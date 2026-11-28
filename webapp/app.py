"""FastAPI entry point for the Pinoy Recipe RAG web interface.

HOW TO USE
    From the repository root:
        uvicorn app:app --app-dir webapp --reload --port 8000
    or:
        python webapp/app.py

    Then open http://localhost:8000

Endpoints
    GET  /            the single-page interface
    GET  /api/health  assets loaded? Ollama reachable? model pulled?
    POST /api/search  retrieval only (no LLM, always works offline)
    POST /api/chat    streamed answer via Server-Sent Events

The app is read-only with respect to the repository: it reads the artifacts in
data/processed/ and calls Ollama if it is available.
"""

from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import rag  # noqa: E402

STATIC = HERE / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the heavy assets once; keep the server up even if setup is incomplete."""
    try:
        app.state.assets = rag.load_assets()
        app.state.error = None
    except Exception as exc:  # missing artifacts, broken parquet, ...
        app.state.assets = None
        app.state.error = str(exc)
        print(f"[webapp] assets not loaded: {exc}")
    yield


app = FastAPI(title="Pinoy Recipe RAG", version="1.0.0", lifespan=lifespan)

Method = Literal["overlap", "bm25", "dense", "hybrid"]
Mode = Literal["auto", "demo", "llm"]


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    method: Method = "hybrid"
    k: int = Field(default=rag.TOP_K, ge=1, le=10)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatIn(BaseModel):
    pantry: str = Field(min_length=1, max_length=2000)
    messages: list[ChatTurn] = Field(min_length=1)
    method: Method = "hybrid"
    mode: Mode = "auto"
    focus_row: int | None = Field(default=None, ge=0)
    k: int = Field(default=rag.TOP_K, ge=1, le=10)


def assets_or_503():
    assets = app.state.assets
    if assets is None:
        raise HTTPException(
            status_code=503,
            detail=app.state.error or "Recipe data is not loaded.",
        )
    return assets


@app.get("/api/health")
def health(fresh: bool = False):
    assets = app.state.assets
    return {
        "ok": assets is not None,
        "error": app.state.error,
        "recipes": len(assets.df) if assets is not None else 0,
        "ollama": rag.ollama_info(force=fresh),
        "llm_model": rag.LLM_MODEL,
        "embed_model": rag.EMBED_MODEL,
    }


@app.post("/api/search")
def api_search(body: SearchIn):
    assets = assets_or_503()
    hits, canon, unmapped = rag.search(assets, body.query, body.method, body.k)
    return {
        "method": body.method,
        "normalized": canon,
        "unmapped": unmapped,
        "recipes": rag.summarize_hits(assets, hits, canon),
    }


@app.post("/api/chat")
def api_chat(body: ChatIn):
    assets = assets_or_503()
    turns = [turn.model_dump() for turn in body.messages]
    last_user = next(
        (turn["content"] for turn in reversed(turns) if turn["role"] == "user"),
        body.pantry,
    )
    focus_row = (
        body.focus_row
        if body.focus_row is not None and body.focus_row < len(assets.df)
        else None
    )

    # Query understanding: pantry canonicals, their generic parents, and the
    # diet / time constraints the user stated anywhere in the conversation.
    search_text = rag.retrieval_query(body.pantry, last_user, assets.lookup)
    pantry_canon, pantry_unmapped = rag.parse_query(body.pantry, assets.lookup)
    constraints = rag.merge_constraints(body.pantry, last_user)
    pantry_expanded = rag.expand_pantry(assets, pantry_canon)

    # Retrieve a wider candidate pool, then apply diet/time filters on top so
    # "vegetarian, under 30 minutes" can skip several irrelevant hits. Diets
    # need the widest pool: most retrieved dishes usually contain meat.
    if constraints["diet"]:
        pool_k = 24
    elif constraints["max_minutes"] is not None:
        pool_k = 12
    else:
        pool_k = body.k
    hits, _, _ = rag.search(assets, search_text, body.method, pool_k)
    hits, constraint_report = rag.apply_constraints(assets, hits, pantry_expanded, constraints)
    hits = hits[: body.k]

    recipes = rag.summarize_hits(assets, hits, pantry_expanded)
    versions = (
        rag.version_cards(assets, hits[0][0], pantry_expanded) if hits else []
    )
    more_count = max(0, len(hits) - 1)

    info = rag.ollama_info()
    use_llm = body.mode == "llm" or (
        body.mode == "auto" and info["available"] and info["has_model"]
    )
    is_followup = len([turn for turn in turns if turn["role"] == "user"]) > 1
    meta = {
        "normalized": pantry_canon,
        "unmapped": pantry_unmapped,
        "recipes": recipes,
        "versions": versions,
        "constraints": constraint_report,
        "more_count": more_count,
        "method": body.method,
    }

    def stream():
        if not use_llm:
            yield rag.sse("meta", {**meta, "mode": "demo"})
            if is_followup:
                text = rag.demo_followup(
                    assets, hits, pantry_expanded, last_user, focus_row
                )
            else:
                text = rag.demo_answer(
                    assets, hits, pantry_expanded,
                    constraints=constraint_report, more_count=more_count,
                )
            for piece in rag.chunk_text(text):
                yield rag.sse("token", {"text": piece})
                time.sleep(0.012)
            yield rag.sse("done", {"mode": "demo", "ok": True})
            return

        yield rag.sse("meta", {**meta, "mode": "llm"})
        if not info["available"]:
            yield rag.sse(
                "error",
                {
                    "message": (
                        f"Ollama is not reachable at {info['host']}. "
                        "Start it with 'ollama serve' and try again."
                    )
                },
            )
            yield rag.sse("done", {"mode": "llm", "ok": False})
            return
        if not info["has_model"]:
            yield rag.sse(
                "error",
                {
                    "message": (
                        f"Ollama is running, but {rag.LLM_MODEL} is not pulled. "
                        f"Run: ollama pull {rag.LLM_MODEL}"
                    )
                },
            )
            yield rag.sse("done", {"mode": "llm", "ok": False})
            return

        try:
            messages = rag.build_chat_messages(
                assets,
                body.pantry,
                hits,
                pantry_expanded,
                turns,
                focus_row=focus_row,
                constraints=constraint_report,
            )
            for piece in rag.generate(messages):
                yield rag.sse("token", {"text": piece})
        except (rag.LLMUnavailable, ValueError) as exc:
            yield rag.sse("error", {"message": str(exc)})
            yield rag.sse("done", {"mode": "llm", "ok": False})
            return
        yield rag.sse("done", {"mode": "llm", "ok": True})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
