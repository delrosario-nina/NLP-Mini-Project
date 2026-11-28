"""Retrieval and answer generation for the Pinoy Recipe RAG web interface.

HOW TO USE
    From the repository root:
        uvicorn app:app --app-dir webapp --reload --port 8000
    or:
        python webapp/app.py

    Testing without Ollama:
        python webapp/tools/mock_ollama.py --port 11435
        $env:OLLAMA_HOST = "http://localhost:11435"

    Ollama Cloud instead of a local install (key from ollama.com/settings/keys):
        $env:OLLAMA_HOST = "https://ollama.com"
        $env:OLLAMA_API_KEY = "<your key>"
        $env:LLM_MODEL = "gpt-oss:20b"   # any model listed at ollama.com/api/tags

    The ollama client picks up all three variables automatically; nothing is
    hard-coded, so the same code serves local, LAN, and cloud.

This module is a thin port of notebooks/04_rag_pipeline.ipynb. The notebook is
still the source of truth for retrieval behaviour; if the scoring changes there,
mirror the change here.

It only reads local artifacts and never writes to tracked files:
    data/processed/cleaned_recipes.parquet   (notebooks 01 + 02)
    data/processed/alias_map.json            (notebook 02)
    data/processed/recipe_vecs_*.npy         (notebook 04; rebuilt if absent)
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dish_groups import version_facts  # noqa: E402
from ingredient_utils import build_lookup, load_alias_map, normalize_text  # noqa: E402

# Same constants as notebook 04. LLM_MODEL can be overridden for Ollama Cloud,
# e.g. $env:LLM_MODEL = "gpt-oss:20b" with $env:OLLAMA_HOST = "https://ollama.com".
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.2:3b")
EMBED_MODEL = "intfloat/multilingual-e5-small"
TOP_K = 3
RRF_K = 60
SEED = 42
HYBRID_PARTS = ["bm25", "dense"]
GIVE_MATCH_HINTS = True
METHODS = ("overlap", "bm25", "dense", "hybrid")
MAX_STEPS = 8

# Demo-only substitution ideas, copied from examples/demo_answer.py. These are
# illustrative, not learned from the dataset.
SUBSTITUTIONS = {
    "annatto": "achuete seeds steeped in warm oil",
    "annatto oil": "achuete seeds steeped in warm oil",
    "annatto seed": "achuete seeds steeped in warm oil",
    "palm vinegar": "coconut vinegar or white vinegar",
    "calamansi": "lemon juice",
    "calamansi juice": "lemon juice",
    "coconut vinegar": "white vinegar with a squeeze of calamansi",
    "lemongrass": "tanglad, or 1/2 teaspoon dried lemongrass",
    "margarine": "butter",
    "brown sugar": "white sugar with a little molasses",
    "whole chicken": "chicken thighs or breast pieces",
    "bay leaf": "dried bay leaves",
    "black pepper": "ground white pepper or crushed peppercorns",
    "bird's eye chili": "a pinch of red pepper flakes",
    "soda": "a pinch of sugar plus extra calamansi juice",
    "lemon": "calamansi juice",
}


class AssetsMissing(RuntimeError):
    """Raised when the notebook artifacts the app depends on are absent."""


class LLMUnavailable(RuntimeError):
    """Raised when Ollama cannot answer (not running, model missing, ...)."""


@dataclass
class Assets:
    """Everything loaded once at startup, shared by every request."""

    df: pd.DataFrame
    lookup: dict
    bm25: object
    embedder: object
    vecs: np.ndarray


def load_assets(progress=print) -> Assets:
    """Load recipes, alias map, BM25 index and embeddings.

    The embedding cache is rebuilt (and saved) only when it is missing or does
    not match the current recipe count, exactly like notebook 04.
    """
    parquet = ROOT / "data/processed/cleaned_recipes.parquet"
    alias_path = ROOT / "data/processed/alias_map.json"
    missing = [p for p in (parquet, alias_path) if not p.exists()]
    if missing:
        names = ", ".join(str(p.relative_to(ROOT)) for p in missing)
        raise AssetsMissing(
            "Recipe artifacts are missing: "
            f"{names}. Run notebooks 01 and 02 first, then restart the app."
        )

    df = pd.read_parquet(parquet)
    required = {"recipe_name", "ingredients", "instructions", "canonical_ingredients"}
    absent = required - set(df.columns)
    if absent:
        raise AssetsMissing(
            "The recipe parquet is missing columns "
            f"{sorted(absent)}. Re-run notebooks 01 and 02."
        )

    alias_map = load_alias_map(alias_path)
    lookup, _ = build_lookup(alias_map)
    df["canon_list"] = df["canonical_ingredients"].map(list)
    df["canon_set"] = df["canon_list"].map(set)

    bm25 = BM25Okapi([[x.replace(" ", "_") for x in lst] for lst in df["canon_list"]])

    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(EMBED_MODEL)
    cache = ROOT / "data/processed" / f"recipe_vecs_{EMBED_MODEL.split('/')[-1]}.npy"
    vecs = np.load(cache) if cache.exists() else None
    if vecs is None or vecs.shape[0] != len(df):
        vecs = _embed_passages(df, embedder, cache, progress=progress)

    progress(f"Loaded {len(df)} recipes | {len(alias_map)} canonical ingredients")
    return Assets(df=df, lookup=lookup, bm25=bm25, embedder=embedder, vecs=vecs)


def _embed_passages(df: pd.DataFrame, embedder, cache: Path, progress=print) -> np.ndarray:
    """Build the passage embeddings and cache them, mirroring notebook 04."""
    doc_text = (
        df["recipe_name"].astype(str)
        + ". Ingredients: "
        + df["canon_list"].map(lambda items: ", ".join(items))
    ).tolist()
    passages = ["passage: " + text for text in doc_text]
    chunks = []
    for start in range(0, len(passages), 200):
        chunks.append(
            embedder.encode(
                passages[start : start + 200],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )
        progress(f"Embedded {min(start + 200, len(passages))}/{len(passages)} recipes")
    vecs = np.vstack(chunks)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, vecs)
    return vecs


def parse_query(query: str, lookup: dict) -> tuple[list[str], list[str]]:
    """Raw pantry text (English, Tagalog, Taglish) -> (canonical names, unmapped tokens)."""
    parts = re.split(r"[,;\n]|\bat\b|\band\b", query, flags=re.IGNORECASE)
    canonicals: list[str] = []
    unmapped: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        found, left = normalize_text(part, lookup)
        for item in found:
            if item not in canonicals:
                canonicals.append(item)
        unmapped.extend(left)
    return canonicals, unmapped


def retrieval_query(pantry: str, followup: str, lookup: dict) -> str:
    """Keep the pantry, but extend it when a follow-up names new ingredients."""
    base, _ = parse_query(pantry, lookup)
    extra, _ = parse_query(followup, lookup)
    if set(extra) - set(base):
        return f"{pantry}, {followup}"
    return pantry


def overlap_scores(assets: Assets, query_canon: Sequence[str]) -> np.ndarray:
    """Matched-count score with ingredient coverage as a tie-breaker."""
    q = set(query_canon)
    out = np.zeros(len(assets.df))
    if not q:
        return out
    for i, canon in enumerate(assets.df["canon_set"]):
        matched = len(q & canon)
        out[i] = matched + 0.5 * (matched / max(1, len(canon)))
    return out


def bm25_scores(assets: Assets, query_canon: Sequence[str]) -> np.ndarray:
    tokens = [item.replace(" ", "_") for item in query_canon]
    return assets.bm25.get_scores(tokens) if tokens else np.zeros(len(assets.df))


def dense_scores(assets: Assets, query: str, query_canon: Sequence[str]) -> np.ndarray:
    text = ("Ingredients: " + ", ".join(query_canon)) if query_canon else query
    vec = assets.embedder.encode(["query: " + text], normalize_embeddings=True)[0]
    return assets.vecs @ vec


def rank_of(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def search(
    assets: Assets,
    query: str,
    method: str = "hybrid",
    k: int = TOP_K,
) -> tuple[list[tuple[int, float]], list[str], list[str]]:
    """Return ([(row_position, score), ...], canonicals, unmapped tokens)."""
    if method not in METHODS:
        raise ValueError(f"Unknown retrieval method {method!r}. Choose from {METHODS}.")

    canon, unmapped = parse_query(query, assets.lookup)
    parts = {
        "overlap": overlap_scores(assets, canon),
        "bm25": bm25_scores(assets, canon),
        "dense": dense_scores(assets, query, canon),
    }
    if method == "hybrid":
        scores = sum(1.0 / (RRF_K + rank_of(parts[name])) for name in HYBRID_PARTS)
    else:
        scores = parts[method]

    top = np.argsort(-scores, kind="stable")[:k]
    hits = [(int(pos), float(scores[pos])) for pos in top]
    return hits, canon, unmapped


SYSTEM_PROMPT = (
    "You are a Filipino recipe assistant. "
    "Use ONLY the recipes provided in the context. "
    "Do not invent ingredients, quantities, cooking times, or steps that are not in the context. "
    "Reply in the same language as the user's message (English, Tagalog, or Taglish)."
)
TASK_MATCH = (
    "2. Report the matched and missing ingredients exactly as listed in the context for the recommended recipe."
    if GIVE_MATCH_HINTS
    else "2. List the matched ingredients (the user has them) and the missing ingredients."
)


def _context_block(assets: Assets, pos: int, query_canon: Sequence[str], label: str, max_chars: int) -> str:
    """One recipe as prompt context, with deterministic match hints."""
    row = assets.df.iloc[pos]
    hint = ""
    if GIVE_MATCH_HINTS:
        have = set(query_canon)
        matched = [item for item in row["canon_list"] if item in have]
        missing = [item for item in row["canon_list"] if item not in have]
        hint = (
            f"\nMatched ingredients (the user has): {', '.join(matched) or 'none'}"
            f"\nMissing ingredients: {', '.join(missing) or 'none'}"
        )
    return (
        f"{label} {row['recipe_name']}\n"
        f"Ingredients: {row['ingredients']}{hint}\n"
        f"Instructions: {str(row['instructions'])[:max_chars]}"
    )


def format_context(
    assets: Assets,
    hits,
    query_canon: Sequence[str],
    max_chars: int = 1200,
    extra_rows: Sequence[int] = (),
) -> str:
    """Render retrieved recipes for the prompt, plus an optional focused recipe."""
    blocks = [
        _context_block(assets, pos, query_canon, f"[Recipe {number}]", max_chars)
        for number, (pos, _) in enumerate(hits, start=1)
    ]
    hit_positions = {pos for pos, _ in hits}
    for pos in extra_rows:
        if pos is not None and pos not in hit_positions:
            blocks.append(_context_block(assets, pos, query_canon, "[Focused recipe]", max_chars))
    return "\n\n".join(blocks)


def _context_prompt(
    pantry: str,
    query_canon: Sequence[str],
    context: str,
    with_tasks: bool,
) -> str:
    base = (
        f"The user has these ingredients: {pantry}\n"
        f"(Normalized ingredient names: {', '.join(query_canon) or 'none recognized'})\n\n"
        f"Context recipes:\n{context}"
    )
    if not with_tasks:
        return base + "\n\nUse only these recipes. Answer the user's latest message."

    return (
        base
        + "\n\nTasks:\n"
        "1. Recommend the best matching dish from the context and briefly explain why it fits.\n"
        + TASK_MATCH
        + "\n3. For each missing ingredient, suggest a substitute only if it is reasonable; otherwise say there is none.\n"
        "4. Rewrite the cooking instructions as short numbered steps. Keep all quantities and times."
    )


def build_messages(assets: Assets, query: str, hits, query_canon: Sequence[str]) -> list[dict]:
    """Single-turn prompt, kept for scripts that only need one answer."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _context_prompt(
                query,
                query_canon,
                format_context(assets, hits, query_canon),
                with_tasks=True,
            ),
        },
    ]


def build_chat_messages(
    assets: Assets,
    pantry: str,
    hits,
    query_canon: Sequence[str],
    turns: Sequence[dict],
    focus_row: int | None = None,
) -> list[dict]:
    """Multi-turn prompt: pantry context first, then the real conversation.

    ``turns`` is the full visible conversation including the new user message.
    The first user turn is replaced by the pantry/context block so the model is
    not given the pantry twice.
    """
    turns = [turn for turn in turns if turn.get("content")]
    user_indexes = [i for i, turn in enumerate(turns) if turn.get("role") == "user"]
    if not user_indexes:
        raise ValueError("The conversation needs at least one user message.")

    last_user = user_indexes[-1]
    prior = turns[:last_user]
    new_text = turns[last_user]["content"]

    context = format_context(
        assets,
        hits,
        query_canon,
        extra_rows=(focus_row,) if focus_row is not None else (),
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if not prior:
        messages.append(
            {
                "role": "user",
                "content": _context_prompt(pantry, query_canon, context, with_tasks=True),
            }
        )
        return messages

    messages.append(
        {
            "role": "user",
            "content": _context_prompt(pantry, query_canon, context, with_tasks=False),
        }
    )
    # Drop the original pantry turn; it is represented by the context prompt.
    rest = prior[1:] if prior[0].get("role") == "user" else prior
    messages.extend({"role": turn["role"], "content": turn["content"]} for turn in rest)
    messages.append({"role": "user", "content": new_text})
    return messages


def generate(messages: list[dict]) -> Iterator[str]:
    """Stream answer tokens from Ollama. Raises LLMUnavailable with a usable message."""
    try:
        import ollama
    except ImportError as exc:
        raise LLMUnavailable(
            "The 'ollama' Python package is not installed. Run: pip install ollama"
        ) from exc

    try:
        for chunk in ollama.chat(
            model=LLM_MODEL,
            messages=messages,
            stream=True,
            options={"temperature": 0, "seed": SEED, "num_predict": 500},
        ):
            piece = (chunk.get("message") or {}).get("content") or ""
            if piece:
                yield piece
    except Exception as exc:  # ollama raises its own connection/response errors
        raise LLMUnavailable(friendly_llm_error(exc)) from exc


def ollama_host() -> str:
    """Resolve OLLAMA_HOST the same way the Python client does."""
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").strip()
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host.rstrip("/")


def ollama_is_remote(host: str | None = None) -> bool:
    """True when OLLAMA_HOST points at another machine or Ollama Cloud."""
    host = (host or ollama_host()).lower()
    return not any(
        host.startswith(f"{scheme}://{machine}")
        for scheme in ("http", "https")
        for machine in ("localhost", "127.0.0.1", "0.0.0.0", "[::1]")
    )


def ollama_api_key() -> str:
    """The OLLAMA_API_KEY value, when one is configured (Ollama Cloud)."""
    return os.environ.get("OLLAMA_API_KEY", "").strip()


_OLLAMA_CACHE: dict[str, object] = {"at": 0.0, "host": None, "info": None}
_OLLAMA_TTL = 5.0


def ollama_info(timeout: float | None = None, force: bool = False) -> dict:
    """Cheap readiness probe for the health endpoint and the UI banner.

    Results are cached for a few seconds: every chat turn asks whether Ollama is
    up, and probing a closed port would otherwise add a timeout to each request.
    Remote hosts (a groupmate's machine or https://ollama.com) get a longer
    timeout than a localhost probe.
    """
    host = ollama_host()
    if timeout is None:
        timeout = 4.0 if ollama_is_remote(host) else 1.5
    now = time.monotonic()
    cached = _OLLAMA_CACHE["info"]
    if (
        not force
        and cached is not None
        and _OLLAMA_CACHE["host"] == host
        and now - float(_OLLAMA_CACHE["at"]) < _OLLAMA_TTL
    ):
        return dict(cached)  # type: ignore[arg-type]

    request = Request(f"{host}/api/tags")
    key = ollama_api_key()
    if key:
        request.add_header("Authorization", f"Bearer {key}")

    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        # 401/403 means the host is there but the key was rejected; that is
        # a configuration problem the chat call should explain, not "offline".
        if key and exc.code in (401, 403):
            info = {
                "available": True,
                "host": host,
                "models": [],
                "has_model": False,
                "auth_error": "API key was rejected. Check OLLAMA_API_KEY.",
            }
        else:
            info = {"available": False, "host": host, "models": [], "has_model": False}
    except Exception:
        info = {"available": False, "host": host, "models": [], "has_model": False}
    else:
        models = [str(model.get("name", "")) for model in payload.get("models", [])]
        has_model = any(
            name == LLM_MODEL or name == LLM_MODEL + ":latest" for name in models
        )
        info = {
            "available": True,
            "host": host,
            "models": models,
            "has_model": has_model,
        }

    _OLLAMA_CACHE.update({"at": now, "host": host, "info": info})
    return dict(info)


def friendly_llm_error(exc: Exception) -> str:
    """Turn an Ollama exception into something a student can act on."""
    text = str(exc)
    lowered = text.lower()
    host = ollama_host()

    if any(word in lowered for word in ("unauthorized", "invalid api key", "401")):
        return (
            "Ollama rejected the API key. Check that OLLAMA_API_KEY is set "
            "correctly (create one at ollama.com/settings/keys) and that the "
            "terminal running uvicorn has it."
        )
    if any(word in lowered for word in ("forbidden", "403")):
        return (
            f"Ollama at {host} refused the request. If this is Ollama Cloud, "
            "check your key's access and remaining credit on ollama.com."
        )
    if any(word in lowered for word in ("connect", "connection", "refused", "timed out")):
        if ollama_is_remote(host):
            return (
                f"Could not reach Ollama at {host}. Check your internet "
                "connection and that the host is correct, then try again."
            )
        return (
            f"Could not reach Ollama at {host}. Start it with 'ollama serve' "
            "and try again."
        )
    if "not found" in lowered or "pull" in lowered:
        if ollama_is_remote(host):
            return (
                f"{host} does not offer {LLM_MODEL}. Pick a model from "
                f"{host}/api/tags and set LLM_MODEL to its name."
            )
        return f"Ollama is running, but {LLM_MODEL} is not available. Run: ollama pull {LLM_MODEL}"
    return f"Ollama call failed: {text}"


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else round(number, 1)


def _steps(row: pd.Series, limit: int = MAX_STEPS) -> list[str]:
    parts = [part.strip() for part in str(row.get("instructions", "")).split("|")]
    return [part for part in parts if part][:limit]


def _ingredient_lines(row: pd.Series) -> list[str]:
    """Raw ingredient lines with quantities, e.g. '2 lbs chicken'."""
    parts = [
        re.sub(r"\s*\(note\s*\d+\)", "", part.strip(), flags=re.IGNORECASE)
        for part in str(row.get("ingredients", "")).split("|")
    ]
    return [part for part in parts if part]


def _servings_text(value) -> str:
    """Render servings without a trailing '.0' (4.0 -> '4')."""
    if value in (None, "", 0):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def summarize_hits(assets: Assets, hits, query_canon: Sequence[str]) -> list[dict]:
    """JSON-safe recipe cards for the UI, reusing dish_groups.version_facts."""
    cards = []
    for rank, (pos, score) in enumerate(hits, start=1):
        row = assets.df.iloc[pos]
        facts = version_facts(row, query_canon)
        cards.append(
            {
                "rank": rank,
                "row": pos,
                "score": round(float(score), 6),
                "name": facts["name"],
                "time": facts["display_time"],
                "active_minutes": _number(facts["active_minutes"]),
                "servings": _servings_text(facts["servings"]),
                "matched": facts["matched"],
                "missing": facts["missing"],
                "missing_cost": facts["missing_cost"],
                "techniques": facts["techniques"],
                "steps": _steps(row),
                "instructions": _steps(row, limit=40),
                "ingredient_lines": _ingredient_lines(row),
                "ingredients": [str(item) for item in row["canon_list"]],
            }
        )
    return cards


def demo_answer(assets: Assets, hits, query_canon: Sequence[str]) -> str:
    """Deterministic answer computed from the top recipe. No model call.

    Mirrors examples/demo_answer.py so the UI can be tested without Ollama and
    so the group can compare the LLM wording against a known-good baseline.
    """
    if not hits:
        return (
            "I couldn't find a recipe for those ingredients. Try adding a main "
            "ingredient such as chicken, pork, fish, or tofu."
        )

    row = assets.df.iloc[hits[0][0]]
    facts = version_facts(row, query_canon)
    lines = [
        f"Recommended: {facts['name']}",
        f"- Total time: ~{facts['display_time']}",
    ]
    if _servings_text(facts["servings"]):
        lines.append(f"- Servings: {_servings_text(facts['servings'])}")

    lines += [
        "",
        "Matched ingredients (you have): " + (", ".join(facts["matched"]) or "none"),
        "",
        "Missing ingredients: " + (", ".join(facts["missing"]) or "none"),
        "",
        "Suggested substitutions:",
    ]
    found = False
    for item in facts["missing"]:
        if item in SUBSTITUTIONS:
            lines.append(f"- {item} -> {SUBSTITUTIONS[item]}")
            found = True
    if not found:
        lines.append("- No reliable substitution from the retrieved recipe.")

    lines += ["", "Simplified steps:"]
    for number, step in enumerate(_steps(row), start=1):
        lines.append(f"{number}. {step}")

    lines += [
        "",
        "Demo answer generated locally from the retrieved recipe. No LLM call was made.",
    ]
    return "\n".join(lines)


def _demo_summary(assets: Assets, pos: int, query_canon: Sequence[str], label: str) -> str:
    facts = version_facts(assets.df.iloc[pos], query_canon)
    lines = [f"{label}: {facts['name']}", f"- Total time: ~{facts['display_time']}"]
    if _servings_text(facts["servings"]):
        lines.append(f"- Servings: {_servings_text(facts['servings'])}")
    lines += [
        "",
        "Matched ingredients (you have): " + (", ".join(facts["matched"]) or "none"),
        "",
        "Missing ingredients: " + (", ".join(facts["missing"]) or "none"),
    ]
    return "\n".join(lines)


def demo_followup(
    assets: Assets,
    hits,
    query_canon: Sequence[str],
    message: str,
    focus_row: int | None = None,
) -> str:
    """Deterministic answers for common follow-ups when Ollama is unavailable."""
    positions = [pos for pos, _ in hits]
    if focus_row is not None and 0 <= focus_row < len(assets.df):
        positions = [focus_row] + [pos for pos in positions if pos != focus_row]
    if not positions:
        return demo_answer(assets, hits, query_canon)

    text = message.lower()

    if any(word in text for word in ("fast", "quick", "short")):
        pos = min(
            positions,
            key=lambda p: version_facts(assets.df.iloc[p], query_canon)["budget_minutes"],
        )
        return _demo_summary(assets, pos, query_canon, "Fastest option")

    if any(word in text for word in ("fewest", "least", "missing", "complete")):
        def missing_key(pos: int) -> tuple[int, float, int]:
            facts = version_facts(assets.df.iloc[pos], query_canon)
            return (len(facts["missing"]), facts["missing_cost"], pos)

        pos = min(positions, key=missing_key)
        return _demo_summary(assets, pos, query_canon, "Fewest missing ingredients")

    if any(word in text for word in ("substitut", "replace", "instead")):
        facts = version_facts(assets.df.iloc[positions[0]], query_canon)
        lines = [f"Substitutions for {facts['name']}:", ""]
        found = False
        for item in facts["missing"]:
            if item in SUBSTITUTIONS:
                lines.append(f"- {item} -> {SUBSTITUTIONS[item]}")
                found = True
        if not found:
            lines.append("- No reliable substitution from the retrieved recipe.")
        return "\n".join(lines)

    if any(word in text for word in ("step", "how", "cook", "method", "instruction")):
        pos = positions[0]
        facts = version_facts(assets.df.iloc[pos], query_canon)
        lines = [f"Steps for {facts['name']}:", ""]
        for number, step in enumerate(_steps(assets.df.iloc[pos]), start=1):
            lines.append(f"{number}. {step}")
        return "\n".join(lines)

    return (
        "Ollama isn't connected, so I'm answering with the local demo. I can cover "
        "the fastest option, the fewest missing ingredients, substitutions, or the "
        "cooking steps. Start Ollama for free-form follow-ups."
    )


def chunk_text(text: str, words: int = 3) -> Iterator[str]:
    """Split text into small pieces so demo mode streams like the real thing."""
    tokens = re.findall(r"\S+\s*", text)
    for start in range(0, len(tokens), words):
        yield "".join(tokens[start : start + words])


def sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
