"""LLM prompt construction and the Ollama call for the web interface.

Split from the original rag.py so each file has one job:
    retrieval.py   parquet -> assets -> hybrid hits -> JSON cards
    prompts.py     hits -> LLM prompt -> streamed answer               (this file)
    demo.py        hits -> deterministic answer + SSE helpers

The prompt makes the LLM a presenter, not a decider: retrieval + pantry fit
already chose the [Primary recipe] the interface displays, and the model is
told to present exactly that. Backups are only mentioned on request.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Iterator, Sequence
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dish_groups import (  # noqa: E402
    normalize_name,
    version_facts,
    version_positions,
)
from retrieval import Assets  # noqa: E402

LLM_MODEL = "llama3.2:3b"
SEED = 42
GIVE_MATCH_HINTS = True


class LLMUnavailable(RuntimeError):
    """Raised when Ollama cannot answer (not running, model missing, ...)."""


SYSTEM_PROMPT = (
    "You are a Filipino recipe assistant. "
    "Use ONLY the recipes provided in the context. "
    "Do not invent ingredients, quantities, cooking times, or steps that are not in the context. "
    "A substitute must be a DIFFERENT ingredient the user could reasonably have; "
    "never name the missing ingredient as its own substitute. "
    "If the recipe is not the dish type the user asked for, say so instead of pretending. "
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
    """Render retrieved recipes for the prompt, plus an optional focused recipe.

    The first hit is the PRIMARY recipe the interface displays; the rest are
    backups the model may only mention when the user asks for more options.
    """
    blocks = []
    for number, (pos, _) in enumerate(hits, start=1):
        label = "[Primary recipe]" if number == 1 else f"[Backup recipe {number}]"
        blocks.append(_context_block(assets, pos, query_canon, label, max_chars))
    hit_positions = {pos for pos, _ in hits}
    for pos in extra_rows:
        if pos is not None and pos not in hit_positions:
            blocks.append(_context_block(assets, pos, query_canon, "[Focused recipe]", max_chars))
    return "\n\n".join(blocks)


def _constraint_prompt(constraints: dict | None) -> str:
    """One prompt line describing the user's filters, if any."""
    if not constraints:
        return ""
    parts = []
    if constraints.get("diet"):
        parts.append(f"The user is {constraints['diet']}.")
    if constraints.get("max_minutes") is not None:
        parts.append(f"The user has at most {constraints['max_minutes']} minutes to cook.")
    if constraints.get("category"):
        parts.append(f"The user asked for a {constraints['category']} dish.")
    if constraints.get("allergens"):
        parts.append(
            "The user is allergic to: "
            + ", ".join(constraints["allergens"])
            + ". Never recommend those."
        )
    if constraints.get("diet_fallback"):
        parts.append(
            "No fully diet-friendly recipe matched, so the closest one is included; "
            "say so honestly."
        )
    if constraints.get("time_fallback"):
        parts.append(
            "No recipe fits the time budget, so the fastest close match is included; "
            "say so honestly."
        )
    if constraints.get("category_fallback"):
        parts.append(
            "No matching dish of that type exists, so the closest one is included; "
            "say so honestly."
        )
    if constraints.get("allergen_fallback"):
        parts.append(
            "Every close match contains the allergens; say so honestly and tell "
            "the user to adapt the recipe."
        )
    return ("\n" + "\n".join(parts) + "\n") if parts else ""


def _primary_note(assets: Assets, hits, query_canon: Sequence[str]) -> str:
    """Tell the model which recipe the interface is showing as the top card.

    Retrieval + pantry fit already picked the best match deterministically;
    the model's job is to present it, not to re-decide. Without this line the
    model may recommend a different context recipe than the displayed card.
    """
    if not hits:
        return ""
    facts = version_facts(assets.df.iloc[hits[0][0]], query_canon)
    return (
        "The [Primary recipe] above is the recommendation the interface is already "
        f"showing the user: '{facts['name']}'. Present that dish as your "
        "recommendation. The [Backup recipe] entries are NOT recommendations — "
        "mention them only if the user explicitly asks for more options."
    )


def _context_prompt(
    pantry: str,
    query_canon: Sequence[str],
    context: str,
    with_tasks: bool,
    constraints: dict | None = None,
    versions_note: str = "",
    primary_note: str = "",
) -> str:
    base = (
        f"The user has these ingredients: {pantry}\n"
        f"(Normalized ingredient names: {', '.join(query_canon) or 'none recognized'})\n"
        + _constraint_prompt(constraints)
        + f"\nContext recipes:\n{context}"
    )
    if versions_note:
        base += f"\n\n{versions_note}"
    if primary_note:
        base += f"\n\n{primary_note}"
    if not with_tasks:
        return base + "\n\nUse only these recipes. Answer the user's latest message."

    return (
        base
        + "\n\nTasks:\n"
        "1. Present the [Primary recipe] as your recommendation and briefly explain why "
        "it fits (it matches the most pantry ingredients and misses the fewest). Do not "
        "present a [Backup recipe] unless the user asks for more options.\n"
        + TASK_MATCH
        + "\n3. For each missing ingredient, suggest a substitute only if it is reasonable; otherwise say there is none.\n"
        "4. Rewrite the cooking instructions as short numbered steps. Keep all quantities and times."
    )


def _versions_note(assets: Assets, top_pos: int | None, pantry: Sequence[str]) -> str:
    """Tell the model when the top dish exists in several versions."""
    if top_pos is None:
        return ""
    row = assets.df.iloc[top_pos]
    group = assets.groups.get(normalize_name(row["recipe_name"]))
    if group is None or len(group.rows) < 2:
        return ""
    positions = version_positions(group, assets.df, pantry, max_versions=4)
    others = [pos for pos in positions if pos != top_pos][:3]
    if not others:
        return ""
    names = ", ".join(
        f"'{version_facts(assets.df.iloc[pos], pantry)['name']}' "
        f"(~{version_facts(assets.df.iloc[pos], pantry)['display_time']})"
        for pos in others
    )
    return (
        "The recommended dish also has these other versions in the dataset: "
        f"{names}. You may mention them in one short line, but recommend only one."
    )


def build_chat_messages(
    assets: Assets,
    pantry: str,
    hits,
    query_canon: Sequence[str],
    turns: Sequence[dict],
    focus_row: int | None = None,
    constraints: dict | None = None,
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
    top_pos = hits[0][0] if hits else None
    versions_note = _versions_note(assets, top_pos, query_canon)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if not prior:
        messages.append(
            {
                "role": "user",
                "content": _context_prompt(
                    pantry,
                    query_canon,
                    context,
                    with_tasks=True,
                    constraints=constraints,
                    versions_note=versions_note,
                    primary_note=_primary_note(assets, hits, query_canon),
                ),
            }
        )
        return messages

    messages.append(
        {
            "role": "user",
            "content": _context_prompt(
                pantry,
                query_canon,
                context,
                with_tasks=False,
                constraints=constraints,
                versions_note=versions_note,
            ),
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
    import os

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").strip()
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host.rstrip("/")


_OLLAMA_CACHE: dict[str, object] = {"at": 0.0, "host": None, "info": None}
_OLLAMA_TTL = 5.0


def ollama_info(timeout: float = 1.5, force: bool = False) -> dict:
    """Cheap readiness probe for the health endpoint and the UI banner.

    Results are cached for a few seconds: every chat turn asks whether Ollama is
    up, and probing a closed port would otherwise add a timeout to each request.
    """
    host = ollama_host()
    now = time.monotonic()
    cached = _OLLAMA_CACHE["info"]
    if (
        not force
        and cached is not None
        and _OLLAMA_CACHE["host"] == host
        and now - float(_OLLAMA_CACHE["at"]) < _OLLAMA_TTL
    ):
        return dict(cached)  # type: ignore[arg-type]

    try:
        with urlopen(f"{host}/api/tags", timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
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
    if any(word in lowered for word in ("connect", "connection", "refused", "timed out")):
        return (
            f"Could not reach Ollama at {host}. Start it with 'ollama serve' "
            "and try again."
        )
    if "not found" in lowered or "pull" in lowered:
        return f"Ollama is running, but {LLM_MODEL} is not available. Run: ollama pull {LLM_MODEL}"
    return f"Ollama call failed: {text}"
