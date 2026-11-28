"""Retrieval and answer generation for the Pinoy Recipe RAG web interface.

HOW TO USE
    From the repository root:
        uvicorn app:app --app-dir webapp --reload --port 8000
    or:
        python webapp/app.py

    Testing without Ollama:
        python webapp/tools/mock_ollama.py --port 11435
        $env:OLLAMA_HOST = "http://localhost:11435"

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
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence
from urllib.request import urlopen

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dish_groups import (  # noqa: E402
    build_groups,
    ingredient_importance,
    normalize_name,
    version_facts,
    version_positions,
)
from alias_map import parents_map  # noqa: E402
from ingredient_utils import (  # noqa: E402
    build_lookup,
    expand_canonicals,
    load_alias_map,
    normalize_text,
)

# Same constants as notebook 04.
LLM_MODEL = "llama3.2:3b"
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
    parents: dict
    groups: dict
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
    parents = parents_map()
    df["canon_list"] = df["canonical_ingredients"].map(list)
    df["canon_set"] = df["canon_list"].map(set)
    # Matching set: a recipe's canonicals plus their generic parents, so a
    # pantry "pork" matches a recipe that uses pork belly, and "fish" matches
    # a bangus dish. Exact-cut recipes still outrank generic ones because they
    # contain BOTH tokens.
    df["canon_match"] = df["canon_list"].map(
        lambda lst: set(lst) | {parents[c] for c in lst if c in parents}
    )

    bm25 = BM25Okapi([[x.replace(" ", "_") for x in sorted(s)] for s in df["canon_match"]])
    groups = {group.name_key: group for group in build_groups(df)}

    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(EMBED_MODEL)
    cache = ROOT / "data/processed" / f"recipe_vecs_{EMBED_MODEL.split('/')[-1]}.npy"
    vecs = np.load(cache) if cache.exists() else None
    if vecs is None or vecs.shape[0] != len(df):
        vecs = _embed_passages(df, embedder, cache, progress=progress)

    progress(f"Loaded {len(df)} recipes | {len(alias_map)} canonical ingredients")
    return Assets(
        df=df, lookup=lookup, parents=parents, groups=groups, bm25=bm25,
        embedder=embedder, vecs=vecs,
    )


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


# --- query understanding: time budget, diet, and pantry expansion -------------

_TIME_NUMBERS = {
    "half": 0.5, "an": 1, "a": 1, "isang": 1,
    "one": 1, "una": 1, "two": 2, "dalawang": 2, "three": 3, "tatlong": 3,
    "four": 4, "apat": 4, "five": 5, "limang": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20,
    "thirty": 30, "forty": 40, "forty-five": 45, "forty five": 45,
    "sixty": 60, "ninety": 90,
}
_TIME_UNITS_MIN = {"min", "mins", "minute", "minutes", "minuto", "minutos", "mins"}
_TIME_UNITS_HOUR = {"hour", "hours", "hr", "hrs", "oras"}
_NUMBER = r"\d+(?:\.\d+)?|" + "|".join(_TIME_NUMBERS)
_TIME_RE = re.compile(
    rf"\b({_NUMBER})\s*(hour|hours|hr|hrs|oras|minute|minutes|min|mins|minuto|minutos)\b",
    flags=re.IGNORECASE,
)
_HALF_HOUR_RE = re.compile(
    r"\bhalf\s+(?:an?\s+)?(?:hour|oras)\b|kalahating\s+oras", flags=re.IGNORECASE
)

_DIET_RES: list[tuple[str, re.Pattern]] = [
    ("vegan", re.compile(r"\bvegan\b|plant.?based|walang\s+karne", re.IGNORECASE)),
    (
        "pescatarian",
        re.compile(
            r"\bpescatarian\b|\bpescetarian\b|pesco.?vegetarian|\bpesco\b", re.IGNORECASE
        ),
    ),
    (
        "vegetarian",
        re.compile(
            r"\bvegetarian\b|\bveggie\b|\bvegy\b|meatless|no\s+meat|walang\s+karneng"
            r"|gulay\s+lang|vegetables\s+only",
            re.IGNORECASE,
        ),
    ),
]

# Canonical exclusion sets per diet, in diet order: vegan > vegetarian > pescatarian.
_MEAT = {
    "pork", "pork_belly", "pork_hock", "ground_pork", "pork_ribs", "pork_shoulder",
    "pork_loin", "pork intestine", "pig ears", "kidney", "pork blood", "liver",
    "liver spread", "bacon", "ham", "longganisa", "chorizo de bilbao", "chicharon",
    "lechon", "leftover_pork_adobo", "adobo", "beef", "beef shank", "beef tripe",
    "goat meat", "corned beef", "hotdog", "pork bouillon", "beef bouillon",
}
_POULTRY = {
    "chicken", "chicken_breast", "chicken_thigh", "whole chicken",
    "leftover_chicken", "chicken bouillon",
}
_SEAFOOD = {
    "milkfish", "fish", "tilapia", "salmon", "tuna", "lapu lapu", "galunggong",
    "tulingan", "tanigue", "pompano", "sardines", "dried fish", "tinapa flakes",
    "fish flakes", "shrimp", "squid", "mussel", "crab", "clams", "anchovy",
    "fish sauce", "shrimp paste", "fish powder",
}
_ANIMAL = {
    "egg", "quail egg", "salted egg", "milk", "evaporated milk", "condensed milk",
    "cheese", "kesong puti", "butter", "margarine", "all purpose cream",
    "mayonnaise", "honey",
}
_DIET_EXCLUSIONS = {
    "vegan": _MEAT | _POULTRY | _SEAFOOD | _ANIMAL,
    "vegetarian": _MEAT | _POULTRY | _SEAFOOD,
    "pescatarian": _MEAT | _POULTRY,
}


def parse_constraints(text: str) -> dict:
    """Read the time budget and diet out of a user message.

    Returns {"diet": str|None, "max_minutes": int|None} so retrieval can filter
    before any answer is written. Constraint phrases are ignored for ingredient
    matching (they are not ingredients), which parse_query already handles.
    """
    lowered = str(text).lower()

    diet = None
    for name, pattern in _DIET_RES:
        if pattern.search(lowered):
            diet = name
            break

    minutes: list[float] = []
    for value, unit in _TIME_RE.findall(lowered):
        number = _TIME_NUMBERS.get(value.lower())
        if number is None:
            try:
                number = float(value)
            except ValueError:
                continue
        multiplier = 60 if unit.lower() in _TIME_UNITS_HOUR else 1
        minutes.append(number * multiplier)
    if _HALF_HOUR_RE.search(lowered):
        minutes.append(30.0)

    return {
        "diet": diet,
        "max_minutes": int(min(minutes)) if minutes else None,
    }


def merge_constraints(*texts: str) -> dict:
    """Union constraints across pantry and the latest follow-up."""
    diet = None
    max_minutes: list[int] = []
    for text in texts:
        found = parse_constraints(text or "")
        if found["diet"]:
            diet = found["diet"]
        if found["max_minutes"] is not None:
            max_minutes.append(found["max_minutes"])
    return {
        "diet": diet,
        "max_minutes": min(max_minutes) if max_minutes else None,
    }


def expand_pantry(assets: Assets, canon: Sequence[str]) -> list[str]:
    """Pantry canonicals plus their generic parents (kasim -> pork shoulder, pork)."""
    return expand_canonicals(list(canon), assets.parents)


def apply_constraints(
    assets: Assets, hits, pantry: Sequence[str], constraints: dict
) -> tuple[list, dict]:
    """Filter retrieved recipes by diet and time, with graceful fallback.

    Returns (filtered_hits, report). If a filter would remove everything, it is
    skipped and flagged in the report so the answer can say so honestly.
    """
    report = {"diet": constraints["diet"], "max_minutes": constraints["max_minutes"],
              "diet_fallback": False, "time_fallback": False}
    kept = list(hits)

    if constraints["diet"]:
        banned = _DIET_EXCLUSIONS[constraints["diet"]]
        survivors = [
            hit for hit in kept if not (banned & assets.df.iloc[hit[0]]["canon_match"])
        ]
        if survivors:
            kept = survivors
        else:
            report["diet_fallback"] = True

    if constraints["max_minutes"] is not None:
        limit = constraints["max_minutes"]

        def within(hit) -> bool:
            return version_facts(assets.df.iloc[hit[0]], pantry)["budget_minutes"] <= limit

        survivors = [hit for hit in kept if within(hit)]
        if survivors:
            kept = survivors
        else:
            report["time_fallback"] = True

    return kept, report


def rank_by_pantry_fit(assets: Assets, hits, pantry: Sequence[str]):
    """Order retrieved recipes by how well they use the user's pantry.

    Retrieval (hybrid BM25 + dense) decides which dishes are candidates; this
    decides which one to SHOW first: most (importance-weighted) matched
    ingredients, then fewest missing, then fastest, then retrieval score.
    So when two versions match the same pantry items, the one missing fewer
    ingredients wins.
    """
    def key(hit):
        pos, score = hit
        facts = version_facts(assets.df.iloc[pos], pantry)
        matched_weight = sum(ingredient_importance(item) for item in facts["matched"])
        return (
            -matched_weight,
            facts["missing_cost"],
            facts["budget_minutes"],
            -score,
        )

    return sorted(hits, key=key)


def version_cards(assets: Assets, top_pos: int, pantry: Sequence[str], limit: int = 3) -> list[dict]:
    """Cards for the other versions of the top dish (pork adobo has five)."""
    row = assets.df.iloc[top_pos]
    group = assets.groups.get(normalize_name(row["recipe_name"]))
    if group is None or len(group.rows) < 2:
        return []
    positions = version_positions(group, assets.df, pantry, max_versions=limit + 1)
    others = [pos for pos in positions if pos != top_pos][:limit]
    if not others:
        return []
    cards = summarize_hits(assets, [(pos, 0.0) for pos in others], pantry)
    for card in cards:
        card["is_version"] = True
    return cards


def overlap_scores(assets: Assets, query_canon: Sequence[str]) -> np.ndarray:
    """Matched-count score with ingredient coverage as a tie-breaker.

    Uses ``canon_match`` (canonicals + their generic parents) so a generic
    pantry item also hits specific-cut recipes.
    """
    q = set(query_canon)
    out = np.zeros(len(assets.df))
    if not q:
        return out
    for i, canon in enumerate(assets.df["canon_match"]):
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


def _constraint_prompt(constraints: dict | None) -> str:
    """One prompt line describing the user's diet and time budget, if any."""
    if not constraints:
        return ""
    parts = []
    if constraints.get("diet"):
        parts.append(f"The user is {constraints['diet']}.")
    if constraints.get("max_minutes") is not None:
        parts.append(f"The user has at most {constraints['max_minutes']} minutes to cook.")
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
    return ("\n" + "\n".join(parts) + "\n") if parts else ""


def _context_prompt(
    pantry: str,
    query_canon: Sequence[str],
    context: str,
    with_tasks: bool,
    constraints: dict | None = None,
    versions_note: str = "",
) -> str:
    base = (
        f"The user has these ingredients: {pantry}\n"
        f"(Normalized ingredient names: {', '.join(query_canon) or 'none recognized'})\n"
        + _constraint_prompt(constraints)
        + f"\nContext recipes:\n{context}"
    )
    if versions_note:
        base += f"\n\n{versions_note}"
    if not with_tasks:
        return base + "\n\nUse only these recipes. Answer the user's latest message."

    return (
        base
        + "\n\nTasks:\n"
        "1. Recommend the single best matching dish from the context and briefly explain "
        "why it fits. Do not list the other retrieved recipes unless the user asks for "
        "more options.\n"
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


def build_messages(
    assets: Assets,
    query: str,
    hits,
    query_canon: Sequence[str],
    constraints: dict | None = None,
) -> list[dict]:
    """Single-turn prompt, kept for scripts that only need one answer."""
    top_pos = hits[0][0] if hits else None
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _context_prompt(
                query,
                query_canon,
                format_context(assets, hits, query_canon),
                with_tasks=True,
                constraints=constraints,
                versions_note=_versions_note(assets, top_pos, query_canon),
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
    """Render servings without a trailing '.0' (4.0 / '4.0' -> '4')."""
    if value in (None, "", 0):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value)
    if text.endswith(".0"):
        return text[:-2]
    return text


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


def _constraint_notes(constraints: dict) -> list[str]:
    """Honest one-liners about how diet/time filters were applied."""
    notes = []
    if constraints["diet"]:
        notes.append(f"Diet: {constraints['diet']}.")
    if constraints["max_minutes"] is not None:
        notes.append(f"Time budget: {constraints['max_minutes']} minutes or less.")
    if constraints.get("diet_fallback"):
        notes.append(
            "No fully " + constraints["diet"] + "-friendly recipe matched those "
            "ingredients, so here is the closest one. Ask for another if needed."
        )
    if constraints.get("time_fallback"):
        notes.append(
            f"Nothing fits within {constraints['max_minutes']} minutes, so here "
            "is the fastest close match."
        )
    return notes


def demo_answer(
    assets: Assets,
    hits,
    query_canon: Sequence[str],
    constraints: dict | None = None,
    more_count: int = 0,
) -> str:
    """Deterministic answer computed from the top recipe. No model call.

    Mirrors examples/demo_answer.py so the UI can be tested without Ollama and
    so the group can compare the LLM wording against a known-good baseline.
    Recommends ONE recipe; versions and extra matches are surfaced separately
    in the interface.
    """
    if not hits:
        return (
            "I couldn't find a recipe for those ingredients. Try adding a main "
            "ingredient such as chicken, pork, fish, or tofu."
        )

    constraints = constraints or {}
    row = assets.df.iloc[hits[0][0]]
    facts = version_facts(row, query_canon)
    lines = [f"Recommended: {facts['name']}", f"- Total time: ~{facts['display_time']}"]
    if _servings_text(facts["servings"]):
        lines.append(f"- Servings: {_servings_text(facts['servings'])}")

    notes = _constraint_notes(constraints)
    if notes:
        lines.append("- " + notes[0])
        for note in notes[1:]:
            lines.append(f"- {note}")

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

    if more_count:
        lines += ["", f"{more_count} other recipe(s) matched — say 'show me more' to see them."]

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

    if any(
        phrase in text
        for phrase in ("more recipe", "other recipe", "more options", "what else",
                        "anything else", "another recipe", "other options", "more dishes",
                        "what other")
    ):
        others = positions[1:4] if focus_row is None else [p for p in positions if p != focus_row][:3]
        if not others:
            return "That's everything I retrieved for your pantry. Add another ingredient or relax a filter to see more."
        lines = ["Other recipes that matched your pantry:", ""]
        for pos in others:
            facts = version_facts(assets.df.iloc[pos], query_canon)
            missing = len(facts["missing"])
            lines.append(
                f"- {facts['name']} (~{facts['display_time']}, "
                f"{missing} missing ingredient{'s' if missing != 1 else ''})"
            )
        lines += ["", "Ask for the full recipe of any of them."]
        return "\n".join(lines)

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
        "the fastest option, the fewest missing ingredients, substitutions, the "
        "cooking steps, or more recipes. Start Ollama for free-form follow-ups."
    )


def chunk_text(text: str, words: int = 3) -> Iterator[str]:
    """Split text into small pieces so demo mode streams like the real thing."""
    tokens = re.findall(r"\S+\s*", text)
    for start in range(0, len(tokens), words):
        yield "".join(tokens[start : start + words])


def sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
