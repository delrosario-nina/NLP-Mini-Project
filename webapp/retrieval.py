"""Data loading, retrieval scoring, and recipe cards for the web interface.

Split from the original rag.py so each file has one job:
    retrieval.py   parquet -> assets -> hybrid hits -> JSON cards   (this file)
    prompts.py     hits -> LLM prompt -> streamed answer
    demo.py        hits -> deterministic answer + SSE helpers

The NLP mechanism (alias map, phrase matching, query understanding, dish
groups) lives in ``src/`` and is imported here; this module only handles the
webapp's data shape (the Assets dataclass) and the retrieval port of
notebooks/04_rag_pipeline.ipynb.

HOW TO USE
    From the repository root:
        uvicorn app:app --app-dir webapp --reload --port 8000

It only reads local artifacts and never writes to tracked files:
    data/processed/cleaned_recipes.parquet   (notebooks 01 + 02)
    data/processed/alias_map.json            (notebook 02)
    data/processed/recipe_vecs_*.npy         (notebook 04; rebuilt if absent)
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from alias_map import parents_map  # noqa: E402
from dish_groups import (  # noqa: E402
    build_groups,
    normalize_name,
    version_facts,
    version_positions,
)
from ingredient_utils import build_lookup, load_alias_map, parse_query  # noqa: E402

# Same constants as notebook 04.
EMBED_MODEL = "intfloat/multilingual-e5-small"
TOP_K = 3
RRF_K = 60
HYBRID_PARTS = ["bm25", "dense"]
METHODS = ("overlap", "bm25", "dense", "hybrid")
MAX_STEPS = 8


class AssetsMissing(RuntimeError):
    """Raised when the notebook artifacts the app depends on are absent."""


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


def retrieval_query(pantry: str, followup: str, lookup: dict) -> str:
    """Keep the pantry, but extend it when a follow-up names new ingredients."""
    base, _ = parse_query(pantry, lookup)
    extra, _ = parse_query(followup, lookup)
    if set(extra) - set(base):
        return f"{pantry}, {followup}"
    return pantry


# --- query understanding lives in src/query_understanding.py -----------------
# The mechanism (time budgets, diets and their negations, dish types, allergies,
# pantry accumulation, constraint filtering, pantry-fit ranking) is a project
# module shared with the notebook pipeline; this file only adapts it to the
# Assets dataclass the webapp loads.

from query_understanding import (  # noqa: E402
    accumulate_pantry,
    category_positions,
    merge_constraints,
    pantry_changed,
)
from query_understanding import apply_constraints as _apply_query_constraints  # noqa: E402
from query_understanding import expand_pantry as _expand_canonicals  # noqa: E402
from query_understanding import rank_by_pantry_fit as _rank_by_pantry_fit  # noqa: E402


def expand_pantry(assets: Assets, canon: Sequence[str]) -> list[str]:
    """Pantry canonicals plus their generic parents (kasim -> pork shoulder, pork)."""
    return _expand_canonicals(assets.parents, canon)


def apply_constraints(assets: Assets, hits, pantry, constraints):
    """Filter retrieved recipes by the resolved constraints (see src module)."""
    return _apply_query_constraints(assets.df, hits, pantry, constraints)


def rank_by_pantry_fit(assets: Assets, hits, pantry: Sequence[str]):
    """Order retrieved recipes by pantry fit (see src module)."""
    return _rank_by_pantry_fit(assets.df, hits, pantry)


# --- retrieval scoring (port of notebook 04) ---------------------------------


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
    restrict_positions: Sequence[int] | None = None,
) -> tuple[list[tuple[int, float]], list[str], list[str]]:
    """Return ([(row_position, score), ...], canonicals, unmapped tokens).

    ``restrict_positions`` limits scoring to the given rows — used when the
    user asks for a dish type (soup, breakfast, ...), which defines the
    candidate universe instead of being a post-filter on a pool that hybrid
    fusion may have filled with non-soup dishes.
    """
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

    if restrict_positions is not None:
        mask = np.full(len(scores), -np.inf)
        allowed = np.asarray(list(restrict_positions), dtype=int)
        if len(allowed):
            mask[allowed] = 0.0
        scores = scores + mask

    top = np.argsort(-scores, kind="stable")[:k]
    hits = [(int(pos), float(scores[pos])) for pos in top if np.isfinite(scores[pos])]
    return hits, canon, unmapped


# --- JSON-safe recipe cards --------------------------------------------------


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
