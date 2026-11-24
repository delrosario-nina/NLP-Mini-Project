"""Group same-name recipe rows and treat them as comparable *versions*.

This module is opt-in: it does not change the existing alias map, retrievers,
or RAG pipeline. It only reorganises rows that already share a recipe name.

Two behaviours matter:

1. Rows with an identical normalised name are grouped, so a top-k list does
   not waste two slots on the same dish.
2. Within a group, rows that are practically identical (high ingredient
   overlap, near-identical instructions, similar time) are merged silently.
   Only genuinely different versions are offered to the user.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

_NAME_SUFFIX = re.compile(r"\s+recipe\s*$", flags=re.IGNORECASE)
_WS = re.compile(r"\s+")


def normalize_name(name: object) -> str:
    """Normalise a recipe name for grouping: lowercase, collapse spaces, drop trailing 'Recipe'."""
    if not isinstance(name, str):
        return ""
    s = _NAME_SUFFIX.sub("", name.strip())
    return _WS.sub(" ", s).lower()


@dataclass
class VersionGroup:
    """One dish name plus the row positions of its versions."""

    name_key: str
    display_name: str
    rows: list[int] = field(default_factory=list)
    merged_rows: list[int] = field(default_factory=list)
    score: float = 0.0


def _row_ingredient_tokens(row: pd.Series) -> set[str]:
    """Canonical tokens if notebook 02 was run, otherwise the raw ingredient list."""
    for col in ("canonical_ingredients", "ingredient_list", "ingredient_names"):
        value = row.get(col, None)
        if isinstance(value, (list, tuple, set)):
            return {str(x).strip().lower() for x in value if str(x).strip()}
        if isinstance(value, str) and value.strip():
            return {x.strip().lower() for x in value.split("|") if x.strip()}
    return set()


def _time_min(row: pd.Series, column: str) -> float:
    value = row.get(column, None)
    if value is None or pd.isna(value):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def _instruction_similarity(a: object, b: object) -> float:
    return SequenceMatcher(None, str(a), str(b)).ratio()


def collapse_near_identical(
    df: pd.DataFrame,
    rows: Sequence[int],
    ing_jaccard: float = 0.8,
    instr_sim: float = 0.75,
    time_diff: float = 10.0,
) -> tuple[list[int], list[int]]:
    """Collapse rows that are practically the same recipe.

    Keeps the row with the longest instructions per cluster, matching the rule
    already used in notebook 01, and returns (kept_rows, merged_rows).
    """
    rows = list(rows)
    if len(rows) < 2:
        return rows, []

    parent = {r: r for r in rows}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            ra, rb = df.iloc[a], df.iloc[b]
            ta, tb = _time_min(ra, "total_time_min"), _time_min(rb, "total_time_min")
            dt = abs(ta - tb) if not (np.isnan(ta) or np.isnan(tb)) else float("inf")
            if (
                _jaccard(_row_ingredient_tokens(ra), _row_ingredient_tokens(rb)) >= ing_jaccard
                and _instruction_similarity(ra.get("instructions", ""), rb.get("instructions", "")) >= instr_sim
                and dt <= time_diff
            ):
                parent[find(a)] = find(b)

    clusters: dict[int, list[int]] = {}
    for r in rows:
        clusters.setdefault(find(r), []).append(r)

    kept, merged = [], []
    for cluster in clusters.values():
        keeper = max(cluster, key=lambda r: len(str(df.iloc[r].get("instructions", ""))))
        kept.append(keeper)
        merged.extend(r for r in cluster if r != keeper)
    return sorted(kept), sorted(merged)


def build_groups(
    df: pd.DataFrame,
    merge_near_identical: bool = True,
    ing_jaccard: float = 0.8,
    instr_sim: float = 0.75,
    time_diff: float = 10.0,
) -> list[VersionGroup]:
    """Group every row by normalised recipe name.

    With ``merge_near_identical=True``, near-identical rows inside a group are
    removed from ``rows`` and recorded in ``merged_rows`` for auditing.
    """
    buckets: dict[str, list[int]] = {}
    for pos, name in enumerate(df["recipe_name"]):
        buckets.setdefault(normalize_name(name), []).append(pos)

    groups = []
    for key, rows in buckets.items():
        if len(rows) > 1 and merge_near_identical:
            kept, merged = collapse_near_identical(
                df, rows, ing_jaccard=ing_jaccard, instr_sim=instr_sim, time_diff=time_diff
            )
        else:
            kept, merged = rows, []
        groups.append(VersionGroup(key, str(df.iloc[kept[0]]["recipe_name"]), kept, merged))
    return sorted(groups, key=lambda g: g.rows[0])


def rank_groups(groups: Iterable[VersionGroup], row_scores: Sequence[float], k: int = 3) -> list[VersionGroup]:
    """Rank groups by their best-scoring row, then keep the top ``k`` groups."""
    scores = np.asarray(list(row_scores), dtype=float)
    ranked = []
    for group in groups:
        group.score = float(max(scores[r] for r in group.rows))
        ranked.append(group)
    ranked.sort(key=lambda g: (-g.score, g.rows[0]))
    return ranked[:k]
