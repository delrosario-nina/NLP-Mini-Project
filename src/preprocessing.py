"""Shared preprocessing helpers for the Pinoy Recipe RAG project.

HOW TO USE
    import sys; sys.path.insert(0, "src")
    from preprocessing import prepare_dataset
    raw = pd.read_parquet("data/raw/raw_recipes.parquet")
    clean, report = prepare_dataset(raw)   # writes nothing; you save it yourself

    Notebook 01 still works; this module is the maintained version of its
    cleaning, time parsing, and near-duplicate logic. It uses only pandas,
    numpy, and the standard library.

The functions are intentionally pure: they take a DataFrame and return a new
one, so they can be unit-tested without running the whole notebook.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd

from ingredient_utils import DROP

try:  # notebook 01 already installs rapidfuzz; keep the same similarity measure
    from rapidfuzz.fuzz import token_set_ratio
except ImportError:  # dependency-free fallback for unit tests
    token_set_ratio = None

TEXT_COLUMNS = [
    "recipe_name", "prep_time", "cook_time", "total_time",
    "servings", "ingredients", "instructions", "ingredient_names",
]
TIME_COLUMNS = ["prep_time", "cook_time", "total_time"]

_TIME_UNITS = {"day": 1440, "hr": 60, "min": 1}
_TIME_RE = re.compile(r"(\d+(?:\s+\d+/\d+)?|\d+/\d+|\d+\.\d+)\s*(day|hr|min)")


def clean_text(value: object) -> object:
    """Normalise unicode, strip HTML tags, and collapse whitespace."""
    if not isinstance(value, str):
        return value
    s = unicodedata.normalize("NFKC", value)
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def split_pipe(value: object) -> list[str]:
    """Split a pipe-separated field into cleaned, non-empty parts."""
    if not isinstance(value, str):
        return []
    return [item for item in (clean_text(part) for part in value.split("|")) if item]


def _to_float(token: str) -> float:
    token = token.strip()
    if " " in token:
        whole, frac = token.split(maxsplit=1)
        return float(whole) + _to_float(frac)
    if "/" in token:
        num, den = token.split("/", 1)
        return float(num) / float(den) if float(den) else 0.0
    return float(token)


def parse_minutes(value: object) -> float | None:
    """Parse time strings to minutes.

    Handles '15 mins', '1 hour 30 minutes', '1 1/2 hours', '2 days'.
    Returns None when no time expression is found.
    """
    if not isinstance(value, str):
        return None
    s = value.lower()
    s = re.sub(r"\bmins?\b|\bminutes?\b", "min", s)
    s = re.sub(r"\bhrs?\b|\bhours?\b", "hr", s)
    s = re.sub(r"\bdays?\b", "day", s)
    found = _TIME_RE.findall(s)
    if not found:
        return None
    return int(sum(_to_float(amount) * _TIME_UNITS[unit] for amount, unit in found))


def parse_servings(value: object) -> tuple[int | None, int | None]:
    """Return (min, max) servings found in a string such as '4 to 6'."""
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", str(value))] if isinstance(value, str) else []
    return (int(min(nums)), int(max(nums))) if nums else (None, None)


def ingredient_tokens(lines: Iterable[str]) -> set[str]:
    """Token set used for duplicate detection (units and prep words removed)."""
    tokens: set[str] = set()
    for line in lines:
        s = re.sub(r"\([^)]*\)", " ", str(line).lower())
        s = re.sub(r"[^a-z\s]", " ", s)
        tokens |= {t for t in s.split() if t not in DROP and len(t) > 1}
    return tokens


def prepare_recipes(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean a raw recipe DataFrame without dropping or renumbering anything yet."""
    df = raw.copy()
    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(clean_text)

    df["recipe_name"] = df["recipe_name"].str.replace(r"\s+recipe$", "", regex=True, case=False)
    df["ingredient_lines"] = df["ingredients"].map(split_pipe)
    df["ingredient_list"] = df["ingredient_names"].map(
        lambda s: list(dict.fromkeys(x.lower() for x in split_pipe(s)))
    )

    for col in TIME_COLUMNS:
        df[col + "_min"] = df[col].map(parse_minutes)
    df[["servings_min", "servings_max"]] = pd.DataFrame(
        df["servings"].map(parse_servings).tolist(),
        columns=["servings_min", "servings_max"],
        index=df.index,
    )

    before = len(df)
    df = df[(df["ingredient_lines"].str.len() > 0) & (df["instructions"].str.len() > 0)]
    return df, {"dropped_empty": before - len(df)}


def find_near_duplicate_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Score same-name row pairs by ingredient overlap, instruction similarity, and time gap."""
    rows = []
    for name, group in df.groupby(df["recipe_name"].str.lower()):
        if len(group) < 2:
            continue
        for (i, a), (j, b) in combinations(group.iterrows(), 2):
            ta, tb = a.get("total_time_min"), b.get("total_time_min")
            t_diff = abs(float(ta or 0) - float(tb or 0))
            if token_set_ratio is not None:
                instr_sim = token_set_ratio(str(a["instructions"]), str(b["instructions"])) / 100
            else:
                instr_sim = SequenceMatcher(None, str(a["instructions"]), str(b["instructions"])).ratio()
            rows.append((
                name, i, j,
                round(_jaccard(ingredient_tokens(a["ingredient_lines"]),
                               ingredient_tokens(b["ingredient_lines"])), 2),
                round(instr_sim, 2),
                t_diff,
            ))
    return pd.DataFrame(rows, columns=["name", "i", "j", "ing_jaccard", "instr_sim", "time_diff"])


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def duplicate_clusters(
    pairs: pd.DataFrame,
    df: pd.DataFrame | None = None,
    high_jaccard: float = 0.8,
    high_time_diff: float = 10,
    review_jaccard: float = 0.4,
    review_instr_sim: float = 0.75,
    review_time_diff: float = 20,
    confirmed_pairs: Iterable[tuple[int, int]] = (),
) -> tuple[list[list[int]], list[int], pd.DataFrame]:
    """Return (clusters to merge, rows to drop, pairs needing review).

    Tier 1 (very high ingredient overlap, small time gap) and explicitly
    confirmed pairs are merged; tier 2 is only reported for manual review, as
    in notebook 01. Within each cluster the longest instruction row is kept.
    """
    tier1 = pairs[(pairs.ing_jaccard >= high_jaccard) & (pairs.time_diff <= high_time_diff)]
    tier2 = pairs[
        (pairs.ing_jaccard < high_jaccard)
        & (pairs.ing_jaccard >= review_jaccard)
        & (pairs.instr_sim >= review_instr_sim)
        & (pairs.time_diff <= review_time_diff)
    ]
    edges = set(zip(tier1["i"], tier1["j"])) | set(confirmed_pairs)

    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)

    groups: dict[int, list[int]] = defaultdict(list)
    for row in list(parent):
        groups[find(row)].append(row)
    clusters = sorted((sorted(v) for v in groups.values() if len(v) > 1), key=lambda c: c[0])
    if df is None:
        keep = {cluster[-1] for cluster in clusters}
    else:
        keep = {
            max(cluster, key=lambda r: len(str(df.loc[r, "instructions"])))
            for cluster in clusters
        }
    dropped = sorted(row for cluster in clusters for row in cluster if row not in keep)
    return clusters, dropped, tier2


def prepare_dataset(
    raw: pd.DataFrame,
    drop_near_duplicates: bool = True,
    confirmed_pairs: Iterable[tuple[int, int]] = (),
) -> tuple[pd.DataFrame, dict]:
    """Full preprocessing pipeline. Returns (clean DataFrame, small report dict).

    ``confirmed_pairs`` are tier-2 pairs a human approved for merging, matching
    notebook 01's ``confirmed`` list.
    """
    df, report = prepare_recipes(raw)
    pairs = find_near_duplicate_pairs(df)
    report["pairs_compared"] = len(pairs)
    if drop_near_duplicates and not pairs.empty:
        clusters, dropped, review = duplicate_clusters(pairs, df, confirmed_pairs=confirmed_pairs)
        report["review_pairs"] = len(review)
        df = df.drop(index=dropped)
        report["clusters"] = len(clusters)
        report["rows_removed"] = len(dropped)
    else:
        report["clusters"] = 0
        report["rows_removed"] = 0

    df = df.reset_index(drop=True)
    df.insert(0, "recipe_id", range(len(df)))
    report["rows_final"] = len(df)
    return df, report
