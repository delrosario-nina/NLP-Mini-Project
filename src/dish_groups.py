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


# --- Version comparison ---------------------------------------------------

STAPLES = {
    "salt", "water", "cooking oil", "vegetable oil", "canola oil", "olive oil",
    "sugar", "brown sugar", "black pepper", "pepper", "ground black pepper",
}
SEASONINGS = {
    "soy sauce", "fish sauce", "vinegar", "oyster sauce", "shrimp paste",
    "bagoong", "patis", "toyo", "maggi magic sarap", "liquid seasoning",
    "bouillon", "stock", "broth", "sinigang mix", "knorr", "brown sugar",
}
MAIN_INGREDIENTS = {
    "pork", "pork belly", "pork hock", "ground pork", "chicken", "chicken breast",
    "chicken thigh", "beef", "fish", "tilapia", "bangus", "shrimp", "squid",
    "milkfish", "egg", "tofu", "eggplant", "pork adobo", "leftover pork adobo",
}


def ingredient_importance(name: str) -> float:
    """Weight how costly it is to be missing an ingredient.

    Main ingredients matter most; pantry staples barely matter; unknown items
    are treated conservatively so the system does not silently understate cost.
    """
    key = str(name).strip().lower()
    if key in STAPLES:
        return 0.25
    if key in SEASONINGS:
        return 1.0
    if key in MAIN_INGREDIENTS:
        return 3.0
    if any(word in key for word in ("sauce", "powder", "mix", "seasoning", "cube")):
        return 1.0
    if any(word in key for word in ("leaf", "leaves", "onion", "garlic", "tomato", "chili")):
        return 2.0
    return 1.5


def missing_cost(row: pd.Series, pantry: Iterable[str]) -> float:
    """Sum of importance-weighted ingredients the user does not have."""
    have = {str(x).strip().lower() for x in pantry}
    return sum(ingredient_importance(i) for i in _row_ingredient_tokens(row) - have)


def _time_to_display(row: pd.Series) -> str:
    raw = row.get("total_time", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    minutes = _time_min(row, "total_time_min")
    return f"{int(minutes)} minutes" if not np.isnan(minutes) else "time unknown"


def _budget_minutes(row: pd.Series) -> float:
    """Prefer cook time for 'I have X minutes to cook'; fall back to total."""
    for col in ("cook_time_min", "total_time_min"):
        value = _time_min(row, col)
        if not np.isnan(value):
            return value
    return float("inf")


def _technique_hints(row: pd.Series) -> list[str]:
    text = f"{row.get('recipe_name', '')} {row.get('instructions', '')}".lower()
    markers = {
        "overnight": "overnight soak",
        "marinate": "marinated",
        "instant pot": "Instant Pot",
        "deep fry": "deep-fried",
        "pan fry": "pan-fried",
        "grill": "grilled",
        "bake": "baked",
        "soda": "uses soda",
        "sprite": "uses soda",
        "knorr": "uses Knorr",
        "maggi": "uses Maggi",
    }
    return [label for needle, label in markers.items() if needle in text]


def version_facts(row: pd.Series, pantry: Iterable[str]) -> dict:
    """Deterministic facts about one version, safe to hand to the LLM as-is."""
    have = {str(x).strip().lower() for x in pantry}
    tokens = _row_ingredient_tokens(row)
    matched, missing = sorted(tokens & have), sorted(tokens - have)
    servings = row.get("servings", None)
    return {
        "name": str(row.get("recipe_name", "")),
        "display_time": _time_to_display(row),
        "budget_minutes": _budget_minutes(row),
        "servings": str(servings) if servings is not None and not pd.isna(servings) else "",
        "ingredient_count": len(tokens),
        "matched": matched,
        "missing": missing,
        "missing_cost": round(missing_cost(row, pantry), 2),
        "techniques": _technique_hints(row),
    }


def _version_sort_key(item: tuple[int, dict]) -> tuple[float, float, int]:
    pos, facts = item
    return (facts["missing_cost"], facts["budget_minutes"], pos)


def format_version_comparison(
    group: VersionGroup,
    df: pd.DataFrame,
    pantry: Iterable[str],
    max_versions: int = 3,
) -> str:
    """Render a short comparison of a group's versions.

    Similarities/differences are computed from the rows, never guessed by the
    LLM. The base for the ingredient delta is the version with the lowest
    missing cost, so '+' means extra shopping and '-' means a simpler version.
    """
    facts = [(pos, version_facts(df.iloc[pos], pantry)) for pos in group.rows]
    facts.sort(key=_version_sort_key)
    facts = facts[:max_versions]
    base_pos, base_facts = facts[0]
    base_tokens = _row_ingredient_tokens(df.iloc[base_pos])

    lines = [f"We found {len(facts)} versions of {group.display_name} that fit your ingredients:"]
    for label, (pos, f) in zip("ABCDE", facts):
        tokens = _row_ingredient_tokens(df.iloc[pos])
        extra, absent = sorted(tokens - base_tokens), sorted(base_tokens - tokens)
        parts = [f"~{f['display_time']}", f"{len(f['missing'])} missing"]
        if f["techniques"]:
            parts.append("; ".join(f["techniques"]))
        detail = ", ".join(f["missing"][:4]) if f["missing"] else "nothing"
        line = f"{label}. {f['name']} — {parts[0]} · {parts[1]} ({detail})"
        if len(parts) > 2:
            line += f" · {parts[2]}"
        if extra or absent:
            delta = []
            if extra:
                delta.append("adds " + ", ".join(extra[:4]))
            if absent:
                delta.append("skips " + ", ".join(absent[:4]))
            line += " · " + "; ".join(delta) + " vs A"
        lines.append(line)
    lines.append("Which version? Or say 'pick the fastest' / 'pick the one with fewest missing'.")
    return "\n".join(lines)


def select_version(
    group: VersionGroup,
    df: pd.DataFrame,
    pantry: Iterable[str],
    choice: str | int | None = None,
) -> int:
    """Resolve a user's choice to one row position.

    Accepts a 1-based label ('1', 'A'), 'fastest', 'fewest missing', or a
    substring of the version name. Defaults to the lowest missing cost.
    """
    facts = [(pos, version_facts(df.iloc[pos], pantry)) for pos in group.rows]
    if choice is None or str(choice).strip().lower() in {"", "best", "recommended"}:
        return min(facts, key=_version_sort_key)[0]

    text = str(choice).strip().lower()
    if text in {"fastest", "quickest", "shortest"}:
        return min(facts, key=lambda item: (item[1]["budget_minutes"], item[0]))[0]
    if text in {"fewest missing", "least missing", "most complete"}:
        return min(facts, key=lambda item: (item[1]["missing_cost"], item[0]))[0]

    ordered = sorted(facts, key=_version_sort_key)
    if text.isdigit():
        idx = int(text) - 1
        if 0 <= idx < len(ordered):
            return ordered[idx][0]
    if len(text) == 1 and "a" <= text <= "z":
        idx = ord(text) - ord("a")
        if idx < len(ordered):
            return ordered[idx][0]
    for pos, f in ordered:
        if text in f["name"].lower():
            return pos
    return min(facts, key=_version_sort_key)[0]
