"""Deterministic demo answers and SSE helpers for the web interface.

Split from the original rag.py so each file has one job:
    retrieval.py   parquet -> assets -> hybrid hits -> JSON cards
    prompts.py     hits -> LLM prompt -> streamed answer
    demo.py        hits -> deterministic answer + SSE helpers        (this file)

The demo path exists so the whole interface works without Ollama and so the
group can compare LLM wording against a known-good, data-derived baseline.
Mirrors examples/demo_answer.py.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterator, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dish_groups import version_facts  # noqa: E402
from retrieval import Assets, _steps, _servings_text  # noqa: E402

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


def _constraint_notes(constraints: dict) -> list[str]:
    """Honest one-liners about how the filters were applied."""
    notes = []
    if constraints.get("diet"):
        notes.append(f"Diet: {constraints['diet']}.")
    if constraints.get("max_minutes") is not None:
        notes.append(f"Time budget: {constraints['max_minutes']} minutes or less.")
    if constraints.get("category"):
        notes.append(f"Dish type: {constraints['category']}.")
    if constraints.get("allergens"):
        notes.append("Allergies: excludes " + ", ".join(constraints["allergens"]) + ".")
    if constraints.get("diet_fallback"):
        notes.append(
            "No fully " + str(constraints.get("diet")) + "-friendly recipe matched those "
            "ingredients, so here is the closest one. Ask for another if needed."
        )
    if constraints.get("time_fallback"):
        notes.append(
            f"Nothing fits within {constraints.get('max_minutes')} minutes, so here "
            "is the fastest close match."
        )
    if constraints.get("category_fallback"):
        notes.append(
            f"No {constraints.get('category')} dish matched those ingredients, so here "
            "is the closest one."
        )
    if constraints.get("allergen_fallback"):
        notes.append(
            "Every close match contains your allergens, so here is the best of them "
            "— please adapt it yourself."
        )
    return notes


def demo_answer(
    assets: Assets,
    hits,
    query_canon: Sequence[str],
    constraints: dict | None = None,
    more_count: int = 0,
    pantry_additions: Sequence[str] = (),
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
    lines = ["Recommended: " + facts["name"]]
    if pantry_additions:
        lines.append(
            "- Updated pantry: now includes "
            + ", ".join(pantry_additions)
            + ". Here is the best dish for it."
        )
    lines.append(f"- Total time: ~{facts['display_time']}")
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
