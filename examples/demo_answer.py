"""Manual demo: print a conversation that looks like an LLM answer.

HOW TO USE
    From the repository root:  python examples/demo_answer.py
    Automated check:           python -m unittest discover -s tests -v

Everything printed here is computed from the recipe rows by src/dish_groups.py;
no model call is made. This is a deterministic stand-in you can run to inspect
the exact wording the user would see, before we wire in Ollama.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from dish_groups import (  # noqa: E402
    build_groups,
    format_version_comparison,
    select_version,
    version_facts,
    version_positions,
)

PANTRY = ["chicken", "vinegar", "garlic", "onion"]
DISH = "chicken inasal"

# Demo-only substitution ideas. These are illustrative, not learned from the dataset.
SUBSTITUTIONS = {
    "annatto oil": "achuete seeds steeped in warm oil",
    "annatto seed": "achuete seeds steeped in warm oil",
    "palm vinegar": "coconut vinegar or white vinegar",
    "calamansi juice": "lemon juice",
    "coconut vinegar": "white vinegar with a squeeze of calamansi",
    "lemongrass": "tanglad, or 1/2 teaspoon dried lemongrass",
    "margarine": "butter",
    "brown sugar": "white sugar with a little molasses",
    "whole chicken": "chicken thighs or breast pieces",
    "bay leaf": "dried bay leaves",
}


def load_recipes() -> tuple[pd.DataFrame, str]:
    for rel in (
        "data/processed/cleaned_recipes.parquet",
        "notebooks/data/processed/cleaned_recipes.parquet",
    ):
        path = ROOT / rel
        if path.exists():
            return pd.read_parquet(path), rel
    raise SystemExit(
        "No recipe parquet found. Run notebooks 01 and 02 first, or regenerate "
        "data/processed/cleaned_recipes.parquet."
    )


def simplify_steps(row: pd.Series, limit: int = 6) -> list[str]:
    steps = [s.strip() for s in str(row.get("instructions", "")).split("|") if s.strip()]
    return steps[:limit]


def render_answer(row: pd.Series, pantry: list[str]) -> str:
    facts = version_facts(row, pantry)
    lines = [
        f"Recommended: {facts['name']} (dataset row {int(row.name)})",
        f"- Total time: ~{facts['display_time']}; active cook time: ~{int(facts['active_minutes'])} min",
    ]
    if facts["servings"]:
        lines.append(f"- Servings: {facts['servings']}")

    lines.append("Matched ingredients (you have): " + (", ".join(facts["matched"]) or "none"))
    lines.append("Missing ingredients: " + (", ".join(facts["missing"]) or "none"))

    lines.append("Suggested substitutions:")
    found = False
    for item in facts["missing"]:
        if item in SUBSTITUTIONS:
            lines.append(f"- {item} -> {SUBSTITUTIONS[item]}")
            found = True
    if not found:
        lines.append("- No reliable substitution from the retrieved recipe.")

    lines.append("Simplified steps:")
    for i, step in enumerate(simplify_steps(row), start=1):
        lines.append(f"{i}. {step}")
    return "\n".join(lines)


def main() -> None:
    df, rel = load_recipes()
    groups = build_groups(df)
    group = next((g for g in groups if g.name_key == DISH), None)
    if group is None:
        raise SystemExit(f"No rows named {DISH!r} found in {rel}")

    eligible = version_positions(group, df, PANTRY)
    comparison = format_version_comparison(group, df, PANTRY)

    print("=== SIMULATED CONVERSATION (deterministic, no Ollama) ===")
    print("Data source:", rel)
    print()
    print("USER: I have chicken, vinegar, garlic and onion. Show me versions of chicken inasal.")
    print()
    print("ASSISTANT:")
    print(comparison)
    print()
    print("USER: pick the fastest")
    print()
    chosen = select_version(group, df, PANTRY, "fastest", eligible=eligible)
    print("ASSISTANT:")
    print(render_answer(df.iloc[chosen], PANTRY))


if __name__ == "__main__":
    main()
