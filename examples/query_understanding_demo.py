"""Demo the project's query understanding without the web interface.

HOW TO USE
    python examples/query_understanding_demo.py
    python examples/query_understanding_demo.py "i want a soup, under 30 minutes"

Shows how the mechanism layer (src/query_understanding.py + src/ingredient_utils.py)
reads a conversation the way the pipeline does: canonical pantry across all
messages, plus diet / time budget / dish type / allergies with most-recent-
statement-wins resolution.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alias_map import parents_map  # noqa: E402
from ingredient_utils import build_lookup, load_alias_map  # noqa: E402
from query_understanding import (  # noqa: E402
    accumulate_pantry,
    expand_pantry,
    merge_constraints,
    parse_constraints,
)


def show(text: str, lookup) -> None:
    found = parse_constraints(text, lookup)
    print(f"  {text!r}")
    print(f"    diet={found['diet']} negated={found['diet_negated']} "
          f"time={found['max_minutes']} cleared={found['time_cleared']}")
    print(f"    type={found['category']} allergens={found['allergens']}")


def main() -> None:
    lookup, collisions = build_lookup(load_alias_map())
    if collisions:
        print("ALIAS COLLISIONS:", collisions)

    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]), lookup)
        return

    print("== single messages ==")
    show("may sitaw, bawang, at manok ako", lookup)
    show("i only have half an hour, vegetarian ako", lookup)
    show("im allergic to shrimp, isda at kamatis", lookup)
    show("i want a soup, keep it under an hour", lookup)
    show("breakfast sana, longganisa at itlog", lookup)
    show("no longer a vegetarian, no time limit", lookup)

    print("\n== whole conversation (most recent statement wins) ==")
    conversation = [
        "vegetarian, sitaw, bawang, i only have half an hour",
        "actually i now have pork belly as well",
        "im not a vegetarian anymore, i have two hours now",
    ]
    constraints = merge_constraints(lookup, conversation)
    print("  resolved:", constraints)

    pantry, unmapped = accumulate_pantry(lookup, conversation)
    print("  pantry:  ", pantry, "| unmapped:", unmapped)
    print("  expanded:", expand_pantry(parents_map(), pantry))


if __name__ == "__main__":
    main()
