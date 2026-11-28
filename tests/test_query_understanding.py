"""Tests for src/query_understanding.py — the constraint and pantry mechanism.

HOW TO USE
    From the repository root:  python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alias_map import parents_map  # noqa: E402
from ingredient_utils import build_lookup, load_alias_map  # noqa: E402
from query_understanding import (  # noqa: E402
    accumulate_pantry,
    apply_constraints,
    category_positions,
    expand_pantry,
    is_novelty,
    merge_constraints,
    pantry_changed,
    parse_constraints,
    rank_by_pantry_fit,
)

import pandas as pd  # noqa: E402


def fake_df():
    """A tiny recipes frame with the columns the filters and facts read."""
    return pd.DataFrame(
        {
            "recipe_name": [
                "Pork Adobo",
                "Chicken Tinola",
                "Chicken Tinola Onigiri Bento",
                "Ginisang Gulay",
                "Shrimp Sinigang",
            ],
            "canonical_ingredients": [
                ["pork_belly", "soy sauce", "vinegar", "garlic"],
                ["chicken", "ginger", "garlic", "onion"],
                ["rice", "chicken", "cheese", "tomato"],
                ["string beans", "garlic", "tomato"],
                ["shrimp", "tomato", "onion", "water"],
            ],
            "ingredients": [
                "1 lb pork belly|2 tbsp soy sauce|1/4 cup vinegar|3 cloves garlic",
                "1 whole chicken|2 thumbs ginger|3 cloves garlic|1 onion",
                "2 cups rice|1 chicken breast|1 slice cheese|1 tomato",
                "200 g string beans|3 cloves garlic|2 tomatoes",
                "1/2 kg shrimp|3 tomatoes|1 onion|2 cups water",
            ],
            "instructions": [
                "Marinate.|Simmer until tender.|Reduce the sauce.",
                "Saute aromatics.|Add chicken and water.|Simmer.",
                "Mold rice balls.|Decorate with cheese and nori.",
                "Saute garlic.|Add string beans and tomatoes.|Toss.",
                "Boil water.|Add vegetables and shrimp.|Simmer briefly.",
            ],
            "total_time_min": [45.0, 40.0, 30.0, 23.0, 35.0],
            "cook_time_min": [30.0, 25.0, 30.0, 15.0, 20.0],
            "total_time": ["45 minutes", "40 minutes", "30 minutes", "23 minutes", "35 minutes"],
            "servings": [4, 4, 2, 4, 4],
        }
    )


class QueryUnderstandingTestCase(unittest.TestCase):
    def setUp(self):
        self.lookup, _ = build_lookup(load_alias_map())
        self.parents = parents_map()
        self.df = fake_df()
        self.df["canon_list"] = self.df["canonical_ingredients"].map(list)
        self.df["canon_match"] = self.df["canon_list"].map(
            lambda lst: set(lst) | {self.parents[c] for c in lst if c in self.parents}
        )

    def constraints(self, **kwargs):
        base = {"diet": None, "max_minutes": None, "category": None, "allergens": []}
        base.update(kwargs)
        return base

    # --- time budgets -------------------------------------------------------

    def test_time_budgets_parse(self):
        cases = {
            "i only have half an hour": 30,
            "keep it under an hour": 60,
            "45 minutes max": 45,
            "isang oras lang": 60,
            "30 minuto lang ako magluto": 30,
            "i have two hours": 120,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_constraints(text)["max_minutes"], expected)

    def test_time_clearing(self):
        for text in ("no time limit", "no rush", "kahit gaano katagal"):
            with self.subTest(text=text):
                self.assertTrue(parse_constraints(text)["time_cleared"])

    def test_new_time_replaces_old_not_min(self):
        merged = merge_constraints(self.lookup, [
            "sitaw, under an hour",
            "actually i have two hours now",
        ])
        self.assertEqual(merged["max_minutes"], 120)

    # --- diets and their negations ------------------------------------------

    def test_diet_detection(self):
        cases = {
            "vegetarian ako": "vegetarian",
            "im a pescatarian": "pescatarian",
            "vegan please": "vegan",
            "gulay lang ako": "vegetarian",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_constraints(text, self.lookup)["diet"], expected)

    def test_diet_negation_clears_the_tag(self):
        # "no longer a vegetarian" contains the word "vegetarian"; the parser
        # must treat it as a negation, not as the diet itself.
        cases = [
            "no longer a vegetarian",
            "im not vegetarian anymore",
            "hindi na ako vegetarian",
            "i eat meat again",
            "quit being vegan",
        ]
        for text in cases:
            with self.subTest(text=text):
                found = parse_constraints(text, self.lookup)
                self.assertIsNone(found["diet"])
                self.assertTrue(found["diet_negated"])

    def test_conversation_resolution_most_recent_wins(self):
        merged = merge_constraints(self.lookup, [
            "vegetarian, sitaw, i only have half an hour",
            "im not a vegetarian anymore, i have two hours now",
        ])
        self.assertIsNone(merged["diet"])
        self.assertEqual(merged["max_minutes"], 120)

    # --- dish types -----------------------------------------------------------

    def test_dish_type_detection(self):
        cases = {
            "i want a soup with tomatoes": "soup",
            "give me a stew with beef": "stew",
            "breakfast sana, longganisa": "breakfast",
            "pancit for merienda": "noodles",
            "something grilled, liempo": "grilled",
            "dessert, gata": "dessert",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_constraints(text)["category"], expected)

    def test_novelty_recipes_are_not_dishes_of_a_type(self):
        self.assertTrue(is_novelty("Chicken Tinola Onigiri Bento"))
        self.assertFalse(is_novelty("Chicken Tinola"))

    def test_category_positions_exclude_novelty(self):
        positions = category_positions(self.df, "soup")
        names = {self.df.iloc[p]["recipe_name"] for p in positions}
        self.assertIn("Chicken Tinola", names)
        self.assertNotIn("Chicken Tinola Onigiri Bento", names)

    # --- allergies ------------------------------------------------------------

    def test_allergen_phrases(self):
        cases = {
            "im allergic to shrimp": ["shrimp"],
            "peanut allergy": ["peanut", "peanut butter"],
            "may allergy ako sa hipon at mani": ["shrimp", "peanut"],
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_constraints(text, self.lookup)["allergens"], expected)

    def test_allergen_group_expands_seafood(self):
        bans = parse_constraints("seafood allergy", self.lookup)["allergens"]
        self.assertIn("shrimp", bans)
        self.assertIn("fish", bans)
        self.assertIn("fish sauce", bans)

    def test_allergies_accumulate_across_messages(self):
        merged = merge_constraints(self.lookup, [
            "manok, kamatis",
            "im allergic to shrimp",
            "also allergic to peanuts",
        ])
        self.assertIn("shrimp", merged["allergens"])
        self.assertIn("peanut butter", merged["allergens"])

    # --- pantry ------------------------------------------------------------------

    def test_pantry_accumulates_across_messages(self):
        canon, _ = accumulate_pantry(self.lookup, [
            "toyo, suka, bawang",
            "actually i now have pork belly as well",
        ])
        self.assertIn("pork_belly", canon)
        self.assertIn("soy sauce", canon)

    def test_pantry_changed_on_new_ingredient(self):
        texts = ["toyo, suka, bawang", "actually i now have pork belly as well"]
        self.assertTrue(pantry_changed(self.lookup, texts[:-1], texts))

    def test_pantry_unchanged_on_plain_questions(self):
        texts = ["toyo, suka, bawang", "show me the cooking steps"]
        self.assertFalse(pantry_changed(self.lookup, texts[:-1], texts))

    def test_expand_pantry_adds_parents(self):
        self.assertEqual(
            expand_pantry(self.parents, ["pork_belly"]),
            ["pork_belly", "pork"],
        )
        self.assertEqual(expand_pantry(self.parents, ["tilapia"]), ["tilapia", "fish"])

    # --- filtering and ranking -----------------------------------------------------

    def test_diet_filter_and_honest_fallback(self):
        constraints = self.constraints(diet="vegetarian")
        hits = [(0, 1.0), (1, 0.9), (3, 0.8)]
        kept, report = apply_constraints(self.df, hits, ["string beans"], constraints)
        self.assertEqual([pos for pos, _ in kept], [3])
        self.assertFalse(report["diet_fallback"])

        # Only meat dishes retrieved -> filter gives up and says so.
        kept, report = apply_constraints(self.df, [(0, 1.0)], ["pork"], constraints)
        self.assertEqual([pos for pos, _ in kept], [0])
        self.assertTrue(report["diet_fallback"])

    def test_time_filter_uses_cook_time_first(self):
        constraints = self.constraints(max_minutes=20)
        kept, report = apply_constraints(
            self.df, [(0, 1.0), (3, 0.9), (4, 0.8)], ["string beans"], constraints
        )
        self.assertEqual([pos for pos, _ in kept], [3, 4])
        self.assertFalse(report["time_fallback"])

    def test_allergen_filter_removes_shrimp_dishes(self):
        constraints = self.constraints(allergens=["shrimp"])
        kept, report = apply_constraints(
            self.df, [(1, 1.0), (4, 0.9)], ["chicken", "tomato"], constraints
        )
        self.assertEqual([pos for pos, _ in kept], [1])
        self.assertFalse(report["allergen_fallback"])

    def test_rank_by_pantry_fit_prefers_fewer_missing_on_ties(self):
        # Row 4 (Shrimp Sinigang) matches shrimp like row 2 does but is missing
        # more of what the user has around; row 2 must rank first.
        pantry = ["shrimp", "tomato", "onion"]
        ranked = rank_by_pantry_fit(self.df, [(4, 1.0), (1, 0.9)], pantry)
        self.assertEqual(ranked[0][0], 4)
        # Chicken Tinola has no shrimp, so it loses on matched weight.
        self.assertEqual(ranked[1][0], 1)

    def test_rank_demotes_novelty_recipes_on_ties(self):
        pantry = ["tomato"]
        # Both rows match tomato with the same importance weight; the tie must
        # be broken by novelty, so the real dish ranks ahead of the bento.
        # (A novelty recipe that matches strictly MORE of the pantry can still
        # win — the demotion is a tie-breaker, not an override.)
        ranked = rank_by_pantry_fit(self.df, [(2, 1.0), (4, 0.9)], pantry)
        self.assertEqual(ranked[0][0], 4)
        self.assertEqual(ranked[-1][0], 2)


if __name__ == "__main__":
    unittest.main()
