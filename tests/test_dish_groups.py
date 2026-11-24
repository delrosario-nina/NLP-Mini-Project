"""Small dependency-light checks for the dish-grouping core.

HOW TO USE
    From the repository root:  python -m unittest discover -s tests -v
    Just this file:            python -m unittest tests.test_dish_groups -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dish_groups import (  # noqa: E402
    build_groups,
    collapse_near_identical,
    format_version_comparison,
    ingredient_importance,
    normalize_name,
    rank_groups,
    select_version,
    version_facts,
    version_positions,
)


def make_df(rows):
    return pd.DataFrame(
        [
            {
                "recipe_name": name,
                "ingredients": " | ".join(ings),
                "instructions": instr,
                "ingredient_list": ings,
                "total_time_min": minutes,
            }
            for name, ings, instr, minutes in rows
        ]
    )


class TestNormalizeName(unittest.TestCase):
    def test_strips_suffix_case_and_spaces(self):
        self.assertEqual(normalize_name("  Pork Adobo Recipe  "), "pork adobo")
        self.assertEqual(normalize_name("Chicken Inasal"), "chicken inasal")

    def test_empty_for_non_string(self):
        self.assertEqual(normalize_name(None), "")


class TestBuildGroups(unittest.TestCase):
    def setUp(self):
        self.df = make_df(
            [
                ("Pork Adobo", ["pork", "vinegar", "soy sauce"], "brown pork then simmer", 70),
                ("Pork Adobo", ["pork", "vinegar", "soy sauce", "soda"], "brown pork, add soda, simmer", 70),
                ("Sinigang", ["pork", "tamarind"], "boil everything", 45),
            ]
        )

    def test_same_name_rows_group_together(self):
        groups = build_groups(self.df, merge_near_identical=False)
        self.assertEqual([g.name_key for g in groups], ["pork adobo", "sinigang"])
        self.assertEqual(groups[0].rows, [0, 1])
        self.assertEqual(groups[1].rows, [2])

    def test_near_identical_rows_are_merged_but_audited(self):
        df = make_df(
            [
                ("Dinuguan", ["pork", "vinegar", "oil"], "saute pork, add vinegar and simmer", 60),
                ("Dinuguan", ["pork", "vinegar", "oil"], "saute pork in oil, add vinegar and simmer until thick", 60),
            ]
        )
        groups = build_groups(df)
        self.assertEqual(len(groups[0].rows), 1)
        self.assertEqual(groups[0].merged_rows, [0])  # shorter instructions dropped

    def test_different_versions_are_kept(self):
        df = make_df(
            [
                ("Crispy Pata", ["pork leg", "salt"], "boil then fry", 100),
                ("Crispy Pata", ["pork leg", "soda", "garlic"], "soak overnight then fry", 630),
            ]
        )
        groups = build_groups(df)
        self.assertEqual(sorted(groups[0].rows), [0, 1])


class TestRankGroups(unittest.TestCase):
    def test_group_uses_best_row_and_does_not_repeat(self):
        groups = build_groups(self.df(), merge_near_identical=False)
        ranked = rank_groups(groups, [0.1, 0.9, 0.5], k=2)
        self.assertEqual([g.name_key for g in ranked], ["pork adobo", "sinigang"])
        self.assertAlmostEqual(ranked[0].score, 0.9)
        self.assertEqual(sorted(r for g in ranked for r in g.rows), [0, 1, 2])

    @staticmethod
    def df():
        return make_df(
            [
                ("Pork Adobo", ["pork"], "a", 70),
                ("Pork Adobo", ["pork", "soda"], "b", 70),
                ("Sinigang", ["pork"], "c", 45),
            ]
        )


class TestVersionComparison(unittest.TestCase):
    def setUp(self):
        # Two genuinely different versions: the second needs one more item.
        self.df = make_df(
            [
                ("Chicken Inasal", ["chicken", "vinegar", "garlic"], "grill the chicken", 35),
                ("Chicken Inasal", ["chicken", "vinegar", "garlic", "soda", "soy sauce"], "marinate overnight then grill", 78),
            ]
        )
        self.group = build_groups(self.df)[0]

    def test_importance_weights(self):
        self.assertEqual(ingredient_importance("salt"), 0.25)
        self.assertEqual(ingredient_importance("pork belly"), 3.0)
        self.assertEqual(ingredient_importance("soy sauce"), 1.0)
        self.assertEqual(ingredient_importance("mystery ingredient"), 1.5)

    def test_canonical_array_from_parquet_is_used(self):
        row = pd.Series({
            "recipe_name": "Adobo",
            "canonical_ingredients": np.array(["chicken", "vinegar"]),
            "ingredient_list": ["raw item"],
        })
        self.assertEqual(version_facts(row, ["chicken"])["missing"], ["vinegar"])

    def test_facts_report_matched_and_missing(self):
        f = version_facts(self.df.iloc[0], ["chicken", "vinegar"])
        self.assertEqual(f["missing"], ["garlic"])
        self.assertEqual(f["matched"], ["chicken", "vinegar"])
        self.assertGreater(f["missing_cost"], 0)

    def test_comparison_is_deterministic_and_mentions_delta(self):
        text = format_version_comparison(self.group, self.df, ["chicken", "vinegar"])
        self.assertIn("2 versions of Chicken Inasal", text)
        self.assertIn("Which version?", text)
        self.assertIn("adds", text)
        self.assertIn("overnight", text)

    def test_selection_choices(self):
        self.assertEqual(select_version(self.group, self.df, [], "fastest"), 0)
        self.assertEqual(select_version(self.group, self.df, [], "fewest missing"), 0)
        self.assertEqual(select_version(self.group, self.df, [], "2"), 1)
        self.assertEqual(select_version(self.group, self.df, [], "A"), 0)

    def test_default_uses_weighted_cost_but_fewest_missing_uses_count(self):
        df = make_df(
            [
                ("Adobo", ["pork", "vinegar"], "a", 60),
                ("Adobo", ["salt", "water", "sugar", "garlic"], "b", 90),
            ]
        )
        group = build_groups(df)[0]
        self.assertEqual(select_version(group, df, []), 1)  # staples are cheap to be missing
        self.assertEqual(
            select_version(group, df, [], "fewest missing"), 0
        )  # but 1 < 4 literal items

    def test_display_always_includes_fastest_version(self):
        df = make_df(
            [
                ("Pata", ["pork"], "a", 600),
                ("Pata", ["pork", "salt"], "b", 540),
                ("Pata", ["pork", "soy sauce"], "c", 40),
            ]
        )
        group = build_groups(df)[0]
        self.assertIn(2, version_positions(group, df, [], max_versions=2))


if __name__ == "__main__":
    unittest.main()
