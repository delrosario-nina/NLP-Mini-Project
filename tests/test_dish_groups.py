"""Small dependency-light checks for the dish-grouping core."""

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dish_groups import build_groups, collapse_near_identical, normalize_name, rank_groups  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
