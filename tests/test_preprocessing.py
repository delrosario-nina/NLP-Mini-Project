"""Tests for the shared preprocessing helpers.

HOW TO USE
    From the repository root:  python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import (  # noqa: E402
    clean_text,
    duplicate_clusters,
    find_near_duplicate_pairs,
    parse_minutes,
    parse_servings,
    prepare_dataset,
    prepare_recipes,
    split_pipe,
)


class TestParsers(unittest.TestCase):
    def test_parse_minutes(self):
        cases = {
            "15 mins": 15,
            "15 minutes": 15,
            "1 hour 30 minutes": 90,
            "1 1/2 hours": 90,
            "2 days": 2880,
            "garbage": None,
            None: None,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_minutes(text), expected)

    def test_parse_servings(self):
        self.assertEqual(parse_servings("4 to 6"), (4, 6))
        self.assertEqual(parse_servings("serves 4"), (4, 4))
        self.assertEqual(parse_servings(""), (None, None))

    def test_clean_text_and_split_pipe(self):
        self.assertEqual(clean_text("  <b>Pork</b>   Adobo  "), "Pork Adobo")
        self.assertEqual(split_pipe("a | b|  |c"), ["a", "b", "c"])


def raw_df():
    return pd.DataFrame([
        {
            "recipe_name": "Pork Adobo Recipe",
            "prep_time": "10 minutes",
            "cook_time": "1 hour",
            "total_time": "1 hour 10 minutes",
            "servings": "4 to 6",
            "ingredients": "pork | vinegar | soy sauce",
            "ingredient_names": "pork | vinegar | soy sauce",
            "instructions": "brown pork | add vinegar and simmer",
        },
        {
            "recipe_name": "Pork Adobo",
            "prep_time": "10 mins",
            "cook_time": "1 hour",
            "total_time": "1 hour 10 mins",
            "servings": "4",
            "ingredients": "pork | vinegar | soy sauce",
            "ingredient_names": "pork | vinegar | soy sauce",
            "instructions": "brown pork | add vinegar and simmer | serve hot",
        },
    ])


class TestPrepareRecipes(unittest.TestCase):
    def test_columns_and_values(self):
        clean, report = prepare_recipes(raw_df())
        self.assertEqual(report["dropped_empty"], 0)
        self.assertEqual(list(clean["recipe_name"]), ["Pork Adobo", "Pork Adobo"])
        self.assertEqual(clean.loc[0, "total_time_min"], 70)
        self.assertEqual(clean.loc[1, "total_time_min"], 70)
        self.assertEqual(clean.loc[0, "servings_min"], 4)
        self.assertEqual(clean.loc[0, "servings_max"], 6)
        self.assertEqual(clean.loc[0, "ingredient_lines"], ["pork", "vinegar", "soy sauce"])

    def test_near_identical_rows_are_detected_and_merged(self):
        clean, _ = prepare_recipes(raw_df())
        pairs = find_near_duplicate_pairs(clean)
        self.assertEqual(len(pairs), 1)
        clusters, dropped, review = duplicate_clusters(pairs, clean)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(dropped, [0])  # shorter instruction row removed
        self.assertTrue(review.empty)

    def test_prepare_dataset_reports_and_renumbers(self):
        clean, report = prepare_dataset(raw_df())
        self.assertEqual(len(clean), 1)
        self.assertEqual(report["rows_removed"], 1)
        self.assertEqual(report["rows_final"], 1)
        self.assertEqual(list(clean["recipe_id"]), [0])


if __name__ == "__main__":
    unittest.main()
