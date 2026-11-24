"""Tests for the maintained alias map and compound-aware normalization.

HOW TO USE
    From the repository root:  python -m unittest discover -s tests -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alias_map import alias_map, parents_map, write_alias_map  # noqa: E402
from ingredient_utils import (  # noqa: E402
    build_lookup,
    expand_canonicals,
    match_canonicals,
    normalize_text,
)


class TestAliasMap(unittest.TestCase):
    def setUp(self):
        self.aliases = alias_map()
        self.parents = parents_map()
        self.lookup, self.collisions = build_lookup(self.aliases)

    def test_no_collisions_or_whitespace_keys(self):
        self.assertEqual(self.collisions, [])
        for canon, terms in self.aliases.items():
            self.assertEqual(canon, canon.strip().lower())
            self.assertTrue(canon)
            for term in terms:
                self.assertEqual(term, term.strip().lower())

    def test_parent_targets_are_known_canonicals(self):
        for child, parent in self.parents.items():
            with self.subTest(child=child):
                self.assertIn(child, self.aliases)
                self.assertIn(parent, self.aliases)

    def test_compound_ingredients_stay_compound(self):
        cases = {
            "leftover pork adobo": "leftover_pork_adobo",
            "adobo flakes": "leftover_pork_adobo",
            "shredded adobo flakes from leftover adobo": "leftover_pork_adobo",
            "pork belly": "pork_belly",
            "liempo": "pork_belly",
            "pork hock": "pork_hock",
            "ground pork": "ground_pork",
            "giniling": "ground_pork",
            "chicken thighs": "chicken_thigh",
            "Maggi Magic Sarap": "maggi magic sarap",
            "Magic Sarap": "maggi magic sarap",
            "pork cube bouillon": "pork bouillon",
            "basil leaves": "basil",
            "ground cumin": "cumin",
            "breadcrumbs": "bread crumbs",
            "patola": "sponge gourd",
            "sweet corn": "corn",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                canon, left = normalize_text(text, self.lookup)
                self.assertEqual(canon, [expected])
                self.assertEqual(left, [])

    def test_generic_cuts_are_not_lost_inside_a_compound(self):
        canon, left = match_canonicals("adobo flakes from leftover adobo", self.lookup)
        self.assertEqual(canon, ["leftover_pork_adobo"])
        self.assertNotIn("from", left)

    def test_parents_expand_generic_pantry_matches(self):
        self.assertEqual(
            expand_canonicals(["pork_belly", "ground_pork"], self.parents),
            ["pork_belly", "pork", "ground_pork"],
        )

    def test_write_alias_map_produces_loadable_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_alias_map(Path(tmp) / "alias_map.json")
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["pork_belly"], sorted(set(self.aliases["pork_belly"])))
            self.assertEqual(data, {k: sorted(set(v)) for k, v in sorted(self.aliases.items())})


class TestLookupRobustness(unittest.TestCase):
    def test_collision_keeps_first_definition(self):
        lookup, collisions = build_lookup({"first": ["shared"], "second": ["shared"]})
        self.assertEqual(lookup["shared"], "first")
        self.assertEqual(len(collisions), 1)


if __name__ == "__main__":
    unittest.main()
