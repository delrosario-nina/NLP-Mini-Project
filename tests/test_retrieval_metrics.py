import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retrieval_metrics import ranking_metrics


class TestRankingMetrics(unittest.TestCase):
    def test_metrics_for_relevant_recipe_at_rank_two(self):
        metrics = ranking_metrics({"recipe-1"}, ["recipe-2", "recipe-1"], k=3)

        self.assertAlmostEqual(metrics["precision_at_k"], 1 / 3)
        self.assertEqual(metrics["recall_at_k"], 1)
        self.assertEqual(metrics["mrr_at_k"], 0.5)
        self.assertAlmostEqual(metrics["ndcg_at_k"], 1 / math.log2(3))

    def test_metrics_are_zero_when_no_relevant_recipe_is_retrieved(self):
        metrics = ranking_metrics({"recipe-1"}, ["recipe-2", "recipe-3"], k=2)

        self.assertEqual(metrics["precision_at_k"], 0)
        self.assertEqual(metrics["recall_at_k"], 0)
        self.assertEqual(metrics["mrr_at_k"], 0)
        self.assertEqual(metrics["ndcg_at_k"], 0)

    def test_invalid_cutoff_and_empty_relevance_are_rejected(self):
        with self.assertRaises(ValueError):
            ranking_metrics({"recipe-1"}, ["recipe-1"], k=0)
        with self.assertRaises(ValueError):
            ranking_metrics(set(), ["recipe-1"], k=5)


if __name__ == "__main__":
    unittest.main()
