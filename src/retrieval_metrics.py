"""Ranking metrics for evaluating recipe retrieval results."""

import math
from typing import Iterable, Sequence


def ranking_metrics(
    relevant_ids: Iterable,
    ranked_ids: Sequence,
    k: int = 5,
) -> dict:
    """Calculate Precision@k, Recall@k, MRR@k, and nDCG@k."""
    if k < 1:
        raise ValueError("k must be at least 1")

    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("relevant_ids must contain at least one ID")

    top_ids = list(ranked_ids)[:k]
    relevant_ranks = [
        rank
        for rank, recipe_id in enumerate(top_ids, start=1)
        if recipe_id in relevant
    ]

    precision = len(relevant_ranks) / k
    recall = len(relevant_ranks) / len(relevant)
    reciprocal_rank = 1 / relevant_ranks[0] if relevant_ranks else 0.0

    dcg = sum(1 / math.log2(rank + 1) for rank in relevant_ranks)
    ideal_count = min(len(relevant), k)
    ideal_dcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )

    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "mrr_at_k": reciprocal_rank,
        "ndcg_at_k": dcg / ideal_dcg,
    }
