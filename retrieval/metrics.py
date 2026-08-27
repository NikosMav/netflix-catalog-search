"""Information-retrieval metrics for binary relevance judgments."""

from __future__ import annotations

import math
from typing import Iterable


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Recall@k = |top-k ∩ relevant| / |relevant|."""
    if not relevant_ids:
        return float("nan")
    top = retrieved_ids[:k]
    return len(set(top) & relevant_ids) / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank contribution for one query (0 if no hit)."""
    if not relevant_ids:
        return float("nan")
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """nDCG@k with binary relevance."""
    if not relevant_ids:
        return float("nan")
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        rel = 1.0 if doc_id in relevant_ids else 0.0
        dcg += rel / math.log2(i + 1)
    ideal_hits = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def aggregate_mean(values: Iterable[float]) -> float:
    vals = [v for v in values if v == v]  # drop NaN
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)
