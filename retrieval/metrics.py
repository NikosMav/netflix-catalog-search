"""Information-retrieval metrics.

``recall_at_k``, ``mrr``, and ``ndcg_at_k`` stay binary for the v1 evaluation.
``ndcg_at_k_graded`` is the eval v2 amendment: gain is ``2^grade - 1``.
"""

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


def grade_gain(grade: int) -> float:
    """Exponential gain ``2^grade - 1`` for grades 0, 1, 2, 3."""
    value = int(grade)
    if value < 0:
        raise ValueError("grade must be >= 0")
    return float((1 << value) - 1)


def ndcg_at_k_graded(
    retrieved_ids: list[str],
    grades: dict[str, int],
    k: int,
    *,
    judged_only: bool = False,
) -> float:
    """nDCG@k with gain ``2^grade - 1``.

    A missing id has gain 0. If ``judged_only`` is set, ids that are not in
    ``grades`` are dropped from the ranked list and do not take a discount
    slot. Queries whose highest grade is 0 return NaN.
    """
    if not grades or max(int(g) for g in grades.values()) <= 0:
        return float("nan")
    ranked = list(retrieved_ids)
    if judged_only:
        ranked = [doc_id for doc_id in ranked if doc_id in grades]
    dcg = 0.0
    for i, doc_id in enumerate(ranked[:k], start=1):
        dcg += grade_gain(int(grades.get(doc_id, 0))) / math.log2(i + 1)
    ideal = sorted(
        (grade_gain(int(g)) for g in grades.values() if int(g) > 0),
        reverse=True,
    )
    idcg = sum(gain / math.log2(i + 1) for i, gain in enumerate(ideal[:k], start=1))
    if idcg == 0.0:
        return float("nan")
    return dcg / idcg


def aggregate_mean(values: Iterable[float]) -> float:
    vals = [v for v in values if v == v]  # drop NaN
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)
