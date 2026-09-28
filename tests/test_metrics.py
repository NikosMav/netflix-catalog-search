"""Unit tests for IR metrics (no model download required)."""

from __future__ import annotations

import math

from retrieval.metrics import aggregate_mean, grade_gain, mrr, ndcg_at_k, ndcg_at_k_graded, recall_at_k


def test_recall_at_k():
    rel = {"a", "b", "c"}
    assert recall_at_k(["a", "x", "b", "y"], rel, 3) == 2 / 3
    assert recall_at_k(["x", "y"], rel, 2) == 0.0


def test_mrr():
    rel = {"a", "b"}
    assert mrr(["x", "a"], rel) == 0.5
    assert mrr(["a"], rel) == 1.0
    assert mrr(["x", "y"], rel) == 0.0


def test_ndcg_binary():
    rel = {"a", "b", "c"}
    retrieved = ["a", "x", "b"]
    dcg = 1.0 / math.log2(2) + 0.0 + 1.0 / math.log2(4)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3) + 1.0 / math.log2(4)
    assert abs(ndcg_at_k(retrieved, rel, 3) - dcg / idcg) < 1e-9


def test_graded_ndcg_uses_exponential_gain_and_skips_all_zero():
    assert [grade_gain(g) for g in range(4)] == [0.0, 1.0, 3.0, 7.0]
    grades = {"a": 3, "b": 1}
    retrieved = ["b", "a"]
    # gains 1 then 7. Ideal is 7 then 1.
    dcg = 1.0 / math.log2(2) + 7.0 / math.log2(3)
    idcg = 7.0 / math.log2(2) + 1.0 / math.log2(3)
    assert abs(ndcg_at_k_graded(retrieved, grades, 10) - dcg / idcg) < 1e-9
    assert math.isnan(ndcg_at_k_graded(["a"], {"a": 0, "b": 0}, 10))
    # Grade 1 is not relevant for recall. Grade 2 is.
    assert math.isnan(recall_at_k(["a"], set(), 10))
    rel = {"b"}
    assert recall_at_k(["a", "b"], rel, 2) == 1.0


def test_judged_only_ndcg_drops_unlabeled_slots():
    grades = {"a": 2}
    # Unlabeled x would otherwise sit at rank 1. judged_only keeps a at rank 1.
    full = ndcg_at_k_graded(["x", "a"], grades, 10)
    judged = ndcg_at_k_graded(["x", "a"], grades, 10, judged_only=True)
    assert full < judged
    assert abs(judged - 1.0) < 1e-9


def test_aggregate_mean_skips_nan():
    assert aggregate_mean([1.0, float("nan"), 3.0]) == 2.0
