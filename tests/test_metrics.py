"""Unit tests for IR metrics (no model download required)."""

from __future__ import annotations

import math

from retrieval.metrics import aggregate_mean, mrr, ndcg_at_k, recall_at_k


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


def test_aggregate_mean_skips_nan():
    assert aggregate_mean([1.0, float("nan"), 3.0]) == 2.0
