"""merge_metrics_preserving_existing keeps prior committed numbers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from retrieval.evaluate import merge_metrics_preserving_existing


def test_merge_preserves_old_rows(tmp_path: Path):
    old = {
        "n_queries": 28,
        "metrics": [
            {
                "method": "bm25",
                "recall@5": 0.5248,
                "recall@10": 0.5645,
                "ndcg@5": 0.5167,
                "ndcg@10": 0.5253,
                "mrr": 0.5637,
            },
            {
                "method": "dense(title+desc)",
                "recall@5": 0.5849,
                "recall@10": 0.6209,
                "ndcg@5": 0.571,
                "ndcg@10": 0.5656,
                "mrr": 0.6304,
            },
        ],
    }
    path = tmp_path / "eval_metrics.json"
    path.write_text(json.dumps(old), encoding="utf-8")
    new = pd.DataFrame(
        [
            {
                "method": "bm25(desc+plot)",
                "n_queries": 28,
                "recall@5": 0.5,
                "recall@10": 0.55,
                "ndcg@5": 0.5,
                "ndcg@10": 0.52,
                "mrr": 0.56,
            }
        ]
    )
    merged = merge_metrics_preserving_existing(path, new)
    by = {r["method"]: r for _, r in merged.iterrows()}
    assert by["bm25"]["recall@5"] == 0.5248
    assert by["dense(title+desc)"]["mrr"] == 0.6304
    assert by["bm25(desc+plot)"]["recall@5"] == 0.5
    assert set(merged["n_queries"].unique()) == {28}
