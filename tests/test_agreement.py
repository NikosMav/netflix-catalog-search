"""Graded spot-check agreement. No catalog and no model download."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _agreement():
    path = ROOT / "scripts" / "eval_v2_agreement.py"
    spec = importlib.util.spec_from_file_location("eval_v2_agreement", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_agreement_reports_weighted_and_binary_kappa_and_system_split():
    module = _agreement()
    model = {
        "test:q:s1": 3,
        "test:q:s2": 0,
        "test:q:s3": 2,
    }
    human = [
        {"sample_id": "test:q:s1", "grade": 3},
        {"sample_id": "test:q:s2", "grade": 1},
        {"sample_id": "test:q:s3", "grade": 0},
    ]
    top10 = {
        "systems": ["bm25", "hybrid+rerank"],
        "queries": [
            {
                "split": "test",
                "query_id": "q",
                "rankings": {
                    "bm25": ["s2", "s3"],
                    "hybrid+rerank": ["s1"],
                },
            }
        ],
    }
    payload = module.agreement_from_labels(model, human, top10)
    assert payload["status"] == "scored"
    assert payload["n"] == 3
    assert payload["quadratic_weighted_kappa"] is not None
    # s3 is grade 2 vs 0, so binarised labels disagree on that pair.
    assert payload["binary_kappa"] is not None
    by = {row["method"]: row for row in payload["by_system"]}
    assert by["bm25"]["n"] == 2
    assert by["hybrid+rerank"]["n"] == 1
    assert by["hybrid+rerank"]["raw_agreement"] == 1.0
    human_ndcg = {row["method"]: row for row in payload["human_ndcg@10"]}
    idcg = 7.0 / math.log2(2) + 1.0 / math.log2(3)
    hybrid = (7.0 / math.log2(2)) / idcg
    bm25 = (1.0 / math.log2(2)) / idcg
    assert abs(human_ndcg["hybrid+rerank"]["ndcg@10"] - hybrid) < 1e-9
    assert abs(human_ndcg["bm25"]["ndcg@10"] - bm25) < 1e-9


def test_spotcheck_page_does_not_read_model_grades():
    source = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    spot = source.split("def render_spotcheck")[1].split("def render_how")[0]
    assert "judgments.json" not in spot
    assert "Grade {grade}" in source or "Grade {grade}" in spot
    assert "--grade" not in spot
    assert "grade" in spot
