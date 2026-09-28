"""Docs tables are copied from the committed JSON."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sync():
    path = ROOT / "scripts" / "sync_metrics_docs.py"
    spec = importlib.util.spec_from_file_location("sync_metrics_docs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_test_table_copies_json_intervals():
    sync = _sync()
    payload = json.loads((ROOT / "results" / "eval_v2" / "metrics.json").read_text(encoding="utf-8"))
    table = sync.test_table(payload)
    methods = [line.split("|")[1].strip() for line in table.splitlines()[2:]]
    assert methods == [row["method"] for row in payload["test"]["systems"]]
    assert len(methods) == len(payload["reported_systems"])
    hybrid = next(row for row in payload["test"]["systems"] if row["method"] == "hybrid+rerank")
    comp = next(row for row in payload["test"]["comparisons_vs_bm25"] if row["method"] == "hybrid+rerank")
    assert sync.num(hybrid["ndcg@10"]) in table
    assert f"[{sync.num(comp['ci_low'])}, {sync.num(comp['ci_high'])}]" in table
    assert sync.num(comp["sign_p_value"]) in table
    assert "baseline" in table


def test_notes_say_spotcheck_pending_until_scored():
    sync = _sync()
    payload = json.loads((ROOT / "results" / "eval_v2" / "metrics.json").read_text(encoding="utf-8"))
    sample = json.loads((ROOT / "data" / "eval_v2" / "spotcheck_sample.json").read_text(encoding="utf-8"))
    agreement = json.loads((ROOT / "results" / "eval_v2" / "spotcheck_agreement.json").read_text(encoding="utf-8"))
    notes = sync.test_notes(payload, sample, agreement)
    assert payload["judge"] in notes
    assert "spot-check pending" in notes
    assert sync.num(sample["n_sample"]) in notes
    scored = {
        "status": "scored",
        "n": 2,
        "n_agree": 2,
        "agreement": 1.0,
        "kappa": 1.0,
    }
    scored_notes = sync.test_notes(payload, sample, scored)
    assert "Cohen's kappa 1.0" in scored_notes


def test_label_overrides_are_the_only_disclosed_edits_and_match_judgments():
    sync = _sync()
    overrides = json.loads((ROOT / "data" / "eval_v2" / "label_overrides.json").read_text(encoding="utf-8"))
    judgments = json.loads((ROOT / "data" / "eval_v2" / "judgments.json").read_text(encoding="utf-8"))
    by_query = {item["query_id"]: item for item in judgments["queries"]}
    block = sync.overrides_block(overrides)
    assert "There are no other overrides." in block
    assert sync.num(len(overrides["overrides"])) in block
    seen = set()
    for row in overrides["overrides"]:
        key = (row["query_id"], row["show_id"])
        assert key not in seen
        seen.add(key)
        match = next(j for j in by_query[row["query_id"]]["judgments"] if j["show_id"] == row["show_id"])
        assert int(match["relevant"]) == row["final_label"]
        assert row["raw_label"] != row["final_label"]
        assert row["reason"] in block
        assert row["show_id"] in block


def test_v1_legacy_table_comes_from_legacy_json():
    sync = _sync()
    payload = json.loads((ROOT / "results" / "eval_metrics.json").read_text(encoding="utf-8"))
    table = sync.v1_table(payload)
    assert "ndcg@5" in table
    assert sync.num(payload["n_queries"]) in table
    assert sync.num(payload["metrics"][0]["ndcg@10"]) in table
