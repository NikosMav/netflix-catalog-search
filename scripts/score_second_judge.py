"""Compare both judgments and rescore the frozen test rankings with audited grades."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from retrieval.significance import paired_bootstrap_ci, two_sided_sign_test
from retrieval.agreement import agreement_from_labels, model_grade_map
from scripts.run_eval_v2 import _mean_row, _query_scores

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "eval_v2"
RESULTS = ROOT / "results" / "eval_v2"
FUSION = "hybrid(bm25+dense,meta)"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare(scores, method, baseline):
    left, right = np.asarray(scores[method]), np.asarray(scores[baseline])
    keep = np.isfinite(left) & np.isfinite(right)
    left, right = left[keep], right[keep]
    return {"method": method, "baseline": baseline, **paired_bootstrap_ci(left, right),
            "sign_test": two_sided_sign_test(left - right)}


def rescore_test(judgments, rankings, overrides):
    queries = {q["query_id"]: q for q in judgments["queries"] if q["split"] == "test"}
    rows = defaultdict(list)
    per_query = []
    for q in rankings["queries"]:
        if q["split"] != "test":
            continue
        key = q["query_id"]
        grades = {j["show_id"]: overrides.get(f"test:{key}:{j['show_id']}", j["grade"])
                  for j in queries[key]["judgments"]}
        ndcg = {}
        for method, ranking in q["rankings"].items():
            metrics = _query_scores(ranking, grades)
            rows[method].append(metrics)
            ndcg[method] = metrics["ndcg@10"] if np.isfinite(metrics["ndcg@10"]) else None
        per_query.append({"query_id": key, "ndcg@10": ndcg})
    scores = {name: [r["ndcg@10"] for r in records] for name, records in rows.items()}
    comparisons = [compare(scores, name, "bm25") for name in rankings["systems"] if name != "bm25"]
    comparisons.append(compare(scores, "hybrid+rerank", FUSION))
    return {"systems": {name: _mean_row(records) for name, records in rows.items()},
            "comparisons": comparisons, "per_query": per_query}


def audit_report():
    sample = read(DATA / "spotcheck_sample.json")["pairs"]
    cards = {c["sample_id"]: c for c in sample}
    audit = read(DATA / "spotcheck_second_judge.json")
    labels = audit["labels"]
    grades = {row["sample_id"]: row["grade"] for row in labels}
    if len(labels) != len(sample) or set(grades) != set(cards):
        raise ValueError("Both reviewers must cover exactly the frozen sample, once each")
    if any(type(row["grade"]) is not int or row["grade"] not in range(4)
           or not row["rationale"].strip() for row in labels):
        raise ValueError("Every reviewed card needs an integer grade 0–3 and a rationale")
    judgments = read(DATA / "judgments.json")
    original = model_grade_map(judgments)
    rankings = read(RESULTS / "top10.json")
    agreement = agreement_from_labels(original, labels, rankings)
    matrix = [[0] * 4 for _ in range(4)]
    disagreements = []
    for r in labels:
        key = r["sample_id"]
        old, new = original[key], r["grade"]
        matrix[old][new] += 1
        if old != new:
            disagreements.append({"sample_id": key, "query": cards[key]["query"], "title": cards[key]["title"],
                                  "original_grade": old, "second_grade": new})
    by_split = {}
    for split in ("dev", "test"):
        subset = [r for r in labels if cards[r["sample_id"]]["split"] == split]
        by_split[split] = {k: v for k, v in agreement_from_labels(original, subset).items()
                          if k in ("n", "raw_agreement", "quadratic_weighted_kappa", "binary_agreement", "binary_kappa")}
    return {"judge": audit["judge"], "status": "second-judge audit complete",
            "compared_to": judgments["judge"], "n_pooled": len(original),
            "agreement": agreement, "by_split": by_split, "matrix_original_rows_second_columns": matrix,
            "disagreements": disagreements,
            "test_sensitivity": rescore_test(judgments, rankings, grades),
            "test_original": rescore_test(judgments, rankings, {})}


if __name__ == "__main__":
    path = RESULTS / "second_judge_audit.json"
    path.write_text(json.dumps(audit_report(), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
    print(path)
