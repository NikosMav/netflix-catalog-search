#!/usr/bin/env python3
"""Score the pre-registered eval v2 systems and write results/eval_v2/metrics.json.

Reads judgments produced from the blinded pools. Does not edit v1 files.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from retrieval.catalog import load_catalog
from retrieval.evaluate import build_methods
from retrieval.metrics import aggregate_mean, mrr, ndcg_at_k_graded, recall_at_k
from retrieval.significance import paired_bootstrap_ci, two_sided_sign_test

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "eval_v2.yaml"
V1_LABELS = ROOT / "data" / "labeled_queries.json"
JUDGMENTS = ROOT / "data" / "eval_v2" / "judgments.json"
POOL_STATS = ROOT / "data" / "eval_v2" / "pool_stats.json"
OUT_PATH = ROOT / "results" / "eval_v2" / "metrics.json"
TOP10_PATH = ROOT / "results" / "eval_v2" / "top10.json"
RELEVANT_AT = 2

REPORTED = (
    "boolean",
    "tf-idf",
    "bm25",
    "dense(title+desc)",
    "hybrid(bm25+dense,meta)",
    "hybrid+rerank",
)
BASELINE = "bm25"
METRICS = ("recall@5", "recall@10", "ndcg@10", "mrr")


def _config_int(key: str) -> int:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    match = re.search(rf"^\s*{re.escape(key)}:\s*(\d+)\s*$", text, flags=re.MULTILINE)
    if not match:
        raise SystemExit(f"{key} missing from protocol")
    return int(match.group(1))


def _query_scores(retrieved: list[str], grades: dict[str, int]) -> dict[str, float]:
    relevant = {sid for sid, grade in grades.items() if grade >= RELEVANT_AT}
    return {
        "recall@5": recall_at_k(retrieved, relevant, 5),
        "recall@10": recall_at_k(retrieved, relevant, 10),
        "ndcg@10": ndcg_at_k_graded(retrieved, grades, 10),
        "mrr": mrr(retrieved, relevant),
    }


def _mean_row(per_query: list[dict[str, float]]) -> dict[str, float | None]:
    row: dict[str, float | None] = {}
    for metric in METRICS:
        vals = [item[metric] for item in per_query]
        mean = aggregate_mean(vals)
        row[metric] = None if mean != mean else round(float(mean), 4)
    return row


def _round_scores(scores: dict[str, float]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for metric in METRICS:
        value = scores[metric]
        out[metric] = None if value != value else round(float(value), 6)
    return out


def main() -> None:
    judgments = json.loads(JUDGMENTS.read_text(encoding="utf-8"))
    pool = json.loads(POOL_STATS.read_text(encoding="utf-8"))
    v1 = json.loads(V1_LABELS.read_text(encoding="utf-8"))
    v1_by_id = {item["id"]: set(item["relevant_show_ids"]) for item in v1["queries"]}

    by_split: dict[str, list[dict]] = {"dev": [], "test": []}
    for item in judgments["queries"]:
        by_split[item["split"]].append(item)

    catalog = load_catalog()
    methods = build_methods(catalog, show_progress=True)
    by_name = {method.name: method for method in methods}
    missing = [name for name in REPORTED if name not in by_name]
    if missing:
        raise SystemExit(f"reported systems missing from build_methods: {missing}")

    n_resamples = _config_int("resamples")
    bootstrap_seed = None
    # The protocol has more than one `seed` key. The bootstrap seed is the
    # integer on the `seed:` line directly under `bootstrap:`.
    config_text = CONFIG_PATH.read_text(encoding="utf-8")
    boot = re.search(r"bootstrap:\n(?:.*\n)*?\s*seed:\s*(\d+)", config_text)
    if not boot:
        raise SystemExit("bootstrap seed missing")
    bootstrap_seed = int(boot.group(1))

    splits_out: dict[str, dict] = {}
    label_counts: dict[str, dict] = {}
    per_query_vectors: dict[str, dict[str, list[float]]] = {}

    for split, queries in by_split.items():
        # Ablations are scored on dev only. The test table is the six reported systems.
        names = list(by_name) if split == "dev" else list(REPORTED)
        system_rows = []
        per_query_out: dict[str, list[dict]] = {name: [] for name in names}
        vectors: dict[str, list[float]] = {name: [] for name in names}
        n_relevant_pairs = 0
        n_positive_grades = 0
        n_judged = 0
        n_scored_ndcg = 0
        n_scored_recall = 0
        top10_rows = []
        for item in queries:
            grades = {j["show_id"]: int(j["grade"]) for j in item["judgments"]}
            relevant = {sid for sid, grade in grades.items() if grade >= RELEVANT_AT}
            n_judged += len(grades)
            n_relevant_pairs += len(relevant)
            n_positive_grades += sum(1 for grade in grades.values() if grade > 0)
            has_ndcg = any(grade > 0 for grade in grades.values())
            has_recall = bool(relevant)
            if has_ndcg:
                n_scored_ndcg += 1
            if has_recall:
                n_scored_recall += 1
            rankings: dict[str, list[str]] = {}
            for name in names:
                hits = by_name[name].retriever.query(item["query"], top_k=10)
                retrieved = hits["show_id"].astype(str).tolist()
                if name in REPORTED:
                    rankings[name] = retrieved
                if not has_ndcg and not has_recall:
                    empty = {metric: None for metric in METRICS}
                    per_query_out[name].append({"query_id": item["query_id"], **empty})
                    vectors[name].append(float("nan"))
                    continue
                scores = _query_scores(retrieved, grades)
                if not has_ndcg:
                    scores["ndcg@10"] = float("nan")
                if not has_recall:
                    scores["recall@5"] = float("nan")
                    scores["recall@10"] = float("nan")
                    scores["mrr"] = float("nan")
                per_query_out[name].append({"query_id": item["query_id"], **_round_scores(scores)})
                vectors[name].append(scores["ndcg@10"])
            if rankings:
                top10_rows.append(
                    {
                        "split": split,
                        "query_id": item["query_id"],
                        "rankings": rankings,
                    }
                )
        for name in names:
            finite = [v for v in vectors[name] if v == v]
            summary = {}
            for metric in METRICS:
                vals = [row[metric] for row in per_query_out[name] if row[metric] is not None]
                mean = aggregate_mean(vals)
                summary[metric] = None if mean != mean else round(float(mean), 4)
            recall_n = sum(
                1
                for row in per_query_out[name]
                if row["recall@10"] is not None
            )
            system_rows.append(
                {
                    "method": name,
                    "n_scored": len(finite),
                    "n_scored_recall": recall_n,
                    **summary,
                }
            )
        label_counts[split] = {
            "n_queries": len(queries),
            "n_queries_with_positive_grade": n_scored_ndcg,
            "n_queries_with_relevant": n_scored_recall,
            "n_judged_pairs": n_judged,
            "n_positive_grade_pairs": n_positive_grades,
            "n_relevant_pairs": n_relevant_pairs,
        }
        splits_out[split] = {
            "systems": system_rows,
            "per_query": per_query_out,
            "top10": top10_rows,
        }
        per_query_vectors[split] = vectors

    # Paired tests on TEST, reported systems against BM25. Uses unrounded
    # per-query nDCG stored above before the 6-decimal JSON rounding.
    # Recompute from the in-memory vectors, which skipped empty-relevant queries
    # as NaN. Drop those pairs from the comparison so both sides stay aligned.
    test_vectors = per_query_vectors["test"]
    base_raw = test_vectors[BASELINE]
    comparisons = []
    for name in REPORTED:
        if name == BASELINE:
            continue
        paired_system = []
        paired_base = []
        for sys_v, base_v in zip(test_vectors[name], base_raw):
            if sys_v == sys_v and base_v == base_v:
                paired_system.append(sys_v)
                paired_base.append(base_v)
        boot = paired_bootstrap_ci(
            paired_system,
            paired_base,
            n_resamples=n_resamples,
            seed=bootstrap_seed,
        )
        signs = two_sided_sign_test(
            [s - b for s, b in zip(paired_system, paired_base)]
        )
        comparisons.append(
            {
                "method": name,
                "baseline": BASELINE,
                "metric": "ndcg@10",
                "n": boot["n"],
                "mean_difference": round(float(boot["mean_difference"]), 4),
                "ci_low": round(float(boot["ci_low"]), 4),
                "ci_high": round(float(boot["ci_high"]), 4),
                "ci_level": boot["level"],
                "n_resamples": boot["n_resamples"],
                "bootstrap_seed": boot["seed"],
                "sign_n_positive": signs["n_positive"],
                "sign_n_negative": signs["n_negative"],
                "sign_n_ties": signs["n_ties"],
                "sign_p_value": signs["p_value"],
            }
        )

    missed = []
    for item in by_split["dev"]:
        v2_rel = {j["show_id"] for j in item["judgments"] if int(j["grade"]) >= RELEVANT_AT}
        v1_rel = v1_by_id[item["query_id"]]
        for sid in sorted(v2_rel - v1_rel):
            missed.append({"query_id": item["query_id"], "show_id": sid})

    dev_systems = splits_out["dev"]["systems"]
    reported_dev = [row for row in dev_systems if row["method"] in REPORTED]
    ablation_dev = [row for row in dev_systems if row["method"] not in REPORTED]
    # Keep build_methods order.
    order = {name: i for i, name in enumerate(by_name)}
    reported_dev.sort(key=lambda row: order[row["method"]])
    ablation_dev.sort(key=lambda row: order[row["method"]])
    test_systems = splits_out["test"]["systems"]
    test_systems.sort(key=lambda row: REPORTED.index(row["method"]))

    grade_counts = {str(grade): 0 for grade in range(4)}
    for item in judgments["queries"]:
        for judgment in item["judgments"]:
            grade_counts[str(int(judgment["grade"]))] += 1

    top10_payload = {
        "depth": 10,
        "systems": list(REPORTED),
        "queries": splits_out["dev"]["top10"] + splits_out["test"]["top10"],
    }
    TOP10_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOP10_PATH.write_text(json.dumps(top10_payload, indent=2) + "\n", encoding="utf-8")

    payload = {
        "protocol": "eval_v2",
        "amendment": "graded_labels",
        "headline_metric": "ndcg@10",
        "ndcg_gain": "2^grade - 1",
        "relevant_if_grade_at_least": RELEVANT_AT,
        "baseline": BASELINE,
        "judge": judgments.get("judge"),
        "metrics": list(METRICS),
        "reported_systems": list(REPORTED),
        "grade_counts": grade_counts,
        "pool": {
            "pool_depth": pool["pool_depth"],
            "n_systems": pool["n_systems"],
            "systems": pool["systems"],
            "n_queries": pool["n_queries"],
            "n_candidate_slots": pool["n_candidate_slots"],
            "candidates_min": pool["candidates_min"],
            "candidates_max": pool["candidates_max"],
        },
        "labels": label_counts,
        "v1_missed_on_dev": {
            "n_v2_relevant_titles_absent_from_v1": len(missed),
            "pairs": missed,
        },
        "test": {
            "n_queries": label_counts["test"]["n_queries"],
            "systems": test_systems,
            "comparisons_vs_bm25": comparisons,
            "per_query": splits_out["test"]["per_query"],
        },
        "dev": {
            "n_queries": label_counts["dev"]["n_queries"],
            "systems": reported_dev,
            "ablations": ablation_dev,
            "per_query": {
                name: splits_out["dev"]["per_query"][name]
                for name in list(REPORTED) + [row["method"] for row in ablation_dev]
            },
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
