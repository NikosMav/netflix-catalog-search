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
from retrieval.metrics import aggregate_mean, mrr, ndcg_at_k, recall_at_k
from retrieval.significance import paired_bootstrap_ci, two_sided_sign_test

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "eval_v2.yaml"
V1_LABELS = ROOT / "data" / "labeled_queries.json"
JUDGMENTS = ROOT / "data" / "eval_v2" / "judgments.json"
POOL_STATS = ROOT / "data" / "eval_v2" / "pool_stats.json"
OUT_PATH = ROOT / "results" / "eval_v2" / "metrics.json"

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


def _query_scores(retrieved: list[str], relevant: set[str]) -> dict[str, float]:
    return {
        "recall@5": recall_at_k(retrieved, relevant, 5),
        "recall@10": recall_at_k(retrieved, relevant, 10),
        "ndcg@10": ndcg_at_k(retrieved, relevant, 10),
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
        n_judged = 0
        n_scored = 0
        for item in queries:
            relevant = {j["show_id"] for j in item["judgments"] if j["relevant"]}
            n_judged += len(item["judgments"])
            n_relevant_pairs += len(relevant)
            if not relevant:
                for name in names:
                    empty = {metric: None for metric in METRICS}
                    per_query_out[name].append({"query_id": item["query_id"], **empty})
                    vectors[name].append(float("nan"))
                continue
            n_scored += 1
            for name in names:
                hits = by_name[name].retriever.query(item["query"], top_k=10)
                retrieved = hits["show_id"].astype(str).tolist()
                scores = _query_scores(retrieved, relevant)
                per_query_out[name].append({"query_id": item["query_id"], **_round_scores(scores)})
                vectors[name].append(scores["ndcg@10"])
        for name in names:
            finite = [v for v in vectors[name] if v == v]
            # Means use the same finite-query rule as retrieval.metrics.aggregate_mean.
            per_metric = []
            for metric in METRICS:
                vals = []
                for row in per_query_out[name]:
                    value = row[metric]
                    if value is None:
                        continue
                    vals.append(value)
                per_metric.append({metric: vals})
            summary = {}
            for metric in METRICS:
                vals = [row[metric] for row in per_query_out[name] if row[metric] is not None]
                mean = aggregate_mean(vals)
                summary[metric] = None if mean != mean else round(float(mean), 4)
            system_rows.append({"method": name, "n_scored": len(finite), **summary})
        label_counts[split] = {
            "n_queries": len(queries),
            "n_queries_with_relevant": n_scored,
            "n_judged_pairs": n_judged,
            "n_relevant_pairs": n_relevant_pairs,
        }
        splits_out[split] = {"systems": system_rows, "per_query": per_query_out}
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
        v2_rel = {j["show_id"] for j in item["judgments"] if j["relevant"]}
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

    payload = {
        "protocol": "eval_v2",
        "headline_metric": "ndcg@10",
        "baseline": BASELINE,
        "judge": judgments.get("judge"),
        "metrics": list(METRICS),
        "reported_systems": list(REPORTED),
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
