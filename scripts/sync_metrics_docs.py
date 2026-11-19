#!/usr/bin/env python3
"""Fill README.md and RETRIEVAL.md metric blocks from committed JSON.

Every evaluation number in those blocks is copied from:

- results/eval_v2/metrics.json
- results/eval_metrics.json (v1 legacy table only)
- data/eval_v2/spotcheck_sample.json
- results/eval_v2/spotcheck_agreement.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2_PATH = ROOT / "results" / "eval_v2" / "metrics.json"
V1_PATH = ROOT / "results" / "eval_metrics.json"
SAMPLE_PATH = ROOT / "data" / "eval_v2" / "spotcheck_sample.json"
AGREEMENT_PATH = ROOT / "results" / "eval_v2" / "spotcheck_agreement.json"
OVERRIDES_PATH = ROOT / "data" / "eval_v2" / "label_overrides.json"

METRIC_COLS = ["recall@5", "recall@10", "ndcg@10", "mrr"]


def num(value: object) -> str:
    """Render a JSON number without adding precision."""
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return json.dumps(value)
    return str(value)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _system_cells(row: dict) -> list[str]:
    return [num(row[col]) for col in METRIC_COLS]


def unscored_query_ids(split_payload: dict) -> list[str]:
    rows = split_payload["per_query"]["bm25"]
    return [row["query_id"] for row in rows if row["ndcg@10"] is None]


def recall_unscored_query_ids(split_payload: dict) -> list[str]:
    rows = split_payload["per_query"]["bm25"]
    return [row["query_id"] for row in rows if row["recall@10"] is None]


def _id_list(query_ids: list[str]) -> str:
    return ", ".join(f"`{qid}`" for qid in query_ids) if query_ids else "none"


def test_table(payload: dict) -> str:
    comparisons = {row["method"]: row for row in payload["test"]["comparisons_vs_bm25"]}
    headers = [
        "method",
        "n_scored",
        *METRIC_COLS,
        "nDCG@10 difference vs bm25",
        "95% CI",
        "sign-test p",
    ]
    rows: list[list[str]] = []
    for system in payload["test"]["systems"]:
        name = system["method"]
        cells = [name, num(system["n_scored"]), *_system_cells(system)]
        if name == payload["baseline"]:
            cells.extend(["baseline", "", ""])
        else:
            comp = comparisons[name]
            cells.append(num(comp["mean_difference"]))
            cells.append(f"[{num(comp['ci_low'])}, {num(comp['ci_high'])}]")
            cells.append(num(comp["sign_p_value"]))
        rows.append(cells)
    return _md_table(headers, rows)


def _ci_phrase(comp: dict) -> str:
    low = comp["ci_low"]
    high = comp["ci_high"]
    if low > 0:
        side = "entirely above 0"
    elif high < 0:
        side = "entirely below 0"
    else:
        side = "includes 0"
    return (
        f"`{comp['method']}` minus `{comp['baseline']}` mean {comp['metric']} "
        f"difference {num(comp['mean_difference'])} "
        f"({num(comp['ci_level'])} paired bootstrap interval "
        f"[{num(low)}, {num(high)}], {side}; "
        f"{num(comp['n_resamples'])} resamples, seed {num(comp['bootstrap_seed'])}). "
        f"Two-sided sign test p {num(comp['sign_p_value'])} "
        f"(+{num(comp['sign_n_positive'])} / −{num(comp['sign_n_negative'])} / "
        f"ties {num(comp['sign_n_ties'])}, n={num(comp['n'])})."
    )


def test_notes(payload: dict, sample: dict, agreement: dict) -> str:
    test = payload["test"]
    n_scored = test["systems"][0]["n_scored"]
    lines = [
        f"Judge: {payload['judge']}.",
        (
            f"Test split: {num(test['n_queries'])} queries, "
            f"{num(n_scored)} scored for nDCG. "
            f"Unscored for nDCG because the highest grade is 0: {_id_list(unscored_query_ids(test))}. "
            f"Unscored for recall and MRR because no grade is 2 or 3: "
            f"{_id_list(recall_unscored_query_ids(test))}."
        ),
        (
            "Each interval is the paired bootstrap of the per-query "
            f"{payload['headline_metric']} difference against `{payload['baseline']}`."
        ),
    ]
    lines.extend(_ci_phrase(comp) for comp in test["comparisons_vs_bm25"])
    lines.append(_spotcheck_sentence(sample, agreement))
    grades = _grade_sentence(payload)
    if grades:
        lines.append(grades)
    return "\n\n".join(lines)


def split_table(systems: list[dict]) -> str:
    headers = ["method", "n_scored", *METRIC_COLS]
    rows = [[row["method"], num(row["n_scored"]), *_system_cells(row)] for row in systems]
    return _md_table(headers, rows)


def dev_block(payload: dict) -> str:
    dev = payload["dev"]
    n_scored = dev["systems"][0]["n_scored"]
    intro = (
        f"Dev split: {num(dev['n_queries'])} queries, {num(n_scored)} scored for nDCG. "
        f"Unscored for nDCG because the highest grade is 0: {_id_list(unscored_query_ids(dev))}. "
        f"Unscored for recall and MRR because no grade is 2 or 3: "
        f"{_id_list(recall_unscored_query_ids(dev))}. "
        "No test-set interval is computed on this split."
    )
    return intro + "\n\n" + split_table(dev["systems"])


def ablation_block(payload: dict) -> str:
    return split_table(payload["dev"]["ablations"])


def _spotcheck_sentence(sample: dict, agreement: dict) -> str:
    status = agreement.get("status", sample.get("status", ""))
    base = (
        f"Spot-check sample: {num(sample['n_sample'])} pairs "
        f"from {num(sample['n_judged_pairs'])} judged pairs "
        f"(fraction {num(sample['fraction'])}, seed {num(sample['seed'])}). "
        f"Status: {status}."
    )
    if status == "scored":
        return (
            base
            + f" Raw agreement {num(agreement['raw_agreement'])}, "
            + f"quadratic-weighted kappa {num(agreement['quadratic_weighted_kappa'])}, "
            + f"binary kappa {num(agreement['binary_kappa'])}, "
            + f"n={num(agreement['n'])}."
        )
    return base


def stats_block(payload: dict, sample: dict, agreement: dict) -> str:
    labels = payload["labels"]
    pool = payload["pool"]
    missed = payload["v1_missed_on_dev"]["n_v2_relevant_titles_absent_from_v1"]
    lines = [
        (
            f"Pool depth {num(pool['pool_depth'])} across {num(pool['n_systems'])} systems "
            f"and {num(pool['n_queries'])} queries "
            f"({num(pool['n_candidate_slots'])} candidate slots; "
            f"{num(pool['candidates_min'])} to {num(pool['candidates_max'])} unique titles per query)."
        ),
        (
            f"Dev labels: {num(labels['dev']['n_relevant_pairs'])} pairs graded 2 or 3 "
            f"out of {num(labels['dev']['n_judged_pairs'])} judged pairs "
            f"({num(labels['dev']['n_queries_with_relevant'])} of "
            f"{num(labels['dev']['n_queries'])} queries have a grade of 2 or 3)."
        ),
        (
            f"Test labels: {num(labels['test']['n_relevant_pairs'])} pairs graded 2 or 3 "
            f"out of {num(labels['test']['n_judged_pairs'])} judged pairs "
            f"({num(labels['test']['n_queries_with_relevant'])} of "
            f"{num(labels['test']['n_queries'])} queries have a grade of 2 or 3)."
        ),
        (
            f"v2-relevant dev titles absent from that query's v1 relevant set: {num(missed)}."
        ),
        _spotcheck_sentence(sample, agreement),
    ]
    grades = _grade_sentence(payload)
    if grades:
        lines.append(grades)
    return "\n\n".join(lines)


def overrides_block(payload: dict) -> str:
    rows_in = payload["overrides"]
    intro = (
        f"{num(len(rows_in))} committed grades differ from the blinded re-judge. "
        "There are no other overrides."
    )
    headers = ["query_id", "show_id", "title", "raw grade", "final grade", "reason"]
    rows = [
        [
            row["query_id"],
            row["show_id"],
            row["title"],
            num(row["raw_grade"]),
            num(row["final_grade"]),
            row["reason"],
        ]
        for row in rows_in
    ]
    return intro + "\n\n" + _md_table(headers, rows)


def _grade_sentence(payload: dict) -> str:
    counts = payload.get("grade_counts")
    if not counts:
        return ""
    return (
        "Grade counts: "
        + ", ".join(f"{grade}={num(counts[str(grade)])}" for grade in range(4))
        + f". nDCG gain is {payload.get('ndcg_gain', '2^grade - 1')}. "
        + "Recall and MRR count a title when its grade is at least "
        + f"{num(payload.get('relevant_if_grade_at_least', 2))}."
    )


def v1_table(payload: dict) -> str:
    cols = ["method", "recall@5", "recall@10", "ndcg@5", "ndcg@10", "mrr"]
    rows: list[list[str]] = []
    for row in payload["metrics"]:
        cells = []
        for col in cols:
            value = row[col]
            cells.append(num(value) if isinstance(value, (int, float)) else str(value))
        rows.append(cells)
    intro = (
        f"v1 scored {num(payload['n_queries'])} queries. "
        "This table is the legacy file, not the v2 test result."
    )
    return intro + "\n\n" + _md_table(cols, rows)


def replace_block(text: str, begin: str, end: str, body: str) -> str:
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), flags=re.DOTALL)
    if not pattern.search(text):
        raise SystemExit(f"Markers not found: {begin}")
    return pattern.sub(f"{begin}\n{body}\n{end}", text, count=1)


def render(
    readme: str,
    retrieval: str,
    v2: dict,
    v1: dict,
    sample: dict,
    agreement: dict,
    overrides: dict,
) -> tuple[str, str]:
    readme = replace_block(
        readme,
        "<!-- METRICS_TABLE_BEGIN -->",
        "<!-- METRICS_TABLE_END -->",
        test_table(v2),
    )
    readme = replace_block(
        readme,
        "<!-- EVAL_V2_NOTES_BEGIN -->",
        "<!-- EVAL_V2_NOTES_END -->",
        test_notes(v2, sample, agreement),
    )
    retrieval = replace_block(
        retrieval,
        "<!-- EVAL_V2_DEV_BEGIN -->",
        "<!-- EVAL_V2_DEV_END -->",
        dev_block(v2),
    )
    retrieval = replace_block(
        retrieval,
        "<!-- EVAL_V2_ABLATIONS_BEGIN -->",
        "<!-- EVAL_V2_ABLATIONS_END -->",
        ablation_block(v2),
    )
    retrieval = replace_block(
        retrieval,
        "<!-- EVAL_V2_STATS_BEGIN -->",
        "<!-- EVAL_V2_STATS_END -->",
        stats_block(v2, sample, agreement),
    )
    retrieval = replace_block(
        retrieval,
        "<!-- V1_METRICS_TABLE_BEGIN -->",
        "<!-- V1_METRICS_TABLE_END -->",
        v1_table(v1),
    )
    retrieval = replace_block(
        retrieval,
        "<!-- EVAL_V2_OVERRIDES_BEGIN -->",
        "<!-- EVAL_V2_OVERRIDES_END -->",
        overrides_block(overrides),
    )
    return readme, retrieval


def main() -> None:
    v2 = json.loads(V2_PATH.read_text(encoding="utf-8"))
    v1 = json.loads(V1_PATH.read_text(encoding="utf-8"))
    sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    agreement = json.loads(AGREEMENT_PATH.read_text(encoding="utf-8"))
    overrides = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    readme_path = ROOT / "README.md"
    retrieval_path = ROOT / "RETRIEVAL.md"
    readme, retrieval = render(
        readme_path.read_text(encoding="utf-8"),
        retrieval_path.read_text(encoding="utf-8"),
        v2,
        v1,
        sample,
        agreement,
        overrides,
    )
    readme_path.write_text(readme, encoding="utf-8")
    retrieval_path.write_text(retrieval, encoding="utf-8")
    print(f"Updated README.md and RETRIEVAL.md from {V2_PATH.name} and {V1_PATH.name}")


if __name__ == "__main__":
    main()
