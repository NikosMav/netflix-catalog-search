"""Agreement and condensed-ranking diagnostics for two sets of relevance grades."""

from __future__ import annotations

from retrieval.metrics import aggregate_mean, ndcg_at_k_graded
from retrieval.significance import cohens_kappa, quadratic_weighted_kappa

RELEVANT_AT = 2
REPORTED = (
    "boolean",
    "tf-idf",
    "bm25",
    "dense(title+desc)",
    "hybrid(bm25+dense,meta)",
    "hybrid+rerank",
)


def model_grade_map(judgments: dict) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for item in judgments["queries"]:
        for judgment in item["judgments"]:
            key = f"{item['split']}:{item['query_id']}:{judgment['show_id']}"
            mapping[key] = int(judgment["grade"])
    return mapping


def _pair_grades(model: dict[str, int], second_labels: list[dict]) -> tuple[list[int], list[int], list[str]]:
    model_grades = []
    second_grades = []
    keys = []
    missing = []
    for row in second_labels:
        key = row["sample_id"]
        if key not in model:
            missing.append(key)
            continue
        if "grade" not in row:
            raise SystemExit(f"{key} has no grade")
        grade = int(row["grade"])
        if grade not in (0, 1, 2, 3):
            raise SystemExit(f"{key} grade {grade} is outside 0-3")
        model_grades.append(model[key])
        second_grades.append(grade)
        keys.append(key)
    if missing:
        raise SystemExit(f"second-judge labels not in the model judgments: {missing}")
    return model_grades, second_grades, keys


def _stats(model_grades: list[int], second_grades: list[int]) -> dict:
    weighted = quadratic_weighted_kappa(model_grades, second_grades, n_classes=4)
    binary = cohens_kappa(
        [1 if grade >= RELEVANT_AT else 0 for grade in model_grades],
        [1 if grade >= RELEVANT_AT else 0 for grade in second_grades],
    )
    return {
        "n": weighted["n"],
        "raw_agreement": weighted["agreement"],
        "quadratic_weighted_kappa": weighted["kappa"],
        "binary_agreement": binary["agreement"],
        "binary_kappa": binary["kappa"],
    }


def sample_ndcg_by_system(
    second_labels: list[dict],
    top10: dict,
) -> list[dict]:
    """nDCG@10 from second-judge grades only.

    For each query, documents without a second-judge grade are dropped from the
    system's top 10. Ideal gain uses every second-judge grade on that query.
    Queries whose second-judge grades are all 0 are omitted.
    """
    by_query: dict[tuple[str, str], dict[str, int]] = {}
    for row in second_labels:
        split, query_id, show_id = row["sample_id"].split(":", 2)
        by_query.setdefault((split, query_id), {})[show_id] = int(row["grade"])
    rankings = {
        (item["split"], item["query_id"]): item["rankings"]
        for item in top10["queries"]
    }
    rows = []
    for name in top10.get("systems", REPORTED):
        scores = []
        for key, grades in by_query.items():
            ranking = rankings.get(key, {}).get(name)
            if ranking is None:
                continue
            value = ndcg_at_k_graded(ranking, grades, 10, judged_only=True)
            if value == value:
                scores.append(value)
        mean = aggregate_mean(scores)
        rows.append(
            {
                "method": name,
                "n_scored": len(scores),
                "ndcg@10": None if mean != mean else mean,
            }
        )
    return rows


def agreement_from_labels(
    model: dict[str, int],
    second_labels: list[dict],
    top10: dict | None = None,
) -> dict:
    model_grades, second_grades, keys = _pair_grades(model, second_labels)
    payload = {"status": "scored", **_stats(model_grades, second_grades)}
    if not top10:
        payload["by_system"] = []
        payload["sample_only_condensed_ndcg@10"] = []
        return payload
    retrieved: dict[str, set[str]] = {name: set() for name in top10.get("systems", REPORTED)}
    for item in top10["queries"]:
        prefix = f"{item['split']}:{item['query_id']}:"
        for name, ranking in item["rankings"].items():
            for show_id in ranking[:10]:
                retrieved.setdefault(name, set()).add(prefix + show_id)
    by_system = []
    key_to_pos = {key: i for i, key in enumerate(keys)}
    for name in top10.get("systems", REPORTED):
        idxs = [key_to_pos[key] for key in keys if key in retrieved.get(name, set())]
        if not idxs:
            by_system.append({"method": name, "n": 0})
            continue
        stats = _stats(
            [model_grades[i] for i in idxs],
            [second_grades[i] for i in idxs],
        )
        by_system.append({"method": name, **stats})
    payload["by_system"] = by_system
    payload["sample_only_condensed_ndcg@10"] = sample_ndcg_by_system(second_labels, top10)
    return payload
