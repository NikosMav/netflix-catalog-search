"""Build a portable viewer of the frozen test rankings with the standard library."""

import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".cache" / "explorer"


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def explorer_data():
    metrics = load("results/eval_v2/metrics.json")["test"]
    audit = load("results/eval_v2/second_judge_audit.json")
    second = {r["sample_id"]: r for r in load("data/eval_v2/spotcheck_second_judge.json")["labels"]}
    judgments = {q["query_id"]: q for q in load("data/eval_v2/judgments.json")["queries"]
                 if q["split"] == "test"}
    scores = {method: {q["query_id"]: q["ndcg@10"] for q in rows}
              for method, rows in metrics["per_query"].items()}
    with (ROOT / "data/netflix_titles.csv").open(encoding="utf-8-sig", newline="") as f:
        catalog = {r["show_id"]: r for r in csv.DictReader(f)}
    queries = {}
    for record in load("results/eval_v2/top10.json")["queries"]:
        if record["split"] != "test":
            continue
        key = record["query_id"]
        judged = judgments[key]
        grades = {j["show_id"]: j for j in judged["judgments"]}
        cards = {}
        for ranking in record["rankings"].values():
            for sid in ranking:
                row = catalog[sid]
                cards[sid] = {field: row[field] for field in (
                    "title", "description", "type", "release_year", "listed_in", "cast", "director"
                )}
                cards[sid].update(grade=grades[sid]["grade"], rationale=grades[sid]["rationale"])
                reviewed = second.get(f"test:{key}:{sid}")
                if reviewed:
                    cards[sid]["second_review"] = {"grade": reviewed["grade"], "rationale": reviewed["rationale"]}
        queries[key] = {
            "query": judged["query"], "rankings": record["rankings"], "cards": cards,
            "scores": {method: values[key] for method, values in scores.items()},
        }
    return {"queries": queries, "systems": metrics["systems"],
            "comparisons": metrics["comparisons_vs_bm25"],
            "review": {k: audit["agreement"][k] for k in ("n", "raw_agreement", "binary_agreement")},
            "sensitivity": next(c for c in audit["test_sensitivity"]["comparisons"]
                                if c["method"] == "hybrid+rerank" and c["baseline"] == "bm25")}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for asset in ("index.html", "style.css", "explorer.js"):
        shutil.copyfile(ROOT / "explorer" / asset, OUTPUT / asset)
    data = json.dumps(explorer_data(), ensure_ascii=True, separators=(",", ":"))
    (OUTPUT / "results.js").write_text("const benchmark = " + data + ";\n", encoding="utf-8")
    print(OUTPUT / "index.html")


if __name__ == "__main__":
    main()
