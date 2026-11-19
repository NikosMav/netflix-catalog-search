from collections import defaultdict

import numpy as np

from retrieval.metrics import ndcg_at_k_graded
from retrieval.significance import paired_bootstrap_ci, two_sided_sign_test
from scripts.build_explorer import explorer_data, load


def test_replay_reproduces_test_scores_and_paired_intervals():
    viewer = explorer_data()
    test = load("results/eval_v2/metrics.json")["test"]
    raw_scores = defaultdict(list)
    queries = [q for q in load("data/eval_v2/judgments.json")["queries"] if q["split"] == "test"]
    assert len(queries) == len(viewer["queries"]) == test["n_queries"]
    for item in queries:
        replay = viewer["queries"][item["query_id"]]
        grades = {j["show_id"]: j["grade"] for j in item["judgments"]}
        for method, ranking in replay["rankings"].items():
            score = ndcg_at_k_graded(ranking, grades, 10)
            assert round(score, 6) == replay["scores"][method]
            raw_scores[method].append(score)
    for row in viewer["systems"]:
        assert round(float(np.mean(raw_scores[row["method"]])), 4) == row["ndcg@10"]
    for row in viewer["comparisons"]:
        model, baseline = raw_scores[row["method"]], raw_scores["bm25"]
        effect = paired_bootstrap_ci(model, baseline, n_resamples=row["n_resamples"],
                                     seed=row["bootstrap_seed"])
        for field in ("mean_difference", "ci_low", "ci_high"):
            assert round(effect[field], 4) == row[field]
        signs = two_sided_sign_test(np.subtract(model, baseline))
        assert signs["p_value"] == row["sign_p_value"]
