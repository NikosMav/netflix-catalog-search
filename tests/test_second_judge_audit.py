import json

import pytest

from scripts.score_second_judge import RESULTS, audit_report


def test_independent_grades_rescore_the_same_rankings_and_reproduce_the_report():
    audit = audit_report()
    stored = json.loads((RESULTS / "second_judge_audit.json").read_text(encoding="utf-8"))
    original = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))["test"]
    assert audit["disagreements"] == stored["disagreements"]
    matrix = audit["matrix_original_rows_second_columns"]
    assert sum(map(sum, matrix)) == 303
    assert sum(matrix[i][i] for i in range(4)) + len(audit["disagreements"]) == 303
    assert audit["by_split"]["test"]["n"] == 187
    assert audit["by_split"]["dev"]["n"] == 116
    for row in original["systems"]:
        assert audit["test_original"]["systems"][row["method"]]["ndcg@10"] == row["ndcg@10"]
    for branch in ("test_original", "test_sensitivity"):
        assert stored[branch]["systems"] == audit[branch]["systems"]
        for current, saved in zip(audit[branch]["comparisons"], stored[branch]["comparisons"], strict=True):
            assert current["n"] == 30
            assert current["method"] == saved["method"]
            assert current["baseline"] == saved["baseline"]
            for metric in ("mean_difference", "ci_low", "ci_high"):
                assert current[metric] == pytest.approx(saved[metric], abs=1e-10)
