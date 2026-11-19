import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from retrieval.review import read_grades, save_grade
from scripts import eval_v2_agreement as agreement
from scripts import eval_v2_spotcheck as spotcheck


def test_draw_reproduces_every_frozen_blind_card():
    judgments = json.loads(spotcheck.JUDGMENTS.read_text(encoding="utf-8"))
    sample = json.loads(spotcheck.SAMPLE_PATH.read_text(encoding="utf-8"))
    cards = spotcheck.draw_sample(judgments, *spotcheck._config_numbers())
    assert cards == sample["pairs"]
    assert len({card["sample_id"] for card in cards}) == 303
    assert all(not {"grade", "relevant", "rationale", "rank", "score", "method"} & card.keys()
               for card in cards)


def test_concurrent_saves_and_revisions_preserve_other_grades(tmp_path):
    path = tmp_path / "human.json"
    keys = {f"test:query:{i}" for i in range(12)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda key: save_grade(path, keys, key, 2), keys))
    save_grade(path, keys, "test:query:0", 0)
    assert read_grades(path, keys) == {k: 0 if k.endswith(":0") else 2 for k in keys}


@pytest.mark.parametrize("rows", [
    [{"sample_id": "unknown", "grade": 0}],
    [{"sample_id": "a", "grade": 1}] * 2,
    [{"sample_id": "a", "grade": 1.5}],
])
def test_invalid_progress_is_preserved_for_recovery(tmp_path, rows):
    path = tmp_path / "human.json"
    original = json.dumps({"judge": "human", "labels": rows})
    path.write_text(original)
    with pytest.raises(ValueError):
        save_grade(path, {"a", "b"}, "b", 3)
    assert path.read_text() == original


def test_agreement_waits_for_last_card_then_lists_disagreements(tmp_path, monkeypatch):
    cards = json.loads(spotcheck.SAMPLE_PATH.read_text(encoding="utf-8"))["pairs"][:2]
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps({"pairs": cards}))
    human, report = tmp_path / "human.json", tmp_path / "agreement.json"
    monkeypatch.setattr(agreement, "SAMPLE_PATH", sample)
    monkeypatch.setattr(agreement, "HUMAN_PATH", human)
    monkeypatch.setattr(agreement, "OUT_PATH", report)
    model = agreement.model_grade_map(json.loads(agreement.JUDGMENTS.read_text(encoding="utf-8")))
    keys = {c["sample_id"] for c in cards}
    for i, card in enumerate(cards):
        key = card["sample_id"]
        save_grade(human, keys, key, (model[key] + 1) % 4)
        agreement.main()
        payload = json.loads(report.read_text())
        assert payload["n"] == i + 1
        if i == 0:
            assert payload == {"status": "spot-check in progress", "n": 1, "n_sample": 2}
    assert payload["status"] == "scored"
    assert {d["sample_id"] for d in payload["disagreements"]} == keys
    assert payload["raw_agreement"] == 0
