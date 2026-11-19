import json

from scripts import eval_v2_spotcheck as spotcheck


def test_draw_reproduces_every_frozen_blind_card():
    judgments = json.loads(spotcheck.JUDGMENTS.read_text(encoding="utf-8"))
    sample = json.loads(spotcheck.SAMPLE_PATH.read_text(encoding="utf-8"))
    cards = spotcheck.draw_sample(judgments, *spotcheck._config_numbers())
    assert cards == sample["pairs"]
    assert len({card["sample_id"] for card in cards}) == 303
    assert all(not {"grade", "relevant", "rationale", "rank", "score", "method"} & card.keys()
               for card in cards)
