#!/usr/bin/env python3
"""Reproduce the frozen stratified sample from its original binary strata."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "eval_v2.yaml"
JUDGMENTS = ROOT / "data" / "eval_v2" / "judgments.json"
BLINDED = ROOT / "data" / "eval_v2" / "blinded"
SAMPLE_PATH = ROOT / "data" / "eval_v2" / "spotcheck_sample.json"
STRATA = ROOT / "data" / "eval_v2" / "spotcheck_strata.json"


def _config_numbers() -> tuple[float, int, int]:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    fraction = re.search(r"^\s*fraction:\s*([0-9.]+)\s*$", text, flags=re.MULTILINE)
    minimum = re.search(
        r"^\s*minimum_per_present_label:\s*(\d+)\s*$", text, flags=re.MULTILINE
    )
    # spotcheck seed is the seed line after `spotcheck:`.
    block = re.search(r"spotcheck:\n(.*)", text, flags=re.DOTALL)
    if not fraction or not minimum or not block:
        raise SystemExit("spotcheck settings missing from the protocol")
    seed = re.search(r"^\s*seed:\s*(\d+)\s*$", block.group(1), flags=re.MULTILINE)
    if not seed:
        raise SystemExit("spotcheck seed missing")
    return float(fraction.group(1)), int(minimum.group(1)), int(seed.group(1))


def sample_id(split: str, query_id: str, show_id: str) -> str:
    return f"{split}:{query_id}:{show_id}"


def draw_sample(judgments: dict, fraction: float, minimum: int, seed: int) -> list[dict]:
    """Stratified draw. Label 0, then label 1, so the seed is stable.

    Within a label, k = round(fraction * n), raised to `minimum` when the
    label is present, and never larger than n. The selected cards are then
    shuffled with the same generator so the form is not blocked by class.
    """
    groups: dict[int, list[dict]] = {0: [], 1: []}
    positive = set(json.loads(STRATA.read_text(encoding="utf-8"))["positive_sample_ids"])
    for item in judgments["queries"]:
        blinded_path = BLINDED / f"{item['split']}__{item['query_id']}.json"
        blinded = json.loads(blinded_path.read_text(encoding="utf-8"))
        cards = {c["show_id"]: c for c in blinded["candidates"]}
        for judgment in item["judgments"]:
            card = cards[judgment["show_id"]]
            key = sample_id(item["split"], item["query_id"], judgment["show_id"])
            groups[int(key in positive)].append(
                {
                    "sample_id": key,
                    "split": item["split"],
                    "query_id": item["query_id"],
                    "query": item["query"],
                    "show_id": judgment["show_id"],
                    "title": card["title"],
                    "type": card["type"],
                    "release_year": card["release_year"],
                    "listed_in": card["listed_in"],
                    "cast": card["cast"],
                    "director": card["director"],
                    "description": card["description"],
                }
            )
    rng = np.random.default_rng(seed)
    chosen: list[dict] = []
    for label in (0, 1):
        group = groups[label]
        if not group:
            continue
        k = int(round(fraction * len(group)))
        if k < minimum:
            k = minimum
        k = min(k, len(group))
        picks = rng.choice(len(group), size=k, replace=False)
        chosen.extend(group[int(i)] for i in picks)
    order = rng.permutation(len(chosen))
    return [chosen[int(i)] for i in order]


def write_sample() -> Path:
    fraction, minimum, seed = _config_numbers()
    judgments = json.loads(JUDGMENTS.read_text(encoding="utf-8"))
    pairs = draw_sample(judgments, fraction, minimum, seed)
    payload = {
        "fraction": fraction,
        "minimum_per_present_label": minimum,
        "seed": seed,
        "n_sample": len(pairs),
        "n_judged_pairs": sum(len(item["judgments"]) for item in judgments["queries"]),
        "pairs": pairs,
    }
    if SAMPLE_PATH.exists():
        if json.loads(SAMPLE_PATH.read_text(encoding="utf-8")) != payload:
            raise ValueError("Regenerated sample differs from the frozen review sample")
        return SAMPLE_PATH
    SAMPLE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return SAMPLE_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["draw"], nargs="?", default="draw")
    parser.parse_args()
    path = write_sample()
    print(f"Reproduced the frozen sample at {path}")
