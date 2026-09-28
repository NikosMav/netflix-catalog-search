#!/usr/bin/env python3
"""Draw a blinded spot-check sample and record human labels.

The sample file does not contain the model judgment. Human labels go to
data/eval_v2/spotcheck_human.json only when someone passes --relevant or
--not-relevant. This script never writes those labels by itself.

Agreement is a separate command and stays pending until the human file exists.
"""

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
HUMAN_PATH = ROOT / "data" / "eval_v2" / "spotcheck_human.json"


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
    for item in judgments["queries"]:
        blinded_path = BLINDED / f"{item['split']}__{item['query_id']}.json"
        blinded = json.loads(blinded_path.read_text(encoding="utf-8"))
        cards = {c["show_id"]: c for c in blinded["candidates"]}
        for judgment in item["judgments"]:
            card = cards[judgment["show_id"]]
            groups[1 if judgment["relevant"] else 0].append(
                {
                    "sample_id": sample_id(item["split"], item["query_id"], judgment["show_id"]),
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
        "status": "spot-check pending",
        "fraction": fraction,
        "minimum_per_present_label": minimum,
        "seed": seed,
        "n_sample": len(pairs),
        "n_judged_pairs": sum(len(item["judgments"]) for item in judgments["queries"]),
        "pairs": pairs,
    }
    for pair in pairs:
        if "relevant" in pair or "rationale" in pair:
            raise SystemExit("sample card would leak a model label")
    SAMPLE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return SAMPLE_PATH


def _load_human() -> dict:
    if not HUMAN_PATH.exists():
        return {"judge": "human", "labels": []}
    return json.loads(HUMAN_PATH.read_text(encoding="utf-8"))


def label_pair(sample_id_value: str, relevant: bool) -> None:
    sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    known = {pair["sample_id"] for pair in sample["pairs"]}
    if sample_id_value not in known:
        raise SystemExit(f"unknown sample_id {sample_id_value}")
    human = _load_human()
    labels = [row for row in human.get("labels", []) if row["sample_id"] != sample_id_value]
    labels.append({"sample_id": sample_id_value, "relevant": bool(relevant)})
    labels.sort(key=lambda row: row["sample_id"])
    HUMAN_PATH.write_text(
        json.dumps({"judge": "human", "labels": labels}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Blinded eval v2 spot-check for a human rater")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("draw", help="Write the seeded sample. Does not write human labels.")
    show = sub.add_parser("show", help="Print one unlabeled card")
    show.add_argument("--index", type=int, default=0)
    mark = sub.add_parser("label", help="Record one human label")
    mark.add_argument("sample_id")
    group = mark.add_mutually_exclusive_group(required=True)
    group.add_argument("--relevant", action="store_true")
    group.add_argument("--not-relevant", action="store_true")
    args = parser.parse_args()

    if args.command == "draw":
        path = write_sample()
        payload = json.loads(path.read_text(encoding="utf-8"))
        print(f"Wrote {path} ({payload['n_sample']} pairs). spot-check pending")
        return
    if args.command == "show":
        sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
        labeled = {row["sample_id"] for row in _load_human().get("labels", [])}
        pending = [pair for pair in sample["pairs"] if pair["sample_id"] not in labeled]
        if not pending:
            print("No unlabeled pairs left in the sample.")
            return
        card = pending[args.index]
        print(f"sample_id: {card['sample_id']}")
        print(f"query: {card['query']}")
        print(f"{card['title']} ({card['type']}, {card['release_year']})")
        print(card["listed_in"])
        if card["director"]:
            print(f"director: {card['director']}")
        if card["cast"]:
            print(f"cast: {card['cast']}")
        print(card["description"])
        print("spot-check pending")
        return
    label_pair(args.sample_id, relevant=args.relevant)
    print(f"Recorded {args.sample_id}")


if __name__ == "__main__":
    main()
