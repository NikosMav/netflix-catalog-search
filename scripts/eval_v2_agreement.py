#!/usr/bin/env python3
"""Raw agreement and Cohen's kappa for the eval v2 spot-check.

Prints 'spot-check pending' until data/eval_v2/spotcheck_human.json exists
and contains a label. Does not invent a kappa.
"""

from __future__ import annotations

import json
from pathlib import Path

from retrieval.significance import cohens_kappa

ROOT = Path(__file__).resolve().parents[1]
JUDGMENTS = ROOT / "data" / "eval_v2" / "judgments.json"
HUMAN_PATH = ROOT / "data" / "eval_v2" / "spotcheck_human.json"
OUT_PATH = ROOT / "results" / "eval_v2" / "spotcheck_agreement.json"


def model_label_map(judgments: dict) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for item in judgments["queries"]:
        for judgment in item["judgments"]:
            key = f"{item['split']}:{item['query_id']}:{judgment['show_id']}"
            mapping[key] = 1 if judgment["relevant"] else 0
    return mapping


def agreement_from_labels(model: dict[str, int], human_labels: list[dict]) -> dict:
    model_bits = []
    human_bits = []
    missing = []
    for row in human_labels:
        key = row["sample_id"]
        if key not in model:
            missing.append(key)
            continue
        model_bits.append(model[key])
        human_bits.append(1 if row["relevant"] else 0)
    if missing:
        raise SystemExit(f"human labels not in the model judgments: {missing}")
    stats = cohens_kappa(model_bits, human_bits)
    return {
        "status": "scored",
        "n": stats["n"],
        "n_agree": stats["n_agree"],
        "agreement": stats["agreement"],
        "kappa": stats["kappa"],
    }


def main() -> None:
    if not HUMAN_PATH.exists():
        payload = {"status": "spot-check pending"}
        print("spot-check pending")
    else:
        human = json.loads(HUMAN_PATH.read_text(encoding="utf-8"))
        labels = human.get("labels") or []
        if not labels:
            payload = {"status": "spot-check pending"}
            print("spot-check pending")
        else:
            judgments = json.loads(JUDGMENTS.read_text(encoding="utf-8"))
            payload = agreement_from_labels(model_label_map(judgments), labels)
            print(
                f"n={payload['n']} agreement={payload['agreement']} kappa={payload['kappa']}"
            )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
