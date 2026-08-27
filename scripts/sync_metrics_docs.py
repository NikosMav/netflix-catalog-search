#!/usr/bin/env python3
"""Sync README.md / RETRIEVAL.md metric tables from results/eval_metrics.json.

Never invent numbers — only copy what is already in the committed JSON.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "results" / "eval_metrics.json"
BEGIN = "<!-- METRICS_TABLE_BEGIN -->"
END = "<!-- METRICS_TABLE_END -->"


def table_from_json(payload: dict) -> str:
    cols = ["method", "recall@5", "recall@10", "ndcg@5", "ndcg@10", "mrr"]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in payload["metrics"]:
        cells = []
        for c in cols:
            val = row[c]
            if isinstance(val, float):
                cells.append(f"{val:.4f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def replace_table(path: Path, table: str) -> None:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        re.escape(BEGIN) + r".*?" + re.escape(END),
        flags=re.DOTALL,
    )
    replacement = f"{BEGIN}\n{table}\n{END}"
    if not pattern.search(text):
        raise SystemExit(f"Markers not found in {path}")
    path.write_text(pattern.sub(replacement, text), encoding="utf-8")


def main() -> None:
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    table = table_from_json(payload)
    for rel in ("README.md", "RETRIEVAL.md"):
        replace_table(ROOT / rel, table)
        print(f"Updated {rel} from {JSON_PATH.name} (n_queries={payload['n_queries']})")


if __name__ == "__main__":
    main()
