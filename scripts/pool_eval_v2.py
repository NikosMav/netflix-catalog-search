#!/usr/bin/env python3
"""Build blinded eval-v2 judging pools.

For every dev and test query, take the union of the top 10 titles from every
retriever in ``build_methods`` (including field and plot ablations). Write one
JSON file per query with catalog fields only, in an order shuffled by the
pre-registered seed.

The files omit method names, ranks, scores, and the v1 relevance labels.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from retrieval.catalog import load_catalog
from retrieval.evaluate import build_methods

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "eval_v2.yaml"
DEV_QUERIES = ROOT / "data" / "labeled_queries.json"
TEST_QUERIES = ROOT / "data" / "eval_v2" / "test_queries.json"
OUT_DIR = ROOT / "data" / "eval_v2" / "blinded"
STATS_PATH = ROOT / "data" / "eval_v2" / "pool_stats.json"

CANDIDATE_FIELDS = (
    "show_id",
    "title",
    "type",
    "release_year",
    "listed_in",
    "cast",
    "director",
    "description",
)
FORBIDDEN_KEYS = {"method", "rank", "score", "relevant_show_ids", "v1_label", "notes", "intent"}


def _config_int(text: str, key: str) -> int:
    match = re.search(rf"^\s*{re.escape(key)}:\s*(\d+)\s*$", text, flags=re.MULTILINE)
    if not match:
        raise SystemExit(f"{key} not found in {CONFIG_PATH}")
    return int(match.group(1))


def load_protocol() -> tuple[int, int]:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    depth = _config_int(text, "depth")
    seed = _config_int(text, "shuffle_seed")
    return depth, seed


def load_split_queries() -> list[dict[str, str]]:
    """Query text only. v1 relevance labels are not read into the pool."""
    dev = json.loads(DEV_QUERIES.read_text(encoding="utf-8"))
    test = json.loads(TEST_QUERIES.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    for item in dev["queries"]:
        rows.append({"split": "dev", "query_id": item["id"], "query": item["query"]})
    for item in test["queries"]:
        rows.append({"split": "test", "query_id": item["id"], "query": item["query"]})
    return rows


def _year(value: object) -> int | None:
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
    except Exception:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return int(float(text))


def candidate_from_row(row) -> dict:
    record = {
        "show_id": str(row["show_id"]),
        "title": str(row["title"] or ""),
        "type": str(row["type"] or ""),
        "release_year": _year(row["release_year"]),
        "listed_in": str(row["listed_in"] or ""),
        "cast": str(row["cast"] or ""),
        "director": str(row["director"] or ""),
        "description": str(row["description"] or ""),
    }
    extra = set(record) - set(CANDIDATE_FIELDS)
    if extra or FORBIDDEN_KEYS & set(record):
        raise RuntimeError(f"candidate fields drifted: {sorted(record)}")
    return record


def _assert_blinded(payload: dict) -> None:
    allowed_top = {"split", "query_id", "query", "candidates"}
    if set(payload) != allowed_top:
        raise RuntimeError(f"unexpected blinded keys: {sorted(set(payload) - allowed_top)}")
    for candidate in payload["candidates"]:
        if set(candidate) != set(CANDIDATE_FIELDS):
            raise RuntimeError(f"unexpected candidate keys: {sorted(candidate)}")
        if FORBIDDEN_KEYS & set(candidate):
            raise RuntimeError("forbidden key in candidate")


def pool_query(methods, query: str, depth: int) -> list[str]:
    """Stable union: method order, then rank within a method. Scores are dropped."""
    ordered: list[str] = []
    seen: set[str] = set()
    for method in methods:
        hits = method.retriever.query(query, top_k=depth)
        for sid in hits["show_id"].astype(str).tolist():
            if sid not in seen:
                seen.add(sid)
                ordered.append(sid)
    return ordered


def main() -> None:
    depth, seed = load_protocol()
    queries = load_split_queries()
    catalog = load_catalog()
    by_id = {str(sid): i for i, sid in enumerate(catalog["show_id"].astype(str).tolist())}
    print(f"Building {depth}-deep pools for {len(queries)} queries (shuffle_seed={seed})")
    methods = build_methods(catalog, show_progress=True)
    system_names = [method.name for method in methods]

    rng = np.random.default_rng(seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_query = []

    for item in queries:
        pooled = pool_query(methods, item["query"], depth)
        order = rng.permutation(len(pooled))
        shuffled = [pooled[int(i)] for i in order]
        candidates = []
        for sid in shuffled:
            idx = by_id.get(sid)
            if idx is None:
                raise RuntimeError(f"pooled show_id {sid} missing from catalog")
            candidates.append(candidate_from_row(catalog.iloc[idx]))
        payload = {
            "split": item["split"],
            "query_id": item["query_id"],
            "query": item["query"],
            "candidates": candidates,
        }
        _assert_blinded(payload)
        blob = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        out = OUT_DIR / f"{item['split']}__{item['query_id']}.json"
        out.write_text(blob, encoding="utf-8")
        per_query.append(
            {
                "split": item["split"],
                "query_id": item["query_id"],
                "n_candidates": len(candidates),
            }
        )
        print(f"  {item['split']}/{item['query_id']}: {len(candidates)} candidates")

    sizes = [row["n_candidates"] for row in per_query]
    stats = {
        "pool_depth": depth,
        "shuffle_seed": seed,
        "n_systems": len(system_names),
        "systems": system_names,
        "n_queries": len(per_query),
        "n_dev_queries": sum(1 for row in per_query if row["split"] == "dev"),
        "n_test_queries": sum(1 for row in per_query if row["split"] == "test"),
        "n_candidate_slots": sum(sizes),
        "candidates_min": min(sizes) if sizes else 0,
        "candidates_max": max(sizes) if sizes else 0,
        "per_query": per_query,
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_DIR} and {STATS_PATH}")


if __name__ == "__main__":
    main()
