"""Reproducible evaluation over the hand-labeled query set."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from retrieval.bm25 import BM25Retriever
from retrieval.boolean_retriever import BooleanRetriever
from retrieval.catalog import load_catalog
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.metrics import aggregate_mean, mrr, ndcg_at_k, recall_at_k
from retrieval.rerank import CrossEncoderReranker
from retrieval.sparse import SparseTfidfRetriever

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = REPO_ROOT / "data" / "labeled_queries.json"
DEFAULT_RESULTS = REPO_ROOT / "results" / "eval_metrics.json"

# Cross-encoder second stage: rerank top-50 of the first-stage list.
RERANK_CANDIDATE_K = 50


@dataclass
class MethodSpec:
    name: str
    retriever: Any


def load_labeled_queries(path: str | Path | None = None) -> list[dict[str, Any]]:
    p = Path(path) if path else DEFAULT_LABELS
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    queries = data["queries"]
    for q in queries:
        if "relevant_show_ids" not in q:
            raise ValueError(f"Query {q.get('id')} missing relevant_show_ids")
    return queries


def validate_labels(catalog: pd.DataFrame, queries: list[dict[str, Any]]) -> None:
    known = set(catalog["show_id"].astype(str))
    for q in queries:
        missing = [sid for sid in q["relevant_show_ids"] if sid not in known]
        if missing:
            raise ValueError(f"Query {q['id']}: unknown show_ids {missing}")


def build_methods(catalog: pd.DataFrame, show_progress: bool = False) -> list[MethodSpec]:
    """Build the full ablation suite for offline eval.

    Same 28 labeled queries; methods vary representation / ranking only.
    ``text`` = title+description; ``text_meta`` adds listed_in/cast/director/country.
    """
    boolean = BooleanRetriever(catalog, text_field="text")
    tfidf = SparseTfidfRetriever(catalog, text_field="text")
    tfidf_meta = SparseTfidfRetriever(catalog, text_field="text_meta")
    bm25 = BM25Retriever(catalog, text_field="text")
    bm25_meta = BM25Retriever(catalog, text_field="text_meta")
    dense = DenseRetriever(catalog, text_field="text", show_progress=show_progress)
    dense_meta = DenseRetriever(catalog, text_field="text_meta", show_progress=show_progress)
    dense_title = DenseRetriever(catalog, text_field="title_text", show_progress=show_progress)
    hybrid = HybridRetriever(
        catalog,
        retrievers=[tfidf, dense],
        name="hybrid(tfidf+dense)",
    )
    hybrid_bm25 = HybridRetriever(
        catalog,
        retrievers=[bm25_meta, dense_meta],
        name="hybrid(bm25+dense,meta)",
    )
    dense_rerank = CrossEncoderReranker(
        catalog,
        base=dense,
        text_field="text_meta",
        candidate_k=RERANK_CANDIDATE_K,
        name="dense+rerank",
        show_progress=show_progress,
    )
    hybrid_rerank = CrossEncoderReranker(
        catalog,
        base=hybrid,
        text_field="text_meta",
        candidate_k=RERANK_CANDIDATE_K,
        name="hybrid+rerank",
        show_progress=show_progress,
    )
    return [
        MethodSpec("boolean", boolean),
        MethodSpec("tf-idf", tfidf),
        MethodSpec("tf-idf(desc+meta)", tfidf_meta),
        MethodSpec("bm25", bm25),
        MethodSpec("bm25(desc+meta)", bm25_meta),
        MethodSpec("dense(title+desc)", dense),
        MethodSpec("dense(title+desc+meta)", dense_meta),
        MethodSpec("dense(title-only)", dense_title),
        MethodSpec("hybrid(tfidf+dense)", hybrid),
        MethodSpec("hybrid(bm25+dense,meta)", hybrid_bm25),
        MethodSpec("dense+rerank", dense_rerank),
        MethodSpec("hybrid+rerank", hybrid_rerank),
    ]


def evaluate_method(
    method: MethodSpec,
    queries: list[dict[str, Any]],
    ks: tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    max_k = max(ks)
    per_q: dict[str, list[float]] = {f"recall@{k}": [] for k in ks}
    per_q.update({f"ndcg@{k}": [] for k in ks})
    per_q["mrr"] = []

    for item in queries:
        hits = method.retriever.query(item["query"], top_k=max_k)
        retrieved = hits["show_id"].astype(str).tolist()
        relevant = set(item["relevant_show_ids"])
        for k in ks:
            per_q[f"recall@{k}"].append(recall_at_k(retrieved, relevant, k))
            per_q[f"ndcg@{k}"].append(ndcg_at_k(retrieved, relevant, k))
        per_q["mrr"].append(mrr(retrieved, relevant))

    return {metric: round(aggregate_mean(vals), 4) for metric, vals in per_q.items()}


def run_evaluation(
    labels_path: str | Path | None = None,
    csv_path: str | Path | None = None,
    show_progress: bool = True,
    ks: tuple[int, ...] = (5, 10),
) -> pd.DataFrame:
    catalog = load_catalog(csv_path)
    queries = load_labeled_queries(labels_path)
    validate_labels(catalog, queries)
    methods = build_methods(catalog, show_progress=show_progress)

    rows = []
    for method in methods:
        metrics = evaluate_method(method, queries, ks=ks)
        rows.append({"method": method.name, "n_queries": len(queries), **metrics})
    return pd.DataFrame(rows)


def results_to_markdown(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if c != "n_queries"]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            if isinstance(val, float):
                cells.append(f"{val:.4f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_results(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    out = Path(path) if path else DEFAULT_RESULTS
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_queries": int(df["n_queries"].iloc[0]) if len(df) else 0,
        "metrics": df.drop(columns=["n_queries"]).to_dict(orient="records"),
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def qualitative_failures(
    catalog: pd.DataFrame,
    methods: list[MethodSpec],
    queries: list[dict[str, Any]],
    examples: list[str] | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Side-by-side top-k for selected query ids (for README failure cases)."""
    by_id = {q["id"]: q for q in queries}
    ids = examples or [q["id"] for q in queries[:4]]
    out = []
    for qid in ids:
        item = by_id[qid]
        block = {"id": qid, "query": item["query"], "relevant_show_ids": item["relevant_show_ids"], "runs": {}}
        for method in methods:
            hits = method.retriever.query(item["query"], top_k=top_k)
            block["runs"][method.name] = [
                {"show_id": r.show_id, "title": r.title, "score": float(r.score)}
                for r in hits.itertuples(index=False)
            ]
        out.append(block)
    return out
