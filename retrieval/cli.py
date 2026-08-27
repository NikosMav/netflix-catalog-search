"""CLI: `python -m retrieval <command>`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from retrieval.boolean_retriever import BooleanRetriever
from retrieval.catalog import load_catalog
from retrieval.dense import DenseRetriever
from retrieval.evaluate import (
    build_methods,
    load_labeled_queries,
    qualitative_failures,
    results_to_markdown,
    run_evaluation,
    save_results,
)
from retrieval.hybrid import HybridRetriever
from retrieval.sparse import SparseTfidfRetriever

REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_retriever(name: str, catalog, show_progress: bool = False):
    name = name.lower()
    if name in {"boolean", "bow"}:
        return BooleanRetriever(catalog)
    if name in {"tfidf", "tf-idf"}:
        return SparseTfidfRetriever(catalog)
    if name in {"dense", "dense(title+desc)"}:
        return DenseRetriever(catalog, text_field="text", show_progress=show_progress)
    if name in {"dense-title", "dense(title-only)"}:
        return DenseRetriever(catalog, text_field="title_text", show_progress=show_progress)
    if name in {"hybrid", "hybrid(tfidf+dense)"}:
        tfidf = SparseTfidfRetriever(catalog)
        dense = DenseRetriever(catalog, text_field="text", show_progress=show_progress)
        return HybridRetriever(catalog, retrievers=[tfidf, dense], name="hybrid(tfidf+dense)")
    raise SystemExit(f"Unknown method: {name}. Choose boolean|tfidf|dense|dense-title|hybrid")


def cmd_query(args: argparse.Namespace) -> int:
    catalog = load_catalog(args.data)
    retriever = _build_retriever(args.method, catalog, show_progress=not args.quiet)
    hits = retriever.query(args.query, top_k=args.top_k)
    if args.json:
        print(hits.to_json(orient="records", indent=2))
    else:
        print(f"method={args.method}  query={args.query!r}  top_k={args.top_k}")
        print("-" * 72)
        for row in hits.itertuples(index=False):
            desc = (row.description or "")[:80].replace("\n", " ")
            print(f"{row.rank:2d}. {row.score:7.4f}  {row.title}  [{row.show_id}]")
            print(f"    {desc}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    df = run_evaluation(
        labels_path=args.labels,
        csv_path=args.data,
        show_progress=not args.quiet,
        ks=tuple(args.ks),
    )
    md = results_to_markdown(df)
    print(md)
    out = save_results(df, args.out)
    print(f"\nWrote {out}")

    if args.failures:
        catalog = load_catalog(args.data)
        methods = build_methods(catalog, show_progress=False)
        queries = load_labeled_queries(args.labels)
        # Pick queries that illustrate sparse vs dense wins
        example_ids = [
            "vietnam_war",
            "mickey_mouse",
            "semantic_cooking_show",
            "semantic_scandi_crime",
            "chess_drama",
            "exact_title_stranger_things",
        ]
        example_ids = [e for e in example_ids if any(q["id"] == e for q in queries)]
        failures = qualitative_failures(catalog, methods, queries, examples=example_ids, top_k=5)
        fail_path = Path(args.out).with_name("qualitative_examples.json") if args.out else (
            REPO_ROOT / "results" / "qualitative_examples.json"
        )
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        fail_path.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {fail_path}")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    """Precompute and cache dense embeddings for title+desc and title-only."""
    catalog = load_catalog(args.data)
    print(f"Indexing {len(catalog)} titles…")
    DenseRetriever(catalog, text_field="text", show_progress=not args.quiet)
    DenseRetriever(catalog, text_field="title_text", show_progress=not args.quiet)
    print("Dense embedding caches ready under .cache/embeddings/")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m retrieval",
        description="Netflix catalog text retrieval (Boolean / TF-IDF / dense / hybrid)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--data", type=Path, default=None, help="Path to netflix_titles.csv")
        sp.add_argument("--quiet", action="store_true", help="Less progress output")

    q = sub.add_parser("query", help="Run a natural-language catalog query")
    add_common(q)
    q.add_argument("query", type=str, help="Query text")
    q.add_argument(
        "--method",
        default="dense",
        help="boolean|tfidf|dense|dense-title|hybrid (default: dense)",
    )
    q.add_argument("--top-k", type=int, default=10)
    q.add_argument("--json", action="store_true")
    q.set_defaults(func=cmd_query)

    e = sub.add_parser("eval", help="Evaluate all methods on the labeled query set")
    add_common(e)
    e.add_argument("--labels", type=Path, default=None)
    e.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "eval_metrics.json")
    e.add_argument("--ks", type=int, nargs="+", default=[5, 10])
    e.add_argument("--failures", action="store_true", help="Also write qualitative examples JSON")
    e.set_defaults(func=cmd_eval)

    i = sub.add_parser("index", help="Precompute dense embedding caches")
    add_common(i)
    i.set_defaults(func=cmd_index)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
