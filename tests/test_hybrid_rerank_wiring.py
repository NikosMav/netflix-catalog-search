"""hybrid+rerank must wrap hybrid(bm25+dense,meta), not hybrid(tfidf+dense)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from retrieval.bm25 import BM25Retriever
from retrieval.cli import _build_retriever
from retrieval.evaluate import build_methods
from retrieval.hybrid import HybridRetriever
from retrieval.sparse import SparseTfidfRetriever


class _FakeDense:
    """Stand-in for DenseRetriever: no model download / encode."""

    def __init__(self, catalog, text_field="text", show_progress=False, **_kwargs):
        self.catalog = catalog
        self.text_field = text_field

    def rank_indices(self, text: str, top_k: int = 100):
        n = min(top_k, len(self.catalog))
        return np.arange(n, dtype=int), np.ones(n, dtype=float)

    def query(self, text: str, top_k: int = 10) -> pd.DataFrame:
        idxs, scores = self.rank_indices(text, top_k=top_k)
        hits = self.catalog.iloc[idxs][["show_id", "title", "description", "type"]].copy()
        hits.insert(0, "rank", np.arange(1, len(hits) + 1))
        hits.insert(1, "score", scores)
        hits.insert(2, "method", f"dense:{self.text_field}")
        return hits.reset_index(drop=True)


def _tiny_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "show_id": ["s1", "s2", "s3"],
            "title": ["Vietnam War Doc", "Cooking Show", "Space Opera"],
            "description": [
                "A film about the war between Vietnam and the USA.",
                "Feel-good chefs compete in a cooking competition.",
                "Starships battle across a galaxy far away.",
            ],
            "type": ["Movie", "TV Show", "Movie"],
            "text": [
                "Vietnam War Doc A film about the war between Vietnam and the USA.",
                "Cooking Show Feel-good chefs compete in a cooking competition.",
                "Space Opera Starships battle across a galaxy far away.",
            ],
            "text_meta": [
                "Vietnam War Doc A film about the war between Vietnam and the USA. Documentaries United States",
                "Cooking Show Feel-good chefs compete in a cooking competition. Reality TV United Kingdom",
                "Space Opera Starships battle across a galaxy far away. Sci-Fi & Fantasy United States",
            ],
            "title_text": ["Vietnam War Doc", "Cooking Show", "Space Opera"],
        }
    )


def test_build_methods_hybrid_rerank_uses_bm25_meta_hybrid(monkeypatch):
    monkeypatch.setattr("retrieval.evaluate.DenseRetriever", _FakeDense)
    methods = {m.name: m.retriever for m in build_methods(_tiny_catalog(), show_progress=False)}
    hybrid_rerank = methods["hybrid+rerank"]
    base = hybrid_rerank.base
    assert isinstance(base, HybridRetriever)
    assert base.name == "hybrid(bm25+dense,meta)"
    assert isinstance(base.retrievers[0], BM25Retriever)
    assert base.retrievers[0].text_field == "text_meta"
    assert isinstance(base.retrievers[1], _FakeDense)
    assert base.retrievers[1].text_field == "text_meta"
    # Weaker TF-IDF hybrid must remain a separate method, not the CE first stage.
    weak = methods["hybrid(tfidf+dense)"]
    assert weak.name == "hybrid(tfidf+dense)"
    assert isinstance(weak.retrievers[0], SparseTfidfRetriever)


def test_cli_hybrid_rerank_uses_bm25_meta_hybrid(monkeypatch):
    monkeypatch.setattr("retrieval.cli.DenseRetriever", _FakeDense)
    retriever = _build_retriever("hybrid-rerank", _tiny_catalog(), show_progress=False)
    base = retriever.base
    assert isinstance(base, HybridRetriever)
    assert base.name == "hybrid(bm25+dense,meta)"
    assert isinstance(base.retrievers[0], BM25Retriever)
    assert base.retrievers[0].text_field == "text_meta"
    assert isinstance(base.retrievers[1], _FakeDense)
    assert base.retrievers[1].text_field == "text_meta"
