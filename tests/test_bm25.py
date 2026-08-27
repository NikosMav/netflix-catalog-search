"""BM25 retriever smoke tests (no dense model download)."""

from __future__ import annotations

import pandas as pd
import pytest

from retrieval.bm25 import BM25Retriever, tokenize


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
        }
    )


def test_tokenize_lowercase():
    assert tokenize("Vietnam War!") == ["vietnam", "war"]


def test_bm25_ranks_lexical_match_first():
    retriever = BM25Retriever(_tiny_catalog(), text_field="text")
    hits = retriever.query("war between vietnam and usa", top_k=2)
    assert hits.iloc[0]["show_id"] == "s1"
    assert list(hits["rank"]) == [1, 2]


def test_bm25_meta_uses_genre_tokens():
    retriever = BM25Retriever(_tiny_catalog(), text_field="text_meta")
    hits = retriever.query("Documentaries about Vietnam", top_k=1)
    assert hits.iloc[0]["show_id"] == "s1"


def test_bm25_empty_query_returns_empty():
    retriever = BM25Retriever(_tiny_catalog(), text_field="text")
    idxs, scores = retriever.rank_indices("!!!", top_k=5)
    assert len(idxs) == 0
    assert len(scores) == 0


def test_bm25_rejects_non_positive_top_k():
    retriever = BM25Retriever(_tiny_catalog(), text_field="text")
    with pytest.raises(ValueError, match="top_k"):
        retriever.rank_indices("vietnam", top_k=0)
