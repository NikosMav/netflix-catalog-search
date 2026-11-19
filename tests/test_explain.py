"""Explanations must use the retrievers' real scores. No model download."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from retrieval.bm25 import BM25Retriever, tokenize
from retrieval.boolean_retriever import BooleanRetriever
from retrieval.evaluate import RERANK_CANDIDATE_K
from retrieval.explain import (
    bm25_term_contributions,
    cosine,
    explain_results,
    highlight_html,
    how_it_works,
    metric_notes,
    rrf_contribution,
)
from retrieval.hybrid import HybridRetriever
from retrieval.rerank import CrossEncoderReranker
from retrieval.sparse import SparseTfidfRetriever


def _catalog() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "show_id": ["s1", "s2", "s3"],
            "title": ["Vietnam War Doc", "Cooking Show", "Space Opera"],
            "description": [
                "A film about the war between Vietnam and the USA.",
                "Feel-good chefs compete in a cooking competition.",
                "Starships battle across a galaxy far away.",
            ],
            "type": ["Movie", "TV Show", "Movie"],
            "listed_in": ["Documentaries", "Reality TV", "Sci-Fi & Fantasy"],
            "cast": ["", "Chef Amy", ""],
            "director": ["Ken Burns", "", ""],
            "country": ["United States", "United Kingdom", "United States"],
            "release_year": [2017, 2020, 2021],
        }
    )
    frame["text"] = (frame["title"] + " " + frame["description"]).str.strip()
    frame["text_meta"] = (
        frame["text"]
        + " "
        + frame["listed_in"]
        + " "
        + frame["cast"]
        + " "
        + frame["director"]
        + " "
        + frame["country"]
    ).str.strip()
    return frame


def test_bm25_term_contributions_sum_to_score():
    retriever = BM25Retriever(_catalog(), text_field="text")
    query = "war between vietnam and usa"
    scores = retriever.bm25.get_scores(tokenize(query))
    for doc_index, score in enumerate(scores):
        parts = bm25_term_contributions(retriever, query, doc_index)
        assert sum(parts.values()) == pytest.approx(float(score))

    hits = retriever.query(query, top_k=3)
    explained = explain_results(retriever, query, hits)
    assert len(explained) == len(hits)
    for exp, score in zip(explained, hits["score"]):
        assert sum(term.contribution for term in exp.terms) == pytest.approx(float(score))
        assert exp.score == pytest.approx(float(score))


def test_repeated_query_term_is_counted_each_time():
    retriever = BM25Retriever(_catalog(), text_field="text")
    once = bm25_term_contributions(retriever, "vietnam", 0)
    twice = bm25_term_contributions(retriever, "vietnam vietnam", 0)
    assert twice["vietnam"] == pytest.approx(once["vietnam"] * 2)
    scores = retriever.bm25.get_scores(tokenize("vietnam vietnam"))
    assert twice["vietnam"] == pytest.approx(float(scores[0]))


def test_bm25_names_matched_fields():
    cat = _catalog()
    lexical = BM25Retriever(cat, text_field="text")
    hits = lexical.query("vietnam war", top_k=1)
    exp = explain_results(lexical, "vietnam war", hits)[0]
    fields = {term.term: term.fields for term in exp.terms}
    assert "title" in fields["vietnam"]
    assert "description" in fields["vietnam"]
    assert "title" in fields["war"]

    meta = BM25Retriever(cat, text_field="text_meta")
    hits = meta.query("Documentaries", top_k=1)
    exp = explain_results(meta, "Documentaries", hits)[0]
    genres = next(term for term in exp.terms if term.term == "documentaries")
    assert genres.fields == ("genres",)


def test_bm25_empty_query_explains_nothing():
    retriever = BM25Retriever(_catalog(), text_field="text")
    hits = retriever.query("!!!", top_k=5)
    assert explain_results(retriever, "!!!", hits) == []


def test_tfidf_term_contributions_sum_to_cosine():
    retriever = SparseTfidfRetriever(_catalog(), text_field="text", max_df=1.0)
    query = "vietnam war"
    hits = retriever.query(query, top_k=3)
    explained = explain_results(retriever, query, hits)
    assert any("vietnam" in term.term for exp in explained for term in exp.terms)
    for exp, score in zip(explained, hits["score"]):
        assert sum(term.contribution for term in exp.terms) == pytest.approx(float(score), abs=1e-6)
        assert exp.kind == "tfidf"


def test_boolean_jaccard_matches_retriever():
    retriever = BooleanRetriever(_catalog(), text_field="text", max_df=1.0)
    query = "vietnam war"
    hits = retriever.query(query, top_k=3)
    explained = explain_results(retriever, query, hits)
    for exp, score in zip(explained, hits["score"]):
        assert exp.score == pytest.approx(float(score))
        assert exp.kind == "boolean"
    assert explained[0].highlight_terms


def test_rrf_contribution_formula():
    assert rrf_contribution(None, 60) == 0.0
    assert rrf_contribution(1, 60) == pytest.approx(1 / 61)
    assert rrf_contribution(1, 60) + rrf_contribution(2, 60) == pytest.approx(1 / 61 + 1 / 62)


class _ListRanker:
    def __init__(self, order: list[int], name: str):
        self.order = list(order)
        self.name = name

    def rank_indices(self, text: str, top_k: int = 100):
        idxs = np.array(self.order[:top_k], dtype=int)
        scores = np.ones(len(idxs), dtype=float)
        return idxs, scores


def test_hybrid_rrf_contributions_sum_to_fused_score():
    cat = _catalog()
    hybrid = HybridRetriever(
        cat,
        retrievers=[_ListRanker([0, 1], "list-a"), _ListRanker([1, 2], "list-b")],
        rrf_k=60,
        candidate_k=10,
    )
    hits = hybrid.query("anything", top_k=3)
    explained = explain_results(hybrid, "anything", hits)
    by_id = {sid: exp for sid, exp in zip(hits["show_id"], explained)}

    assert by_id["s1"].sources[0].rank == 1
    assert by_id["s1"].sources[1].rank is None
    assert by_id["s1"].sources[1].contribution == 0.0
    assert by_id["s1"].sources[0].contribution == pytest.approx(1 / 61)

    assert by_id["s2"].sources[0].rank == 2
    assert by_id["s2"].sources[1].rank == 1
    assert sum(source.contribution for source in by_id["s2"].sources) == pytest.approx(1 / 61 + 1 / 62)

    for exp, score in zip(explained, hits["score"]):
        assert sum(source.contribution for source in exp.sources) == pytest.approx(float(score))
        assert exp.score == pytest.approx(float(score))
        assert "1/(60 + rank)" in exp.summary


class _ScoreByPosition:
    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        return [float(i) for i in range(len(pairs))]


def test_rerank_reports_cross_encoder_score_and_movement():
    cat = _catalog()
    base = _ListRanker([0, 1, 2], "first-stage")
    reranker = CrossEncoderReranker(
        cat,
        base=base,
        text_field="text",
        candidate_k=3,
        name="hybrid+rerank",
    )
    reranker._model = _ScoreByPosition()
    hits = reranker.query("vietnam", top_k=3)
    explained = explain_results(reranker, "vietnam", hits)
    by_id = {sid: exp for sid, exp in zip(hits["show_id"], explained)}

    # Base order is s1, s2, s3. The fake cross-encoder prefers later candidates,
    # so s3 moves from first-stage rank 3 to rank 1.
    assert by_id["s3"].rerank.first_stage_rank == 3
    assert by_id["s3"].rerank.final_rank == 1
    assert by_id["s3"].rerank.places == 2
    assert "Moved up 2" in by_id["s3"].summary
    assert "Cross-encoder score" in by_id["s3"].summary

    assert by_id["s2"].rerank.places == 0
    assert "Stayed at rank 2" in by_id["s2"].summary

    assert by_id["s1"].rerank.places == -2
    assert "Moved down 2" in by_id["s1"].summary

    for exp, score in zip(explained, hits["score"]):
        assert exp.score == pytest.approx(float(score))
        assert exp.rerank.cross_encoder_score == pytest.approx(float(score))
        assert exp.kind == "rerank"


def test_cosine_of_unit_vectors():
    assert cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert cosine(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 0.0


def test_highlight_html_marks_whole_words_and_escapes():
    assert highlight_html("The Vietnam War", ("vietnam", "war")) == (
        "The <mark>Vietnam</mark> <mark>War</mark>"
    )
    assert "<mark>" not in highlight_html("warfare", ("war",))
    escaped = highlight_html("<b>Vietnam</b>", ("vietnam",))
    assert "<b>" not in escaped
    assert "<mark>Vietnam</mark>" in escaped


def test_how_it_works_uses_pipeline_constants():
    text = how_it_works()
    assert "1/(60 + rank)" in text
    assert f"top {RERANK_CANDIDATE_K}" in text
    assert RERANK_CANDIDATE_K == 50
    assert "watch history" in text


def test_metric_notes_use_the_given_count_and_no_result_values():
    text = metric_notes(28)
    assert "28" in text
    assert "R@5" in text and "R@10" in text
    assert "nDCG@10" in text and "MRR" in text
    assert "0.7001" not in text
    assert "0.3159" not in text
    assert "28" not in metric_notes(3)


def test_streamlit_app_parses():
    from pathlib import Path

    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    compile(source, "app/streamlit_app.py", "exec")
    assert "results/eval_v2/metrics.json" in source
    assert "spotcheck_sample.json" in source
    assert "judgments.json" not in source
    assert "eval_metrics.json" not in source
