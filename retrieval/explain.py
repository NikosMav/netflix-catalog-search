"""Plain-language explanations computed from the retrievers that ranked a hit.

The numbers come from the same indexes and formulas as retrieval. Nothing here
is a hand-written score.
"""

from __future__ import annotations

import html
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from retrieval.bm25 import BM25Retriever, tokenize
from retrieval.boolean_retriever import BooleanRetriever
from retrieval.dense import DenseRetriever
from retrieval.evaluate import RERANK_CANDIDATE_K
from retrieval.hybrid import HybridRetriever
from retrieval.rerank import CrossEncoderReranker
from retrieval.sparse import SparseTfidfRetriever

# Columns that can be named in a "where did this term match?" line.
_FIELD_SOURCES = {
    "text": ("title", "description"),
    "text_meta": ("title", "description", "listed_in", "cast", "director", "country"),
    "text_plot": ("title", "description", "plot"),
    "title_text": ("title",),
}
_FIELD_LABEL = {
    "title": "title",
    "description": "description",
    "listed_in": "genres",
    "cast": "cast",
    "director": "director",
    "country": "country",
    "plot": "plot",
}
_INDEX_PHRASE = {
    "text": "title and description",
    "text_meta": "title, description, genres, cast, director, and country",
    "text_plot": "title, description, and plot",
    "title_text": "the title",
}
_SCORE_NAME = {
    "bm25": "BM25",
    "tfidf": "TF-IDF",
    "boolean": "Jaccard",
    "dense": "cosine",
    "hybrid": "RRF",
    "rerank": "cross-encoder",
}
_MAX_BOOLEAN_TERMS = 10
_MARK_RE_CACHE: dict[tuple[str, ...], re.Pattern[str]] = {}


@dataclass(frozen=True)
class TermContribution:
    term: str
    contribution: float
    fields: tuple[str, ...]


@dataclass(frozen=True)
class SourceContribution:
    name: str
    rank: int | None
    contribution: float


@dataclass(frozen=True)
class RerankMove:
    cross_encoder_score: float
    first_stage_rank: int | None
    final_rank: int

    @property
    def places(self) -> int | None:
        if self.first_stage_rank is None:
            return None
        return self.first_stage_rank - self.final_rank


@dataclass(frozen=True)
class Explanation:
    kind: str
    summary: str
    details: tuple[str, ...]
    score: float
    rank: int
    highlight_terms: tuple[str, ...] = ()
    terms: tuple[TermContribution, ...] = ()
    sources: tuple[SourceContribution, ...] = ()
    rerank: RerankMove | None = None

    def score_text(self) -> str:
        places = 6 if self.kind == "hybrid" else 4
        return f"{_SCORE_NAME[self.kind]} {format_score(self.score, places)}"


def format_score(value: float, places: int = 4) -> str:
    return f"{value:.{places}f}"


def rrf_contribution(rank: int | None, rrf_k: int) -> float:
    """One list's share of a reciprocal-rank-fusion score.

    A title missing from that list contributes 0. Otherwise the share is
    ``1 / (rrf_k + rank)`` with ``rank`` starting at 1, matching
    ``retrieval.hybrid.reciprocal_rank_fusion``.
    """
    if rank is None:
        return 0.0
    if rank < 1:
        raise ValueError("rank must be >= 1")
    if rrf_k < 0:
        raise ValueError("rrf_k must be >= 0")
    return 1.0 / (rrf_k + rank)


def bm25_term_contributions(
    retriever: BM25Retriever, query: str, doc_index: int
) -> dict[str, float]:
    """Per query term, the BM25 weight for one document.

    Repeated query terms are counted once per occurrence, matching
    ``BM25Okapi.get_scores``. Zero-weight terms are omitted. The values sum
    to that document's BM25 score.
    """
    bm25 = retriever.bm25
    tokens = tokenize(query)
    if not tokens or doc_index < 0 or doc_index >= bm25.corpus_size:
        return {}
    counts = Counter(tokens)
    order: list[str] = []
    for token in tokens:
        if token not in order:
            order.append(token)
    dl = bm25.doc_len[doc_index]
    tfs = bm25.doc_freqs[doc_index]
    k1 = bm25.k1
    b = bm25.b
    avgdl = bm25.avgdl
    length_norm = k1 * (1.0 - b + b * dl / avgdl)
    out: dict[str, float] = {}
    for term in order:
        tf = tfs.get(term) or 0
        idf = bm25.idf.get(term) or 0.0
        denom = tf + length_norm
        once = 0.0 if denom == 0 else idf * (tf * (k1 + 1.0) / denom)
        total = float(once * counts[term])
        if total != 0.0:
            out[term] = total
    return out


def tfidf_term_contributions(
    retriever: SparseTfidfRetriever, query: str, doc_index: int
) -> dict[str, float]:
    """Per overlapping feature, its share of the TF-IDF cosine.

    The retriever stores L2-normalized TF-IDF vectors, so the cosine is the
    dot product. Each share is ``query_weight * document_weight``. The values
    sum to the cosine.
    """
    query_vec = retriever.vectorizer.transform([query])
    doc_vec = retriever.doc_matrix[doc_index]
    product = query_vec.multiply(doc_vec).tocoo()
    names = retriever.vectorizer.get_feature_names_out()
    acc: dict[str, float] = {}
    for col, val in zip(product.col, product.data):
        weight = float(val)
        if weight == 0.0:
            continue
        term = str(names[int(col)])
        acc[term] = acc.get(term, 0.0) + weight
    return acc


def cosine(query_vec: np.ndarray, doc_vec: np.ndarray) -> float:
    """Cosine similarity, the dense retriever's score."""
    query_arr = np.asarray(query_vec, dtype=np.float64).ravel()
    doc_arr = np.asarray(doc_vec, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(query_arr) * np.linalg.norm(doc_arr))
    if denom == 0.0:
        return 0.0
    return float(np.dot(query_arr, doc_arr) / denom)


def how_it_works() -> str:
    """Short description of the default pipeline, using the code's constants."""
    rrf_k = int(HybridRetriever.__dataclass_fields__["rrf_k"].default)
    pool = RERANK_CANDIDATE_K
    return (
        "Search ranks this fixed Netflix catalog from title text and the other "
        "catalog fields below. Ranking does not use watch history.\n\n"
        "BM25 scores overlapping words. MiniLM encodes the query and each title "
        "and ranks them by cosine similarity, so a paraphrase can match when the "
        "wording differs. "
        f"Hybrid search adds 1/({rrf_k} + rank) from the BM25 list and the MiniLM "
        "list, which keeps those two score scales separate. "
        f"Hybrid + rerank asks a cross-encoder to rescore the top {pool} fused "
        "titles and returns that order.\n\n"
        "The BM25, TF-IDF, and Boolean options read the title and description. "
        "Hybrid and hybrid + rerank also read genres, cast, director, and country. "
        "The first dense or reranked query downloads the MiniLM models. Later "
        "queries stay on CPU and use the local cache."
    )


def metric_notes(n_queries: int) -> str:
    """Definitions for the committed metrics. ``n_queries`` comes from the JSON."""
    return (
        f"These means are over {n_queries} scored queries. "
        "R@5 and R@10 are the fraction of labeled relevant titles that appear in "
        "the top 5 and top 10. "
        "nDCG@10 scores the top 10 with binary labels, giving more credit when a "
        "relevant title is closer to rank 1. "
        "MRR is 1 divided by the rank of the first relevant title, averaged over "
        "the queries."
    )


def highlight_html(text: str, terms: tuple[str, ...] | list[str]) -> str:
    """Escape ``text`` and wrap whole-word matches in ``<mark>``."""
    raw = text or ""
    tokens: list[str] = []
    seen: set[str] = set()
    for term in terms:
        for part in str(term).lower().split():
            if len(part) < 2 or not part.isalnum() or part in seen:
                continue
            seen.add(part)
            tokens.append(part)
    if not tokens or not raw:
        return html.escape(raw)
    key = tuple(sorted(tokens, key=len, reverse=True))
    pattern = _MARK_RE_CACHE.get(key)
    if pattern is None:
        pattern = re.compile(r"\b(" + "|".join(re.escape(tok) for tok in key) + r")\b", re.IGNORECASE)
        _MARK_RE_CACHE[key] = pattern
    pieces: list[str] = []
    last = 0
    for match in pattern.finditer(raw):
        pieces.append(html.escape(raw[last : match.start()]))
        pieces.append("<mark>" + html.escape(match.group(0)) + "</mark>")
        last = match.end()
    pieces.append(html.escape(raw[last:]))
    return "".join(pieces)


def explain_results(
    retriever,
    query: str,
    hits: pd.DataFrame,
) -> list[Explanation]:
    """One explanation per row of ``hits``, which must come from ``retriever.query``."""
    if hits is None or len(hits) == 0:
        return []
    rows = _hit_rows(retriever.catalog, hits)
    if isinstance(retriever, CrossEncoderReranker):
        return _explain_rerank(retriever, query, rows)
    if isinstance(retriever, HybridRetriever):
        return _explain_hybrid(retriever, query, rows)
    if isinstance(retriever, BM25Retriever):
        return _explain_bm25(retriever, query, rows)
    if isinstance(retriever, SparseTfidfRetriever):
        return _explain_tfidf(retriever, query, rows)
    if isinstance(retriever, BooleanRetriever):
        return _explain_boolean(retriever, query, rows)
    if isinstance(retriever, DenseRetriever):
        return _explain_dense(retriever, query, rows)
    raise TypeError(f"Cannot explain {type(retriever).__name__}")


def _hit_rows(catalog: pd.DataFrame, hits: pd.DataFrame) -> list[tuple[int, int, float]]:
    id_map: dict[str, int] = {}
    for i, sid in enumerate(catalog["show_id"].tolist()):
        id_map.setdefault(str(sid), i)
    rows: list[tuple[int, int, float]] = []
    for i in range(len(hits)):
        sid = str(hits.iloc[i]["show_id"])
        if sid not in id_map:
            raise KeyError(f"show_id {sid} is not in the catalog")
        rows.append((id_map[sid], int(hits.iloc[i]["rank"]), float(hits.iloc[i]["score"])))
    return rows


def _phrase_for(retriever) -> str:
    field = getattr(retriever, "text_field", "text")
    return _INDEX_PHRASE.get(field, "the indexed text")


def _columns_for(retriever) -> tuple[str, ...]:
    field = getattr(retriever, "text_field", "text")
    return _FIELD_SOURCES.get(field, ("title", "description"))


def _field_blob(row: pd.Series, column: str) -> str:
    if column not in row.index:
        return ""
    val = row[column]
    if pd.isna(val):
        return ""
    return str(val)


def _bm25_fields(row: pd.Series, columns: tuple[str, ...], term: str) -> tuple[str, ...]:
    matched = []
    for column in columns:
        if term in set(tokenize(_field_blob(row, column))):
            matched.append(_FIELD_LABEL[column])
    return tuple(matched)


def _analyzed_fields(analyzer, row: pd.Series, columns: tuple[str, ...]) -> dict[str, set[str]]:
    return {column: set(analyzer(_field_blob(row, column))) for column in columns}


def _fields_from_analysis(
    analysis: dict[str, set[str]], columns: tuple[str, ...], term: str
) -> tuple[str, ...]:
    return tuple(_FIELD_LABEL[column] for column in columns if term in analysis.get(column, ()))


def _highlight_tokens(terms: list[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for term in terms:
        for part in term.lower().split():
            if len(part) < 2 or not part.isalnum() or part in seen:
                continue
            seen.add(part)
            out.append(part)
    return tuple(out)


def _field_phrase(fields: tuple[str, ...]) -> str:
    return ", ".join(fields) if fields else "indexed text"


def _explain_bm25(retriever: BM25Retriever, query: str, rows: list[tuple[int, int, float]]) -> list[Explanation]:
    columns = _columns_for(retriever)
    phrase = _phrase_for(retriever)
    explanations: list[Explanation] = []
    for doc_index, rank, _score in rows:
        parts = bm25_term_contributions(retriever, query, doc_index)
        total = float(sum(parts.values()))
        row = retriever.catalog.iloc[doc_index]
        ordered = sorted(parts.items(), key=lambda item: (-item[1], item[0]))
        terms: list[TermContribution] = []
        details: list[str] = []
        for term, weight in ordered:
            fields = _bm25_fields(row, columns, term)
            terms.append(TermContribution(term, weight, fields))
            details.append(f"`{term}` — {format_score(weight)} — {_field_phrase(fields)}")
        summary = (
            f"BM25 score {format_score(total)} on {phrase}. "
            "Each term's contribution is listed below. They add up to this score."
        )
        if not details:
            summary = f"BM25 score {format_score(total)} on {phrase}. No query term contributed."
        explanations.append(
            Explanation(
                kind="bm25",
                summary=summary,
                details=tuple(details),
                score=total,
                rank=rank,
                highlight_terms=_highlight_tokens([term.term for term in terms]),
                terms=tuple(terms),
            )
        )
    return explanations


def _explain_tfidf(
    retriever: SparseTfidfRetriever, query: str, rows: list[tuple[int, int, float]]
) -> list[Explanation]:
    columns = _columns_for(retriever)
    phrase = _phrase_for(retriever)
    analyzer = retriever.vectorizer.build_analyzer()
    explanations: list[Explanation] = []
    for doc_index, rank, _score in rows:
        parts = tfidf_term_contributions(retriever, query, doc_index)
        total = float(sum(parts.values()))
        row = retriever.catalog.iloc[doc_index]
        analysis = _analyzed_fields(analyzer, row, columns)
        ordered = sorted(parts.items(), key=lambda item: (-item[1], item[0]))
        terms: list[TermContribution] = []
        details: list[str] = []
        for term, weight in ordered:
            fields = _fields_from_analysis(analysis, columns, term)
            terms.append(TermContribution(term, weight, fields))
            details.append(f"`{term}` — {format_score(weight)} — {_field_phrase(fields)}")
        summary = (
            f"TF-IDF cosine {format_score(total)} on {phrase}. "
            "Each number is that term's share of the dot product. They add up to this cosine."
        )
        if not details:
            summary = (
                f"TF-IDF cosine {format_score(total)} on {phrase}. "
                "No query term was indexed for this title."
            )
        explanations.append(
            Explanation(
                kind="tfidf",
                summary=summary,
                details=tuple(details),
                score=total,
                rank=rank,
                highlight_terms=_highlight_tokens([term.term for term in terms]),
                terms=tuple(terms),
            )
        )
    return explanations


def _explain_boolean(
    retriever: BooleanRetriever, query: str, rows: list[tuple[int, int, float]]
) -> list[Explanation]:
    columns = _columns_for(retriever)
    phrase = _phrase_for(retriever)
    names = retriever.vectorizer.get_feature_names_out()
    analyzer = retriever.vectorizer.build_analyzer()
    explanations: list[Explanation] = []
    for doc_index, rank, _score in rows:
        query_vec = retriever.vectorizer.transform([query])
        doc_vec = retriever.doc_matrix[doc_index]
        query_idx = {int(i) for i in query_vec.indices}
        doc_idx = {int(i) for i in doc_vec.indices}
        overlap = query_idx & doc_idx
        union = query_idx | doc_idx
        score = (len(overlap) / len(union)) if union else 0.0
        row = retriever.catalog.iloc[doc_index]
        analysis = _analyzed_fields(analyzer, row, columns)
        labels = sorted(str(names[i]) for i in overlap)
        shown = labels[:_MAX_BOOLEAN_TERMS]
        details = []
        for term in shown:
            fields = _fields_from_analysis(analysis, columns, term)
            details.append(f"`{term}` — {_field_phrase(fields)}")
        hidden = len(labels) - len(shown)
        if hidden > 0:
            details.append(f"{hidden} more shared terms")
        if labels:
            summary = (
                f"Jaccard {format_score(score)} on {phrase}: "
                f"{len(labels)} shared terms out of {len(union)} in the union."
            )
        else:
            summary = f"Jaccard {format_score(score)} on {phrase}. No shared terms."
        explanations.append(
            Explanation(
                kind="boolean",
                summary=summary,
                details=tuple(details),
                score=float(score),
                rank=rank,
                highlight_terms=_highlight_tokens(labels),
            )
        )
    return explanations


def _explain_dense(retriever: DenseRetriever, query: str, rows: list[tuple[int, int, float]]) -> list[Explanation]:
    query_vec = retriever._encode([query])[0]
    phrase = _phrase_for(retriever)
    explanations: list[Explanation] = []
    for doc_index, rank, _score in rows:
        similarity = cosine(query_vec, retriever.embeddings[doc_index])
        explanations.append(
            Explanation(
                kind="dense",
                summary=(
                    f"Cosine similarity {format_score(similarity)} between the query "
                    f"and this title ({phrase}, MiniLM)."
                ),
                details=(),
                score=float(similarity),
                rank=rank,
            )
        )
    return explanations


def _source_name(retriever) -> str:
    if isinstance(retriever, BM25Retriever):
        return "BM25"
    if isinstance(retriever, SparseTfidfRetriever):
        return "TF-IDF"
    if isinstance(retriever, BooleanRetriever):
        return "Boolean"
    if isinstance(retriever, DenseRetriever):
        return "Dense"
    name = getattr(retriever, "name", "")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return type(retriever).__name__


def _source_names(retrievers: list) -> list[str]:
    names = [_source_name(retriever) for retriever in retrievers]
    if len(names) == len(set(names)):
        return names
    return [f"{name} {i}" for i, name in enumerate(names, start=1)]


def _rank_map(idxs: np.ndarray) -> dict[int, int]:
    ranks: dict[int, int] = {}
    for rank, idx in enumerate(np.asarray(idxs), start=1):
        ranks.setdefault(int(idx), rank)
    return ranks


def _explain_hybrid(
    retriever: HybridRetriever, query: str, rows: list[tuple[int, int, float]]
) -> list[Explanation]:
    names = _source_names(list(retriever.retrievers))
    rank_maps = []
    for source in retriever.retrievers:
        idxs, _scores = source.rank_indices(query, top_k=retriever.candidate_k)
        rank_maps.append(_rank_map(idxs))
    explanations: list[Explanation] = []
    for doc_index, rank, _score in rows:
        sources: list[SourceContribution] = []
        details: list[str] = []
        for name, ranks in zip(names, rank_maps):
            source_rank = ranks.get(doc_index)
            share = rrf_contribution(source_rank, retriever.rrf_k)
            sources.append(SourceContribution(name, source_rank, share))
            if source_rank is None:
                details.append(
                    f"{name} — outside top {retriever.candidate_k} — {format_score(share, 6)}"
                )
            else:
                details.append(f"{name} — rank {source_rank} — {format_score(share, 6)}")
        total = float(sum(source.contribution for source in sources))
        summary = (
            f"RRF score {format_score(total, 6)}. "
            f"Each list adds 1/({retriever.rrf_k} + rank), or 0 if the title is outside that list. "
            "Contributions add up to this score."
        )
        explanations.append(
            Explanation(
                kind="hybrid",
                summary=summary,
                details=tuple(details),
                score=total,
                rank=rank,
                sources=tuple(sources),
            )
        )
    return explanations


def rerank_summary(score: float, first_stage_rank: int | None, final_rank: int) -> str:
    score_txt = f"Cross-encoder score {format_score(score)}."
    if first_stage_rank is None:
        return f"{score_txt} This title was outside the first-stage candidate list."
    delta = first_stage_rank - final_rank
    if delta > 0:
        move = f"Moved up {delta} from first-stage rank {first_stage_rank} to rank {final_rank}."
    elif delta < 0:
        move = f"Moved down {-delta} from first-stage rank {first_stage_rank} to rank {final_rank}."
    else:
        move = f"Stayed at rank {final_rank} after the first stage."
    return f"{score_txt} {move}"


def _explain_rerank(
    retriever: CrossEncoderReranker, query: str, rows: list[tuple[int, int, float]]
) -> list[Explanation]:
    pool = min(retriever.candidate_k, len(retriever.catalog))
    base_idxs, _scores = retriever.base.rank_indices(query, top_k=pool)
    first_stage = _rank_map(base_idxs)
    explanations: list[Explanation] = []
    for doc_index, rank, score in rows:
        first = first_stage.get(doc_index)
        move = RerankMove(score, first, rank)
        explanations.append(
            Explanation(
                kind="rerank",
                summary=rerank_summary(score, first, rank),
                details=(),
                score=float(score),
                rank=rank,
                rerank=move,
            )
        )
    return explanations
