"""Hybrid retrieval via Reciprocal Rank Fusion (RRF).

Combines sparse (TF-IDF) and dense rank lists without calibrating raw scores:
    score(d) = Σ_r 1 / (rrf_k + rank_r(d))

Default fuses TF-IDF + dense(title+description). Boolean can be included too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from retrieval.validation import bounded_top_k


class Ranker(Protocol):
    def rank_indices(self, text: str, top_k: int = 100) -> tuple[np.ndarray, np.ndarray]: ...


def reciprocal_rank_fusion(
    ranked_lists: list[np.ndarray],
    rrf_k: int = 60,
    top_k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse multiple ranked index lists with RRF. Returns (indices, scores)."""
    scores: dict[int, float] = {}
    for ranking in ranked_lists:
        for rank, idx in enumerate(ranking, start=1):
            scores[int(idx)] = scores.get(int(idx), 0.0) + 1.0 / (rrf_k + rank)
    k = bounded_top_k(top_k, len(scores))
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    if not ordered:
        return np.array([], dtype=int), np.array([], dtype=float)
    idxs = np.array([i for i, _ in ordered], dtype=int)
    vals = np.array([s for _, s in ordered], dtype=float)
    return idxs, vals


@dataclass
class HybridRetriever:
    catalog: pd.DataFrame
    retrievers: list[Any]
    rrf_k: int = 60
    candidate_k: int = 100
    name: str = "hybrid"

    def __post_init__(self) -> None:
        if not self.retrievers:
            raise ValueError("retrievers must not be empty")
        if self.rrf_k < 0:
            raise ValueError("rrf_k must be >= 0")
        if self.candidate_k < 1:
            raise ValueError("candidate_k must be >= 1")

    def query(self, text: str, top_k: int = 10) -> pd.DataFrame:
        idxs, scores = self.rank_indices(text, top_k=top_k)
        hits = self.catalog.iloc[idxs][["show_id", "title", "description", "type"]].copy()
        hits.insert(0, "rank", np.arange(1, len(hits) + 1))
        hits.insert(1, "score", scores)
        hits.insert(2, "method", self.name)
        return hits.reset_index(drop=True)

    def rank_indices(self, text: str, top_k: int = 100) -> tuple[np.ndarray, np.ndarray]:
        lists = []
        for r in self.retrievers:
            idxs, _ = r.rank_indices(text, top_k=self.candidate_k)
            lists.append(idxs)
        return reciprocal_rank_fusion(lists, rrf_k=self.rrf_k, top_k=top_k)
