"""BM25 Okapi sparse retrieval (proper lexical baseline vs Boolean Jaccard)."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from retrieval.validation import bounded_top_k

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens — shared by index and query."""
    return _TOKEN_RE.findall((text or "").lower())


@dataclass
class BM25Retriever:
    catalog: pd.DataFrame
    text_field: str = "text"
    # BM25Okapi defaults; exposed for experiments without changing the package API.
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        if self.text_field not in self.catalog.columns:
            raise ValueError(f"Unknown text_field: {self.text_field}")
        corpus = [tokenize(t) for t in self.catalog[self.text_field].tolist()]
        self.bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)
        self._n_docs = len(corpus)

    def query(self, text: str, top_k: int = 10) -> pd.DataFrame:
        order, scores = self.rank_indices(text, top_k=top_k)
        hits = self.catalog.iloc[order][["show_id", "title", "description", "type"]].copy()
        hits.insert(0, "rank", np.arange(1, len(hits) + 1))
        hits.insert(1, "score", scores)
        hits.insert(2, "method", f"bm25:{self.text_field}")
        return hits.reset_index(drop=True)

    def rank_indices(self, text: str, top_k: int = 100) -> tuple[np.ndarray, np.ndarray]:
        k = bounded_top_k(top_k, self._n_docs)
        tokens = tokenize(text)
        if not tokens:
            # Empty query → empty ranking (caller gets no hits).
            return np.array([], dtype=int), np.array([], dtype=float)
        scores = np.asarray(self.bm25.get_scores(tokens), dtype=float)
        # argpartition then sort the top slice — O(n) + O(k log k) vs full argsort
        if k < self._n_docs:
            part = np.argpartition(-scores, k)[:k]
            order = part[np.argsort(-scores[part])]
        else:
            order = np.argsort(-scores)[:k]
        return order, scores[order]
