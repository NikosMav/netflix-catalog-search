"""Boolean / bag-of-words retrieval with set-Jaccard scoring.

Binary term presence (uni+bigrams), scored with Jaccard against the query
vector. Efficient sparse implementation — no dense Hamming matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def _jaccard_scores(query_vec, doc_matrix) -> np.ndarray:
    """Set Jaccard between a binary query row and each binary document row."""
    q = query_vec.copy()
    q.data = np.ones_like(q.data)
    X = doc_matrix.copy()
    X.data = np.ones_like(X.data)

    intersection = np.asarray(X.dot(q.T).todense()).ravel().astype(float)
    q_sum = float(q.sum())
    doc_sums = np.asarray(X.sum(axis=1)).ravel().astype(float)
    union = doc_sums + q_sum - intersection
    return intersection / np.maximum(union, 1e-12)


@dataclass
class BooleanRetriever:
    catalog: pd.DataFrame
    text_field: str = "text"
    max_features: int = 10_000
    ngram_range: tuple[int, int] = (1, 2)
    max_df: float = 0.4
    min_df: int = 1

    def __post_init__(self) -> None:
        if self.text_field not in self.catalog.columns:
            raise ValueError(f"Unknown text_field: {self.text_field}")
        self.vectorizer = CountVectorizer(
            binary=True,
            ngram_range=self.ngram_range,
            stop_words="english",
            max_df=self.max_df,
            min_df=self.min_df,
            max_features=self.max_features,
            lowercase=True,
        )
        self.doc_matrix = self.vectorizer.fit_transform(self.catalog[self.text_field])

    def query(self, text: str, top_k: int = 10) -> pd.DataFrame:
        q = self.vectorizer.transform([text])
        scores = _jaccard_scores(q, self.doc_matrix)
        order = np.argsort(-scores)[:top_k]
        hits = self.catalog.iloc[order][["show_id", "title", "description", "type"]].copy()
        hits.insert(0, "rank", np.arange(1, len(hits) + 1))
        hits.insert(1, "score", scores[order])
        hits.insert(2, "method", "boolean")
        return hits.reset_index(drop=True)

    def rank_indices(self, text: str, top_k: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, scores) for the top_k documents — used by eval/hybrid."""
        q = self.vectorizer.transform([text])
        scores = _jaccard_scores(q, self.doc_matrix)
        order = np.argsort(-scores)[:top_k]
        return order, scores[order]
