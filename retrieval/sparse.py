"""TF-IDF sparse retrieval over catalog text fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SparseTfidfRetriever:
    catalog: pd.DataFrame
    text_field: str = "text"
    max_features: int = 10_000
    ngram_range: tuple[int, int] = (1, 2)
    max_df: float = 0.4
    min_df: int = 1

    def __post_init__(self) -> None:
        if self.text_field not in self.catalog.columns:
            raise ValueError(f"Unknown text_field: {self.text_field}")
        self.vectorizer = TfidfVectorizer(
            use_idf=True,
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            max_df=self.max_df,
            min_df=self.min_df,
            smooth_idf=True,
            lowercase=True,
            stop_words="english",
        )
        self.doc_matrix = self.vectorizer.fit_transform(self.catalog[self.text_field])

    def query(self, text: str, top_k: int = 10) -> pd.DataFrame:
        q = self.vectorizer.transform([text])
        scores = cosine_similarity(q, self.doc_matrix).ravel()
        order = np.argsort(-scores)[:top_k]
        hits = self.catalog.iloc[order][["show_id", "title", "description", "type"]].copy()
        hits.insert(0, "rank", np.arange(1, len(hits) + 1))
        hits.insert(1, "score", scores[order])
        hits.insert(2, "method", "tf-idf")
        return hits.reset_index(drop=True)

    def rank_indices(self, text: str, top_k: int = 100) -> tuple[np.ndarray, np.ndarray]:
        q = self.vectorizer.transform([text])
        scores = cosine_similarity(q, self.doc_matrix).ravel()
        order = np.argsort(-scores)[:top_k]
        return order, scores[order]
