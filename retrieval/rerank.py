"""Second-stage cross-encoder reranking over a first-stage candidate list.

CPU-only by default (`cross-encoder/ms-marco-MiniLM-L-6-v2`). No paid API.
Takes the top-``candidate_k`` hits from a base retriever and reorders them
with pairwise (query, document) scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class CrossEncoderReranker:
    """Wrap any retriever that exposes ``rank_indices`` with a CE second stage."""

    catalog: pd.DataFrame
    base: Any
    text_field: str = "text"
    model_name: str = DEFAULT_CROSS_ENCODER
    candidate_k: int = 50
    name: str = "rerank"
    batch_size: int = 32
    show_progress: bool = False
    _model: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.text_field not in self.catalog.columns:
            raise ValueError(f"Unknown text_field: {self.text_field}")
        if self.candidate_k < 1:
            raise ValueError("candidate_k must be >= 1")

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            # device omitted → sentence-transformers picks CPU when CUDA unavailable
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rank_indices(self, text: str, top_k: int = 100) -> tuple[np.ndarray, np.ndarray]:
        pool_k = max(self.candidate_k, top_k)
        base_idxs, _ = self.base.rank_indices(text, top_k=pool_k)
        if len(base_idxs) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        # Cap at candidate_k for the expensive CE pass.
        cand = np.asarray(base_idxs[: self.candidate_k], dtype=int)
        docs = self.catalog.iloc[cand][self.text_field].tolist()
        pairs = [[text, doc] for doc in docs]
        model = self._get_model()
        scores = np.asarray(
            model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress,
            ),
            dtype=float,
        )
        order = np.argsort(-scores)[:top_k]
        return cand[order], scores[order]

    def query(self, text: str, top_k: int = 10) -> pd.DataFrame:
        idxs, scores = self.rank_indices(text, top_k=top_k)
        hits = self.catalog.iloc[idxs][["show_id", "title", "description", "type"]].copy()
        hits.insert(0, "rank", np.arange(1, len(hits) + 1))
        hits.insert(1, "score", scores)
        hits.insert(2, "method", self.name)
        return hits.reset_index(drop=True)
