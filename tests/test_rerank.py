"""Cross-encoder reranker unit tests with a fake CE model (no download)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from retrieval.bm25 import BM25Retriever
from retrieval.rerank import CrossEncoderReranker


class _FakeCE:
    """Scores pairs by whether the doc contains 'vietnam' (case-insensitive)."""

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        return np.array(
            [10.0 if "vietnam" in doc.lower() else 0.0 for _, doc in pairs],
            dtype=float,
        )


def _catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "show_id": ["s1", "s2", "s3"],
            "title": ["A", "B", "C"],
            "description": ["unrelated cooking", "vietnam war story", "space"],
            "type": ["Movie", "Movie", "Movie"],
            "text": [
                "A unrelated cooking",
                "B vietnam war story",
                "C space",
            ],
            "text_meta": [
                "A unrelated cooking Reality",
                "B vietnam war story Documentaries",
                "C space Sci-Fi",
            ],
        }
    )


def test_reranker_reorders_with_cross_encoder():
    cat = _catalog()
    base = BM25Retriever(cat, text_field="text")
    rerank = CrossEncoderReranker(
        cat,
        base=base,
        text_field="text_meta",
        candidate_k=3,
        name="dense+rerank",
    )
    rerank._model = _FakeCE()  # bypass HuggingFace download
    hits = rerank.query("war history", top_k=2)
    assert hits.iloc[0]["show_id"] == "s2"
    assert hits.iloc[0]["method"] == "dense+rerank"
