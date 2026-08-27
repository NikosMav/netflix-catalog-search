"""Dense embedding retrieval with a local sentence-transformers model.

Default path is CPU-friendly (`all-MiniLM-L6-v2`). Optional OpenAI backend when
`OPENAI_API_KEY` is set — see `backend="openai"`.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from retrieval.validation import bounded_top_k

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "embeddings"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"


def _cache_key(texts: list[str], model_name: str, text_field: str) -> str:
    h = hashlib.sha256()
    h.update(f"{model_name}|{text_field}".encode())
    h.update(b"\0")
    h.update(str(len(texts)).encode())
    h.update("\n".join(texts).encode("utf-8"))
    return h.hexdigest()[:32]


@dataclass
class DenseRetriever:
    catalog: pd.DataFrame
    text_field: str = "text"
    model_name: str = DEFAULT_MODEL
    backend: str = "sentence-transformers"
    cache_dir: Path = field(default_factory=lambda: DEFAULT_CACHE_DIR)
    batch_size: int = 64
    show_progress: bool = False

    def __post_init__(self) -> None:
        if self.backend not in {"sentence-transformers", "openai"}:
            raise ValueError("backend must be 'sentence-transformers' or 'openai'")
        if self.text_field not in self.catalog.columns:
            raise ValueError(f"Unknown text_field: {self.text_field}")
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        texts = self.catalog[self.text_field].tolist()
        self.embeddings = self._load_or_encode(texts)
        self._nn = NearestNeighbors(metric="cosine", algorithm="brute")
        self._nn.fit(self.embeddings)

    def _get_st_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self.backend == "sentence-transformers":
            model = self._get_st_model()
            emb = model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return np.asarray(emb, dtype=np.float32)

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "backend='openai' requires OPENAI_API_KEY. "
                "Default path uses local sentence-transformers (no paid API)."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install openai for backend='openai': pip install openai"
            ) from exc

        client = OpenAI(api_key=api_key)
        model = self.model_name if self.model_name != DEFAULT_MODEL else DEFAULT_OPENAI_MODEL
        vectors: list[list[float]] = []
        chunk = 128
        for i in range(0, len(texts), chunk):
            batch = texts[i : i + chunk]
            resp = client.embeddings.create(model=model, input=batch)
            ordered = sorted(resp.data, key=lambda d: d.index)
            vectors.extend([d.embedding for d in ordered])
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return arr / norms

    def _load_or_encode(self, texts: list[str]) -> np.ndarray:
        key = _cache_key(texts, f"{self.backend}:{self.model_name}", self.text_field)
        path = self.cache_dir / f"{key}.npy"
        if path.exists():
            return np.load(path)
        emb = self._encode(texts)
        np.save(path, emb)
        return emb

    def query(self, text: str, top_k: int = 10) -> pd.DataFrame:
        idxs, sims = self.rank_indices(text, top_k=top_k)
        hits = self.catalog.iloc[idxs][["show_id", "title", "description", "type"]].copy()
        hits.insert(0, "rank", np.arange(1, len(hits) + 1))
        hits.insert(1, "score", sims)
        hits.insert(2, "method", f"dense:{self.text_field}")
        return hits.reset_index(drop=True)

    def rank_indices(self, text: str, top_k: int = 100) -> tuple[np.ndarray, np.ndarray]:
        k = bounded_top_k(top_k, len(self.catalog))
        if k == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        q = self._encode([text])
        distances, indices = self._nn.kneighbors(q, n_neighbors=k)
        sims = 1.0 - distances.ravel()
        return indices.ravel(), sims
