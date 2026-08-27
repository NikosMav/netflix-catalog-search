"""Netflix catalog retrieval: sparse, dense, and hybrid text search."""

from retrieval.boolean_retriever import BooleanRetriever
from retrieval.catalog import load_catalog
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.sparse import SparseTfidfRetriever

__all__ = [
    "load_catalog",
    "BooleanRetriever",
    "SparseTfidfRetriever",
    "DenseRetriever",
    "HybridRetriever",
]
