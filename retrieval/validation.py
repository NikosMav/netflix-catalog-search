"""Shared validation helpers for retrieval APIs and the CLI."""

from __future__ import annotations


def bounded_top_k(top_k: int, n_items: int) -> int:
    """Validate ``top_k`` and cap it at the available collection size."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if n_items < 0:
        raise ValueError("n_items must be >= 0")
    return min(top_k, n_items)
