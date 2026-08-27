"""Retrieval boundary and CLI validation tests."""

from __future__ import annotations

import pytest

from retrieval.cli import build_parser
from retrieval.hybrid import reciprocal_rank_fusion
from retrieval.validation import bounded_top_k


def test_bounded_top_k_caps_to_collection_size():
    assert bounded_top_k(10, 3) == 3
    assert bounded_top_k(2, 3) == 2


@pytest.mark.parametrize("top_k", [0, -1])
def test_bounded_top_k_rejects_non_positive_values(top_k):
    with pytest.raises(ValueError, match="top_k"):
        bounded_top_k(top_k, 3)


def test_rrf_rejects_non_positive_top_k():
    with pytest.raises(ValueError, match="top_k"):
        reciprocal_rank_fusion([], top_k=0)


@pytest.mark.parametrize(
    "argv",
    [
        ["query", "war", "--top-k", "0"],
        ["query", "   "],
        ["eval", "--ks", "5", "0"],
    ],
)
def test_cli_rejects_invalid_query_arguments(argv):
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)
