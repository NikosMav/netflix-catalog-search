"""Catalog text-field construction (no model download)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from retrieval.catalog import METADATA_FIELDS, build_text_meta, load_catalog


def test_load_catalog_text_fields():
    cat = load_catalog()
    assert {"text", "text_meta", "title_text"} <= set(cat.columns)
    assert len(cat) > 1000
    # Description-only text is title + description (no cast dump by default).
    row = cat.iloc[0]
    assert row["title"] in row["text"]
    assert row["description"] in row["text"]
    # Meta field is a strict enrichment of description-only text when metadata exists.
    assert row["text"] in row["text_meta"] or row["text_meta"].startswith(row["title"])


def test_text_meta_includes_unused_columns():
    cat = load_catalog()
    # Find a row that has cast + listed_in populated.
    mask = (cat["cast"].str.len() > 0) & (cat["listed_in"].str.len() > 0)
    row = cat.loc[mask].iloc[0]
    for col in ("listed_in", "cast"):
        assert row[col] in row["text_meta"]
        assert row[col] not in row["text"] or row[col] in row["description"]


def test_build_text_meta_skips_empty():
    row = pd.Series(
        {
            "title": "T",
            "description": "D",
            "listed_in": "Dramas",
            "cast": "",
            "director": "",
            "country": "United States",
        }
    )
    text = build_text_meta(row)
    assert text == "T D Dramas United States"
    assert METADATA_FIELDS == ("listed_in", "cast", "director", "country")


def test_labeled_query_count_unchanged():
    """Gold set size must stay 28 so metrics remain comparable to main."""
    import json

    path = Path(__file__).resolve().parents[1] / "data" / "labeled_queries.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["queries"]) == 28
