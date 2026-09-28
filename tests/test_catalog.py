"""Catalog text-field construction (no model download)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from retrieval.catalog import METADATA_FIELDS, build_text_meta, load_catalog


def test_load_catalog_text_fields():
    cat = load_catalog()
    assert {"text", "text_meta", "title_text", "text_plot", "plot"} <= set(cat.columns)
    assert len(cat) > 1000
    # Description-only text is title + description (no cast dump by default).
    row = cat.iloc[0]
    assert row["title"] in row["text"]
    assert row["description"] in row["text"]
    # Meta field is a strict enrichment of description-only text when metadata exists.
    assert row["text"] in row["text_meta"] or row["text_meta"].startswith(row["title"])
    # Plot field equals text when no Wikipedia plot is attached.
    if not row["plot"]:
        assert row["text_plot"] == row["text"]
    else:
        assert row["plot"] in row["text_plot"]
        assert row["text"] in row["text_plot"] or row["text_plot"].startswith(row["title"])


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


def test_build_text_plot_appends_plot():
    from retrieval.catalog import build_text_plot

    row = pd.Series({"title": "T", "description": "D", "plot": "Longer plot summary."})
    assert build_text_plot(row) == "T D Longer plot summary."
    row_empty = pd.Series({"title": "T", "description": "D", "plot": ""})
    assert build_text_plot(row_empty) == "T D"


def test_labeled_query_blob_unchanged():
    """Gold labels file must remain byte-identical to the committed blob."""
    import json
    import subprocess

    path = Path(__file__).resolve().parents[1] / "data" / "labeled_queries.json"
    digest = subprocess.check_output(["git", "hash-object", str(path)], text=True).strip()
    assert digest == "f70b57ef5f99e871915a94062b1f1e0c4dac0b51"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["queries"]) == 28


def test_v1_eval_metrics_blob_unchanged():
    """Legacy metrics file must remain byte-identical to the committed blob."""
    import subprocess

    path = Path(__file__).resolve().parents[1] / "results" / "eval_metrics.json"
    digest = subprocess.check_output(["git", "hash-object", str(path)], text=True).strip()
    assert digest == "3029fdcc055763a26d4f9c45a92c5d53dc05a075"
