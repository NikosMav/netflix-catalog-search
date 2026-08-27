"""Load the Netflix catalog and build document text fields for retrieval."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "data" / "netflix_titles.csv"

# Catalog columns concatenated into the richer indexed text (already in CSV).
METADATA_FIELDS = ("listed_in", "cast", "director", "country")


def _fill_str(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def build_text_meta(row: pd.Series) -> str:
    """Title + description + listed_in / cast / director / country."""
    parts = [row["title"], row["description"]]
    for col in METADATA_FIELDS:
        val = str(row.get(col, "") or "").strip()
        if val:
            parts.append(val)
    return " ".join(parts).strip()


def load_catalog(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Return a working frame with retrieval text fields.

    Fields:
      - ``title_text``: title only (dense ablation)
      - ``text``: title + description (description-only baseline; keeps old metrics comparable)
      - ``text_meta``: title + description + listed_in + cast + director + country

    Missing string cells become empty strings. Row order matches the CSV so
    integer positions stay stable across retrievers.
    """
    path = Path(csv_path) if csv_path else DEFAULT_CSV
    df = pd.read_csv(path)
    required = {"show_id", "title", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    out = df.copy()
    out["show_id"] = out["show_id"].astype(str)
    out["title"] = _fill_str(out["title"])
    out["description"] = _fill_str(out["description"])
    for col in METADATA_FIELDS:
        if col not in out.columns:
            out[col] = ""
        else:
            out[col] = _fill_str(out[col])

    out["title_text"] = out["title"]
    out["text"] = (out["title"] + " " + out["description"]).str.strip()
    out["text_meta"] = out.apply(build_text_meta, axis=1)
    return out.reset_index(drop=True)


def show_id_to_index(catalog: pd.DataFrame) -> dict[str, int]:
    """Map show_id → row index (first occurrence if duplicates)."""
    mapping: dict[str, int] = {}
    for i, sid in enumerate(catalog["show_id"].tolist()):
        mapping.setdefault(sid, i)
    return mapping
