"""Load the Netflix catalog and build document text fields for retrieval."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "data" / "netflix_titles.csv"


def load_catalog(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Return a working frame with `text` (title+description) and `title_text`.

    Missing titles/descriptions become empty strings. Row order matches the CSV
    so integer positions stay stable across retrievers.
    """
    path = Path(csv_path) if csv_path else DEFAULT_CSV
    df = pd.read_csv(path)
    required = {"show_id", "title", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    out = df.copy()
    out["show_id"] = out["show_id"].astype(str)
    out["title"] = out["title"].fillna("").astype(str)
    out["description"] = out["description"].fillna("").astype(str)
    out["title_text"] = out["title"].str.strip()
    out["text"] = (out["title"] + " " + out["description"]).str.strip()
    return out.reset_index(drop=True)


def show_id_to_index(catalog: pd.DataFrame) -> dict[str, int]:
    """Map show_id → row index (first occurrence if duplicates)."""
    mapping: dict[str, int] = {}
    for i, sid in enumerate(catalog["show_id"].tolist()):
        mapping.setdefault(sid, i)
    return mapping
