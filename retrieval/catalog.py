"""Load the Netflix catalog and build document text fields for retrieval."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "data" / "netflix_titles.csv"
DEFAULT_WIKI_PLOTS = REPO_ROOT / "data" / "wikipedia_plots.jsonl"

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


def build_text_plot(row: pd.Series) -> str:
    """Title + description + Wikipedia plot (empty plot → same as ``text``)."""
    parts = [row["title"], row["description"]]
    plot = str(row.get("plot", "") or "").strip()
    if plot:
        parts.append(plot)
    return " ".join(parts).strip()


def load_wikipedia_plots(path: str | Path | None = None) -> pd.DataFrame:
    """Load committed Wikipedia plot enrichment (CC BY-SA; may be sparse).

    Returns an empty frame with the expected columns when the file is absent so
    notebooks and tests still load the base catalog.
    """
    p = Path(path) if path else DEFAULT_WIKI_PLOTS
    cols = ["show_id", "plot", "wiki_title", "wiki_url", "plot_status", "plot_confidence"]
    if not p.exists():
        return pd.DataFrame(columns=cols)

    rows: list[dict] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            plot = str(rec.get("plot_text") or "").strip()
            status = str(rec.get("status") or "")
            # Only attach plot text for confident matches that yielded a section.
            if status != "matched" or not plot:
                plot = ""
            rows.append(
                {
                    "show_id": str(rec.get("show_id", "")),
                    "plot": plot,
                    "wiki_title": rec.get("wiki_title") or "",
                    "wiki_url": rec.get("wiki_url") or "",
                    "plot_status": status,
                    "plot_confidence": rec.get("confidence") or "none",
                }
            )
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    # First row wins if duplicates ever appear.
    return out.drop_duplicates(subset=["show_id"], keep="first").reset_index(drop=True)


def load_catalog(
    csv_path: str | Path | None = None,
    wiki_plots_path: str | Path | None = None,
) -> pd.DataFrame:
    """Return a working frame with retrieval text fields.

    Fields:
      - ``title_text``: title only (dense ablation)
      - ``text``: title + description (description-only baseline; keeps old metrics comparable)
      - ``text_meta``: title + description + listed_in + cast + director + country
      - ``text_plot``: title + description + Wikipedia plot (plot ablation; empty if unmatched)
      - ``plot``: raw plot text when matched (else empty)

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

    plots = load_wikipedia_plots(wiki_plots_path)
    if len(plots):
        out = out.merge(plots, on="show_id", how="left")
    else:
        out["plot"] = ""
        out["wiki_title"] = ""
        out["wiki_url"] = ""
        out["plot_status"] = ""
        out["plot_confidence"] = ""
    for col in ("plot", "wiki_title", "wiki_url", "plot_status", "plot_confidence"):
        out[col] = _fill_str(out[col])

    out["title_text"] = out["title"]
    out["text"] = (out["title"] + " " + out["description"]).str.strip()
    out["text_meta"] = out.apply(build_text_meta, axis=1)
    out["text_plot"] = out.apply(build_text_plot, axis=1)
    return out.reset_index(drop=True)


def show_id_to_index(catalog: pd.DataFrame) -> dict[str, int]:
    """Map show_id → row index (first occurrence if duplicates)."""
    mapping: dict[str, int] = {}
    for i, sid in enumerate(catalog["show_id"].tolist()):
        mapping.setdefault(sid, i)
    return mapping
