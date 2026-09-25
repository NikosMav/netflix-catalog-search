#!/usr/bin/env python3
"""Enrich the Netflix catalog with English Wikipedia plot/premise text.

Matches each title using release year + type (film / TV series), extracts a
Plot/Premise/Overview section, and writes a committed JSONL artifact so
clone-and-run does not need Wikipedia access.

Wikipedia text is CC BY-SA — see data/WIKIPEDIA_ATTRIBUTION.md.

Examples:
  python scripts/enrich_wikipedia_plots.py
  python scripts/enrich_wikipedia_plots.py --gold-only
  python scripts/enrich_wikipedia_plots.py --report-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval.wikipedia_plot import (  # noqa: E402
    CandidatePage,
    MatchDecision,
    candidate_title_queries,
    choose_match,
    coverage_stats,
    extract_plot_from_wikitext,
    is_disambiguation_wikitext,
    media_type_from_catalog,
    truncate_plot,
)

DEFAULT_CSV = ROOT / "data" / "netflix_titles.csv"
DEFAULT_LABELS = ROOT / "data" / "labeled_queries.json"
DEFAULT_OUT = ROOT / "data" / "wikipedia_plots.jsonl"
DEFAULT_SUMMARY = ROOT / "data" / "wikipedia_plots_coverage.json"
DEFAULT_CACHE = ROOT / ".cache" / "wikipedia"
USER_AGENT = (
    "NetflixCatalogSearch/0.1 "
    "(https://github.com/NikosMav/netflix-catalog-search; "
    "offline IR research; contact via GitHub issues)"
)
API_URL = "https://en.wikipedia.org/w/api.php"
LICENSE = "CC BY-SA 4.0"
ATTRIBUTION = "Wikipedia (https://en.wikipedia.org/), license CC BY-SA 4.0"


class WikiClient:
    """Thin MediaWiki API client with on-disk JSON cache and rate limiting."""

    def __init__(
        self,
        cache_dir: Path,
        min_interval_s: float = 0.2,
        timeout_s: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._last_request = 0.0
        self.hits = 0
        self.misses = 0

    def _cache_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)

    def get(self, params: dict[str, Any], cache_key: str | None = None) -> dict[str, Any]:
        key = cache_key or urllib.parse.urlencode(sorted((k, str(v)) for k, v in params.items()))
        path = self._cache_path(key)
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))
        self.misses += 1
        last_error: Exception | None = None
        for attempt in range(6):
            self._throttle()
            try:
                resp = self.session.get(API_URL, params=params, timeout=self.timeout_s)
                self._last_request = time.monotonic()
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", 5 + attempt * 5))
                    time.sleep(retry_after)
                    last_error = requests.HTTPError(f"429 for {params.get('action')}")
                    continue
                resp.raise_for_status()
                data = resp.json()
                path.write_text(json.dumps(data), encoding="utf-8")
                return data
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                time.sleep(2 ** attempt)
        assert last_error is not None
        raise last_error

    def resolve_titles(self, titles: list[str]) -> dict[str, CandidatePage]:
        """Resolve titles (any length; chunked at 50) → CandidatePage."""
        out: dict[str, CandidatePage] = {}
        uniq: list[str] = []
        seen: set[str] = set()
        for t in titles:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        for i in range(0, len(uniq), 50):
            chunk = uniq[i : i + 50]
            out.update(self._resolve_chunk(chunk))
        return out

    def _resolve_chunk(self, titles: list[str]) -> dict[str, CandidatePage]:
        params = {
            "action": "query",
            "titles": "|".join(titles),
            "prop": "pageprops|info",
            "ppprop": "disambiguation|wikibase-shortdesc",
            "redirects": 1,
            "format": "json",
            "formatversion": 2,
        }
        data = self.get(params, cache_key=f"resolve_v2:{'|'.join(titles)}")
        query = data.get("query", {})
        redirects = {r["from"]: r["to"] for r in query.get("redirects", [])}
        normalized = {n["from"]: n["to"] for n in query.get("normalized", [])}
        pages_by_title = {p["title"]: p for p in query.get("pages", [])}

        def final_title(t: str) -> str:
            cur = normalized.get(t, t)
            for _ in range(3):
                if cur in redirects:
                    cur = redirects[cur]
                else:
                    break
            return cur

        out: dict[str, CandidatePage] = {}
        for t in titles:
            dest = final_title(t)
            page = pages_by_title.get(dest)
            if page is None:
                out[t] = CandidatePage(title=dest, is_missing=True)
                continue
            props = page.get("pageprops") or {}
            is_missing = bool(page.get("missing"))
            is_disambig = "disambiguation" in props
            short = props.get("wikibase-shortdesc") or ""
            out[t] = CandidatePage(
                title=page.get("title", dest),
                pageid=None if is_missing else page.get("pageid"),
                short_description=short,
                is_missing=is_missing,
                is_disambiguation=is_disambig,
            )
        return out

    def fetch_page_wikitext(self, title: str) -> str:
        params = {
            "action": "query",
            "titles": title,
            "prop": "revisions",
            "rvslots": "main",
            "rvprop": "content",
            "redirects": 1,
            "format": "json",
            "formatversion": 2,
        }
        data = self.get(params, cache_key=f"wikitext:{title}")
        pages = data.get("query", {}).get("pages") or []
        if not pages or pages[0].get("missing"):
            return ""
        revisions = pages[0].get("revisions") or []
        if not revisions:
            return ""
        slots = revisions[0].get("slots") or {}
        main = slots.get("main") or {}
        return str(main.get("content") or "")

    def fetch_sections(self, title: str) -> list[dict[str, Any]]:
        params = {
            "action": "parse",
            "page": title,
            "prop": "sections",
            "format": "json",
            "formatversion": 2,
        }
        data = self.get(params, cache_key=f"sections:{title}")
        if "error" in data:
            return []
        return list(data.get("parse", {}).get("sections") or [])

    def fetch_section_wikitext(self, title: str, section_index: str) -> str:
        params = {
            "action": "parse",
            "page": title,
            "prop": "wikitext",
            "section": section_index,
            "format": "json",
            "formatversion": 2,
        }
        data = self.get(params, cache_key=f"section:{title}:{section_index}")
        if "error" in data:
            return ""
        return str(data.get("parse", {}).get("wikitext") or "")

    def fetch_lead_wikitext(self, title: str) -> str:
        params = {
            "action": "parse",
            "page": title,
            "prop": "wikitext",
            "section": 0,
            "format": "json",
            "formatversion": 2,
        }
        data = self.get(params, cache_key=f"lead:{title}")
        if "error" in data:
            return ""
        return str(data.get("parse", {}).get("wikitext") or "")


def load_gold_ids(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    ids: set[str] = set()
    for q in data["queries"]:
        ids.update(str(s) for s in q["relevant_show_ids"])
    return ids


def record_from_decision(
    *,
    show_id: str,
    title: str,
    release_year: int | None,
    type_value: str,
    decision: MatchDecision,
    plot_section: str | None = None,
    plot_text: str = "",
    status_override: str | None = None,
) -> dict[str, Any]:
    status = status_override or decision.status
    return {
        "show_id": show_id,
        "title": title,
        "release_year": release_year,
        "type": type_value,
        "status": status,
        "confidence": decision.confidence if status == "matched" else "none",
        "wiki_title": decision.wiki_title,
        "wiki_pageid": decision.wiki_pageid,
        "wiki_url": decision.wiki_url,
        "plot_section": plot_section,
        "plot_text": plot_text,
        "match_reason": decision.reason,
        "candidates_considered": list(decision.candidates_considered),
        "license": LICENSE,
        "attribution": ATTRIBUTION,
    }


def row_meta(row: dict[str, Any]) -> tuple[str, str, str, int | None, list[str]]:
    title = str(row["title"])
    show_id = str(row["show_id"])
    type_value = str(row.get("type") or "")
    media = media_type_from_catalog(type_value)
    year_raw = row.get("release_year")
    try:
        year = int(year_raw) if pd.notna(year_raw) else None
    except (TypeError, ValueError):
        year = None
    queries = candidate_title_queries(title, year, media)
    return show_id, title, type_value, year, queries


def attach_plot(client: WikiClient, decision: MatchDecision) -> tuple[str | None, str, str | None]:
    """Return (plot_section, plot_text, status_override).

    One wikitext fetch per matched page; plot/premise extracted locally.
    """
    if decision.status != "matched" or not decision.wiki_title:
        return None, "", None
    wikitext = client.fetch_page_wikitext(decision.wiki_title)
    if not wikitext:
        return None, "", "no_plot"
    if is_disambiguation_wikitext(wikitext):
        return None, "", "ambiguous"
    section_name, plain = extract_plot_from_wikitext(wikitext)
    plain = truncate_plot(plain)
    if not section_name or not plain:
        return section_name, "", "no_plot"
    return section_name, plain, None


def enrich_rows(client: WikiClient, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enrich many rows with batched title resolution."""
    metas = [row_meta(r) for r in rows]
    all_queries: list[str] = []
    for *_, queries in metas:
        all_queries.extend(queries)
    resolved = client.resolve_titles(all_queries)

    # Disambiguation check for pages lacking short descriptions
    need_lead: list[str] = []
    for page in resolved.values():
        if (
            not page.is_missing
            and not page.is_disambiguation
            and page.pageid is not None
            and not page.short_description
        ):
            need_lead.append(page.title)
    disambig_titles: set[str] = set()
    for title in dict.fromkeys(need_lead):
        lead = client.fetch_lead_wikitext(title)
        if is_disambiguation_wikitext(lead):
            disambig_titles.add(title)

    records: list[dict[str, Any]] = []
    for row, (show_id, title, type_value, year, queries) in zip(rows, metas):
        media = media_type_from_catalog(type_value)
        candidates: list[tuple[str, CandidatePage]] = []
        for q in queries:
            page = resolved.get(q) or CandidatePage(title=q, is_missing=True)
            if page.title in disambig_titles:
                page = CandidatePage(
                    title=page.title,
                    pageid=page.pageid,
                    short_description=page.short_description,
                    is_missing=page.is_missing,
                    is_disambiguation=True,
                )
            candidates.append((q, page))
        decision = choose_match(
            catalog_title=title,
            release_year=year,
            media=media,
            candidates=candidates,
        )
        plot_section, plot_text, status_override = attach_plot(client, decision)
        records.append(
            record_from_decision(
                show_id=show_id,
                title=title,
                release_year=year,
                type_value=type_value,
                decision=decision,
                plot_section=plot_section,
                plot_text=plot_text,
                status_override=status_override,
            )
        )
    return records


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[str(rec["show_id"])] = rec
    return out


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_enrichment(args: argparse.Namespace) -> int:
    catalog = pd.read_csv(args.data)
    gold_ids = load_gold_ids(args.labels)
    if args.gold_only:
        catalog = catalog[catalog["show_id"].astype(str).isin(gold_ids)].copy()
    if args.limit:
        catalog = catalog.head(args.limit).copy()

    existing = {} if args.force else load_existing(args.out)
    client = WikiClient(args.cache_dir, min_interval_s=args.interval)

    pending_rows: list[dict[str, Any]] = []
    for row in catalog.to_dict(orient="records"):
        sid = str(row["show_id"])
        if sid in existing and not args.force:
            continue
        pending_rows.append(row)

    print(f"Enriching {len(pending_rows)} / {len(catalog)} rows (cached={len(catalog)-len(pending_rows)})")
    batch_size = max(1, args.batch_size)
    for i in range(0, len(pending_rows), batch_size):
        batch = pending_rows[i : i + batch_size]
        recs = enrich_rows(client, batch)
        for rec in recs:
            existing[str(rec["show_id"])] = rec
        write_jsonl(args.out, list(existing.values()))
        print(
            f"[{min(i + batch_size, len(pending_rows))}/{len(pending_rows)}] "
            f"cache_hits={client.hits} cache_misses={client.misses} "
            f"batch_statuses={[r['status'] for r in recs[:5]]}",
            flush=True,
        )

    final_records = list(existing.values())
    write_jsonl(args.out, final_records)
    stats = coverage_stats(final_records, gold_show_ids=gold_ids)
    args.summary.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Wrote {args.out} ({len(final_records)} rows)")
    print(f"Wrote {args.summary}")
    return 0


def run_report(args: argparse.Namespace) -> int:
    records = list(load_existing(args.out).values())
    if not records:
        print(f"No records in {args.out}", file=sys.stderr)
        return 1
    gold_ids = load_gold_ids(args.labels)
    stats = coverage_stats(records, gold_show_ids=gold_ids)
    args.summary.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_CSV)
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--interval", type=float, default=0.35, help="Min seconds between API calls")
    p.add_argument("--batch-size", type=int, default=25, help="Catalog rows per resolve batch")
    p.add_argument("--gold-only", action="store_true", help="Only enrich labeled relevant titles")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N catalog rows")
    p.add_argument("--force", action="store_true", help="Re-fetch even if show_id already in out file")
    p.add_argument("--report-only", action="store_true", help="Summarize existing JSONL; no network")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.report_only:
        return run_report(args)
    return run_enrichment(args)


if __name__ == "__main__":
    raise SystemExit(main())
