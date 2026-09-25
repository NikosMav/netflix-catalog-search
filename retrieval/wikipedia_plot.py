"""Match Netflix catalog rows to English Wikipedia plot/premise text.

Pure matching and extraction helpers live here so unit tests never hit the
network. The fetch/cache CLI is ``scripts/enrich_wikipedia_plots.py``.

Wikipedia text is licensed under CC BY-SA; see ``data/WIKIPEDIA_ATTRIBUTION.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

# Section headings we accept as plot/premise (case-insensitive, HTML stripped).
PLOT_SECTION_NAMES = frozenset(
    {
        "plot",
        "plots",
        "synopsis",
        "premise",
        "overview",
        "summary",
        "story",
        "storyline",
        "plot summary",
        "series overview",
    }
)

# Hard stop sections when scanning plain wikitext without a section index.
STOP_SECTION_NAMES = frozenset(
    {
        "cast",
        "cast and characters",
        "characters",
        "production",
        "development",
        "filming",
        "release",
        "reception",
        "critical reception",
        "accades",
        "accolades",
        "awards",
        "soundtrack",
        "music",
        "episodes",
        "episode list",
        "see also",
        "references",
        "external links",
        "notes",
        "bibliography",
        "further reading",
        "themes",
        "legacy",
        "franchise",
        "adaptations",
        "home media",
        "marketing",
        "box office",
    }
)

DISAMBIGUATION_MARKERS = (
    "{{disambiguation",
    "{{disambig",
    "{{dab",
    "{{hndis",
)

YEAR_IN_TITLE = re.compile(r"\((\d{4})\s*([^)]*)\)\s*$")
YEAR_IN_DESC = re.compile(r"\b((?:19|20)\d{2})\b")
YEAR_RANGE = re.compile(r"\b((?:19|20)\d{2})\s*[–\-]\s*((?:19|20)\d{2}|present)\b", re.I)

# Strip common wikitext constructs when turning a section into plain text.
_RE_COMMENT = re.compile(r"<!--.*?-->", re.S)
_RE_REF = re.compile(r"<ref\b[^>]*>.*?</ref>", re.I | re.S)
_RE_REF_SINGLE = re.compile(r"<ref\b[^/]*/>", re.I)
_RE_HTML = re.compile(r"<[^>]+>")
_RE_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_RE_LINK = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
_RE_EXTERNAL = re.compile(r"\[https?://[^\]\s]+\s+([^\]]+)\]")
_RE_BOLDITAL = re.compile(r"'{2,}")
_RE_TABLE = re.compile(r"^\s*\{\|.*?\|\}\s*$", re.M | re.S)
_RE_FILE = re.compile(r"\[\[(?:File|Image):[^\]]+\]\]", re.I)
_RE_MULTI_WS = re.compile(r"[ \t]+\n")
_RE_BLANK = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class CandidatePage:
    """A Wikipedia page considered for a catalog row."""

    title: str
    pageid: int | None = None
    short_description: str = ""
    is_missing: bool = False
    is_disambiguation: bool = False
    wikitext: str = ""


@dataclass(frozen=True)
class MatchDecision:
    """Result of applying the confidence rule to one catalog row."""

    status: str  # matched | ambiguous | not_found | no_plot | skipped
    confidence: str  # high | medium | low | none
    wiki_title: str | None = None
    wiki_pageid: int | None = None
    wiki_url: str | None = None
    plot_section: str | None = None
    plot_text: str = ""
    reason: str = ""
    candidates_considered: tuple[str, ...] = ()


def normalize_title(title: str) -> str:
    """Light normalization for Wikipedia title lookup."""
    t = (title or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def media_type_from_catalog(type_value: str) -> str:
    """Map Netflix ``type`` to ``film`` or ``tv``."""
    t = (type_value or "").strip().lower()
    if t in {"movie", "film"}:
        return "film"
    if t in {"tv show", "tv", "series", "television"}:
        return "tv"
    return "unknown"


def candidate_title_queries(title: str, year: int | None, media: str) -> list[str]:
    """Ordered Wikipedia title candidates for exact/resolve lookup.

    Prefer year-qualified film/TV titles first so ambiguous bare names are not
    chosen when a precise page exists.
    """
    base = normalize_title(title)
    if not base:
        return []
    out: list[str] = []
    if media == "film":
        if year:
            out.append(f"{base} ({year} film)")
        out.append(f"{base} (film)")
    elif media == "tv":
        if year:
            out.append(f"{base} ({year} TV series)")
            out.append(f"{base} ({year} television series)")
            out.append(f"{base} ({year} miniseries)")
        out.append(f"{base} (TV series)")
        out.append(f"{base} (television series)")
        out.append(f"{base} (miniseries)")
        out.append(f"{base} (TV program)")
    # Bare title last — only accepted with corroborating short description.
    out.append(base)
    # Deduplicate preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for t in out:
        key = t.casefold()
        if key not in seen:
            seen.add(key)
            uniq.append(t)
    return uniq


def _title_year_and_kind(wiki_title: str) -> tuple[int | None, str | None]:
    m = YEAR_IN_TITLE.search(wiki_title or "")
    if not m:
        # Also catch "(TV series)" without year
        low = (wiki_title or "").lower()
        kind = None
        if "film" in low and low.rstrip().endswith(")"):
            kind = "film"
        elif any(k in low for k in ("tv series", "television series", "miniseries", "tv program")):
            kind = "tv"
        return None, kind
    year = int(m.group(1))
    kind_raw = (m.group(2) or "").strip().lower()
    if "film" in kind_raw or "movie" in kind_raw:
        kind = "film"
    elif any(k in kind_raw for k in ("tv", "television", "miniseries", "series")):
        kind = "tv"
    else:
        kind = None
    return year, kind


def _years_from_short_description(desc: str) -> list[int]:
    if not desc:
        return []
    years: list[int] = []
    for m in YEAR_RANGE.finditer(desc):
        years.append(int(m.group(1)))
    for m in YEAR_IN_DESC.finditer(desc):
        y = int(m.group(1))
        if y not in years:
            years.append(y)
    return years


def _kind_from_short_description(desc: str) -> str | None:
    d = (desc or "").lower()
    if not d:
        return None
    tv_markers = (
        "television series",
        "tv series",
        "miniseries",
        "web series",
        "television program",
        "tv program",
        "television show",
        "sitcom",
    )
    film_markers = ("film", "movie")
    if any(m in d for m in tv_markers):
        return "tv"
    if any(m in d for m in film_markers):
        return "film"
    return None


def year_compatible(release_year: int | None, page_years: Iterable[int], tol: int = 1) -> bool:
    """True if no year constraint, or any page year is within ``tol`` of release."""
    if release_year is None:
        return True
    page_years = list(page_years)
    if not page_years:
        return False
    return any(abs(int(y) - int(release_year)) <= tol for y in page_years)


def score_candidate(
    *,
    catalog_title: str,
    release_year: int | None,
    media: str,
    page: CandidatePage,
    queried_as: str,
) -> tuple[str, str]:
    """Return ``(confidence, reason)`` for one candidate page.

    Confidence levels:
      - high: year-qualified film/TV title match, or bare/alt title whose short
        description confirms both type and year
      - medium: type-qualified title (film / TV series) with year confirmed in
        short description, or year in title matching release year
      - low: weak corroboration — not accepted by the default rule
      - none: missing, disambiguation, wrong type/year
    """
    if page.is_missing or not page.title:
        return "none", "missing"
    if page.is_disambiguation:
        return "none", "disambiguation"

    title_year, title_kind = _title_year_and_kind(page.title)
    desc = page.short_description or ""
    desc_years = _years_from_short_description(desc)
    desc_kind = _kind_from_short_description(desc)
    kind = title_kind or desc_kind

    if media in {"film", "tv"} and kind and kind != media:
        return "none", f"type_mismatch:{kind}"

    years = []
    if title_year:
        years.append(title_year)
    years.extend(desc_years)

    # Year-qualified query that resolved cleanly.
    q_year, q_kind = _title_year_and_kind(queried_as)
    if q_year is not None and q_kind == media and year_compatible(release_year, [q_year], tol=0):
        if release_year is None or year_compatible(release_year, years or [q_year], tol=1):
            return "high", "year_qualified_title"

    if title_year is not None and year_compatible(release_year, [title_year], tol=0):
        if kind == media or (kind is None and media == "film" and "film" in page.title.lower()):
            return "high", "title_embeds_matching_year"

    if kind == media and year_compatible(release_year, desc_years, tol=1):
        # Type in title like "X (film)" plus year in short description.
        if title_kind == media or queried_as.rstrip().endswith(f"({media})") or (
            media == "tv"
            and any(
                queried_as.endswith(s)
                for s in ("(TV series)", "(television series)", "(miniseries)", "(TV program)")
            )
        ):
            return "medium", "type_title_plus_desc_year"
        # Bare title whose short description is clearly the right work.
        base = normalize_title(catalog_title).casefold()
        page_base = YEAR_IN_TITLE.sub("", page.title).strip().casefold()
        if page_base == base and desc_kind == media:
            return "high", "bare_title_desc_corroborated"
        if desc_kind == media:
            return "medium", "desc_type_and_year"

    if kind == media and release_year is None:
        return "low", "type_only_no_year"

    if kind == media and not desc_years and title_year is None:
        return "low", "type_without_year_corroboration"

    if years and not year_compatible(release_year, years, tol=1):
        return "none", "year_mismatch"

    return "none", "insufficient_evidence"


def choose_match(
    *,
    catalog_title: str,
    release_year: int | None,
    media: str,
    candidates: list[tuple[str, CandidatePage]],
) -> MatchDecision:
    """Apply the skip-if-ambiguous rule over scored candidates.

    Accept when exactly one candidate is ``high``, or exactly one is
    ``medium`` and none are ``high``. Multiple medium/high → ``ambiguous``.
    """
    considered = tuple(p.title for _, p in candidates if p.title and not p.is_missing)
    scored: list[tuple[str, str, str, CandidatePage]] = []
    for queried_as, page in candidates:
        conf, reason = score_candidate(
            catalog_title=catalog_title,
            release_year=release_year,
            media=media,
            page=page,
            queried_as=queried_as,
        )
        if conf in {"high", "medium"}:
            scored.append((conf, reason, queried_as, page))

    # Deduplicate by resolved page title
    by_title: dict[str, tuple[str, str, str, CandidatePage]] = {}
    rank = {"high": 2, "medium": 1}
    for conf, reason, queried_as, page in scored:
        key = page.title.casefold()
        prev = by_title.get(key)
        if prev is None or rank[conf] > rank[prev[0]]:
            by_title[key] = (conf, reason, queried_as, page)

    unique = list(by_title.values())
    highs = [x for x in unique if x[0] == "high"]
    mediums = [x for x in unique if x[0] == "medium"]

    if len(highs) > 1:
        return MatchDecision(
            status="ambiguous",
            confidence="none",
            reason="multiple_high_confidence",
            candidates_considered=considered,
        )
    if len(highs) == 1:
        conf, reason, _, page = highs[0]
        return MatchDecision(
            status="matched",
            confidence=conf,
            wiki_title=page.title,
            wiki_pageid=page.pageid,
            wiki_url=wiki_url_for(page.title),
            reason=reason,
            candidates_considered=considered,
        )
    if len(mediums) > 1:
        return MatchDecision(
            status="ambiguous",
            confidence="none",
            reason="multiple_medium_confidence",
            candidates_considered=considered,
        )
    if len(mediums) == 1:
        conf, reason, _, page = mediums[0]
        return MatchDecision(
            status="matched",
            confidence=conf,
            wiki_title=page.title,
            wiki_pageid=page.pageid,
            wiki_url=wiki_url_for(page.title),
            reason=reason,
            candidates_considered=considered,
        )
    if considered:
        return MatchDecision(
            status="not_found",
            confidence="none",
            reason="no_confident_match",
            candidates_considered=considered,
        )
    return MatchDecision(
        status="not_found",
        confidence="none",
        reason="no_candidates",
        candidates_considered=considered,
    )


def wiki_url_for(title: str) -> str:
    from urllib.parse import quote

    return f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"


def is_disambiguation_wikitext(wikitext: str) -> bool:
    low = (wikitext or "")[:4000].lower()
    return any(m in low for m in DISAMBIGUATION_MARKERS)


def find_plot_section(sections: list[dict[str, Any]]) -> tuple[str, str] | None:
    """Pick the best plot/premise section from a ``parse.sections`` list.

    Returns ``(section_index, section_line)`` or None.
    """
    level2: list[tuple[str, str]] = []
    for sec in sections:
        level = str(sec.get("level", ""))
        if level != "2":
            continue
        line = re.sub(r"<[^>]+>", "", str(sec.get("line", ""))).strip()
        index = str(sec.get("index", ""))
        level2.append((index, line))

    by_name = {line.casefold(): (index, line) for index, line in level2}
    for name in (
        "plot",
        "plot summary",
        "synopsis",
        "premise",
        "story",
        "storyline",
        "overview",
        "series overview",
        "summary",
        "plots",
    ):
        if name in by_name:
            return by_name[name]
    # Partial contains, e.g. "Plot synopsis"
    for index, line in level2:
        low = line.casefold()
        if any(n in low for n in PLOT_SECTION_NAMES):
            if not any(s in low for s in ("cast", "production", "reception")):
                return index, line
    return None


def strip_wikitext(wikitext: str) -> str:
    """Convert a wikitext section body to readable plain text."""
    text = wikitext or ""
    # Drop the leading == Heading == line
    text = re.sub(r"^\s*==+[^=].*?==+\s*", "", text, count=1)
    text = _RE_COMMENT.sub("", text)
    text = _RE_REF.sub("", text)
    text = _RE_REF_SINGLE.sub("", text)
    text = _RE_FILE.sub("", text)
    # Nested templates: iteratively peel simple ones
    for _ in range(8):
        new = _RE_TEMPLATE.sub("", text)
        if new == text:
            break
        text = new
    # Leftover unmatched braces from complex templates
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.S)
    text = re.sub(r"\{\{[^}]*$", "", text)
    text = _RE_LINK.sub(r"\1", text)
    text = _RE_EXTERNAL.sub(r"\1", text)
    text = _RE_HTML.sub("", text)
    text = _RE_BOLDITAL.sub("", text)
    text = text.replace("{{", " ").replace("}}", " ")
    text = text.replace("[[", " ").replace("]]", " ")
    text = _RE_MULTI_WS.sub("\n", text)
    text = _RE_BLANK.sub("\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_plot_from_wikitext(wikitext: str) -> tuple[str | None, str]:
    """Extract plot plain text from full-page wikitext when section API is unavailable.

    Returns ``(section_name, plain_text)``.
    """
    if not wikitext:
        return None, ""
    pattern = re.compile(r"^==\s*([^=]+?)\s*==\s*$", re.M)
    matches = list(pattern.finditer(wikitext))
    for i, m in enumerate(matches):
        name = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        low = name.casefold()
        if low not in PLOT_SECTION_NAMES and not any(n in low for n in PLOT_SECTION_NAMES):
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(wikitext)
        body = wikitext[start:end]
        # If next heading is still level-3 under plot, keep until next level-2 (already)
        plain = strip_wikitext(f"== {name} ==\n{body}")
        if plain:
            return name, plain
    return None, ""


def truncate_plot(text: str, max_chars: int = 4000) -> str:
    """Cap plot length for storage and embedding models."""
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # Prefer sentence boundary
    for sep in (". ", ".\n", "? ", "! "):
        pos = cut.rfind(sep)
        if pos >= max_chars // 2:
            return cut[: pos + 1].strip()
    return cut.rsplit(" ", 1)[0].strip() + "…"


def build_text_plot(title: str, description: str, plot: str) -> str:
    """Concatenation field for the plot ablation: title + description + plot."""
    parts = [str(title or "").strip(), str(description or "").strip(), str(plot or "").strip()]
    return " ".join(p for p in parts if p).strip()


def coverage_stats(records: list[dict[str, Any]], gold_show_ids: set[str] | None = None) -> dict[str, Any]:
    """Summarize match coverage, optionally restricted to gold show_ids."""

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        matched = [r for r in rows if r.get("status") == "matched" and (r.get("plot_text") or "").strip()]
        by_status: dict[str, int] = {}
        by_conf: dict[str, int] = {}
        for r in rows:
            by_status[r.get("status", "unknown")] = by_status.get(r.get("status", "unknown"), 0) + 1
            if r.get("status") == "matched":
                by_conf[r.get("confidence", "none")] = by_conf.get(r.get("confidence", "none"), 0) + 1
        return {
            "n": n,
            "with_plot": len(matched),
            "coverage": round(len(matched) / n, 4) if n else 0.0,
            "by_status": by_status,
            "matched_by_confidence": by_conf,
        }

    out: dict[str, Any] = {"catalog": summarize(records)}
    if gold_show_ids is not None:
        gold_rows = [r for r in records if str(r.get("show_id")) in gold_show_ids]
        # Include gold ids with no enrichment row as not_found
        present = {str(r.get("show_id")) for r in gold_rows}
        for sid in gold_show_ids - present:
            gold_rows.append({"show_id": sid, "status": "not_found", "plot_text": ""})
        out["gold"] = summarize(gold_rows)
        out["gold_show_ids"] = len(gold_show_ids)
    return out
