# Wikipedia plot enrichment — attribution & license

Plot and premise text in [`wikipedia_plots.jsonl`](wikipedia_plots.jsonl) is
extracted from **English Wikipedia** via the public MediaWiki API.

## License

Wikipedia text is available under the
[Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

That is a **different** license from this repository’s MIT code license
([`LICENSE.md`](../LICENSE.md)). Reuse or redistribution of the plot text must
comply with CC BY-SA (attribution + share-alike).

## Attribution

For each matched row, `wikipedia_plots.jsonl` records:

| Field | Meaning |
| --- | --- |
| `wiki_title` | Resolved English Wikipedia article title |
| `wiki_url` | Canonical article URL |
| `wiki_pageid` | MediaWiki page id when available |
| `plot_section` | Section heading used (Plot, Premise, Overview, …) |
| `license` | `CC BY-SA 4.0` |
| `attribution` | Short credit string |

Credit form used in the JSONL:

> Wikipedia (https://en.wikipedia.org/), license CC BY-SA 4.0

Article-level credit should name the specific page (`wiki_title` / `wiki_url`).

## Why this file is committed

The enrichment script (`scripts/enrich_wikipedia_plots.py`) is reproducible and
caches API responses under `.cache/wikipedia/` (gitignored). The **JSONL is
committed** so `pip install` + `python -m retrieval eval` works offline without
re-hitting Wikipedia, and so evaluation numbers stay tied to a fixed text
snapshot.

Regenerate (network required):

```bash
python scripts/enrich_wikipedia_plots.py
python scripts/enrich_wikipedia_plots.py --report-only
```

## Match rule (skip if unsure)

Matching uses Netflix `title` + `release_year` + `type` (Movie → film, TV Show →
TV series). Ambiguous or weakly corroborated pages are stored as `ambiguous` /
`not_found` / `no_plot` **without** plot text — we do not guess.
