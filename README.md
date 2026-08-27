# Netflix catalog text search

**Author:** Nikolaos (Nikos) Mavrapidis ([NikosMav](https://github.com/NikosMav))

## 5-minute walk

**What it is:** offline **catalog text search** over a public Netflix titles dump — Boolean, TF-IDF, BM25, dense MiniLM, hybrid RRF, and a CPU cross-encoder rerank, with metrics on **28 author-labeled** queries.

**What it isn’t:** a recommender, collaborative filtering, or RAG-over-the-web. No invented metrics; no production IR claims. Demo models only (MiniLM + ms-marco MiniLM CE). **n=28, author labels — no confidence intervals.**

| Command | What you get |
|---------|----------------|
| `python -m retrieval query "…" --method bm25` | Lexical catalog search |
| `python -m retrieval query "…" --method hybrid-rerank` | Strong first-stage hybrid + CE rerank |
| `python -m retrieval eval --failures` | Regenerates [`results/eval_metrics.json`](results/eval_metrics.json) |
| `python -m retrieval index` | Precomputes dense embedding caches |

Full method notes and limits: [`RETRIEVAL.md`](RETRIEVAL.md). Original EDA + Boolean/TF-IDF chapter stays in [`netflix_data_analysis.ipynb`](netflix_data_analysis.ipynb).

### Clone and run

```bash
git clone https://github.com/NikosMav/DataAnalysis-Netflix.git
cd DataAnalysis-Netflix

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m retrieval query "war between vietnam and usa" --method bm25 --top-k 10
python -m retrieval query "feel-good cooking competition show" --method dense-rerank
python -m retrieval query "dark crime thriller set in Scandinavia" --method hybrid-rerank
python -m retrieval eval --failures   # regenerates results/*.json from the gold labels
```

First dense/rerank run downloads MiniLM + ms-marco CE (~80MB each) and embeds ~7.8k rows; then caches under `.cache/`.

### Headline metrics

Source of truth: committed [`results/eval_metrics.json`](results/eval_metrics.json) (same 28 queries). Do not hand-edit this table — regenerate with `python -m retrieval eval`, then optionally `python scripts/sync_metrics_docs.py`.

`hybrid+rerank` = cross-encoder over **`hybrid(bm25+dense,meta)`** (not the weaker TF-IDF hybrid).

<!-- METRICS_TABLE_BEGIN -->
| method | recall@5 | recall@10 | ndcg@5 | ndcg@10 | mrr |
| --- | --- | --- | --- | --- | --- |
| boolean | 0.3159 | 0.4012 | 0.3185 | 0.3527 | 0.4440 |
| tf-idf | 0.4502 | 0.5446 | 0.4555 | 0.4912 | 0.5013 |
| tf-idf(desc+meta) | 0.4192 | 0.5446 | 0.3875 | 0.4374 | 0.4659 |
| bm25 | 0.5248 | 0.5645 | 0.5167 | 0.5253 | 0.5637 |
| bm25(desc+meta) | 0.5020 | 0.5524 | 0.5233 | 0.5353 | 0.6002 |
| dense(title+desc) | 0.5849 | 0.6209 | 0.5710 | 0.5656 | 0.6304 |
| dense(title+desc+meta) | 0.5735 | 0.6856 | 0.5825 | 0.6193 | 0.6786 |
| dense(title-only) | 0.4241 | 0.4499 | 0.4286 | 0.4269 | 0.5081 |
| hybrid(tfidf+dense) | 0.5059 | 0.6922 | 0.4941 | 0.5626 | 0.5853 |
| hybrid(bm25+dense,meta) | 0.6552 | 0.7421 | 0.6351 | 0.6605 | 0.7065 |
| dense+rerank | 0.6167 | 0.7062 | 0.6179 | 0.6469 | 0.6930 |
| hybrid+rerank | 0.7001 | 0.7817 | 0.6958 | 0.7185 | 0.7583 |
<!-- METRICS_TABLE_END -->

**Headline (from JSON):** hybrid+rerank R@5 **0.7001** / R@10 **0.7817** / MRR **0.7583** vs dense(title+desc) **0.5849** / **0.6209** / **0.6304**.

## Two tracks

1. **EDA + Boolean / TF-IDF** — original case study ([`netflix_data_analysis.ipynb`](netflix_data_analysis.ipynb))
2. **Catalog retrieval product** — package + CLI ([`retrieval/`](retrieval/), [`RETRIEVAL.md`](RETRIEVAL.md))

## What’s in the box

| Path | Purpose |
|------|---------|
| [`netflix_data_analysis.ipynb`](netflix_data_analysis.ipynb) | EDA + Boolean/TF-IDF case study (kept intact) |
| [`retrieval/`](retrieval/) | Package + CLI (`query`, `eval`, `index`) |
| [`data/labeled_queries.json`](data/labeled_queries.json) | 28 author-labeled queries (`relevant_show_ids`) |
| [`results/eval_metrics.json`](results/eval_metrics.json) | Committed metric table (source of truth) |
| [`RETRIEVAL.md`](RETRIEVAL.md) | Full case study: method, results, limits |
| [`netflix_dense_retrieval.ipynb`](netflix_dense_retrieval.ipynb) | Thin package walkthrough (not a second metrics source) |
| [`.github/workflows/unit-tests.yml`](.github/workflows/unit-tests.yml) | CI: `pytest tests/` only (no full catalog eval) |

## Data

Under [`data/`](data/): Netflix titles dump, slim IMDb join for the EDA chart, and retrieval labels. Provenance in [`data/README.md`](data/README.md).

## Limitations

- **No invented metrics.** Tables come from committed JSON / `python -m retrieval eval`.
- **28 queries, author-labeled.** Honest demo, not a public IR leaderboard; no CIs.
- **Catalog search ≠ recommender ≠ web RAG.**
- **CPU MiniLM + ms-marco CE are demo models.**

## License

MIT — see [LICENSE.md](LICENSE.md). Netflix catalog © Netflix (public dump via Kaggle). IMDb data subject to [IMDb non-commercial terms](https://developer.imdb.com/non-commercial-datasets/).
