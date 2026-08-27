# Netflix catalog: EDA → sparse retrieval → dense / hybrid / rerank

**Author:** Nikolaos (Nikos) Mavrapidis ([NikosMav](https://github.com/NikosMav))

Two showcase tracks in one repo:

1. **EDA + Boolean / TF-IDF** — original 2023 case study, cleaned and runnable ([`netflix_data_analysis.ipynb`](netflix_data_analysis.ipynb))
2. **Catalog retrieval product** — Boolean, TF-IDF, BM25, dense MiniLM, hybrid RRF, and CPU cross-encoder rerank with honest offline metrics ([`RETRIEVAL.md`](RETRIEVAL.md), `python -m retrieval`)

This is **catalog text search**. It is **not** a production recommender, **not** collaborative filtering, **not** TESSI, and **not** RAG-over-the-web.

## 5-minute story

| Step | What | Where |
|------|------|-------|
| Explore the catalog | Cleaning + charts | `netflix_data_analysis.ipynb` |
| Sparse retrieval | Boolean / TF-IDF / BM25 | notebook + `python -m retrieval` |
| Dense / hybrid / rerank | Embed → fuse → CE rerank → eval | `python -m retrieval`, `RETRIEVAL.md` |

**Headline** (same 28 hand-labeled queries — regenerate with `python -m retrieval eval`):

Baseline on `main`:

| method | recall@5 | recall@10 | ndcg@5 | ndcg@10 | mrr |
| --- | --- | --- | --- | --- | --- |
| boolean | 0.3159 | 0.4012 | 0.3185 | 0.3527 | 0.4440 |
| tf-idf | 0.4502 | 0.5446 | 0.4555 | 0.4912 | 0.5013 |
| dense(title+desc) | 0.5849 | 0.6209 | 0.5710 | 0.5656 | 0.6304 |
| dense(title-only) | 0.4241 | 0.4499 | 0.4286 | 0.4269 | 0.5081 |
| hybrid(tfidf+dense) | 0.5059 | 0.6922 | 0.4941 | 0.5626 | 0.5853 |

Extended ablations (BM25, metadata text, CE rerank):

| method | recall@5 | recall@10 | ndcg@5 | ndcg@10 | mrr |
| --- | --- | --- | --- | --- | --- |
| bm25 | 0.5248 | 0.5645 | 0.5167 | 0.5253 | 0.5637 |
| dense(title+desc+meta) | 0.5735 | 0.6856 | 0.5825 | 0.6193 | 0.6786 |
| hybrid(bm25+dense,meta) | 0.6552 | 0.7421 | 0.6351 | 0.6605 | 0.7065 |
| dense+rerank | 0.6167 | 0.7062 | 0.6179 | 0.6469 | 0.6930 |
| hybrid+rerank | **0.6882** | **0.7627** | **0.6862** | **0.7053** | **0.7601** |

BM25 replaces Boolean as the serious lexical baseline. Metadata helps dense recall@10 / MRR more than sparse early precision. Cross-encoder rerank over top-50 gives the largest early-rank lift. Full table + interpretation: [`RETRIEVAL.md`](RETRIEVAL.md).

## Clone and run

```bash
git clone https://github.com/NikosMav/DataAnalysis-Netflix.git
cd DataAnalysis-Netflix

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# --- Track 1: EDA + sparse TF-IDF notebook ---
jupyter notebook netflix_data_analysis.ipynb

# --- Track 2: retrieval product (no paid API, CPU default) ---
python -m retrieval query "war between vietnam and usa" --method bm25 --top-k 10
python -m retrieval query "feel-good cooking competition show" --method dense-rerank
python -m retrieval query "dark crime thriller set in Scandinavia" --method hybrid-rerank
python -m retrieval eval --failures   # regenerates results/*.json

# Optional walkthrough notebook
jupyter notebook netflix_dense_retrieval.ipynb
```

**Runtime:** sparse notebook needs a few GB RAM for pairwise matrices. Dense first run downloads MiniLM (~80MB) and embeds ~7.8k rows; rerank downloads ms-marco MiniLM CE (~80MB). Then caches under `.cache/`.

## What’s in the box

| Path | Purpose |
|------|---------|
| [`netflix_data_analysis.ipynb`](netflix_data_analysis.ipynb) | EDA + Boolean/TF-IDF case study (kept intact) |
| [`retrieval/`](retrieval/) | Package + CLI (`query`, `eval`, `index`) |
| [`data/labeled_queries.json`](data/labeled_queries.json) | 28 author-labeled queries with `relevant_show_ids` (unchanged) |
| [`results/eval_metrics.json`](results/eval_metrics.json) | Last regenerated metric table |
| [`RETRIEVAL.md`](RETRIEVAL.md) | Full case study: method, baseline + extended results, limits |
| [`netflix_dense_retrieval.ipynb`](netflix_dense_retrieval.ipynb) | Thin package walkthrough |

## Data

Under [`data/`](data/): Netflix titles dump, slim IMDb join for the EDA chart, and retrieval labels. Provenance in [`data/README.md`](data/README.md).

## Limitations (read these)

- **No invented metrics.** Tables above come from `python -m retrieval eval` on the shipped labels.
- **28 queries, author-labeled.** Enough for an honest demo, not a public IR leaderboard.
- **Metadata ablation is uneven.** Genre/cast/director/country help some methods and hurt others — see `RETRIEVAL.md`.
- **Catalog search ≠ recommender ≠ TESSI ≠ web RAG.**

## License

MIT — see [LICENSE.md](LICENSE.md). Netflix catalog © Netflix (public dump via Kaggle). IMDb data subject to [IMDb non-commercial terms](https://developer.imdb.com/non-commercial-datasets/).
