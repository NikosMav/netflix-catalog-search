# Catalog retrieval case study

**Author:** Nikolaos (Nikos) Mavrapidis ([NikosMav](https://github.com/NikosMav))

Five-minute walkthrough. This is **catalog text search** over a public Netflix titles dump. It is **not** a production recommender, **not** collaborative filtering, and **not** RAG-over-the-web (no LLM generation step).

The older EDA + Boolean/TF-IDF notebook remains: [`netflix_data_analysis.ipynb`](netflix_data_analysis.ipynb). This document covers the **retrieval product** next to it.

## Problem

Given a natural-language query, rank Netflix catalog rows by text similarity. Compare classical sparse methods, dense embeddings, hybrid fusion, and a CPU cross-encoder reranker on the **same** 28 labeled queries.

## Method

| Method | Representation | Scoring |
|--------|----------------|---------|
| **Boolean** | Binary bag-of-words (uni+bigrams) | Set Jaccard |
| **TF-IDF** | Weighted sparse terms (uni+bigrams, 10k features) | Cosine |
| **BM25** | Okapi BM25 (`rank-bm25`) | BM25 score |
| **Dense** | `sentence-transformers/all-MiniLM-L6-v2` (CPU) | Cosine NN |
| **Hybrid** | Sparse + dense rank lists | Reciprocal Rank Fusion (k=60) |
| **+rerank** | Top-50 of dense, or of `hybrid(bm25+dense,meta)` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

**Document-text ablations**

| Field | Contents |
|-------|----------|
| `text` (desc-only) | `title` + `description` |
| `text_meta` (desc+meta) | above + `listed_in` + `cast` + `director` + `country` |
| `title_text` | title only (dense ablation) |

No new catalog rows; metadata columns were already in `netflix_titles.csv`.

## How to run

```bash
git clone https://github.com/NikosMav/netflix-catalog-search.git
cd netflix-catalog-search
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Query the catalog (no paid API; CPU default)
python -m retrieval query "war between vietnam and usa" --method bm25 --top-k 10
python -m retrieval query "feel-good cooking competition show" --method dense-rerank
python -m retrieval query "dark crime thriller set in Scandinavia" --method hybrid-rerank

# Regenerate the metrics table from a clean checkout
python -m retrieval eval --failures
# → results/eval_metrics.json  (+ qualitative_examples.json)
```

First dense / rerank run downloads MiniLM bi-encoder (~80MB) and the ms-marco cross-encoder (~80MB), then embeds ~7.8k titles (a few minutes on CPU). Caches under `.cache/`.

Optional OpenAI embeddings: `DenseRetriever(backend="openai")` + `OPENAI_API_KEY` (not required).

## Results (28 hand-labeled queries)

Labels: [`data/labeled_queries.json`](data/labeled_queries.json) — author judgments with `relevant_show_ids` from this dump. Binary relevance. **Same 28 queries** (unchanged gold set). Metrics from `python -m retrieval eval`; committed [`results/eval_metrics.json`](results/eval_metrics.json) is the source of truth. **n=28, author labels — no confidence intervals.**

`hybrid+rerank` = cross-encoder over **`hybrid(bm25+dense,meta)`** (BM25 + dense on `text_meta`), not over `hybrid(tfidf+dense)`.

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

Numbers above match [`results/eval_metrics.json`](results/eval_metrics.json) from the last `python -m retrieval eval` run. After changing retrieval wiring, re-run eval (and `python scripts/sync_metrics_docs.py`) before quoting headline numbers.

### What the numbers mean

- **Recall@k** — fraction of labeled relevant titles found in the top-k.
- **MRR** — how early the first relevant title appears (1 = rank 1).
- **nDCG@k** — ranking quality with binary gains (order matters).

### Takeaways (honest)

1. **BM25 beats Boolean and TF-IDF** on this set (R@5 0.52 vs 0.32 / 0.45). Boolean Jaccard was a coarse demo baseline; BM25 is the proper lexical comparator.
2. **Metadata is mixed, not free lift.** Appending genre/cast/director/country *hurts* TF-IDF early ranks (cast-name noise) and slightly lowers BM25/dense Recall@5, but **helps dense Recall@10 and MRR**. Genre tokens help topical recall; long cast strings dilute sparse IDF.
3. **Cross-encoder rerank lifts early ranks** over a fixed top-50 pool. Headline path: `hybrid+rerank` = CE over `hybrid(bm25+dense,meta)` (R@5 0.7001 / R@10 0.7817 / MRR 0.7583 on this set).
4. **Strong first-stage fusion still matters.** `hybrid(bm25+dense,meta)` already beats `hybrid(tfidf+dense)` on every metric before any CE pass.
5. Gains are real but **set-specific** — 28 author-labeled queries, not a public IR benchmark. Catalog search ≠ recommender.

## Ablations & failure cases

### Dense / rerank wins (paraphrase)

Query: `feel-good cooking competition show`  
Dense and `dense+rerank` return competition cooking shows. Boolean/TF-IDF often latch onto isolated tokens (`feel`, `cook`).

Query: `dark crime thriller set in Scandinavia`  
Dense surfaces Nordic crime; sparse returns generic US crime — “Scandinavia” rarely appears verbatim. Metadata (`country`) can help when the first stage indexes `text_meta`.

### Sparse / exact tokens win

Query: `war between vietnam and usa`  
BM25 / TF-IDF place Vietnam War titles high via distinctive tokens. Dense is competitive but can insert topical near-misses (`V Wars`).

Query: `Kevin Hart stand-up comedy special` / `Stranger Things`  
Exact name/title tokens favor sparse overlap; BM25 is especially strong here.

### Mickey Mouse (ambiguous lexical match)

Query: `Mickey Mouse`  
Sparse ranks other “mouse” titles. Dense ranks Disney shorts (mentions Mickey) higher — still noisy.

Full side-by-side dumps: [`results/qualitative_examples.json`](results/qualitative_examples.json).

## Repo map

| Path | Role |
|------|------|
| `retrieval/` | Package + `python -m retrieval` CLI |
| `retrieval/bm25.py` | BM25 Okapi baseline |
| `retrieval/rerank.py` | Cross-encoder second stage |
| `data/netflix_titles.csv` | Catalog |
| `data/labeled_queries.json` | Eval labels (show_ids) — **unchanged 28 queries** |
| `results/` | Regenerated metrics + qualitative JSON |
| `netflix_data_analysis.ipynb` | Original EDA + Boolean/TF-IDF showcase |
| `netflix_dense_retrieval.ipynb` | Optional walkthrough notebook |

## Limitations

- **Catalog search ≠ recommender.** No watch history, no CF, no popularity re-rank.
- **Not web RAG.** No chunking of long docs, no tool-using agent, no generation.
- **Tiny labeled set.** 28 author queries; useful for honesty, not SOTA claims.
- **Binary labels only.** No graded relevance → nDCG uses 0/1 gains.
- **Short marketing blurbs.** Bad descriptions limit every method; metadata helps unevenly.
- **CPU MiniLM + ms-marco CE are demo models.** Stronger embedders / larger cross-encoders would change ranks.
- **Hybrid is RRF, not learned fusion.** Rerank is greedy over a fixed top-50 pool.
- **No external enrichment.** No TMDB / Wikipedia scrape in this pass.

## License

MIT — see [LICENSE.md](LICENSE.md). Netflix catalog © Netflix (public dump). Labels are original to this repo.
