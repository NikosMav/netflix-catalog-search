# Netflix catalog: EDA → sparse retrieval → dense / hybrid retrieval

**Author:** Nikolaos (Nikos) Mavrapidis ([NikosMav](https://github.com/NikosMav))

Two showcase tracks in one repo:

1. **EDA + Boolean / TF-IDF** — original 2023 case study, cleaned and runnable ([`netflix_data_analysis.ipynb`](netflix_data_analysis.ipynb))
2. **Catalog retrieval product** — Boolean, TF-IDF, dense MiniLM, and hybrid RRF with honest offline metrics ([`RETRIEVAL.md`](RETRIEVAL.md), `python -m retrieval`)

This is **catalog text search** (title + description). It is **not** a production recommender, **not** collaborative filtering, **not** TESSI, and **not** RAG-over-the-web.

## 5-minute story

| Step | What | Where |
|------|------|-------|
| Explore the catalog | Cleaning + charts | `netflix_data_analysis.ipynb` |
| Sparse retrieval | Boolean vs TF-IDF demos | same notebook |
| Dense / hybrid product | Embed → index → query → eval | `python -m retrieval`, `RETRIEVAL.md` |

**Headline result** (28 hand-labeled queries — regenerate with `python -m retrieval eval`):

| method | recall@5 | recall@10 | ndcg@5 | ndcg@10 | mrr |
| --- | --- | --- | --- | --- | --- |
| boolean | 0.3159 | 0.4012 | 0.3185 | 0.3527 | 0.4440 |
| tf-idf | 0.4502 | 0.5446 | 0.4555 | 0.4912 | 0.5013 |
| dense(title+desc) | **0.5849** | 0.6209 | **0.5710** | 0.5656 | **0.6304** |
| dense(title-only) | 0.4241 | 0.4499 | 0.4286 | 0.4269 | 0.5081 |
| hybrid(tfidf+dense) | 0.5059 | **0.6922** | 0.4941 | 0.5626 | 0.5853 |

Dense wins early ranking (R@5 / nDCG@5 / MRR). Hybrid wins Recall@10. Title-only embeddings underperform title+description. Full interpretation and failure cases: [`RETRIEVAL.md`](RETRIEVAL.md).

## Clone and run

```bash
git clone https://github.com/NikosMav/DataAnalysis-Netflix.git
cd DataAnalysis-Netflix

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# --- Track 1: EDA + sparse TF-IDF notebook ---
jupyter notebook netflix_data_analysis.ipynb

# --- Track 2: retrieval product (no paid API) ---
python -m retrieval query "war between vietnam and usa" --method dense --top-k 10
python -m retrieval query "feel-good cooking competition show" --method hybrid
python -m retrieval eval --failures   # regenerates results/*.json

# Optional walkthrough notebook
jupyter notebook netflix_dense_retrieval.ipynb
```

**Runtime:** sparse notebook needs a few GB RAM for pairwise matrices. Dense first run downloads MiniLM (~80MB) and embeds ~7.8k rows (a few minutes on CPU); then caches under `.cache/`.

## What’s in the box

| Path | Purpose |
|------|---------|
| [`netflix_data_analysis.ipynb`](netflix_data_analysis.ipynb) | EDA + Boolean/TF-IDF case study (kept intact) |
| [`retrieval/`](retrieval/) | Package + CLI (`query`, `eval`, `index`) |
| [`data/labeled_queries.json`](data/labeled_queries.json) | 28 author-labeled queries with `relevant_show_ids` |
| [`results/eval_metrics.json`](results/eval_metrics.json) | Last regenerated metric table |
| [`RETRIEVAL.md`](RETRIEVAL.md) | Full case study: method, results, ablations, limits |
| [`netflix_dense_retrieval.ipynb`](netflix_dense_retrieval.ipynb) | Thin package walkthrough |

## Data

Under [`data/`](data/): Netflix titles dump, slim IMDb join for the EDA chart, and retrieval labels. Provenance in [`data/README.md`](data/README.md).

## Limitations (read these)

- **No invented metrics.** Table above comes from `python -m retrieval eval` on the shipped labels.
- **28 queries, author-labeled.** Enough for an honest demo, not a public IR leaderboard.
- **Text only.** Genre/cast/popularity unused in the rankers.
- **Catalog search ≠ recommender ≠ TESSI ≠ web RAG.**

## License

MIT — see [LICENSE.md](LICENSE.md). Netflix catalog © Netflix (public dump via Kaggle). IMDb data subject to [IMDb non-commercial terms](https://developer.imdb.com/non-commercial-datasets/).
