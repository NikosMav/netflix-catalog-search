# Netflix Catalog Search

Offline text search over a public Netflix title catalog. The project compares
Boolean retrieval, TF-IDF, BM25, dense embeddings, hybrid reciprocal rank fusion
(RRF), and cross-encoder reranking on one labeled query set.

This is a catalog-search project, not a personalized recommender: it uses title,
description, genre, cast, director, and country metadata, but no viewing history or
collaborative-filtering signals.

## What this project demonstrates

- Lexical, semantic, hybrid, and two-stage retrieval in one package
- Metadata-field ablations and explicit first-stage/reranker wiring
- Reproducible Recall, nDCG, and MRR evaluation
- A command-line interface for querying, indexing, and evaluation
- Model and embedding caches for repeatable local runs
- Unit tests and lightweight CI without model downloads

## Retrieval pipeline

```text
query -> BM25 ------------------+
                                 +-> RRF -> top 50 -> cross-encoder -> results
query -> MiniLM dense retrieval -+
```

Exact names and titles often favor BM25, while dense retrieval handles paraphrases.
RRF combines their rank lists without mixing incompatible raw score scales; the
cross-encoder then spends more compute only on the strongest candidates.

## Results

Metrics come from 28 author-labeled queries with binary relevance judgments. The
committed [`results/eval_metrics.json`](results/eval_metrics.json) is the source of
truth and can be regenerated with `python -m retrieval eval`.

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

`hybrid+rerank` means cross-encoder reranking over
`hybrid(bm25+dense,meta)`. Full method notes, qualitative examples, and failure
analysis are in [`RETRIEVAL.md`](RETRIEVAL.md).

## Quick start

```bash
git clone https://github.com/NikosMav/netflix-catalog-search.git
cd netflix-catalog-search
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

python -m retrieval query "war between vietnam and usa" --method bm25 --top-k 10
python -m retrieval query "feel-good cooking competition show" --method hybrid-rerank
pytest
```

The BM25, Boolean, and TF-IDF paths need no model download. The first dense or
reranked query downloads the MiniLM bi-encoder and MS MARCO cross-encoder and
caches catalog embeddings under `.cache/`.

Precompute all dense indexes or regenerate evaluation artifacts:

```bash
python -m retrieval index
python -m retrieval eval --failures
python scripts/sync_metrics_docs.py
```

Install the full exploratory-notebook stack with `pip install -r requirements.txt`.

## Repository map

| Path | Purpose |
| --- | --- |
| `retrieval/` | Catalog loading, retrievers, reranker, metrics, and CLI |
| `tests/` | Fast unit and wiring tests |
| `data/labeled_queries.json` | Evaluation queries and relevant show IDs |
| `results/` | Metrics and qualitative comparisons |
| `RETRIEVAL.md` | Detailed retrieval case study |
| `netflix_data_analysis.ipynb` | Original EDA and sparse-retrieval chapter |
| `netflix_dense_retrieval.ipynb` | Short package walkthrough |

## Evaluation boundaries

- The 28 queries and relevance judgments were created by the project author.
- Labels are binary and have no inter-annotator agreement or confidence intervals.
- Results describe this fixed catalog snapshot, query set, and model configuration.
- Full model evaluation is intentionally separate from lightweight CI.

## Data and license

The catalog is the public Netflix Movies and TV Shows dataset distributed by
[Shivam Bansal on Kaggle](https://www.kaggle.com/shivamb/netflix-shows) and mirrored
by [TidyTuesday](https://github.com/rfordatascience/tidytuesday/tree/master/data/2021/2021-04-20).
Provenance and IMDb-join caveats are documented in [`data/README.md`](data/README.md).

Code is released under the [MIT License](LICENSE.md). Netflix catalog content remains
the property of Netflix; IMDb-derived data is subject to IMDb's non-commercial terms.
