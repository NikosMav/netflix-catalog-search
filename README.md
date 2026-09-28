# Netflix Catalog Search

Offline text search over a public Netflix title catalog. The project compares
Boolean retrieval, TF-IDF, BM25, dense embeddings, hybrid reciprocal rank fusion
(RRF), and cross-encoder reranking on a dev split and a held-out test split.

This is a catalog-search project, not a personalized recommender: it uses title,
description, genre, cast, director, and country metadata (plus an optional
Wikipedia plot ablation), but no viewing history or collaborative-filtering signals.

## What this project demonstrates

- Lexical, semantic, hybrid, and two-stage retrieval in one package
- Metadata-field ablations and explicit first-stage/reranker wiring
- Reproducible Recall, nDCG, and MRR evaluation
- A command-line interface for querying, indexing, and evaluation
- A local UI that explains each result from the model that ranked it
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

The headline table is the held-out **test** split. The reported systems, the BM25
baseline, Recall@5, Recall@10, nDCG@10, MRR, and the paired bootstrap and sign
test were declared in [`configs/eval_v2.yaml`](configs/eval_v2.yaml) before that
split was scored. `hybrid+rerank` is cross-encoder reranking over
`hybrid(bm25+dense,meta)`.

Judgments are grades from 0 to 3. nDCG@10 uses gain `2^grade - 1`. Recall and
MRR count a title only when the grade is 2 or 3. The judge saw a shuffled
candidate card (title, type, year, genres, cast, director, description) and did
not see system names, ranks, scores, or the older labels. Queries on both splits
were written for this project. The dev split reuses the earlier query texts with
new labels. The human spot-check of the judge is not filled in.

Dev results, field ablations, pool and label counts, and the v1 table are in
[`RETRIEVAL.md`](RETRIEVAL.md). The blocks below are written by
`python scripts/sync_metrics_docs.py` from the committed JSON.

<!-- METRICS_TABLE_BEGIN -->
| method | n_scored | recall@5 | recall@10 | ndcg@10 | mrr | nDCG@10 difference vs bm25 | 95% CI | sign-test p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| boolean | 29 | 0.5259 | 0.5632 | 0.5151 | 0.5184 | -0.1301 | [-0.2494, -0.0355] | 0.14599609375 |
| tf-idf | 29 | 0.5989 | 0.6477 | 0.5755 | 0.5917 | -0.0697 | [-0.1719, 0.0131] | 0.7744140625 |
| bm25 | 29 | 0.6592 | 0.6966 | 0.6452 | 0.6759 | baseline |  |  |
| dense(title+desc) | 29 | 0.6017 | 0.6603 | 0.5995 | 0.6003 | -0.0456 | [-0.1696, 0.0607] | 1.0 |
| hybrid(bm25+dense,meta) | 29 | 0.6459 | 0.745 | 0.6754 | 0.6707 | 0.0302 | [-0.0994, 0.1483] | 0.60723876953125 |
| hybrid+rerank | 29 | 0.8579 | 0.9037 | 0.8859 | 0.8908 | 0.2408 | [0.1269, 0.3676] | 0.0001220703125 |
<!-- METRICS_TABLE_END -->

<!-- EVAL_V2_NOTES_BEGIN -->
Judge: LLM judge (grok-4.7 via Cursor cloud agent), blinded to system and rank.

Test split: 30 queries, 29 scored for nDCG. Unscored for nDCG because the highest grade is 0: `regency_london_romance`.

Each interval is the paired bootstrap of the per-query ndcg@10 difference against `bm25`.

`boolean` minus `bm25` mean ndcg@10 difference -0.1301 (0.95 paired bootstrap interval [-0.2494, -0.0355], entirely below 0; 10000 resamples, seed 20260928). Two-sided sign test p 0.14599609375 (+3 / −9 / ties 17, n=29).

`tf-idf` minus `bm25` mean ndcg@10 difference -0.0697 (0.95 paired bootstrap interval [-0.1719, 0.0131], includes 0; 10000 resamples, seed 20260928). Two-sided sign test p 0.7744140625 (+5 / −7 / ties 17, n=29).

`dense(title+desc)` minus `bm25` mean ndcg@10 difference -0.0456 (0.95 paired bootstrap interval [-0.1696, 0.0607], includes 0; 10000 resamples, seed 20260928). Two-sided sign test p 1.0 (+6 / −7 / ties 16, n=29).

`hybrid(bm25+dense,meta)` minus `bm25` mean ndcg@10 difference 0.0302 (0.95 paired bootstrap interval [-0.0994, 0.1483], includes 0; 10000 resamples, seed 20260928). Two-sided sign test p 0.60723876953125 (+9 / −6 / ties 14, n=29).

`hybrid+rerank` minus `bm25` mean ndcg@10 difference 0.2408 (0.95 paired bootstrap interval [0.1269, 0.3676], entirely above 0; 10000 resamples, seed 20260928). Two-sided sign test p 0.0001220703125 (+14 / −0 / ties 15, n=29).

Spot-check sample: 303 pairs from 3032 judged pairs (fraction 0.1, seed 20260930). Status: spot-check pending.
<!-- EVAL_V2_NOTES_END -->

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

## Run the UI

```bash
pip install -e ".[ui]"
streamlit run app/streamlit_app.py
```

![Search page with a per-result explanation](docs/ui-search.png)

The app is a search page, an evaluation page, and a spot-check page. Search runs
BM25, dense MiniLM, hybrid RRF, or hybrid plus cross-encoder rerank (the default).
Open "Why this result?" on a hit for the score breakdown from that method.
Evaluation reads [`results/eval_v2/metrics.json`](results/eval_v2/metrics.json)
and can switch between the test and dev splits. Replay ranks a v2 labeled query
with the method you pick. Spot-check shows a sampled card and writes
`data/eval_v2/spotcheck_human.json` without displaying the model judgment.

The first dense or reranked search downloads the MiniLM models and caches
embeddings under `.cache/`. Later runs stay on CPU.

Precompute all dense indexes or regenerate evaluation artifacts:

```bash
python -m retrieval index
python scripts/run_eval_v2.py
python scripts/sync_metrics_docs.py
```

Wikipedia plot enrichment is already committed under `data/wikipedia_plots.jsonl`
(CC BY-SA). To regenerate from the MediaWiki API:

```bash
pip install -e ".[enrich]"
python scripts/enrich_wikipedia_plots.py
```

Install the full exploratory-notebook stack with `pip install -r requirements.txt`.

## Repository map

| Path | Purpose |
| --- | --- |
| `retrieval/` | Catalog loading, retrievers, reranker, metrics, and CLI |
| `retrieval/explain.py` | Score explanations used by the UI |
| `app/streamlit_app.py` | Local search and evaluation UI |
| `tests/` | Fast unit and wiring tests |
| `data/labeled_queries.json` | v1 labels, kept byte-identical |
| `data/eval_v2/` | v2 queries, blinded pools, judgments, spot-check sample |
| `configs/eval_v2.yaml` | Pre-registered v2 systems, metrics, and tests |
| `data/wikipedia_plots.jsonl` | Wikipedia plot enrichment (CC BY-SA) |
| `results/eval_v2/` | v2 metrics and spot-check agreement |
| `results/eval_metrics.json` | v1 metrics, kept byte-identical |
| `RETRIEVAL.md` | Detailed retrieval case study |
| `netflix_data_analysis.ipynb` | Original EDA and sparse-retrieval chapter |
| `netflix_dense_retrieval.ipynb` | Short package walkthrough |

## Evaluation boundaries

- v2 labels are grades from 0 to 3, defined in
  [`docs/eval_v2_rubric.md`](docs/eval_v2_rubric.md). An LLM judge applied that
  rubric to the blinded card only. The judge string in the metrics JSON names
  that setup. nDCG uses gain `2^grade - 1`. Recall and MRR use grades 2 and 3.
- Test queries were written by browsing the catalog, before pooling. Dev queries
  are the earlier query texts with new labels. Both were written for this project.
- nDCG means skip queries whose highest grade is 0. Recall and MRR means skip
  queries with no grade of 2 or 3. The metrics JSON records both counts.
  Intervals are paired bootstrap differences against BM25 on the test split.
- The human spot-check is pending until `data/eval_v2/spotcheck_human.json` exists.
  The sample does not include the model label.
- Settings listed as tuned in `configs/eval_v2.yaml` were frozen from the earlier
  labels on the dev query texts. They were not re-selected on the test split or
  on the v2 labels.
- v1 (`data/labeled_queries.json`, `results/eval_metrics.json`) stays byte-identical.
  That evaluation used one query set for both tuning and reporting and has no interval.
- Results describe this catalog snapshot and the frozen configuration.
- Lightweight CI does not download embedding models. It checks that the committed
  docs match `python scripts/sync_metrics_docs.py`.

## Data and license

The catalog is the public Netflix Movies and TV Shows dataset distributed by
[Shivam Bansal on Kaggle](https://www.kaggle.com/shivamb/netflix-shows) and mirrored
by [TidyTuesday](https://github.com/rfordatascience/tidytuesday/tree/master/data/2021/2021-04-20).
Provenance and IMDb-join caveats are documented in [`data/README.md`](data/README.md).

Code is released under the [MIT License](LICENSE.md). Netflix catalog content remains
the property of Netflix; IMDb-derived data is subject to IMDb's non-commercial terms.
Wikipedia plot text in [`data/wikipedia_plots.jsonl`](data/wikipedia_plots.jsonl) is
[CC BY-SA 4.0](data/WIKIPEDIA_ATTRIBUTION.md).
