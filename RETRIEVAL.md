# Catalog retrieval case study

**Author:** Nikolaos (Nikos) Mavrapidis ([NikosMav](https://github.com/NikosMav))

Five-minute walkthrough. This is **catalog text search** over a public Netflix titles dump. It is **not** a production recommender, **not** collaborative filtering, and **not** RAG-over-the-web (no LLM generation step).

The older EDA + Boolean/TF-IDF notebook remains: [`netflix_data_analysis.ipynb`](netflix_data_analysis.ipynb). This document covers the **retrieval product** next to it.

## Problem

Given a natural-language query, rank Netflix catalog rows by text similarity. Compare classical sparse methods, dense embeddings, hybrid fusion, and a CPU cross-encoder reranker. Evaluation v2 splits those comparisons into a dev set and a held-out test set.

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
| `text_plot` (desc+plot) | `title` + `description` + Wikipedia plot/premise (when matched) |
| `title_text` | title only (dense ablation) |

No new catalog rows for metadata; `listed_in` / cast / director / country were already in
`netflix_titles.csv`. Plot text comes from the committed
[`data/wikipedia_plots.jsonl`](data/wikipedia_plots.jsonl) enrichment (CC BY-SA; see
[`data/WIKIPEDIA_ATTRIBUTION.md`](data/WIKIPEDIA_ATTRIBUTION.md)).
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

# Regenerate v2 metrics (downloads MiniLM; not part of CI), then refresh the docs
python scripts/run_eval_v2.py
python scripts/sync_metrics_docs.py
# Legacy v1 file, still produced by:
# python -m retrieval eval --failures  → results/eval_metrics.json
```

First dense / rerank run downloads MiniLM bi-encoder (~80MB) and the ms-marco cross-encoder (~80MB), then embeds ~7.8k titles (a few minutes on CPU). Caches under `.cache/`.

Optional OpenAI embeddings: `DenseRetriever(backend="openai")` + `OPENAI_API_KEY` (not required).

## Evaluation v2

Protocol: [`configs/eval_v2.yaml`](configs/eval_v2.yaml), committed before the test split was scored. Rubric: [`docs/eval_v2_rubric.md`](docs/eval_v2_rubric.md). Headline numbers for the test split are in the README. This section is the dev split, the ablations, and the legacy v1 table.

`hybrid+rerank` is the cross-encoder over `hybrid(bm25+dense,meta)` (BM25 + dense on `text_meta`), not over `hybrid(tfidf+dense)`.

The reported systems were frozen before the test run. What was tuned on the dev query texts used the earlier v1 labels for those texts, not the v2 judgments, and was not re-selected afterward:

- The headline hybrid first stage is BM25 + dense MiniLM on `text_meta`. TF-IDF + dense on title and description stays an ablation.
- The headline reranker rescores that hybrid's candidate pool. Dense-only rerank stays an ablation.
- Reported dense retrieval is title + description. Metadata and plot dense indexes stay ablations.
- The BM25 baseline is title + description, not the metadata or plot BM25 ablation.
- BM25 `k1` and `b`, RRF `k`, and the TF-IDF feature cap are library defaults. They were not grid-searched.

The judge is named in the metrics JSON: an LLM judge (grok-4.7 via Cursor cloud agent), blinded to system and rank. Each decision used the query and the candidate card only. Queries with an empty relevant set stay in the per-query file and are left out of the means.

### Label corrections

A few committed labels were changed after the blinded judge batches. The list is copied from [`data/eval_v2/label_overrides.json`](data/eval_v2/label_overrides.json). `judgments.json` points at that file.

<!-- EVAL_V2_OVERRIDES_BEGIN -->
2 committed labels differ from the blinded judge batches. There are no other overrides.

| query_id | show_id | title | batch | raw label | final label | reason |
| --- | --- | --- | --- | --- | --- | --- |
| office_mockumentary | s6719 | The Office (U.S.) | F | 0 | 1 | The card is a TV comedy about office workers at the Dunder Mifflin paper company, so the pair was marked relevant even though the card never says mockumentary. |
| vietnam_war | s1570 | Da 5 Bloods | C | 1 | 0 | The card is a decades-later return to recover remains and buried gold, so the pair was marked not relevant to a query about the war itself. |
<!-- EVAL_V2_OVERRIDES_END -->

- **Recall@k** — fraction of labeled relevant titles found in the top k.
- **MRR** — how early the first relevant title appears (1 means rank 1).
- **nDCG@k** — ranking quality with binary gains.

The blocks below are written by `python scripts/sync_metrics_docs.py`.

### Dev split

<!-- EVAL_V2_DEV_BEGIN -->
Dev split: 28 queries, 25 scored. Unscored because the relevant set is empty: `semantic_scandi_crime`, `anime_death_note`, `dark_german_series`. No test-set interval is computed on this split.

| method | n_scored | recall@5 | recall@10 | ndcg@10 | mrr |
| --- | --- | --- | --- | --- | --- |
| boolean | 25 | 0.2747 | 0.3942 | 0.3033 | 0.3674 |
| tf-idf | 25 | 0.4969 | 0.5588 | 0.4744 | 0.5191 |
| bm25 | 25 | 0.5422 | 0.5879 | 0.5291 | 0.6027 |
| dense(title+desc) | 25 | 0.6552 | 0.7524 | 0.6744 | 0.716 |
| hybrid(bm25+dense,meta) | 25 | 0.6999 | 0.7852 | 0.7205 | 0.7647 |
| hybrid+rerank | 25 | 0.7575 | 0.8676 | 0.8138 | 0.8433 |
<!-- EVAL_V2_DEV_END -->

### Dev ablations

These runs are not in the headline table. They are scored on the dev split only. Metadata and Wikipedia plot text are mixed: compare each ablation row with the reported system that uses title and description. Plot match coverage is recorded in [`data/wikipedia_plots_coverage.json`](data/wikipedia_plots_coverage.json).

<!-- EVAL_V2_ABLATIONS_BEGIN -->
| method | n_scored | recall@5 | recall@10 | ndcg@10 | mrr |
| --- | --- | --- | --- | --- | --- |
| tf-idf(desc+meta) | 25 | 0.4212 | 0.5407 | 0.4364 | 0.4827 |
| tf-idf(desc+plot) | 25 | 0.3775 | 0.4842 | 0.4222 | 0.498 |
| bm25(desc+meta) | 25 | 0.5632 | 0.68 | 0.5953 | 0.6284 |
| bm25(desc+plot) | 25 | 0.5484 | 0.6245 | 0.5796 | 0.6683 |
| dense(title+desc+meta) | 25 | 0.6853 | 0.7956 | 0.7189 | 0.7767 |
| dense(title+desc+plot) | 25 | 0.6245 | 0.7085 | 0.636 | 0.6733 |
| dense(title-only) | 25 | 0.4672 | 0.5234 | 0.498 | 0.59 |
| hybrid(tfidf+dense) | 25 | 0.5985 | 0.7463 | 0.6113 | 0.6213 |
| dense+rerank | 25 | 0.7129 | 0.8165 | 0.7762 | 0.8233 |
<!-- EVAL_V2_ABLATIONS_END -->

### Labels, pool, and spot-check

Candidates are the union of each system's top of the pool, including the ablations, shuffled without system names, ranks, scores, or v1 labels.

<!-- EVAL_V2_STATS_BEGIN -->
Pool depth 10 across 15 systems and 58 queries (3032 candidate slots; 21 to 76 unique titles per query).

Dev labels: 126 relevant pairs out of 1276 judged pairs (25 of 28 queries have a relevant title).

Test labels: 81 relevant pairs out of 1756 judged pairs (29 of 30 queries have a relevant title).

v2-relevant dev titles absent from that query's v1 relevant set: 73.

Spot-check sample: 303 pairs from 3032 judged pairs (fraction 0.1, seed 20260930). Status: spot-check pending.
<!-- EVAL_V2_STATS_END -->

Nikos can label the sample without seeing the model judgment. The CLI and the Streamlit spot-check page both write `data/eval_v2/spotcheck_human.json`. Neither shows the model label.

```bash
python scripts/eval_v2_spotcheck.py show
python scripts/eval_v2_spotcheck.py label <sample_id> --relevant
python scripts/eval_v2_spotcheck.py label <sample_id> --not-relevant
python scripts/eval_v2_agreement.py
```

Until that human file has labels, the agreement file says spot-check pending. Do not fill the human file in as the model.

### Limits

The splits are small, the judge is an LLM, and the queries were written by the project author and agent rather than drawn from a search log. A difference whose interval includes 0 is not a stable gain over BM25 on this test set. Hybrid RRF was not re-chosen after the test run. Read the test intervals in the README before treating a hybrid row as better than BM25.

## v1 (legacy)

v1 scored every ablation on the same queries that were used to choose the headline configuration. There is no held-out split and no interval in that file. [`data/labeled_queries.json`](data/labeled_queries.json) and [`results/eval_metrics.json`](results/eval_metrics.json) are unchanged and are not the v2 labels.

<!-- V1_METRICS_TABLE_BEGIN -->
v1 scored 28 queries. This table is the legacy file, not the v2 test result.

| method | recall@5 | recall@10 | ndcg@5 | ndcg@10 | mrr |
| --- | --- | --- | --- | --- | --- |
| boolean | 0.3159 | 0.4012 | 0.3185 | 0.3527 | 0.444 |
| tf-idf | 0.4502 | 0.5446 | 0.4555 | 0.4912 | 0.5013 |
| tf-idf(desc+meta) | 0.4192 | 0.5446 | 0.3875 | 0.4374 | 0.4659 |
| tf-idf(desc+plot) | 0.3639 | 0.4722 | 0.3672 | 0.4038 | 0.4956 |
| bm25 | 0.5248 | 0.5645 | 0.5167 | 0.5253 | 0.5637 |
| bm25(desc+meta) | 0.502 | 0.5524 | 0.5233 | 0.5353 | 0.6002 |
| bm25(desc+plot) | 0.5069 | 0.6116 | 0.5093 | 0.5494 | 0.5972 |
| dense(title+desc) | 0.5849 | 0.6209 | 0.571 | 0.5656 | 0.6304 |
| dense(title+desc+meta) | 0.5735 | 0.6856 | 0.5825 | 0.6193 | 0.6786 |
| dense(title+desc+plot) | 0.5569 | 0.6506 | 0.5524 | 0.5738 | 0.6376 |
| dense(title-only) | 0.4241 | 0.4499 | 0.4286 | 0.4269 | 0.5081 |
| hybrid(tfidf+dense) | 0.5059 | 0.6922 | 0.4941 | 0.5626 | 0.5853 |
| hybrid(bm25+dense,meta) | 0.6552 | 0.7421 | 0.6351 | 0.6605 | 0.7065 |
| dense+rerank | 0.6167 | 0.7062 | 0.6179 | 0.6469 | 0.693 |
| hybrid+rerank | 0.7001 | 0.7817 | 0.6958 | 0.7185 | 0.7583 |
<!-- V1_METRICS_TABLE_END -->

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
| `data/labeled_queries.json` | v1 labels, kept byte-identical |
| `data/eval_v2/` | v2 queries, blinded pools, judgments, spot-check sample |
| `results/eval_metrics.json` | v1 metrics, kept byte-identical |
| `results/eval_v2/` | v2 metrics and spot-check agreement |
| `netflix_data_analysis.ipynb` | Original EDA + Boolean/TF-IDF showcase |
| `netflix_dense_retrieval.ipynb` | Optional walkthrough notebook |

## Limitations

- **Catalog search ≠ recommender.** No watch history, no CF, no popularity re-rank.
- **Not web RAG.** No chunking of long docs, no tool-using agent, no generation.
- **Small labeled splits, an LLM judge, and queries written for this project.** Useful for honesty, not a public IR benchmark. The human spot-check is pending.
- **Binary labels only.** No graded relevance, so nDCG uses 0/1 gains. Queries with an empty relevant set are omitted from the means and the paired tests.
- **Short marketing blurbs.** Bad descriptions limit every method; metadata helps unevenly.
  Wikipedia plot enrichment (`text_plot`) is an optional ablation — match errors and missing
  pages remain a caveat (see coverage JSON).
- **CPU MiniLM + ms-marco CE are demo models.** Stronger embedders / larger cross-encoders would change ranks.
- **Hybrid is RRF, not learned fusion.** Rerank is greedy over a fixed top-50 pool.
## License

MIT — see [LICENSE.md](LICENSE.md). Netflix catalog © Netflix (public dump). Labels are original to this repo.
Wikipedia plot excerpts in `data/wikipedia_plots.jsonl` are **CC BY-SA 4.0** — see
[`data/WIKIPEDIA_ATTRIBUTION.md`](data/WIKIPEDIA_ATTRIBUTION.md).
