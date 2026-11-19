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

# Run all models again and inspect the metrics and top ten in .cache/eval_v2/
python scripts/run_eval_v2.py
# Optional: regenerate candidate cards and pool statistics under .cache/eval_v2/
python scripts/pool_eval_v2.py
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

The judge is named in the metrics JSON: an LLM judge (grok-4.7 via Cursor cloud agent), blinded to system and rank. Each decision used the query and the candidate card only. Grades are 0 (not relevant), 1 (marginal), 2 (relevant), and 3 (highly relevant). nDCG@10 uses gain `2^grade - 1`. Recall and MRR treat grade 2 or 3 as relevant. A query with no positive grade is left out of nDCG. A query with no grade of 2 or 3 is left out of recall and MRR.

### Held-out comparison

<!-- EVAL_V2_TEST_BEGIN -->
| method | n_scored | recall@5 | recall@10 | ndcg@10 | mrr | nDCG@10 difference vs bm25 | 95% CI | sign-test p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| boolean | 30 | 0.5362 | 0.5701 | 0.419 | 0.5299 | -0.125 | [-0.221, -0.0457] | 0.0025768280029296875 |
| tf-idf | 30 | 0.5954 | 0.6437 | 0.4973 | 0.5986 | -0.0467 | [-0.1296, 0.0227] | 0.6900379657745361 |
| bm25 | 30 | 0.6592 | 0.6937 | 0.544 | 0.6897 | baseline |  |  |
| dense(title+desc) | 30 | 0.592 | 0.6517 | 0.5391 | 0.6003 | -0.0049 | [-0.1, 0.0783] | 0.711071103811264 |
| hybrid(bm25+dense,meta) | 30 | 0.6431 | 0.7393 | 0.6202 | 0.6764 | 0.0763 | [-0.0109, 0.1656] | 0.16863754391670227 |
| hybrid+rerank | 30 | 0.8493 | 0.9002 | 0.8094 | 0.8908 | 0.2655 | [0.1728, 0.3644] | 8.430331945419312e-06 |
<!-- EVAL_V2_TEST_END -->

All 30 queries contribute to nDCG. Recall and reciprocal rank use the 29 queries with a pooled grade of at least 2. Reciprocal rank uses the first such title in the saved top ten. A paired interval measures variation across queries conditional on the recorded judgments.

### Fresh model run

A Windows CPU rerun of all six systems on the 30 test queries reproduced every reranked top ten and the published mean nDCG@10 of 0.8094. BM25 reproduced 27 top tens exactly and the same mean of 0.5440. Standalone dense retrieval and hybrid fusion also reproduced their published means.

Boolean scored 0.4196 versus the frozen 0.4190, and TF-IDF scored 0.4977 versus 0.4973. Live top tens included 12 Boolean slots and one TF-IDF slot outside the original judged pool. The live calculation assigns zero gain to unjudged slots. Sparse rankings can order equal-scoring titles differently across numerical library versions. The browser explorer replays the original rankings and CI recomputes their scores and intervals directly from the frozen judgments.

Rerun environment: NumPy 2.5.5, scikit-learn 1.9.1, sentence-transformers 3.4.1 and PyTorch 2.15.0.

### Label corrections

A few committed labels were changed after the blinded judge batches. The list is copied from [`data/eval_v2/label_overrides.json`](data/eval_v2/label_overrides.json). `judgments.json` points at that file.

<!-- EVAL_V2_OVERRIDES_BEGIN -->
2 committed grades differ from the blinded re-judge. There are no other overrides.

| query_id | show_id | title | raw grade | final grade | reason |
| --- | --- | --- | --- | --- | --- |
| office_mockumentary | s6719 | The Office (U.S.) | 3 | 2 | The blinded re-judge assigned grade 3, but the card never says mockumentary, so the carried relevant decision is grade 2 rather than exact intent. |
| vietnam_war | s1570 | Da 5 Bloods | 1 | 0 | The blinded re-judge assigned grade 1. The carried decision stays not relevant: the card is a decades-later return for remains and gold, not the war itself. |
<!-- EVAL_V2_OVERRIDES_END -->

- **Recall@k** — fraction of titles graded 2 or 3 found in the top k.
- **MRR** — how early the first title graded 2 or 3 appears (1 means rank 1).
- **nDCG@k** — ranking quality with gain `2^grade - 1`.

The blocks below are written by `python scripts/sync_metrics_docs.py`.

### Dev split

<!-- EVAL_V2_DEV_BEGIN -->
Dev split: 28 queries, 28 scored for nDCG. Unscored for nDCG because the highest grade is 0: none. Unscored for recall and MRR because no grade is 2 or 3: `semantic_scandi_crime`, `anime_death_note`. No test-set interval is computed on this split.

| method | n_scored | recall@5 | recall@10 | ndcg@10 | mrr |
| --- | --- | --- | --- | --- | --- |
| boolean | 28 | 0.2773 | 0.3828 | 0.4223 | 0.4204 |
| tf-idf | 28 | 0.493 | 0.5429 | 0.577 | 0.5381 |
| bm25 | 28 | 0.5584 | 0.6262 | 0.6161 | 0.634 |
| dense(title+desc) | 28 | 0.6222 | 0.7105 | 0.7184 | 0.6917 |
| hybrid(bm25+dense,meta) | 28 | 0.6833 | 0.7873 | 0.7609 | 0.7564 |
| hybrid+rerank | 28 | 0.7375 | 0.8724 | 0.857 | 0.8494 |
<!-- EVAL_V2_DEV_END -->

### Dev ablations

These runs are not in the headline table. They are scored on the dev split only. Metadata and Wikipedia plot text are mixed: compare each ablation row with the reported system that uses title and description. Plot match coverage is recorded in [`data/wikipedia_plots_coverage.json`](data/wikipedia_plots_coverage.json).

<!-- EVAL_V2_ABLATIONS_BEGIN -->
| method | n_scored | recall@5 | recall@10 | ndcg@10 | mrr |
| --- | --- | --- | --- | --- | --- |
| tf-idf(desc+meta) | 28 | 0.4154 | 0.5035 | 0.5033 | 0.4737 |
| tf-idf(desc+plot) | 28 | 0.3321 | 0.5155 | 0.4442 | 0.4459 |
| bm25(desc+meta) | 28 | 0.5398 | 0.6514 | 0.6428 | 0.6508 |
| bm25(desc+plot) | 28 | 0.5505 | 0.6555 | 0.5882 | 0.6426 |
| dense(title+desc+meta) | 28 | 0.6425 | 0.7463 | 0.7352 | 0.7532 |
| dense(title+desc+plot) | 28 | 0.5962 | 0.6595 | 0.6668 | 0.6506 |
| dense(title-only) | 28 | 0.4473 | 0.5033 | 0.5462 | 0.5908 |
| hybrid(tfidf+dense) | 28 | 0.5766 | 0.7457 | 0.692 | 0.6125 |
| dense+rerank | 28 | 0.6525 | 0.7719 | 0.8016 | 0.7917 |
<!-- EVAL_V2_ABLATIONS_END -->

### Labels, pool, and spot-check

Candidates are the union of each system's top of the pool, including the ablations, shuffled without system names, ranks, scores, or v1 labels.

<!-- EVAL_V2_STATS_BEGIN -->
Pool depth 10 across 15 systems and 58 queries (3032 candidate slots; 21 to 76 unique titles per query).

Dev labels: 139 pairs graded 2 or 3 out of 1276 judged pairs (26 of 28 queries have a grade of 2 or 3).

Test labels: 86 pairs graded 2 or 3 out of 1756 judged pairs (29 of 30 queries have a grade of 2 or 3).

v2-relevant dev titles absent from that query's v1 relevant set: 87.

A second LLM reviewed **303 of 3,032 pairs**. The judges agreed on the exact grade for 83.2% and on relevance (grade 2 or 3) for 98.0%. Substituting the second grades on the sampled test cards gave a reranking gain of **+0.2587** versus BM25 (paired 95% interval +0.1670 to +0.3559).

Grade counts: 0=2096, 1=711, 2=40, 3=185. nDCG gain is 2^grade - 1. Recall and MRR count a title when its grade is at least 2.
<!-- EVAL_V2_STATS_END -->

### Second-judge findings

A second LLM scored all 303 original sample cards using the same 0–3 rubric, before comparing grades. The sample contains 116 development and 187 test pairs. Sampling used the original binary strata. The reviewer saw each query and catalog evidence with system, rank and original grade hidden from the review cards. The reviewer had previously seen project results and selected examples. Both judgments rely on language models and the same catalog evidence.

The judges agreed exactly on 252 of 303 cards (83.2%, quadratic-weighted kappa 0.867). At the operational relevance threshold of grade 2 they agreed on 297 (98.0%, binary kappa 0.864). Only 21 sampled cards met this threshold for the original judge; the second judge retained all 21 and added six. Every disagreement was one grade apart. Of the 51 disagreements, 44 were between 0 and 1, including 42 where the second judge gave partial credit. For example, the second judge gave another comic’s stand-up special a 1 for a John Mulaney query because it matched the format. Both judges treated that special as irrelevant for recall.

| Original grade ↓ / second grade → | 0 | 1 | 2 | 3 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 168 | 42 | 0 | 0 |
| 1 | 2 | 64 | 6 | 0 |
| 2 | 0 | 0 | 4 | 0 |
| 3 | 0 | 0 | 1 | 16 |

The table below substitutes second grades for the 187 sampled held-out pairs. The remaining 1,569 held-out pairs keep their original grades. All six frozen rankings are scored again against each query’s full judged pool. Intervals resample the 30 queries in pairs and condition on these labels. The direct comparison with the hybrid first stage is exploratory.

| Difference in mean nDCG@10 | Original grades | Sample substituted | Paired 95% interval after substitution | Wins / losses / ties |
| --- | ---: | ---: | --- | --- |
| Hybrid fusion minus BM25 | +0.0763 | +0.0645 | −0.0216 to +0.1517 | 17 / 10 / 3 |
| Hybrid + rerank minus BM25 | +0.2655 | +0.2587 | +0.1670 to +0.3559 | 27 / 3 / 0 |
| Hybrid + rerank minus its own first stage | +0.1892 | +0.1942 | +0.1109 to +0.2845 | 26 / 4 / 0 |

People-focused queries showed the largest descriptive mean gain over BM25 after substitution (+0.420, six wins in six queries). The other four intent groups each contain only six queries as well. These figures describe where to look in the explorer; the sample is too small to establish that one intent benefits more in general. The three queries where BM25 won concerned the Central Park case, the Lagos trafficking drama and the karate rivalry. In each original ranking BM25 already put the intended title first.

Recompute from the repository root with `python -m scripts.score_second_judge`. The report includes split-specific and per-system agreement, every disagreement, all six metric rows, and paired test results. Per-system sample counts overlap because methods can retrieve the same title. [Reviewed cards and reasons](data/eval_v2/spotcheck_second_judge.json) · [Audit results](results/eval_v2/second_judge_audit.json)

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
- **Small labeled splits, an LLM judge, and queries written for this project.** Useful for honesty, not a public IR benchmark. The review count appears above.
- **Grades are 0-3.** nDCG uses gain `2^grade - 1`. Recall and MRR use grades 2 and 3. Queries with no positive grade are omitted from nDCG. Queries with no grade of 2 or 3 are omitted from recall and MRR.
- **Short marketing blurbs.** Bad descriptions limit every method; metadata helps unevenly.
  Wikipedia plot enrichment (`text_plot`) is an optional ablation — match errors and missing
  pages remain a caveat (see coverage JSON).
- **CPU MiniLM + ms-marco CE are demo models.** Stronger embedders / larger cross-encoders would change ranks.
- **Hybrid is RRF, not learned fusion.** Rerank is greedy over a fixed top-50 pool.
## License

MIT — see [LICENSE](LICENSE). Netflix catalog © Netflix (public dump). Labels are original to this repo.
Wikipedia plot excerpts in `data/wikipedia_plots.jsonl` are **CC BY-SA 4.0** — see
[`data/WIKIPEDIA_ATTRIBUTION.md`](data/WIKIPEDIA_ATTRIBUTION.md).
