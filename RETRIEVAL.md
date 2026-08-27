# Catalog retrieval case study

**Author:** Nikolaos (Nikos) Mavrapidis ([NikosMav](https://github.com/NikosMav))

Five-minute walkthrough for hiring managers. This is **catalog text search** over a public Netflix titles dump — title + description only. It is **not** a production recommender, **not** collaborative filtering, **not** TESSI, and **not** RAG-over-the-web (no LLM generation step).

The older EDA + Boolean/TF-IDF notebook remains: [`netflix_data_analysis.ipynb`](netflix_data_analysis.ipynb). This document covers the **retrieval product** next to it.

## Problem

Given a natural-language query, rank Netflix catalog rows by text similarity to `title + description`. Compare classical sparse methods to local dense embeddings on the **same** labeled queries.

## Method

| Method | Representation | Scoring |
|--------|----------------|---------|
| **Boolean** | Binary bag-of-words (uni+bigrams) | Set Jaccard |
| **TF-IDF** | Weighted sparse terms (uni+bigrams, 10k features) | Cosine |
| **Dense** | `sentence-transformers/all-MiniLM-L6-v2` (CPU) | Cosine NN |
| **Hybrid** | TF-IDF + dense rank lists | Reciprocal Rank Fusion (k=60) |

**Ablation:** dense on `title` only vs `title + description`.

## How to run

```bash
git clone https://github.com/NikosMav/DataAnalysis-Netflix.git
cd DataAnalysis-Netflix
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Query the catalog (no paid API)
python -m retrieval query "war between vietnam and usa" --method dense --top-k 10
python -m retrieval query "feel-good cooking competition show" --method hybrid

# Regenerate the metrics table from a clean checkout
python -m retrieval eval --failures
# → results/eval_metrics.json  (+ qualitative_examples.json)
```

First dense run downloads MiniLM (~80MB) and embeds ~7.8k titles (a few minutes on CPU). Embeddings cache under `.cache/embeddings/`.

Optional OpenAI embeddings: `DenseRetriever(backend="openai")` + `OPENAI_API_KEY` (not required).

## Results (28 hand-labeled queries)

Labels: [`data/labeled_queries.json`](data/labeled_queries.json) — author judgments with `relevant_show_ids` from this dump. Binary relevance. Metrics recomputed by `python -m retrieval eval`.

| method | recall@5 | recall@10 | ndcg@5 | ndcg@10 | mrr |
| --- | --- | --- | --- | --- | --- |
| boolean | 0.3159 | 0.4012 | 0.3185 | 0.3527 | 0.4440 |
| tf-idf | 0.4502 | 0.5446 | 0.4555 | 0.4912 | 0.5013 |
| dense(title+desc) | 0.5849 | 0.6209 | 0.5710 | 0.5656 | 0.6304 |
| dense(title-only) | 0.4241 | 0.4499 | 0.4286 | 0.4269 | 0.5081 |
| hybrid(tfidf+dense) | 0.5059 | 0.6922 | 0.4941 | 0.5626 | 0.5853 |

Numbers match [`results/eval_metrics.json`](results/eval_metrics.json) from the last local eval run.

### What the numbers mean

- **Recall@k** — fraction of labeled relevant titles found in the top-k.
- **MRR** — how early the first relevant title appears (1 = rank 1).
- **nDCG@k** — ranking quality with binary gains (order matters).

### Takeaways

1. **Dense (title+description) wins early precision / MRR** on this set — best Recall@5, nDCG@5, MRR.
2. **Hybrid wins Recall@10** — RRF recovers sparse lexical hits dense misses, and vice versa. Hybrid is not always best at ranks 1–5 on paraphrase queries (e.g. cooking competition: dense alone is cleaner early; RRF can re-surface TF-IDF near-misses like `Cook Off`).
3. **Title-only embeddings are weaker** than title+description (descriptions carry the topical signal).
4. **TF-IDF beats Boolean**; Boolean is a coarse baseline, not a production sparse ranker.
5. Gains are real but **modest and set-specific** — 28 author-labeled queries, not a public IR benchmark.

## Ablations & failure cases

### Dense wins (paraphrase)

Query: `feel-good cooking competition show`  
Dense returns competition cooking shows (`The Chefs' Line`, `Crazy Delicious`, baking contests). Boolean/TF-IDF often latch onto isolated tokens (`feel`, `cook`) and rank unrelated titles.

Query: `dark crime thriller set in Scandinavia`  
Dense surfaces Nordic crime (`The Valhalla Murders`, `Bordertown`). Sparse returns generic US crime (`Longmire`, `Brooklyn's Finest`) — “Scandinavia” rarely appears verbatim.

### Sparse / exact tokens win

Query: `war between vietnam and usa`  
TF-IDF and hybrid place Vietnam War titles high via distinctive tokens (`vietnam`, `war`). Dense is competitive but can insert topical near-misses (`V Wars`).

Query: `Kevin Hart stand-up comedy special` / `Stranger Things`  
Exact name/title tokens favor sparse overlap; dense still works but is not uniquely necessary.

### Mickey Mouse (ambiguous lexical match)

Query: `Mickey Mouse`  
Sparse ranks other “mouse” titles (`Tip the Mouse`, `Danger Mouse`). Dense ranks `Walt Disney Animation Studios Short Films Collection` (mentions Mickey) at #2 — still noisy, but semantically closer than pure token overlap.

Full side-by-side dumps: [`results/qualitative_examples.json`](results/qualitative_examples.json).

## Repo map

| Path | Role |
|------|------|
| `retrieval/` | Package + `python -m retrieval` CLI |
| `data/netflix_titles.csv` | Catalog |
| `data/labeled_queries.json` | Eval labels (show_ids) |
| `results/` | Regenerated metrics + qualitative JSON |
| `netflix_data_analysis.ipynb` | Original EDA + Boolean/TF-IDF showcase |
| `netflix_dense_retrieval.ipynb` | Optional walkthrough notebook |

## Limitations

- **Catalog search ≠ recommender.** No watch history, no CF, no popularity re-rank.
- **Not TESSI / not web RAG.** No chunking of long docs, no tool-using agent, no generation.
- **Tiny labeled set.** 28 author queries; useful for honesty, not SOTA claims.
- **Binary labels only.** No graded relevance → nDCG uses 0/1 gains.
- **Short marketing blurbs.** Bad descriptions limit every method.
- **CPU MiniLM is a demo embedder.** Stronger models / cross-encoders would change ranks.
- **Hybrid is RRF, not learned fusion.**

## License

MIT — see [LICENSE.md](LICENSE.md). Netflix catalog © Netflix (public dump). Labels are original to this repo.
