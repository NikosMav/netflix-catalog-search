# Netflix catalog EDA + sparse text retrieval

Early (2023) work by **Nikolaos (Nikos) Mavrapidis** ([NikosMav](https://github.com/NikosMav)): exploratory analysis of a public Netflix titles dump, then a small **Boolean / bag-of-words vs TF-IDF** experiment on `title + description`.

This is **EDA + classical sparse text retrieval**. It is **not** a production recommender, **not** collaborative filtering, **not** embeddings / vector search / RAG, and **not** a TESSI project. It sits honestly on a path toward retrieval engineering: first understand a catalog and score short text with sparse features; dense retrieval comes later.

Originally written as a university data-mining assignment; reframed here so a hiring manager can clone it, run it, and read the claims at face value.

## What’s in the notebook

| Section | What it does |
|--------|----------------|
| **Cleaning** | Fill missing director/cast/country/date/rating with documented defaults |
| **EDA** | Movies vs TV, yearly mix, countries, genres, cast, ratings/age bands, monthly adds, directors, seasons, IMDb-joined top-rated movies |
| **Sparse retrieval** | BoW + Hamming / Jaccard-style vs TF-IDF + cosine; title→neighbors and free-text→neighbors demos |

## Data

Shipped under [`data/`](data/):

- `netflix_titles.csv` — Netflix Movies and TV Shows catalog ([Kaggle / Shivam Bansal](https://www.kaggle.com/shivamb/netflix-shows), via [TidyTuesday](https://github.com/rfordatascience/tidytuesday/tree/master/data/2021/2021-04-20))
- `imdb_ratings_netflix_join.csv` — slim title∩IMDb ratings join from the [IMDb non-commercial datasets](https://developer.imdb.com/non-commercial-datasets/) (see [`data/README.md`](data/README.md))

## How to run

```bash
git clone https://github.com/NikosMav/DataAnalysis-Netflix.git
cd DataAnalysis-Netflix

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook netflix_data_analysis.ipynb
# or: jupyter nbconvert --to notebook --execute netflix_data_analysis.ipynb
```

Or open the notebook in Colab via the badge at the top (paths assume the `data/` folder from this repo).

**Runtime note:** building dense BoW/TF-IDF frames and full pairwise similarity matrices needs on the order of a few GB of RAM and a couple of minutes on a laptop. The free-text Boolean query path is slower than TF-IDF (row-wise scores); a few demo queries are fine.

## What you learn

- How a real catalog looks after light cleaning (and how fill-ins bias charts)
- How **Boolean/BoW** overlap differs from **TF-IDF + cosine** on short marketing descriptions
- Why short, sparse text struggles under coarse Boolean features (qualitative — see notebook examples)
- How classical sparse retrieval relates to later embedding/RAG work: same *retrieve by similarity* idea, different representation

## Limitations (read these)

- **No ranking metrics.** Neighbor lists are judged by eye on a few queries. No Precision@K, nDCG, or user study.
- **Text only.** Genre, cast, and popularity are unused in the ranker.
- **Boolean precompute ≠ Jaccard query.** Precomputed neighbors use Hamming similarity on the count matrix; the free-text helper uses `jaccard_score`. Both are “Boolean-ish,” not identical.
- **IMDb join is fuzzy.** Matching on title strings produces collisions and misses; the top-rated chart is exploratory.
- **Not production.** Re-fitting a vectorizer on query+corpus is a notebook convenience, not how a real inverted index works.

### Sparse retrieval vs embeddings / RAG

TF-IDF here is **sparse text retrieval**: weighted term vectors + cosine. Dense **embeddings** + vector search (and LLM/RAG pipelines) reuse the retrieve-by-similarity pattern with continuous representations and usually a separate generation step. This repo stops at the sparse step on purpose.

## License

MIT — see [LICENSE.md](LICENSE.md). Netflix catalog © Netflix (public dump via Kaggle). IMDb data subject to [IMDb non-commercial terms](https://developer.imdb.com/non-commercial-datasets/).
