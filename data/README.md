# Data

Files used by the notebooks and the `retrieval` package.

## `netflix_titles.csv`

Public **Netflix Movies and TV Shows** catalog compiled by [Shivam Bansal on Kaggle](https://www.kaggle.com/shivamb/netflix-shows). This copy matches the [TidyTuesday 2021-04-20](https://github.com/rfordatascience/tidytuesday/tree/master/data/2021/2021-04-20) mirror (~7.8k rows).

Columns include `show_id`, `type`, `title`, `director`, `cast`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in`, `description`.

## `imdb_ratings_netflix_join.csv`

Slim join for the “top-rated on Netflix” chart. Built from the official [IMDb non-commercial datasets](https://developer.imdb.com/non-commercial-datasets/) (`title.basics.tsv.gz` + `title.ratings.tsv.gz`): keep movie/TV rows whose `primaryTitle` or `originalTitle` appears in this Netflix snapshot, then attach `averageRating`.

| Column | Meaning |
|--------|---------|
| `title` | IMDb primary title (join key vs Netflix `title`) |
| `year` | Start year |
| `genre` | IMDb genres (`\|`-separated) |
| `weighted_average_vote` | IMDb `averageRating` (name kept for continuity with the 2023 Kaggle IMDb dump) |
| `num_votes` | IMDb vote count |
| `title_type` | e.g. `movie`, `tvSeries` |
| `tconst` | IMDb id |

**Caveats:** title-only matching is imperfect; multiple IMDb rows can share a name; this file is for exploratory charts, not a gold ranking. Redistribute only under IMDb’s non-commercial terms.

### Rebuild (optional)

```bash
curl -fsSL -o /tmp/title.basics.tsv.gz https://datasets.imdbws.com/title.basics.tsv.gz
curl -fsSL -o /tmp/title.ratings.tsv.gz https://datasets.imdbws.com/title.ratings.tsv.gz
# then re-run the join logic used to create this CSV (see repo history / notebook notes)
```

## `labeled_queries.json`

Hand-authored retrieval evaluation set (**28 queries**) with binary `relevant_show_ids` from `netflix_titles.csv`. Used by `python -m retrieval eval`. Author judgments for catalog search — not crowd-sourced, not personalization labels.
