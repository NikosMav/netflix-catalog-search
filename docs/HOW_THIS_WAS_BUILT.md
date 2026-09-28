# How this was built

Most of the catalog-search code in this repository was written by AI coding assistants, mainly Cursor cloud agents, under Nikos's direction. Six of the seven merged pull requests are on `cursor/` branches, and five of those include a Cursor cloud-agent footer. The other pull request, on a `codex/` branch, added package metadata and replaced the README. Before that, the history is a 2023 Colaboratory notebook on the public Netflix titles dump. Those pull requests kept the notebook and added the retrieval package beside it.

Nikos set the scope: offline catalog text search over that titles dump, explicitly not a recommender. He chose which methods to compare. They landed across those pull requests as Boolean retrieval, TF-IDF, dense MiniLM embeddings, hybrid reciprocal rank fusion, BM25, metadata fields, a cross-encoder reranker, and a Wikipedia plot-text ablation. The last merged pull request added the local Streamlit UI.

He required that every reported number be regenerated from the committed eval JSON by script. `python -m retrieval eval` writes `results/eval_metrics.json`, and `scripts/sync_metrics_docs.py` copies that JSON into the README and `RETRIEVAL.md`. The unit-test workflow checks that the docs still match the file.

He also decided to publish negative and mixed results. The Wikipedia plot enrichment pull request is titled as mixed results and was merged that way: extra plot text did not help every method. Metadata-field comparisons that did not help were merged too. Each pull request was reviewed before merge, with AI assistance, under his sign-off.
