"""Local catalog search UI.

Run from the repo root:

    pip install -e ".[ui]"
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# CPU-only. Set before any sentence-transformers import.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Netflix catalog search", layout="centered")

from retrieval.catalog import load_catalog, show_id_to_index
from retrieval.cli import _build_retriever
from retrieval.evaluate import RERANK_CANDIDATE_K
from retrieval.explain import explain_results, highlight_html, how_it_works, metric_notes

METHODS = [
    ("hybrid-rerank", "Hybrid + rerank"),
    ("hybrid-bm25", "Hybrid (BM25 + dense)"),
    ("dense", "Dense (MiniLM)"),
    ("bm25", "BM25"),
    ("tfidf", "TF-IDF"),
    ("boolean", "Boolean"),
    ("dense-rerank", "Dense + rerank"),
    ("hybrid", "Hybrid (TF-IDF + dense)"),
]
METHOD_IDS = {label: method_id for method_id, label in METHODS}
SLOW_METHODS = {"dense", "hybrid", "hybrid-bm25", "dense-rerank", "hybrid-rerank"}
BLURBS = {
    "hybrid-rerank": (
        f"Cross-encoder on the top {RERANK_CANDIDATE_K} from BM25 + dense "
        "(title, description, genres, cast, director, and country)."
    ),
    "hybrid-bm25": (
        "Reciprocal rank fusion of BM25 and MiniLM on title, description, "
        "genres, cast, director, and country."
    ),
    "dense": "MiniLM cosine similarity on title and description.",
    "bm25": "Okapi BM25 on title and description.",
    "tfidf": "TF-IDF cosine similarity on title and description.",
    "boolean": "Jaccard overlap of binary terms on title and description.",
    "dense-rerank": (
        f"Cross-encoder on the top {RERANK_CANDIDATE_K} MiniLM hits. "
        "The first stage reads title and description; the cross-encoder also reads metadata."
    ),
    "hybrid": "Reciprocal rank fusion of TF-IDF and MiniLM on title and description.",
}
CHART_LABELS = {
    "recall@5": "R@5",
    "recall@10": "R@10",
    "ndcg@10": "nDCG@10",
    "mrr": "MRR",
}
TABLE_LABELS = {
    "recall@5": "R@5",
    "recall@10": "R@10",
    "ndcg@5": "nDCG@5",
    "ndcg@10": "nDCG@10",
    "mrr": "MRR",
}
DESC_LIMIT = 280


@st.cache_resource(show_spinner=False)
def get_catalog() -> pd.DataFrame:
    return load_catalog()


@st.cache_resource(show_spinner=False)
def get_retriever(method_id: str):
    return _build_retriever(method_id, get_catalog(), show_progress=False)


def shorten(text: str, limit: int = DESC_LIMIT) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    cut = collapsed[:limit].rsplit(" ", 1)[0]
    return (cut or collapsed[:limit]).rstrip() + "…"


def meta_line(row: pd.Series) -> str:
    parts: list[str] = []
    kind = str(row.get("type", "") or "").strip()
    if kind and kind.lower() != "nan":
        parts.append(kind)
    if "release_year" in row.index and not pd.isna(row["release_year"]):
        parts.append(str(int(row["release_year"])))
    genres = str(row.get("listed_in", "") or "").strip()
    if genres and genres.lower() != "nan":
        parts.append(genres)
    return " · ".join(parts)


def render_card(row: pd.Series, explanation, relevant: bool = False) -> None:
    title = highlight_html(str(row.get("title", "")), explanation.highlight_terms)
    description = shorten(str(row.get("description", "") or ""))
    body = highlight_html(description, explanation.highlight_terms) if description else "No description."
    with st.container(border=True):
        st.markdown(
            f"<div style='font-weight:600'>{explanation.rank}. {title}</div>",
            unsafe_allow_html=True,
        )
        meta = meta_line(row)
        if meta:
            st.caption(meta)
        if relevant:
            st.caption("Labeled relevant")
        st.markdown(f"<p style='margin:0.35rem 0 0.2rem'>{body}</p>", unsafe_allow_html=True)
        st.caption(explanation.score_text())
        with st.expander("Why this result?"):
            st.markdown(explanation.summary)
            for line in explanation.details:
                st.markdown(f"- {line}")


def _search(method_id: str, query: str, top_k: int):
    catalog = get_catalog()
    if method_id in SLOW_METHODS:
        message = (
            f"Building the index for {len(catalog):,} titles. "
            "The first dense or rerank search downloads MiniLM and can take a few minutes."
        )
    else:
        message = "Searching the catalog…"
    with st.spinner(message):
        retriever = get_retriever(method_id)
        hits = retriever.query(query, top_k=top_k)
        explanations = explain_results(retriever, query, hits)
    return hits, explanations


def render_result_list(query: str, method_id: str, hits: pd.DataFrame, explanations, relevant_ids=None) -> None:
    st.caption(BLURBS[method_id])
    if hits is None or len(hits) == 0:
        st.info("No titles matched that query.")
        return
    if len(explanations) != len(hits):
        st.error("Could not explain every result.")
        return
    catalog = get_catalog()
    id_map = show_id_to_index(catalog)
    relevant_ids = relevant_ids or set()
    for i, explanation in enumerate(explanations):
        sid = str(hits.iloc[i]["show_id"])
        row = catalog.iloc[id_map[sid]]
        render_card(row, explanation, relevant=sid in relevant_ids)


def render_search() -> None:
    with st.form("search_form"):
        query = st.text_input("Query", placeholder="war between vietnam and usa")
        method_label = st.selectbox("Method", [label for _method_id, label in METHODS], index=0)
        top_k = st.slider("Results", min_value=1, max_value=20, value=10)
        submitted = st.form_submit_button("Search", type="primary")

    if submitted:
        text = query.strip()
        if not text:
            st.warning("Enter a query.")
        else:
            method_id = METHOD_IDS[method_label]
            try:
                hits, explanations = _search(method_id, text, int(top_k))
            except Exception as exc:
                st.error(f"Search failed: {exc}")
            else:
                st.session_state["search_result"] = {
                    "query": text,
                    "method_id": method_id,
                    "hits": hits,
                    "explanations": explanations,
                }

    saved = st.session_state.get("search_result")
    if saved:
        st.subheader(f"Results for “{saved['query']}”")
        render_result_list(saved["query"], saved["method_id"], saved["hits"], saved["explanations"])


def _v2_metrics() -> dict:
    path = ROOT / "results" / "eval_v2" / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _v2_queries(split: str) -> list[dict]:
    path = ROOT / "data" / "eval_v2" / f"{split}_labels.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["queries"]


def _label_spotcheck(sample_id: str, grade: int) -> None:
    import importlib.util

    path = ROOT / "scripts" / "eval_v2_spotcheck.py"
    spec = importlib.util.spec_from_file_location("eval_v2_spotcheck", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.label_pair(sample_id, grade)


def render_chart(frame: pd.DataFrame) -> None:
    import altair as alt

    chart_keys = [key for key in CHART_LABELS if key in frame.columns]
    if not chart_keys:
        return
    default = chart_keys.index("ndcg@10") if "ndcg@10" in chart_keys else 0
    chosen = st.selectbox(
        "Chart",
        chart_keys,
        index=default,
        format_func=lambda key: CHART_LABELS[key],
    )
    chart_df = pd.DataFrame({"method": frame["method"], "value": frame[chosen].astype(float)})
    methods = chart_df["method"].tolist()
    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("value:Q", title=CHART_LABELS[chosen]),
            y=alt.Y("method:N", sort=methods, title=None),
            tooltip=[
                alt.Tooltip("method:N", title="method"),
                alt.Tooltip("value:Q", title=CHART_LABELS[chosen], format=".4f"),
            ],
        )
        .properties(height=max(220, 24 * len(methods)))
    )
    st.altair_chart(chart, width="stretch")


def render_labeled_query(split: str) -> None:
    st.subheader("Try a labeled query")
    st.caption(
        "Relevant titles are the v2 labels for this split. "
        "This list is that method's ranking."
    )
    queries = _v2_queries(split)
    labels = [f"{item['id']}: {item['query']}" for item in queries]
    selected = st.selectbox("Labeled query", labels, key=f"eval_query_{split}")
    bm25_index = next(i for i, (method_id, _label) in enumerate(METHODS) if method_id == "bm25")
    method_label = st.selectbox(
        "Method",
        [label for _method_id, label in METHODS],
        index=bm25_index,
        key="eval_method",
    )
    if st.button("Show ranking"):
        item = next(query for query in queries if f"{query['id']}: {query['query']}" == selected)
        method_id = METHOD_IDS[method_label]
        try:
            hits, explanations = _search(method_id, item["query"], top_k=10)
        except Exception as exc:
            st.error(f"Search failed: {exc}")
        else:
            st.session_state["eval_run"] = {
                "split": split,
                "query": item["query"],
                "method_id": method_id,
                "relevant": [str(sid) for sid in item["relevant_show_ids"]],
                "hits": hits,
                "explanations": explanations,
            }

    saved = st.session_state.get("eval_run")
    if not saved or saved.get("split") != split:
        return
    relevant = set(saved["relevant"])
    hits = saved["hits"]
    shown = {str(sid) for sid in hits["show_id"].tolist()} if len(hits) else set()
    n_hit = sum(1 for sid in saved["relevant"] if sid in shown)
    st.markdown(f"**{saved['query']}**")
    st.caption(f"{n_hit} of {len(saved['relevant'])} labeled relevant titles appear in these results.")
    missed = [sid for sid in saved["relevant"] if sid not in shown]
    if missed:
        catalog = get_catalog()
        id_map = show_id_to_index(catalog)
        names = []
        for sid in missed:
            idx = id_map.get(sid)
            names.append(str(catalog.iloc[idx]["title"]) if idx is not None else sid)
        st.caption("Not in these results: " + "; ".join(names))
    render_result_list(
        saved["query"],
        saved["method_id"],
        hits,
        saved["explanations"],
        relevant_ids=relevant,
    )


def _format_cell(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return json.dumps(value)
    return str(value)


def _format_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for column in display.columns:
        if column == "method":
            continue
        display[column] = display[column].map(_format_cell)
    return display


def render_eval() -> None:
    st.subheader("Evaluation")
    payload = _v2_metrics()
    split = st.radio("Split", ["test", "dev"], horizontal=True, key="eval_split")
    block = payload[split]
    systems = pd.DataFrame(block["systems"])
    n_scored = int(systems["n_scored"].iloc[0])
    st.markdown(metric_notes(n_scored))
    if payload.get("ndcg_gain"):
        st.caption(
            f"nDCG gain is {payload['ndcg_gain']}. "
            f"Recall and MRR count grade {payload.get('relevant_if_grade_at_least', 2)} or higher."
        )
    st.caption(
        f"{split} split: {int(block['n_queries'])} queries, {n_scored} scored. "
        f"Judge: {payload['judge']}. "
        "Source: results/eval_v2/metrics.json. The app does not recompute these averages."
    )
    show = systems[["method", "n_scored", *CHART_LABELS.keys()]]
    st.dataframe(
        _format_metrics(show).rename(columns=TABLE_LABELS),
        hide_index=True,
        width="stretch",
        height=36 * (len(show) + 1),
    )
    render_chart(systems)
    if split == "test":
        st.subheader("Difference vs BM25")
        comparisons = pd.DataFrame(block["comparisons_vs_bm25"])
        keep = [
            "method",
            "n",
            "mean_difference",
            "ci_low",
            "ci_high",
            "sign_p_value",
            "sign_n_positive",
            "sign_n_negative",
            "sign_n_ties",
        ]
        st.dataframe(_format_metrics(comparisons[keep]), hide_index=True, width="stretch")
        st.caption(
            "Paired bootstrap interval for the mean nDCG@10 difference "
            "(system minus BM25) on the scored test queries, and a two-sided sign test."
        )
    else:
        st.subheader("Dev ablations")
        ablations = pd.DataFrame(block["ablations"])
        show_ablations = ablations[["method", "n_scored", *CHART_LABELS.keys()]]
        st.dataframe(
            _format_metrics(show_ablations).rename(columns=TABLE_LABELS),
            hide_index=True,
            width="stretch",
            height=36 * (len(show_ablations) + 1),
        )
    render_labeled_query(split)


def _human_labels() -> set[str]:
    path = ROOT / "data" / "eval_v2" / "spotcheck_human.json"
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["sample_id"] for row in payload.get("labels", [])}


def render_spotcheck() -> None:
    st.subheader("Spot-check")
    sample = json.loads((ROOT / "data" / "eval_v2" / "spotcheck_sample.json").read_text(encoding="utf-8"))
    labeled = _human_labels()
    pending = [pair for pair in sample["pairs"] if pair["sample_id"] not in labeled]
    st.caption(f"Status: {sample['status']}. {len(labeled)} labeled, {len(pending)} still open.")
    st.caption("This page does not show the model judgment.")
    if not pending:
        st.info("The sample is labeled. Run python scripts/eval_v2_agreement.py for agreement and kappa.")
        return
    card = pending[0]
    st.markdown(f"**Query:** {card['query']}")
    year = card.get("release_year", "")
    st.markdown(f"**{card['title']}**")
    st.caption(f"{card.get('type', '')} · {year} · {card.get('listed_in', '')}")
    if card.get("director"):
        st.caption(f"Director: {card['director']}")
    if card.get("cast"):
        st.caption(f"Cast: {card['cast']}")
    st.write(card.get("description") or "")
    st.caption(f"sample_id: {card['sample_id']}")
    st.caption("Grade 0 not relevant, 1 marginal, 2 relevant, 3 highly relevant.")
    columns = st.columns(4)
    for grade, column in enumerate(columns):
        if column.button(f"Grade {grade}", key=f"spot_grade_{grade}"):
            _label_spotcheck(card["sample_id"], grade)
            st.rerun()


def render_how() -> None:
    st.subheader("How it works")
    st.markdown(how_it_works())


def main() -> None:
    st.markdown(
        """
        <style>
        mark { background: #fff3bf; color: inherit; padding: 0 0.12em; border-radius: 2px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Netflix catalog search")
    st.caption("Offline text search over the public Netflix titles catalog.")
    page = st.sidebar.radio("Page", ["Search", "Evaluation", "Spot-check"])
    if page == "Search":
        render_search()
    elif page == "Evaluation":
        render_eval()
    else:
        render_spotcheck()
    render_how()


main()
