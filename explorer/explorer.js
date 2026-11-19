"use strict";

const names = {
  boolean: "Boolean", "tf-idf": "TF-IDF", bm25: "BM25",
  "dense(title+desc)": "Dense MiniLM", "hybrid(bm25+dense,meta)": "Hybrid RRF",
  "hybrid+rerank": "Hybrid + rerank",
};
const query = document.querySelector("#query");
const method = document.querySelector("#method");
const comparisons = benchmark.comparisons;
const rerank = "hybrid+rerank";
const initial = new URLSearchParams(location.search);

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}

function panel(item, system) {
  const group = node("section", undefined, "panel");
  group.append(node("h3", names[system] + " · " + item.scores[system].toFixed(3)));
  item.rankings[system].forEach((id, index) => {
    const card = item.cards[id];
    const details = node("details");
    const summary = node("summary");
    summary.append(node("span", index + 1, "rank"), node("span", card.title),
      node("span", "Grade " + card.grade, "grade"));
    details.append(summary, node("p", [card.type, card.release_year, card.listed_in].join(" · ")),
      node("p", card.description));
    if (card.cast) details.append(node("p", "Cast · " + card.cast));
    if (card.director) details.append(node("p", "Director · " + card.director));
    details.append(node("p", "Judge’s rationale · " + card.rationale));
    group.append(details);
  });
  return group;
}

function render() {
  const item = benchmark.queries[query.value];
  const delta = item.scores[method.value] - item.scores.bm25;
  document.querySelector("#delta").textContent = "nDCG@10 difference on this query · " +
    (delta > 0 ? "+" : "") + delta.toFixed(3);
  document.querySelector("#lists").replaceChildren(panel(item, "bm25"), panel(item, method.value));
}

for (const [id, item] of Object.entries(benchmark.queries).sort((a, b) => a[1].query.localeCompare(b[1].query))) {
  query.add(new Option(item.query, id));
}
for (const row of comparisons) method.add(new Option(names[row.method], row.method));
query.value = Object.hasOwn(benchmark.queries, initial.get("q")) ? initial.get("q") : "birmingham_gangsters";
method.value = comparisons.some(row => row.method === initial.get("method")) ? initial.get("method") : rerank;
for (const control of [query, method]) {
  control.addEventListener("change", () => {
    if (location.protocol !== "file:") {
      const url = new URL(location.href);
      url.searchParams.set("q", query.value);
      url.searchParams.set("method", method.value);
      history.replaceState(null, "", url);
    }
    render();
  });
}
for (const row of [...benchmark.systems].sort((a, b) => b["ndcg@10"] - a["ndcg@10"])) {
  const bar = node("div", undefined, row.method === rerank ? "bar best" : "bar");
  const track = node("div", undefined, "track");
  const fill = node("div", undefined, "fill");
  fill.style.width = (100 * row["ndcg@10"]) + "%";
  track.append(fill);
  bar.append(node("span", names[row.method]), track, node("span", row["ndcg@10"].toFixed(3), "value"));
  document.querySelector("#bars").append(bar);
}
const effect = comparisons.find(row => row.method === rerank);
const fusion = comparisons.find(row => row.method === "hybrid(bm25+dense,meta)");
document.querySelector("#effect").textContent = "Hybrid + rerank beat BM25 on " + effect.sign_n_positive +
  " of " + effect.n + " queries. Its average gain was " + effect.mean_difference +
  " nDCG@10 (paired 95% interval " + effect.ci_low + " to " + effect.ci_high + ").";
document.querySelector("#fusion").textContent = "Hybrid fusion alone gained " + fusion.mean_difference +
  " on average. Its interval (" + fusion.ci_low + " to " + fusion.ci_high + ") includes zero.";
document.querySelector("#review").textContent = "Human review · " + benchmark.review + ".";
render();
