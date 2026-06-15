# Graph Analysis Extension Plan

This is the current working plan for a possible extension of the RPG reproduction project. It is not a fixed experimental protocol, and it should not be treated as proven correct. It is the best idea so far, based on the supervisor's suggestion and the current understanding of the codebase and paper. We should refine it element by element before implementing or reporting results.

## Motivation

RPG performs graph-constrained decoding over a sparse item-item graph built from semantic-ID token embeddings. In the paper's Figure 6, performance appears to saturate quickly once a modest inference budget is used. A natural question is whether this saturation can be explained by the structure of the constructed graph and by how decoding explores that graph.

The tentative research question is:

> Why does RPG's graph-constrained decoding saturate after a small resource budget?

The extension should ideally connect graph properties to recommendation behavior, not only describe the graph in isolation.

## Experiment Block A: Static Graph Analysis

Static experiments analyze the constructed item-item graph independently of any user query.

For the first static pass, use a narrow and reproducible setup:

- Dataset: `Sports_and_Outdoors` only.
- Graph source: cached original-pool flat graph, top-100, if possible.
- Effective graph sizes: slice the cached top-100 graph to `k = 10, 20, 30, 50, 100`.
- Node set: exclude padding item `0`; analyze real item IDs only.
- Self-edges: exclude self-edges from edge-level metrics if they appear in the adjacency.
- Directed graph: use directed edges for similarity, Hamming distance, reciprocity, and hubness.
- Graph library: use `igraph` for graph-level metrics such as connected components and clustering.
- Similarity definition: use the same RPG graph-construction similarity based on trained semantic-token embeddings, not raw text embeddings.
- Random baseline: sample random item pairs with the same number of pairs as graph edges, using fixed random seeds such as `2024, 2025, 2026`.

### A1. Neighbor Similarity

Measure the similarity distribution of graph edges and compare it against random item pairs.

Question: are graph neighbors meaningfully more similar than random nodes?

Design:

- For each directed edge `i -> j`, compute RPG item-item similarity between `i` and `j`.
- Compare against random directed pairs `(i, j)` with `i != j`.
- Use all graph edges for Sports if feasible.
- For each effective `k`, report mean, median, p10, and p90 for graph edges and random pairs.
- Plot graph-edge similarity against random-pair similarity.

Interpretation:

- If graph-edge similarity is much higher than random-pair similarity, the constructed graph is semantically coherent.
- If similarity drops sharply as `k` increases, larger decoding budgets may add weaker neighbors.

### A2. Semantic-ID Distance

Measure token Hamming distance between each item and its graph neighbors.

Question: does the graph mostly connect items that differ in only a few semantic-ID digits?

Design:

- For each directed edge `i -> j`, compute Hamming distance between the semantic-ID token sequence of item `i` and item `j`.
- Report both raw Hamming distance and normalized Hamming distance, `distance / m`, where `m` is semantic-ID length.
- Compare against random directed item pairs.
- Exact token IDs are sufficient for equality checks; codebook-local IDs are not necessary for this metric.
- For each effective `k`, report mean, median, p10, and p90.
- Plot Hamming distance distributions or mean normalized Hamming distance vs `k`.

Interpretation:

- Low graph-edge Hamming distance means graph neighbors share many semantic-ID tokens.
- If Hamming distances are high but RPG similarity is high, the graph may be driven more by learned token embedding similarity than by exact token overlap.

### A3. Reciprocity

In the directed top-k graph, measure how often `i -> j` also implies `j -> i`.

Question: is the graph locally symmetric, or does it contain many one-way neighbor relations?

Design:

- For each directed edge `i -> j`, check whether `j -> i` is also present in the same effective top-`k` graph.
- Reciprocity is `reciprocal directed edges / total directed edges`.
- Also compute node-level reciprocity: for each node, the fraction of its outgoing edges that are reciprocated.
- Exclude self-edges.
- Compute for `k = 10, 20, 30, 50, 100`.
- Plot global reciprocity vs `k` and optionally the node-level reciprocity distribution.

Interpretation:

- High reciprocity means local neighborhoods are stable and symmetric.
- Low reciprocity suggests hub effects or asymmetric nearest-neighbor structure.

### A4. Hubness

Count how often each item appears in other items' neighbor lists.

Question: are a few items overrepresented as graph hubs?

Design:

- Compute in-degree in the directed top-`k` graph.
- Since out-degree is approximately fixed by `k`, in-degree variation captures hubness.
- Exclude self-edges.
- For each effective `k`, report mean in-degree, standard deviation, max in-degree, Gini coefficient, and top-1% share of incoming edges.
- Compare the in-degree distribution against a random directed graph baseline with the same number of nodes and edges.
- Plot the in-degree distribution, preferably with a log-scale y-axis.

Interpretation:

- Strong hubness means graph propagation may repeatedly enter central semantic areas.
- This could help explain performance saturation if extra budget visits redundant hubs rather than diverse useful items.

### A5. Connected Components

Convert the graph to an undirected graph and compute component sizes.

Question: is most of the item pool in one giant component, or is the graph fragmented?

Design:

- For each effective `k`, build the directed top-`k` graph from the cached adjacency.
- Exclude padding item `0`.
- Remove self-edges.
- Convert the directed graph to an undirected graph by adding an undirected edge `{i, j}` if either `i -> j` or `j -> i` exists.
- Use `igraph` to compute connected components on the undirected graph.
- Do not use strong connected components as the main result, because strong connectivity is too strict for a directed kNN-style graph.
- Compute this for `k = 10, 20, 30, 50, 100`.

Report:

- directed density of the top-`k` graph after removing self-edges
- number of undirected edges after symmetrization
- undirected density after symmetrization
- number of connected components
- largest connected component size
- largest connected component fraction
- second-largest connected component size
- number of isolated nodes
- median component size
- p90 component size
- fraction of nodes in components with size `< 10`

Interpretation:

- If the largest connected component is already close to the full graph at small `k`, then global fragmentation is unlikely to explain performance saturation.
- If many small components remain at low `k`, random initial beams may fail when the relevant item is located in another component.
- This is a global static metric only. It does not prove query-level reachability, because RPG only propagates for a few steps.

### A6. Clustering

Estimate local clustering coefficient or triangle rate.

Question: do neighbors of a node also tend to be neighbors of each other?

Design:

- Use the same undirected symmetrized graph as A5.
- Use `igraph` to compute local clustering coefficients for all real item nodes.
- Use unweighted clustering for the first version.
- Compute this for `k = 10, 20, 30, 50, 100`.
- Compare against a simple random undirected graph baseline with the same number of nodes and undirected edges, using fixed random seeds such as `2024, 2025, 2026`.

Report:

- average clustering coefficient
- median clustering coefficient
- p10 clustering coefficient
- p90 clustering coefficient
- fraction of nodes with clustering coefficient equal to zero
- random-baseline average clustering coefficient
- clustering lift over random baseline, if the random baseline is not near zero

Interpretation:

- High clustering means graph neighborhoods are locally redundant and semantically tight.
- This would support the saturation hypothesis: additional propagation steps may keep exploring the same local semantic neighborhoods rather than discovering diverse new regions.
- Low clustering would suggest that saturation is less likely to come from local graph redundancy and may instead come from model scoring, beam pruning, or query-level reachability limits.

### A7. Popularity Bias

Compare in-degree or hubness against item frequency in the training data.

Question: does the graph structure favor popular items?

Popularity means how often an item appears in the training interactions:

```text
train_frequency(item) = number of times item appears in the training split
```

Use the training split only, because this is the data available to the model and graph construction process. Do not use validation or test labels to define popularity.

Design:

- For each effective `k`, build the directed top-`k` graph.
- Exclude padding item `0`.
- Remove self-edges.
- Compute graph in-degree for every item.
- Compute `train_frequency` for every item by counting item occurrences in RPG's training split.
- Compare `train_frequency` against graph in-degree.
- Compute this for `k = 10, 20, 30, 50, 100`.

Report:

- Spearman correlation between `train_frequency` and graph in-degree.
- Pearson correlation between `log1p(train_frequency)` and `log1p(in_degree)`.
- Mean graph in-degree by simple training-frequency bucket.
- A scatter plot of `log1p(train_frequency)` vs `log1p(in_degree)`.
- A top-hub summary for the top 1% highest in-degree items.

Use these simple training-frequency buckets:

```text
0-5
6-10
11-20
21-50
51+
```

For each bucket, report:

- number of items
- mean training frequency
- mean in-degree
- median in-degree
- p90 in-degree
- max in-degree

For the top 1% highest in-degree items, report:

- mean training frequency
- median training frequency
- share of total training interactions covered by those items
- comparison against all items

Interpretation:

- Strong positive correlation means graph hubs tend to be popular training items.
- Weak correlation means graph hubness is probably not mainly popularity-driven.
- If top hubs cover a large share of training interactions, graph propagation may be biased toward popular semantic regions.
- Popularity bias could help explain saturation if additional propagation budget repeatedly expands through popular hubs rather than diverse long-tail regions.

## Experiment Block B: Dynamic / Query-Conditioned Analysis

Dynamic experiments analyze the actual RPG decoding process for test users. This should probably be the main contribution, because it directly connects graph structure to recommendation performance.

### B1. Target Reachability

For each test user, measure whether the ground-truth next item is reachable from the randomly initialized beams within `q` graph-propagation steps.

Default inference setting:

- Use the released README Sports inference hyperparameters as the default dynamic-analysis setting: `num_beams = 100`, `n_edges = 30`, and `propagation_steps = 5`.
- The paper-appendix/perf setting, `num_beams = 10`, `n_edges = 100`, and `propagation_steps = 2`, can be kept as a secondary comparison only if needed.

Potential variables:

- `q`: propagation steps, for example `0, 1, 2, 3, 5`
- `k`: graph neighbors per node, for example `10, 20, 50, 100, 200`
- `b`: beam size, for example `10, 20, 50, 100`
- eval seed, because RPG's initial beams are random

Possible outputs:

- reachable target rate vs `q`
- reachable target rate vs `k`
- reachable target rate vs `b`
- NDCG@10 vs reachable target rate
- performance when target is reachable vs unreachable

### B2. Marginal Utility Per Propagation Step

For each propagation step, measure what new information is gained.

Possible metrics:

- number of new unique nodes discovered
- overlap with previously visited nodes
- average model score of newly discovered nodes
- fraction of final top-10 items first discovered at this step
- fraction of target items first discovered at this step

This is one of the most direct ways to explain performance saturation. If later steps add many nodes but almost no useful high-scoring nodes, then the graph search budget has diminishing returns.

### B3. Visited-Set Oracle

Separate graph-search failure from model-ranking failure.

For each user:

- Was the target item visited?
- If visited, what was the target's rank among visited nodes?
- What Recall@10 would be possible with an oracle reranker over only visited nodes?

Interpretation:

- target not visited: graph/search budget failure
- target visited but ranked low: model scoring failure

### B4. Score Gap Analysis

Compare scores among visited and unvisited items.

Possible metrics:

- best model score among visited nodes
- best model score among all items, if feasible for the original item pool
- target item score
- final recommended item scores
- score gap between best visited and best global item

Question: does graph decoding quickly reach the high-scoring part of the item space?

### B5. Seed Sensitivity

Repeat reachability and visited-set analysis across multiple eval seeds.

Question: how much of RPG's metric variance comes from random initial beam selection?

## Visualization Ideas

Full-graph visualizations are likely to be unreadable for Sports, because the graph has about 18k nodes and up to 100 outgoing edges per node. Visualization should therefore be selective.

Potential visualizations:

- ego-net plots for selected items
- query-trace plots showing initial beams, step-wise visited nodes, target item, and final top-10 recommendations
- UMAP or t-SNE layout of item representations with sampled graph edges overlaid
- community-level visualization after clustering the undirected graph
- histograms for edge similarity, in-degree, clustering coefficient, and semantic-ID distance

The most useful visualization is probably a query-trace plot, because it makes the decoding process interpretable.

## Current Graph Availability

Normal RPG training and evaluation do not save the graph. The upstream model builds `model.adjacency` in memory and discards it when the process exits.

The repo-owned performance profiling path does save adjacency caches for Sports under:

```text
artifacts/rpg/perf/sports/graphs/
```

The currently available cached graph for the original Sports item pool is:

```text
artifacts/rpg/perf/sports/graphs/rpg_sports_and_outdoors_pool18357_backend-flat_topk-100_seed-2024_783be92f6501.pt
```

There are also enlarged-pool HNSW graph caches for Sports with pool sizes `20k`, `50k`, `100k`, `200k`, and `500k`.

I did not find saved graph caches for Beauty, Toys, or CDs. I also did not find saved eval-time traces such as initial beams, visited nodes, or per-step frontiers. Those would need to be instrumented if we pursue the dynamic analysis.

For Sports, the saved top-100 graph can likely support analyses for smaller `k` values by taking the first `k` neighbors per row, assuming the cached adjacency preserves sorted neighbor order.

## Recommended First Version

The first implementation should probably be narrow:

1. Use Sports only.
2. Use the cached original-pool top-100 graph if possible.
3. Compute static graph summaries: neighbor similarity, reciprocity, hubness, connected components, and clustering.
4. Instrument or reimplement graph propagation to record per-user visited sets and per-step frontiers.
5. Run target reachability and marginal utility analysis for a small grid of `q`, `k`, and `b`.
6. Add one or two query-trace visualizations only after the quantitative results are clear.

This keeps the extension aligned with the paper's Figure 6 while limiting scope.

## Open Questions

- Should the paper-appendix inference settings be included as a secondary dynamic-analysis comparison, or should we keep only the released README settings?
- Should the dynamic analysis use the upstream dense graph or the repo-owned cached flat graph?
- How many eval seeds are enough for stable reachability estimates?
- Do we need all datasets, or is Sports enough because Figure 6 is based on Sports?
- Should visualization be part of the core contribution or only supporting material?
