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

Dynamic experiments analyze the actual RPG decoding process for test users. This should probably be the strongest part of the extension, because it directly connects graph structure to recommendation performance.

The goal is not to implement every possible dynamic diagnostic at once. The first dynamic pass should focus on B1, B2, B4, and B6:

- B1: is the target item reached by graph decoding?
- B2: does extra budget add genuinely new regions, or mostly redundant neighbors?
- B4: if the target is reached, how many propagation steps are needed?
- B6: do these dynamic diagnostics saturate at the same time as recommendation performance?

After inspecting the first dynamic results, add one lightweight diagnostic:

- B7: does increasing RPG's coupled beam/search budget solve failures after reachability?

The implementation should save raw decoding traces first, then compute aggregate metrics from those traces. This is safer than saving only aggregates, because it lets us fix or redefine metrics later without rerunning decoding.

Important implementation rule:

- Tracing must not change decoding behavior.
- The first validation should compare normal RPG evaluation metrics against RPG evaluation with tracing enabled. If the metrics differ, the dynamic analysis is invalid.

### B1. Target Reachability

Question: does graph decoding ever reach the ground-truth next item?

For each test example, record the initial candidate items, the step-wise visited set, the final prediction, and the ground-truth item. Then check whether the target appears anywhere in the visited set.

Default inference setting:

- Use the released README Sports inference hyperparameters as the default dynamic-analysis setting: `num_beams = 100`, `n_edges = 30`, and `propagation_steps = 5`.
- Do not use the paper-appendix/perf setting as the default for B. It can be kept as a secondary comparison only if needed.

First sweep:

- `n_edges = [10, 20, 30, 50, 100]`
- `propagation_steps = 5`
- `num_beams = 100`
- `temperature = 0.03`

Optional second sweep:

- `n_edges = 30`
- `propagation_steps = [1, 2, 3, 5]`
- `num_beams = 100`
- `temperature = 0.03`

Per-example trace fields:

- example id
- user/session id if available
- target item
- budget config
- initial items
- visited items by step
- final visited set
- final prediction
- target reachable
- target first reached step
- target selected in final recommendation

Aggregate metrics:

- reachable target rate
- reachable target rate by budget
- target selected rate
- reachable-but-not-selected rate
- mean visited set size
- target first-reached-step distribution

Interpretation:

- If target reachability saturates early, then additional graph budget has limited room to improve recommendation metrics.
- If many targets are never reached, then the bottleneck is graph search or initial beam selection.
- If targets are reached but not selected, then the bottleneck is ranking/scoring rather than graph reachability.

### B2. Redundancy of Visited Candidates

Question: when the budget grows, are the newly visited items genuinely new semantic regions, or mostly redundant neighbors?

For each test example and budget, compare newly added items at each propagation step against the items already visited before that step.

Core metrics:

- number of new unique nodes discovered
- visited set size by step
- marginal new items by step
- mean RPG similarity between newly added items and previous visited items
- mean semantic-ID Hamming distance between newly added items and previous visited items
- unique semantic-ID prefixes covered by the visited set

Simple prefix choices:

- prefix length 1
- prefix length 2
- prefix length 4

Keep prefix analysis simple at first. It is only meant to indicate semantic-region diversity, not define a new clustering algorithm.

Interpretation:

- If the visited set grows but prefix diversity saturates, extra budget is expanding within the same semantic regions.
- If newly added items are highly similar to already visited items, extra budget is likely redundant.
- If new items become less similar and prefix diversity keeps growing, then saturation is less likely to be explained by local redundancy.

### B4. First-Hit / Path-Depth Analysis

Question: if the target is reachable, how many graph-expansion steps are needed?

This is related to B1 but focuses on depth. B1 asks whether the target is reached at all; B4 asks when it is first reached.

Core metrics:

- fraction of targets reached at step 0
- fraction first reached at step 1
- fraction first reached at step 2
- fraction first reached at step 3
- fraction first reached at step 5
- fraction never reached
- mean first reached step among reachable targets
- median first reached step among reachable targets

Optional metric:

- shortest graph distance from the initial candidate set to the target, if this is cheap to compute from the saved graph and trace.

The optional shortest-distance metric should not block the first implementation. The step-wise trace already gives the most important quantity: when RPG actually reaches the target under its real beam pruning.

Interpretation:

- If most reachable targets appear within the first one or two propagation steps, increasing propagation depth should have limited benefit.
- If many targets require deeper paths but are pruned before reaching them, beam pruning may be the real bottleneck.
- If targets are theoretically close in the graph but not reached dynamically, model scoring or beam pruning is steering search away from them.

### B6. Saturation Curves

Question: do dynamic diagnostics saturate at the same time as recommendation performance?

This is the synthesis experiment. For each budget setting, plot recommendation metrics and dynamic diagnostics together.

Budget axes:

- main axis: `n_edges = [10, 20, 30, 50, 100]`, with `propagation_steps = 5`
- secondary axis if needed: `propagation_steps = [1, 2, 3, 5]`, with `n_edges = 30`

Report per budget:

- Recall@K and/or NDCG@K
- reachable target rate
- mean visited set size
- mean new items per step
- target first-hit distribution
- redundancy metrics from B2
- semantic-prefix diversity

Interpretation:

- If performance, reachability, and diversity all saturate together, the graph-search explanation is strong.
- If performance saturates but reachability keeps increasing, the bottleneck is probably ranking/scoring after reachability.
- If visited set size keeps growing but reachability and diversity saturate, extra budget is mostly adding redundant candidates.
- If none of the dynamic diagnostics saturate, then the static graph explanation is probably incomplete.

### B7. Beam-Budget Diagnostic

Question: if we increase RPG's total beam/search budget, do reached targets become selected more often?

This is a small follow-up diagnostic, not another full dynamic sweep. Keep graph width and depth fixed, then sweep `num_beams`:

- dataset: `Sports_and_Outdoors`
- graph width: `n_edges = 100`
- propagation depth: `propagation_steps = 5`
- eval seed: `2024`
- test subset: fixed first `2000` users
- beam sizes: `num_beams = [50, 100, 200, 500]`

Important caveat: in the current RPG implementation, `num_beams` controls the number of random initial candidates, the number of frontier nodes expanded, the number of raw neighbors considered, and the number of candidates kept after each propagation step. Therefore B7 is a coupled beam-budget diagnostic, not a perfectly isolated pruning intervention.

For each user and beam size, classify the target into exactly one outcome bucket:

- `not_reached`: target never appears in the considered graph candidates
- `considered_never_in_beam`: target appears in candidates but never survives into the beam in that run
- `in_beam_not_selected`: target survives into the beam but is not in final top-k
- `selected`: target is in the final top-k

Report:

- bucket rates by `num_beams`
- target considered rate
- target in-beam rate
- target selected rate
- Recall@K and NDCG@K
- mean visited item count

Interpretation:

- If larger beams improve target considered/in-beam rates and Recall@K, then extra RPG beam/search budget helps.
- If larger beams improve considered/in-beam rates but Recall@K barely improves, simply spending more beam budget is not enough; final scoring/ranking is likely still limiting.
- If `in_beam_not_selected` grows with larger beams, targets survive into the beam more often but are still not ranked high enough.
- If bucket rates barely change, beam budget is probably not the right lever; score misalignment or local distractors are more likely.

### B Implementation Notes

Use one dynamic trace runner rather than separate scripts for each B experiment.

The raw trace should be saved under the graph-inspection/graph-analysis artifact tree, not mixed into normal evaluation outputs. A possible structure is:

```text
artifacts/rpg/graph_analysis/sports/<session>/
  graphs/
  static/
  dynamic/
    traces/
    summaries/
```

Suggested raw trace output:

```text
dynamic_traces.parquet
```

CSV is acceptable for the first implementation if arrays are stored in a simple JSON-string column, but Parquet is cleaner for nested trace fields.

Suggested aggregate outputs:

```text
dynamic_reachability_summary.csv
dynamic_redundancy_summary.csv
dynamic_first_hit_summary.csv
dynamic_saturation_summary.csv
pruning_summary.csv
```

The first implementation milestone should be:

1. Run RPG evaluation with tracing enabled.
2. Confirm recommendation metrics match normal RPG evaluation.
3. Save per-example visited-set traces.
4. Compute B1 reachability.
5. Add B2 redundancy and B4 first-hit metrics from the same traces.
6. Build B6 plots only after the trace fields are trusted.

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

The first implementation should be narrow:

1. Use Sports only.
2. Keep the static A outputs as the structural baseline.
3. Instrument graph propagation to record per-example visited sets and per-step frontiers.
4. Validate that tracing does not change RPG evaluation metrics.
5. Run the main `n_edges` sweep: `[10, 20, 30, 50, 100]` with README defaults for the other dynamic hyperparameters.
6. Compute B1, B2, B4, and B6 from saved traces.
7. Add query-trace visualizations only after the quantitative dynamic results are clear.

This keeps the extension aligned with the paper's Figure 6 while limiting scope.

## Open Questions

- Should the paper-appendix inference settings be included as a secondary comparison, or should B use only the released README settings?
- Should B reuse the prepared static top-100 graph, or should it build the decoding graph exactly through the upstream RPG path for every budget?
- How many eval seeds are needed for stable reachability estimates?
- Is Sports enough for the extension, or do we need one additional dataset after the method is stable?
- Should shortest-path analysis be included in B4, or is first-hit step from actual decoding enough?
- Should visualization be part of the core contribution or only supporting material?
