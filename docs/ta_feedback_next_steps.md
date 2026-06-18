# TA Feedback: Next Steps

This note summarizes what the TA/supervisor feedback implies for the graph-analysis extension. It is a planning document, not a finalized protocol.

## Huge Next Directions

These are larger research directions. They are scientifically interesting, but probably too large to treat as small fixes to the current notebook.

### 1. Compare Against SASRec

The TA suggested moving some RPG design choices into SASRec to understand where the gains come from.

Motivation:

```text
RPG mixes several ideas: SID tokenization, shared token embeddings, MTP loss, and graph decoding.
SASRec gives a simpler baseline where these components can be added one by one.
```

Possible comparison ladder:

- `SASRec`: standard item embeddings and standard scoring.
- `SASRec + SID/shared embeddings`: replace item embeddings with SID-token-based item representations.
- `SASRec + MTP-style loss`: predict semantic-ID tokens instead of item IDs.
- `SASRec + graph decoding`: use a similar item graph to accelerate prediction.
- `RPG`: full method.

Main research questions:

- Is the gain from semantic-ID tokenization?
- Is shared embedding useful?
- Is MTP loss important compared with standard cross-entropy or sampled negative ranking loss?
- Can graph decoding accelerate a SASRec-style model while maintaining performance?

### 2. Efficient Graph Decoding Beyond RPG

The TA pointed out that graph methods usually decouple:

- number of initial seeds
- neighbor count `k`
- beam/search width
- total visited-node budget

In current RPG, `num_beams` controls several of these at once. A larger direction would be to design or test a cleaner graph-search procedure where these knobs are separated.

Main research question:

```text
Can we preserve recommendation quality while visiting far fewer nodes?
```

This direction should report both:

- accuracy metrics such as Recall@10 and NDCG@10
- efficiency metrics such as visited nodes, memory, and latency

## Next Analysis Steps

These are the most useful immediate analyses to make the current conclusion cleaner.

### 1. Brute-Force All-Item RPG Scoring Baseline

This is the most important next analysis.

Current dynamic analysis shows:

```text
The target is often reached, but not selected.
```

The TA suggested disentangling this into two independent questions:

```text
1. Does the model give high score to the true target item?
2. Does graph decoding find the highest-scored items under that same score?
```

The brute-force baseline answers this by scoring all items with RPG's existing semantic-token scoring rule, without graph decoding or beam search.

Compare:

- brute-force all-item Recall@10/NDCG@10
- graph-decoding Recall@10/NDCG@10
- memory usage
- latency

Interpretation:

- If brute force is much better than graph decoding, graph search is losing high-scored items.
- If brute force is also weak, the scoring function itself is the bottleneck.

### 2. Strong Connectivity Check

The current largest-component plot uses weak/undirected connectivity with `igraph`.

The TA noted that strong connectivity is often considered for directed graph search indexes. Therefore, add a directed strong-component analysis.

Report:

- number of strongly connected components
- largest strongly connected component size/fraction
- weak largest component size/fraction beside it
- maybe source/sink component counts in the condensation graph

Interpretation:

- Weak connectivity says the graph is not globally fragmented if direction is ignored.
- Strong connectivity says whether directed traversal can move between regions in both directions.

### 3. Local Intrinsic Dimension of Trained Item Representations

The TA suggested estimating local intrinsic dimension to understand whether long SIDs bring high-dimensional information.

Candidate representations:

- RPG SID-token-mean item embeddings
- possibly graph-construction token-based item vectors
- SASRec item embeddings, if used later

Possible simple metrics:

- local PCA dimension needed to explain 90% or 95% variance among nearest neighbors
- participation ratio
- two-nearest-neighbor intrinsic dimension estimator

Interpretation:

- High local intrinsic dimension would support the idea that long SIDs encode rich item structure.
- Low local intrinsic dimension would suggest redundancy or locally low-dimensional geometry.

## Fixes To The Current Analysis

These should improve the current notebook and claims without changing the overall direction.

## Easy Fixes

### 1. Separate Reachability and Ranking Metrics in Plots

The TA noted that reachability rate and ranking metrics should not share the same y-axis.

Fix:

- Put reachability in one subplot.
- Put Recall@10/NDCG@10 in another subplot.
- Avoid implying they are on the same scale.

### 2. Rename B7 as Beam-Budget Diagnostic

Current B7 sweeps `num_beams`, but in RPG this changes multiple things:

- initial random pool size
- frontier size
- number of raw candidates considered
- number of candidates kept after scoring

Therefore it should not be called a pure pruning experiment.

Use:

```text
B7. Beam-Budget Diagnostic
```

Interpretation should be:

```text
Increasing RPG's coupled beam/search budget improves target access, but does not materially improve Recall@10.
```

Do not claim:

```text
Pruning is causally not the bottleneck.
```

### 3. Clarify Ground Truth Definition

In reachability plots, ground truth means:

```text
the actual held-out/test target item for the user
```

It does not mean:

```text
nearest item to the user hidden state
```

Add this sentence near the B1 plot.

### 4. Clarify Re-Ranking Wording

The current plots do not include an additional reranker.

Clarify:

```text
Current analysis uses standard RPG graph decoding.
Reranking is only a proposed future direction over the candidate pool.
```

### 5. Clarify Cosine Similarity Trial

The quick reranking trial used a naive cosine similarity:

```text
normalized final user hidden state vs normalized SID-token-mean item embedding
```

The TA noted this is not expected to work well because RPG's MTP training uses token-logit/dot-product-style scoring, so the cosine trial is inconsistent with the training objective.

## Less Easy Fixes

### 1. Add Strong-Component Static Metrics

This is a real analysis addition, but it should be lightweight because `igraph` already supports directed components.

Need to decide:

- use full directed top-k graph after removing self-edges
- compute strong components for each `k`
- report beside the existing weak/undirected component plot

### 2. Add Brute-Force All-Item Scoring

This is the key next experiment but requires implementation and a compute run.

Need to implement:

- all-item RPG semantic-token scoring
- batched scoring over all items
- Recall@K/NDCG@K computation
- latency and memory logging

Need to compare against:

- current graph-decoding metrics
- possibly different `n_edges` budgets

### 3. Better Candidate-Ranking Diagnostic

After brute-force scoring exists, analyze:

- target global rank under brute-force RPG score
- whether graph decoding visited the brute-force top-k items
- overlap between graph-decoding top-k and brute-force top-k
- whether graph decoding misses high-scored items or finds them but does not keep them

This directly answers the TA's decomposition:

```text
model scoring quality vs graph decoding quality
```

### 4. Local Intrinsic Dimension Analysis

This requires selecting the item representation and estimator.

Start simple:

- use SID-token-mean item embeddings
- compute nearest neighbors
- estimate local PCA dimension or participation ratio

Then decide whether it is useful enough to include.

## Current Priority

The most useful next action is:

```text
Implement brute-force all-item RPG scoring and compare it with graph decoding.
```

This directly addresses the TA's main point: disentangle whether saturation comes from weak model scores or from graph decoding failing to recover the highest-scored items.
