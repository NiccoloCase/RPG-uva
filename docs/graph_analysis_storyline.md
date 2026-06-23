# RPG graph analysis: evidence, filtering, and possible storyline

This file is a working synthesis of the graph-analysis results across the four datasets. It is not a final paper section. The goal is to separate three steps:

1. Extract what we measured.
2. Decide what is worth reporting.
3. Only then formulate the research question and narrative.

## 1. What was analyzed

All results below come from the saved graph-analysis artifacts:

| Dataset | Session | Items | SID digits | Max graph k |
|---|---:|---:|---:|---:|
| Beauty | `20260618T220235Z_job24012690` | 12,102 | 32 | 200 |
| CDs and Vinyl | `20260618T192046Z_job24010304` | 64,444 | 64 | 500 |
| Sports and Outdoors | `20260615T153949Z_job23882755` | 18,358 | 16 | 100 |
| Toys and Games | `20260621T090036Z_jobchain` | 11,925 | 16 | 100 |

The comparable cross-dataset experiments are:

- Static graph structure: edge similarity, SID Hamming distance, weak components, clustering, reciprocity, hubness, popularity correlation.
- Dynamic decoding behavior: whether the ground-truth item is reached, whether it is selected, how many items are visited, and how much candidate duplication appears.
- Brute-force scorer comparison: same RPG scoring computation over all items, used as an upper bound for graph decoding under the same scorer.
- Inference profiling: median runtime and peak CUDA memory for graph decoding versus brute-force all-item scoring.

Sports has additional exploratory diagnostics:

- Beam-budget diagnostic, previously called pruning analysis.
- Novelty / visited-masked traversal.
- Naive reranking and visited-pool reranking.

Those Sports-only diagnostics are useful for interpretation, but they are not comparable across datasets and should not be central in the main claim.

## 2. Extracted evidence

### 2.1 Static graph structure

At the smallest and largest graph budgets, the largest weak component contains all nodes for every dataset. The graph is therefore not globally fragmented. The largest-component number uses the undirected/weak view of the directed kNN graph.

| Dataset | k | Largest weak comp. | Edge cosine | Random cosine | Norm. SID Hamming | Random norm. Hamming | Reciprocity | Clustering lift | Indegree Gini |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Beauty | 20 | 100.0% | 0.574 | 0.501 | 0.870 | 0.996 | 74.9% | 73.9x | 0.222 |
| Beauty | 200 | 100.0% | 0.535 | 0.501 | 0.950 | 0.996 | 93.2% | 2.6x | 0.065 |
| CDs | 50 | 100.0% | 0.543 | 0.508 | 0.948 | 0.996 | 79.7% | 24.6x | 0.169 |
| CDs | 500 | 100.0% | 0.531 | 0.508 | 0.966 | 0.996 | 88.5% | 2.4x | 0.102 |
| Sports | 10 | 100.0% | 0.637 | 0.501 | 0.743 | 0.996 | 63.0% | 324.4x | 0.313 |
| Sports | 100 | 100.0% | 0.569 | 0.501 | 0.883 | 0.996 | 87.2% | 12.9x | 0.121 |
| Toys | 10 | 100.0% | 0.631 | 0.501 | 0.752 | 0.995 | 63.7% | 206.5x | 0.307 |
| Toys | 100 | 100.0% | 0.563 | 0.501 | 0.894 | 0.995 | 85.7% | 7.2x | 0.130 |

Interpretation:

- The graph is connected in the weak sense even at the smallest tested k.
- Edges are consistently more similar than random item pairs.
- SID Hamming distances are much lower than random, especially for Sports and Toys with 16-digit SIDs.
- Reciprocity increases with k, meaning many nearest-neighbor relations become mutual at larger graph budgets.
- Clustering lift is high at small k and decreases as k grows, which is expected: small-k neighborhoods are more local; large-k neighborhoods become less selective.
- Hubness is present at small k but becomes milder at larger k.

### 2.2 Dynamic graph decoding

Increasing graph budget strongly increases target reachability and visited items, but the final recommendation metric improves much less.

| Dataset | n_edges | Reachable | Selected / Recall@10 | NDCG@10 | Mean visited | Final duplicate candidates | Final novelty |
|---|---:|---:|---:|---:|---:|---:|---:|
| Beauty | 20 | 16.7% | 5.20% | 3.06% | 714 | 49.3% | 48.7% |
| Beauty | 200 | 68.3% | 7.83% | 4.54% | 6,202 | 37.9% | 32.6% |
| CDs | 50 | 15.2% | 4.67% | 2.74% | 2,229 | 44.6% | 14.8% |
| CDs | 500 | 50.9% | 7.16% | 4.05% | 18,856 | 34.2% | 2.7% |
| Sports | 10 | 26.5% | 3.82% | 2.21% | 1,700 | 63.9% | 11.0% |
| Sports | 100 | 78.0% | 4.43% | 2.55% | 11,224 | 56.3% | 0.04% |
| Toys | 10 | 39.7% | 7.53% | 4.28% | 2,860 | 48.1% | 30.1% |
| Toys | 100 | 96.1% | 8.62% | 4.84% | 11,095 | 61.9% | 1.5% |

Interpretation:

- Reachability grows a lot with graph budget.
- Recall@10 grows only modestly and then saturates.
- This creates the central tension: the graph can often reach the target, but reaching it does not imply selecting it.
- Final novelty becomes very low for Sports, Toys, and CDs at large k, suggesting late propagation mostly revisits already-explored regions or very redundant neighborhoods.

### 2.3 Brute-force RPG scoring comparison

This is the most important diagnostic for deciding whether graph search is the bottleneck. The brute-force baseline scores every item with the same RPG scorer and reports the top-k. It is not an oracle; it is the upper bound of the current RPG scorer if graph search missed no candidates.

| Dataset | n_edges | Graph Recall@10 | Brute-force Recall@10 | Loss vs BF | Graph NDCG@10 | BF NDCG@10 | Reachable | BF top-10 missed by graph | Graph/BF top-10 overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Beauty | 20 | 5.23% | 7.85% | 2.62 pp | 3.06% | 4.55% | 17.0% | 3.85% | 43.4% |
| Beauty | 200 | 7.83% | 7.85% | 0.02 pp | 4.54% | 4.55% | 68.3% | 0.08% | 96.3% |
| CDs | 50 | 4.61% | 7.41% | 2.80 pp | 2.71% | 4.19% | 15.0% | 3.78% | 43.2% |
| CDs | 500 | 7.16% | 7.41% | 0.25 pp | 4.05% | 4.19% | 50.9% | 0.59% | 80.6% |
| Sports | 10 | 3.81% | 4.43% | 0.62 pp | 2.22% | 2.55% | 26.7% | 1.09% | 72.9% |
| Sports | 100 | 4.43% | 4.43% | 0.002 pp | 2.55% | 2.55% | 78.0% | 0.006% | 99.8% |
| Toys | 10 | 7.59% | 8.62% | 1.03 pp | 4.33% | 4.84% | 39.8% | 1.72% | 76.6% |
| Toys | 100 | 8.62% | 8.62% | 0.00 pp | 4.84% | 4.84% | 96.1% | 0.00% | 100.0% |

Interpretation:

- At large graph budgets, graph decoding almost exactly matches brute-force RPG scoring on Beauty, Sports, and Toys.
- CDs still has a small gap, but it is much smaller than the low-budget gap.
- Therefore, for these checkpoints and datasets, the main performance ceiling is not graph connectivity or graph search at sufficiently large k. The ceiling is mostly the RPG scorer/model itself.
- This explains why increasing graph budget gives limited gains: once the graph recovers the same top items as brute-force RPG scoring, more graph traversal cannot exceed the scorer's own upper bound.

### 2.4 Inference profiling

This experiment compares median runtime per user and peak CUDA allocation for graph decoding versus vectorized brute-force all-item scoring under the graph-analysis implementation.

| Dataset | n_edges | Graph ms/user | BF ms/user | Graph slowdown | Graph peak alloc. | BF peak alloc. | Graph Recall@10 | BF Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Beauty | 20 | 1.07 | 0.31 | 3.4x | 0.25 GB | 0.25 GB | 5.23% | 7.85% |
| Beauty | 200 | 2.20 | 0.31 | 7.0x | 0.27 GB | 0.25 GB | 7.83% | 7.85% |
| CDs | 50 | 1.92 | 0.58 | 3.3x | 0.51 GB | 2.10 GB | 4.70% | 7.41% |
| CDs | 500 | 6.27 | 0.58 | 10.8x | 0.73 GB | 2.10 GB | 7.15% | 7.41% |
| Sports | 10 | 1.66 | 0.25 | 6.6x | 0.15 GB | 0.18 GB | 3.81% | 4.43% |
| Sports | 100 | 4.31 | 0.25 | 17.2x | 0.16 GB | 0.18 GB | 4.43% | 4.43% |
| Toys | 10 | 1.42 | 0.25 | 5.8x | 0.14 GB | 0.14 GB | 7.51% | 8.62% |
| Toys | 100 | 3.48 | 0.25 | 14.2x | 0.15 GB | 0.14 GB | 8.62% | 8.62% |

Interpretation:

- In this implementation and item-count regime, vectorized brute-force all-item scoring is faster per user than graph decoding.
- CDs is the exception in memory: brute-force uses substantially more peak CUDA memory than graph decoding.
- This should be framed carefully. It does not prove graph decoding is always inefficient; it shows that the current graph decoding path is not an obvious speed win under this profiling setup.
- The result is still useful because it changes the engineering question: if graph decoding does not improve accuracy over brute force and is slower here, then the current graph traversal implementation is not the main path to better recommendation quality.

### 2.5 Sports-only exploratory diagnostics

Beam-budget diagnostic:

- Increasing `num_beams` from 50 to 500 raises target-considered rate from 60.3% to 98.9%.
- Recall@10 barely changes: 4.0% to 4.1%.
- This is not a clean pruning-only causal test because `num_beams` also changes the number of seeds, expanded frontier nodes, and explored regions.
- The safe interpretation is that larger beam/search budget improves coverage but not selection.

Novelty / visited-masked traversal:

- Visited-masked traversal increases reachability but badly hurts Recall@10 on Sports.
- This suggests that simply forcing novelty breaks the learned scoring/traversal dynamics.
- It should not be a main result unless expanded and redesigned.

Naive reranking:

- Naive cosine reranking on Sports did not improve metrics.
- Visited-pool reranking produced essentially no change in Recall@10.
- These are useful negative results, but they are too preliminary to report as a main experiment.

## 3. What is worth reporting

### Main results worth reporting

1. Static graph health.

Report the largest weak component, edge cosine versus random, SID Hamming versus random, reciprocity, clustering lift, and indegree Gini. These directly answer whether the constructed graph is pathological. The answer is no: it is connected, locally structured, reciprocal, and not dominated by extreme hubness at large k.

2. Reachability versus ranking saturation.

Report reachability, selected rate / Recall@10, and mean visited items as k increases. This directly addresses the original motivation from Figure 6: increasing graph resources makes many more targets reachable, but performance saturates quickly.

3. Brute-force RPG scorer upper bound.

This is the strongest result. It shows that high-budget graph decoding nearly matches all-item RPG scoring. Therefore, the remaining performance limitation is mostly not graph search, but the scorer/model ranking quality.

4. Runtime and memory as an engineering caveat.

Report the profiling result, but carefully. It is not the main scientific claim. It shows that in this graph-analysis implementation, graph decoding is slower per user than vectorized brute-force scoring, while sometimes using less memory.

### Secondary / appendix-only results

- Detailed histograms of edge similarity, Hamming distance, and indegree.
- Popularity buckets and popularity-hubness correlations.
- First-hit step summaries.
- Redundancy curves beyond a compact novelty/duplicate summary.
- Sports-only beam-budget diagnostic.
- Sports-only novelty and reranking attempts.

These results can support the discussion, but they should not dominate the main story because they either duplicate the main evidence or are not cross-dataset.

### Results probably not worth reporting in the main text

- Raw item-level metrics.
- Sample traces, except maybe one qualitative example.
- The naive cosine reranking result as a proposed solution.
- The visited-masked novelty result as a proposed solution.
- A causal pruning conclusion from B7. It should only be described as a beam/search-budget diagnostic.
- Large tables with many mixed units. They obscure the story and should be replaced by small focused tables.

## 4. Hypotheses and what the evidence says

### H1: Performance saturates because the graph is structurally poor.

Evidence:

- Largest weak component is 100% for all datasets and tested budgets.
- Edge similarity is consistently higher than random.
- SID Hamming distances are consistently lower than random.
- Clustering and reciprocity are high.

Conclusion:

- Mostly rejected. The graph does not look globally broken or random.

### H2: Performance saturates because the graph cannot reach the correct item.

Evidence:

- Reachability is low at small budgets.
- Reachability becomes high for Sports and Toys, moderate for Beauty, and improves substantially for CDs at large budgets.
- However, selected rate / Recall@10 remains much lower than reachability.

Conclusion:

- Partly true at low graph budgets, but not enough to explain saturation at larger budgets.

### H3: Performance saturates because reached items are not ranked into the final top-k.

Evidence:

- Reachability can be 68% to 96%, while Recall@10 remains 4% to 9%.
- Sports beam-budget diagnostic shows target-considered rate can approach 99%, while Recall@10 stays around 4%.

Conclusion:

- Supported. Reaching the target is not sufficient; the scoring/ranking stage is the main bottleneck once budget is large enough.

### H4: Graph search is the main bottleneck compared with the RPG scorer's all-item upper bound.

Evidence:

- At large k, graph decoding nearly matches brute-force RPG scoring: zero or near-zero Recall@10 loss for Beauty, Sports, and Toys; small loss for CDs.
- Graph/BF top-10 overlap reaches 96% to 100% on Beauty, Sports, and Toys, and 81% on CDs.

Conclusion:

- Mostly rejected. At sufficient graph budget, graph search reproduces the scorer's own top-k. The limiting factor is the scorer/model, not the graph.

### H5: More graph compute is an efficient way to improve results.

Evidence:

- Runtime increases with k.
- The final metric saturates quickly.
- In this profiling setup, graph decoding is slower than vectorized brute-force scoring.
- Final novelty becomes very low at large k for several datasets.

Conclusion:

- Not supported. More graph traversal gives diminishing returns and may be computationally inefficient in the current implementation.

## 5. Proposed final research question

The research question that best fits the evidence is:

> Why does increasing RPG graph-decoding resources give only limited recommendation gains: is the bottleneck graph structure/reachability, or the downstream RPG scoring and ranking once candidates are reachable?

A more paper-style version:

> We analyze whether the saturation of RPG graph decoding is caused by limitations of the constructed item graph, by failure to reach relevant items during decoding, or by the ranking capacity of the RPG scorer itself.

## 6. Proposed storyline

The original motivation is the saturation behavior: after a modest graph budget, giving RPG more graph resources yields limited gains.

First, inspect the graph itself. If the graph were fragmented, random, or dominated by hubs, then saturation could be explained as a graph-construction failure. The static analysis does not support that explanation. Across all four datasets, the graph is weakly connected, locally structured, more similar than random, more SID-consistent than random, and increasingly reciprocal as k grows.

Second, inspect decoding dynamics. Larger graph budgets do what they are supposed to do: they reach many more items and visit much larger candidate sets. However, Recall@10 increases only modestly. This separates reachability from selection. The target may be present somewhere in the explored graph region, but the model often does not rank it into the final recommendation list.

Third, compare graph decoding against brute-force scoring with the same RPG scorer. This is the key control experiment. At large graph budgets, graph decoding almost matches all-item RPG scoring on Beauty, Sports, and Toys, and is close on CDs. Therefore, the graph is not leaving much performance on the table once k is large enough. The performance ceiling is mostly the RPG scorer's own ranking quality.

Finally, look at efficiency. Increasing k increases runtime and often explores increasingly redundant neighborhoods. Under the current profiling setup, graph decoding is slower per user than vectorized brute-force scoring, although it can use less memory on the largest dataset. This suggests that future work should not simply allocate more budget to graph traversal. The more promising direction is to improve the scorer/ranking objective, or redesign efficient decoding only if it preserves the brute-force scorer's top-k at much lower cost.

## 7. Main takeaways

1. The graph is not obviously broken.

The constructed graphs are connected in the weak sense, locally clustered, reciprocal, and semantically/SID structured relative to random baselines.

2. Reachability improves with graph budget, but recommendation quality saturates.

This is the central empirical pattern. More graph search finds more of the graph, but final Recall@10 does not increase proportionally.

3. The scorer, not the graph, appears to be the main performance ceiling at large budgets.

The brute-force RPG scorer gives almost the same Recall@10 as high-budget graph decoding. If all-item scoring itself only reaches roughly 4% to 9% Recall@10, graph traversal cannot exceed that ceiling.

4. Larger graph budgets are not an attractive standalone solution.

They increase visited items and runtime, while gains shrink quickly. The graph already recovers the scorer's preferred top-k at sufficient k.

5. The best next scientific direction is model/scorer analysis, not more graph traversal.

Examples: compare against SASRec-style scoring, test whether SID tokenization or embedding sharing helps, isolate the MTP objective, and evaluate whether a graph index can accelerate another scorer without changing its predictions.

## 8. Suggested reporting structure

### Section 1: Motivation

Start from the saturation observation: performance stops improving much after a moderate resource budget.

### Section 2: Is the graph structurally pathological?

Report the static graph table/plots. Conclude that graph construction is not the obvious failure mode.

### Section 3: Does more graph budget improve reachability?

Report reachability and visited items versus Recall@10. Conclude that reachability improves much faster than final ranking quality.

### Section 4: Is graph search still losing good candidates?

Report brute-force RPG scoring versus graph decoding. Conclude that high-budget graph decoding is already close to the scorer upper bound.

### Section 5: Efficiency caveat

Report runtime/memory as an engineering diagnostic, not as the central paper claim. Conclude that the current implementation does not show an efficiency win over vectorized all-item scoring in these datasets, although memory may favor graph decoding for larger item sets.

### Section 6: Implications

State that the most promising future work is to improve the scorer/ranking side or to transfer graph-style efficient decoding to stronger scorers, rather than just increasing RPG graph budget.

## 9. Concrete next steps

1. Keep the main cross-dataset figures small:

- Static graph structure: one compact table or 2-3 panels.
- Reachability versus Recall@10: separate y-axes or separate panels.
- Brute-force RPG upper bound: one table/plot showing loss versus BF.
- Runtime/memory: one caveated efficiency table.

2. Move noisy diagnostics to appendix:

- Histograms.
- Popularity buckets.
- First-hit distributions.
- Sports-only B7/novelty/rerank.

3. Tighten terminology:

- Say "largest weak component" for the current component analysis.
- Say "beam/search-budget diagnostic", not "pruning causal test", for B7.
- Say "brute-force RPG scorer upper bound", not "oracle".
- Say "naive cosine reranking" for the failed reranking attempt, not generic reranking.

4. If adding one new experiment, prefer scorer/model comparison:

- Same scorer, all-item scoring already tells us graph search is not the large-budget bottleneck.
- The next open question is why the scorer upper bound is low.
- A SASRec/RPG comparison or objective/embedding-sharing ablation would target that question more directly than another graph-budget sweep.
