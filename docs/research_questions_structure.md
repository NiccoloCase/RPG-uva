# Research questions and storyline

This is a compact working structure for the paper/presentation. The goal is to connect reproducibility, baseline fairness, fairness-oriented metrics, graph analysis, brute-force checks, and performance into one coherent narrative.

## Overall thesis

RPG is a strong and interesting semantic-ID recommender, but its empirical story is more nuanced than the original paper suggests. The reproduction shows that results depend on evaluation details and baseline tuning. The graph analysis shows that the constructed graph is meaningful and can recover the RPG scorer's preferred candidates, but final performance is mostly limited by scoring/ranking rather than graph reachability.

## Research questions

```text
RQ0: Can we reproduce RPG's reported accuracy and efficiency claims under the released setup, and how do these claims change under stronger baseline tuning?

RQ1: How does RPG compare to tuned sequential recommendation baselines, especially SASRec, under accuracy and fairness-oriented metrics?

RQ2: What structural properties does the RPG item graph exhibit, and do these properties explain why performance saturates as graph-decoding budget increases?

RQ3: Is RPG's performance primarily limited by graph reachability/search, or by the scoring and ranking capacity of the model once candidates are reachable?

RQ4: Does graph decoding provide practical latency and memory advantages over alternative decoding strategies, including brute-force scoring with the same RPG scorer?
```

## RQ0: Reproducibility and baseline validity

**Question.** Can we reproduce RPG's reported accuracy and efficiency claims, and how do they change under stronger baseline tuning?

**Why it matters.** This establishes whether the original empirical claims are stable enough to build on. It also motivates the rest of the project: if results depend on protocol details or baseline strength, then a deeper audit is useful.

**Hypotheses.**

- RPG is approximately reproducible, but exact one-run equality is unlikely because graph decoding is stochastic.
- Some discrepancies come from protocol details, not necessarily implementation bugs.
- The original SASRec comparison may be weaker than a properly tuned or matched SASRec baseline.

**Evidence to show.**

- RPG reproduced metrics versus paper metrics.
- Dataset accounting issues, especially the Sports user/item and interaction-count mismatch.
- Seed/eval variability from stochastic graph decoding.
- SASRec results under released and stronger/matched hyperparameters.

**Current answer.** RPG is broadly reproducible, but exact numbers depend on stochastic graph decoding and evaluation details. The SASRec comparison is sensitive to baseline configuration, so RPG's advantage should be stated relative to explicit baseline settings.

**Caveat.** Do not say RPG is "not reproducible". Say that exact reproduction and baseline conclusions are protocol-sensitive.

## RQ1: Accuracy and fairness-oriented comparison

**Question.** How does RPG compare to tuned sequential baselines, especially SASRec, under accuracy and fairness-oriented metrics?

**Why it matters.** Average Recall/NDCG is not the full evaluation. RPG's semantic IDs and graph decoding may change recommendation behavior across item/user groups, popularity regions, and cold-start buckets.

**Hypotheses.**

- RPG's average advantage may shrink when SASRec is tuned fairly.
- RPG and SASRec may differ on fairness-oriented metrics, even when average accuracy is similar.
- The conclusion may be dataset-dependent.

**Evidence to show.**

- RPG versus SASRec average Recall@K/NDCG@K.
- Results on the added datasets.
- Cold-start / interaction-history buckets.
- Popularity-oriented metrics, long-tail exposure, APLT, ARP, or group-level performance if these are the implemented fairness metrics.

**Current answer.** The comparison should be framed as metric- and dataset-dependent. A stronger SASRec baseline and fairness-oriented metrics make the story less one-dimensional than "RPG is simply better".

**Caveat.** Define the fairness metrics precisely. Do not collapse cold-start robustness, popularity bias, and long-tail exposure into one generic "fairness" claim.

## RQ2: Static and dynamic graph structure

**Question.** What structural properties does the RPG item graph exhibit, and do these properties explain saturation as graph-decoding budget increases?

**Why it matters.** Graph decoding is one of RPG's main contributions. If performance saturates quickly, one possible explanation is that the graph is fragmented, noisy, hub-dominated, or redundant.

**Hypotheses.**

- If the graph is structurally poor, saturation may come from graph construction.
- If the graph is connected and semantically coherent, global graph pathology is unlikely.
- If the graph is highly clustered/redundant, extra graph budget may explore similar regions without improving final ranking.

**Evidence to show.**

- Largest weak component.
- Edge cosine similarity versus random item pairs.
- SID Hamming distance versus random item pairs.
- Reciprocity, clustering lift, and indegree Gini.
- Reachability, selected rate / Recall@10, visited items, and novelty/duplication as `n_edges` increases.

**Current answer.** The graph is not obviously broken. Across datasets, it is weakly connected, more similar than random, more SID-consistent than random, reciprocal, and clustered. Reachability increases strongly with budget, but Recall@10 saturates.

**Key takeaway.** Saturation is not explained by a disconnected or random graph. The graph is meaningful, but graph structure alone does not guarantee better recommendation quality.

**Caveat.** Weak connectivity is a static undirected property. It does not mean every target is reachable within a few directed RPG propagation steps.

## RQ3: Graph search versus scorer bottleneck

**Question.** Is RPG limited by graph reachability/search, or by the scoring and ranking capacity of the model once candidates are reachable?

**Why it matters.** This is the core explanatory question. Dynamic analysis shows many targets become reachable, but few are selected. The brute-force scorer check separates graph-search failure from scorer/ranker failure.

**Hypotheses.**

- If graph search is the bottleneck, brute-force all-item RPG scoring should substantially outperform graph decoding at large budget.
- If the scorer is the bottleneck, high-budget graph decoding should approach brute-force RPG scoring.
- If larger beam/search budget increases coverage but not Recall@10, selection/ranking is the main issue.

**Evidence to show.**

- Graph Recall@10/NDCG@10 versus brute-force RPG Recall@10/NDCG@10.
- Graph top-10 overlap with brute-force top-10.
- Brute-force top-10 missed by graph.
- Reachable rate versus selected rate.

**Current answer.** At sufficient graph budget, graph decoding nearly matches brute-force RPG scoring. Beauty, Sports, and Toys have almost zero Recall@10 loss versus brute force; CDs has a small remaining gap. This means graph search is not the main large-budget accuracy bottleneck. The scorer/ranker is.

**Key takeaway.** Once graph decoding recovers the RPG scorer's preferred top-k, more graph traversal cannot exceed the scorer's own upper bound.

**Caveat.** Brute-force RPG scoring is not an oracle. It is only the upper bound of the same RPG scorer without graph-search misses.

## RQ4: Practical efficiency of graph decoding

**Question.** Does graph decoding provide practical latency and memory advantages over alternatives, including brute-force scoring with the same RPG scorer?

**Why it matters.** RPG motivates graph decoding partly as an efficiency mechanism. But efficiency depends on the comparison: autoregressive decoding, graph decoding, brute-force RPG scoring, SASRec full sort, exact graph, or approximate graph.

**Hypotheses.**

- Graph decoding may reduce memory for large item pools.
- Graph decoding should reduce latency only if traversal overhead is lower than scoring many items.
- In the current implementation, Python/control-flow overhead and redundant traversal may reduce the expected speed benefit.

**Evidence to show.**

- Paper-style performance profiling: candidate-pool scaling, graph construction excluded, runtime delta memory, median repeats.
- Graph-analysis profiling: graph decoding versus vectorized brute-force RPG scoring, ms/user, peak CUDA memory, and Recall@10.

**Current answer.** The efficiency story is nuanced. In the graph-analysis profiling setup, vectorized brute-force RPG scoring is faster per user than graph decoding on the tested datasets, while graph decoding can use less memory on the largest dataset. Therefore graph decoding is not automatically a latency win; its practical value depends on scale, implementation, batching, and backend.

**Caveat.** Do not use the graph-analysis brute-force timing alone to refute all original efficiency claims. It is a different methodology from paper-style candidate-pool scaling and autoregressive decoding comparisons.

## Narrative flow

1. Start with reproduction: the original claims are important but protocol-sensitive.
2. Show that baseline fairness matters: a stronger SASRec changes the comparison.
3. Add fairness-oriented metrics: average accuracy is not the whole story.
4. Move to graph decoding: because it is RPG's main mechanism and performance saturates.
5. Show the graph is meaningful, not broken.
6. Show reachability improves but selection saturates.
7. Use brute-force RPG scoring to show the scorer/ranker is the main large-budget bottleneck.
8. End with efficiency: graph decoding may help scaling/memory, but in the reproduced setup it is not automatically faster than vectorized brute force.

## Suggested paper structure

1. Introduction: RPG claims, reproduction goal, and RQs.
2. Reproduction and baseline validity: RQ0.
3. Accuracy and fairness-oriented evaluation: RQ1.
4. Graph structure and decoding dynamics: RQ2.
5. Graph search versus scorer bottleneck: RQ3.
6. Efficiency profiling: RQ4.
7. Discussion: what RPG's graph is good for, what it does not solve, and where future work should focus.

## Final short answers

- **RQ0:** RPG is approximately reproducible, but exact conclusions depend on stochastic decoding, dataset accounting, and baseline tuning.
- **RQ1:** RPG's advantage is not universal; it depends on dataset, metric, and how fairly SASRec is tuned.
- **RQ2:** The graph is structurally meaningful and not globally fragmented, so saturation is not due to an obviously poor graph.
- **RQ3:** At high graph budget, graph decoding nearly reaches the brute-force RPG scorer upper bound; the main accuracy bottleneck is the scorer/ranker.
- **RQ4:** Graph decoding is not automatically faster in the reproduced setup, although it may offer memory/scaling advantages in larger regimes or with better graph backends.
