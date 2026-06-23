# Repository Summary and Important Results

This document is a detailed handoff for the current state of `RPG-uva`.
It summarizes what was implemented in this repository, what experiments were run, and which results matter most.

Scope rules for this summary:

- It focuses on `RPG`, graph analysis, performance analysis, datasets, and ablations.
- It explicitly excludes Diffuser/DRPG as a research direction to continue. That line was later removed from the main branch (`git log` contains `91d6a24 remove Diffusion RPG`), so it is treated here as cut off.
- It only reports claims that are supported by files currently present in this repository.
- When a result appears to rely on shared artifacts that are not checked in locally, that gap is called out directly.

## 1. What This Repository Became

This repo started as a wrapper and experiment layer around the upstream RPG implementation stored in the read-only `third_party/` submodule.
The important repo-owned work happened outside the submodule:

- root-level RPG wrapper and configs
- reproduction jobs for the four paper datasets
- SASRec baselines implemented in-repo
- SASRec-modernized implementation and evaluation pipeline
- hyperparameter sweep and result collection tooling
- graph-analysis diagnostics and notebooks
- performance profiling harnesses for RPG and SASRec-style models
- new-dataset pipelines for `Video_Games` and `Pet_Supplies`
- seed-robustness, inference-grid, and ablation studies

The recent commit history reflects these phases:

- `7550e2e`: SASRec performance, cold-start, and seeded evaluation tooling
- `18265c3`, `d36ec9d`, `ba7cc77`: train/inference hyperparameter sweeps and result collection
- `f4f1539` to `9476c6c`: new-dataset support and documentation
- `b259827` to `7e39f7e` and `dd38f8e`: static/dynamic RPG graph diagnostics
- `922da77`, `0e084db`: brute-force and graph/performance comparisons
- `316d8e1`: SASRec parameter-matched ablation workflow
- `6166f45`: additional hyperparameter sweep work
- `91d6a24`: diffusion line removed from main

## 2. Core Repo Structure

The main owned areas are:

- `scripts/`: repo-owned runners, profiling entrypoints, data prep, and result collectors
- `configs/rpg/`, `configs/sasrec/`, `configs/sasrec_modernized/`: experiment presets
- `jobs/reproduction/` and `jobs/new_datasets/`: Snellius job scripts
- `results/` and `results/tables/`: compact CSV summaries for the important conclusions
- `docs/`: reproduction notes, graph-analysis plans, and performance documentation
- `output/`: checked-in scheduler outputs and some experiment CSVs
- `notebooks/`: analysis notebooks, especially graph, cold-start, and ablation notebooks

Operationally, the repo enforces a clean separation:

- job definitions under `jobs/`
- scheduler logs under `output/`
- runtime artifacts under `artifacts/`

That layout is consistent with `AGENTS.md` and the Snellius guidance stored in `docs/snellius/`.

## 3. RPG Reproduction Work

### 3.1 What was implemented

The repo wraps upstream RPG so experiments can be run without modifying the submodule:

- `scripts/rpg.py`
- `configs/rpg/root.yaml`
- `configs/rpg/repro/*.yaml`
- `jobs/reproduction/rpg/...`

The main paper-facing datasets are:

- `Sports_and_Outdoors`
- `Beauty`
- `Toys_and_Games`
- `CDs_and_Vinyl`

### 3.2 Reproduction quality against the paper

The compact verdict is in `results/tables/claim2_verdict.csv`.

Best semantic-ID length comparison:

| Dataset | Paper best m | Repo best m | Paper NDCG@10 | Repo NDCG@10 | Gap |
| --- | --- | --- | --- | --- | --- |
| Sports | 16 | 16 | 0.0263 | 0.0254 | -3.5% |
| Beauty | 32 | 32 | 0.0464 | 0.0469 | +1.1% |
| Toys | 16 | 32 | 0.0490 | 0.0465 to 0.0467 | about -5% |
| CDs | 64 | 64 | 0.0415 | 0.0403 | -3.0% |

Interpretation:

- Reproduction is close to paper quality on all four datasets.
- Beauty essentially matches or slightly exceeds the paper number.
- Sports and CDs are within a few percent.
- Toys is the least aligned dataset: the repo prefers `m=32`, while the paper reports `m=16`.

### 3.3 Important caveat on Sports

`docs/reproduction_notes.md` explains the Sports mismatch carefully.
The important points are:

- the paper’s Sports user/item counts appear swapped relative to the code logs
- the interaction count mismatch is explainable by whether the test item has already been held out
- RPG graph decoding is stochastic, so eval-only reruns can differ from the training job’s final built-in evaluation

The repo conclusion is that the training job’s final test evaluation is the better single-run comparator for the paper than a separate eval-only rerun.

### 3.4 Training hyperparameter retuning results

`results/tables/rpg_train_best.csv` and `results/tables/train_grid_best.csv` show that retuning training hyperparameters improved over the released README settings.

Best RPG training settings found:

| Dataset | Released config | Released test | Selected config | Selected test | Gain |
| --- | --- | --- | --- | --- | --- |
| Sports | `lr=0.003, t=0.03` | 0.0255 | `lr=0.001, t=0.01` | 0.0264 | +0.0008 |
| Beauty | `lr=0.01, t=0.03` | 0.0460 | `lr=0.003, t=0.01` | 0.0463 | +0.0003 |
| Toys | `lr=0.003, t=0.03` | 0.0468 | `lr=0.003, t=0.01` | 0.0483 | +0.0015 |

Seed robustness from `results/tables/seed_robustness.csv`:

- Sports: tuned-vs-selected intervals do not overlap, so the improvement looks real
- Beauty and Toys: improvements exist, but overlap is larger, so the evidence is weaker than Sports

### 3.5 Inference-grid results

The inference-side decode grid is summarized in:

- `results/tables/infer_grid_best.csv`
- `results/tables/decode_val_selected.csv`
- `results/tables/decode_test_selected.csv`
- `results/tables/decode_grid_summary.csv`
- `results/fig6_grid.csv`

The important selected decode settings are:

| Dataset | Selected `(b,k,q)` | Selected test NDCG@10 |
| --- | --- | --- |
| Sports | `(200,100,2)` | 0.0257 |
| Beauty | `(50,500,2)` | 0.0470 |
| Toys | `(200,500,1)` | 0.0471 |
| CDs | `(200,500,4)` | 0.0414 |

Important interpretation:

- decode retuning gives small but real improvements over the released README defaults
- the improvements are modest compared with the total performance level
- many nearby decode configurations have nearly identical test performance
- on several datasets, the validation argmax configuration is not uniquely best on test, indicating a broad plateau rather than a sharp optimum

This already hints at the later graph-analysis conclusion: more graph budget often does not translate into much extra ranking quality.

## 4. Main Graph Study

This is one of the most important research contributions of the repo.
The narrative sources are:

- `README.md`
- `docs/graph_analysis_extension_plan.md`
- `docs/ta_feedback_next_steps.md`
- `notebooks/graph_analysis/sports_and_outdoors.ipynb`
- `notebooks/graph_analysis/cds_and_vinyl.ipynb`
- `notebooks/graph_analysis/cross_dataset_comparison.ipynb`

### 4.1 Question being studied

The graph-analysis line asks:

> Why does RPG’s graph-constrained decoding saturate after a modest inference budget?

This was decomposed into:

- static graph quality questions
- dynamic traversal and reachability questions
- ranking-vs-reachability questions
- brute-force scoring comparisons

### 4.2 Static graph conclusions

The repo-level summary in `README.md` is:

- graph neighbors are much more similar than random item pairs
- the graph becomes globally connected quickly
- the graph is locally clustered
- hubness/popularity exists but is not the whole story

Static conclusion:

> The graph itself is coherent, connected, and locally redundant.
> Figure 6 saturation is unlikely to be caused mainly by bad graph construction or disconnected components.

This is important because it rejects the simplest failure hypothesis.
The graph is not obviously broken.

### 4.3 Dynamic decoding conclusions

The dynamic analysis is more important than the static analysis because it follows actual decoding.
The key conclusions repeated in `README.md` are:

- increasing graph width strongly improves target access
- recommendation quality improves only slightly
- later propagation steps add fewer useful new candidates
- reachable targets are often found early
- increasing the coupled beam/search budget improves access more than final Recall/NDCG

Dynamic conclusion:

> The main bottleneck seems to be candidate scoring/ranking after access, not graph reachability alone.

This is probably the single most important scientific finding in the repo.

### 4.4 Quantitative evidence for saturation

The clearest compact evidence is in `results/fig6_grid.csv`.

Examples:

- Sports with `b=10, q=2`: NDCG@10 rises from `0.0034` at `k=10` to `0.0238` at `k=300`, but the final tuned system only reaches `0.0257` even with much larger decode budgets
- Beauty with `b=10, q=2`: NDCG@10 rises quickly from `0.0066` at `k=10` to `0.0458` at `k=300`
- Toys shows the same pattern: rapid early gains, then a plateau
- CDs behaves similarly, with larger absolute graph budgets needed because the candidate pool is much larger

The practical reading is:

- very small graph budgets are too restrictive
- once the decoder can reach enough plausible candidates, extra search mostly adds redundant items
- ranking quality then becomes the limiting factor

### 4.5 TA-driven interpretation and next-step refinement

`docs/ta_feedback_next_steps.md` is important because it tightened the research framing:

- the repo should compare graph decoding against brute-force all-item RPG scoring
- reachability and ranking should be separated
- strong connectivity should be analyzed alongside weak connectivity
- the beam-budget diagnostic should not be mislabeled as pure pruning
- naive cosine reranking should not be overinterpreted because it mismatches the RPG training objective

That feedback pushed the project from “interesting plots” toward a more defensible causal story.

### 4.6 Overall graph-study conclusion

The current best summary is:

1. RPG’s item graph appears structurally healthy.
2. Increasing graph budget makes the true item reachable much more often.
3. Final top-10 ranking improves much less than reachability.
4. Therefore the dominant problem is not just graph access; it is candidate scoring and selection after access.

## 5. Performance Analysis

There are two performance lines in the repo:

- RPG performance profiling
- SASRec / SASRec-modernized performance profiling

The main RPG documentation is `docs/perf_profiling.md`.

### 5.1 RPG performance harness

The repo adds a profiling layer around RPG without editing `third_party/`.
It measures:

- enlarged-pool inference
- offline sparse-graph build cost
- inference-only epoch time
- total and runtime-delta CUDA memory
- visited item counts and visited ratio

This is a method contribution of the repo, even where the checked-in result set is incomplete.

### 5.2 Graph validation result

`artifacts/rpg/perf/sports/20260606T085403711257Z_job23517619/graphs/validate_graph_report.json` validates the replacement graph builder on Sports.

Important numbers for dense-vs-flat:

- exact top-100 match rate across items: `0.9924`
- mean overlap rate: `0.9999`
- only `140` of `18,357` items have any mismatch

Interpretation:

- the exact FAISS flat formulation reproduces the released dense graph almost perfectly
- this is strong evidence that the repo-owned scalable profiling path is faithful enough to compare against the original RPG graph construction

### 5.3 Checked-in brute-force RPG performance result

One checked-in summary is:

- `artifacts/rpg/perf/sports/20260619T102453985310Z_job24019933/summaries/profile_summary.csv`

It contains a brute-force RPG scoring row on a 500k candidate pool:

- method: `RPG-BruteForce`
- pool size: `500000`
- median epoch time: `26.26s`
- runtime-delta CUDA allocated: `0.393 GB`
- visited items: `500000`
- NDCG@10: `0.01014`
- Recall@10: `0.01185`

Important limitation:

- the checked-in file currently exposes only the brute-force row, not the paired graph-decoding rows that would complete the paper-style comparison inside the repo

### 5.4 Graph-analysis notebook performance takeaway

The Sports graph-analysis notebook explicitly states a practical conclusion:

> under the graph-analysis hyperparameters, graph decoding is still much slower than vectorized all-item scoring on Sports, even though it uses slightly less runtime CUDA delta memory

That statement matters, but it should be kept scoped to the notebook’s specific setup.
The notebook itself warns that this should not be generalized to the paper’s large-pool performance setting without a matched rerun.

### 5.5 Performance-analysis conclusion

What the repo supports today is:

- a faithful graph-construction replacement path
- a profiling framework for larger candidate pools
- evidence that performance needs to be discussed jointly with ranking quality and graph budget
- a strong methodological base for future matched RPG-vs-brute-force and RPG-vs-SASRec comparisons

## 6. SASRec Baselines and Modernized SASRec

This repo does not only reproduce RPG.
It also builds strong non-RPG baselines.

### 6.1 SASRec baseline work

The repo includes:

- in-repo SASRec implementation under `models/sasrec`
- data preparation script `scripts/sasrec_prepare_data.py`
- reproduction configs and jobs under `configs/sasrec/` and `jobs/reproduction/sasrec/`

`docs/reproduction_notes.md` documents the intended SASRec contract and the paper target metrics.

### 6.2 SASRec training-grid improvements

`results/tables/sasrec_train_best.csv` and `results/tables/sasrec_grid_best.csv` show that the released SASRec settings were not optimal.

Best found settings vs released settings:

| Dataset | Released test | Selected test | Gain |
| --- | --- | --- | --- |
| Sports | 0.0180 | 0.0194 | +0.0014 |
| Beauty | 0.0294 | 0.0324 | +0.0030 |
| Toys | 0.0381 | 0.0420 | +0.0039 |

Interpretation:

- the repo substantially strengthens the SASRec baseline relative to naive released settings
- this matters for any fair RPG-vs-SASRec comparison

### 6.3 SASRec-modernized line

The repo later added a “modernized” SASRec path and then a parameter-matched ablation workflow.
The implementation evidence is in:

- `models/sasrec_modernized`
- `configs/sasrec_modernized`
- `jobs/reproduction/sasrec_modernized`
- `output/reproduction/sasrec_modernized/...`

This line appears to be the main non-RPG comparison vehicle for later ablation and performance work.

## 7. Datasets Beyond the Paper

The new-dataset pipeline is fully documented in:

- `jobs/new_datasets/README.md`
- `jobs/new_datasets/rpg/README.md`

The two added categories are:

- `Video_Games`
- `Pet_Supplies`

The full pipeline was designed to support:

1. RPG semantic-ID preparation
2. RPG training
3. RPG multi-seed evaluation
4. SASRec basic data prep and training
5. SASRec-modernized training and multi-seed evaluation

### 7.1 New-dataset SASRec results

The checked-in summary `results/sasrec_new_datasets_grid_results.md` is the clearest local evidence for these datasets.

Best Video Games SASRec training setting:

- `lr=0.0003`, `dropout=0.5`, `blocks=2`
- best validation epoch `126`
- NDCG@10 `0.0627`
- Recall@10 `0.1200`

Best Pet Supplies SASRec training setting:

- `lr=0.0003`, `dropout=0.2`, `blocks=2`
- best validation epoch `95`
- NDCG@10 `0.0360`
- Recall@10 `0.0683`

Important interpretation:

- the new datasets are viable and produce sensible nontrivial metrics
- `Video_Games` is easier than `Pet_Supplies` under this baseline
- low learning rate (`0.0003`) was clearly favored on both datasets

### 7.2 What is missing locally

The repo docs say both new datasets completed the full five-stage pipeline at least once.
However, the local workspace does not currently contain:

- checked-in RPG new-dataset multi-seed `summary.json` files
- checked-in SASRec/SASRec-modernized new-dataset multi-seed `summary.json` files

Also, `output/new_datasets/rpg/` only contains `.gitkeep`, not checked-in result logs.

So the important honest conclusion is:

- the pipeline exists and SASRec grid results are present
- but the final RPG-vs-SASRec new-dataset comparison numbers are not available locally in this repo snapshot

## 8. Ablations

There are several ablation directions in the repo.

### 8.1 Semantic-ID length ablation for RPG

`results/tables/msweep_best.csv` summarizes the `m` sweep:

| Dataset | Repo best m | Repo NDCG@10 | Paper best m | Paper NDCG@10 |
| --- | --- | --- | --- | --- |
| Sports | 16 | 0.0254 | 16 | 0.0263 |
| Beauty | 32 | 0.0469 | 32 | 0.0464 |
| Toys | 32 | 0.0467 | 16 | 0.0490 |
| CDs | 64 | 0.0403 | 64 | 0.0415 |

Takeaways:

- the semantic-ID length matters materially
- the paper’s preferred `m` is reproduced on 3 of 4 datasets
- Toys remains the main inconsistency

### 8.2 SASRec parameter-matched ablation

`git log` shows `316d8e1 Add SASRec parameter-matched ablation workflow`.
The repo contains:

- `configs/sasrec/param_matched/*.yaml`
- `output/reproduction/sasrec_modernized/ablation_size/...`
- `notebooks/size_ablation_comparison.ipynb`

This ablation line exists to answer a fairness question:

- how much of RPG’s performance advantage survives when SASRec is given matched or near-matched model capacity?

The checked-in notebook is the main analysis surface, but there is no compact checked-in markdown/CSV in `results/` that states a final single-sentence verdict.
So the correct summary is:

- the size/parameter-matched ablation workflow was implemented
- the run outputs are present
- but the final distilled conclusion is still embedded mainly in notebook analysis rather than a clean text artifact

### 8.3 Graph-budget ablation

The decode grid doubles as a graph-budget ablation:

- varying `num_beams`
- varying `n_edges`
- varying `propagation_steps`

This is one of the strongest completed ablations because it yielded a stable conclusion:

- more budget raises reachability much more than it raises ranking performance

### 8.4 Search/pruning/reranking diagnostics

The graph notebooks and commit history show several diagnostics were added:

- dynamic graph experiments
- pruning / beam-budget diagnostic
- reranking attempts
- brute-force comparison
- visited-pool analysis

These do not all end in a single productionized pipeline, but together they support the same story:

- the interesting failures occur after graph access, not only before it

## 9. Important Negative Results and Constraints

These are just as important as the positive results.

### 9.1 The graph is not obviously the main failure

The repo rules out several easy explanations:

- the graph is not random
- the graph is not badly disconnected
- simply increasing graph width does not solve the quality problem

### 9.2 More inference budget is not a complete fix

The decode sweeps show broad saturation.
This means:

- improving only traversal budget has diminishing returns
- a better final scoring/reranking mechanism is a more promising next direction

### 9.3 Some result artifacts are not local

Not every documented experiment has a local final summary artifact.
Notable missing pieces in the current checkout:

- some new-dataset multi-seed summary files
- a complete paired RPG performance table for graph-decoding vs brute-force in `results/`
- a compact text verdict for the SASRec parameter-matched ablation

### 9.4 The repo is in a partly in-progress state

`git status --short` shows uncommitted work around:

- `scripts/sasrec_cold_start.py`
- `scripts/sasrec_perf.py`
- `notebooks/perf_sports_analysis.ipynb`
- new grid/perf job folders and outputs

So the repo contains both completed conclusions and ongoing work.

## 10. Best High-Level Summary

If someone needs the shortest honest description of what this repository achieved, it is this:

1. It reproduced RPG on the main paper datasets reasonably closely, with especially good alignment on Beauty and decent alignment on Sports, Toys, and CDs.
2. It showed that retuning RPG training hyperparameters and decode hyperparameters gives small but real improvements over the released defaults.
3. It built stronger SASRec baselines than the released settings, which is important for fair comparisons.
4. It added a substantial graph-analysis line and reached the key conclusion that RPG’s sparse graph often reaches the right region, but the final ranking/scoring stage does not convert that extra access into equally large top-k gains.
5. It built the tooling for performance profiling, larger-pool experiments, new datasets, and size/parameter-matched ablations.
6. It cut off the diffuser direction and left the most valuable future work centered on better scoring/reranking, stronger brute-force-vs-graph comparisons, and cleaner final summaries of the ablation studies.

## 11. Recommended “What Matters Most” Results to Reuse

If you only keep a few results from this repo for a report, presentation, or continuation, these are the ones that matter most:

- `results/tables/claim2_verdict.csv`
  RPG reproduction quality vs the paper.

- `results/tables/rpg_train_best.csv`
  RPG training retuning gains.

- `results/tables/sasrec_train_best.csv`
  SASRec strengthening relative to released defaults.

- `results/tables/decode_grid_summary.csv`
  Best decode settings and their test performance.

- `results/fig6_grid.csv`
  The clearest quantitative evidence of graph-budget saturation.

- `docs/reproduction_notes.md`
  The Sports mismatch explanation and stochastic-eval caveat.

- `README.md` graph-analysis takeaways
  The most compact statement of the main scientific conclusion.

- `docs/ta_feedback_next_steps.md`
  The best roadmap for turning the current graph results into a stronger final story.

## 12. Bottom Line

The repository’s most important contribution is not just “we reran RPG.”
It is:

- a stronger and more careful reproduction stack,
- better baseline infrastructure,
- and a convincing analysis that RPG’s graph search is not the whole problem.

The current evidence points to this central conclusion:

> RPG’s sparse graph usually provides enough access to relevant candidates.
> The bigger remaining issue is how those candidates are scored and selected into the final recommendation list.
