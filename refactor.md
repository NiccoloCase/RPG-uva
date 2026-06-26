# Publication Refactor

## Scope

This branch, `refactor_publish`, prepares the repository for publication around the accompanying report.

The refactor goal is not to redesign the research code from scratch. The goal is to make the repository understandable, reproducible, and honest about which code is canonical, which code is preserved for provenance, and how the job tree maps to the paper.

## Non-Negotiable Decisions

### 1. Keep `third_party/`

`third_party/` remains in the public repository.

Reason:

- it preserves the exact upstream RPG dependency boundary
- it makes the provenance of RPG explicit
- it avoids pretending that repo-owned wrapper code is the same thing as the original RPG source

This refactor does not modify `third_party/`.

### 2. Canonical SASRec Means The Former `sasrec_modernized`

The repo previously exposed two SASRec variants:

- `sasrec`
- `sasrec_modernized`

That is not acceptable for a public release because it creates baseline ambiguity.

The publication rule is:

- the former `sasrec_modernized` path is now the canonical public `sasrec`
- the older duplicate SASRec path is removed from the published surface

### 3. Paper-First Navigation

The public reader should be able to navigate the repository by the paper's structure:

1. Reproduction
2. Accuracy and fairness
3. Graph structure and dynamics
4. Search vs scorer
5. Efficiency

The runnable job scripts still live in implementation-oriented trees, but the repo now provides paper-index folders under `jobs/01_*` through `jobs/05_*`.

## Canonical Public Surface

### Root-level user-facing entrypoints

- `scripts/rpg.py`
- `scripts/sasrec.py`
- `scripts/sasrec_eval.py`
- `scripts/sasrec_cold_start.py`
- `scripts/sasrec_prepare_data.py`
- `scripts/rpg_prepare_semantic_ids.py`

### Canonical config roots

- `configs/rpg/`
- `configs/sasrec/`

### Canonical job roots

- `jobs/reproduction/rpg/`
- `jobs/reproduction/sasrec/`
- `jobs/new_datasets/`

### Canonical artifact roots

- `artifacts/rpg/`
- `artifacts/sasrec/`
- `output/reproduction/rpg/`
- `output/reproduction/sasrec/`

## Removed Duplicate Surface

The following duplicate SASRec surfaces are removed from the published repo:

- `scripts/sasrec_modernized.py`
- `scripts/sasrec_legacy.py`
- `configs/sasrec_modernized/`
- `configs/sasrec_legacy/`
- `models/sasrec_modernized/`
- `models/sasrec_legacy/`
- `jobs/reproduction/sasrec_modernized/`
- `jobs/reproduction/sasrec_legacy/`

## Implemented Mapping

### Model package mapping

- public `models/sasrec/` now contains the former modernized implementation

### Config mapping

- public `configs/sasrec/` now contains the former modernized dataset and root presets
- public `configs/sasrec/param_matched/` and `configs/sasrec/eval_seeds/` are retained for the paper analyses

### Job mapping

- public `jobs/reproduction/sasrec/` now contains the former modernized reproduction, ablation, grid, cold-start, eval-seed, and performance jobs

### Script mapping

- public `scripts/sasrec.py` is the canonical SASRec trainer/evaluator

## Paper Structure Mapping

### 01 Reproduction

Paper question:

- can the main RPG and SASRec numbers be reproduced cleanly?

Repo entrypoints:

- `jobs/01_reproduction/`
- `jobs/reproduction/rpg/<dataset>/`
- `jobs/reproduction/sasrec/<dataset>/`

### 02 Accuracy And Fairness

Paper question:

- how robust are the reported conclusions across seeds, cold-start buckets, and extended datasets?

Repo entrypoints:

- `jobs/02_accuracy_and_fairness/`
- `jobs/reproduction/eval_seeds/`
- `jobs/reproduction/sasrec/eval_seeds/`
- `jobs/reproduction/rpg/cold_start/`
- `jobs/reproduction/sasrec/cold_start/`
- `jobs/new_datasets/`

### 03 Graph Structure And Dynamics

Paper question:

- what structural and dynamic properties does the RPG graph exhibit?

Repo entrypoints:

- `jobs/03_graph_structure_and_dynamics/`
- `jobs/reproduction/rpg/graph_analysis/`
- `rpg_graph_analysis/`

### 04 Search Vs Scorer

Paper question:

- is RPG primarily limited by graph search or by the scorer/ranker after reachability?

Repo entrypoints:

- `jobs/04_search_vs_scorer/`
- `jobs/reproduction/rpg/grid/`
- `jobs/reproduction/sasrec/ablation_size/`

### 05 Efficiency

Paper question:

- what is the practical efficiency story for RPG and SASRec-style graph decoding?

Repo entrypoints:

- `jobs/05_efficiency/`
- `jobs/reproduction/rpg/perf/`

## README Contract

The public README now does four things only:

1. explains the repository boundary
2. states that `third_party/` is preserved and read-only here
3. defines canonical vs legacy SASRec naming
4. gives explicit reproduction commands organized by paper section

The README intentionally does not try to explain every exploratory notebook or intermediate result file.

## What Was Intentionally Not Refactored

### 1. `third_party/`

Preserved untouched.

### 2. Research notebooks

The notebooks remain available as research artifacts. They are not the primary public entrypoints.

### 3. Existing results files

Collected CSV and Markdown result summaries remain in place. The public release should expose them, but the README should not depend on them for basic navigation.

### 4. Internal exploratory notes

Files such as `FAILED.md` and other research notes are left as internal context rather than being promoted to public navigation anchors.

## Naming Rules After The Refactor

### Canonical names

- `sasrec` means the publication baseline
- `rpg` means the repo-owned wrapper around the preserved upstream RPG code

### Reserved names

- `sasrec_modernized` should not be used for public-facing commands, docs, or job references

## Public Directory Contract

### `scripts/`

Contains the primary command-line entrypoints used in the README and job scripts.

### `configs/`

Contains canonical presets for:

- dataset-specific reproduction
- eval-seed runs
- performance profiling

### `jobs/`

Contains only Slurm job definitions.

The paper-oriented navigation layer is:

- `jobs/01_reproduction/`
- `jobs/02_accuracy_and_fairness/`
- `jobs/03_graph_structure_and_dynamics/`
- `jobs/04_search_vs_scorer/`
- `jobs/05_efficiency/`

The implementation-oriented runnable trees remain:

- `jobs/reproduction/rpg/`
- `jobs/reproduction/sasrec/`
- `jobs/new_datasets/`

### `output/`

Contains scheduler logs only.

### `artifacts/`

Contains generated checkpoints, caches, summaries, profiling outputs, and other runtime artifacts.

## Validation Checklist

The publication refactor is considered correct if all of the following are true:

- `third_party/` remains untouched
- `scripts/sasrec.py` is the canonical public SASRec entrypoint
- there is no second SASRec implementation tree in the published repo
- the public SASRec configs live under `configs/sasrec/`
- the public SASRec jobs live under `jobs/reproduction/sasrec/`
- README reproduction commands use the canonical names
- the paper can be followed from `jobs/01_*` through `jobs/05_*`
- the duplicate SASRec trees are removed

## Recommended Review Focus

When reviewing this branch, focus on:

1. whether any public-facing command still uses `sasrec_modernized`
2. whether README commands point to real runnable job paths
3. whether SASRec artifact and output directories now consistently use `artifacts/sasrec/` and `output/reproduction/sasrec/`
4. whether the paper index directories match the actual narrative of the report
5. whether any deleted transitional path is still accidentally wired into the canonical public surface
