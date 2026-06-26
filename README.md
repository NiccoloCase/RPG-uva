# RPG-uva

This repository packages two things for publication:

- the original RPG implementation, preserved as a read-only dependency in `third_party/`
- the repo-owned reproduction, analysis, and baseline code used for the paper in `docs/RPG_NEW_report (1).pdf`

The public baseline name is `SASRec`. The older duplicate SASRec trees were removed so the repo exposes one baseline surface only.

## Repository Structure

- `third_party/`: pinned upstream RPG dependency. Do not edit it here.
- `scripts/`: canonical repo entrypoints. Use `scripts/rpg.py` and `scripts/sasrec.py`.
- `configs/`: repo-owned experiment presets. Use `configs/rpg/` and `configs/sasrec/`.
- `jobs/`: Snellius Slurm jobs. Start with the paper index folders:
  `jobs/01_reproduction/`, `jobs/02_accuracy_and_fairness/`, `jobs/03_graph_structure_and_dynamics/`, `jobs/04_search_vs_scorer/`, `jobs/05_efficiency/`.
- `artifacts/`: checkpoints, caches, and runtime outputs.
- `output/`: scheduler stdout/stderr logs.
- `results/`: collected tables and summaries.
- `docs/`: paper, notes, and Snellius guidance.

## Setup

Initialize the submodule:

```bash
git submodule update --init --recursive
```

Create the environment from the repo root:

```bash
conda env create -p "$(pwd)/artifacts/conda/rpg-uva" -f environment.yml
conda activate "$(pwd)/artifacts/conda/rpg-uva"
```

On Snellius, prefer the checked-in Slurm jobs. Submit every job from its own job directory.

## Canonical Entry Points

RPG:

```bash
python3 scripts/rpg.py --preset beauty
```

SASRec:

```bash
python3 scripts/sasrec.py --preset beauty --dataset Beauty
```

## Reproduction Commands

All commands below are paper-facing Snellius commands. Change datasets as needed.

### 1. Reproduction

Canonical SASRec full run:

```bash
cd /gpfs/home6/$USER/RPG-uva/jobs/reproduction/sasrec
DATASET=beauty bash ./submit_all.sh
```

RPG train and eval:

```bash
cd /gpfs/home6/$USER/RPG-uva/jobs/reproduction/rpg/beauty
sbatch ./train.sh
sbatch ./eval.sh
```

### 2. Accuracy And Fairness

SASRec multi-seed evaluation:

```bash
cd /gpfs/home6/$USER/RPG-uva/jobs/reproduction/sasrec/eval_seeds/released_readme
sbatch ./run_sports_and_outdoors.sh
```

RPG multi-seed evaluation:

```bash
cd /gpfs/home6/$USER/RPG-uva/jobs/reproduction/eval_seeds/released_readme
sbatch ./run_sports_and_outdoors.sh
```

SASRec cold-start analysis:

```bash
cd /gpfs/home6/$USER/RPG-uva/jobs/reproduction/sasrec/cold_start
sbatch ./run_cold_start.sh
```

RPG cold-start analysis:

```bash
cd /gpfs/home6/$USER/RPG-uva/jobs/reproduction/rpg/cold_start
sbatch ./run_cold_start.sh
```

### 3. Graph Structure And Dynamics

Static graph analysis:

```bash
cd /gpfs/home6/$USER/RPG-uva/jobs/reproduction/rpg/graph_analysis
sbatch ./run_static_sports.sh
```

Dynamic graph analysis:

```bash
cd /gpfs/home6/$USER/RPG-uva/jobs/reproduction/rpg/graph_analysis
sbatch ./run_dynamic_sports.sh
```

### 4. Search Vs Scorer

RPG decode grid:

```bash
cd /gpfs/home6/$USER/RPG-uva/jobs/reproduction/rpg/grid
sbatch ./run_decode_grid.sh
```

RPG decode confirmation:

```bash
cd /gpfs/home6/$USER/RPG-uva/jobs/reproduction/rpg/grid
sbatch ./run_decode_confirm.sh
```

SASRec size ablation:

```bash
cd /gpfs/home6/$USER/RPG-uva/jobs/reproduction/sasrec/ablation_size
sbatch ./train_sports.sh
```

### 5. Efficiency

RPG inference profiling:

```bash
cd /gpfs/home6/$USER/RPG-uva/jobs/reproduction/rpg/perf
sbatch ./build_graphs.sh /abs/path/to/rpg_checkpoint.pth
sbatch ./profile_inference.sh /abs/path/to/rpg_checkpoint.pth
```

## New-Dataset Extension

The paper datasets live under `jobs/reproduction/`. Extra datasets such as `video_games` and `pet_supplies` live under `jobs/new_datasets/`.

Use:

```bash
cd /gpfs/home6/$USER/RPG-uva/jobs/new_datasets
```

Then follow `jobs/new_datasets/README.md`.

## Notes

- `third_party/` stays in the public repository as the preserved original RPG source boundary.
- The paper-facing job index lives in `jobs/01_reproduction/` through `jobs/05_efficiency/`.
- The canonical SASRec artifacts now live under `artifacts/sasrec/` and `output/reproduction/sasrec/`.
