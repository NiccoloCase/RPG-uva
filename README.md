# RPG-uva

This repo keeps the upstream RPG implementation in the `third_party/` submodule and adds a root-level wrapper so you can run experiments from here without editing the submodule.

## Layout

- `third_party/`: upstream `facebookresearch/RPG_KDD2025` submodule. Read-only in this repo.
- `scripts/rpg.py`: root-level runner that imports the submodule code and exposes config overlays.
- `environment.yml`: root-level conda environment mirroring the upstream Python dependencies.
- `configs/rpg/root.yaml`: repo-owned defaults for output paths.
- `configs/rpg/repro/*.yaml`: paper reproduction presets.
- `configs/rpg/local.example.yaml`: template for untracked machine-specific overrides.

## Setup

Initialize the submodule:

```bash
git submodule update --init --recursive
```

Create the conda environment from the repo root:

```bash
conda env create -p "$(pwd)/artifacts/conda/rpg-uva" -f environment.yml
conda activate "$(pwd)/artifacts/conda/rpg-uva"
```

`environment.yml` mirrors the upstream dependency list and keeps the upstream CUDA 12.9 PyTorch index. If you need CPU-only PyTorch or a different CUDA build, change the `torch` lines there instead of touching `third_party/requirements.txt`.

## Root Commands

Show the wrapper help:

```bash
python3 scripts/rpg.py --help
```

Run one of the paper presets from the repo root:

```bash
python3 scripts/rpg.py --preset sports_and_outdoors
python3 scripts/rpg.py --preset beauty
python3 scripts/rpg.py --preset toys_and_games
python3 scripts/rpg.py --preset cds_and_vinyl
```

Run a custom command from the repo root:

```bash
python3 scripts/rpg.py --category Sports_and_Outdoors --lr 0.003 --temperature 0.03
```

Add extra YAML overrides on top of the preset:

```bash
python3 scripts/rpg.py \
  --preset beauty \
  --config path/to/experiment.yaml \
  --run_id beauty_debug
```

## Performance Profiling

The repo also includes a repo-owned profiling layer for reproducing the RPG side of the paper's inference-efficiency study without modifying `third_party/`.

Validate the exact sparse graph on the original Sports pool:

```bash
python3 scripts/rpg_perf.py \
  validate-graph \
  --checkpoint /abs/path/to/checkpoint.pth \
  --config configs/rpg/perf/sports.yaml
```

Prebuild enlarged-pool adjacency caches and then run inference-only profiling:

```bash
python3 scripts/rpg_perf.py \
  profile \
  --checkpoint /abs/path/to/checkpoint.pth \
  --config configs/rpg/perf/sports.yaml \
  --prepare-only

python3 scripts/rpg_perf.py \
  profile \
  --checkpoint /abs/path/to/checkpoint.pth \
  --config configs/rpg/perf/sports.yaml \
  --profile-only
```

More detailed usage, artifact layout, and Snellius job wrappers are documented in [docs/perf_profiling.md](docs/perf_profiling.md).

## Config Overrides

The root wrapper loads configs in this order:

1. `third_party/genrec/default.yaml`
2. `third_party/genrec/datasets/.../config.yaml`
3. `third_party/genrec/models/.../config.yaml`
4. `configs/rpg/root.yaml`
5. `configs/rpg/local.yaml` if present
6. `configs/rpg/repro/*.yaml` when `--preset` is used
7. Any extra `--config path/to/file.yaml`
8. CLI overrides like `--lr 0.003` or `--lr=0.003`

To keep machine-specific settings out of Git, create an untracked local override:

```bash
cp configs/rpg/local.example.yaml configs/rpg/local.yaml
```

Then edit `configs/rpg/local.yaml` with your own settings, for example batch sizes, FAISS threading, or secrets such as `openai_api_key`.

## Full Hyperparameter Sweep on a New Dataset

The full sweep tunes three groups of hyperparameters: **semantic-ID length `m`**,
**training** (learning rate, temperature), and **inference decoding** (beam size
`b`, graph degree `k`, propagation steps `q`). Following the original paper, `m` is
tuned **jointly** with lr/temperature (the paper's `3 lr × 3 temp × 5 m = 45`-cell
grid), then the inference parameters are tuned by re-decoding the selected
checkpoint. The four paper datasets are already done; this section is for adding a
new Amazon category. All jobs run on Snellius.

Throughout, `<ds>` is the snake_case preset name (e.g. `office_products`) and
`<Category>` is the Title_Case cache name (e.g. `Office_Products`). 

### Prerequisites

1. Snellius access and the group project space; clone the repo there with
   `git submodule update --init --recursive`.
2. The conda env: `module load 2025 && module load Anaconda3/2025.06-1`, then use
   `conda run -n rpg-uva …` (the job scripts do this).
3. An OpenAI API key for `text-embedding-3-large`. Put it in an untracked
   `configs/rpg/local.yaml` as `openai_api_key: sk-…`.
4. The new category available to `genrec` as a dataset (raw Amazon Reviews 2014
   data + dataset config) and a preset `configs/rpg/repro/<ds>.yaml` (copy an
   existing preset and change the `dataset`/`category`). The four existing presets
   under `configs/rpg/repro/` are the templates.

### 1. Prepare semantic IDs (all `m`)

```bash
for M in 4 8 16 32 64; do
  conda run -n rpg-uva python scripts/rpg_prepare_semantic_ids.py --preset <ds> --n_codebook $M
done
```

The first run computes and caches the OpenAI embeddings; later `m` reuse the cache
and only re-run OPQ tokenization.

### 2. Joint training sweep (`m × lr × temperature`)

```bash
cd jobs/reproduction/rpg/grid
mkdir -p ../../../../output/reproduction/rpg/grid/train
DATASETS="<ds>" sbatch --array=0-44 -p gpu_h100 run_train_grid.sh   # 5 m × 3 lr × 3 temp
```

`run_train_grid.sh` reads `DATASETS`, `MVALS`, `LRS`, `TEMPS`, `SEEDS` from the
environment, so no per-dataset edits are needed. Recompute `--array` as
`N_DS × N_M × N_LR × N_TEMP × N_SEED − 1` (one dataset at one seed = `0-44`; two
datasets = `0-89`). Grid checkpoints are discarded; selection comes from the logs.

### 3. Collect and pick the winner

```bash
cd /projects/prjs2120/groups/group_16/code/RPG-uva
python scripts/collect_grid_results.py
```

`train_grid.csv` now has an `m` column. Per dataset, pick the `(m, lr, temperature)`
row with the highest `val_ndcg10_mean` — call it `m*`, `lr*`, `t*`.

### 4. Link the winning checkpoint

The sweep kept every cell's checkpoint. Link the winner `(m*, lr*, t*)` to the name
the inference sweeps glob (`rpg_sweep_m<m*>_<ds>-*.pth`):

```bash
cd artifacts/rpg/ckpt
ln -s rpg_sweep_m<m*>_<ds>_lr<lr*>_t<t*>_s2024-*.pth rpg_sweep_m<m*>_<ds>-best.pth
```

The `m`-scaling curve (Claim 2) is the slice of `train_grid.csv` at `lr*`, `t*` — no
re-decoding needed. To reclaim quota, delete the non-winning grid checkpoints once
the winner is linked.

### 5. Decode-parameter grid (validation) then confirm (test)

Append the new dataset to the hard-coded arrays in **`run_decode_grid.sh`**,
**`run_decode_confirm.sh`**, and **`run_infer_grid.sh`** (lines ~28–32 in each):
add `<ds>` to `DATASETS`, `<Category>` to `CATEGORIES`, and `m*` to `BEST_M`; for
`run_infer_grid.sh` also append the per-dataset base to `BASE_B`/`BASE_K`/`BASE_Q`.
Then bump each `#SBATCH --array` (decode grid = `N_DS × 6 − 1`; confirm/infer =
`N_DS − 1`).

```bash
sbatch -p gpu_h100 run_decode_grid.sh          # full b×k×q on validation, 3 seeds
python ../../../../scripts/collect_grid_results.py   # writes results/decode_val_cluster.csv
sbatch -p gpu_h100 run_decode_confirm.sh       # re-decode the val-selected cluster on test, 10 seeds
python ../../../../scripts/collect_grid_results.py   # writes results/decode_test_selected.csv
```

### 6. Inference candidate budget (NDCG@50/@100)

```bash
sbatch -p gpu_h100 run_infer_grid.sh
python ../../../../scripts/collect_grid_results.py   # writes infer_grid.csv
```

### 7. SASRec baseline (optional, for comparison)

SASRec jobs are per-dataset directories under `jobs/reproduction/sasrec/<ds>/`
(`prepare_data.sh`, `train.sh`, `eval.sh`). Copy an existing dataset directory,
change the preset, and submit via that directory's scripts.

### 8. Analyze

`scp` `results/*.csv` and `results/figures/*` back to your machine;
`notebooks/results_analysis.ipynb` reads them.

### Notes

- Selection is always on **validation**; reported scores are on **test** (the
  authors' protocol).
- `run_train_grid.sh` is fully env-driven, but `run_decode_grid.sh`,
  `run_decode_confirm.sh`, and `run_infer_grid.sh` still use hard-coded dataset
  arrays — extend those arrays as in step 5. `BEST_M` for the new dataset is only
  known after step 3, so these arrays cannot be filled earlier.
- Disk quota: the eval scripts can write large `per_user_metrics.*` dumps. Pass
  `--no-per-user-output` (already set in `run_decode_confirm.sh`) or delete them
  under `output/reproduction/rpg/grid/` if quota runs low; only `summary.json` is
  consumed by the collector.
