#!/bin/bash

# Joint RPG training sweep over semantic-ID length m x learning rate x temperature.
# Mirrors the original paper's tuning (lr x temp x #digit = 3 x 3 x 5 = 45 cells) and
# runs the whole grid in one job array, like the decode/inference sweeps. Each cell
# logs val + test metrics (parsed by scripts/collect_grid_results.py) and keeps its
# checkpoint, so the inference sweeps can re-decode the selected one.
#
# Sweep a new dataset (5 m x 3 lr x 3 temp x 1 seed = 45 cells):
#   cd jobs/reproduction/rpg/grid
#   mkdir -p ../../../../output/reproduction/rpg/grid/train
#   DATASETS="office_products" sbatch --array=0-44 -p gpu_h100 run_train_grid.sh
#   # recompute --array as N_DS * N_M * N_LR * N_TEMP * N_SEED - 1
#
# Env overrides (space-separated): DATASETS, MVALS, LRS, TEMPS, SEEDS.

#SBATCH --job-name=rpg_train_grid
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-179
#SBATCH --output=../../../../output/reproduction/rpg/grid/train/%x-%A_%a.out
#SBATCH --error=../../../../output/reproduction/rpg/grid/train/%x-%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"

# Datasets and grid axes. The lr/temp grid is the paper's (uniform across datasets),
# so new datasets need no per-dataset defaults.
read -r -a DATASETS <<< "${DATASETS:-sports_and_outdoors beauty toys_and_games cds_and_vinyl}"
read -r -a MVALS    <<< "${MVALS:-4 8 16 32 64}"
read -r -a LRS      <<< "${LRS:-0.001 0.003 0.01}"
read -r -a TEMPS    <<< "${TEMPS:-0.03 0.05 0.07}"
read -r -a SEEDS    <<< "${SEEDS:-2024}"

N_M=${#MVALS[@]}
N_LR=${#LRS[@]}
N_TEMP=${#TEMPS[@]}
N_SEED=${#SEEDS[@]}
CELLS_PER_DS=$(( N_M * N_LR * N_TEMP * N_SEED ))
TOTAL=$(( ${#DATASETS[@]} * CELLS_PER_DS ))

idx=${SLURM_ARRAY_TASK_ID:-0}
if (( idx >= TOTAL )); then
  echo "ERROR: array index ${idx} exceeds ${TOTAL} cells; set --array=0-$(( TOTAL - 1 ))" >&2
  exit 2
fi

# Nested decode: ds, then m, then lr, then temp, then seed.
ds_idx=$(( idx / CELLS_PER_DS ))
rem=$(( idx % CELLS_PER_DS ))
m_idx=$(( rem / (N_LR * N_TEMP * N_SEED) ))
rem2=$(( rem % (N_LR * N_TEMP * N_SEED) ))
lr_idx=$(( rem2 / (N_TEMP * N_SEED) ))
rem3=$(( rem2 % (N_TEMP * N_SEED) ))
temp_idx=$(( rem3 / N_SEED ))
seed_idx=$(( rem3 % N_SEED ))

DS=${DATASETS[$ds_idx]}
M=${MVALS[$m_idx]}
LR=${LRS[$lr_idx]}
TEMP=${TEMPS[$temp_idx]}
SEED=${SEEDS[$seed_idx]}
RUN_ID="rpg_sweep_m${M}_${DS}_lr${LR}_t${TEMP}_s${SEED}"

echo "TRAIN_GRID_START dataset=${DS} m=${M} lr=${LR} temperature=${TEMP} seed=${SEED} run_id=${RUN_ID}"

module purge
module load 2025
module load Anaconda3/2025.06-1
cd "${REPO_ROOT}"

conda run -n rpg-uva python scripts/rpg.py \
  --preset "${DS}" \
  --n_codebook "${M}" \
  --lr "${LR}" \
  --temperature "${TEMP}" \
  --rand_seed "${SEED}" \
  --run_id "${RUN_ID}"

echo "TRAIN_GRID_END dataset=${DS} m=${M} lr=${LR} temperature=${TEMP} seed=${SEED}"
