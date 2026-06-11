#!/bin/bash

# Training-hyperparameter grid for RPG: full retrain per (dataset, lr, temperature)
# cell, then the pipeline's built-in test evaluation. Codebook length m is held at
# each dataset's best-m (Sports 16, Beauty 32) 
#
# Flattened array over 2 datasets x len(LRS) x len(TEMPS) cells (default 2x3x3 = 18).
# Each task trains one cell from scratch and logs "Test Results: {...}" to its .err.
#
# Submit from this directory:
#   cd jobs/reproduction/rpg/grid
#   mkdir -p ../../../../output/reproduction/rpg/grid/train
#   sbatch run_train_grid.sh
# Override grids via env: LRS, TEMPS (space-separated). Keep --array in sync with
# the cell count: 2 * |LRS| * |TEMPS| - 1.

#SBATCH --job-name=rpg_train_grid
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-17
#SBATCH --output=../../../../output/reproduction/rpg/grid/train/%x-%A_%a.out
#SBATCH --error=../../../../output/reproduction/rpg/grid/train/%x-%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"

DATASETS=(sports_and_outdoors beauty)
BEST_M=(16 32)
read -r -a LRS   <<< "${LRS:-0.001 0.003 0.01}"
read -r -a TEMPS <<< "${TEMPS:-0.01 0.03 0.07}"

N_LR=${#LRS[@]}
N_TEMP=${#TEMPS[@]}
CELLS_PER_DS=$(( N_LR * N_TEMP ))

idx=${SLURM_ARRAY_TASK_ID:-0}
ds_idx=$(( idx / CELLS_PER_DS ))
rem=$(( idx % CELLS_PER_DS ))
lr_idx=$(( rem / N_TEMP ))
temp_idx=$(( rem % N_TEMP ))

if (( ds_idx >= ${#DATASETS[@]} )); then
  echo "ERROR: array index ${idx} exceeds ${#DATASETS[@]}x${CELLS_PER_DS} cells; fix --array" >&2
  exit 2
fi

DS=${DATASETS[$ds_idx]}
M=${BEST_M[$ds_idx]}
LR=${LRS[$lr_idx]}
TEMP=${TEMPS[$temp_idx]}
RUN_ID="rpg_grid_${DS}_lr${LR}_t${TEMP}"
OUT_DIR="${REPO_ROOT}/output/reproduction/rpg/grid/train/${DS}"

mkdir -p "${OUT_DIR}"

echo "TRAIN_GRID_START dataset=${DS} m=${M} lr=${LR} temperature=${TEMP} run_id=${RUN_ID}"

module purge
module load 2025
module load Anaconda3/2025.06-1
cd "${REPO_ROOT}"

conda run -n rpg-uva python scripts/rpg.py \
  --preset "${DS}" \
  --n_codebook "${M}" \
  --lr "${LR}" \
  --temperature "${TEMP}" \
  --run_id "${RUN_ID}"

echo "TRAIN_GRID_END dataset=${DS} lr=${LR} temperature=${TEMP}"
