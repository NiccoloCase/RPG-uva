#!/bin/bash
# Training-hyperparameter grid for DRPG: full retrain per (dataset, lr, diffusion_heads)
# cell, then the pipeline's built-in test evaluation.
#
# For fair comparison with RPG, codebook length (m) is tied to the RPG values:
#   - beauty: 32
#   - toys_and_games: 16
#   - sports_and_outdoors: 16
#
# n_views and denoise_steps are held constant at 1.
# Model execution is routed through diffusion/genrec via scripts/drpg.py.
#
# Flattened array over 3 datasets x len(LRS) x len(DHEADS) cells (3x2x2 = 12).
#
# Submit from this directory:
#   cd jobs/reproduction/drpg/grid
#   mkdir -p ../../../../output/reproduction/drpg/grid/train
#   sbatch run_train_grid.sh

#SBATCH --job-name=drpg_train_grid
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-11    # 12 total jobs
#SBATCH --output=../../../../output/reproduction/drpg/grid/train/%x-%A_%a.out
#SBATCH --error=../../../../output/reproduction/drpg/grid/train/%x-%A_%a.err

set -euo pipefail
SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"

# The datasets and their corresponding n_codebook values
DATASETS=(beauty toys_and_games sports_and_outdoors)
BEST_M=(32 16 16)

# Read hyperparameter options
read -r -a LRS    <<< "${LRS:-0.003 0.01}"
read -r -a DHEADS <<< "${DHEADS:-4 8}"

N_LR=${#LRS[@]}
N_HEADS=${#DHEADS[@]}
CELLS_PER_DS=$(( N_LR * N_HEADS ))

idx=${SLURM_ARRAY_TASK_ID:-0}

# Modulo math to unflatten the 1D array index
ds_idx=$(( idx / CELLS_PER_DS ))
rem=$(( idx % CELLS_PER_DS ))
lr_idx=$(( rem / N_HEADS ))
head_idx=$(( rem % N_HEADS ))

# Safety check
if (( ds_idx >= ${#DATASETS[@]} )); then
  echo "ERROR: array index ${idx} exceeds ${#DATASETS[@]}x${CELLS_PER_DS} cells; fix --array" >&2
  exit 2
fi

DS=${DATASETS[$ds_idx]}
M=${BEST_M[$ds_idx]}
LR=${LRS[$lr_idx]}
DH=${DHEADS[$head_idx]}

# Fixed parameters for this run
NV=1
DS_STEPS=1

RUN_ID="drpg_grid_${DS}_lr${LR}_dh${DH}"
OUT_DIR="${REPO_ROOT}/output/reproduction/drpg/grid/train/${DS}"
mkdir -p "${OUT_DIR}"

echo "TRAIN_GRID_START dataset=${DS} m=${M} lr=${LR} diffusion_heads=${DH} n_views=${NV} denoise_steps=${DS_STEPS} run_id=${RUN_ID}"

module purge
module load 2025
module load Anaconda3/2025.06-1
cd "${REPO_ROOT}"

# Call the drpg script (which points to diffusion/genrec)
conda run -n rpg-uva python scripts/drpg.py \
  --preset "${DS}" \
  --n_codebook "${M}" \
  --lr "${LR}" \
  --diffusion_heads "${DH}" \
  --n_views "${NV}" \
  --denoise_steps "${DS_STEPS}" \
  --run_id "${RUN_ID}"

echo "TRAIN_GRID_END dataset=${DS} lr=${LR} diffusion_heads=${DH}"
