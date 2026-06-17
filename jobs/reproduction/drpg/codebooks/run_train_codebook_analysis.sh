#!/bin/bash
# Ablation study for DRPG: tests various codebook sizes
# while holding n_views and denoise_steps constant (and equal to each other).
#
# Default dataset is 'beauty' and n_views/denoise_steps is 1.
# Options for n_codebook are {16, 32, 64}.
#
# Submit from this directory:
#   cd jobs/reproduction/drpg/grid
#   mkdir -p ../../../../output/reproduction/drpg/grid/train
#
# Default run (Beauty, views/steps=1):
#   sbatch run_ablation_codebook.sh
#
# Override dataset and the shared views/steps value:
#   DATASET=toys_and_games SHARED_STEPS=8 sbatch run_ablation_codebook.sh

#SBATCH --job-name=drpg_ablation_codebook
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --array=0-2    # Exactly 3 jobs (0 through 2)
#SBATCH --output=../../../../output/reproduction/drpg/grid/train/%x-%A_%a.out
#SBATCH --error=../../../../output/reproduction/drpg/grid/train/%x-%A_%a.err

set -euo pipefail
SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"

# 1. Read variables with defaults (allows easy user overrides)
DATASET="${DATASET:-beauty}"
SHARED_STEPS="${SHARED_STEPS:-1}" # This acts as the value for BOTH views and steps

# 2. Define the array of codebook sizes to test
OPTIONS=(16 32 64)

idx=${SLURM_ARRAY_TASK_ID:-0}

# Safety check
if (( idx >= ${#OPTIONS[@]} )); then
  echo "ERROR: array index ${idx} exceeds total options ${#OPTIONS[@]}" >&2
  exit 2
fi

# 3. Extract the specific value for this SLURM array task
N_CODEBOOK=${OPTIONS[$idx]}
N_VIEWS=$SHARED_STEPS
DENOISE_STEPS=$SHARED_STEPS

RUN_ID="drpg_codebook_ablation_${DATASET}_m${N_CODEBOOK}_nv${N_VIEWS}_ds${DENOISE_STEPS}"
OUT_DIR="${REPO_ROOT}/output/reproduction/drpg/grid/train/${DATASET}"
mkdir -p "${OUT_DIR}"

echo "START ABLATION: dataset=${DATASET} m=${N_CODEBOOK} views=${N_VIEWS} steps=${DENOISE_STEPS} run_id=${RUN_ID}"

module purge
module load 2025
module load Anaconda3/2025.06-1
cd "${REPO_ROOT}"

# 4. Call the drpg script
conda run -n rpg-uva python scripts/drpg.py \
  --preset "${DATASET}" \
  --n_codebook "${N_CODEBOOK}" \
  --n_views "${N_VIEWS}" \
  --denoise_steps "${DENOISE_STEPS}" \
  --run_id "${RUN_ID}"

echo "END ABLATION: dataset=${DATASET} m=${N_CODEBOOK} views=${N_VIEWS} steps=${DENOISE_STEPS}"
