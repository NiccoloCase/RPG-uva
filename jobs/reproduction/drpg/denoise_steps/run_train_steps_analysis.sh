#!/bin/bash
# Ablation study for DRPG: tests various coupled values of (n_views, denoise_steps)
# while holding n_codebook constant.
#
# Default dataset is 'beauty' and n_codebook is 64.
# Options for (n_views == denoise_steps) are {1, 4, 8, 16, 32}.
#
# Submit from this directory:
#   cd jobs/reproduction/drpg/grid
#   mkdir -p ../../../../output/reproduction/drpg/grid/train
#
# Default run (Beauty, m=64):
#   sbatch run_ablation_views_steps.sh
#
# Override dataset/codebook:
#   DATASET=toys_and_games N_CODEBOOK=16 sbatch run_ablation_views_steps.sh

#SBATCH --job-name=drpg_ablation
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --array=0-3    # Exactly 4 jobs (0 through 3)
#SBATCH --output=../../../../output/reproduction/drpg/grid/train/%x-%A_%a.out
#SBATCH --error=../../../../output/reproduction/drpg/grid/train/%x-%A_%a.err

set -euo pipefail
SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"

# 1. Read variables with defaults (allows easy user overrides)
DATASET="${DATASET:-beauty}"
N_CODEBOOK="${N_CODEBOOK:-64}"

# 2. Define the coupled options for n_views and denoise_steps
# Both will use the exact same value from this list for a given job.
OPTIONS=(1 4 8 16)

idx=${SLURM_ARRAY_TASK_ID:-0}

# Safety check
if (( idx >= ${#OPTIONS[@]} )); then
  echo "ERROR: array index ${idx} exceeds total options ${#OPTIONS[@]}" >&2
  exit 2
fi

# 3. Extract the specific value for this SLURM array task
VAL=${OPTIONS[$idx]}
N_VIEWS=$VAL
DENOISE_STEPS=$VAL

RUN_ID="drpg_step_ablation_${DATASET}_m${N_CODEBOOK}_nv${N_VIEWS}_ds${DENOISE_STEPS}"
OUT_DIR="${REPO_ROOT}/output/reproduction/drpg/grid/train/${DATASET}"
mkdir -p "${OUT_DIR}"

echo "START ABLATION: dataset=${DATASET} m=${N_CODEBOOK} views=${N_VIEWS} steps=${DENOISE_STEPS} run_id=${RUN_ID}"

module purge
module load 2025
module load Anaconda3/2025.06-1
cd "${REPO_ROOT}"

# 4. Call the drpg script
conda run -n rpg-uva python scripts/drpg.py \
  --preset="${DATASET}" \
  --n_codebook="${N_CODEBOOK}" \
  --n_views="${N_VIEWS}" \
  --denoise_steps="${DENOISE_STEPS}" \
  --run_id="${RUN_ID}"

echo "END ABLATION: dataset=${DATASET} m=${N_CODEBOOK} views=${N_VIEWS} steps=${DENOISE_STEPS}"
