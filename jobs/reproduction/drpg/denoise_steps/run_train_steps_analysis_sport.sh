#!/bin/bash
# Ablation study for DRPG: tests various coupled values of (n_views, denoise_steps)
# while holding n_codebook constant.
#
# Submit from this directory:
#   cd jobs/reproduction/drpg/denoise_steps
#   mkdir -p ../../../../output/reproduction/drpg/denoise_steps/train
#

#SBATCH --job-name=drpg_denoise_steps
#SBATCH --partition=gpu_h100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --array=0-3    # Exactly 4 jobs (0 through 3)
#SBATCH --output=../../../../output/reproduction/drpg/denoise_steps/train/%x-%A_%a.out
#SBATCH --error=../../../../output/reproduction/drpg/denoise_steps/train/%x-%A_%a.err

set -euo pipefail
SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"

# 1. Read variables with defaults (allows easy user overrides)
DATASET="sports_and_outdoors"
N_CODEBOOK="16"

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

RUN_ID="drpg_diff_steps_${DATASET}_m${N_CODEBOOK}_nv${N_VIEWS}_ds${DENOISE_STEPS}"
OUT_DIR="${REPO_ROOT}/output/reproduction/drpg/denoise_steps/train/${DATASET}"
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
  --denoise_inference_steps "${DENOISE_STEPS}" \
  --run_id "${RUN_ID}"

echo "END ABLATION: dataset=${DATASET} m=${N_CODEBOOK} views=${N_VIEWS} steps=${DENOISE_STEPS}"
