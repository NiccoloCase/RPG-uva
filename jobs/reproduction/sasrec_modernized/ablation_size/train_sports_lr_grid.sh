#!/bin/bash

# Submit from jobs/reproduction/sasrec_modernized/ablation_size so these relative output paths resolve correctly.
# Partition checked on 2026-06-19: gpu_a100 and gpu_h100 each had 3 idle nodes;
# prefer gpu_a100 per repo guidance for normal full GPU runs.
#SBATCH --partition=gpu_a100
#SBATCH --job-name=sasrec_mod_sports_size_lr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --array=0-2
#SBATCH --output=../../../../output/reproduction/sasrec_modernized/ablation_size/lr_grid/%x-%A_%a.out
#SBATCH --error=../../../../output/reproduction/sasrec_modernized/ablation_size/lr_grid/%x-%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}" && pwd -P)"
PWD_REAL="$(pwd -P)"
if [[ "${PWD_REAL}" != "${SCRIPT_DIR}" ]]; then
  echo "ERROR: run this script from ${SCRIPT_DIR}" >&2
  echo "Run:" >&2
  echo "  cd ${SCRIPT_DIR}" >&2
  echo "  sbatch ./train_sports_lr_grid.sh" >&2
  exit 2
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"
OUTPUT_DIR="${REPO_ROOT}/output/reproduction/sasrec_modernized/ablation_size/lr_grid"
CKPT_DIR="${REPO_ROOT}/artifacts/sasrec/ckpt/ablation_size/lr_grid"
DATA_FILE="${REPO_ROOT}/artifacts/sasrec/data/Sports_and_Outdoors/Sports_and_Outdoors.txt"
ENV_PREFIX="${REPO_ROOT}/artifacts/conda/rpg-uva"

read -r -a LRS <<< "${LRS:-0.001 0.0005 0.0003}"
IDX=${SLURM_ARRAY_TASK_ID:-0}
if (( IDX < 0 || IDX >= ${#LRS[@]} )); then
  echo "ERROR: array index ${IDX} is out of range for ${#LRS[@]} learning rates; fix --array" >&2
  exit 2
fi
LR="${LRS[$IDX]}"
LR_TAG="${LR//./p}"
RUN_ID="sasrec_modernized_sports_and_outdoors_size_match_e300_lr${LR_TAG}"

mkdir -p "${OUTPUT_DIR}" "${CKPT_DIR}"
source "${REPO_ROOT}/jobs/lib/runtime_stats.sh"
runtime_stats_init "${OUTPUT_DIR}" "${RUN_ID}"
runtime_stats_start_gpu_monitor
trap runtime_stats_finish EXIT

if [[ ! -f "${DATA_FILE}" ]]; then
  echo "ERROR: missing SASRec data file: ${DATA_FILE}" >&2
  exit 3
fi

module purge
module load 2025
module load Anaconda3/2025.06-1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

cd "${REPO_ROOT}"

runtime_stats_run python3 scripts/sasrec_modernized.py \
  --preset sports_and_outdoors \
  --dataset Sports_and_Outdoors \
  --epochs 300 \
  --lr "${LR}" \
  --ckpt_dir artifacts/sasrec/ckpt/ablation_size/lr_grid \
  --run_id "${RUN_ID}" \
  --hidden_size 326
