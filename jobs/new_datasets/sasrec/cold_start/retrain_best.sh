#!/bin/bash

# Submit from jobs/new_datasets/sasrec/cold_start so these relative output paths resolve correctly.
#SBATCH --job-name=sasrec_new_best_retrain
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=../../../../output/new_datasets/sasrec/cold_start/%x-%j.out
#SBATCH --error=../../../../output/new_datasets/sasrec/cold_start/%x-%j.err

set -euo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  PWD_REAL="$(pwd -P)"
  if [[ "${PWD_REAL}" == "${RUNNER_DIR}" ]]; then
    SCRIPT_DIR="${RUNNER_DIR}"
  else
    SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
  fi
else
  SCRIPT_DIR="${RUNNER_DIR}"
  PWD_REAL="$(pwd -P)"
  if [[ "${PWD_REAL}" != "${SCRIPT_DIR}" ]]; then
    echo "ERROR: run this script from ${SCRIPT_DIR}" >&2
    echo "Run:" >&2
    echo "  cd ${SCRIPT_DIR}" >&2
    echo "  COLD_START_DATASET_SLUG=video_games sbatch ./retrain_best.sh" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output/new_datasets/sasrec/cold_start"
ENV_PREFIX="${REPO_ROOT}/artifacts/conda/rpg-uva"
COLD_START_DATASET_SLUG="${COLD_START_DATASET_SLUG:-video_games}"

BEST_INFO="$("${ENV_PREFIX}/bin/python" "${REPO_ROOT}/scripts/find_best_new_dataset_sasrec_grid.py" --dataset "${COLD_START_DATASET_SLUG}" --format shell)"
eval "${BEST_INFO}"

DATA_FILE="${REPO_ROOT}/artifacts/sasrec/data/${DATASET_CATEGORY}/${DATASET_CATEGORY}.txt"

mkdir -p "${OUTPUT_DIR}"
source "${REPO_ROOT}/jobs/lib/runtime_stats.sh"
runtime_stats_init "${OUTPUT_DIR}" "sasrec_retrain_best_${COLD_START_DATASET_SLUG}"
runtime_stats_start_gpu_monitor
trap runtime_stats_finish EXIT

if [[ ! -f "${DATA_FILE}" ]]; then
  echo "ERROR: missing SASRec data file: ${DATA_FILE}" >&2
  exit 3
fi

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "${REPO_ROOT}"

runtime_stats_run "${ENV_PREFIX}/bin/python" scripts/sasrec.py \
  --preset "${DATASET_SLUG}" \
  --dataset "${DATASET_CATEGORY}" \
  --run_id "${RUN_ID}" \
  --seed 2024 \
  --lr "${LR}" \
  --hidden_dropout_prob "${DROPOUT}" \
  --attention_probs_dropout_prob "${DROPOUT}" \
  --num_hidden_layers "${BLOCKS}"
