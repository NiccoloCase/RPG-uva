#!/bin/bash

# Submit from jobs/new_datasets/sasrec/cold_start so these relative output paths resolve correctly.
#SBATCH --job-name=sasrec_new_cold_start
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=04:00:00
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
    echo "  COLD_START_DATASET_SLUG=video_games sbatch ./run_cold_start.sh" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output/new_datasets/sasrec/cold_start"
ENV_PREFIX="${REPO_ROOT}/artifacts/conda/rpg-uva"
COLD_START_DATASET_SLUG="${COLD_START_DATASET_SLUG:-video_games}"
COLD_START_OUTPUT_DIR="${COLD_START_OUTPUT_DIR:-${REPO_ROOT}/artifacts/sasrec/cold_start/new_datasets/${COLD_START_DATASET_SLUG}}"
CHECKPOINT_PATH="${1:-${CHECKPOINT_PATH:-}}"
shift_count=0
if [[ $# -ge 1 ]]; then
  shift_count=1
fi

BEST_INFO="$(${ENV_PREFIX}/bin/python ${REPO_ROOT}/scripts/find_best_new_dataset_sasrec_grid.py --dataset ${COLD_START_DATASET_SLUG} --format shell)"
eval "${BEST_INFO}"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="${REPO_ROOT}/artifacts/sasrec/ckpt/${RUN_ID}.pt"
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: SASRec checkpoint not found: ${CHECKPOINT_PATH}" >&2
  echo "Retrain the best grid cell first from ${SCRIPT_DIR}/retrain_best.sh" >&2
  exit 3
fi

if [[ ${shift_count} -eq 1 ]]; then
  shift
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${COLD_START_OUTPUT_DIR}"
source "${REPO_ROOT}/jobs/lib/runtime_stats.sh"
runtime_stats_init "${OUTPUT_DIR}" "sasrec_cold_start_${COLD_START_DATASET_SLUG}"
runtime_stats_start_gpu_monitor
trap runtime_stats_finish EXIT

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "${REPO_ROOT}"

runtime_stats_run ${ENV_PREFIX}/bin/python scripts/sasrec_cold_start.py \
  run \
  --checkpoint "${CHECKPOINT_PATH}" \
  --model-family sasrec \
  --preset "${DATASET_SLUG}" \
  --dataset "${DATASET_CATEGORY}" \
  --output-dir "${COLD_START_OUTPUT_DIR}" \
  --plot-title "SASRec Cold-Start Analysis (${DATASET_TITLE})" \
  "$@"
