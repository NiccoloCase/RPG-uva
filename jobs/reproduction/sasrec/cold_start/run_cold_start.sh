#!/bin/bash

# Submit from jobs/reproduction/sasrec/cold_start so these relative output paths resolve correctly.
#SBATCH --job-name=sasrec_cold_start
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=../../../../output/reproduction/sasrec/cold_start/%x-%j.out
#SBATCH --error=../../../../output/reproduction/sasrec/cold_start/%x-%j.err

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
    echo "  bash ./run_cold_start.sh" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output/reproduction/sasrec/cold_start"
ENV_PREFIX="${REPO_ROOT}/artifacts/conda/rpg-uva"
COLD_START_DATASET_SLUG="${COLD_START_DATASET_SLUG:-sports_and_outdoors}"
COLD_START_PRESET="${COLD_START_PRESET:-${COLD_START_DATASET_SLUG}}"
COLD_START_DATASET="${COLD_START_DATASET:-Sports_and_Outdoors}"
COLD_START_OUTPUT_DIR="${COLD_START_OUTPUT_DIR:-${REPO_ROOT}/artifacts/sasrec/cold_start}"
LOCAL_CHECKPOINT_DIR="${REPO_ROOT}/artifacts/sasrec_modernized/ckpt"
SHARED_CHECKPOINT_DIR="/projects/prjs2120/groups/group_16/artifacts/sasrec_modernized/ckpt"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
CHECKPOINT_PATH="${1:-${CHECKPOINT_PATH:-}}"
shift_count=0
if [[ $# -ge 1 ]]; then
  shift_count=1
fi

DEFAULT_CHECKPOINT_BASENAME="sasrec_modernized_${COLD_START_DATASET_SLUG}.pt"
if [[ -z "${CHECKPOINT_DIR}" ]]; then
  if [[ -f "${LOCAL_CHECKPOINT_DIR}/${DEFAULT_CHECKPOINT_BASENAME}" ]]; then
    CHECKPOINT_DIR="${LOCAL_CHECKPOINT_DIR}"
  elif [[ -f "${SHARED_CHECKPOINT_DIR}/${DEFAULT_CHECKPOINT_BASENAME}" ]]; then
    CHECKPOINT_DIR="${SHARED_CHECKPOINT_DIR}"
  else
    CHECKPOINT_DIR="${LOCAL_CHECKPOINT_DIR}"
  fi
fi

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="${CHECKPOINT_DIR}/${DEFAULT_CHECKPOINT_BASENAME}"
fi

if [[ "${CHECKPOINT_PATH}" == *"<"* || "${CHECKPOINT_PATH}" == *">"* ]]; then
  echo "ERROR: checkpoint path contains angle-bracket placeholders: ${CHECKPOINT_PATH}" >&2
  echo "Use a real path, for example:" >&2
  echo "  ${LOCAL_CHECKPOINT_DIR}/${DEFAULT_CHECKPOINT_BASENAME}" >&2
  echo "  ${SHARED_CHECKPOINT_DIR}/${DEFAULT_CHECKPOINT_BASENAME}" >&2
  exit 3
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: SASRec checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 4
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

runtime_stats_run conda run -p "${ENV_PREFIX}" python scripts/sasrec_cold_start.py \
  run \
  --checkpoint "${CHECKPOINT_PATH}" \
  --preset "${COLD_START_PRESET}" \
  --dataset "${COLD_START_DATASET}" \
  --output-dir "${COLD_START_OUTPUT_DIR}" \
  "$@"
