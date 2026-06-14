#!/bin/bash

# Submit from jobs/new_datasets/sasrec/eval_seeds so these relative output paths resolve correctly.
#SBATCH --job-name=sasrec_eval
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=../../../../output/new_datasets/sasrec/eval_seeds/%x-%j.out
#SBATCH --error=../../../../output/new_datasets/sasrec/eval_seeds/%x-%j.err

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
    echo "  bash ./run_eval.sh" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output/new_datasets/sasrec/eval_seeds"
ENV_PREFIX="${REPO_ROOT}/artifacts/conda/rpg-uva"
SASREC_EVAL_MODE="${SASREC_EVAL_MODE:-eval_seeds}"
SASREC_EVAL_SEEDS="${SASREC_EVAL_SEEDS:-2024,2025,2026,2027,2028,2029,2030,2031,2032,2033}"
SASREC_EVAL_SEED="${SASREC_EVAL_SEED:-2024}"
SASREC_EVAL_DATASET_SLUG="${SASREC_EVAL_DATASET_SLUG:-video_games}"
SASREC_EVAL_PRESET="${SASREC_EVAL_PRESET:-${SASREC_EVAL_DATASET_SLUG}}"
SASREC_EVAL_DATASET="${SASREC_EVAL_DATASET:-Video_Games}"
SASREC_EVAL_CONFIG_DEFAULT="${REPO_ROOT}/configs/sasrec/eval_seeds/new_datasets/${SASREC_EVAL_DATASET_SLUG}.yaml"
SASREC_EVAL_CONFIG="${SASREC_EVAL_CONFIG:-${SASREC_EVAL_CONFIG_DEFAULT}}"
SASREC_EVAL_OUTPUT_DIR="${SASREC_EVAL_OUTPUT_DIR:-${REPO_ROOT}/artifacts/sasrec/eval_seeds/new_datasets/${SASREC_EVAL_DATASET_SLUG}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/artifacts/sasrec_modernized/ckpt}"
CHECKPOINT_PATH="${1:-${CHECKPOINT_PATH:-}}"
shift_count=0
if [[ $# -ge 1 ]]; then
  shift_count=1
fi

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="${CHECKPOINT_DIR}/sasrec_modernized_${SASREC_EVAL_DATASET_SLUG}.pt"
fi

if [[ "${CHECKPOINT_PATH}" == *"<"* || "${CHECKPOINT_PATH}" == *">"* ]]; then
  echo "ERROR: checkpoint path contains angle-bracket placeholders: ${CHECKPOINT_PATH}" >&2
  echo "Use a real path under ${REPO_ROOT}, for example:" >&2
  echo "  ${CHECKPOINT_DIR}/sasrec_modernized_${SASREC_EVAL_DATASET_SLUG}.pt" >&2
  exit 3
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: SASRec checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 4
fi

if [[ ! -f "${SASREC_EVAL_CONFIG}" ]]; then
  echo "ERROR: SASRec eval config not found: ${SASREC_EVAL_CONFIG}" >&2
  exit 5
fi

if [[ ${shift_count} -eq 1 ]]; then
  shift
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SASREC_EVAL_OUTPUT_DIR}"
source "${REPO_ROOT}/jobs/lib/runtime_stats.sh"
runtime_stats_init "${OUTPUT_DIR}" "sasrec_eval_${SASREC_EVAL_DATASET_SLUG}_${SASREC_EVAL_MODE}"
runtime_stats_start_gpu_monitor
trap runtime_stats_finish EXIT

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "${REPO_ROOT}"

runtime_stats_run "${ENV_PREFIX}/bin/python" scripts/sasrec_eval.py \
  --checkpoint "${CHECKPOINT_PATH}" \
  --eval-mode "${SASREC_EVAL_MODE}" \
  --eval-seed "${SASREC_EVAL_SEED}" \
  --eval-seeds "${SASREC_EVAL_SEEDS}" \
  --preset "${SASREC_EVAL_PRESET}" \
  --dataset "${SASREC_EVAL_DATASET}" \
  --config "${SASREC_EVAL_CONFIG}" \
  --output-dir "${SASREC_EVAL_OUTPUT_DIR}" \
  "$@"
