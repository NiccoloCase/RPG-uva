#!/bin/bash

# Submit from jobs/new_datasets/eval_seeds so these relative output paths resolve correctly.
#SBATCH --job-name=rpg_eval_seeds
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=../../../output/new_datasets/eval_seeds/%x-%j.out
#SBATCH --error=../../../output/new_datasets/eval_seeds/%x-%j.err

set -euo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  SCRIPT_DIR="${RUNNER_DIR}"
  PWD_REAL="$(pwd -P)"
  if [[ "${PWD_REAL}" != "${SCRIPT_DIR}" ]]; then
    echo "ERROR: run this script from ${SCRIPT_DIR}" >&2
    echo "Run:" >&2
    echo "  cd ${SCRIPT_DIR}" >&2
    echo "  bash ./run_eval_seeds.sh" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${RUNNER_DIR}/../../.." && pwd)"
RPG_ARTIFACTS_ROOT="${RPG_ARTIFACTS_ROOT:-${REPO_ROOT}/artifacts/rpg}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RPG_ARTIFACTS_ROOT}/ckpt}"
OUTPUT_DIR="${REPO_ROOT}/output/new_datasets/eval_seeds"
EVAL_CONFIG_DEFAULT="${REPO_ROOT}/configs/rpg/eval_seeds/new_datasets/video_games.yaml"
CHECKPOINT_PATH="${1:-${CHECKPOINT_PATH:-}}"
shift_count=0
if [[ $# -ge 1 ]]; then
  shift_count=1
fi
EVAL_CONFIG="${EVAL_CONFIG:-${EVAL_CONFIG_DEFAULT}}"
EVAL_SEEDS="${EVAL_SEEDS:-2024,2025,2026,2027,2028,2029,2030,2031,2032,2033}"
EVAL_DATASET_SLUG="${EVAL_DATASET_SLUG:-video_games}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${RPG_ARTIFACTS_ROOT}/eval_seeds/${EVAL_DATASET_SLUG}}"

resolve_checkpoint_path() {
  local dataset_slug="$1"
  find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name "rpg_repro_${dataset_slug}-*.pth" | sort | tail -n 1
}

if [[ -z "${CHECKPOINT_PATH}" && -n "${EVAL_DATASET_SLUG}" ]]; then
  CHECKPOINT_PATH="$(resolve_checkpoint_path "${EVAL_DATASET_SLUG}")"
fi

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: provide the checkpoint path as the first argument or CHECKPOINT_PATH env var." >&2
  echo "If omitted, the script tries to resolve the latest dataset checkpoint from:" >&2
  echo "  ${CHECKPOINT_DIR}" >&2
  exit 3
fi

if [[ "${CHECKPOINT_PATH}" == *"<"* || "${CHECKPOINT_PATH}" == *">"* ]]; then
  echo "ERROR: checkpoint path contains angle-bracket placeholders: ${CHECKPOINT_PATH}" >&2
  echo "Use a real absolute path under ${REPO_ROOT}, for example:" >&2
  echo "  ${CHECKPOINT_DIR}/rpg_repro_${EVAL_DATASET_SLUG}-<timestamp>.pth" >&2
  exit 4
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: checkpoint file not found: ${CHECKPOINT_PATH}" >&2
  exit 5
fi

if [[ ! -f "${EVAL_CONFIG}" ]]; then
  echo "ERROR: eval config file not found: ${EVAL_CONFIG}" >&2
  exit 6
fi

if [[ ${shift_count} -eq 1 ]]; then
  shift
fi

ENV_PREFIX="${REPO_ROOT}/artifacts/conda/rpg-uva"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${EVAL_OUTPUT_DIR}"

source "${REPO_ROOT}/jobs/lib/runtime_stats.sh"
runtime_stats_init "${OUTPUT_DIR}" "rpg_eval_seeds_${EVAL_DATASET_SLUG}"
runtime_stats_start_gpu_monitor
trap runtime_stats_finish EXIT

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "${REPO_ROOT}"

cmd=(
  runtime_stats_run "${ENV_PREFIX}/bin/python" scripts/rpg_eval_seeds.py
  --checkpoint "${CHECKPOINT_PATH}"
  --config "${EVAL_CONFIG}"
  --eval-seeds "${EVAL_SEEDS}"
  --output-dir "${EVAL_OUTPUT_DIR}"
)

cmd+=("$@")
"${cmd[@]}"
