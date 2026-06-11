#!/bin/bash

# Submit from jobs/reproduction/rpg/perf so these relative output paths resolve correctly.
#SBATCH --job-name=rpg_perf_profile
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=../../../../output/reproduction/rpg/perf/%x-%j.out
#SBATCH --error=../../../../output/reproduction/rpg/perf/%x-%j.err

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PWD_REAL="$(pwd -P)"
  if [[ "${PWD_REAL}" != "${SCRIPT_DIR}" ]]; then
    echo "ERROR: run this script from ${SCRIPT_DIR}" >&2
    echo "Run:" >&2
    echo "  cd ${SCRIPT_DIR}" >&2
    echo "  bash ./profile_inference.sh $(cd "${SCRIPT_DIR}/../../.." && pwd)/artifacts/rpg/ckpt/model.pth" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output/reproduction/perf"
PERF_CONFIG_DEFAULT="${REPO_ROOT}/configs/rpg/perf/sports.yaml"
CHECKPOINT_PATH="${1:-${CHECKPOINT_PATH:-}}"
shift_count=0
if [[ $# -ge 1 ]]; then
  shift_count=1
fi
PERF_CONFIG="${PERF_CONFIG:-${PERF_CONFIG_DEFAULT}}"
PERF_DATASET_SLUG="${PERF_DATASET_SLUG:-sports_and_outdoors}"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="$(find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name "rpg_repro_${PERF_DATASET_SLUG}-*.pth" | sort | tail -n 1)"
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
  echo "  ${CHECKPOINT_DIR}/rpg_repro_${PERF_DATASET_SLUG}-<timestamp>.pth" >&2
  exit 4
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: checkpoint file not found: ${CHECKPOINT_PATH}" >&2
  exit 5
fi

if [[ ${shift_count} -eq 1 ]]; then
  shift
fi

mkdir -p "${OUTPUT_DIR}"

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "${REPO_ROOT}"

conda run -p "${ENV_PREFIX}" python scripts/rpg_perf.py \
  profile \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config "${PERF_CONFIG}" \
  --profile-only \
  "$@"
