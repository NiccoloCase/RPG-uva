#!/bin/bash

# Submit from jobs/reproduction/rpg/cold_start so these relative output paths resolve correctly.
#SBATCH --job-name=rpg_cold_start
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=../../../../output/reproduction/rpg/cold_start/%x-%j.out
#SBATCH --error=../../../../output/reproduction/rpg/cold_start/%x-%j.err

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
    echo "  bash ./run_cold_start.sh $(cd "${SCRIPT_DIR}/../../../.." && pwd)/artifacts/rpg/ckpt/model.pth" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output/reproduction/rpg/cold_start"
ENV_PREFIX="${REPO_ROOT}/artifacts/conda/rpg-uva"
COLD_START_CONFIG_DEFAULT="${REPO_ROOT}/configs/rpg/repro/sports_and_outdoors.yaml"
CHECKPOINT_PATH="${1:-${CHECKPOINT_PATH:-}}"
shift_count=0
if [[ $# -ge 1 ]]; then
  shift_count=1
fi
COLD_START_CONFIG="${COLD_START_CONFIG:-${COLD_START_CONFIG_DEFAULT}}"
COLD_START_OUTPUT_DIR="${COLD_START_OUTPUT_DIR:-${REPO_ROOT}/artifacts/rpg/cold_start}"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: provide the checkpoint path as the first argument or CHECKPOINT_PATH env var." >&2
  exit 3
fi

if [[ "${CHECKPOINT_PATH}" == *"<"* || "${CHECKPOINT_PATH}" == *">"* ]]; then
  echo "ERROR: checkpoint path contains angle-bracket placeholders: ${CHECKPOINT_PATH}" >&2
  echo "Use a real absolute path under ${REPO_ROOT}, for example:" >&2
  echo "  ${REPO_ROOT}/artifacts/rpg/ckpt/model.pth" >&2
  exit 4
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: checkpoint file not found: ${CHECKPOINT_PATH}" >&2
  exit 5
fi

if [[ ! -f "${COLD_START_CONFIG}" ]]; then
  echo "ERROR: cold-start config file not found: ${COLD_START_CONFIG}" >&2
  exit 6
fi

if [[ ${shift_count} -eq 1 ]]; then
  shift
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${COLD_START_OUTPUT_DIR}"

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "${REPO_ROOT}"

conda run -p "${ENV_PREFIX}" python scripts/rpg_cold_start.py \
  run \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config "${COLD_START_CONFIG}" \
  --output-dir "${COLD_START_OUTPUT_DIR}" \
  "$@"
