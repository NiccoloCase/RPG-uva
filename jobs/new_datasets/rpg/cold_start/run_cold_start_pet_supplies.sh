#!/bin/bash
#SBATCH --job-name=rpg_cold_start_pet_supplies
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=../../../../output/new_datasets/rpg/cold_start/%x-%j.out
#SBATCH --error=../../../../output/new_datasets/rpg/cold_start/%x-%j.err

set -euo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  SCRIPT_DIR="${RUNNER_DIR}"
  if [[ "$(pwd -P)" != "${SCRIPT_DIR}" ]]; then
    echo "ERROR: run this script from ${SCRIPT_DIR}" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output/new_datasets/rpg/cold_start"
ENV_PREFIX="${REPO_ROOT}/artifacts/conda/rpg-uva"
COLD_START_DATASET_SLUG="pet_supplies"
COLD_START_OUTPUT_DIR="${REPO_ROOT}/artifacts/rpg/cold_start/new_datasets/${COLD_START_DATASET_SLUG}"
COLD_START_CONFIG="${REPO_ROOT}/configs/rpg/new_datasets/${COLD_START_DATASET_SLUG}.yaml"
CHECKPOINT_PATH="${REPO_ROOT}/artifacts/rpg/ckpt/rpg_sweep_m32_pet_supplies_lr0.003_t0.03_s2024.pth"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: RPG checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 3
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${COLD_START_OUTPUT_DIR}"

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "${REPO_ROOT}"

"${ENV_PREFIX}/bin/python" scripts/rpg_cold_start.py \
  run \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config "${COLD_START_CONFIG}" \
  --output-dir "${COLD_START_OUTPUT_DIR}" \
  --plot-title "RPG Cold-Start Analysis (${COLD_START_DATASET_SLUG})"
