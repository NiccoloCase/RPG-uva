#!/bin/bash
#SBATCH --job-name=sasrec_cold_start_video_games
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
  SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  SCRIPT_DIR="${RUNNER_DIR}"
  if [[ "$(pwd -P)" != "${SCRIPT_DIR}" ]]; then
    echo "ERROR: run this script from ${SCRIPT_DIR}" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output/new_datasets/sasrec/cold_start"
ENV_PREFIX="${REPO_ROOT}/artifacts/conda/rpg-uva"
COLD_START_DATASET_SLUG="video_games"
COLD_START_OUTPUT_DIR="${REPO_ROOT}/artifacts/sasrec/cold_start/new_datasets/${COLD_START_DATASET_SLUG}"

BEST_INFO="$(${ENV_PREFIX}/bin/python ${REPO_ROOT}/scripts/find_best_new_dataset_sasrec_grid.py --dataset ${COLD_START_DATASET_SLUG} --format shell)"
eval "${BEST_INFO}"

CHECKPOINT_PATH="${REPO_ROOT}/artifacts/sasrec/ckpt/${RUN_ID}.pt"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: SASRec checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 3
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${COLD_START_OUTPUT_DIR}"

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "${REPO_ROOT}"

${ENV_PREFIX}/bin/python scripts/sasrec_cold_start.py \
  run \
  --checkpoint "${CHECKPOINT_PATH}" \
  --model-family sasrec \
  --preset "${COLD_START_DATASET_SLUG}" \
  --output-dir "${COLD_START_OUTPUT_DIR}" \
  --plot-title "SASRec Cold-Start Analysis (${COLD_START_DATASET_SLUG})"
