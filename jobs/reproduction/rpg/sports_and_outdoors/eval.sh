#!/bin/bash

# Submit from jobs/reproduction/rpg/sports_and_outdoors so these relative output paths resolve correctly.
#SBATCH --partition=gpu_a100
#SBATCH --job-name=rpg_sports_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --output=../../../../output/reproduction/rpg/sports_and_outdoors/%x-%j.out
#SBATCH --error=../../../../output/reproduction/rpg/sports_and_outdoors/%x-%j.err

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
    echo "  bash ./eval.sh" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output/reproduction/rpg/sports_and_outdoors"
DATA_DIR="${REPO_ROOT}/artifacts/rpg/cache/AmazonReviews2014/Sports_and_Outdoors"
CKPT_DIR="${REPO_ROOT}/artifacts/rpg/ckpt"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
EVAL_SEED="${EVAL_SEED:-2024}"

mkdir -p "${OUTPUT_DIR}"
source "${REPO_ROOT}/jobs/lib/runtime_stats.sh"
runtime_stats_init "${OUTPUT_DIR}" "rpg_sports_eval"
runtime_stats_start_gpu_monitor
trap runtime_stats_finish EXIT

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "ERROR: missing cached dataset directory: ${DATA_DIR}" >&2
  exit 3
fi

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="$(
    find "${CKPT_DIR}" -maxdepth 1 -type f -name 'rpg_repro_sports_and_outdoors-*.pth' -printf '%T@ %p\n' \
      | sort -nr \
      | sed -n '1s/^[^ ]* //p'
  )"
fi

if [[ -z "${CHECKPOINT_PATH}" || ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: could not find a Sports checkpoint in ${CKPT_DIR}" >&2
  echo "Set CHECKPOINT_PATH=/path/to/checkpoint.pth to override." >&2
  exit 4
fi

module purge
module load 2025
module load Anaconda3/2025.06-1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rpg-uva

cd "${REPO_ROOT}"

echo "Evaluating checkpoint: ${CHECKPOINT_PATH}"
echo "Eval seed: ${EVAL_SEED}"
runtime_stats_run python3 scripts/rpg_eval.py \
  --preset sports_and_outdoors \
  --checkpoint "${CHECKPOINT_PATH}" \
  --eval-seed "${EVAL_SEED}"
