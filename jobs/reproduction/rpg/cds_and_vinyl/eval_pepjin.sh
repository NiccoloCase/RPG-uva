#!/bin/bash

# Submit from jobs/reproduction/rpg/cds_and_vinyl so these relative output paths resolve correctly.
#SBATCH --partition=gpu_a100
#SBATCH --job-name=rpg_cds_and_vinyl_pepjin_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --output=../../../../output/reproduction/rpg/cds_and_vinyl/%x-%j.out
#SBATCH --error=../../../../output/reproduction/rpg/cds_and_vinyl/%x-%j.err

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
    echo "  bash ./eval_pepjin.sh" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output/reproduction/rpg/cds_and_vinyl"
DATA_DIR="${REPO_ROOT}/artifacts/rpg/cache/AmazonReviews2014/CDs_and_Vinyl"
CKPT_DIR="${REPO_ROOT}/artifacts/pepjin/ckpt"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
EVAL_SEED="${EVAL_SEED:-2024}"

mkdir -p "${OUTPUT_DIR}"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "ERROR: missing cached CDs and Vinyl dataset directory: ${DATA_DIR}" >&2
  exit 3
fi

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="$(
    find "${CKPT_DIR}" -maxdepth 1 -type f -name '*CDs_and_Vinyl*.pth' -printf '%T@ %p\n' \
      | sort -nr \
      | sed -n '1s/^[^ ]* //p'
  )"
fi

if [[ -z "${CHECKPOINT_PATH}" || ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: could not find a CDs and Vinyl checkpoint in ${CKPT_DIR}" >&2
  echo "Set CHECKPOINT_PATH=/path/to/checkpoint.pth to override." >&2
  exit 4
fi

module purge
module load 2025
module load Anaconda3/2025.06-1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rpg-uva

cd "${REPO_ROOT}"

python3 scripts/rpg_eval.py \
  --preset cds_and_vinyl \
  --checkpoint "${CHECKPOINT_PATH}" \
  --eval-seed "${EVAL_SEED}" \
  --no-root-config \
  --config configs/rpg/root_pepjin.yaml \
  --cache_dir artifacts/rpg/cache \
  --num_beams 20 \
  --n_edges 500 \
  --propagation_steps 5
