#!/bin/bash

# Submit from jobs/reproduction/rpg/sports_and_outdoors so these relative output paths resolve correctly.
#SBATCH --partition=gpu_a100
#SBATCH --job-name=rpg_sports_pepjin_eval
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
    echo "  bash ./eval_pepjin.sh" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output/reproduction/rpg/sports_and_outdoors"
DATA_DIR="${REPO_ROOT}/artifacts/pepjin/cache/AmazonReviews2014/Sports_and_Outdoors"
SEM_IDS_PATH="${DATA_DIR}/processed/text-embedding-3-large_OPQ16,IVF1,PQ16x8.sem_ids"
CKPT_DIR="${REPO_ROOT}/artifacts/pepjin/ckpt"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
EVAL_SEED="${EVAL_SEED:-2024}"

mkdir -p "${OUTPUT_DIR}"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "ERROR: missing cached Sports dataset directory: ${DATA_DIR}" >&2
  exit 3
fi

if [[ ! -f "${SEM_IDS_PATH}" ]]; then
  echo "ERROR: missing Pepjin Sports semantic IDs: ${SEM_IDS_PATH}" >&2
  echo "This checkpoint must be evaluated with the semantic IDs used during training." >&2
  echo "Ask for the matching processed/text-embedding-3-large_OPQ16,IVF1,PQ16x8.sem_ids file." >&2
  exit 5
fi

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="$(
    find "${CKPT_DIR}" -maxdepth 1 -type f -name '*Sports_and_Outdoors*.pth' -printf '%T@ %p\n' \
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
echo "Using cache: ${DATA_DIR}"
echo "Using semantic IDs: ${SEM_IDS_PATH}"
echo "Eval seed: ${EVAL_SEED}"

python3 scripts/rpg_eval.py \
  --preset sports_and_outdoors \
  --checkpoint "${CHECKPOINT_PATH}" \
  --eval-seed "${EVAL_SEED}" \
  --no-root-config \
  --config configs/rpg/root_pepjin.yaml \
  --num_beams 100 \
  --n_edges 30 \
  --propagation_steps 5
