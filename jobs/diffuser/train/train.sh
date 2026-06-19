#!/bin/bash

# Submit from jobs/diffuser/train so these relative output paths resolve correctly.
#SBATCH --partition=gpu_a100
#SBATCH --job-name=drpg_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --output=../../../output/diffuser/train/%x-%j.out
#SBATCH --error=../../../output/diffuser/train/%x-%j.err

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
    echo "  bash ./train.sh" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output/diffuser/train"
ENV_PREFIX="${REPO_ROOT}/artifacts/conda/rpg-uva"

PRESET="${PRESET:-beauty}"
case "${PRESET}" in
  beauty)
    CATEGORY="${CATEGORY:-Beauty}"
    ;;
  sports_and_outdoors)
    CATEGORY="${CATEGORY:-Sports_and_Outdoors}"
    ;;
  toys_and_games)
    CATEGORY="${CATEGORY:-Toys_and_Games}"
    ;;
  cds_and_vinyl)
    CATEGORY="${CATEGORY:-CDs_and_Vinyl}"
    ;;
  *)
    echo "ERROR: unsupported PRESET=${PRESET}" >&2
    echo "Supported presets: beauty, sports_and_outdoors, toys_and_games, cds_and_vinyl" >&2
    exit 2
    ;;
esac

DATA_DIR="${REPO_ROOT}/artifacts/rpg/cache/AmazonReviews2014/${CATEGORY}"
RUN_ID="${RUN_ID:-drpg_${PRESET}}"
RESET_SEM_IDS="${RESET_SEM_IDS:-0}"

mkdir -p "${OUTPUT_DIR}"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "ERROR: missing cached dataset directory: ${DATA_DIR}" >&2
  exit 3
fi

if [[ "${RESET_SEM_IDS}" == "1" ]]; then
  PROCESSED_DIR="${DATA_DIR}/processed"
  if [[ -d "${PROCESSED_DIR}" ]]; then
    while IFS= read -r sem_ids_path; do
      echo "Removing stale semantic IDs: ${sem_ids_path}"
      rm -f "${sem_ids_path}"
    done < <(find "${PROCESSED_DIR}" -maxdepth 1 -type f -name '*.sem_ids' | sort)
  fi
else
  echo "RESET_SEM_IDS disabled"
fi

module purge
module load 2025
module load Anaconda3/2025.06-1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

cd "${REPO_ROOT}"

declare -a EXTRA_OVERRIDES=()

if [[ -n "${DIFFUSION_MASK_COUNTS:-}" ]]; then
  EXTRA_OVERRIDES+=(--diffusion_mask_counts "${DIFFUSION_MASK_COUNTS}")
fi
if [[ -n "${DIFFUSION_MASK_RATIOS:-}" ]]; then
  EXTRA_OVERRIDES+=(--diffusion_mask_ratios "${DIFFUSION_MASK_RATIOS}")
fi
if [[ -n "${DIFFUSION_MIN_MASKS:-}" ]]; then
  EXTRA_OVERRIDES+=(--diffusion_min_masks "${DIFFUSION_MIN_MASKS}")
fi
if [[ -n "${DIFFUSION_FINAL_LOGITS:-}" ]]; then
  EXTRA_OVERRIDES+=(--diffusion_final_logits "${DIFFUSION_FINAL_LOGITS}")
fi
if [[ -n "${EPOCHS:-}" ]]; then
  EXTRA_OVERRIDES+=(--epochs "${EPOCHS}")
fi
if [[ -n "${STEPS:-}" ]]; then
  EXTRA_OVERRIDES+=(--steps "${STEPS}")
fi
if [[ -n "${LR:-}" ]]; then
  EXTRA_OVERRIDES+=(--lr "${LR}")
fi
if [[ -n "${TRAIN_BATCH_SIZE:-}" ]]; then
  EXTRA_OVERRIDES+=(--train_batch_size "${TRAIN_BATCH_SIZE}")
fi
if [[ -n "${EVAL_BATCH_SIZE:-}" ]]; then
  EXTRA_OVERRIDES+=(--eval_batch_size "${EVAL_BATCH_SIZE}")
fi

echo "Training DRPG"
echo "PRESET=${PRESET}"
echo "CATEGORY=${CATEGORY}"
echo "RUN_ID=${RUN_ID}"
echo "DATA_DIR=${DATA_DIR}"
echo "RESET_SEM_IDS=${RESET_SEM_IDS}"
echo "LR=${LR:-preset}"
echo "Extra overrides: ${EXTRA_OVERRIDES[*]:-<none>}"

python3 scripts/rpg.py \
  --model DRPG \
  --preset "${PRESET}" \
  --run_id "${RUN_ID}" \
  "${EXTRA_OVERRIDES[@]}"
