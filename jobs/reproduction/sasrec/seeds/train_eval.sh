#!/bin/bash

# Submit from jobs/reproduction/sasrec/seeds so these relative output paths resolve correctly.
#SBATCH --job-name=sasrec_seeds
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=../../../../output/reproduction/sasrec/seeds/%x-%j.out
#SBATCH --error=../../../../output/reproduction/sasrec/seeds/%x-%j.err

# Train SASRec (S3-Rec recipe, BCE loss) for several seeds and test every
# checkpoint under both evaluation protocols: past items removed from the
# ranking (S3-Rec code, as during training) and past items kept (as RPG,
# TIGER and full-softmax SASRec). Seeds whose checkpoint already exists are
# only evaluated. Usage:
#   cd jobs/reproduction/sasrec/seeds
#   sbatch --job-name=sasrec_seeds_beauty_h64 ./train_eval.sh beauty Beauty 64 2 "2024 2025 2026"
#   sbatch --job-name=sasrec_seeds_beauty_pm ./train_eval.sh beauty Beauty 540 3 "2024 2025 2026"
# Parameter-matched cells: sports_and_outdoors 326/2, beauty 540/3, toys_and_games 396/2, cds_and_vinyl 328/2.
# Arguments: PRESET, DATASET, HIDDEN_SIZE, NUM_HIDDEN_LAYERS, SEEDS.
# Env vars: RUN_PREFIX (default sasrec_<preset>_h<hidden>_L<layers>_e300_lr0p0003),
#           CKPT_DIR (default artifacts/sasrec/ckpt/seeds), LR (default 0.0003),
#           EXTRA_ARGS (more config overrides for training, e.g. "--val_max_users 20000" on Books).
# Test metrics: last line of artifacts/sasrec/logs/<run_id>_{removed,kept}.log

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ "$(pwd -P)" != "${SCRIPT_DIR}" ]]; then
    echo "ERROR: run this script from ${SCRIPT_DIR}" >&2
    exit 2
  fi
fi

PRESET="${1:?PRESET}"; DATASET="${2:?DATASET}"; HIDDEN_SIZE="${3:?HIDDEN_SIZE}"
NUM_HIDDEN_LAYERS="${4:?NUM_HIDDEN_LAYERS}"; SEEDS="${5:-2024 2025 2026}"
LR="${LR:-0.0003}"
RUN_PREFIX="${RUN_PREFIX:-sasrec_${PRESET}_h${HIDDEN_SIZE}_L${NUM_HIDDEN_LAYERS}_e300_lr${LR//./p}}"
CKPT_DIR="${CKPT_DIR:-artifacts/sasrec/ckpt/seeds}"

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output/reproduction/sasrec/seeds"
source "${REPO_ROOT}/jobs/lib/env.sh"
repo_env_activate "${REPO_ROOT}"

mkdir -p "${OUTPUT_DIR}"
source "${REPO_ROOT}/jobs/lib/runtime_stats.sh"
runtime_stats_init "${OUTPUT_DIR}" "${RUN_PREFIX}"
runtime_stats_start_gpu_monitor
trap runtime_stats_finish EXIT

cd "${REPO_ROOT}"
common=(--preset "${PRESET}" --dataset "${DATASET}" --hidden_size "${HIDDEN_SIZE}" --num_hidden_layers "${NUM_HIDDEN_LAYERS}")
for seed in ${SEEDS}; do
  RUN_ID="${RUN_PREFIX}_s${seed}"
  CHECKPOINT="${CKPT_DIR}/${RUN_ID}.pt"
  if [[ ! -f "${CHECKPOINT}" ]]; then
    runtime_stats_run python scripts/sasrec/train.py "${common[@]}" --epochs 300 --lr "${LR}" --seed "${seed}" \
      --ckpt_dir "${CKPT_DIR}" --run_id "${RUN_ID}" ${EXTRA_ARGS:-}
  fi
  for protocol in removed kept; do
    mask=true; [[ "${protocol}" == kept ]] && mask=false
    runtime_stats_run python scripts/sasrec/train.py "${common[@]}" --eval-only --checkpoint "${CHECKPOINT}" \
      --mask_history "${mask}" --run_id "${RUN_ID}_${protocol}"
  done
done
echo "SASREC_SEEDS_DONE"
