#!/bin/bash

# Submit from jobs/baselines/memgen so these relative output paths resolve correctly.
#SBATCH --job-name=memgen_eval_removed
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=../../../output/baselines/memgen/%x-%j.out
#SBATCH --error=../../../output/baselines/memgen/%x-%j.err

# Test the trained full-softmax SASRec checkpoints again with the items of the
# input sequence removed from the ranking (the S3-Rec protocol), for the
# comparison of evaluation protocols. No training.
#   cd jobs/baselines/memgen && sbatch ./eval_removed.sh
# Env vars: PRESETS (default: sports beauty toys cds), SEEDS (2024 2025 2026).
# Test metrics: "Test Results" in
# artifacts/memgen/logs/AmazonReviews2014/SASRec/memgen_sasrec_<preset>_seed<seed>_removed-*.log

set -uo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ "$(pwd -P)" != "${SCRIPT_DIR}" ]]; then
    echo "ERROR: run this script from ${SCRIPT_DIR}" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output/baselines/memgen"
PRESETS="${PRESETS:-sports beauty toys cds}"
SEEDS="${SEEDS:-2024 2025 2026}"
source "${REPO_ROOT}/jobs/lib/env.sh"
repo_env_activate "${REPO_ROOT}"
export HF_HUB_OFFLINE=1

mkdir -p "${OUTPUT_DIR}"
source "${REPO_ROOT}/jobs/lib/runtime_stats.sh"
runtime_stats_init "${OUTPUT_DIR}" "memgen_eval_removed"
trap runtime_stats_finish EXIT

cd "${REPO_ROOT}"
for preset in ${PRESETS}; do
  for seed in ${SEEDS}; do
    ckpt="$(ls -t artifacts/memgen/ckpt/memgen_sasrec_${preset}_seed${seed}-*.pth 2>/dev/null | head -n 1)"
    if [[ -z "${ckpt}" ]]; then runtime_stats_log "SKIP ${preset} seed ${seed}: no checkpoint"; continue; fi
    runtime_stats_run python scripts/memgen/train.py --model SASRec --preset "${preset}" --seed "${seed}" \
      --eval-only --checkpoint "${ckpt}" --mask-history --run-tag removed
  done
done
echo "MEMGEN_EVAL_REMOVED_DONE"
