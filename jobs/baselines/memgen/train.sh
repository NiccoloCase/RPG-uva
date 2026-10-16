#!/bin/bash

# Submit from jobs/baselines/memgen so these relative output paths resolve correctly.
#SBATCH --job-name=memgen_train
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=../../../output/baselines/memgen/%x-%j.out
#SBATCH --error=../../../output/baselines/memgen/%x-%j.err

# Train and test one MemGen-GR baseline (SASRec with softmax loss, or TIGER) on
# one dataset for several seeds, one after the other, so runs of the same
# dataset never write the same cache files at once. Run jobs/data/prepare_data.sh
# first. Usage:
#   cd jobs/baselines/memgen
#   sbatch --job-name=memgen_sasrec_sports ./train.sh SASRec sports "2024 2025 2026"
#   sbatch --job-name=memgen_sasrec_books  ./train.sh SASRec books "2024" --eval_interval=1 --patience=5
# Arguments: MODEL (SASRec | TIGER), PRESET (sports | beauty | toys | cds | books),
# SEEDS (space-separated), then optional --key=value MemGen-GR overrides.
# Test metrics: "Test Results" at the end of
# artifacts/memgen/logs/AmazonReviews2014/<MODEL>/memgen_<model>_<preset>_seed<seed>-*.log

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

MODEL="${1:?MODEL (SASRec or TIGER)}"
PRESET="${2:?PRESET (sports, beauty, toys, cds, books)}"
SEEDS="${3:-2024 2025 2026}"
shift 3 || shift $#

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output/baselines/memgen"
source "${REPO_ROOT}/jobs/lib/env.sh"
repo_env_activate "${REPO_ROOT}"
export HF_HUB_OFFLINE=1   # sentence-T5 is cached by prepare_data.sh

mkdir -p "${OUTPUT_DIR}"
source "${REPO_ROOT}/jobs/lib/runtime_stats.sh"
runtime_stats_init "${OUTPUT_DIR}" "memgen_${MODEL,,}_${PRESET}"
runtime_stats_start_gpu_monitor
trap runtime_stats_finish EXIT

cd "${REPO_ROOT}"
for seed in ${SEEDS}; do
  runtime_stats_log "train ${MODEL} ${PRESET} seed ${seed} $*"
  runtime_stats_run python scripts/memgen/train.py --model "${MODEL}" --preset "${PRESET}" --seed "${seed}" "$@"
done
echo "MEMGEN_DONE"
