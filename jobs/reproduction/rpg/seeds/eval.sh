#!/bin/bash

# Submit from jobs/reproduction/rpg/seeds so these relative output paths resolve correctly.
#SBATCH --job-name=rpg_seeds_eval
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=../../../../output/reproduction/rpg/seeds/%x-%j.out
#SBATCH --error=../../../../output/reproduction/rpg/seeds/%x-%j.err

# Evaluate the RPG checkpoints of every dataset and training seed the same way:
# graph-constrained decoding with the released settings (exact graph built in
# row blocks), one fixed evaluation seed, all test users.
#   cd jobs/reproduction/rpg/seeds && sbatch ./eval.sh
# Env vars: DATASETS (default: the four paper datasets), SEEDS (2024 2025 2026),
#           EVAL_SEED (2024).
# Checkpoints: seed 2024 is rpg_loss_token_ce_<dataset>-*, other seeds
# rpg_loss_token_ce_<dataset>_s<seed>-* or rpg_repro_<dataset>_s<seed>-* (newest match).
# Results: artifacts/rpg/eval_seeds/seeds/<dataset>/s<seed>/<session>/summary.json

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

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output/reproduction/rpg/seeds"
DATASETS="${DATASETS:-sports_and_outdoors beauty toys_and_games cds_and_vinyl}"
SEEDS="${SEEDS:-2024 2025 2026}"
EVAL_SEED="${EVAL_SEED:-2024}"
source "${REPO_ROOT}/jobs/lib/env.sh"
repo_env_activate "${REPO_ROOT}"

mkdir -p "${OUTPUT_DIR}"
source "${REPO_ROOT}/jobs/lib/runtime_stats.sh"
runtime_stats_init "${OUTPUT_DIR}" "rpg_seeds_eval"
runtime_stats_start_gpu_monitor
trap runtime_stats_finish EXIT

cd "${REPO_ROOT}"
newest() { ls -t $@ 2>/dev/null | head -n 1; }
for dataset in ${DATASETS}; do
  for seed in ${SEEDS}; do
    if [[ "${seed}" == 2024 ]]; then
      checkpoint="$(newest "artifacts/rpg/ckpt/rpg_loss_token_ce_${dataset}-*.pth")"
    else
      checkpoint="$(newest "artifacts/rpg/ckpt/rpg_loss_token_ce_${dataset}_s${seed}-*.pth" "artifacts/rpg/ckpt/rpg_repro_${dataset}_s${seed}-*.pth")"
    fi
    if [[ -z "${checkpoint}" ]]; then runtime_stats_log "SKIP ${dataset} seed ${seed}: no checkpoint"; continue; fi
    runtime_stats_log "${dataset} seed ${seed}: $(basename "${checkpoint}" | cut -c1-80)"
    runtime_stats_run python scripts/rpg/eval_seeds.py --checkpoint "${checkpoint}" \
      --config "configs/rpg/repro/${dataset}.yaml" --no-local-config \
      --decoding graph --graph-backend exact_gpu --eval-seeds "${EVAL_SEED}" \
      --output-dir "artifacts/rpg/eval_seeds/seeds/${dataset}/s${seed}" --no-per-user-output
  done
done
echo "RPG_SEEDS_EVAL_DONE"
