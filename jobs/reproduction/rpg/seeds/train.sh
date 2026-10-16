#!/bin/bash

# Submit from jobs/reproduction/rpg/seeds so these relative output paths resolve correctly.
#SBATCH --job-name=rpg_seeds
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=16:00:00
#SBATCH --output=../../../../output/reproduction/rpg/seeds/%x-%j.out
#SBATCH --error=../../../../output/reproduction/rpg/seeds/%x-%j.err

# Train and test RPG with the released settings of one dataset for extra
# training seeds (the released seed is 2024). Usage:
#   cd jobs/reproduction/rpg/seeds
#   sbatch --job-name=rpg_seeds_cds_s2025 ./train.sh cds_and_vinyl "2025"
# Arguments: PRESET (configs/rpg/repro/<preset>.yaml), SEEDS.
# Test metrics: "Test Results" at the end of artifacts/rpg/logs/AmazonReviews2014/RPG/rpg_repro_<preset>_s<seed>-*.log

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

PRESET="${1:?PRESET}"; SEEDS="${2:-2025 2026}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output/reproduction/rpg/seeds"
source "${REPO_ROOT}/jobs/lib/env.sh"
repo_env_activate "${REPO_ROOT}"

mkdir -p "${OUTPUT_DIR}"
source "${REPO_ROOT}/jobs/lib/runtime_stats.sh"
runtime_stats_init "${OUTPUT_DIR}" "rpg_${PRESET}_seeds"
runtime_stats_start_gpu_monitor
trap runtime_stats_finish EXIT

cd "${REPO_ROOT}"
for seed in ${SEEDS}; do
  runtime_stats_run python scripts/rpg/train.py --preset "${PRESET}" --rand_seed "${seed}" --run_id "rpg_repro_${PRESET}_s${seed}"
done
echo "RPG_SEEDS_DONE"
