#!/bin/bash

# Submit from jobs/new_datasets/rpg/pet_supplies so these relative output paths resolve correctly.
#SBATCH --partition=gpu_a100
#SBATCH --job-name=rpg_pet_supplies_prep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --output=../../../../output/new_datasets/rpg/pet_supplies/%x-%j.out
#SBATCH --error=../../../../output/new_datasets/rpg/pet_supplies/%x-%j.err

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
    echo "  bash ./prepare_semantic_ids.sh" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output/new_datasets/rpg/pet_supplies"
ENV_PREFIX="${REPO_ROOT}/artifacts/conda/rpg-uva"

mkdir -p "${OUTPUT_DIR}"
source "${REPO_ROOT}/jobs/lib/runtime_stats.sh"
runtime_stats_init "${OUTPUT_DIR}" "rpg_pet_supplies_prep"
runtime_stats_start_gpu_monitor
trap runtime_stats_finish EXIT

module purge
module load 2025
module load Anaconda3/2025.06-1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

cd "${REPO_ROOT}"

runtime_stats_run python3 scripts/rpg_prepare_semantic_ids.py --preset pet_supplies
