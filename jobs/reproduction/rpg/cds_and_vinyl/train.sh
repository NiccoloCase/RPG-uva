#!/bin/bash

# Submit from jobs/reproduction/rpg/cds_and_vinyl so these relative output paths resolve correctly.
#SBATCH --partition=gpu_a100
#SBATCH --job-name=rpg_cds_and_vinyl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=48:00:00
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
    echo "  bash ./train.sh" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output/reproduction/rpg/cds_and_vinyl"
DATA_DIR="${REPO_ROOT}/artifacts/rpg/cache/AmazonReviews2014/CDs_and_Vinyl"
ENV_PREFIX="${REPO_ROOT}/artifacts/conda/rpg-uva"

mkdir -p "${OUTPUT_DIR}"
source "${REPO_ROOT}/jobs/lib/runtime_stats.sh"
runtime_stats_init "${OUTPUT_DIR}" "rpg_cds_and_vinyl"
runtime_stats_start_gpu_monitor
trap runtime_stats_finish EXIT

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "ERROR: missing cached dataset directory: ${DATA_DIR}" >&2
  exit 3
fi

module purge
module load 2025
module load Anaconda3/2025.06-1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

cd "${REPO_ROOT}"

runtime_stats_run python3 scripts/rpg.py --preset cds_and_vinyl
