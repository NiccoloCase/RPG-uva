#!/bin/bash

# Submit from jobs/reproduction/rpg/beauty so these relative output paths resolve correctly.
#SBATCH --partition=gpu_h100
#SBATCH --job-name=rpg_beauty
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --output=../../../../output/reproduction/rpg/beauty/%x-%j.out
#SBATCH --error=../../../../output/reproduction/rpg/beauty/%x-%j.err


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
OUTPUT_DIR="${REPO_ROOT}/output/reproduction/rpg/beauty"
DATA_DIR="${REPO_ROOT}/artifacts/rpg/cache/AmazonReviews2014/Beauty"
PROCESSED_DIR="${DATA_DIR}/processed"
SENT_EMB_PATH="${PROCESSED_DIR}/text-embedding-3-large.sent_emb"
ENV_PREFIX="${REPO_ROOT}/artifacts/conda/rpg-uva"
RESET_SEM_IDS="${RESET_SEM_IDS:-1}"

mkdir -p "${OUTPUT_DIR}"
source "${REPO_ROOT}/jobs/lib/runtime_stats.sh"
runtime_stats_init "${OUTPUT_DIR}" "rpg_beauty"
runtime_stats_start_gpu_monitor
trap runtime_stats_finish EXIT

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "ERROR: missing cached dataset directory: ${DATA_DIR}" >&2
  exit 3
fi

if [[ ! -f "${SENT_EMB_PATH}" ]]; then
  echo "ERROR: missing cached Beauty sentence embeddings: ${SENT_EMB_PATH}" >&2
  echo "This job is meant to reuse embeddings and regenerate semantic IDs locally." >&2
  exit 4
fi

if [[ "${RESET_SEM_IDS}" == "1" ]]; then
  while IFS= read -r sem_ids_path; do
    runtime_stats_log "removing_stale_semantic_ids=${sem_ids_path}"
    rm -f "${sem_ids_path}"
  done < <(find "${PROCESSED_DIR}" -maxdepth 1 -type f -name '*.sem_ids' | sort)
else
  runtime_stats_log "reset_semantic_ids=disabled"
fi

module purge
module load 2025
module load Anaconda3/2025.06-1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

cd "${REPO_ROOT}"

echo "Training RPG Beauty from scratch"
echo "Using cache: ${DATA_DIR}"
echo "Reusing sentence embeddings: ${SENT_EMB_PATH}"
echo "RESET_SEM_IDS=${RESET_SEM_IDS}"

runtime_stats_run python3 scripts/rpg.py --preset beauty
