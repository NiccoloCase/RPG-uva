#!/bin/bash

# Submit from jobs/diagnostics/drpg.
#SBATCH --partition=gpu_h100
#SBATCH --job-name=drpg_pregraph_beauty
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --output=../../../output/diagnostics/drpg/%x-%j.out
#SBATCH --error=../../../output/diagnostics/drpg/%x-%j.err

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
    echo "  sbatch ./pregraph_beauty.sh" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ENV_PREFIX="${REPO_ROOT}/artifacts/conda/rpg-uva"
OUTPUT_JSON="${OUTPUT_JSON:-${REPO_ROOT}/artifacts/rpg/diagnostics/drpg_beauty_pregraph_diagnostics.json}"
MAX_EXAMPLES="${MAX_EXAMPLES:-1024}"
BATCH_SIZE="${BATCH_SIZE:-64}"

mkdir -p "${REPO_ROOT}/output/diagnostics/drpg"
mkdir -p "$(dirname "${OUTPUT_JSON}")"

module purge
module load 2025
module load Anaconda3/2025.06-1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

cd "${REPO_ROOT}"

echo "Running DRPG pre-graph diagnostics"
echo "REPO_ROOT=${REPO_ROOT}"
echo "OUTPUT_JSON=${OUTPUT_JSON}"
echo "MAX_EXAMPLES=${MAX_EXAMPLES}"
echo "BATCH_SIZE=${BATCH_SIZE}"

python scripts/drpg_pregraph_diagnostics.py \
  --output "${OUTPUT_JSON}" \
  --max-examples "${MAX_EXAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --device cuda
