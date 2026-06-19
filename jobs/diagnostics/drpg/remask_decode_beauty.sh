#!/bin/bash

# Submit from jobs/diagnostics/drpg.
#SBATCH --partition=gpu_h100
#SBATCH --job-name=drpg_remask_decode_beauty
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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
    echo "  sbatch ./remask_decode_beauty.sh" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ENV_PREFIX="${REPO_ROOT}/artifacts/conda/rpg-uva"
OUTPUT_JSON="${OUTPUT_JSON:-${REPO_ROOT}/artifacts/rpg/diagnostics/drpg_beauty_remask_decode_eval.json}"
MAX_EXAMPLES="${MAX_EXAMPLES:-1024}"
BATCH_SIZE="${BATCH_SIZE:-64}"
REVEAL_COUNTS="${REVEAL_COUNTS:-8,4,2}"
RANDOM_ORACLE_SEED="${RANDOM_ORACLE_SEED:-17}"
RECENT_HISTORY="${RECENT_HISTORY:-3}"
GRAPH_CANDIDATES_PER_SEED="${GRAPH_CANDIDATES_PER_SEED:-200}"
INCLUDE_GRAPH="${INCLUDE_GRAPH:-1}"

mkdir -p "${REPO_ROOT}/output/diagnostics/drpg"
mkdir -p "$(dirname "${OUTPUT_JSON}")"

module purge
module load 2025
module load Anaconda3/2025.06-1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

cd "${REPO_ROOT}"

echo "Running DRPG remask/prefix decode eval"
echo "REPO_ROOT=${REPO_ROOT}"
echo "OUTPUT_JSON=${OUTPUT_JSON}"
echo "MAX_EXAMPLES=${MAX_EXAMPLES}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "REVEAL_COUNTS=${REVEAL_COUNTS}"
echo "RANDOM_ORACLE_SEED=${RANDOM_ORACLE_SEED}"
echo "RECENT_HISTORY=${RECENT_HISTORY}"
echo "GRAPH_CANDIDATES_PER_SEED=${GRAPH_CANDIDATES_PER_SEED}"
echo "INCLUDE_GRAPH=${INCLUDE_GRAPH}"

ARGS=(
  --output "${OUTPUT_JSON}"
  --max-examples "${MAX_EXAMPLES}"
  --batch-size "${BATCH_SIZE}"
  --reveal-counts "${REVEAL_COUNTS}"
  --random-oracle-seed "${RANDOM_ORACLE_SEED}"
  --recent-history "${RECENT_HISTORY}"
  --graph-candidates-per-seed "${GRAPH_CANDIDATES_PER_SEED}"
  --device cuda
)

if [[ "${INCLUDE_GRAPH}" == "1" ]]; then
  ARGS+=(--include-graph)
fi

python scripts/drpg_remask_decode_eval.py "${ARGS[@]}"
