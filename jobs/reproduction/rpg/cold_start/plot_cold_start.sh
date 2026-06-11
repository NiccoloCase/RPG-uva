#!/bin/bash

# Submit from jobs/reproduction/rpg/cold_start so these relative output paths resolve correctly.
#SBATCH --job-name=rpg_cold_start_plot
#SBATCH --partition=genoa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=../../../../output/reproduction/rpg/cold_start/%x-%j.out
#SBATCH --error=../../../../output/reproduction/rpg/cold_start/%x-%j.err

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
    echo "  bash ./plot_cold_start.sh $(cd "${SCRIPT_DIR}/../../../.." && pwd)/artifacts/rpg/cold_start/sports_debug_run/tables/cold_start_summary.json" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output/reproduction/rpg/cold_start"
ENV_PREFIX="${REPO_ROOT}/artifacts/conda/rpg-uva"
SUMMARY_PATH="${1:-${SUMMARY_PATH:-}}"
PLOT_OUTPUT_PATH="${2:-${PLOT_OUTPUT_PATH:-}}"
shift_count=0
if [[ $# -ge 2 ]]; then
  shift_count=2
fi

if [[ -z "${SUMMARY_PATH}" ]]; then
  echo "ERROR: provide the cold-start summary JSON as the first argument or SUMMARY_PATH env var." >&2
  exit 3
fi

if [[ "${SUMMARY_PATH}" == *"<"* || "${SUMMARY_PATH}" == *">"* ]]; then
  echo "ERROR: summary path contains angle-bracket placeholders: ${SUMMARY_PATH}" >&2
  exit 4
fi

if [[ ! -f "${SUMMARY_PATH}" ]]; then
  echo "ERROR: cold-start summary JSON not found: ${SUMMARY_PATH}" >&2
  exit 5
fi

if [[ -z "${PLOT_OUTPUT_PATH}" ]]; then
  PLOT_OUTPUT_PATH="$(dirname "${SUMMARY_PATH}")/../figures/ndcg_at_10.png"
fi

if [[ ${shift_count} -eq 2 ]]; then
  shift 2
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "$(dirname "${PLOT_OUTPUT_PATH}")"

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "${REPO_ROOT}"

conda run -p "${ENV_PREFIX}" python scripts/rpg_cold_start.py \
  plot \
  --input "${SUMMARY_PATH}" \
  --output "${PLOT_OUTPUT_PATH}" \
  "$@"
