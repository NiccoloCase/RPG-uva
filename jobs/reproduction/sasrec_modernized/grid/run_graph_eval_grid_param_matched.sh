#!/bin/bash

# SASRecModernized parameter-matched graph-decoding hyperparameter grid on the original Sports pool.
# Submit from this directory:
#   cd jobs/reproduction/sasrec_modernized/grid
#   mkdir -p ../../../../output/reproduction/sasrec_modernized/grid/graph_eval_param_matched
#   sbatch ./run_graph_eval_grid_param_matched.sh

#SBATCH --job-name=sasrec_mod_graph_grid_pm
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=../../../../output/reproduction/sasrec_modernized/grid/graph_eval_param_matched/%x-%j.out
#SBATCH --error=../../../../output/reproduction/sasrec_modernized/grid/graph_eval_param_matched/%x-%j.err

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
    echo "  bash ./run_graph_eval_grid_param_matched.sh" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"
OUTPUT_DIR="${REPO_ROOT}/output/reproduction/sasrec_modernized/grid/graph_eval_param_matched"
ENV_PREFIX="${REPO_ROOT}/artifacts/conda/rpg-uva"
GRID_CONFIG="${REPO_ROOT}/configs/sasrec/perf/sports_graph.yaml"
PARAM_MATCHED_CONFIG="${REPO_ROOT}/configs/sasrec/param_matched/sports_and_outdoors.yaml"
GRID_OUTPUT_DIR="${REPO_ROOT}/artifacts/sasrec_modernized/grid/graph_eval/param_matched/sports_and_outdoors"

if [[ ! -f "${PARAM_MATCHED_CONFIG}" ]]; then
  echo "ERROR: parameter-matched config not found: ${PARAM_MATCHED_CONFIG}" >&2
  exit 3
fi

mkdir -p "${OUTPUT_DIR}"
source "${REPO_ROOT}/jobs/lib/runtime_stats.sh"
runtime_stats_init "${OUTPUT_DIR}" "sasrec_modernized_graph_grid_param_matched_sports_and_outdoors"
runtime_stats_start_gpu_monitor
trap runtime_stats_finish EXIT

module purge
module load 2025
module load Anaconda3/2025.06-1

BEST_INFO="$("${ENV_PREFIX}/bin/python" "${REPO_ROOT}/scripts/find_best_sasrec_ablation_checkpoint.py" --dataset sports_and_outdoors --format shell)"
eval "${BEST_INFO}"

if [[ -z "${CHECKPOINT_PATH:-}" || ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: best parameter-matched checkpoint not found for sports_and_outdoors" >&2
  exit 4
fi

cd "${REPO_ROOT}"

runtime_stats_run conda run -p "${ENV_PREFIX}" python scripts/sasrec_perf.py \
  grid-eval \
  --checkpoint "${CHECKPOINT_PATH}" \
  --preset sports_and_outdoors \
  --dataset Sports_and_Outdoors \
  --config "${GRID_CONFIG}" \
  --config "${PARAM_MATCHED_CONFIG}" \
  --output-dir "${GRID_OUTPUT_DIR}" \
  "$@"
