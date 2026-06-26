#!/bin/bash

# SASRec base graph-decoding hyperparameter grid on the original Sports pool.
# Submit from this directory:
#   cd jobs/reproduction/sasrec/grid
#   mkdir -p ../../../../output/reproduction/sasrec/grid/graph_eval_base
#   sbatch ./run_graph_eval_grid_base.sh

#SBATCH --job-name=sasrec_mod_graph_grid_base
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=../../../../output/reproduction/sasrec/grid/graph_eval_base/%x-%j.out
#SBATCH --error=../../../../output/reproduction/sasrec/grid/graph_eval_base/%x-%j.err

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
    echo "  bash ./run_graph_eval_grid_base.sh" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"
OUTPUT_DIR="${REPO_ROOT}/output/reproduction/sasrec/grid/graph_eval_base"
ENV_PREFIX="${REPO_ROOT}/artifacts/conda/rpg-uva"
GRID_CONFIG="${REPO_ROOT}/configs/sasrec/perf/sports_graph.yaml"
GRID_OUTPUT_DIR="${REPO_ROOT}/artifacts/sasrec/grid/graph_eval/base/sports_and_outdoors"
PERF_DATASET_SLUG="${PERF_DATASET_SLUG:-sports_and_outdoors}"
LOCAL_CHECKPOINT_DIR="${REPO_ROOT}/artifacts/sasrec/ckpt"
SHARED_CHECKPOINT_DIR="/projects/prjs2120/groups/group_16/artifacts/sasrec/ckpt"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
CHECKPOINT_PATH="${1:-${CHECKPOINT_PATH:-}}"
shift_count=0
if [[ $# -ge 1 ]]; then
  shift_count=1
fi

DEFAULT_CHECKPOINT_BASENAME="sasrec_${PERF_DATASET_SLUG}.pt"
if [[ -z "${CHECKPOINT_DIR}" ]]; then
  if [[ -f "${LOCAL_CHECKPOINT_DIR}/${DEFAULT_CHECKPOINT_BASENAME}" ]]; then
    CHECKPOINT_DIR="${LOCAL_CHECKPOINT_DIR}"
  elif [[ -f "${SHARED_CHECKPOINT_DIR}/${DEFAULT_CHECKPOINT_BASENAME}" ]]; then
    CHECKPOINT_DIR="${SHARED_CHECKPOINT_DIR}"
  else
    CHECKPOINT_DIR="${LOCAL_CHECKPOINT_DIR}"
  fi
fi

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="${CHECKPOINT_DIR}/${DEFAULT_CHECKPOINT_BASENAME}"
fi

if [[ "${CHECKPOINT_PATH}" == *"<"* || "${CHECKPOINT_PATH}" == *">"* ]]; then
  echo "ERROR: checkpoint path contains angle-bracket placeholders: ${CHECKPOINT_PATH}" >&2
  echo "Use a real path, for example:" >&2
  echo "  ${LOCAL_CHECKPOINT_DIR}/${DEFAULT_CHECKPOINT_BASENAME}" >&2
  echo "  ${SHARED_CHECKPOINT_DIR}/${DEFAULT_CHECKPOINT_BASENAME}" >&2
  exit 3
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: SASRec base checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 4
fi

mkdir -p "${OUTPUT_DIR}"
source "${REPO_ROOT}/jobs/lib/runtime_stats.sh"
runtime_stats_init "${OUTPUT_DIR}" "sasrec_graph_grid_base_${PERF_DATASET_SLUG}"
runtime_stats_start_gpu_monitor
trap runtime_stats_finish EXIT

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "${REPO_ROOT}"

runtime_stats_run conda run -p "${ENV_PREFIX}" python scripts/sasrec_perf.py \
  grid-eval \
  --checkpoint "${CHECKPOINT_PATH}" \
  --preset sports_and_outdoors \
  --dataset Sports_and_Outdoors \
  --config "${GRID_CONFIG}" \
  --output-dir "${GRID_OUTPUT_DIR}" \
  "$@"
