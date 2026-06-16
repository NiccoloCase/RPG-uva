#!/bin/bash

# Submit from jobs/reproduction/rpg/graph_analysis so relative output paths resolve correctly.
# Requires an existing graph-analysis session with graphs/graph_metadata.json.
# If SESSION_DIR is omitted, the latest prepared Sports graph-analysis session is used.
#SBATCH --job-name=rpg_graph_pruning
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=../../../../output/reproduction/rpg/graph_analysis/%x-%j.out
#SBATCH --error=../../../../output/reproduction/rpg/graph_analysis/%x-%j.err

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
    echo "  bash ./run_pruning_sports.sh /gpfs/home6/\$USER/RPG/artifacts/rpg/ckpt/model.pth" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output/reproduction/rpg/graph_analysis"
ENV_PREFIX="${REPO_ROOT}/artifacts/conda/rpg-uva"
GRAPH_CONFIG="${GRAPH_CONFIG:-${REPO_ROOT}/configs/rpg/graph_analysis/sports.yaml}"
PRUNING_CONFIG="${PRUNING_CONFIG:-${REPO_ROOT}/configs/rpg/graph_analysis/sports_pruning.yaml}"
CHECKPOINT_PATH="${1:-${CHECKPOINT_PATH:-}}"
SESSION_ARGS=()

if [[ -n "${SESSION_DIR:-}" ]]; then
  SESSION_ARGS=(--session-dir "${SESSION_DIR}")
fi

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: provide the Sports RPG checkpoint path as the first argument or CHECKPOINT_PATH env var." >&2
  exit 3
fi

if [[ "${CHECKPOINT_PATH}" == *"<"* || "${CHECKPOINT_PATH}" == *">"* ]]; then
  echo "ERROR: checkpoint path contains angle-bracket placeholders: ${CHECKPOINT_PATH}" >&2
  exit 4
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: checkpoint file not found: ${CHECKPOINT_PATH}" >&2
  exit 5
fi

if [[ ! -f "${GRAPH_CONFIG}" ]]; then
  echo "ERROR: graph analysis config not found: ${GRAPH_CONFIG}" >&2
  exit 6
fi

if [[ ! -f "${PRUNING_CONFIG}" ]]; then
  echo "ERROR: pruning graph analysis config not found: ${PRUNING_CONFIG}" >&2
  exit 7
fi

mkdir -p "${OUTPUT_DIR}"
source "${REPO_ROOT}/jobs/lib/runtime_stats.sh"
runtime_stats_init "${OUTPUT_DIR}" "rpg_graph_pruning"
runtime_stats_start_gpu_monitor
trap runtime_stats_finish EXIT

module purge
module load 2025
module load Anaconda3/2025.06-1

export MPLCONFIGDIR="${TMPDIR:-/tmp}/rpg-matplotlib-${USER:-user}"
mkdir -p "${MPLCONFIGDIR}"

cd "${REPO_ROOT}"

runtime_stats_run conda run -p "${ENV_PREFIX}" python scripts/rpg_graph_analysis.py \
  pruning \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config "${GRAPH_CONFIG}" \
  --config "${PRUNING_CONFIG}" \
  "${SESSION_ARGS[@]}"
