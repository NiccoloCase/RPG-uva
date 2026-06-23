#!/bin/bash

# Submit from jobs/reproduction/sasrec_modernized/perf so these relative output paths resolve correctly.
# Prefer gpu_a100 for this full GPU profiling run per repo guidance.
#SBATCH --job-name=sasrec_mod_graph_prof
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=../../../../output/reproduction/sasrec_modernized/perf/graph_profile/%x-%j.out
#SBATCH --error=../../../../output/reproduction/sasrec_modernized/perf/graph_profile/%x-%j.err

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
    echo "  bash ./profile_graph_inference.sh" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"
OUTPUT_DIR="${REPO_ROOT}/output/reproduction/sasrec_modernized/perf/graph_profile"
ENV_PREFIX="${REPO_ROOT}/artifacts/conda/rpg-uva"
PERF_CONFIG_DEFAULT="${REPO_ROOT}/configs/sasrec/perf/sports_graph.yaml"
PERF_CONFIG="${PERF_CONFIG:-${PERF_CONFIG_DEFAULT}}"
PERF_DATASET_SLUG="${PERF_DATASET_SLUG:-sports_and_outdoors}"
LOCAL_CHECKPOINT_DIR="${REPO_ROOT}/artifacts/sasrec_modernized/ckpt"
SHARED_CHECKPOINT_DIR="/projects/prjs2120/groups/group_16/artifacts/sasrec_modernized/ckpt"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
CHECKPOINT_PATH="${1:-${CHECKPOINT_PATH:-}}"
shift_count=0
if [[ $# -ge 1 ]]; then
  shift_count=1
fi

DEFAULT_CHECKPOINT_BASENAME="sasrec_modernized_${PERF_DATASET_SLUG}.pt"
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
  echo "ERROR: SASRecModernized checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 4
fi

if [[ ! -f "${PERF_CONFIG}" ]]; then
  echo "ERROR: SASRecModernized graph perf config not found: ${PERF_CONFIG}" >&2
  exit 5
fi

if [[ ${shift_count} -eq 1 ]]; then
  shift
fi

mkdir -p "${OUTPUT_DIR}"
source "${REPO_ROOT}/jobs/lib/runtime_stats.sh"
runtime_stats_init "${OUTPUT_DIR}" "sasrec_modernized_graph_profile_${PERF_DATASET_SLUG}"
runtime_stats_start_gpu_monitor
trap runtime_stats_finish EXIT

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "${REPO_ROOT}"

runtime_stats_run conda run -p "${ENV_PREFIX}" python scripts/sasrec_perf.py \
  profile \
  --checkpoint "${CHECKPOINT_PATH}" \
  --preset sports_and_outdoors \
  --dataset Sports_and_Outdoors \
  --config "${PERF_CONFIG}" \
  "$@"
