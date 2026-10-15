#!/bin/bash

# Train RPG on Amazon Books on one GPU of a shared server (no Slurm).
# Run from this folder:
#
#   cd jobs/new_datasets/rpg/books
#   GPU=0 bash ./train.sh
#
# Differences from the small-dataset jobs, both because of catalogue size:
#   --train-only      the built-in graph-decoded test would allocate a dense
#                     n x n matrix; evaluation is done by eval.sh instead
#   --val-max-users   early stopping uses a fixed user subset, because
#                     validation scores every item for every user
#
# Optional env vars: VAL_MAX_USERS (20000), EVAL_BATCH_SIZE (16), RUN_TAG.
# Extra arguments are forwarded as config overrides.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(pwd -P)" != "${SCRIPT_DIR}" ]]; then
  echo "ERROR: run this script from ${SCRIPT_DIR}" >&2
  exit 2
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
source "${REPO_ROOT}/jobs/lib/local_gpu.sh"
source "${REPO_ROOT}/jobs/lib/runtime_stats.sh"

: "${GPU:?Set GPU to the index of an idle GPU.}"

local_gpu_require_item_cache "${REPO_ROOT}" "Books"
PYTHON_BIN="$(local_gpu_python "${REPO_ROOT}")"

RUN_ID="rpg_repro_books${RUN_TAG:+_${RUN_TAG}}"
OUTPUT_DIR="${REPO_ROOT}/output/new_datasets/rpg/books"
mkdir -p "${OUTPUT_DIR}"

local_gpu_require_idle "${GPU}"
export CUDA_VISIBLE_DEVICES="${GPU}"

LOG_FILE="${OUTPUT_DIR}/${RUN_ID}-local-$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

runtime_stats_init "${OUTPUT_DIR}" "${RUN_ID}"
runtime_stats_start_gpu_monitor
trap runtime_stats_finish EXIT

cd "${REPO_ROOT}"
runtime_stats_run "${PYTHON_BIN}" scripts/rpg.py \
  --preset books \
  --train-only \
  --val-max-users "${VAL_MAX_USERS:-20000}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-16}" \
  --run_id "${RUN_ID}" \
  "$@"
