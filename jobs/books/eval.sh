#!/bin/bash

# Evaluate the Books RPG checkpoint on one GPU of a shared server (no Slurm).
# Run from this folder:
#
#   cd jobs/books
#   GPU=0 bash ./eval.sh
#
# Runs graph decoding (graph built in row blocks on the GPU, never the dense matrix) and
# exhaustive scoring on the test split.
#
# Optional env vars:
#   CHECKPOINT_PATH  default: newest artifacts/rpg/ckpt/rpg_repro_books[_TAG]-*.pth
#   GRAPH_BACKEND    exact_gpu (default) or hnsw
#   EVAL_SEEDS       default 2024,2025,2026
#   RUN_TAG          must match the tag used for training
# Extra arguments are forwarded as config overrides to both evaluations,
# e.g. --n_codebook 32 for the 32-digit run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(pwd -P)" != "${SCRIPT_DIR}" ]]; then
  echo "ERROR: run this script from ${SCRIPT_DIR}" >&2
  exit 2
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/jobs/lib/local_gpu.sh"

: "${GPU:?Set GPU to the index of an idle GPU.}"
PYTHON_BIN="$(local_gpu_python "${REPO_ROOT}")"

RUN_ID="rpg_repro_books${RUN_TAG:+_${RUN_TAG}}"
CHECKPOINT_DIR="${REPO_ROOT}/artifacts/rpg/ckpt"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$(ls -t "${CHECKPOINT_DIR}/${RUN_ID}-"*.pth 2>/dev/null | head -n 1 || true)}"
if [[ -z "${CHECKPOINT_PATH}" || ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: no checkpoint found for run ${RUN_ID} in ${CHECKPOINT_DIR}" >&2
  exit 3
fi

GRAPH_BACKEND="${GRAPH_BACKEND:-exact_gpu}"
EVAL_SEEDS="${EVAL_SEEDS:-2024,2025,2026}"
EVAL_CONFIG="${REPO_ROOT}/configs/rpg/books.yaml"
RESULT_DIR="${REPO_ROOT}/artifacts/rpg/eval_seeds/new_datasets/books${RUN_TAG:+_${RUN_TAG}}"
OUTPUT_DIR="${REPO_ROOT}/output/new_datasets/rpg/books"
mkdir -p "${RESULT_DIR}" "${OUTPUT_DIR}"

local_gpu_require_idle "${GPU}"
export CUDA_VISIBLE_DEVICES="${GPU}"

LOG_FILE="${OUTPUT_DIR}/${RUN_ID}-eval-local-$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

cd "${REPO_ROOT}"

"${PYTHON_BIN}" scripts/rpg_eval_seeds.py \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config "${EVAL_CONFIG}" \
  --eval-seeds "${EVAL_SEEDS}" \
  --decoding graph \
  --graph-backend "${GRAPH_BACKEND}" \
  --no-per-user-output \
  --output-dir "${RESULT_DIR}/graph" \
  "$@"

"${PYTHON_BIN}" scripts/rpg_eval_seeds.py \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config "${EVAL_CONFIG}" \
  --eval-seeds 2024 \
  --decoding exhaustive \
  --no-per-user-output \
  --output-dir "${RESULT_DIR}/exhaustive" \
  "$@"
