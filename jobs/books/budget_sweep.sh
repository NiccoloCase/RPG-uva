#!/bin/bash

# Decoding-budget sweep on a Books RPG checkpoint, on one GPU of a shared server
# (no Slurm). For each (beam size, edges, steps) setting it builds the exact
# decoding graph on the GPU and runs graph decoding on the test split with one
# eval seed. Run from this folder:
#
#   cd jobs/books
#   GPU=7 RUN_TAG=m32 bash ./budget_sweep.sh --n_codebook 32
#
# Optional env vars:
#   BUDGETS          space-separated "beams:edges:steps" settings
#   CHECKPOINT_PATH  default: newest artifacts/rpg/ckpt/rpg_repro_books[_TAG]-*.pth
#   EVAL_SEED        default 2024
#   RUN_TAG          must match the tag used for training
# Extra arguments are forwarded as config overrides.
#
# Results: artifacts/rpg/eval_seeds/new_datasets/books[_TAG]/budget/b<b>_k<k>_q<q>/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(pwd -P)" != "${SCRIPT_DIR}" ]]; then
  echo "ERROR: run this script from ${SCRIPT_DIR}" >&2
  exit 2
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/jobs/lib/local_gpu.sh"

: "${GPU:?Set GPU to the index of a GPU.}"
PYTHON_BIN="$(local_gpu_python "${REPO_ROOT}")"

RUN_ID="rpg_repro_books${RUN_TAG:+_${RUN_TAG}}"
CHECKPOINT_DIR="${REPO_ROOT}/artifacts/rpg/ckpt"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$(ls -t "${CHECKPOINT_DIR}/${RUN_ID}-"*.pth 2>/dev/null | head -n 1 || true)}"
if [[ -z "${CHECKPOINT_PATH}" || ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: no checkpoint found for run ${RUN_ID} in ${CHECKPOINT_DIR}" >&2
  exit 3
fi

BUDGETS="${BUDGETS:-50:500:5 100:500:5 200:500:5 100:1000:5 200:1000:5 200:2000:5}"
EVAL_SEED="${EVAL_SEED:-2024}"
EVAL_CONFIG="${REPO_ROOT}/configs/rpg/books.yaml"
RESULT_ROOT="${REPO_ROOT}/artifacts/rpg/eval_seeds/new_datasets/books${RUN_TAG:+_${RUN_TAG}}/budget"
OUTPUT_DIR="${REPO_ROOT}/output/new_datasets/rpg/books"
mkdir -p "${RESULT_ROOT}" "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU}"
LOG_FILE="${OUTPUT_DIR}/${RUN_ID}-budget-local-$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

cd "${REPO_ROOT}"
for budget in ${BUDGETS}; do
  IFS=: read -r beams edges steps <<< "${budget}"
  cell_dir="${RESULT_ROOT}/b${beams}_k${edges}_q${steps}"
  if compgen -G "${cell_dir}/*/summary.json" >/dev/null; then
    echo "BUDGET_SKIP b=${beams} k=${edges} q=${steps} (summary exists)"
    continue
  fi
  echo "BUDGET_START b=${beams} k=${edges} q=${steps} $(date -Is)"
  "${PYTHON_BIN}" scripts/rpg/eval_seeds.py \
    --checkpoint "${CHECKPOINT_PATH}" \
    --config "${EVAL_CONFIG}" \
    --eval-seeds "${EVAL_SEED}" \
    --decoding graph \
    --graph-backend exact_gpu \
    --no-per-user-output \
    --output-dir "${cell_dir}" \
    --num_beams "${beams}" \
    --n_edges "${edges}" \
    --propagation_steps "${steps}" \
    "$@" || echo "BUDGET_FAILED b=${beams} k=${edges} q=${steps}"
  echo "BUDGET_END b=${beams} k=${edges} q=${steps} $(date -Is)"
done
echo "BUDGET_SWEEP_DONE"
