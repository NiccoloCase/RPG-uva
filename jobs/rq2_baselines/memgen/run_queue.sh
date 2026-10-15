#!/bin/bash

# MemGen-GR baselines (SASRec with softmax loss, TIGER): train and test a queue
# of (model, dataset) pairs on one GPU of a shared server (no Slurm). Run from
# this folder:
#
#   cd jobs/rq2_baselines/memgen
#   GPU=5 QUEUE="SASRec:sports SASRec:books" bash ./run_queue.sh
#
# QUEUE items are "model:preset[:key=value,key=value]":
#   model    SASRec or TIGER
#   preset   sports, beauty, toys, cds, books (configs/memgen/<model>/)
# Optional env vars:
#   SEED              random seed for every item (default 2024)
#   MEMGEN_ROOT       MemGen-GR checkout (default third_party/MemGen-GR)
#   GPU_MAX_USED_MIB  idle threshold for the GPU check (500)
# Test metrics are the "Test Results" line at the end of
# artifacts/memgen/logs/AmazonReviews2014/<model>/<run_id>-*.log.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(pwd -P)" != "${SCRIPT_DIR}" ]]; then
  echo "ERROR: run this script from ${SCRIPT_DIR}" >&2
  exit 2
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${REPO_ROOT}/jobs/lib/local_gpu.sh"

: "${GPU:?Set GPU to a GPU index.}"
: "${QUEUE:?Set QUEUE to space-separated model:preset items.}"
SEED="${SEED:-2024}"
PYTHON_BIN="$(local_gpu_python "${REPO_ROOT}")"

OUTPUT_DIR="${REPO_ROOT}/output/memgen"
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/queue-gpu${GPU}-$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

local_gpu_require_idle "${GPU}" || exit 1
log "queue on GPU ${GPU}: ${QUEUE} (seed ${SEED})"

for item in ${QUEUE}; do
  IFS=: read -r model preset extra <<< "${item}"
  overrides=()
  if [[ -n "${extra:-}" ]]; then
    IFS=, read -ra pairs <<< "${extra}"
    for pair in "${pairs[@]}"; do overrides+=("--${pair}"); done
  fi
  run_id="memgen_$(tr '[:upper:]' '[:lower:]' <<< "${model}")_${preset}_seed${SEED}"
  log "train ${model} ${preset} run_id=${run_id} ${overrides[*]:-}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${REPO_ROOT}/scripts/memgen_train.py" \
    --model "${model}" --preset "${preset}" --seed "${SEED}" "${overrides[@]}" \
    > "${OUTPUT_DIR}/${run_id}.out" 2>&1
  status=$?
  if (( status == 0 )); then
    log "done ${run_id}: $(grep -o 'Test Results: .*' "${OUTPUT_DIR}/${run_id}.out" | tail -1)"
  else
    log "FAILED ${run_id} (exit ${status}); see ${OUTPUT_DIR}/${run_id}.out"
  fi
done
log "QUEUE_DONE"
