#!/bin/bash

# SASRec loss/capacity audit: train and test a queue of SASRec settings on one
# GPU of a shared server (no Slurm). Run from this folder:
#
#   cd jobs/sasrec_audit
#   GPU=6 QUEUE="sports_and_outdoors:ce:64 beauty:bce:540" bash ./run_queue.sh
#
# QUEUE items are "dataset:loss:hidden_size[:key=value,key=value]":
#   dataset      SASRec preset (sports_and_outdoors, beauty, toys_and_games,
#                cds_and_vinyl, books, ...)
#   loss         bce (one sampled negative, S3-Rec recipe), ce (softmax over all
#                items) or sampled_ce (sampled softmax, for large catalogues)
#   hidden_size  64 for the original width, or the parameter-matched width
# Optional env vars:
#   LR               learning rate for every item (default 0.0003)
#   IDLE_MINUTES     minutes the GPU must stay idle before the first item (5)
#   GPU_MAX_USED_MIB idle threshold (500)
# Items whose checkpoint already exists are skipped. Test metrics are written at
# the end of artifacts/sasrec/logs/<run_id>.log; collect them with
# scripts/results/collect_sasrec_audit.py.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(pwd -P)" != "${SCRIPT_DIR}" ]]; then
  echo "ERROR: run this script from ${SCRIPT_DIR}" >&2
  exit 2
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/jobs/lib/local_gpu.sh"

: "${GPU:?Set GPU to a GPU index.}"
: "${QUEUE:?Set QUEUE to space-separated dataset:loss:hidden items.}"
LR="${LR:-0.0003}"
IDLE_MINUTES="${IDLE_MINUTES:-5}"
GPU_MAX_USED_MIB="${GPU_MAX_USED_MIB:-500}"
PYTHON_BIN="$(local_gpu_python "${REPO_ROOT}")"

OUTPUT_DIR="${REPO_ROOT}/output/sasrec_audit"
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/queue-gpu${GPU}-$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

wait_for_idle_gpu() {
  local idle_since=""
  while true; do
    local used
    used="$(nvidia-smi --id="${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
    if [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= GPU_MAX_USED_MIB )); then
      [[ -z "${idle_since}" ]] && idle_since="$(date +%s)" && log "GPU ${GPU} idle (${used} MiB); waiting ${IDLE_MINUTES} min"
      if (( $(date +%s) - idle_since >= IDLE_MINUTES * 60 )); then
        log "GPU ${GPU} idle for ${IDLE_MINUTES} min; starting"
        return 0
      fi
    else
      idle_since=""
    fi
    sleep 30
  done
}

log "queue on GPU ${GPU}: ${QUEUE} (lr=${LR})"
wait_for_idle_gpu
export CUDA_VISIBLE_DEVICES="${GPU}"
cd "${REPO_ROOT}"

for item in ${QUEUE}; do
  IFS=: read -r dataset loss hidden overrides <<< "${item}"
  run_id="sasrec_audit_${dataset}_${loss}_h${hidden}_lr${LR//./p}"
  if [[ -f "artifacts/sasrec/ckpt/${run_id}.pt" ]] && grep -q '"ndcg@10"' "artifacts/sasrec/logs/${run_id}.log" 2>/dev/null; then
    log "skip ${item}: ${run_id} already trained and tested"
    continue
  fi
  override_args=()
  if [[ -n "${overrides:-}" ]]; then
    IFS=, read -r -a override_list <<< "${overrides}"
    for kv in "${override_list[@]}"; do override_args+=("--${kv}"); done
  fi
  log "train ${item} run_id=${run_id}"
  "${PYTHON_BIN}" scripts/sasrec/train.py \
    --preset "${dataset}" \
    --loss_type "${loss}" \
    --hidden_size "${hidden}" \
    --lr "${LR}" \
    --run_id "${run_id}" \
    "${override_args[@]}" > "${OUTPUT_DIR}/${run_id}.out" 2>&1 \
    || log "FAILED ${item}; see ${OUTPUT_DIR}/${run_id}.out"
done
log "queue on GPU ${GPU} finished"
