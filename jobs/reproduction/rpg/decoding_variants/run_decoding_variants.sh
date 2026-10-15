#!/bin/bash

# RQ3: graph-constrained decoding with the exact graph, with the approximate
# HNSW graph, and exhaustive scoring, on the same trained RPG checkpoints and
# the same evaluation seeds. Runs scripts/rpg_eval_seeds.py once per variant.
# CPU is enough for the four paper datasets (the HNSW graph is built on CPU in
# under a second; graph decoding takes a few ms per user). Run from this folder:
#
#   cd jobs/reproduction/rpg/decoding_variants
#   DATASETS="sports_and_outdoors beauty toys_and_games cds_and_vinyl" bash ./run_decoding_variants.sh
#
# Env vars:
#   DATASETS     space-separated dataset slugs (default: the four paper datasets)
#   CKPT_GLOB    how to find a dataset's checkpoint under artifacts/rpg/ckpt
#                (default "rpg_repro_<dataset>-*.pth"; "*<dataset>*" also works)
#   EVAL_SEEDS   comma-separated eval seeds for graph decoding (default 2024,2025,2026)
#   VARIANTS     subset of "exact_gpu hnsw exhaustive" (default all)
#   GPU          optional GPU index; unset = CPU
# Results: artifacts/rpg/eval_seeds/decoding_variants/<dataset>/<variant>/<session>/summary.json

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(pwd -P)" != "${SCRIPT_DIR}" ]]; then
  echo "ERROR: run this script from ${SCRIPT_DIR}" >&2
  exit 2
fi
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
source "${REPO_ROOT}/jobs/lib/local_gpu.sh"

DATASETS="${DATASETS:-sports_and_outdoors beauty toys_and_games cds_and_vinyl}"
EVAL_SEEDS="${EVAL_SEEDS:-2024,2025,2026}"
VARIANTS="${VARIANTS:-exact_gpu hnsw exhaustive}"
PYTHON_BIN="$(local_gpu_python "${REPO_ROOT}")"
if [[ -n "${GPU:-}" ]]; then
  local_gpu_require_idle "${GPU}" || exit 1
  export CUDA_VISIBLE_DEVICES="${GPU}"
else
  export CUDA_VISIBLE_DEVICES=""
fi

OUTPUT_DIR="${REPO_ROOT}/output/reproduction/rpg/decoding_variants"
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/run-$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

for dataset in ${DATASETS}; do
  local_gpu_resolve_dataset "${REPO_ROOT}" "${dataset}" || exit 1
  pattern="${CKPT_GLOB:-rpg_repro_${dataset}-*.pth}"
  checkpoint="$(ls -t "${REPO_ROOT}"/artifacts/rpg/ckpt/${pattern} 2>/dev/null | head -1)"
  if [[ -z "${checkpoint}" ]]; then
    log "SKIP ${dataset}: no checkpoint matching ${pattern}"
    continue
  fi
  log "${dataset}: checkpoint $(basename "${checkpoint}" | cut -c1-80)"
  for variant in ${VARIANTS}; do
    result_dir="${REPO_ROOT}/artifacts/rpg/eval_seeds/decoding_variants/${dataset}/${variant}"
    if [[ "${variant}" == "exhaustive" ]]; then
      extra=(--decoding exhaustive --eval-seeds 2024)
    else
      extra=(--decoding graph --graph-backend "${variant}" --eval-seeds "${EVAL_SEEDS}")
    fi
    log "  ${variant}"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/rpg_eval_seeds.py" \
      --checkpoint "${checkpoint}" --config "${DATASET_CONFIG}" --no-local-config \
      --output-dir "${result_dir}" --no-per-user-output "${extra[@]}" \
      > "${OUTPUT_DIR}/${dataset}_${variant}.out" 2>&1 \
      && log "  done ${variant}: $(grep -o '"ndcg@10"[^}]*' "$(ls -t "${result_dir}"/*/summary.json | head -1)" | head -1 | cut -c1-80)" \
      || log "  FAILED ${variant}; see ${OUTPUT_DIR}/${dataset}_${variant}.out"
  done
done
log "DECODING_VARIANTS_DONE"
