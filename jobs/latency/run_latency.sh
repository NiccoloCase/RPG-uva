#!/bin/bash

# Per-query latency of RPG graph decoding versus exhaustive scoring on real
# catalogues of increasing size (Sports 18k, CDs 64k, Books 368k items), on GPU
# and CPU, at batch sizes 1, 16 and 128. Runs directly on a shared GPU server
# (no Slurm). Run from this folder:
#
#   cd jobs/latency
#   GPU=4 bash ./run_latency.sh
#
# Optional env vars:
#   DATASETS  subset of: sports cds books_m32 books_m64 (default: all four)
#   GPU_MAX_USED_MIB  idle threshold for the GPU check (500)
# Results: artifacts/rpg/latency/<Category>_<timestamp>/{latency.csv,summary.json}
#
# Checkpoints: the newest file matching each pattern below. The Sports and CDs
# checkpoints were trained on this server with the released RPG settings and
# the original RPG loss; their file names carry the run id of the job that
# trained them (rpg_loss_token_ce_*). Any RPG checkpoint for the dataset works.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(pwd -P)" != "${SCRIPT_DIR}" ]]; then
  echo "ERROR: run this script from ${SCRIPT_DIR}" >&2
  exit 2
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/jobs/lib/local_gpu.sh"

: "${GPU:?Set GPU to the index of an idle GPU.}"
DATASETS="${DATASETS:-sports cds books_m32 books_m64}"
PYTHON_BIN="$(local_gpu_python "${REPO_ROOT}")"
CKPT_DIR="${REPO_ROOT}/artifacts/rpg/ckpt"

OUTPUT_DIR="${REPO_ROOT}/output/latency"
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/latency-gpu${GPU}-$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

newest() { ls -t "${CKPT_DIR}"/$1 2>/dev/null | head -n 1; }

local_gpu_require_idle "${GPU}" || exit 3
export CUDA_VISIBLE_DEVICES="${GPU}"
cd "${REPO_ROOT}"

for dataset in ${DATASETS}; do
  extra=()
  case "${dataset}" in
    sports)
      checkpoint="$(newest 'rpg_loss_token_ce_sports_and_outdoors-*.pth')"
      config="configs/rpg/repro/sports_and_outdoors.yaml"
      budgets="100:30:5"; max_users=2000 ;;
    cds)
      checkpoint="$(newest 'rpg_loss_token_ce_cds_and_vinyl-*.pth')"
      config="configs/rpg/repro/cds_and_vinyl.yaml"
      budgets="20:500:5"; max_users=2000 ;;
    books_m32)
      checkpoint="$(newest 'rpg_repro_books_m32-*.pth')"
      config="configs/rpg/books.yaml"
      budgets="20:500:5,200:2000:5"; max_users=1000; extra=(--n_codebook 32) ;;
    books_m64)
      checkpoint="$(newest 'rpg_repro_books-*.pth')"
      config="configs/rpg/books.yaml"
      budgets="20:500:5,200:2000:5"; max_users=1000 ;;
    *) echo "unknown dataset ${dataset}"; continue ;;
  esac
  if [[ -z "${checkpoint}" ]]; then
    echo "LATENCY_SKIP ${dataset}: no checkpoint"
    continue
  fi
  echo "LATENCY_START ${dataset} $(date -Is) checkpoint=${checkpoint}"
  "${PYTHON_BIN}" scripts/rpg/latency.py \
    --checkpoint "${checkpoint}" \
    --config "${config}" \
    --devices cuda,cpu \
    --batch-sizes 1,16,128 \
    --warmup-batches 2 \
    --max-users "${max_users}" \
    --budgets "${budgets}" \
    "${extra[@]}" || echo "LATENCY_FAILED ${dataset}"
  echo "LATENCY_END ${dataset} $(date -Is)"
done
echo "LATENCY_DONE"
