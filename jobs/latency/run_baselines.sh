#!/bin/bash

# Submit from jobs/latency so these relative output paths resolve correctly.
#SBATCH --job-name=latency_baselines
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=../../output/latency/%x-%j.out
#SBATCH --error=../../output/latency/%x-%j.err

# Per-query latency of the baselines, with the same protocol as run_latency.sh
# (same users, batch sizes 1/16/128, GPU and CPU): SASRec (S3-Rec recipe,
# width 64), full-softmax SASRec and TIGER, on Sports, CDs and Books (no TIGER
# on Books). Uses the seed-2024 checkpoints.
#   cd jobs/latency && sbatch ./run_baselines.sh
# Env vars: DATASETS (subset of: sports cds books; default all)
# Results: artifacts/sasrec/latency/ and artifacts/memgen/latency/

set -uo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ "$(pwd -P)" != "${SCRIPT_DIR}" ]]; then
    echo "ERROR: run this script from ${SCRIPT_DIR}" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output/latency"
DATASETS="${DATASETS:-sports cds books}"
source "${REPO_ROOT}/jobs/lib/env.sh"
repo_env_activate "${REPO_ROOT}"
export HF_HUB_OFFLINE=1

mkdir -p "${OUTPUT_DIR}"
source "${REPO_ROOT}/jobs/lib/runtime_stats.sh"
runtime_stats_init "${OUTPUT_DIR}" "latency_baselines"
runtime_stats_start_gpu_monitor
trap runtime_stats_finish EXIT

cd "${REPO_ROOT}"
newest() { ls -t $1 2>/dev/null | head -n 1; }
timing=(--devices cuda,cpu --batch-sizes 1,16,128 --cpu-threads 16)

for dataset in ${DATASETS}; do
  case "${dataset}" in
    sports) preset=sports_and_outdoors; users=2000 ;;
    cds)    preset=cds_and_vinyl;       users=2000 ;;
    books)  preset=books;               users=1000 ;;
    *) echo "ERROR: unknown dataset ${dataset}" >&2; exit 1 ;;
  esac
  sasrec_ckpt="artifacts/sasrec/ckpt/seeds/sasrec_${preset}_h64_L2_e300_lr0p0003_s2024.pt"
  runtime_stats_log "SASRec ${dataset}: ${sasrec_ckpt}"
  runtime_stats_run python scripts/sasrec/latency.py --preset "${preset}" --checkpoint "${sasrec_ckpt}" \
    --hidden_size 64 --max-users "${users}" "${timing[@]}"
  for model in SASRec TIGER; do
    [[ "${dataset}" == books && "${model}" == TIGER ]] && continue
    ckpt="$(newest "artifacts/memgen/ckpt/memgen_${model,,}_${dataset}_seed2024-*.pth")"
    if [[ -z "${ckpt}" ]]; then runtime_stats_log "SKIP ${model} ${dataset}: no checkpoint"; continue; fi
    runtime_stats_log "${model} ${dataset}: ${ckpt}"
    runtime_stats_run python scripts/memgen/latency.py --model "${model}" --preset "${dataset}" --checkpoint "${ckpt}" \
      --max-users "${users}" "${timing[@]}"
  done
done
echo "LATENCY_BASELINES_DONE"
