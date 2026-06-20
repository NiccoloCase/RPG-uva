#!/bin/bash

# Submit from jobs/reproduction/sasrec_modernized/ablation_size so these relative output paths resolve correctly.
# Partition choice checked on 2026-06-19: gpu_a100 had the most idle capacity among gpu_a100/gpu_h100/gpu_mig.
#SBATCH --partition=gpu_a100
#SBATCH --job-name=sasrec_modernized_sports_size_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --output=../../../../output/reproduction/sasrec_modernized/ablation_size/%x-%j.out
#SBATCH --error=../../../../output/reproduction/sasrec_modernized/ablation_size/%x-%j.err

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
    echo "  bash ./train_sports.sh" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output/reproduction/sasrec_modernized/ablation_size"
DATA_FILE="${REPO_ROOT}/artifacts/sasrec/data/Sports_and_Outdoors/Sports_and_Outdoors.txt"
ENV_PREFIX="${REPO_ROOT}/artifacts/conda/rpg-uva"

mkdir -p "${OUTPUT_DIR}"
source "${REPO_ROOT}/jobs/lib/runtime_stats.sh"
runtime_stats_init "${OUTPUT_DIR}" "sasrec_modernized_sports_size_train"
runtime_stats_start_gpu_monitor
trap runtime_stats_finish EXIT

if [[ ! -f "${DATA_FILE}" ]]; then
  echo "ERROR: missing SASRec data file: ${DATA_FILE}" >&2
  exit 3
fi

module purge
module load 2025
module load Anaconda3/2025.06-1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

cd "${REPO_ROOT}"

runtime_stats_run python3 scripts/sasrec_modernized.py \
  --preset sports_and_outdoors \
  --dataset Sports_and_Outdoors \
  --ckpt_dir artifacts/sasrec/ckpt/ablation_size \
  --run_id sasrec_modernized_sports_and_outdoors_size_match \
  --hidden_size 326
