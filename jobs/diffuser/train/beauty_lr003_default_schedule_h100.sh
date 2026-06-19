#!/bin/bash

# Submit from jobs/diffuser/train when gpu_h100 has better availability.
#SBATCH --partition=gpu_h100
#SBATCH --job-name=drpg_beauty_lr003_s6_h100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --output=../../../output/diffuser/train/%x-%j.out
#SBATCH --error=../../../output/diffuser/train/%x-%j.err

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
    echo "  bash ./beauty_lr003_default_schedule_h100.sh" >&2
    exit 2
  fi
fi

export PRESET=beauty
export RUN_ID=drpg_beauty_lr003_s6_h100
export LR=0.003
export DIFFUSION_MASK_COUNTS="${DIFFUSION_MASK_COUNTS:-32,24,16,8,4,1}"

exec "${SCRIPT_DIR}/train.sh"
