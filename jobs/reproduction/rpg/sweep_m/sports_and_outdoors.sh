#!/bin/bash
# Semantic ID length sweep on Sports: train RPG with m = 4, 8, 16, 32, 64 tokens (one array task each),
# all other settings as released. Re-decode the checkpoints with redecode_configs.sh afterwards.
#   cd jobs/reproduction/rpg/sweep_m && sbatch ./sports_and_outdoors.sh

# Submit from jobs/reproduction/rpg/sweep_m so relative output paths resolve correctly.
#SBATCH --job-name=rpg_sweep_m_sports
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --array=0-4
#SBATCH --output=../../../../output/reproduction/train/sweep_m/%x-%A_%a.out
#SBATCH --error=../../../../output/reproduction/train/sweep_m/%x-%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}" && pwd -P)"
PWD_REAL="$(pwd -P)"
if [[ "${PWD_REAL}" != "${SCRIPT_DIR}" ]]; then
  echo "ERROR: run this script from ${SCRIPT_DIR}" >&2
  echo "Run:" >&2
  echo "  cd ${SCRIPT_DIR}" >&2
  echo "  sbatch ./sports_and_outdoors.sh" >&2
  exit 2
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

M_VALUES=(4 8 16 32 64)
M=${M_VALUES[${SLURM_ARRAY_TASK_ID:-0}]}
RUN_ID="rpg_sweep_m${M}_sports_and_outdoors"

echo "SWEEP_START dataset=Sports_and_Outdoors n_codebook=${M} run_id=${RUN_ID}"

mkdir -p "${REPO_ROOT}/output/reproduction/train/sweep_m"

source "${REPO_ROOT}/jobs/lib/env.sh"
repo_env_activate "${REPO_ROOT}"

cd "${REPO_ROOT}"

python scripts/rpg/train.py \
  --preset sports_and_outdoors \
  --n_codebook "${M}" \
  --run_id "${RUN_ID}"

echo "SWEEP_END dataset=Sports_and_Outdoors n_codebook=${M} run_id=${RUN_ID}"
