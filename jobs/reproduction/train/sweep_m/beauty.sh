#!/bin/bash

# Submit from jobs/reproduction/train/sweep_m so relative output paths resolve correctly.
#SBATCH --job-name=rpg_sweep_m_beauty
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
  echo "  sbatch ./beauty.sh" >&2
  exit 2
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

M_VALUES=(4 8 16 32 64)
M=${M_VALUES[${SLURM_ARRAY_TASK_ID:-0}]}
RUN_ID="rpg_sweep_m${M}_beauty"

echo "SWEEP_START dataset=Beauty n_codebook=${M} run_id=${RUN_ID}"

mkdir -p "${REPO_ROOT}/output/reproduction/train/sweep_m"

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "${REPO_ROOT}"

conda run -n rpg-uva python scripts/rpg.py \
  --preset beauty \
  --n_codebook "${M}" \
  --run_id "${RUN_ID}"

echo "SWEEP_END dataset=Beauty n_codebook=${M} run_id=${RUN_ID}"
