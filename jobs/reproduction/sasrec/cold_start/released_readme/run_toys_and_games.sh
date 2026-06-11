#!/bin/bash

# Submit from jobs/reproduction/sasrec/cold_start/released_readme so these relative output paths resolve correctly.
#SBATCH --job-name=sasrec_cold_toys_rel
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=../../../../../output/reproduction/sasrec/cold_start/released_readme/%x-%j.out
#SBATCH --error=../../../../../output/reproduction/sasrec/cold_start/released_readme/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}" && pwd -P)"
PWD_REAL="$(pwd -P)"
if [[ "${PWD_REAL}" != "${SCRIPT_DIR}" ]]; then
  echo "ERROR: run this script from ${SCRIPT_DIR}" >&2
  echo "Run:" >&2
  echo "  cd ${SCRIPT_DIR}" >&2
  echo "  sbatch ./run_toys_and_games.sh" >&2
  exit 2
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
export COLD_START_DATASET_SLUG="${COLD_START_DATASET_SLUG:-toys_and_games}"
export COLD_START_PRESET="${COLD_START_PRESET:-toys_and_games}"
export COLD_START_DATASET="${COLD_START_DATASET:-Toys_and_Games}"
export COLD_START_OUTPUT_DIR="${COLD_START_OUTPUT_DIR:-${REPO_ROOT}/artifacts/sasrec/cold_start/released_readme/toys_and_games}"

(cd .. && bash ./run_cold_start.sh "$@")
