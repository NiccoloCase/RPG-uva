#!/bin/bash

# Submit from jobs/reproduction/sasrec/cold_start/released_readme so these relative output paths resolve correctly.
#SBATCH --job-name=sasrec_cold_sports_rel
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
  echo "  sbatch ./run_sports_and_outdoors.sh" >&2
  exit 2
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
export COLD_START_DATASET_SLUG="${COLD_START_DATASET_SLUG:-sports_and_outdoors}"
export COLD_START_PRESET="${COLD_START_PRESET:-sports_and_outdoors}"
export COLD_START_DATASET="${COLD_START_DATASET:-Sports_and_Outdoors}"
export COLD_START_OUTPUT_DIR="${COLD_START_OUTPUT_DIR:-${REPO_ROOT}/artifacts/sasrec/cold_start/released_readme/sports_and_outdoors}"

(cd .. && bash ./run_cold_start.sh "$@")
