#!/bin/bash

# Submit from jobs/reproduction/sasrec/eval_seeds/released_readme so these relative output paths resolve correctly.
#SBATCH --job-name=sasrec_eval_beauty_rel
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=../../../../../output/reproduction/sasrec/eval_seeds/released_readme/%x-%j.out
#SBATCH --error=../../../../../output/reproduction/sasrec/eval_seeds/released_readme/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}" && pwd -P)"
PWD_REAL="$(pwd -P)"
if [[ "${PWD_REAL}" != "${SCRIPT_DIR}" ]]; then
  echo "ERROR: run this script from ${SCRIPT_DIR}" >&2
  echo "Run:" >&2
  echo "  cd ${SCRIPT_DIR}" >&2
  echo "  sbatch ./run_beauty.sh" >&2
  exit 2
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
export SASREC_EVAL_DATASET_SLUG="${SASREC_EVAL_DATASET_SLUG:-beauty}"
export SASREC_EVAL_PRESET="${SASREC_EVAL_PRESET:-beauty}"
export SASREC_EVAL_DATASET="${SASREC_EVAL_DATASET:-Beauty}"
export SASREC_EVAL_CONFIG="${SASREC_EVAL_CONFIG:-${REPO_ROOT}/configs/sasrec/eval_seeds/released_readme/beauty.yaml}"
export SASREC_EVAL_OUTPUT_DIR="${SASREC_EVAL_OUTPUT_DIR:-${REPO_ROOT}/artifacts/sasrec/eval_seeds/released_readme/beauty}"

(cd .. && bash ./run_eval.sh "$@")
