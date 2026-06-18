#!/bin/bash

# Submit from jobs/new_datasets/sasrec/eval_seeds/video_games so these relative output paths resolve correctly.
#SBATCH --job-name=sasrec_eval_video_games
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=../../../../../output/new_datasets/sasrec/eval_seeds/video_games/%x-%j.out
#SBATCH --error=../../../../../output/new_datasets/sasrec/eval_seeds/video_games/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}" && pwd -P)"
PWD_REAL="$(pwd -P)"
if [[ "${PWD_REAL}" != "${SCRIPT_DIR}" ]]; then
  echo "ERROR: run this script from ${SCRIPT_DIR}" >&2
  echo "Run:" >&2
  echo "  cd ${SCRIPT_DIR}" >&2
  echo "  sbatch ./run_video_games.sh" >&2
  exit 2
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
export SASREC_EVAL_DATASET_SLUG="${SASREC_EVAL_DATASET_SLUG:-video_games}"
export SASREC_EVAL_DATASET="${SASREC_EVAL_DATASET:-Video_Games}"
export SASREC_EVAL_CONFIG="${SASREC_EVAL_CONFIG:-${REPO_ROOT}/configs/sasrec/eval_seeds/new_datasets/video_games.yaml}"
export SASREC_EVAL_OUTPUT_DIR="${SASREC_EVAL_OUTPUT_DIR:-${REPO_ROOT}/artifacts/sasrec/eval_seeds/new_datasets/video_games}"

(cd .. && bash ./run_eval.sh "$@")
