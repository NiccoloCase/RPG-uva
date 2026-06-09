#!/bin/bash

# Submit from jobs/reproduction/eval_seeds/released_readme so these relative output paths resolve correctly.
#SBATCH --job-name=rpg_eval_sports_rel
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=../../../../output/reproduction/eval_seeds/released_readme/%x-%j.out
#SBATCH --error=../../../../output/reproduction/eval_seeds/released_readme/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}" && pwd -P)"
PWD_REAL="$(pwd -P)"
if [[ "${PWD_REAL}" != "${SCRIPT_DIR}" ]]; then
  echo "ERROR: run this script from ${SCRIPT_DIR}" >&2
  echo "Run:" >&2
  echo "  cd ${SCRIPT_DIR}" >&2
  echo "  sbatch ./run_sports_and_outdoors.sh /gpfs/home6/\$USER/RPG/artifacts/rpg/ckpt/model.pth" >&2
  exit 2
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export EVAL_CONFIG="${EVAL_CONFIG:-${REPO_ROOT}/configs/rpg/eval_seeds/released_readme/sports_and_outdoors.yaml}"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${REPO_ROOT}/artifacts/rpg/eval_seeds/released_readme/sports_and_outdoors}"
export PAPER_METRIC="${PAPER_METRIC:-ndcg@10}"
export PAPER_VALUE="${PAPER_VALUE:-0.0263}"
export EQUIVALENCE_MARGIN="${EQUIVALENCE_MARGIN:-0.001}"

(cd .. && bash ./run_eval_seeds.sh "$@")
