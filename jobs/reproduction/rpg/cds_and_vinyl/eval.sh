#!/bin/bash

# Submit from jobs/reproduction/rpg/cds_and_vinyl so these relative output paths resolve correctly.
# Use EVAL_SEED=2024 to match the default reproduction seed unless explicitly overridden.
# Override decode params with NUM_BEAMS, N_EDGES, and PROPAGATION_STEPS if needed.
# Default values follow the repo entrypoint defaults.
#SBATCH --partition=gpu_a100
#SBATCH --job-name=rpg_cds_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --output=../../../../output/reproduction/rpg/cds_and_vinyl/%x-%j.out
#SBATCH --error=../../../../output/reproduction/rpg/cds_and_vinyl/%x-%j.err

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
    echo "  bash ./eval.sh" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output/reproduction/rpg/cds_and_vinyl"
DATA_DIR="${REPO_ROOT}/artifacts/rpg/cache/AmazonReviews2014/CDs_and_Vinyl"
CKPT_DIR="${REPO_ROOT}/artifacts/rpg/ckpt"
ENV_PREFIX="${REPO_ROOT}/artifacts/conda/rpg-uva"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
EVAL_SEED="${EVAL_SEED:-2024}"
NUM_BEAMS="${NUM_BEAMS:-}"
N_EDGES="${N_EDGES:-}"
PROPAGATION_STEPS="${PROPAGATION_STEPS:-}"

mkdir -p "${OUTPUT_DIR}"
source "${REPO_ROOT}/jobs/lib/runtime_stats.sh"
runtime_stats_init "${OUTPUT_DIR}" "rpg_cds_eval"
runtime_stats_start_gpu_monitor
trap runtime_stats_finish EXIT

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "ERROR: missing cached dataset directory: ${DATA_DIR}" >&2
  exit 3
fi

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="$(
    find "${CKPT_DIR}" -maxdepth 1 -type f -name 'rpg_repro_cds_and_vinyl-*.pth' -printf '%T@ %p\n' \
      | sort -nr \
      | sed -n '1s/^[^ ]* //p'
  )"
fi

if [[ -z "${CHECKPOINT_PATH}" || ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: could not find a CDs_and_Vinyl checkpoint in ${CKPT_DIR}" >&2
  echo "Set CHECKPOINT_PATH=/path/to/checkpoint.pth to override." >&2
  exit 4
fi

module purge
module load 2025
module load Anaconda3/2025.06-1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

cd "${REPO_ROOT}"

echo "Evaluating checkpoint: ${CHECKPOINT_PATH}"
echo "Eval seed: ${EVAL_SEED}"

cmd=(
  python3 scripts/rpg_eval.py
  --preset cds_and_vinyl
  --checkpoint "${CHECKPOINT_PATH}"
  --eval-seed "${EVAL_SEED}"
)

if [[ -n "${NUM_BEAMS}" ]]; then
  cmd+=(--num_beams "${NUM_BEAMS}")
fi
if [[ -n "${N_EDGES}" ]]; then
  cmd+=(--n_edges "${N_EDGES}")
fi
if [[ -n "${PROPAGATION_STEPS}" ]]; then
  cmd+=(--propagation_steps "${PROPAGATION_STEPS}")
fi

runtime_stats_run "${cmd[@]}"
