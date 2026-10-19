#!/bin/bash
# Learning rate x number of blocks grid for parameter-matched SASRec on CDs (one array task per cell).
# The selected setting per dataset is in results/sasrec_train_grid.csv.
#   cd jobs/reproduction/sasrec/tuning && sbatch ./train_cds_lr_depth_grid.sh
# Env vars: LRS (0.001 0.0005 0.0003), LAYERS (1 2 3).

# Submit from jobs/reproduction/sasrec/tuning so these relative output paths resolve correctly.
#SBATCH --partition=gpu_a100
#SBATCH --job-name=sasrec_mod_cds_lr_depth
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --array=0-8
#SBATCH --output=../../../../output/reproduction/sasrec/ablation_size/lr_depth_grid/%x-%A_%a.out
#SBATCH --error=../../../../output/reproduction/sasrec/ablation_size/lr_depth_grid/%x-%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}" && pwd -P)"
PWD_REAL="$(pwd -P)"
if [[ "${PWD_REAL}" != "${SCRIPT_DIR}" ]]; then
  echo "ERROR: run this script from ${SCRIPT_DIR}" >&2
  echo "Run:" >&2
  echo "  cd ${SCRIPT_DIR}" >&2
  echo "  sbatch ./train_cds_lr_depth_grid.sh" >&2
  exit 2
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"
OUTPUT_DIR="${REPO_ROOT}/output/reproduction/sasrec/ablation_size/lr_depth_grid"
CKPT_DIR="${REPO_ROOT}/artifacts/sasrec/ckpt/ablation_size/lr_depth_grid"
DATA_FILE="${REPO_ROOT}/artifacts/sasrec/data/CDs_and_Vinyl/CDs_and_Vinyl.txt"
source "${REPO_ROOT}/jobs/lib/env.sh"
repo_env_activate "${REPO_ROOT}"

read -r -a LRS <<< "${LRS:-0.001 0.0005 0.0003}"
read -r -a LAYERS <<< "${LAYERS:-1 2 3}"
IDX=${SLURM_ARRAY_TASK_ID:-0}
N_LR=${#LRS[@]}
N_LAYER=${#LAYERS[@]}
CELLS=$(( N_LR * N_LAYER ))
if (( IDX < 0 || IDX >= CELLS )); then
  echo "ERROR: array index ${IDX} is out of range for ${CELLS} lr-depth cells; fix --array" >&2
  exit 2
fi

lr_idx=$(( IDX / N_LAYER ))
layer_idx=$(( IDX % N_LAYER ))
LR="${LRS[$lr_idx]}"
NUM_HIDDEN_LAYERS="${LAYERS[$layer_idx]}"
LR_TAG="${LR//./p}"
RUN_ID="sasrec_cds_and_vinyl_size_match_e300_lr${LR_TAG}_L${NUM_HIDDEN_LAYERS}"

mkdir -p "${OUTPUT_DIR}" "${CKPT_DIR}"
source "${REPO_ROOT}/jobs/lib/runtime_stats.sh"
runtime_stats_init "${OUTPUT_DIR}" "${RUN_ID}"
runtime_stats_start_gpu_monitor
trap runtime_stats_finish EXIT

if [[ ! -f "${DATA_FILE}" ]]; then
  echo "ERROR: missing SASRec data file: ${DATA_FILE}" >&2
  exit 3
fi

cd "${REPO_ROOT}"

runtime_stats_run python3 scripts/sasrec/train.py \
  --preset cds_and_vinyl \
  --dataset CDs_and_Vinyl \
  --epochs 300 \
  --lr "${LR}" \
  --num_hidden_layers "${NUM_HIDDEN_LAYERS}" \
  --ckpt_dir artifacts/sasrec/ckpt/ablation_size/lr_depth_grid \
  --run_id "${RUN_ID}" \
  --hidden_size 328
