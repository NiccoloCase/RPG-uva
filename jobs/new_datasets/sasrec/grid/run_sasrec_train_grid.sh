#!/bin/bash

# SASRec training-hyperparameter grid for the two new datasets: full retrain per
# (dataset, lr, dropout, seed) cell, validation-selected by val NDCG@20.
#
# Submit from this directory:
#   cd /gpfs/home6/$USER/RPG-uva/jobs/new_datasets/sasrec/grid
#   sbatch ./run_sasrec_train_grid.sh
# Override via env: CELLS ("lr dropout blocks" triples, newline separated), SEEDS.
# Keep --array in sync: N_DS * N_CELL * N_SEED - 1.

#SBATCH --job-name=sasrec_new_train_grid
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --array=0-21
#SBATCH --output=../../../../output/new_datasets/sasrec/grid/train/%x-%A_%a.out
#SBATCH --error=../../../../output/new_datasets/sasrec/grid/train/%x-%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"

DATASETS=(video_games pet_supplies)
CATEGORIES=(Video_Games Pet_Supplies)
read -r -a SEEDS <<< "${SEEDS:-2024}"
DEFAULT_CELLS="0.0003 0.2 2
0.0003 0.5 2
0.0003 0.8 2
0.001 0.2 2
0.001 0.5 2
0.001 0.8 2
0.003 0.2 2
0.003 0.5 2
0.003 0.8 2
0.001 0.5 1
0.001 0.5 3"
CELL_LIST=()
while IFS= read -r _cell_line; do
  [[ -n "${_cell_line// }" ]] && CELL_LIST+=("${_cell_line}")
done <<< "${CELLS:-$DEFAULT_CELLS}"

N_CELL=${#CELL_LIST[@]}
N_SEED=${#SEEDS[@]}
CELLS_PER_DS=$(( N_CELL * N_SEED ))
TOTAL=$(( ${#DATASETS[@]} * CELLS_PER_DS ))

idx=${SLURM_ARRAY_TASK_ID:-0}
if (( idx >= TOTAL )); then
  echo "ERROR: array index ${idx} exceeds ${TOTAL} tasks; set --array=0-$(( TOTAL - 1 ))" >&2
  exit 2
fi

ds_idx=$(( idx / CELLS_PER_DS ))
rem=$(( idx % CELLS_PER_DS ))
cell_idx=$(( rem / N_SEED ))
seed_idx=$(( rem % N_SEED ))

DS=${DATASETS[$ds_idx]}
CAT=${CATEGORIES[$ds_idx]}
read -r LR DROP LAYERS <<< "${CELL_LIST[$cell_idx]}"
SEED=${SEEDS[$seed_idx]}
RUN_ID="sasrec_grid_${DS}_lr${LR}_d${DROP}_b${LAYERS}_s${SEED}"
OUT_DIR="${REPO_ROOT}/output/new_datasets/sasrec/grid/train"
DATA_FILE="${REPO_ROOT}/artifacts/sasrec/data/${CAT}/${CAT}.txt"

mkdir -p "${OUT_DIR}"

if [[ ! -f "${DATA_FILE}" ]]; then
  echo "ERROR: missing SASRec data file ${DATA_FILE}" >&2
  echo "Run jobs/new_datasets/sasrec/${DS}/prepare_data.sh first." >&2
  exit 3
fi

echo "SASREC_GRID_START dataset=${DS} lr=${LR} dropout=${DROP} blocks=${LAYERS} seed=${SEED} run_id=${RUN_ID}"

module purge
module load 2025
module load Anaconda3/2025.06-1
cd "${REPO_ROOT}"

conda run -n rpg-uva python3 scripts/sasrec.py \
  --preset "${DS}" \
  --dataset "${CAT}" \
  --lr "${LR}" \
  --hidden_dropout_prob "${DROP}" \
  --attention_probs_dropout_prob "${DROP}" \
  --num_hidden_layers "${LAYERS}" \
  --seed "${SEED}" \
  --run_id "${RUN_ID}"

rm -f "${REPO_ROOT}/artifacts/sasrec/ckpt/${RUN_ID}"*

echo "SASREC_GRID_END dataset=${DS} lr=${LR} dropout=${DROP} blocks=${LAYERS} seed=${SEED}"
