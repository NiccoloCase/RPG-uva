#!/bin/bash

# SASRec training-hyperparameter grid: full retrain per (dataset, lr, dropout, seed)
# cell, validation-selected by val NDCG@20 (SASRec's early-stopping metric).
#
# Parallels the RPG sweep so the two are directly comparable in the report. The
# tunable hparams (SASRec paper, Kang & McAuley 2018, implementation details) are
# the number of self-attention blocks b, learning rate, and dropout; batch size and
# sequence length are fixed. We sweep all three, deduplicated into an explicit cell
# list per dataset:
#   - full lr x dropout grid at the released b=2 (the headline heatmap, comparable
#     to the RPG lr x temperature grid):
#       lr      {0.0003, 0.001, 0.003}   released 0.001
#       dropout {0.2, 0.5, 0.8}          released 0.5 (on hidden AND attention dropout)
#   - block-count OFAT b {1, 2, 3} at the released lr=0.001, dropout=0.5 (b=2 is
#     already the grid centre, so only b=1 and b=3 are added).
# That is 9 + 2 = 11 cells per dataset.
#
# All other fields stay at the released SASRec config (hidden size 64, two heads,
# sequence length 50, batch size 256, weight decay 0, early stopping patience 10 on
# val NDCG@20). Each cell is retrained over three training seeds. seed 2024 is the
# value carried by the released runs.
#
# Flattened array over N_DS x N_CELL x N_SEED tasks (default 3 x 11 x 3 = 99). Each
# task retrains one cell and prints the best-val improvements plus the final test
# OrderedDict to its .out.
#
# Submit from this directory:
#   cd jobs/reproduction/sasrec/grid
#   mkdir -p ../../../../output/reproduction/sasrec/grid/train
#   sbatch run_sasrec_train_grid.sh
# Override via env: CELLS ("lr dropout blocks" triples, newline/space separated),
# SEEDS. Keep --array in sync: N_DS * N_CELL * N_SEED - 1.

#SBATCH --job-name=sasrec_train_grid
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --array=0-98
#SBATCH --output=../../../../output/reproduction/sasrec/grid/train/%x-%A_%a.out
#SBATCH --error=../../../../output/reproduction/sasrec/grid/train/%x-%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"
ENV_PREFIX="${REPO_ROOT}/artifacts/conda/rpg-uva"

DATASETS=(sports_and_outdoors beauty toys_and_games)
CATEGORIES=(Sports_and_Outdoors Beauty Toys_and_Games)
read -r -a SEEDS <<< "${SEEDS:-2024 2025 2026}"
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
OUT_DIR="${REPO_ROOT}/output/reproduction/sasrec/grid/train"
DATA_FILE="${REPO_ROOT}/artifacts/sasrec/data/${CAT}/${CAT}.txt"

mkdir -p "${OUT_DIR}"

if [[ ! -f "${DATA_FILE}" ]]; then
  echo "ERROR: missing SASRec data file ${DATA_FILE} (run scripts/sasrec_prepare_data.py first)" >&2
  exit 3
fi

echo "SASREC_GRID_START dataset=${DS} lr=${LR} dropout=${DROP} blocks=${LAYERS} seed=${SEED} run_id=${RUN_ID}"

module purge
module load 2025
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"
cd "${REPO_ROOT}"

python3 scripts/sasrec.py \
  --preset "${DS}" \
  --dataset "${CAT}" \
  --lr "${LR}" \
  --hidden_dropout_prob "${DROP}" \
  --attention_probs_dropout_prob "${DROP}" \
  --num_hidden_layers "${LAYERS}" \
  --seed "${SEED}" \
  --run_id "${RUN_ID}"

echo "SASREC_GRID_END dataset=${DS} lr=${LR} dropout=${DROP} blocks=${LAYERS} seed=${SEED}"
