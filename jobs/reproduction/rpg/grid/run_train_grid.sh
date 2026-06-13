#!/bin/bash

# Training-hyperparameter grid for RPG: full RETRAIN per (dataset, lr, temperature,
# seed) cell, then the pipeline's built-in test evaluation. Codebook length m is held
# at each dataset's best-m (Sports 16, Beauty 32, Toys 16) -- m was already swept in
# Claim 2, so this grid isolates the two trained-in HPs (lr, temperature).
#
# THREE deliberate design choices (vs the earlier single-seed 2-dataset version):
#   1. SEEDS: every cell is retrained over SEEDS (default 2024,2025,2026) via
#      --rand_seed, which reseeds torch/cuda/accelerate at pipeline init
#      (genrec/pipeline.py:52). The sem_ids cache is seed-INDEPENDENT, so each seed
#      reuses identical semantic IDs and we isolate pure training stochasticity
#      (weight init + data order). seed 2024 == genrec default == the old single-seed
#      runs, so those legitimately count as the seed-2024 sample (collector dedups).
#   2. EXTENDED lr: lr grids are widened DOWNWARD because the single-seed Sports
#      optimum sat on the old grid's lower edge (lr=0.001). Each dataset's 4-value lr
#      grid now brackets its tuned optimum on BOTH sides, including a value below it:
#        Sports tuned 0.003 -> {0.0003, 0.001, 0.003, 0.01}
#        Beauty tuned 0.01  -> {0.001, 0.003, 0.01, 0.03}
#        Toys   tuned 0.003 -> {0.0003, 0.001, 0.003, 0.01}
#   3. FULL lr x temperature grid (not OFAT): 4 lr x 3 temp, so we get seeded
#      heatmaps with error bars on BOTH axes.
#
# Everything else (architecture n_embd=448/n_layer=2, dropout embd/attn=0.5, early
# stopping patience=20 on val ndcg@10, 150 max epochs, batch 256) is the released
# RPG default, UNCHANGED -- this is a reproducibility study of their model.
#
# Flattened array over N_DS x N_LR x N_TEMP x N_SEED cells (default 3x4x3x3 = 108).
# Each task retrains one cell from scratch and logs "Test Results: {...}" to its .err.
#
# Submit from this directory:
#   cd jobs/reproduction/rpg/grid
#   mkdir -p ../../../../output/reproduction/rpg/grid/train
#   sbatch run_train_grid.sh
# Override via env: LRS_SPORTS, LRS_BEAUTY, LRS_TOYS, TEMPS, SEEDS (space-separated).
# All three lr grids MUST stay the same length. Keep --array in sync:
#   N_DS * N_LR * N_TEMP * N_SEED - 1.

#SBATCH --job-name=rpg_train_grid
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-107
#SBATCH --output=../../../../output/reproduction/rpg/grid/train/%x-%A_%a.out
#SBATCH --error=../../../../output/reproduction/rpg/grid/train/%x-%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"

DATASETS=(sports_and_outdoors beauty toys_and_games)
BEST_M=(16 32 16)
read -r -a TEMPS      <<< "${TEMPS:-0.01 0.03 0.07}"
read -r -a SEEDS      <<< "${SEEDS:-2024 2025 2026}"
read -r -a LRS_SPORTS <<< "${LRS_SPORTS:-0.0003 0.001 0.003 0.01}"   # tuned 0.003
read -r -a LRS_BEAUTY <<< "${LRS_BEAUTY:-0.001 0.003 0.01 0.03}"     # tuned 0.01
read -r -a LRS_TOYS   <<< "${LRS_TOYS:-0.0003 0.001 0.003 0.01}"     # tuned 0.003

if (( ${#LRS_SPORTS[@]} != ${#LRS_BEAUTY[@]} || ${#LRS_SPORTS[@]} != ${#LRS_TOYS[@]} )); then
  echo "ERROR: LRS_SPORTS, LRS_BEAUTY, LRS_TOYS must all be the same length (uniform array math)" >&2
  exit 2
fi

N_LR=${#LRS_SPORTS[@]}
N_TEMP=${#TEMPS[@]}
N_SEED=${#SEEDS[@]}
CELLS_PER_DS=$(( N_LR * N_TEMP * N_SEED ))
TOTAL=$(( ${#DATASETS[@]} * CELLS_PER_DS ))

idx=${SLURM_ARRAY_TASK_ID:-0}
if (( idx >= TOTAL )); then
  echo "ERROR: array index ${idx} exceeds ${TOTAL} cells; set --array=0-$(( TOTAL - 1 ))" >&2
  exit 2
fi

ds_idx=$(( idx / CELLS_PER_DS ))
rem=$(( idx % CELLS_PER_DS ))
lr_idx=$(( rem / (N_TEMP * N_SEED) ))
rem2=$(( rem % (N_TEMP * N_SEED) ))
temp_idx=$(( rem2 / N_SEED ))
seed_idx=$(( rem2 % N_SEED ))

DS=${DATASETS[$ds_idx]}
M=${BEST_M[$ds_idx]}
case "${ds_idx}" in
  0) LRS=("${LRS_SPORTS[@]}") ;;
  1) LRS=("${LRS_BEAUTY[@]}") ;;
  *) LRS=("${LRS_TOYS[@]}") ;;
esac
LR=${LRS[$lr_idx]}
TEMP=${TEMPS[$temp_idx]}
SEED=${SEEDS[$seed_idx]}
RUN_ID="rpg_grid_${DS}_lr${LR}_t${TEMP}_s${SEED}"
OUT_DIR="${REPO_ROOT}/output/reproduction/rpg/grid/train/${DS}"

mkdir -p "${OUT_DIR}"

echo "TRAIN_GRID_START dataset=${DS} m=${M} lr=${LR} temperature=${TEMP} seed=${SEED} run_id=${RUN_ID}"

module purge
module load 2025
module load Anaconda3/2025.06-1
cd "${REPO_ROOT}"

conda run -n rpg-uva python scripts/rpg.py \
  --preset "${DS}" \
  --n_codebook "${M}" \
  --lr "${LR}" \
  --temperature "${TEMP}" \
  --rand_seed "${SEED}" \
  --run_id "${RUN_ID}"

echo "TRAIN_GRID_END dataset=${DS} lr=${LR} temperature=${TEMP} seed=${SEED}"
