#!/bin/bash

# Infer-grid decode sweep for the two new datasets (Pet Supplies, Video Games),
# mirroring run_infer_grid.sh but with EXPLICIT checkpoint paths instead of
# "newest checkpoint matching prefix" auto-discovery. That auto-discovery is
# unsafe here: each m value has 27 different (lr,temp,seed) checkpoints from
# the joint train_grid sweep, not just one, so "newest" could silently pick a
# non-winning hyperparameter combination.
#
# Submit from this directory:
#   cd jobs/reproduction/rpg/grid
#   mkdir -p ../../../../output/reproduction/rpg/grid/infer
#   sbatch --array=0-1 run_infer_grid_new_datasets.sh
# Override grids via env: BEAMS, EDGES, QSTEPS, HIGH_BEAMS, EVAL_SEEDS, TOPK, FORCE=1.

#SBATCH --job-name=rpg_infer_grid_nd
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=../../../../output/reproduction/rpg/grid/infer/%x-%A_%a.out
#SBATCH --error=../../../../output/reproduction/rpg/grid/infer/%x-%A_%a.err

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"

DATASETS=(pet_supplies video_games)
CATEGORIES=(Pet_Supplies Video_Games)
BEST_M=(32 64)
BEST_LR=(0.003 0.001)
BEST_TEMP=(0.03 0.03)
BEST_SEED=(2024 2024)
# Base b/k/q: currently-deployed (copied) values, used as the held-fixed axis
# while sweeping the other two -- same role as BASE_B/K/Q in run_infer_grid.sh.
BASE_B=(200 100)
BASE_K=(20 30)
BASE_Q=(3 5)

idx=${SLURM_ARRAY_TASK_ID:-0}
DS=${DATASETS[$idx]}
CAT=${CATEGORIES[$idx]}
M=${BEST_M[$idx]}
LR=${BEST_LR[$idx]}
TEMP=${BEST_TEMP[$idx]}
SEED=${BEST_SEED[$idx]}
b0=${BASE_B[$idx]}
k0=${BASE_K[$idx]}
q0=${BASE_Q[$idx]}

read -r -a BEAM_LIST  <<< "${BEAMS:-10 20 50 100 200}"
read -r -a EDGE_LIST  <<< "${EDGES:-30 50 100 200 500}"
read -r -a QSTEP_LIST <<< "${QSTEPS:-1 2 3 5}"
HIGH_BEAMS="${HIGH_BEAMS:-200}"
EVAL_SEEDS="${EVAL_SEEDS:-2024,2025,2026,2027,2028,2029,2030,2031,2032,2033}"

read -r -a CUTOFFS <<< "${CUTOFFS:-5 10 50 100}"
FORCE="${FORCE:-0}"

CKPT_DIR="${REPO_ROOT}/artifacts/rpg/ckpt"
CACHE_DIR="${REPO_ROOT}/artifacts/rpg/cache/AmazonReviews2014/${CAT}/processed"
OUT_ROOT="${REPO_ROOT}/output/reproduction/rpg/grid/infer/${DS}"
CFG="configs/rpg/new_datasets/${DS}.yaml"

mkdir -p "${OUT_ROOT}"

echo "INFER_GRID_START dataset=${CAT} preset=${DS} m=${M} lr=${LR} temp=${TEMP} seed=${SEED} base(b/k/q)=${b0}/${k0}/${q0} cutoffs='${CUTOFFS[*]}' seeds=${EVAL_SEEDS}"

topk_for_beams() {
  local b="$1" out="" c
  for c in "${CUTOFFS[@]}"; do
    (( c <= b )) && out="${out:+$out,}$c"
  done
  [[ -z "$out" ]] && out="$b"
  printf '[%s]' "$out"
}

module purge
module load 2025
module load Anaconda3/2025.06-1
cd "${REPO_ROOT}"

RUN_ID="rpg_sweep_m${M}_${DS}_lr${LR}_t${TEMP}_s${SEED}"
CKPT="$(find "${CKPT_DIR}" -maxdepth 1 -type f -name "${RUN_ID}-*.pth" | sort | tail -n 1)"
SEMIDS="${CACHE_DIR}/text-embedding-3-large_OPQ${M},IVF1,PQ${M}x8.sem_ids"

if [[ -z "${CKPT}" ]]; then
  echo "ERROR: no checkpoint matching ${CKPT_DIR}/${RUN_ID}-*.pth" >&2
  exit 4
fi
if [[ ! -f "${SEMIDS}" ]]; then
  echo "ERROR: missing sem_ids cache ${SEMIDS}" >&2
  exit 5
fi
echo "checkpoint=$(basename "${CKPT}")"

declare -A SEEN=()
CELLS=()
add_cell() {
  local key="$1_$2_$3"
  if [[ -z "${SEEN[$key]:-}" ]]; then
    SEEN[$key]=1
    CELLS+=("$1 $2 $3")
  fi
}
for b in "${BEAM_LIST[@]}";  do add_cell "$b" "$k0" "$q0"; done
for k in "${EDGE_LIST[@]}";  do add_cell "$HIGH_BEAMS" "$k" "$q0"; done
for q in "${QSTEP_LIST[@]}"; do add_cell "$HIGH_BEAMS" "$k0" "$q"; done

echo "Running ${#CELLS[@]} decode cells"

for cell in "${CELLS[@]}"; do
  read -r b k q <<< "${cell}"
  OUT_DIR="${OUT_ROOT}/b${b}_k${k}_q${q}"

  if [[ "${FORCE}" != "1" ]]; then
    existing=("${OUT_DIR}"/*/summary.json)
    if [[ ${#existing[@]} -gt 0 ]]; then
      echo "  SKIP b${b}_k${k}_q${q}: already done (${existing[0]}); set FORCE=1 to redo"
      continue
    fi
  fi

  TOPK_CELL="$(topk_for_beams "${b}")"
  echo "  CELL b=${b} k=${k} q=${q} topk=${TOPK_CELL} -> ${OUT_DIR}"
  conda run -n rpg-uva python scripts/rpg_eval_seeds.py \
    --checkpoint "${CKPT}" \
    --config "${CFG}" \
    --eval-seeds "${EVAL_SEEDS}" \
    --output-dir "${OUT_DIR}" \
    --n_codebook "${M}" \
    --num_beams "${b}" \
    --n_edges "${k}" \
    --propagation_steps "${q}" \
    --topk "${TOPK_CELL}"

  find "${OUT_DIR}" -type f \( -name 'per_user_metrics.csv' -o -name 'per_user_metrics.jsonl' \) -delete
done

echo "INFER_GRID_END dataset=${CAT} preset=${DS}"
