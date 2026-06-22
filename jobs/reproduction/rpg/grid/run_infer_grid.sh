#!/bin/bash


# Submit from this directory:
#   cd jobs/reproduction/rpg/grid
#   mkdir -p ../../../../output/reproduction/rpg/grid/infer
#   sbatch run_infer_grid.sh
# Override grids via env: BEAMS, EDGES, QSTEPS, HIGH_BEAMS, EVAL_SEEDS, TOPK, FORCE=1.

#SBATCH --job-name=rpg_infer_grid
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=../../../../output/reproduction/rpg/grid/infer/%x-%A_%a.out
#SBATCH --error=../../../../output/reproduction/rpg/grid/infer/%x-%A_%a.err

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"

DATASETS=(sports_and_outdoors beauty toys_and_games cds_and_vinyl)
CATEGORIES=(Sports_and_Outdoors Beauty Toys_and_Games CDs_and_Vinyl)
BEST_M=(16 32 16 64)
BASE_B=(100 20 200 20)
BASE_K=(30 200 20 500)
BASE_Q=(5 3 3 5)

idx=${SLURM_ARRAY_TASK_ID:-0}
DS=${DATASETS[$idx]}
CAT=${CATEGORIES[$idx]}
M=${BEST_M[$idx]}
b0=${BASE_B[$idx]}
k0=${BASE_K[$idx]}
q0=${BASE_Q[$idx]}

read -r -a BEAM_LIST  <<< "${BEAMS:-10 20 50 100 200}"
read -r -a EDGE_LIST  <<< "${EDGES:-30 50 100 200 500}"
read -r -a QSTEP_LIST <<< "${QSTEPS:-1 2 3 5}"
HIGH_BEAMS="${HIGH_BEAMS:-200}"          # beam width used while sweeping k and q
EVAL_SEEDS="${EVAL_SEEDS:-2024,2025,2026,2027,2028,2029,2030,2031,2032,2033}"

read -r -a CUTOFFS <<< "${CUTOFFS:-5 10 50 100}"
FORCE="${FORCE:-0}"

CKPT_DIR="${REPO_ROOT}/artifacts/rpg/ckpt"
CACHE_DIR="${REPO_ROOT}/artifacts/rpg/cache/AmazonReviews2014/${CAT}/processed"
OUT_ROOT="${REPO_ROOT}/output/reproduction/rpg/grid/infer/${DS}"
CFG="configs/rpg/repro/${DS}.yaml"

mkdir -p "${OUT_ROOT}"

echo "INFER_GRID_START dataset=${CAT} preset=${DS} m=${M} base(b/k/q)=${b0}/${k0}/${q0} cutoffs='${CUTOFFS[*]}' seeds=${EVAL_SEEDS}"

# Python list literal of cutoffs <= the given beam width (assert needs max(topk) <= beams).
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

newest_checkpoint() {
  local prefix="$1" newest="" f
  for f in "${CKPT_DIR}/${prefix}"-*.pth; do
    [[ -z "$newest" || "$f" -nt "$newest" ]] && newest="$f"
  done
  printf '%s' "$newest"
}

RUN_ID="rpg_sweep_m${M}_${DS}"
CKPT="$(newest_checkpoint "${RUN_ID}")"
SEMIDS="${CACHE_DIR}/text-embedding-3-large_OPQ${M},IVF1,PQ${M}x8.sem_ids"

if [[ -z "${CKPT}" ]]; then
  echo "ERROR: no checkpoint matching ${CKPT_DIR}/${RUN_ID}-*.pth" >&2
  exit 4
fi
if [[ ! -f "${SEMIDS}" ]]; then
  echo "ERROR: missing sem_ids cache ${SEMIDS} (run scripts/rpg_prepare_semantic_ids.py first)" >&2
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
