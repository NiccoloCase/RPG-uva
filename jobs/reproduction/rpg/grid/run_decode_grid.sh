#!/bin/bash


#
# Submit from this directory:
#   cd jobs/reproduction/rpg/grid
#   mkdir -p ../../../../output/reproduction/rpg/grid/decode_val
#   sbatch run_decode_grid.sh                  # all 4 datasets x 6 k = 24 tasks
#   sbatch --array=18-23 run_decode_grid.sh    # CDs only (heaviest; idx 18..23)
# Override via env: BEAMS, EDGES, QSTEPS, SEEDS, TOPK, FORCE=1.

#SBATCH --job-name=rpg_decode_grid
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-23
#SBATCH --output=../../../../output/reproduction/rpg/grid/decode_val/%x-%A_%a.out
#SBATCH --error=../../../../output/reproduction/rpg/grid/decode_val/%x-%A_%a.err

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"

# preset name -> Title_Case cache category -> best-m (from Claim 2 sweep)
DATASETS=(sports_and_outdoors beauty toys_and_games cds_and_vinyl)
CATEGORIES=(Sports_and_Outdoors Beauty Toys_and_Games CDs_and_Vinyl)
BEST_M=(16 32 16 64)

read -r -a BEAM_LIST  <<< "${BEAMS:-10 20 50 100 200}"
read -r -a EDGE_LIST  <<< "${EDGES:-20 30 50 100 200 500}"
read -r -a QSTEP_LIST <<< "${QSTEPS:-1 2 3 4 5}"
SEEDS="${SEEDS:-2024,2025,2026}"          # selection seeds (ranking only)
TOPK="${TOPK:-[5,10]}"                     # selection metric NDCG@10; all b>=10
FORCE="${FORCE:-0}"

N_K=${#EDGE_LIST[@]}
TOTAL=$(( ${#DATASETS[@]} * N_K ))

idx=${SLURM_ARRAY_TASK_ID:-0}
if (( idx >= TOTAL )); then
  echo "ERROR: array index ${idx} exceeds ${TOTAL} tasks; set --array=0-$(( TOTAL - 1 ))" >&2
  exit 2
fi

ds_idx=$(( idx / N_K ))
k_idx=$(( idx % N_K ))
DS=${DATASETS[$ds_idx]}
CAT=${CATEGORIES[$ds_idx]}
M=${BEST_M[$ds_idx]}
K=${EDGE_LIST[$k_idx]}

CKPT_DIR="${REPO_ROOT}/artifacts/rpg/ckpt"
CACHE_DIR="${REPO_ROOT}/artifacts/rpg/cache/AmazonReviews2014/${CAT}/processed"
OUT_ROOT="${REPO_ROOT}/output/reproduction/rpg/grid/decode_val/${DS}"
CFG="configs/rpg/repro/${DS}.yaml"

mkdir -p "${OUT_ROOT}"

echo "DECODE_GRID_START dataset=${CAT} preset=${DS} m=${M} split=val k=${K} \
beams='${BEAM_LIST[*]}' qsteps='${QSTEP_LIST[*]}' seeds=${SEEDS} topk='${TOPK}'"

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

# This task owns one k; loop the full b x q grid for it.
echo "Running $(( ${#BEAM_LIST[@]} * ${#QSTEP_LIST[@]} )) cells for k=${K} (b x q)"

for b in "${BEAM_LIST[@]}"; do
  for q in "${QSTEP_LIST[@]}"; do
    OUT_DIR="${OUT_ROOT}/b${b}_k${K}_q${q}"

    if [[ "${FORCE}" != "1" ]]; then
      existing=("${OUT_DIR}"/*/summary.json)
      if [[ ${#existing[@]} -gt 0 ]]; then
        echo "  SKIP b${b}_k${K}_q${q}: already done (${existing[0]}); set FORCE=1 to redo"
        continue
      fi
    fi

    echo "  CELL b=${b} k=${K} q=${q} split=val topk=${TOPK} -> ${OUT_DIR}"
    conda run -n rpg-uva python scripts/rpg_eval_seeds.py \
      --checkpoint "${CKPT}" \
      --config "${CFG}" \
      --split val \
      --eval-seeds "${SEEDS}" \
      --output-dir "${OUT_DIR}" \
      --n_codebook "${M}" \
      --num_beams "${b}" \
      --n_edges "${K}" \
      --propagation_steps "${q}" \
      --topk "${TOPK}"

    # Drop large per-user dumps once summary.json is written; collector reads only summary.json.
    find "${OUT_DIR}" -type f \( -name 'per_user_metrics.csv' -o -name 'per_user_metrics.jsonl' \) -delete
  done
done

echo "DECODE_GRID_END dataset=${CAT} preset=${DS} k=${K}"
