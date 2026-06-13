#!/bin/bash

# Faithful replication of the paper's Figure 6 (inference hyperparameter analysis).
# DECODE-ONLY: re-decodes each dataset's best-m sweep checkpoint, no retraining.
#
# Unlike run_infer_grid.sh (which pivots around the repo decode config with
# HIGH_BEAMS=200 to keep NDCG@{50,100} valid -- the candidate-budget-cap study),
# this script reproduces Figure 6 EXACTLY:
#   - base operating point = the paper's Fig 6 default:  b=10, k=50, q=2
#   - paper sweep ranges:    b in {10,20,30,40,50}
#                            k in {10,20,50,100,200,300}
#                            q in {0,1,2,3,4,5}   (q=0 = no graph propagation)
#   - one-factor-at-a-time:  when sweeping one param, the OTHER TWO stay at the
#                            paper base (b=10, k=50, q=2) -- this is what makes
#                            each Fig 6 subplot a clean isolation, and is the key
#                            difference from run_infer_grid.sh.
#   - metric = NDCG@10 (paper reports @10; base b=10 caps cutoffs at 10, so topk=[5,10]).
#   - 3 eval seeds -> we report mean +/- std (error bars), strictly stronger than
#     the paper's single line.
#
# The paper only ran Fig 6 on Sports. Array index 0 = Sports = the STRICT
# replication; indices 1-3 (Beauty/Toys/CDs) are a beyond-paper generalisation
# of Fig 6, run at the SAME paper base so the curves are comparable across datasets.
#
# Submit from this directory:
#   cd jobs/reproduction/rpg/grid
#   mkdir -p ../../../../output/reproduction/rpg/grid/fig6
#   sbatch run_fig6_repro.sh                 # all 4 datasets
#   sbatch --array=0 run_fig6_repro.sh       # Sports only (strict replication)
# Override via env: BEAMS, EDGES, QSTEPS, BASE_B/BASE_K/BASE_Q, EVAL_SEEDS, FORCE=1.

#SBATCH --job-name=rpg_fig6_repro
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=../../../../output/reproduction/rpg/grid/fig6/%x-%A_%a.out
#SBATCH --error=../../../../output/reproduction/rpg/grid/fig6/%x-%A_%a.err

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"

# preset name -> Title_Case cache category -> best-m (from Claim 2 sweep)
DATASETS=(sports_and_outdoors beauty toys_and_games cds_and_vinyl)
CATEGORIES=(Sports_and_Outdoors Beauty Toys_and_Games CDs_and_Vinyl)
BEST_M=(16 32 16 64)

idx=${SLURM_ARRAY_TASK_ID:-0}
DS=${DATASETS[$idx]}
CAT=${CATEGORIES[$idx]}
M=${BEST_M[$idx]}

# Paper Fig 6 base + sweep ranges (identical across datasets, by design).
b0="${BASE_B:-10}"
k0="${BASE_K:-50}"
q0="${BASE_Q:-2}"
read -r -a BEAM_LIST  <<< "${BEAMS:-10 20 30 40 50}"
read -r -a EDGE_LIST  <<< "${EDGES:-10 20 50 100 200 300}"
read -r -a QSTEP_LIST <<< "${QSTEPS:-0 1 2 3 4 5}"
EVAL_SEEDS="${EVAL_SEEDS:-2024,2025,2026,2027,2028,2029,2030,2031,2032,2033}"
TOPK="${TOPK:-[5,10]}"          # base b=10 caps cutoffs at 10; paper reports @10
FORCE="${FORCE:-0}"

CKPT_DIR="${REPO_ROOT}/artifacts/rpg/ckpt"
CACHE_DIR="${REPO_ROOT}/artifacts/rpg/cache/AmazonReviews2014/${CAT}/processed"
OUT_ROOT="${REPO_ROOT}/output/reproduction/rpg/grid/fig6/${DS}"
CFG="configs/rpg/repro/${DS}.yaml"

mkdir -p "${OUT_ROOT}"

echo "FIG6_START dataset=${CAT} preset=${DS} m=${M} base(b/k/q)=${b0}/${k0}/${q0} topk='${TOPK}' seeds=${EVAL_SEEDS}"

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

# Deduplicated OFAT cells: each sweep holds the other two params at the paper base.
declare -A SEEN=()
CELLS=()
add_cell() {
  local key="$1_$2_$3"
  if [[ -z "${SEEN[$key]:-}" ]]; then
    SEEN[$key]=1
    CELLS+=("$1 $2 $3")
  fi
}
for b in "${BEAM_LIST[@]}";  do add_cell "$b" "$k0" "$q0"; done   # beam sweep  @ k0,q0
for k in "${EDGE_LIST[@]}";  do add_cell "$b0" "$k" "$q0"; done   # edge sweep  @ b0,q0
for q in "${QSTEP_LIST[@]}"; do add_cell "$b0" "$k0" "$q"; done   # step sweep  @ b0,k0

echo "Running ${#CELLS[@]} decode cells (paper Fig 6 OFAT)"

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

  echo "  CELL b=${b} k=${k} q=${q} topk=${TOPK} -> ${OUT_DIR}"
  conda run -n rpg-uva python scripts/rpg_eval_seeds.py \
    --checkpoint "${CKPT}" \
    --config "${CFG}" \
    --eval-seeds "${EVAL_SEEDS}" \
    --output-dir "${OUT_DIR}" \
    --n_codebook "${M}" \
    --num_beams "${b}" \
    --n_edges "${k}" \
    --propagation_steps "${q}" \
    --topk "${TOPK}"
done

echo "FIG6_END dataset=${CAT} preset=${DS}"
