#!/bin/bash


# Submit from this directory :
#   cd jobs/reproduction/rpg/grid
#   mkdir -p ../../../../output/reproduction/rpg/grid/decode_test_confirm
#   sbatch run_decode_confirm.sh                 # all 4 datasets
#   sbatch --array=3 run_decode_confirm.sh       # CDs only
# Override via env: CLUSTER_CSV, EVAL_SEEDS, TOPK, FORCE=1.

#SBATCH --job-name=rpg_decode_confirm
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=../../../../output/reproduction/rpg/grid/decode_test_confirm/%x-%A_%a.out
#SBATCH --error=../../../../output/reproduction/rpg/grid/decode_test_confirm/%x-%A_%a.err

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"

DATASETS=(sports_and_outdoors beauty toys_and_games cds_and_vinyl)
CATEGORIES=(Sports_and_Outdoors Beauty Toys_and_Games CDs_and_Vinyl)
BEST_M=(16 32 16 64)

idx=${SLURM_ARRAY_TASK_ID:-0}
DS=${DATASETS[$idx]}
CAT=${CATEGORIES[$idx]}
M=${BEST_M[$idx]}

CLUSTER_CSV="${CLUSTER_CSV:-${REPO_ROOT}/results/decode_val_cluster.csv}"
EVAL_SEEDS="${EVAL_SEEDS:-2024,2025,2026,2027,2028,2029,2030,2031,2032,2033}"
TOPK="${TOPK:-[5,10]}"
FORCE="${FORCE:-0}"

CKPT_DIR="${REPO_ROOT}/artifacts/rpg/ckpt"
CACHE_DIR="${REPO_ROOT}/artifacts/rpg/cache/AmazonReviews2014/${CAT}/processed"
OUT_ROOT="${REPO_ROOT}/output/reproduction/rpg/grid/decode_test_confirm/${DS}"
CFG="configs/rpg/repro/${DS}.yaml"

if [[ ! -f "${CLUSTER_CSV}" ]]; then
  echo "ERROR: cluster CSV not found: ${CLUSTER_CSV}" >&2
  echo "       Run run_decode_grid.sh then scripts/collect_grid_results.py first." >&2
  exit 3
fi
mkdir -p "${OUT_ROOT}"

echo "DECODE_CONFIRM_START dataset=${CAT} preset=${DS} m=${M} split=test seeds=${EVAL_SEEDS}"
echo "  cluster CSV=${CLUSTER_CSV}"

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
  echo "ERROR: missing sem_ids cache ${SEMIDS}" >&2
  exit 5
fi
echo "checkpoint=$(basename "${CKPT}")"

# Pull this dataset's cluster cells from the CSV.
# Columns: dataset,num_beams,n_edges,propagation_steps,val_mean,val_std,n_seeds,is_argmax
mapfile -t CELLS < <(awk -F, -v ds="${DS}" 'NR>1 && $1==ds {print $2" "$3" "$4}' "${CLUSTER_CSV}")

if [[ ${#CELLS[@]} -eq 0 ]]; then
  echo "ERROR: no cluster rows for dataset ${DS} in ${CLUSTER_CSV}" >&2
  exit 6
fi
echo "Confirming ${#CELLS[@]} cluster cell(s) on test"

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

  echo "  CELL b=${b} k=${k} q=${q} split=test topk=${TOPK} -> ${OUT_DIR}"
  conda run -n rpg-uva python scripts/rpg_eval_seeds.py \
    --checkpoint "${CKPT}" \
    --config "${CFG}" \
    --split test \
    --eval-seeds "${EVAL_SEEDS}" \
    --output-dir "${OUT_DIR}" \
    --n_codebook "${M}" \
    --num_beams "${b}" \
    --n_edges "${k}" \
    --propagation_steps "${q}" \
    --topk "${TOPK}" \
    --no-per-user-output
done

echo "DECODE_CONFIRM_END dataset=${CAT} preset=${DS}"
