#!/bin/bash

#SBATCH --job-name=rpg_redecode_m
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=../../../../output/reproduction/train/sweep_m_redecode/%x-%A_%a.out
#SBATCH --error=../../../../output/reproduction/train/sweep_m_redecode/%x-%A_%a.err

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"

# dataset preset name (for config paths / run_id) -> Title_Case category (for cache path)
DATASETS=(sports_and_outdoors beauty toys_and_games cds_and_vinyl)
CATEGORIES=(Sports_and_Outdoors Beauty Toys_and_Games CDs_and_Vinyl)

idx=${SLURM_ARRAY_TASK_ID:-0}
DS=${DATASETS[$idx]}
CAT=${CATEGORIES[$idx]}

read -r -a M_LIST <<< "${M_VALUES:-4 8 16 32 64}"
EVAL_SEEDS="${EVAL_SEEDS:-2024,2025,2026}"
FORCE="${FORCE:-0}"

CKPT_DIR="${REPO_ROOT}/artifacts/rpg/ckpt"
CACHE_DIR="${REPO_ROOT}/artifacts/rpg/cache/AmazonReviews2014/${CAT}/processed"
OUT_ROOT="${REPO_ROOT}/output/reproduction/train/sweep_m_redecode/${DS}"

mkdir -p "${REPO_ROOT}/output/reproduction/train/sweep_m_redecode"

declare -A CONFIG_FILE=(
  [repo]="configs/rpg/repro/${DS}.yaml"
  [appendix]="configs/rpg/eval_seeds/paper_appendix/${DS}.yaml"
)

echo "REDECODE_START dataset=${CAT} preset=${DS} seeds=${EVAL_SEEDS} m='${M_LIST[*]}'"

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

for M in "${M_LIST[@]}"; do
  RUN_ID="rpg_sweep_m${M}_${DS}"
  CKPT="$(newest_checkpoint "${RUN_ID}")"
  SEMIDS="${CACHE_DIR}/text-embedding-3-large_OPQ${M},IVF1,PQ${M}x8.sem_ids"

  if [[ -z "${CKPT}" ]]; then
    echo "  SKIP m=${M}: no checkpoint matching ${CKPT_DIR}/${RUN_ID}-*.pth" >&2
    continue
  fi
  if [[ ! -f "${SEMIDS}" ]]; then
    echo "  SKIP m=${M}: missing sem_ids cache ${SEMIDS} (run scripts/rpg_prepare_semantic_ids.py first)" >&2
    continue
  fi
  echo "  m=${M} checkpoint=$(basename "${CKPT}")"

  for TAG in repo appendix; do
    CFG="${CONFIG_FILE[$TAG]}"
    OUT_DIR="${OUT_ROOT}/${TAG}_m${M}"

    if [[ "${FORCE}" != "1" ]]; then
      existing=("${OUT_DIR}"/*/summary.json)
      if [[ ${#existing[@]} -gt 0 ]]; then
        echo "    [${TAG}] already done (${existing[0]}); set FORCE=1 to redo" >&2
        continue
      fi
    fi

    echo "    [${TAG}] config=${CFG} -> ${OUT_DIR}"
    conda run -n rpg-uva python scripts/rpg_eval_seeds.py \
      --checkpoint "${CKPT}" \
      --config "${CFG}" \
      --eval-seeds "${EVAL_SEEDS}" \
      --output-dir "${OUT_DIR}" \
      --n_codebook "${M}"
  done
done

echo "REDECODE_END dataset=${CAT} preset=${DS}"
