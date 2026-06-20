#!/bin/bash

# Submit from jobs/reproduction/sasrec_modernized/perf so these relative output paths resolve correctly.
# Partition checked on 2026-06-19: gpu_a100 had 1 idle node; gpu_h100 and gpu_mig had none.
# Prefer gpu_a100 for normal full GPU runs per repo guidance.
#SBATCH --partition=gpu_a100
#SBATCH --job-name=sasrec_mod_perf_cmp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=../../../../output/reproduction/sasrec_modernized/perf/%x-%A_%a.out
#SBATCH --error=../../../../output/reproduction/sasrec_modernized/perf/%x-%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PWD_REAL="$(pwd -P)"
  if [[ "${PWD_REAL}" != "${SCRIPT_DIR}" ]]; then
    echo "ERROR: run this script from ${SCRIPT_DIR}" >&2
    echo "Run:" >&2
    echo "  cd ${SCRIPT_DIR}" >&2
    echo "  sbatch ./compare_best_ablation.sh" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"
OUTPUT_DIR="${REPO_ROOT}/output/reproduction/sasrec_modernized/perf"
ENV_PREFIX="${REPO_ROOT}/artifacts/conda/rpg-uva"
LOCAL_BASELINE_DIR="${REPO_ROOT}/artifacts/sasrec_modernized/ckpt"
SHARED_BASELINE_DIR="/projects/prjs2120/groups/group_16/artifacts/sasrec_modernized/ckpt"

DATASETS=(beauty cds_and_vinyl sports_and_outdoors toys_and_games)
DATA_NAMES=(Beauty CDs_and_Vinyl Sports_and_Outdoors Toys_and_Games)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX < 0 || IDX >= ${#DATASETS[@]} )); then
  echo "ERROR: array index ${IDX} is out of range for ${#DATASETS[@]} datasets." >&2
  exit 2
fi

DATASET_SLUG="${DATASET_SLUG:-${DATASETS[$IDX]}}"
case "${DATASET_SLUG}" in
  beauty)
    PRESET="beauty"
    DATA_NAME="Beauty"
    ;;
  cds_and_vinyl)
    PRESET="cds_and_vinyl"
    DATA_NAME="CDs_and_Vinyl"
    ;;
  sports_and_outdoors)
    PRESET="sports_and_outdoors"
    DATA_NAME="Sports_and_Outdoors"
    ;;
  toys_and_games)
    PRESET="toys_and_games"
    DATA_NAME="Toys_and_Games"
    ;;
  *)
    echo "ERROR: unsupported DATASET_SLUG=${DATASET_SLUG}" >&2
    exit 2
    ;;
esac

BASELINE_CHECKPOINT="${LOCAL_BASELINE_DIR}/sasrec_modernized_${DATASET_SLUG}.pt"
if [[ ! -f "${BASELINE_CHECKPOINT}" ]]; then
  BASELINE_CHECKPOINT="${SHARED_BASELINE_DIR}/sasrec_modernized_${DATASET_SLUG}.pt"
fi
if [[ ! -f "${BASELINE_CHECKPOINT}" ]]; then
  echo "ERROR: baseline checkpoint not found for ${DATASET_SLUG}" >&2
  exit 3
fi

mkdir -p "${OUTPUT_DIR}"
source "${REPO_ROOT}/jobs/lib/runtime_stats.sh"
runtime_stats_init "${OUTPUT_DIR}" "sasrec_modernized_perf_compare_${DATASET_SLUG}"
runtime_stats_start_gpu_monitor
trap runtime_stats_finish EXIT

module purge
module load 2025
module load Anaconda3/2025.06-1

BEST_INFO="$("${ENV_PREFIX}/bin/python" "${REPO_ROOT}/scripts/find_best_sasrec_ablation_checkpoint.py" --dataset "${DATASET_SLUG}" --format shell)"
eval "${BEST_INFO}"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: best ablation checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 4
fi

BASE_PROFILE_ROOT="artifacts/sasrec_modernized/perf/${DATASET_SLUG}/baseline"
BEST_PROFILE_ROOT="artifacts/sasrec_modernized/perf/${DATASET_SLUG}/best_ablation"
DATA_FILE="${REPO_ROOT}/artifacts/sasrec/data/${DATA_NAME}/${DATA_NAME}.txt"

if [[ ! -f "${DATA_FILE}" ]]; then
  echo "ERROR: SASRec data file not found: ${DATA_FILE}" >&2
  exit 5
fi

POOL_SIZES="$("${ENV_PREFIX}/bin/python" - <<PY
from pathlib import Path
import sys

data_file = Path(${DATA_FILE@Q})
base_sizes = [20000, 50000, 100000, 200000, 500000]
max_item = 0
with data_file.open("r", encoding="utf-8") as handle:
    for line in handle:
        parts = line.strip().split(" ", 1)
        if len(parts) != 2:
            continue
        seq = [int(token) for token in parts[1].split() if token]
        if seq:
            max_item = max(max_item, max(seq))
valid_sizes = [size for size in base_sizes if size >= max_item]
if max_item not in valid_sizes:
    valid_sizes.insert(0, max_item)
print(",".join(str(size) for size in valid_sizes))
PY
)"

if [[ -z "${POOL_SIZES}" ]]; then
  echo "ERROR: failed to derive pool sizes for ${DATASET_SLUG}" >&2
  exit 6
fi

cd "${REPO_ROOT}"

baseline_session="$("${ENV_PREFIX}/bin/python" scripts/sasrec_perf.py \
  profile \
  --checkpoint "${BASELINE_CHECKPOINT}" \
  --preset "${PRESET}" \
  --dataset "${DATA_NAME}" \
  --category "${DATA_NAME}" \
  --data_name "${DATA_NAME}" \
  --output-dir "${BASE_PROFILE_ROOT}" \
  --pool-sizes "${POOL_SIZES}" \
  --repeats 3 \
  --warmup_batches 2 \
  --measure_cuda_memory true \
  --dummy_pool_seed 2024 | tail -n 1)"

best_session="$("${ENV_PREFIX}/bin/python" scripts/sasrec_perf.py \
  profile \
  --checkpoint "${CHECKPOINT_PATH}" \
  --preset "${PRESET}" \
  --dataset "${DATA_NAME}" \
  --category "${DATA_NAME}" \
  --data_name "${DATA_NAME}" \
  --output-dir "${BEST_PROFILE_ROOT}" \
  --pool-sizes "${POOL_SIZES}" \
  --repeats 3 \
  --warmup_batches 2 \
  --measure_cuda_memory true \
  --dummy_pool_seed 2024 | tail -n 1)"

COMPARE_OUTPUT_DIR="${REPO_ROOT}/artifacts/sasrec_modernized/perf/${DATASET_SLUG}/comparison/$(date -u +%Y%m%dT%H%M%SZ)_job${SLURM_JOB_ID:-manual}"
mkdir -p "${COMPARE_OUTPUT_DIR}"

"${ENV_PREFIX}/bin/python" scripts/compare_sasrec_perf.py \
  --baseline-session "${baseline_session}" \
  --candidate-session "${best_session}" \
  --baseline-label "baseline" \
  --candidate-label "best_ablation" \
  --output-dir "${COMPARE_OUTPUT_DIR}"

cat > "${COMPARE_OUTPUT_DIR}/checkpoints.json" <<EOF
{
  "dataset_slug": "${DATASET_SLUG}",
  "data_name": "${DATA_NAME}",
  "baseline_checkpoint": "${BASELINE_CHECKPOINT}",
  "best_ablation_checkpoint": "${CHECKPOINT_PATH}",
  "best_ablation_run_id": "${RUN_ID}",
  "best_ablation_test_ndcg10": ${TEST_NDCG10},
  "best_ablation_test_ndcg20": ${TEST_NDCG20},
  "best_ablation_source_log": "${SOURCE_LOG}",
  "pool_sizes": "${POOL_SIZES}",
  "baseline_session": "${baseline_session}",
  "best_ablation_session": "${best_session}"
}
EOF

echo "comparison_output_dir=${COMPARE_OUTPUT_DIR}"
