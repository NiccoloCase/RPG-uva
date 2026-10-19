#!/bin/bash
# Build and cache the decoding graphs for the enlarged Sports item pools (20k to 500k items with
# dummy items), before profile_inference.sh. Settings: configs/rpg/perf/sports.yaml.
#   cd jobs/reproduction/rpg/perf && sbatch ./build_graphs.sh [checkpoint]

# Submit from jobs/reproduction/rpg/perf so these relative output paths resolve correctly.
#SBATCH --job-name=rpg_perf_graphs
#SBATCH --partition=genoa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=../../../../output/reproduction/rpg/perf/%x-%j.out
#SBATCH --error=../../../../output/reproduction/rpg/perf/%x-%j.err

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
    echo "  bash ./build_graphs.sh $(cd "${SCRIPT_DIR}/../../../.." && pwd)/artifacts/rpg/ckpt/model.pth" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output/reproduction/rpg/perf"
source "${REPO_ROOT}/jobs/lib/env.sh"
repo_env_activate "${REPO_ROOT}"
CHECKPOINT_DIR="${REPO_ROOT}/artifacts/rpg/ckpt"
PERF_CONFIG_DEFAULT="${REPO_ROOT}/configs/rpg/perf/sports.yaml"
CHECKPOINT_PATH="${1:-${CHECKPOINT_PATH:-}}"
PERF_CONFIG="${PERF_CONFIG:-${PERF_CONFIG_DEFAULT}}"
PERF_DATASET_SLUG="${PERF_DATASET_SLUG:-sports_and_outdoors}"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="$(find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name "rpg_repro_${PERF_DATASET_SLUG}-*.pth" | sort | tail -n 1)"
fi

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: provide the checkpoint path as the first argument or CHECKPOINT_PATH env var." >&2
  echo "If omitted, the script tries to resolve the latest dataset checkpoint from:" >&2
  echo "  ${CHECKPOINT_DIR}" >&2
  exit 3
fi

if [[ "${CHECKPOINT_PATH}" == *"<"* || "${CHECKPOINT_PATH}" == *">"* ]]; then
  echo "ERROR: checkpoint path contains angle-bracket placeholders: ${CHECKPOINT_PATH}" >&2
  echo "Use a real absolute path under ${REPO_ROOT}, for example:" >&2
  echo "  ${CHECKPOINT_DIR}/rpg_repro_${PERF_DATASET_SLUG}-<timestamp>.pth" >&2
  exit 4
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: checkpoint file not found: ${CHECKPOINT_PATH}" >&2
  exit 5
fi

mkdir -p "${OUTPUT_DIR}"

cd "${REPO_ROOT}"

python scripts/rpg/perf.py \
  profile \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config "${PERF_CONFIG}" \
  --prepare-only
