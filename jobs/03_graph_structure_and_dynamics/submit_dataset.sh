#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
DATASET="${DATASET:-sports_and_outdoors}"
ANALYSIS="${ANALYSIS:-static}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-}"

case "${DATASET}" in
  beauty) dataset_token="beauty" ;;
  cds_and_vinyl) dataset_token="cds_and_vinyl" ;;
  sports_and_outdoors) dataset_token="sports" ;;
  toys_and_games) dataset_token="toys_and_games" ;;
  *)
    echo "ERROR: unsupported DATASET=${DATASET}" >&2
    exit 2
    ;;
esac

case "${ANALYSIS}" in
  static|dynamic|scoring|perf_inference) ;;
  frontier_memory|novelty|pruning|pool_rerank|rerank)
    if [[ "${dataset_token}" != "sports" ]]; then
      echo "ERROR: ANALYSIS=${ANALYSIS} is only available for DATASET=sports_and_outdoors" >&2
      exit 2
    fi
    ;;
  *)
    echo "ERROR: unsupported ANALYSIS=${ANALYSIS}" >&2
    exit 2
    ;;
esac

account_args=()
if [[ -n "${SBATCH_ACCOUNT}" ]]; then
  account_args=(--account "${SBATCH_ACCOUNT}")
fi

target_script="${REPO_ROOT}/jobs/reproduction/rpg/graph_analysis/run_${ANALYSIS}_${dataset_token}.sh"
if [[ ! -f "${target_script}" ]]; then
  echo "ERROR: missing target job script: ${target_script}" >&2
  exit 3
fi

cd "$(dirname "${target_script}")"
job_id="$(sbatch --parsable "${account_args[@]}" "./$(basename "${target_script}")")"
printf 'submitted=%s\n' "${job_id}"
