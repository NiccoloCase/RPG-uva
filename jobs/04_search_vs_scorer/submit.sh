#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
JOB="${JOB:-decode_grid}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-}"

case "${JOB}" in
  decode_grid) target_script="${REPO_ROOT}/jobs/reproduction/rpg/grid/run_decode_grid.sh" ;;
  decode_confirm) target_script="${REPO_ROOT}/jobs/reproduction/rpg/grid/run_decode_confirm.sh" ;;
  fig6_repro) target_script="${REPO_ROOT}/jobs/reproduction/rpg/grid/run_fig6_repro.sh" ;;
  infer_grid) target_script="${REPO_ROOT}/jobs/reproduction/rpg/grid/run_infer_grid.sh" ;;
  train_grid) target_script="${REPO_ROOT}/jobs/reproduction/rpg/grid/run_train_grid.sh" ;;
  sasrec_lr_grid) target_script="${REPO_ROOT}/jobs/reproduction/sasrec/ablation_size/submit_lr_grid.sh" ;;
  sasrec_lr_depth_grid) target_script="${REPO_ROOT}/jobs/reproduction/sasrec/ablation_size/submit_lr_depth_grid.sh" ;;
  *)
    echo "ERROR: unsupported JOB=${JOB}" >&2
    exit 2
    ;;
esac

account_args=()
if [[ -n "${SBATCH_ACCOUNT}" ]]; then
  account_args=(--account "${SBATCH_ACCOUNT}")
fi

cd "$(dirname "${target_script}")"
job_id="$(sbatch --parsable "${account_args[@]}" "./$(basename "${target_script}")")"
printf 'submitted=%s\n' "${job_id}"
