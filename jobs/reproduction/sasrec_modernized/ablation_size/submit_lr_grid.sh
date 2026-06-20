#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"
DATASET="${DATASET:-all}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-}"

account_args=()
if [[ -n "${SBATCH_ACCOUNT}" ]]; then
  account_args=(--account "${SBATCH_ACCOUNT}")
fi

mkdir -p \
  "${REPO_ROOT}/output/init/env" \
  "${REPO_ROOT}/output/reproduction/sasrec_modernized/ablation_size/lr_grid"

cd "${REPO_ROOT}/jobs/init/env"
install_job_id="$(sbatch --parsable "${account_args[@]}" ./setup_env.sh)"

declare -A TRAIN_SCRIPTS=(
  [sports_and_outdoors]="train_sports_lr_grid.sh"
  [beauty]="train_beauty_lr_grid.sh"
  [toys_and_games]="train_toys_lr_grid.sh"
  [cds_and_vinyl]="train_cds_lr_grid.sh"
)

submit_one() {
  local dataset_key="$1"
  local script_name="${TRAIN_SCRIPTS[$dataset_key]:-}"
  if [[ -z "${script_name}" ]]; then
    echo "ERROR: unknown dataset key: ${dataset_key}" >&2
    exit 2
  fi

  cd "${SCRIPT_DIR}"
  local train_job_id
  train_job_id="$(sbatch --parsable --dependency "afterok:${install_job_id}" "${account_args[@]}" "./${script_name}")"
  printf '%s=%s\n' "${dataset_key}" "${train_job_id}"
}

printf 'Submitted SASRec modernized size-match LR-grid jobs\n'
printf 'install=%s\n' "${install_job_id}"

case "${DATASET}" in
  all)
    submit_one sports_and_outdoors
    submit_one beauty
    submit_one toys_and_games
    submit_one cds_and_vinyl
    ;;
  sports_and_outdoors|beauty|toys_and_games|cds_and_vinyl)
    submit_one "${DATASET}"
    ;;
  *)
    echo "ERROR: unknown DATASET=${DATASET}. Use all, sports_and_outdoors, beauty, toys_and_games, or cds_and_vinyl." >&2
    exit 2
    ;;
esac
