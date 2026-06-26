#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
DATASET="${DATASET:-sports_and_outdoors}"
MODEL="${MODEL:-both}"
RUN_EVAL_SEEDS="${RUN_EVAL_SEEDS:-1}"
RUN_COLD_START="${RUN_COLD_START:-1}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-}"

case "${DATASET}" in
  beauty|cds_and_vinyl|sports_and_outdoors|toys_and_games) ;;
  *)
    echo "ERROR: unsupported DATASET=${DATASET}" >&2
    exit 2
    ;;
esac

case "${MODEL}" in
  sasrec|rpg|both) ;;
  *)
    echo "ERROR: unsupported MODEL=${MODEL}" >&2
    exit 2
    ;;
esac

account_args=()
if [[ -n "${SBATCH_ACCOUNT}" ]]; then
  account_args=(--account "${SBATCH_ACCOUNT}")
fi

job_name_suffix="${DATASET}"
printf 'Submitting accuracy/fairness jobs for DATASET=%s MODEL=%s\n' "${DATASET}" "${MODEL}"

if [[ "${MODEL}" == "rpg" || "${MODEL}" == "both" ]]; then
  if [[ "${RUN_EVAL_SEEDS}" == "1" ]]; then
    cd "${REPO_ROOT}/jobs/reproduction/eval_seeds/released_readme"
    rpg_eval_job_id="$(sbatch --parsable "${account_args[@]}" "./run_${job_name_suffix}.sh")"
    printf 'rpg_eval_seeds=%s\n' "${rpg_eval_job_id}"
  fi
  if [[ "${RUN_COLD_START}" == "1" ]]; then
    cd "${REPO_ROOT}/jobs/reproduction/cold_start/released_readme"
    rpg_cold_job_id="$(sbatch --parsable "${account_args[@]}" "./run_${job_name_suffix}.sh")"
    printf 'rpg_cold_start=%s\n' "${rpg_cold_job_id}"
  fi
fi

if [[ "${MODEL}" == "sasrec" || "${MODEL}" == "both" ]]; then
  if [[ "${RUN_EVAL_SEEDS}" == "1" ]]; then
    cd "${REPO_ROOT}/jobs/reproduction/sasrec/eval_seeds/released_readme"
    sasrec_eval_job_id="$(sbatch --parsable "${account_args[@]}" "./run_${job_name_suffix}.sh")"
    printf 'sasrec_eval_seeds=%s\n' "${sasrec_eval_job_id}"
  fi
  if [[ "${RUN_COLD_START}" == "1" ]]; then
    cd "${REPO_ROOT}/jobs/reproduction/sasrec/cold_start/released_readme"
    sasrec_cold_job_id="$(sbatch --parsable "${account_args[@]}" "./run_${job_name_suffix}.sh")"
    printf 'sasrec_cold_start=%s\n' "${sasrec_cold_job_id}"
  fi
fi
