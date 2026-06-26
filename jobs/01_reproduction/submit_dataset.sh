#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
DATASET="${DATASET:-beauty}"
MODEL="${MODEL:-both}"
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

mkdir -p \
  "${REPO_ROOT}/output/init/env" \
  "${REPO_ROOT}/output/reproduction/rpg/${DATASET}" \
  "${REPO_ROOT}/output/reproduction/sasrec/${DATASET}"

cd "${REPO_ROOT}/jobs/init/env"
install_job_id="$(sbatch --parsable "${account_args[@]}" ./setup_env.sh)"
printf 'install=%s\n' "${install_job_id}"

if [[ "${MODEL}" == "sasrec" || "${MODEL}" == "both" ]]; then
  cd "${REPO_ROOT}/jobs/reproduction/sasrec/${DATASET}"
  sasrec_prepare_job_id="$(sbatch --parsable --dependency "afterok:${install_job_id}" "${account_args[@]}" ./prepare_data.sh)"
  sasrec_train_job_id="$(sbatch --parsable --dependency "afterok:${sasrec_prepare_job_id}" "${account_args[@]}" ./train.sh)"
  sasrec_eval_job_id="$(sbatch --parsable --dependency "afterok:${sasrec_train_job_id}" "${account_args[@]}" ./eval.sh)"
  printf 'sasrec_prepare=%s\nsasrec_train=%s\nsasrec_eval=%s\n' \
    "${sasrec_prepare_job_id}" "${sasrec_train_job_id}" "${sasrec_eval_job_id}"
fi

if [[ "${MODEL}" == "rpg" || "${MODEL}" == "both" ]]; then
  cd "${REPO_ROOT}/jobs/reproduction/rpg/${DATASET}"
  rpg_train_job_id="$(sbatch --parsable --dependency "afterok:${install_job_id}" "${account_args[@]}" ./train.sh)"
  rpg_eval_job_id="$(sbatch --parsable --dependency "afterok:${rpg_train_job_id}" "${account_args[@]}" ./eval.sh)"
  printf 'rpg_train=%s\nrpg_eval=%s\n' "${rpg_train_job_id}" "${rpg_eval_job_id}"
fi

printf 'Submitted reproduction pipeline for DATASET=%s MODEL=%s\n' "${DATASET}" "${MODEL}"
