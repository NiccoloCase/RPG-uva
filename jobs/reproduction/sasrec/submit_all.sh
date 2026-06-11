#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATASET="${DATASET:-beauty}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-}"
TARGET_DIR="${REPO_ROOT}/jobs/reproduction/sasrec/${DATASET}"

if [[ ! -d "${TARGET_DIR}" ]]; then
  echo "ERROR: unknown SASRec dataset directory: ${TARGET_DIR}" >&2
  exit 2
fi

account_args=()
if [[ -n "${SBATCH_ACCOUNT}" ]]; then
  account_args=(--account "${SBATCH_ACCOUNT}")
fi

mkdir -p \
  "${REPO_ROOT}/output/init/env" \
  "${REPO_ROOT}/output/reproduction/sasrec/${DATASET}"

cd "${REPO_ROOT}/jobs/init/env"
install_job_id="$(sbatch --parsable "${account_args[@]}" ./setup_env.sh)"

cd "${TARGET_DIR}"
prepare_job_id="$(sbatch --parsable --dependency "afterok:${install_job_id}" "${account_args[@]}" ./prepare_data.sh)"
train_job_id="$(sbatch --parsable --dependency "afterok:${prepare_job_id}" "${account_args[@]}" ./train.sh)"
eval_job_id="$(sbatch --parsable --dependency "afterok:${train_job_id}" "${account_args[@]}" ./eval.sh)"

printf 'Submitted SASRec jobs for DATASET=%s\n' "${DATASET}"
printf 'install=%s\n' "${install_job_id}"
printf 'prepare=%s\n' "${prepare_job_id}"
printf 'train=%s\n' "${train_job_id}"
printf 'eval=%s\n' "${eval_job_id}"
