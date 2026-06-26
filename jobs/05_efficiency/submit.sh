#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
JOB="${JOB:-profile_inference}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

case "${JOB}" in
  build_graphs) target_script="${REPO_ROOT}/jobs/reproduction/rpg/perf/build_graphs.sh" ;;
  profile_inference) target_script="${REPO_ROOT}/jobs/reproduction/rpg/perf/profile_inference.sh" ;;
  profile_bruteforce) target_script="${REPO_ROOT}/jobs/reproduction/rpg/perf/profile_bruteforce.sh" ;;
  profile_bruteforce_chunk_sweep) target_script="${REPO_ROOT}/jobs/reproduction/rpg/perf/profile_bruteforce_chunk_sweep.sh" ;;
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
if [[ -n "${CHECKPOINT_PATH}" ]]; then
  job_id="$(sbatch --parsable "${account_args[@]}" "./$(basename "${target_script}")" "${CHECKPOINT_PATH}")"
else
  job_id="$(sbatch --parsable "${account_args[@]}" "./$(basename "${target_script}")")"
fi
printf 'submitted=%s\n' "${job_id}"
