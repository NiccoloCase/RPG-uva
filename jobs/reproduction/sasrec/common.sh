#!/bin/bash

set -euo pipefail

require_job_dir() {
  local script_name="$1"

  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
    return
  fi

  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd -P)"
  local pwd_real
  pwd_real="$(pwd -P)"
  if [[ "${pwd_real}" != "${SCRIPT_DIR}" ]]; then
    echo "ERROR: run this script from ${SCRIPT_DIR}" >&2
    echo "Run:" >&2
    echo "  cd ${SCRIPT_DIR}" >&2
    echo "  bash ./${script_name}" >&2
    exit 2
  fi
}

init_sasrec_job() {
  local output_dir="$1"
  local stats_name="$2"

  REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)"
  mkdir -p "${output_dir}"

  # shellcheck source=/dev/null
  source "${REPO_ROOT}/jobs/lib/runtime_stats.sh"
  runtime_stats_init "${output_dir}" "${stats_name}"
  runtime_stats_start_gpu_monitor
  trap runtime_stats_finish EXIT
}

load_sasrec_env() {
  module purge
  module load 2025
  module load Anaconda3/2025.06-1

  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate rpg-uva
}

resolve_checkpoint_path() {
  local checkpoint_arg="${1:-}"
  local ckpt_dir="$2"
  local run_id="$3"

  if [[ -n "${checkpoint_arg}" ]]; then
    if [[ "${checkpoint_arg}" != /* ]]; then
      echo "ERROR: checkpoint path must be absolute: ${checkpoint_arg}" >&2
      exit 4
    fi
    if [[ ! -f "${checkpoint_arg}" ]]; then
      echo "ERROR: checkpoint not found: ${checkpoint_arg}" >&2
      exit 5
    fi
    printf '%s\n' "${checkpoint_arg}"
    return
  fi

  local discovered
  discovered="$(
    find "${ckpt_dir}" -maxdepth 1 -type f -name "${run_id}-*.pth" -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr \
      | sed -n '1s/^[^ ]* //p'
  )"
  if [[ -z "${discovered}" || ! -f "${discovered}" ]]; then
    echo "ERROR: could not find a checkpoint matching ${run_id}-*.pth in ${ckpt_dir}" >&2
    echo "Pass an absolute checkpoint path as the first argument to override autodiscovery." >&2
    exit 6
  fi
  printf '%s\n' "${discovered}"
}
