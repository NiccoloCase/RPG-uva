#!/bin/bash

# Submit from jobs/init/env so these relative output paths resolve correctly.
#SBATCH --partition=cbuild
#SBATCH --job-name=install_rpg_env
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=../../../output/init/env/%x-%j.out
#SBATCH --error=../../../output/init/env/%x-%j.err

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
    echo "  bash ./setup_env.sh" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output/init/env"

mkdir -p "${OUTPUT_DIR}"
source "${REPO_ROOT}/jobs/lib/runtime_stats.sh"
runtime_stats_init "${OUTPUT_DIR}" "install_rpg_env"
trap runtime_stats_finish EXIT

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "${REPO_ROOT}"

if [[ ! -f "${REPO_ROOT}/environment.yml" ]]; then
  echo "ERROR: missing ${REPO_ROOT}/environment.yml" >&2
  exit 3
fi

accept_tos_if_available() {
  local channel_url="$1"
  if ! conda tos accept --override-channels --channel "${channel_url}"; then
    echo "WARNING: could not accept TOS for ${channel_url}; continuing" >&2
  fi
}

accept_tos_if_available "https://repo.anaconda.com/pkgs/main"
accept_tos_if_available "https://repo.anaconda.com/pkgs/r"

if conda env list | awk '$1 == "rpg-uva" {found=1} END {exit !found}'; then
  runtime_stats_run conda env update -n rpg-uva -f environment.yml --prune
else
  runtime_stats_run conda env create -f environment.yml
fi
