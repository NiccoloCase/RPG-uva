#!/bin/bash

# Project environment for job scripts. Source this file, then call
# repo_env_activate "${REPO_ROOT}".
#
# The interpreter is PYTHON_BIN if set, otherwise the repository virtual
# environment at artifacts/venv (see README.md, "Setup"). The function puts its
# bin directory first on PATH, so `python` and `python3` in the job scripts
# resolve to it. On a cluster, load your modules or activate your environment
# before submitting, or set PYTHON_BIN in the submission environment.

repo_env_activate() {
  local repo_root="$1"
  PYTHON_BIN="${PYTHON_BIN:-${repo_root}/artifacts/venv/bin/python}"
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Python interpreter not found: ${PYTHON_BIN}" >&2
    echo "Create the environment (README.md, Setup) or set PYTHON_BIN." >&2
    return 1
  fi
  export PYTHON_BIN
  export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
  export ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-${repo_root}/artifacts}"
}
