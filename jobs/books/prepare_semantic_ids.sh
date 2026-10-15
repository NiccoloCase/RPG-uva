#!/bin/bash

# Build the Books cache: process the raw reviews and metadata, embed the item
# text with OpenAI text-embedding-3-large, and run OPQ to get semantic IDs.
# CPU only. Runs directly on a shared server (no Slurm). Run from this folder:
#
#   cd jobs/books
#   bash ./prepare_semantic_ids.sh
#
# Needs `openai_api_key` in configs/rpg/local.yaml unless the embeddings file
# already exists in the processed cache. Pass --n_codebook 32 for 32-digit IDs.
# TOKENS_PER_MINUTE paces the API requests (default 900000).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(pwd -P)" != "${SCRIPT_DIR}" ]]; then
  echo "ERROR: run this script from ${SCRIPT_DIR}" >&2
  exit 2
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/jobs/lib/local_gpu.sh"
PYTHON_BIN="$(local_gpu_python "${REPO_ROOT}")"

OUTPUT_DIR="${REPO_ROOT}/output/new_datasets/rpg/books"
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/rpg_prepare_books-local-$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

# Semantic-ID preparation does not need a GPU; keep it off the shared cards.
export CUDA_VISIBLE_DEVICES=""

cd "${REPO_ROOT}"
# Embeddings first, with the resumable script: every request is saved as it
# returns, so an interruption never loses paid work. It is a no-op when the
# embeddings file already exists.
"${PYTHON_BIN}" scripts/rpg_embed_items.py --preset books --tokens-per-minute "${TOKENS_PER_MINUTE:-900000}"
"${PYTHON_BIN}" scripts/rpg_prepare_semantic_ids.py --preset books "$@"
