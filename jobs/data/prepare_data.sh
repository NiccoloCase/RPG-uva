#!/bin/bash

# Submit from jobs/data so these relative output paths resolve correctly.
#SBATCH --job-name=prepare_data
#SBATCH --partition=staging
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=../../output/data/%x-%j.out
#SBATCH --error=../../output/data/%x-%j.err

# Download and preprocess every dataset once, before GPU jobs read them in
# parallel: the Amazon 2014 reviews and metadata in the shared genrec cache
# (artifacts/rpg/cache), the SASRec text files (artifacts/sasrec/data) and the
# sentence-T5 model used by TIGER. Then report, per dataset, how often the test
# target already appears in the user's own history. CPU only. Run with
#   cd jobs/data && sbatch ./prepare_data.sh      (or: bash ./prepare_data.sh)
# Env vars:
#   CATEGORIES  space-separated Amazon categories (default: the five used in the paper)

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ "$(pwd -P)" != "${SCRIPT_DIR}" ]]; then
    echo "ERROR: run this script from ${SCRIPT_DIR}" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output/data"
CATEGORIES="${CATEGORIES:-Sports_and_Outdoors Beauty Toys_and_Games CDs_and_Vinyl Books}"
source "${REPO_ROOT}/jobs/lib/env.sh"
repo_env_activate "${REPO_ROOT}"

mkdir -p "${OUTPUT_DIR}"
source "${REPO_ROOT}/jobs/lib/runtime_stats.sh"
runtime_stats_init "${OUTPUT_DIR}" "prepare_data"
trap runtime_stats_finish EXIT

cd "${REPO_ROOT}"
export CATEGORIES

runtime_stats_run python - <<'PY'
import os, sys
sys.path.insert(0, "third_party/MemGen-GR")
from accelerate import Accelerator
from genrec.utils import get_config, get_dataset
for category in os.environ["CATEGORIES"].split():
    config = get_config("SASRec", "AmazonReviews2014", None,
                        {"category": category, "cache_dir": os.path.abspath("artifacts/rpg/cache"), "metadata": "sentence"})
    config["accelerator"] = Accelerator()
    dataset = get_dataset("AmazonReviews2014")(config)
    print(f"[prepare] {category}: {dataset.n_items - 1} items, {len(dataset.all_item_seqs)} users", flush=True)
PY

# shellcheck disable=SC2086
python scripts/sasrec/prepare_data.py --categories ${CATEGORIES}
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/sentence-t5-base'); print('[prepare] sentence-t5-base cached')"

python - <<'PY'
import json, os
for category in os.environ["CATEGORIES"].split():
    seqs = json.load(open(f"artifacts/rpg/cache/AmazonReviews2014/{category}/processed/all_item_seqs.json"))
    repeats = sum(1 for s in seqs.values() if s[-1] in s[:-1])
    print(f"[repeat-check] {category}: test target in own history for {repeats} of {len(seqs)} users ({100 * repeats / len(seqs):.3f}%)")
PY
echo "PREPARE_DONE"
