#!/bin/bash
#SBATCH --job-name=sasrec_mod_cold_start_cds_and_vinyl
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=../../../../output/reproduction/sasrec/cold_start/%x-%j_param_matched.out
#SBATCH --error=../../../../output/reproduction/sasrec/cold_start/%x-%j_param_matched.err

set -euo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  SCRIPT_DIR="${RUNNER_DIR}"
  if [[ "$(pwd -P)" != "${SCRIPT_DIR}" ]]; then
    echo "ERROR: run this script from ${SCRIPT_DIR}" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output/reproduction/sasrec/cold_start"
ENV_PREFIX="${REPO_ROOT}/artifacts/conda/rpg-uva"
COLD_START_DATASET_SLUG="cds_and_vinyl"
COLD_START_OUTPUT_DIR="${REPO_ROOT}/artifacts/sasrec/cold_start/reproduction/${COLD_START_DATASET_SLUG}_param_matched"

module purge
module load 2025
module load Anaconda3/2025.06-1

BEST_INFO="$("${ENV_PREFIX}/bin/python" "${REPO_ROOT}/scripts/find_best_sasrec_ablation_checkpoint.py" --dataset "${COLD_START_DATASET_SLUG}" --format shell)"
eval "${BEST_INFO}"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: SASRec checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 3
fi

# Resolve num_hidden_layers from checkpoint metadata if not set by BEST_INFO
if [[ -z "${NUM_HIDDEN_LAYERS:-}" ]]; then
  NUM_HIDDEN_LAYERS="$("${ENV_PREFIX}/bin/python" - <<EOF
import torch, sys
ckpt = torch.load("${CHECKPOINT_PATH}", map_location="cpu")
sd = ckpt.get("model_state_dict", ckpt)
max_layer = max(
    int(k.split(".")[2]) for k in sd if k.startswith("item_encoder.layer.")
)
print(max_layer + 1)
EOF
  )"
  echo "INFO: Inferred NUM_HIDDEN_LAYERS=${NUM_HIDDEN_LAYERS} from checkpoint" >&2
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${COLD_START_OUTPUT_DIR}"

cd "${REPO_ROOT}"

"${ENV_PREFIX}/bin/python" scripts/sasrec_cold_start.py \
  run \
  --checkpoint "${CHECKPOINT_PATH}" \
  --model-family sasrec \
  --preset "${COLD_START_DATASET_SLUG}" \
  --output-dir "${COLD_START_OUTPUT_DIR}" \
  --plot-title "SASRec Cold-Start Analysis (${COLD_START_DATASET_SLUG})" \
  --hidden_size 328 \
  --num_hidden_layers "${NUM_HIDDEN_LAYERS}"
