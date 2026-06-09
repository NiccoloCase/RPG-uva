#!/bin/bash

# Submit from jobs/reproduction/sasrec/cds_and_vinyl so these relative output paths resolve correctly.
#SBATCH --partition=gpu_a100
#SBATCH --job-name=sasrec_cds_and_vinyl_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --output=../../../../output/reproduction/sasrec/cds_and_vinyl/%x-%j.out
#SBATCH --error=../../../../output/reproduction/sasrec/cds_and_vinyl/%x-%j.err

JOB_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}" && pwd -P)"
# shellcheck source=../common.sh
source "${JOB_DIR}/../common.sh"

require_job_dir "eval.sh"

OUTPUT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)/output/reproduction/sasrec/cds_and_vinyl"
init_sasrec_job "${OUTPUT_DIR}" "sasrec_cds_and_vinyl_eval"
load_sasrec_env

CKPT_DIR="${REPO_ROOT}/artifacts/sasrec/ckpt"
CHECKPOINT_PATH="$(resolve_checkpoint_path "${1:-}" "${CKPT_DIR}" "sasrec_repro_cds_and_vinyl")"
EVAL_SEED="${EVAL_SEED:-2024}"

cd "${REPO_ROOT}"

echo "Evaluating SASRec CDs_and_Vinyl checkpoint: ${CHECKPOINT_PATH}"
echo "Eval seed: ${EVAL_SEED}"

runtime_stats_run python3 scripts/rpg_eval.py \
  --model SASRec \
  --checkpoint "${CHECKPOINT_PATH}" \
  --eval-seed "${EVAL_SEED}" \
  --no-root-config \
  --no-local-config \
  --config configs/sasrec/root.yaml \
  --config configs/sasrec/repro/cds_and_vinyl.yaml
