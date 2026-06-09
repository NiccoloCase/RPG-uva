#!/bin/bash

# Submit from jobs/reproduction/sasrec/toys_and_games so these relative output paths resolve correctly.
#SBATCH --partition=gpu_a100
#SBATCH --job-name=sasrec_toys_and_games_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --output=../../../../output/reproduction/sasrec/toys_and_games/%x-%j.out
#SBATCH --error=../../../../output/reproduction/sasrec/toys_and_games/%x-%j.err

JOB_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}" && pwd -P)"
# shellcheck source=../common.sh
source "${JOB_DIR}/../common.sh"

require_job_dir "eval.sh"

OUTPUT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)/output/reproduction/sasrec/toys_and_games"
init_sasrec_job "${OUTPUT_DIR}" "sasrec_toys_and_games_eval"
load_sasrec_env

CKPT_DIR="${REPO_ROOT}/artifacts/sasrec/ckpt"
CHECKPOINT_PATH="$(resolve_checkpoint_path "${1:-}" "${CKPT_DIR}" "sasrec_repro_toys_and_games")"
EVAL_SEED="${EVAL_SEED:-2024}"

cd "${REPO_ROOT}"

echo "Evaluating SASRec Toys_and_Games checkpoint: ${CHECKPOINT_PATH}"
echo "Eval seed: ${EVAL_SEED}"

runtime_stats_run python3 scripts/rpg_eval.py \
  --model SASRec \
  --checkpoint "${CHECKPOINT_PATH}" \
  --eval-seed "${EVAL_SEED}" \
  --no-root-config \
  --no-local-config \
  --config configs/sasrec/root.yaml \
  --config configs/sasrec/repro/toys_and_games.yaml
