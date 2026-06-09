#!/bin/bash

# Submit from jobs/reproduction/sasrec/sports_and_outdoors so these relative output paths resolve correctly.
#SBATCH --partition=gpu_a100
#SBATCH --job-name=sasrec_sports_and_outdoors
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --output=../../../../output/reproduction/sasrec/sports_and_outdoors/%x-%j.out
#SBATCH --error=../../../../output/reproduction/sasrec/sports_and_outdoors/%x-%j.err

JOB_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}" && pwd -P)"
# shellcheck source=../common.sh
source "${JOB_DIR}/../common.sh"

require_job_dir "train.sh"

OUTPUT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)/output/reproduction/sasrec/sports_and_outdoors"
init_sasrec_job "${OUTPUT_DIR}" "sasrec_sports_and_outdoors"
load_sasrec_env

cd "${REPO_ROOT}"

echo "Training SASRec Sports_and_Outdoors reproduction run"
echo "Config: configs/sasrec/root.yaml + configs/sasrec/repro/sports_and_outdoors.yaml"

runtime_stats_run python3 scripts/rpg.py \
  --model SASRec \
  --no-root-config \
  --no-local-config \
  --config configs/sasrec/root.yaml \
  --config configs/sasrec/repro/sports_and_outdoors.yaml
