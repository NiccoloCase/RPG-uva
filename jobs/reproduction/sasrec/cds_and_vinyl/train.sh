#!/bin/bash

# Submit from jobs/reproduction/sasrec/cds_and_vinyl so these relative output paths resolve correctly.
#SBATCH --partition=gpu_a100
#SBATCH --job-name=sasrec_cds_and_vinyl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --output=../../../../output/reproduction/sasrec/cds_and_vinyl/%x-%j.out
#SBATCH --error=../../../../output/reproduction/sasrec/cds_and_vinyl/%x-%j.err

JOB_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}" && pwd -P)"
# shellcheck source=../common.sh
source "${JOB_DIR}/../common.sh"

require_job_dir "train.sh"

OUTPUT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)/output/reproduction/sasrec/cds_and_vinyl"
init_sasrec_job "${OUTPUT_DIR}" "sasrec_cds_and_vinyl"
load_sasrec_env

cd "${REPO_ROOT}"

echo "Training SASRec CDs_and_Vinyl reproduction run"
echo "Config: configs/sasrec/root.yaml + configs/sasrec/repro/cds_and_vinyl.yaml"

runtime_stats_run python3 scripts/rpg.py \
  --model SASRec \
  --no-root-config \
  --no-local-config \
  --config configs/sasrec/root.yaml \
  --config configs/sasrec/repro/cds_and_vinyl.yaml
