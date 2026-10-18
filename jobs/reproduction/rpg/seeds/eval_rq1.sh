#!/bin/bash

# Submit from jobs/reproduction/rpg/seeds so these relative output paths resolve correctly.
#SBATCH --job-name=rq1_eval
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=../../../../output/reproduction/rpg/seeds/%x-%j.out
#SBATCH --error=../../../../output/reproduction/rpg/seeds/%x-%j.err

# Remaining RQ1 evaluations on the checkpoints of every dataset and seed:
#   1. RPG with the decoding settings of the paper appendix (eval seed 2024),
#   2. cold start: test NDCG@10 by training frequency of the target item, for
#      RPG, SASRec and parameter-matched SASRec, past items kept.
#   cd jobs/reproduction/rpg/seeds && sbatch ./eval_rq1.sh
# Results: artifacts/rpg/eval_seeds/seeds_appendix/<dataset>/s<seed>/<session>/summary.json
#          artifacts/rpg/cold_start/<dataset>_s<seed>/, artifacts/sasrec/cold_start/<dataset>_<variant>_s<seed>/

set -uo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ "$(pwd -P)" != "${SCRIPT_DIR}" ]]; then
    echo "ERROR: run this script from ${SCRIPT_DIR}" >&2
    exit 2
  fi
fi

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output/reproduction/rpg/seeds"
source "${REPO_ROOT}/jobs/lib/env.sh"
repo_env_activate "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"
source "${REPO_ROOT}/jobs/lib/runtime_stats.sh"
runtime_stats_init "${OUTPUT_DIR}" "rq1_eval"
runtime_stats_start_gpu_monitor
trap runtime_stats_finish EXIT

cd "${REPO_ROOT}"
newest() { ls -t $@ 2>/dev/null | head -n 1; }
declare -A HIDDEN=([sports_and_outdoors]=326 [beauty]=540 [toys_and_games]=396 [cds_and_vinyl]=328)
declare -A LAYERS=([sports_and_outdoors]=2 [beauty]=3 [toys_and_games]=2 [cds_and_vinyl]=2)
declare -A CATEGORY=([sports_and_outdoors]=Sports_and_Outdoors [beauty]=Beauty [toys_and_games]=Toys_and_Games [cds_and_vinyl]=CDs_and_Vinyl)
for dataset in sports_and_outdoors beauty toys_and_games cds_and_vinyl; do
  for seed in 2024 2025 2026; do
    if [[ "${seed}" == 2024 ]]; then
      checkpoint="$(newest "artifacts/rpg/ckpt/rpg_loss_token_ce_${dataset}-*.pth")"
    else
      checkpoint="$(newest "artifacts/rpg/ckpt/rpg_loss_token_ce_${dataset}_s${seed}-*.pth" "artifacts/rpg/ckpt/rpg_repro_${dataset}_s${seed}-*.pth")"
    fi
    [[ -z "${checkpoint}" ]] && { runtime_stats_log "SKIP ${dataset} seed ${seed}"; continue; }
    runtime_stats_run python scripts/rpg/eval_seeds.py --checkpoint "${checkpoint}" \
      --config "configs/rpg/repro/${dataset}.yaml" --config "configs/rpg/eval_seeds/paper_appendix/${dataset}.yaml" \
      --no-local-config --decoding graph --graph-backend exact_gpu --eval-seeds 2024 \
      --output-dir "artifacts/rpg/eval_seeds/seeds_appendix/${dataset}/s${seed}" --no-per-user-output
    runtime_stats_run python scripts/rpg/cold_start.py run --checkpoint "${checkpoint}" \
      --config "configs/rpg/repro/${dataset}.yaml" --no-local-config \
      --output-dir "artifacts/rpg/cold_start/${dataset}_s${seed}"
    for variant in h64 pm; do
      if [[ "${variant}" == h64 ]]; then hidden=64; layers=2; else hidden=${HIDDEN[$dataset]}; layers=${LAYERS[$dataset]}; fi
      ckpt="artifacts/sasrec/ckpt/seeds/sasrec_${dataset}_h${hidden}_L${layers}_e300_lr0p0003_s${seed}.pt"
      [[ -f "${ckpt}" ]] || { runtime_stats_log "SKIP SASRec ${variant} ${dataset} ${seed}"; continue; }
      runtime_stats_run python scripts/sasrec/cold_start.py run --checkpoint "${ckpt}" --preset "${dataset}" \
        --dataset "${CATEGORY[$dataset]}" --hidden_size "${hidden}" --num_hidden_layers "${layers}" --mask_history false \
        --output-dir "artifacts/sasrec/cold_start/${dataset}_${variant}_s${seed}"
    done
  done
done
echo "RQ1_EVAL_DONE"
