#!/bin/bash

# Submit from jobs/reproduction/rpg/decoding_variants so these relative output paths resolve correctly.
#SBATCH --job-name=rpg_graph_checks
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=../../../../output/reproduction/rpg/decoding_variants/%x-%j.out
#SBATCH --error=../../../../output/reproduction/rpg/decoding_variants/%x-%j.err

# RQ3 graph checks on the seed-2024 RPG checkpoints of the four paper datasets:
# reachability of the exhaustive top-10 hits (scripts/rpg/verify_decoding.py)
# and neighbour overlap between the exact and HNSW graphs
# (scripts/results/graph_overlap.py).
#   cd jobs/reproduction/rpg/decoding_variants && sbatch ./run_graph_checks.sh

set -uo pipefail
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
source "${REPO_ROOT}/jobs/lib/env.sh"
repo_env_activate "${REPO_ROOT}"
cd "${REPO_ROOT}"
for pair in sports:sports_and_outdoors beauty:beauty toys:toys_and_games cds:cds_and_vinyl; do
  name="${pair%%:*}"; dataset="${pair##*:}"
  checkpoint="$(ls -t artifacts/rpg/ckpt/rpg_loss_token_ce_${dataset}-*.pth | head -n 1)"
  [[ -f "artifacts/rpg/verification/${name}.json" ]] || \
    python scripts/rpg/verify_decoding.py --name "${name}" --checkpoint "${checkpoint}" --config "configs/rpg/repro/${dataset}.yaml"
  python scripts/results/graph_overlap.py --name "${name}" --checkpoint "${checkpoint}" --config "configs/rpg/repro/${dataset}.yaml"
done
echo "GRAPH_CHECKS_DONE"
