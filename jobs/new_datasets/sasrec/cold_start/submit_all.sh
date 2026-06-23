#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

if [[ "$(pwd -P)" != "${SCRIPT_DIR}" ]]; then
  echo "ERROR: run this script from ${SCRIPT_DIR}" >&2
  exit 2
fi

for dataset in video_games pet_supplies; do
  echo "Submitting SASRec best-checkpoint retrain for ${dataset}"
  sbatch --export=ALL,COLD_START_DATASET_SLUG="${dataset}" ./retrain_best.sh
done

echo "After the retrain jobs finish, submit cold-start evaluation with:"
echo "  cd ${SCRIPT_DIR}"
echo "  for dataset in video_games pet_supplies; do sbatch --export=ALL,COLD_START_DATASET_SLUG=\"\${dataset}\" ./run_cold_start.sh; done"
