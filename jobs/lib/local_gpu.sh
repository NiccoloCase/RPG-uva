#!/bin/bash

# Helpers for jobs that run directly on a shared GPU server, without Slurm.
# Source this file; it defines functions only.

# Print the Python interpreter to use: PYTHON_BIN if set, else the repo venv.
local_gpu_python() {
  local repo_root="$1"
  local python_bin="${PYTHON_BIN:-${repo_root}/artifacts/venv/bin/python}"
  if [[ ! -x "${python_bin}" ]]; then
    echo "ERROR: Python interpreter not found: ${python_bin}" >&2
    echo "Create the environment or set PYTHON_BIN." >&2
    return 1
  fi
  echo "${python_bin}"
}

# Refuse to start on a GPU that somebody else is already using.
# Usage: local_gpu_require_idle <gpu_index> [max_used_mib]
local_gpu_require_idle() {
  local gpu_index="$1"
  local max_used_mib="${2:-${GPU_MAX_USED_MIB:-500}}"

  if ! [[ "${gpu_index}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: GPU must be a single GPU index, got '${gpu_index}'." >&2
    return 1
  fi

  local used_mib
  used_mib="$(nvidia-smi --id="${gpu_index}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
  if ! [[ "${used_mib}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: could not read memory usage of GPU ${gpu_index}." >&2
    return 1
  fi

  if (( used_mib > max_used_mib )); then
    echo "ERROR: GPU ${gpu_index} is busy (${used_mib} MiB in use, limit ${max_used_mib} MiB)." >&2
    echo "Pick another GPU, wait, or raise GPU_MAX_USED_MIB if this is intended." >&2
    return 1
  fi
  echo "GPU ${gpu_index} is idle (${used_mib} MiB in use)."
}

# Map a dataset slug to its RPG preset config and Amazon category name.
# Sets DATASET_CONFIG and DATASET_CATEGORY.
local_gpu_resolve_dataset() {
  local repo_root="$1"
  local dataset_slug="$2"
  case "${dataset_slug}" in
    sports_and_outdoors) DATASET_CATEGORY="Sports_and_Outdoors"; DATASET_CONFIG="${repo_root}/configs/rpg/repro/${dataset_slug}.yaml" ;;
    beauty)              DATASET_CATEGORY="Beauty";              DATASET_CONFIG="${repo_root}/configs/rpg/repro/${dataset_slug}.yaml" ;;
    toys_and_games)      DATASET_CATEGORY="Toys_and_Games";      DATASET_CONFIG="${repo_root}/configs/rpg/repro/${dataset_slug}.yaml" ;;
    cds_and_vinyl)       DATASET_CATEGORY="CDs_and_Vinyl";       DATASET_CONFIG="${repo_root}/configs/rpg/repro/${dataset_slug}.yaml" ;;
    video_games)         DATASET_CATEGORY="Video_Games";         DATASET_CONFIG="${repo_root}/configs/rpg/new_datasets/${dataset_slug}.yaml" ;;
    pet_supplies)        DATASET_CATEGORY="Pet_Supplies";        DATASET_CONFIG="${repo_root}/configs/rpg/new_datasets/${dataset_slug}.yaml" ;;
    books)               DATASET_CATEGORY="Books";               DATASET_CONFIG="${repo_root}/configs/rpg/new_datasets/${dataset_slug}.yaml" ;;
    *)
      echo "ERROR: unknown DATASET '${dataset_slug}'." >&2
      return 1
      ;;
  esac
}

# Fail early if the processed cache has neither semantic IDs nor embeddings,
# because the tokenizer would otherwise try to call the OpenAI API.
local_gpu_require_item_cache() {
  local repo_root="$1"
  local category="$2"
  local processed_dir="${repo_root}/artifacts/rpg/cache/AmazonReviews2014/${category}/processed"
  if ! compgen -G "${processed_dir}/*.sem_ids" >/dev/null && ! compgen -G "${processed_dir}/*.sent_emb" >/dev/null; then
    echo "ERROR: no semantic IDs or sentence embeddings under ${processed_dir}" >&2
    echo "Copy the processed cache there, or run the semantic-ID preparation first." >&2
    return 1
  fi
}
