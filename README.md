# RPG-uva

This repo keeps the upstream RPG implementation in the `third_party/` submodule and adds a root-level wrapper so you can run experiments from here without editing the submodule.

## Layout

- `third_party/`: upstream `facebookresearch/RPG_KDD2025` submodule. Read-only in this repo.
- `scripts/rpg.py`: root-level runner that imports the submodule code and exposes config overlays.
- `environment.yml`: root-level conda environment mirroring the upstream Python dependencies.
- `configs/rpg/root.yaml`: repo-owned defaults for output paths.
- `configs/rpg/repro/*.yaml`: paper reproduction presets.
- `configs/rpg/local.example.yaml`: template for untracked machine-specific overrides.

## Setup

Initialize the submodule:

```bash
git submodule update --init --recursive
```

Create the conda environment from the repo root:

```bash
conda env create -f environment.yml
conda activate rpg-uva
```

`environment.yml` mirrors the upstream dependency list and keeps the upstream CUDA 12.9 PyTorch index. If you need CPU-only PyTorch or a different CUDA build, change the `torch` lines there instead of touching `third_party/requirements.txt`.

## Root Commands

Show the wrapper help:

```bash
python3 scripts/rpg.py --help
```

Run one of the paper presets from the repo root:

```bash
python3 scripts/rpg.py --preset sports_and_outdoors
python3 scripts/rpg.py --preset beauty
python3 scripts/rpg.py --preset toys_and_games
python3 scripts/rpg.py --preset cds_and_vinyl
```

Run a custom command from the repo root:

```bash
python3 scripts/rpg.py --category Sports_and_Outdoors --lr 0.003 --temperature 0.03
```

Add extra YAML overrides on top of the preset:

```bash
python3 scripts/rpg.py \
  --preset beauty \
  --config path/to/experiment.yaml \
  --run_id beauty_debug
```

## Performance Profiling

The repo also includes a repo-owned profiling layer for reproducing the RPG side of the paper's inference-efficiency study without modifying `third_party/`.

Validate the exact sparse graph on the original Sports pool:

```bash
python3 scripts/rpg_perf.py \
  validate-graph \
  --checkpoint /abs/path/to/checkpoint.pth \
  --config configs/rpg/perf/sports.yaml
```

Prebuild enlarged-pool adjacency caches and then run inference-only profiling:

```bash
python3 scripts/rpg_perf.py \
  profile \
  --checkpoint /abs/path/to/checkpoint.pth \
  --config configs/rpg/perf/sports.yaml \
  --prepare-only

python3 scripts/rpg_perf.py \
  profile \
  --checkpoint /abs/path/to/checkpoint.pth \
  --config configs/rpg/perf/sports.yaml \
  --profile-only
```

More detailed usage, artifact layout, and Snellius job wrappers are documented in [docs/perf_profiling.md](docs/perf_profiling.md).

## Config Overrides

The root wrapper loads configs in this order:

1. `third_party/genrec/default.yaml`
2. `third_party/genrec/datasets/.../config.yaml`
3. `third_party/genrec/models/.../config.yaml`
4. `configs/rpg/root.yaml`
5. `configs/rpg/local.yaml` if present
6. `configs/rpg/repro/*.yaml` when `--preset` is used
7. Any extra `--config path/to/file.yaml`
8. CLI overrides like `--lr 0.003` or `--lr=0.003`

To keep machine-specific settings out of Git, create an untracked local override:

```bash
cp configs/rpg/local.example.yaml configs/rpg/local.yaml
```

Then edit `configs/rpg/local.yaml` with your own settings, for example batch sizes, FAISS threading, or secrets such as `openai_api_key`.
