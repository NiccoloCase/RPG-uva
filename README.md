# RPG-uva

This repository packages two things for publication:

- the original RPG implementation, preserved as a read-only dependency in `third_party/`
- the repo-owned reproduction, analysis, and baseline code used to produce the accompanying report

The public baseline name is `SASRec`. The older duplicate SASRec trees were removed so the repo exposes one baseline surface only.

## Repository Structure

- `third_party/`: pinned upstream RPG dependency. Do not edit it here.
- `scripts/`: canonical repo entrypoints. Use `scripts/rpg.py` and `scripts/sasrec.py`.
- `configs/`: repo-owned experiment presets. Use `configs/rpg/` and `configs/sasrec/`.
- `jobs/`: Snellius Slurm jobs. Start with the paper index folders:
  `jobs/01_reproduction/`, `jobs/02_accuracy_and_fairness/`, `jobs/03_graph_structure_and_dynamics/`, `jobs/04_search_vs_scorer/`, `jobs/05_efficiency/`.
- `artifacts/`: checkpoints, caches, and runtime outputs.
- `output/`: scheduler stdout/stderr logs.
- `results/`: collected tables and summaries.

## Setup

Initialize the submodule:

```bash
git submodule update --init --recursive
```

Create the environment from the repo root:

```bash
conda env create -p "$(pwd)/artifacts/conda/rpg-uva" -f environment.yml
conda activate "$(pwd)/artifacts/conda/rpg-uva"
```

On Snellius, prefer the checked-in Slurm jobs. Submit every job from its own job directory.

RPG's semantic-ID tokenizer needs item-content embeddings. If a dataset's embeddings are not already cached under `artifacts/rpg/cache/.../processed/`, the tokenizer regenerates them, which by default calls OpenAI's `text-embedding-3-large`. Before running an RPG job for a new dataset, either:

- copy `configs/rpg/local.example.yaml` to `configs/rpg/local.yaml` and set `openai_api_key`, or
- override `sent_emb_model` to a local `sentence-transformers` encoder instead.

`configs/rpg/local.yaml` is gitignored, so this file is machine-specific and is not checked in.

## Canonical Entry Points

RPG:

```bash
python3 scripts/rpg.py --preset beauty
```

SASRec:

```bash
python3 scripts/sasrec.py --preset beauty --dataset Beauty
```

## Reproduction Commands

All commands below call the repo entrypoints directly from the repo root. On Snellius, equivalent checked-in Slurm jobs still live under `jobs/`. Change datasets as needed.

### 1. Reproduction

Canonical SASRec full run:

```bash
cd /gpfs/home6/$USER/RPG-uva
python3 scripts/sasrec_prepare_data.py --categories Beauty
python3 scripts/sasrec.py --preset beauty --dataset Beauty
python3 scripts/sasrec.py \
  --preset beauty \
  --dataset Beauty \
  --eval-only \
  --checkpoint artifacts/sasrec/ckpt/sasrec_beauty.pt
```

RPG train and eval:

```bash
cd /gpfs/home6/$USER/RPG-uva
python3 scripts/rpg.py --preset beauty

CHECKPOINT_PATH="$(find artifacts/rpg/ckpt -maxdepth 1 -type f -name 'rpg_repro_beauty-*.pth' | sort | tail -n 1)"
python3 scripts/rpg_eval.py \
  --preset beauty \
  --checkpoint "${CHECKPOINT_PATH}" \
  --eval-seed 2024 \
  --num_beams 20 \
  --n_edges 200 \
  --propagation_steps 3
```

### 2. Accuracy And Fairness

SASRec multi-seed evaluation:

```bash
cd /gpfs/home6/$USER/RPG-uva
python3 scripts/sasrec_eval.py \
  --checkpoint artifacts/sasrec/ckpt/sasrec_sports_and_outdoors.pt \
  --eval-mode eval_seeds \
  --eval-seed 2024 \
  --eval-seeds 2024,2025,2026,2027,2028,2029,2030,2031,2032,2033 \
  --preset sports_and_outdoors \
  --dataset Sports_and_Outdoors \
  --config configs/sasrec/eval_seeds/released_readme/sports_and_outdoors.yaml \
  --output-dir artifacts/sasrec/eval_seeds/released_readme/sports_and_outdoors
```

RPG multi-seed evaluation:

```bash
cd /gpfs/home6/$USER/RPG-uva
CHECKPOINT_PATH="$(find artifacts/rpg/ckpt -maxdepth 1 -type f -name 'rpg_repro_sports_and_outdoors-*.pth' | sort | tail -n 1)"
python3 scripts/rpg_eval_seeds.py \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config configs/rpg/eval_seeds/released_readme/sports_and_outdoors.yaml \
  --eval-seeds 2024,2025,2026,2027,2028,2029,2030,2031,2032,2033 \
  --output-dir artifacts/rpg/eval_seeds/released_readme/sports_and_outdoors \
  --cache_dir artifacts/rpg/cache
```

SASRec cold-start analysis:

```bash
cd /gpfs/home6/$USER/RPG-uva
python3 scripts/sasrec_cold_start.py \
  run \
  --checkpoint artifacts/sasrec/ckpt/sasrec_sports_and_outdoors.pt \
  --preset sports_and_outdoors \
  --dataset Sports_and_Outdoors \
  --output-dir artifacts/sasrec/cold_start
```

RPG cold-start analysis:

```bash
cd /gpfs/home6/$USER/RPG-uva
CHECKPOINT_PATH="$(find artifacts/rpg/ckpt -maxdepth 1 -type f -name 'rpg_repro_sports_and_outdoors-*.pth' | sort | tail -n 1)"
python3 scripts/rpg_cold_start.py \
  run \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config configs/rpg/repro/sports_and_outdoors.yaml \
  --output-dir artifacts/rpg/cold_start \
  --cache_dir artifacts/rpg/cache
```

### 3. Graph Structure And Dynamics

Static graph analysis:

```bash
cd /gpfs/home6/$USER/RPG-uva
CHECKPOINT_PATH="$(find artifacts/rpg/ckpt -maxdepth 1 -type f -name 'rpg_repro_sports_and_outdoors-*.pth' | sort | tail -n 1)"
SESSION_DIR="artifacts/rpg/graph_analysis/sports/manual_static"
python3 scripts/rpg_graph_analysis.py \
  prepare-graph \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config configs/rpg/graph_analysis/sports.yaml \
  --session-dir "${SESSION_DIR}"
python3 scripts/rpg_graph_analysis.py \
  static \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config configs/rpg/graph_analysis/sports.yaml \
  --session-dir "${SESSION_DIR}"
```

Dynamic graph analysis:

```bash
cd /gpfs/home6/$USER/RPG-uva
CHECKPOINT_PATH="$(find artifacts/rpg/ckpt -maxdepth 1 -type f -name 'rpg_repro_sports_and_outdoors-*.pth' | sort | tail -n 1)"
SESSION_DIR="artifacts/rpg/graph_analysis/sports/manual_static"
python3 scripts/rpg_graph_analysis.py \
  dynamic \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config configs/rpg/graph_analysis/sports.yaml \
  --config configs/rpg/graph_analysis/sports_dynamic.yaml \
  --session-dir "${SESSION_DIR}"
```

### 4. Search Vs Scorer

RPG decode grid:

```bash
cd /gpfs/home6/$USER/RPG-uva
CHECKPOINT_PATH="$(find artifacts/rpg/ckpt -maxdepth 1 -type f -name 'rpg_sweep_m16_sports_and_outdoors-*.pth' | sort | tail -n 1)"
python3 scripts/rpg_eval_seeds.py \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config configs/rpg/repro/sports_and_outdoors.yaml \
  --split val \
  --eval-seeds 2024,2025,2026 \
  --output-dir output/reproduction/rpg/grid/decode_val/sports_and_outdoors/b20_k200_q3 \
  --n_codebook 16 \
  --num_beams 20 \
  --n_edges 200 \
  --propagation_steps 3 \
  --topk "[5,10]"
```

RPG decode confirmation:

```bash
cd /gpfs/home6/$USER/RPG-uva
CHECKPOINT_PATH="$(find artifacts/rpg/ckpt -maxdepth 1 -type f -name 'rpg_sweep_m16_sports_and_outdoors-*.pth' | sort | tail -n 1)"
python3 scripts/rpg_eval_seeds.py \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config configs/rpg/repro/sports_and_outdoors.yaml \
  --split test \
  --eval-seeds 2024,2025,2026,2027,2028,2029,2030,2031,2032,2033 \
  --output-dir output/reproduction/rpg/grid/decode_test_confirm/sports_and_outdoors/b20_k200_q3 \
  --n_codebook 16 \
  --num_beams 20 \
  --n_edges 200 \
  --propagation_steps 3 \
  --topk "[5,10]" \
  --no-per-user-output
```

SASRec size ablation:

```bash
cd /gpfs/home6/$USER/RPG-uva
python3 scripts/sasrec.py \
  --preset sports_and_outdoors \
  --dataset Sports_and_Outdoors \
  --hidden_size 326 \
  --ckpt_dir artifacts/sasrec/ckpt/ablation_size \
  --run_id sasrec_sports_and_outdoors_size_match
```

SASRec parameter ablation:

```bash
cd /gpfs/home6/$USER/RPG-uva

for lr in 0.001 0.0005 0.0003; do
  python3 scripts/sasrec.py \
    --preset sports_and_outdoors \
    --dataset Sports_and_Outdoors \
    --epochs 300 \
    --lr "${lr}" \
    --hidden_size 326 \
    --ckpt_dir artifacts/sasrec/ckpt/ablation_size/lr_grid \
    --run_id "sasrec_sports_and_outdoors_size_match_e300_lr${lr//./p}"
done

for lr in 0.001 0.0005 0.0003; do
  for layers in 1 2 3; do
    python3 scripts/sasrec.py \
      --preset sports_and_outdoors \
      --dataset Sports_and_Outdoors \
      --epochs 300 \
      --lr "${lr}" \
      --num_hidden_layers "${layers}" \
      --hidden_size 326 \
      --ckpt_dir artifacts/sasrec/ckpt/ablation_size/lr_depth_grid \
      --run_id "sasrec_sports_and_outdoors_size_match_e300_lr${lr//./p}_L${layers}"
  done
done
```

This reproduces the size-matched SASRec sweeps used for the Search vs Scorer comparison without Slurm: the first loop is the 3-point learning-rate sweep, and the second loop is the 3x3 learning-rate-by-depth sweep (`num_hidden_layers` in `{1,2,3}`). For the other paper datasets, keep the same pattern and switch to the matching preset/dataset/hidden size: `beauty`/`Beauty`/`540`, `toys_and_games`/`Toys_and_Games`/`396`, or `cds_and_vinyl`/`CDs_and_Vinyl`/`328`. Checkpoints are written under `artifacts/sasrec/ckpt/ablation_size/`.

### 5. Efficiency

RPG inference profiling:

```bash
cd /gpfs/home6/$USER/RPG-uva
CHECKPOINT_PATH="$(find artifacts/rpg/ckpt -maxdepth 1 -type f -name 'rpg_repro_sports_and_outdoors-*.pth' | sort | tail -n 1)"
python3 scripts/rpg_perf.py \
  profile \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config configs/rpg/perf/sports.yaml \
  --prepare-only
python3 scripts/rpg_perf.py \
  profile \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config configs/rpg/perf/sports.yaml \
  --profile-only
```

## New-Dataset Extension

The paper datasets live under `jobs/reproduction/`. Extra datasets such as `video_games` and `pet_supplies` live under `jobs/new_datasets/`.

Use:

```bash
cd /gpfs/home6/$USER/RPG-uva/jobs/new_datasets
```

Then follow `jobs/new_datasets/README.md`.

## Notes

- `third_party/` stays in the public repository as the preserved original RPG source boundary.
- The paper-facing job index lives in `jobs/01_reproduction/` through `jobs/05_efficiency/`.
- The canonical SASRec artifacts now live under `artifacts/sasrec/` and `output/reproduction/sasrec/`.
