# SASRec Reproduction Jobs

This document summarizes the SASRec reproduction work added for the RPG paper checks. User-facing names are `sasrec`, but all new analysis code imports the modernized implementation internally:

- `models.sasrec_modernized`
- helpers from `scripts/sasrec_modernized.py`

The new cold-start, performance, and eval-seed paths do not call legacy `scripts/sasrec.py`.

## What Was Added

### Cold-Start Analysis

Script:

```text
scripts/sasrec_cold_start.py
```

Purpose:

- Reproduce the RPG paper cold-start/Figure 5 protocol for SASRec.
- Bucket test target items by training-prefix frequency.
- Use buckets `[0,5]`, `[6,10]`, `[11,15]`, `[16,20]`.
- Count item frequency from each user sequence prefix `items[:-2]`.
- Evaluate the final test target `items[-1]`.
- Report `recall@5`, `ndcg@5`, `recall@10`, `ndcg@10`.
- Plot `ndcg@10`.
- Use full-sort SASRec evaluation with padding, mask token, and seen train/validation items masked.

Artifacts:

```text
tables/cold_start_summary.json
tables/cold_start_summary.csv
figures/ndcg_at_10.png
manifest.json
```

Default output root:

```text
artifacts/sasrec/cold_start/released_readme/<dataset>/
```

### Sports Performance Analysis

Script:

```text
scripts/sasrec_perf.py
```

Purpose:

- Match the RPG paper performance methodology for the SASRec baseline.
- Use the Sports dataset.
- Enlarge the candidate item pool by deterministic dummy item embedding clones.
- Measure full-sort SASRec inference time.
- Measure peak CUDA memory after the already-loaded model/expanded item table baseline.
- Report `NDCG@10` as a sanity metric.

Default pool sizes:

```text
20000, 50000, 100000, 200000, 500000
```

Artifacts:

```text
raw/profile_runs.csv
raw/profile_runs.jsonl
summaries/profile_summary.csv
summaries/profile_summary.jsonl
manifest.json
```

Default output root:

```text
artifacts/sasrec/perf/sports/
```

### Normal and Eval-Seed Evaluation

Script:

```text
scripts/sasrec_eval.py
```

Purpose:

- Add an evaluation command that can run either the standard single-pass SASRec full-sort evaluation or the RPG-style multi-eval-seed protocol.
- Keep artifact schema close to `scripts/rpg_eval_seeds.py`.
- For eval-seed mode, compute the final metric as:

```text
average each user across eval seeds, then average across users
```

CLI modes:

```bash
--eval-mode normal
--eval-mode eval_seeds
```

Default eval seeds:

```text
2024,2025,2026,2027,2028,2029,2030,2031,2032,2033
```

Artifacts:

```text
per_user_metrics.csv
per_user_metrics.jsonl
per_seed_summary.csv
summary.csv
summary.json
manifest.json
```

Default output root:

```text
artifacts/sasrec/eval_seeds/released_readme/<dataset>/
```

Note: SASRec full-sort evaluation is deterministic in `model.eval()`, so eval-seed variance is expected to be zero or near-zero. The multi-seed mode is still useful because it mirrors the RPG aggregation/reporting protocol.

### Modernized Trainer Masking Fix

File:

```text
models/sasrec_modernized/trainer.py
```

The full-sort evaluation path now masks:

- padding row `0`
- the SASRec mask token
- already-seen train/validation items

Masked candidates are set to `-np.inf`, not `0`, so invalid items cannot still rank above low-scoring valid candidates.

### Notebooks

Updated:

```text
notebooks/cold_start_analysis.ipynb
notebooks/perf_sports_analysis.ipynb
```

Added:

```text
notebooks/sasrec_eval_seed_analysis.ipynb
```

Notebook purposes:

- `cold_start_analysis.ipynb`: loads RPG and SASRec cold-start summaries and plots the Sports Figure 5-style `ndcg@10` cold-start view.
- `perf_sports_analysis.ipynb`: loads RPG and SASRec Sports performance artifacts and creates merged runtime/memory/NDCG plots.
- `sasrec_eval_seed_analysis.ipynb`: loads SASRec normal and eval-seed artifacts, then reports `final_user_avg` and seed variance.

## Checkpoints

The new jobs default to modernized SASRec checkpoints:

```text
artifacts/sasrec_modernized/ckpt/sasrec_modernized_<dataset_slug>.pt
```

Examples:

```text
artifacts/sasrec_modernized/ckpt/sasrec_modernized_sports_and_outdoors.pt
artifacts/sasrec_modernized/ckpt/sasrec_modernized_beauty.pt
artifacts/sasrec_modernized/ckpt/sasrec_modernized_toys_and_games.pt
artifacts/sasrec_modernized/ckpt/sasrec_modernized_cds_and_vinyl.pt
```

You can override the checkpoint by passing it as the first argument to a job script or by setting `CHECKPOINT_PATH`.

## Launching Jobs on Snellius

Submit jobs from the job's own directory. Do not submit from the repo root.

### Cold Start

Shared runner:

```text
jobs/reproduction/sasrec/cold_start/run_cold_start.sh
```

Dataset launchers:

```text
jobs/reproduction/sasrec/cold_start/released_readme/run_sports_and_outdoors.sh
jobs/reproduction/sasrec/cold_start/released_readme/run_beauty.sh
jobs/reproduction/sasrec/cold_start/released_readme/run_toys_and_games.sh
jobs/reproduction/sasrec/cold_start/released_readme/run_cds_and_vinyl.sh
```

Launch Sports:

```bash
cd jobs/reproduction/sasrec/cold_start/released_readme
sbatch ./run_sports_and_outdoors.sh
```

Launch all four datasets:

```bash
cd jobs/reproduction/sasrec/cold_start/released_readme
sbatch ./run_sports_and_outdoors.sh
sbatch ./run_beauty.sh
sbatch ./run_toys_and_games.sh
sbatch ./run_cds_and_vinyl.sh
```

Launch with an explicit checkpoint:

```bash
cd jobs/reproduction/sasrec/cold_start/released_readme
sbatch ./run_sports_and_outdoors.sh /gpfs/home6/$USER/RPG/artifacts/sasrec_modernized/ckpt/sasrec_modernized_sports_and_outdoors.pt
```

Cold-start logs:

```text
output/reproduction/sasrec/cold_start/
output/reproduction/sasrec/cold_start/released_readme/
```

### Performance Profiling

Sports-only job:

```text
jobs/reproduction/sasrec/perf/profile_inference.sh
```

Launch:

```bash
cd jobs/reproduction/sasrec/perf
sbatch ./profile_inference.sh
```

Launch with an explicit checkpoint:

```bash
cd jobs/reproduction/sasrec/perf
sbatch ./profile_inference.sh /gpfs/home6/$USER/RPG/artifacts/sasrec_modernized/ckpt/sasrec_modernized_sports_and_outdoors.pt
```

Override pool sizes:

```bash
cd jobs/reproduction/sasrec/perf
sbatch ./profile_inference.sh /gpfs/home6/$USER/RPG/artifacts/sasrec_modernized/ckpt/sasrec_modernized_sports_and_outdoors.pt --pool_sizes "[20000,50000,100000]"
```

Performance logs:

```text
output/reproduction/sasrec/perf/
```

### Normal and Eval-Seed Evaluation

Shared runner:

```text
jobs/reproduction/sasrec/eval_seeds/run_eval.sh
```

Dataset launchers:

```text
jobs/reproduction/sasrec/eval_seeds/released_readme/run_sports_and_outdoors.sh
jobs/reproduction/sasrec/eval_seeds/released_readme/run_beauty.sh
jobs/reproduction/sasrec/eval_seeds/released_readme/run_toys_and_games.sh
jobs/reproduction/sasrec/eval_seeds/released_readme/run_cds_and_vinyl.sh
```

Launch Sports in default 10-seed mode:

```bash
cd jobs/reproduction/sasrec/eval_seeds/released_readme
sbatch ./run_sports_and_outdoors.sh
```

Launch all four datasets in default 10-seed mode:

```bash
cd jobs/reproduction/sasrec/eval_seeds/released_readme
sbatch ./run_sports_and_outdoors.sh
sbatch ./run_beauty.sh
sbatch ./run_toys_and_games.sh
sbatch ./run_cds_and_vinyl.sh
```

Launch normal single-seed mode:

```bash
cd jobs/reproduction/sasrec/eval_seeds/released_readme
SASREC_EVAL_MODE=normal sbatch ./run_sports_and_outdoors.sh
```

Launch normal mode with a specific eval seed:

```bash
cd jobs/reproduction/sasrec/eval_seeds/released_readme
SASREC_EVAL_MODE=normal SASREC_EVAL_SEED=2024 sbatch ./run_sports_and_outdoors.sh
```

Launch eval-seed mode with a custom seed list:

```bash
cd jobs/reproduction/sasrec/eval_seeds/released_readme
SASREC_EVAL_MODE=eval_seeds SASREC_EVAL_SEEDS=2024,2025,2026 sbatch ./run_sports_and_outdoors.sh
```

Launch with an explicit checkpoint:

```bash
cd jobs/reproduction/sasrec/eval_seeds/released_readme
sbatch ./run_sports_and_outdoors.sh /gpfs/home6/$USER/RPG/artifacts/sasrec_modernized/ckpt/sasrec_modernized_sports_and_outdoors.pt
```

Eval logs:

```text
output/reproduction/sasrec/eval_seeds/
output/reproduction/sasrec/eval_seeds/released_readme/
```

## Direct CLI Usage

The Slurm jobs are the preferred reproducible path. For small local smoke tests or debugging, the scripts can also be run directly.

Cold start:

```bash
python scripts/sasrec_cold_start.py run \
  --checkpoint artifacts/sasrec_modernized/ckpt/sasrec_modernized_sports_and_outdoors.pt \
  --preset sports_and_outdoors \
  --dataset Sports_and_Outdoors \
  --output-dir artifacts/sasrec/cold_start/released_readme/sports_and_outdoors
```

Performance:

```bash
python scripts/sasrec_perf.py profile \
  --checkpoint artifacts/sasrec_modernized/ckpt/sasrec_modernized_sports_and_outdoors.pt \
  --preset sports_and_outdoors \
  --dataset Sports_and_Outdoors \
  --config configs/sasrec/perf/sports.yaml
```

Normal eval:

```bash
python scripts/sasrec_eval.py \
  --checkpoint artifacts/sasrec_modernized/ckpt/sasrec_modernized_sports_and_outdoors.pt \
  --eval-mode normal \
  --eval-seed 2024 \
  --preset sports_and_outdoors \
  --dataset Sports_and_Outdoors \
  --config configs/sasrec/eval_seeds/released_readme/sports_and_outdoors.yaml \
  --output-dir artifacts/sasrec/eval_seeds/released_readme/sports_and_outdoors
```

Ten-seed eval:

```bash
python scripts/sasrec_eval.py \
  --checkpoint artifacts/sasrec_modernized/ckpt/sasrec_modernized_sports_and_outdoors.pt \
  --eval-mode eval_seeds \
  --eval-seeds 2024,2025,2026,2027,2028,2029,2030,2031,2032,2033 \
  --preset sports_and_outdoors \
  --dataset Sports_and_Outdoors \
  --config configs/sasrec/eval_seeds/released_readme/sports_and_outdoors.yaml \
  --output-dir artifacts/sasrec/eval_seeds/released_readme/sports_and_outdoors
```

## Validation Performed

The SASRec additions were checked with:

```bash
python3 -m py_compile scripts/sasrec_modernized.py scripts/sasrec_cold_start.py scripts/sasrec_perf.py scripts/sasrec_eval.py
python3 -m py_compile models/sasrec_modernized/trainer.py models/sasrec_modernized/dataset.py models/sasrec_modernized/model.py models/sasrec_modernized/utils.py
find jobs/reproduction/sasrec/cold_start jobs/reproduction/sasrec/perf jobs/reproduction/sasrec/eval_seeds -type f -name '*.sh' -exec bash -n {} \;
python3 -m json.tool notebooks/cold_start_analysis.ipynb
python3 -m json.tool notebooks/perf_sports_analysis.ipynb
python3 -m json.tool notebooks/sasrec_eval_seed_analysis.ipynb
```

Small local smoke tests were also run for:

- cold-start summary/CSV/figure/manifest writing
- performance pool expansion and profile summaries
- normal single-seed eval artifacts
- multi-seed eval artifacts and `final_user_avg` aggregation

No real Snellius jobs were submitted during local validation.

## Important Caveat

The older pre-existing SASRec train/eval jobs under:

```text
jobs/reproduction/sasrec/<dataset>/
```

still call legacy `scripts/sasrec.py`. They were left untouched. For the modernized-backed reproduction analyses described here, use only:

```text
jobs/reproduction/sasrec/cold_start/
jobs/reproduction/sasrec/perf/
jobs/reproduction/sasrec/eval_seeds/
```
