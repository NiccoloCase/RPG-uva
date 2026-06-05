# Performance Profiling for RPG

This repo ships a repo-owned profiling layer for reproducing the RPG side of the paper's inference-efficiency study without modifying `third_party/`.

## What it measures

- enlarged candidate-pool inference with deterministic dummy items
- offline sparse decoding-graph construction
- inference-only epoch time
- peak CUDA allocated and reserved memory
- average visited items and visited-item ratio

The implementation keeps the original test user histories and ground-truth labels unchanged. Dummy items are added only to the candidate universe used at inference time.

## Preconditions

- The profiling harness rebuilds the dataset and tokenizer around the checkpoint.
- If semantic IDs for the target category are not already cached under `artifacts/rpg/cache/.../processed/`,
  the upstream tokenizer may need to regenerate them.
- With the checked-in defaults, that regeneration path uses `text-embedding-3-large`, so you need either:
  - an existing semantic-ID cache for the dataset/category, or
  - `openai_api_key` in `configs/rpg/local.yaml`, or
  - a config override that switches `sent_emb_model` to a local `sentence-transformers` encoder.

## Main workflow

1. Train RPG normally and keep the checkpoint path.
2. Validate the exact sparse graph builder on the original Sports pool.
3. Prebuild adjacency caches offline for the enlarged pools.
4. Run GPU profiling against the cached graphs.
5. Plot the median summary CSV.

## Config

The checked-in preset for the Sports performance workflow is:

`configs/rpg/perf/sports.yaml`

It uses:

- `category: Sports_and_Outdoors`
- `num_beams: 10`
- `n_edges: 100`
- `propagation_steps: 2`
- pool sizes `20k, 50k, 100k, 200k, 500k`
- `graph_backend: hnsw`
- deterministic dummy duplication with `dummy_pool_seed: 2024`

## Commands

Validate the repo-owned exact sparse graph against the upstream dense graph:

```bash
conda run -n rpg-uva python scripts/rpg_perf.py \
  validate-graph \
  --checkpoint /abs/path/to/checkpoint.pth \
  --config configs/rpg/perf/sports.yaml
```

Prebuild adjacency caches only:

```bash
conda run -n rpg-uva python scripts/rpg_perf.py \
  profile \
  --checkpoint /abs/path/to/checkpoint.pth \
  --config configs/rpg/perf/sports.yaml \
  --prepare-only
```

Run inference-only profiling with cached graphs:

```bash
conda run -n rpg-uva python scripts/rpg_perf.py \
  profile \
  --checkpoint /abs/path/to/checkpoint.pth \
  --config configs/rpg/perf/sports.yaml \
  --profile-only
```

Render the summary plot from a profiling session directory:

```bash
conda run -n rpg-uva python scripts/rpg_perf.py \
  plot \
  --input artifacts/rpg/perf/sports/<session-id> \
  --output artifacts/rpg/perf/sports/<session-id>/plots/perf_rpg.png
```

## Output layout

Each profiling command creates a session directory under:

`artifacts/rpg/perf/sports/`

Each session contains:

- `raw/profile_runs.csv`
- `raw/profile_runs.jsonl`
- `summaries/profile_summary.csv`
- `summaries/profile_summary.jsonl`
- `graphs/graph_builds.csv`
- `graphs/graph_builds.jsonl`
- `manifest.json`

Adjacency caches are stored separately under:

`artifacts/rpg/perf/sports/graphs/`

## Snellius jobs

Repo-owned job scripts live in:

- `jobs/reproduction/perf/build_graphs.sh`
- `jobs/reproduction/perf/profile_inference.sh`

The defaults are aligned with Snellius partition guidance:

- `build_graphs.sh` uses the CPU-only `genoa` partition because adjacency preparation does not need a GPU.
- `profile_inference.sh` uses `gpu_a100` with `1` GPU, `18` CPU cores, and `120G` host memory, matching the 1/4-node A100 allocation documented in the UvA Snellius guide.
- For this course setup, both `gpu_a100` and `gpu_h100` are available GPU partitions on Snellius.
- If you want to run the profiling pass on H100 nodes instead, override the defaults at submit time:

```bash
cd jobs/reproduction/perf
sbatch --partition=gpu_h100 --cpus-per-task=16 --mem=180G \
  ./profile_inference.sh /abs/path/to/checkpoint.pth
```

Reference guide:

`https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial1/Lisa_Cluster.html`

They must be submitted from their own directory:

```bash
cd jobs/reproduction/perf
sbatch ./build_graphs.sh /abs/path/to/checkpoint.pth
sbatch ./profile_inference.sh /abs/path/to/checkpoint.pth
```

Scheduler logs are mirrored under:

`output/reproduction/perf/`

## Notes

- The profiling path does not modify `third_party/`.
- The timed epoch excludes graph construction by design.
- The exact validation path is intended for the original Sports pool only.
- The HNSW backend is the scalable path for the enlarged-pool experiment.
