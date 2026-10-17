# Reproducibility and Efficiency Study of RPG

Code for the paper. It reproduces RPG (Recommendation with Parallel semantic ID
Generation, Hou et al., KDD 2025), compares it with TIGER and SASRec under
controlled model capacity and training objective, analyses its graph-constrained
decoding, and tests the method on Amazon Books, a catalogue of 368k items.

The two upstream code bases are pinned, read-only submodules and are never
edited. Everything else in this repository is ours.

```
third_party/RPG_KDD2025/   RPG, as released by its authors (GPT-2 backbone, OPQ semantic IDs, graph decoding)
third_party/MemGen-GR/     Ding et al.'s framework, used for the SASRec (softmax loss) and TIGER baselines
decoding/                  decoding graph builders (exact, HNSW), exhaustive scoring, evaluation harness, pool scaling
graph_analysis/            analyses of graph decoding: reachability, frontier memory, scoring parity, traced decoder
models/sasrec/             SASRec with the S3-Rec recipe (binary cross-entropy, one negative), the parameter-matched baseline
scripts/rpg/               RPG: train, eval, multi-seed eval, semantic IDs, cold start, decoding analyses, latency
scripts/sasrec/            SASRec (S3-Rec recipe): prepare data, train, eval, cold start, latency
scripts/memgen/            SASRec with softmax loss and TIGER through MemGen-GR: train, latency
scripts/results/           collectors and plots that turn artifacts/ into results/
configs/                   experiment settings: configs/rpg, configs/sasrec, configs/memgen
jobs/                      shell scripts that run the experiments (Slurm headers kept, run fine without Slurm)
results/                   the tables and figures behind the paper
artifacts/, output/        data caches, checkpoints, logs (not in git)
```

## Setup

```bash
git clone <repository> && cd <repository>
git submodule update --init
python3.11 -m venv artifacts/venv            # or: uv venv --python 3.11 artifacts/venv
artifacts/venv/bin/pip install -r requirements.txt
```

`requirements.txt` pins the versions we used (PyTorch 2.11 with CUDA 12.8). Every
job script activates `artifacts/venv`; set `PYTHON_BIN` to use another interpreter.
One GPU with 24 GB is enough for every experiment; most run on a single A5500 or
A100. CPU is enough for the decoding analyses on the four paper datasets.

**Data.** The Amazon Reviews 2014 categories are downloaded and processed
automatically by the upstream code on first use into
`artifacts/rpg/cache/AmazonReviews2014/<Category>/`. RPG's semantic IDs need an
embedding of every item's text with OpenAI's `text-embedding-3-large`. Either use
the embeddings released by the RPG authors (place `<model>.sent_emb` under
`.../processed/`), or set `openai_api_key` in `configs/rpg/local.yaml` (copy
`configs/rpg/local.example.yaml`; the file is ignored by git). For Books,
`scripts/rpg/embed_items.py` embeds the 368k items in resumable batches
(about 63M tokens).

**Checkpoints and caches** that we trained are available from the link in the
paper. Place them under `artifacts/` with the layout used below.

## Experiments

Run each job from its own folder. Jobs with an `#SBATCH` header run with
`sbatch ./x.sh` on a Slurm cluster or `bash ./x.sh` on any machine. Jobs without
one (`jobs/books/*.sh`, `jobs/latency/run_latency.sh`,
`jobs/reproduction/rpg/decoding_variants/`)
were written for a shared GPU server: they take `GPU=<index>` and refuse a GPU
that is already in use. On Slurm, wrap them, for example
`sbatch -p gpu_a100 --gpus=1 --chdir=jobs/books --wrap "GPU=0 bash ./eval.sh"`.
Scripts print `--help`. Outputs go to `artifacts/` and `output/`; the
collectors in the last table turn them into `results/`.

Before the first GPU job, `cd jobs/data && sbatch ./prepare_data.sh` downloads
and preprocesses the five datasets once (shared cache for RPG, MemGen-GR and
SASRec), caches the sentence-T5 model used by TIGER, and reports how often a
test item already appears in the user's history (never, on these datasets).

### RQ1: reproducibility of RPG

| What | Job | Script |
| --- | --- | --- |
| Train and test RPG with the released hyperparameters | `jobs/reproduction/rpg/<dataset>/train.sh`, `eval.sh` | `scripts/rpg/train.py`, `scripts/rpg/eval.py` |
| Extra training seeds (2025, 2026), then one evaluation of every dataset and seed with the same settings, past items kept and removed | `jobs/reproduction/rpg/seeds/train.sh`, `eval.sh` | `scripts/rpg/train.py`, `scripts/rpg/eval_seeds.py`, `scripts/rpg/history_masking.py` |
| Test with 10 evaluation seeds and bootstrap confidence intervals, released-code and paper-appendix decoding settings | `jobs/reproduction/eval_seeds/{released_readme,paper_appendix}/run_<dataset>.sh` | `scripts/rpg/eval_seeds.py` |
| Semantic ID length sweep (m = 4 to 64) and re-decoding | `jobs/reproduction/train/sweep_m/<dataset>.sh`, `redecode_configs.sh` | `scripts/rpg/train.py`, `scripts/rpg/eval_seeds.py` |
| Training and decoding hyperparameter grids | `jobs/reproduction/rpg/grid/run_train_grid.sh`, `run_decode_grid.sh`, `run_decode_confirm.sh`, `run_infer_grid.sh`, `run_fig6_repro.sh` | `scripts/rpg/train.py`, `scripts/rpg/eval_seeds.py` |
| Cold start: test items bucketed by training frequency | `jobs/reproduction/rpg/cold_start/`, `jobs/reproduction/cold_start/released_readme/` | `scripts/rpg/cold_start.py` |

Datasets: `sports_and_outdoors`, `beauty`, `toys_and_games`, `cds_and_vinyl`.
The RPG settings are in `configs/rpg/repro/<dataset>.yaml` (released code) and
`configs/rpg/eval_seeds/paper_appendix/` (paper appendix).

### RQ2: comparison with baselines

| What | Job | Script |
| --- | --- | --- |
| SASRec, S3-Rec recipe (BCE), original width 64 | `jobs/reproduction/sasrec/<dataset>/prepare_data.sh`, `train.sh`, `eval.sh` | `scripts/sasrec/prepare_data.py`, `scripts/sasrec/train.py` |
| SASRec, parameter-matched to RPG (hidden size 326/540/396/328) with learning-rate and depth grids | `jobs/reproduction/sasrec/ablation_size/` | `scripts/sasrec/train.py`, `scripts/sasrec/find_best_ablation_checkpoint.py` |
| SASRec and parameter-matched SASRec, seeds 2024/2025/2026, each tested with past items removed and kept | `jobs/reproduction/sasrec/seeds/train_eval.sh` | `scripts/sasrec/train.py --mask_history {true,false}` |
| SASRec cold start | `jobs/reproduction/sasrec/cold_start/` | `scripts/sasrec/cold_start.py` |
| SASRec with full-softmax loss and TIGER (MemGen-GR implementations), seeds 2024/2025/2026 | `jobs/baselines/memgen/train.sh SASRec sports "2024 2025 2026"` | `scripts/memgen/train.py` |
| Full-softmax SASRec tested again with past items removed (evaluation only) | `jobs/baselines/memgen/eval_removed.sh` | `scripts/memgen/train.py --eval-only --mask-history` |
| Parameter counts of RPG and the SASRec variants | | `scripts/results/count_model_params.py` |
| RPG with and without the user's past items in the ranking (the SASRec protocol removes them) | | `scripts/rpg/history_masking.py` |

`scripts/memgen/train.py` runs the MemGen-GR code unchanged except that its
per-prediction memorisation statistics are replaced by a plain evaluation loop
(same metrics); `--fine-grained-eval` restores the original loop. The MemGen-GR
settings per dataset are in `configs/memgen/<model>/<dataset>.yaml`.

### RQ3: analysis of graph-constrained decoding

| What | Job | Script |
| --- | --- | --- |
| Build the exact decoding graph for a checkpoint | `jobs/reproduction/rpg/graph_analysis/` (`prepare-graph`) | `scripts/rpg/graph_analysis.py prepare-graph` |
| Reachability: which top-ranked items the search visits and keeps, per step | `run_dynamic_<dataset>.sh` | `scripts/rpg/graph_analysis.py dynamic` |
| Graph decoding vs exhaustive scoring of all items, same model | `run_scoring_<dataset>.sh` | `scripts/rpg/graph_analysis.py scoring` |
| Frontier memory: re-entry of visited items and explicit candidate memory | `run_frontier_memory_sports.sh` | `scripts/rpg/graph_analysis.py frontier-memory` |
| Exact graph vs approximate HNSW graph vs exhaustive, same checkpoints and seeds | `jobs/reproduction/rpg/decoding_variants/run_decoding_variants.sh` | `scripts/rpg/eval_seeds.py --graph-backend {exact_gpu,hnsw}` |
| Checks against the upstream code (graph, exhaustive scores, traced decoder) | | `scripts/rpg/verify_decoding.py` |

Graph backends (`decoding/graph.py`): `dense` is the upstream builder (full n x n
similarity matrix on the GPU); `exact_gpu` computes the same similarity and the
same neighbours in row blocks, so the full matrix is never stored; `hnsw` builds
an approximate graph with FAISS over the normalised mean of each item's token
embeddings (a different similarity from upstream, without self-loops).

### RQ4: large catalogue and inference efficiency

| What | Job | Script |
| --- | --- | --- |
| Amazon Books: embeddings, semantic IDs, training, graph and exhaustive evaluation | `jobs/books/prepare_semantic_ids.sh`, `train.sh`, `eval.sh` | `scripts/rpg/embed_items.py`, `scripts/rpg/prepare_semantic_ids.py`, `scripts/rpg/train.py --train-only`, `scripts/rpg/eval_seeds.py` |
| Books at larger decoding budgets (beams, edges, steps) | `jobs/books/budget_sweep.sh` | `scripts/rpg/eval_seeds.py` |
| Per-query latency, graph decoding vs exhaustive, GPU and CPU, batch 1/16/128 | `jobs/latency/run_latency.sh` (RPG), `jobs/latency/run_baselines.sh` (SASRec, full-softmax SASRec, TIGER) | `scripts/rpg/latency.py`, `scripts/sasrec/latency.py`, `scripts/memgen/latency.py` |
| Graph construction time and memory, dense vs row-blocked | | `scripts/rpg/graph_build_timing.py` |
| Inference time and memory with the item pool enlarged by dummy items (the original paper's experiment) | `jobs/reproduction/rpg/perf/` | `scripts/rpg/perf.py` |
| Baselines on Books, one seed: SASRec (S3-Rec recipe, width 64) and full-softmax SASRec | `EXTRA_ARGS="--val_max_users 20000" sbatch jobs/reproduction/sasrec/seeds/train_eval.sh books Books 64 2 "2024"`, `sbatch jobs/baselines/memgen/train.sh SASRec books "2024"` | `scripts/sasrec/train.py`, `scripts/memgen/train.py` |

Books settings: `configs/rpg/books.yaml` (the CDs settings, untuned),
`configs/sasrec/books.yaml`, `configs/memgen/sasrec/books.yaml` (batch sizes
reduced to fit a 40 GB GPU, learning rate 3e-4, validation every epoch with
patience 5). Training uses `--train-only` because the upstream
end-of-training test builds the dense graph, which would need 504 GB on Books,
and early-stops on a fixed 20,000-user validation subset.

## Results

| File | Produced by | Paper |
| --- | --- | --- |
| `results/train_grid.csv`, `infer_grid.csv`, `decode_val_grid.csv`, `decode_val_selected.csv`, `decode_val_cluster.csv`, `decode_test_selected.csv`, `decode_test_confirm.csv`, `fig6_grid.csv` | `scripts/results/collect_grid.py` | RQ1 grids |
| `results/redecode_compare.csv` | `scripts/results/collect_redecode.py` | RQ1 semantic ID length |
| `results/sasrec_train_grid.csv` | `scripts/results/collect_grid.py` | RQ2 |
| `results/main_results.csv` | `scripts/results/collect_main.py` | RQ1, RQ2: every model, dataset, seed and protocol (Tables 2, 3) |
| `results/decoding_variants.csv` | `scripts/results/collect_main.py` | RQ3: exhaustive vs exact graph vs HNSW graph (Table 4) |
| `results/latency.csv` | `scripts/results/collect_main.py` | RQ4: ms per user, all models, GPU and CPU |
| `results/books_scaling.csv` | `scripts/results/collect_books.py` | RQ4 |
| `results/figures/*.pdf` | `scripts/results/plot_books_scaling.py` | RQ3 reachability, RQ4 budget and latency figures |
| `results/tables/*.csv` | summaries of the multi-seed evaluation outputs (`artifacts/*/eval_seeds/*/summary.json`) and of the grids above | RQ1, RQ2 headline tables |

`scripts/results/collect_sweep_m.py` prints the semantic ID length sweep from the
training logs.

## Differences from the original code bases

- RPG is run unmodified. Our scripts load the upstream model and only replace
  the decoding graph (`exact_gpu`, `hnsw`) or score all items (exhaustive
  scoring). `scripts/rpg/verify_decoding.py` checks that the exact graph, the
  exhaustive scores and the traced decoder equal the upstream code.
- Hyperparameters come from the released code (`configs/rpg/repro/`), which
  differ from the paper appendix in beam size, graph width and steps; both are
  evaluated.
- Evaluation protocol: RPG ranks the full catalogue including the user's past
  items; the S3-Rec SASRec code removes them. `models/sasrec` keeps that
  behaviour by default (`mask_history`), and `scripts/rpg/history_masking.py`
  reports RPG both ways.
- Two SASRec recipes are compared: the S3-Rec recipe (BCE, one sampled
  negative) in `models/sasrec`, at the original width and parameter-matched to
  RPG, and MemGen-GR's SASRec (GPT-2 blocks, full softmax over all items). On
  Books the full softmax does not fit at the default batch size, so MemGen-GR's
  SASRec uses batch 64 and learning rate 3e-4 (`configs/memgen/sasrec/books.yaml`).
- All models are trained with seeds 2024, 2025 and 2026. Two TIGER runs did not
  converge (Sports and CDs, seed 2025); Sports was rerun with seed 2027, and
  `scripts/results/collect_main.py` flags both and leaves them out of the means.
- MemGen-GR's evaluation keeps past items in the ranking;
  `scripts/memgen/train.py --eval-only --mask-history` tests a SASRec checkpoint
  with them removed, as in the S3-Rec code. Both RPG and MemGen-GR masks cover
  the items in the model's input window (the last 50 interactions).
- MemGen-GR's evaluation loop computes memorisation statistics per prediction in
  Python; `scripts/memgen/train.py` swaps in a plain loop with identical metrics.
- The HNSW graph follows the second code base: FAISS `IndexHNSWFlat(d, 32)`,
  efConstruction 200, efSearch max(k+1, 50), item vectors are the normalised
  mean of the token embeddings, the item itself is dropped from its neighbours.
- Books uses the CDs hyperparameters without tuning, and the early-stopping
  validation is a fixed 20,000-user subset.

## Licences

Our code is released under the licence in `LICENSE`. The RPG code
(`third_party/RPG_KDD2025`) is CC BY-NC 4.0 and MemGen-GR (`third_party/MemGen-GR`)
carries its own licence; both are included unchanged as submodules.
