# Reproducibility and Efficiency Study of RPG

Code and results for the paper. It reproduces RPG (Hou et al., KDD 2025),
compares it with SASRec and TIGER retrained under one protocol, analyses its
graph-constrained decoding, and tests it on Amazon Books (368k items).

The two upstream code bases are pinned submodules and are not modified.

```
third_party/RPG_KDD2025/   RPG as released by its authors
third_party/MemGen-GR/     Ding et al.'s framework: full-softmax SASRec and TIGER
decoding/                  our decoders for RPG: exact and HNSW graph builders, exhaustive scoring, traced decoder
models/sasrec/             SASRec with the S3-Rec recipe (BCE, one sampled negative)
scripts/{rpg,sasrec,memgen}/   train, evaluate, analyse; every script prints --help
scripts/results/           collectors and plots that turn artifacts/ into results/
configs/                   settings per model and dataset
jobs/                      the experiments as shell scripts (Slurm headers; run with bash without Slurm)
results/                   the tables and figures of the paper
artifacts/, output/        data, checkpoints, logs (not in git)
```

## Setup

```bash
git clone <repository> && cd <repository>
git submodule update --init
python3.11 -m venv artifacts/venv && artifacts/venv/bin/pip install -r requirements.txt
cd jobs/data && sbatch ./prepare_data.sh      # or: bash ./prepare_data.sh
```

`requirements.txt` pins the versions we used (PyTorch 2.11, CUDA 12.8, FAISS 1.12).
A GPU with 24 GB is enough for every experiment except full-softmax SASRec on
Books (40 GB). `prepare_data.sh` downloads and preprocesses the five Amazon
Reviews 2014 categories once, into a cache shared by all models.

RPG's semantic IDs need `text-embedding-3-large` embeddings of the item texts.
For the four original datasets, place the embeddings released by the RPG authors
under `artifacts/rpg/cache/AmazonReviews2014/<Category>/processed/`, or put an
OpenAI key in `configs/rpg/local.yaml` (copy `local.example.yaml`; git-ignored)
and they are computed on first use. For Books, `jobs/books/prepare_semantic_ids.sh`
embeds the 368k items (about 63M tokens).

Our checkpoints and evaluation outputs are available at
https://osf.io/hnmdu/overview?view_only=861c95035c114a25b73ca2b8b4b2ac0f
(`rpg_repro_results.zip`, 9 MB, and `rpg_repro_checkpoints.zip`, 2.4 GB).
Unzip them in the repository root; the collectors below then rebuild every
table without a GPU. Each zip has a README listing its contents.

## Running the experiments

Run each job from its own folder (`cd jobs/... && sbatch ./x.sh`). Jobs without
a Slurm header (`jobs/books/`, `jobs/latency/run_latency.sh`,
`jobs/reproduction/rpg/decoding_variants/`) were written for a shared GPU
server and take `GPU=<index>`; on Slurm wrap them, e.g.
`sbatch -p gpu_a100 --gpus=1 --chdir=jobs/books --wrap "GPU=0 bash ./eval.sh"`.
Each job's header says what it does, what it writes and which variables it takes.

**RQ1: reproducibility of RPG** (Tables 1 and 2, Fig. 3)

| Step | Job |
| --- | --- |
| Train RPG, released settings, seed 2024 | `jobs/reproduction/rpg/<dataset>/train.sh` |
| Train seeds 2025 and 2026 | `jobs/reproduction/rpg/seeds/train.sh <dataset> "2025 2026"` |
| Evaluate every checkpoint with graph decoding | `jobs/reproduction/rpg/seeds/eval.sh` |
| Paper-appendix decoding settings and cold start (RPG and SASRec) | `jobs/reproduction/rpg/seeds/eval_rq1.sh` |
| Semantic ID length m = 4 to 64, then re-decode | `jobs/reproduction/rpg/sweep_m/<dataset>.sh`, `redecode_configs.sh` |

Settings: `configs/rpg/repro/<dataset>.yaml` (released code) and
`configs/rpg/eval_seeds/paper_appendix/<dataset>.yaml` (paper appendix).

**RQ2: baselines** (Table 3, Fig. 4)

| Step | Job |
| --- | --- |
| SASRec and parameter-matched SASRec, seeds 2024/2025/2026 | `jobs/reproduction/sasrec/seeds/train_eval.sh <dataset> <Category> <hidden> <layers> "2024 2025 2026"` |
| Learning rate and depth selection of parameter-matched SASRec | `jobs/reproduction/sasrec/tuning/` |
| Full-softmax SASRec and TIGER (MemGen-GR) | `jobs/baselines/memgen/train.sh {SASRec,TIGER} <dataset> "2024 2025 2026"` |

Hidden sizes of parameter-matched SASRec: 326 (Sports), 540 (Beauty, 3 blocks),
396 (Toys), 328 (CDs); `scripts/results/count_model_params.py` prints the counts.

**RQ3: graph-constrained decoding** (Table 4)

| Step | Job |
| --- | --- |
| Exhaustive scoring, exact graph and HNSW graph on the same checkpoints | `jobs/reproduction/rpg/decoding_variants/run_decoding_variants.sh` |
| Checks against the upstream code and reachability of the exhaustive top 10 | `jobs/reproduction/rpg/decoding_variants/run_graph_checks.sh` |

Graph backends (`decoding/graph.py`): `dense` is the upstream builder (full
N x N matrix), `exact_gpu` gives the same neighbours in row blocks, `hnsw` is a
FAISS HNSW index (32 links, efConstruction 200) over the normalised mean token
embeddings. `scripts/results/graph_overlap.py` measures the HNSW overlap.

**RQ4: Books and efficiency** (Table 5, Figs. 1 and 2)

| Step | Job |
| --- | --- |
| Books: embeddings, semantic IDs, training (m = 64 and 32), evaluation | `jobs/books/prepare_semantic_ids.sh`, `train.sh`, `eval.sh` |
| Books at larger decoding budgets | `jobs/books/budget_sweep.sh` |
| Books baselines, one seed | `EXTRA_ARGS="--val_max_users 20000" sbatch jobs/reproduction/sasrec/seeds/train_eval.sh books Books 64 2 "2024"`, `jobs/baselines/memgen/train.sh SASRec books "2024"` |
| Latency per user, GPU and CPU | `jobs/latency/run_latency.sh` (RPG), `jobs/latency/run_baselines.sh` (SASRec, TIGER) |
| Graph construction memory, dense vs row-blocked | `scripts/rpg/graph_build_timing.py` |
| Item pool enlarged with dummy items | `jobs/reproduction/rpg/perf/build_graphs.sh`, `profile_inference.sh`, `profile_bruteforce.sh` |

Books settings: `configs/rpg/books.yaml` (the CDs settings), `configs/sasrec/books.yaml`,
`configs/memgen/sasrec/books.yaml`. RPG on Books trains with `--train-only`
because the upstream end-of-training test builds the dense graph (540 GB).

## Results

| File | Produced by | Paper |
| --- | --- | --- |
| `results/main_results.csv` | `scripts/results/collect_main.py` | Tables 2 and 3 |
| `results/decoding_variants.csv` | `scripts/results/collect_main.py` | Table 4 |
| `results/latency.csv` | `scripts/results/collect_main.py` | Table 5, Fig. 2 |
| `results/books_scaling.csv` | `scripts/results/collect_books.py` | Table 5, Fig. 1a |
| `results/redecode_compare.csv` | `scripts/results/collect_redecode.py` | Fig. 3 |
| `results/sasrec_train_grid.csv` | tuning logs | parameter-matched SASRec settings |
| `results/figures/` | `scripts/results/plot_books_scaling.py`, `plot_rq1_rq2.py` | Figs. 1 to 4 |

`collect_main.py` flags the two TIGER runs that did not converge (Sports and
CDs, seed 2025) and leaves them out of the means; Sports was rerun with seed 2027.

## Notes on the implementation

- RPG runs unmodified. Our scripts load the upstream model and only replace the
  decoding graph or score all items. `scripts/rpg/verify_decoding.py` checks that
  the row-blocked graph, the exhaustive scores and the traced decoder equal the
  upstream code.
- Past items: all models are tested with the user's past items kept in the
  ranking, as in the RPG code. The S3-Rec code removes them; `models/sasrec`
  keeps that for early stopping and `--mask_history false` turns it off at test time.
- `scripts/memgen/train.py` replaces MemGen-GR's per-prediction memorisation
  statistics by a plain evaluation loop with the same metrics.
- Full-softmax SASRec on Books uses batch size 64 and learning rate 3e-4, the
  only baseline setting changed from Ding et al.

## Licence

Our code is under the MIT licence (`LICENSE`). The submodules keep their own
licences (RPG: CC BY-NC 4.0).
