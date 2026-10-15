# Books (large catalogue)

Amazon 2014 Books (5-core): 603,668 users, 367,982 items, 8.9M interactions,
almost six times the largest catalogue of the original paper. The jobs in this folder run directly on a GPU server without Slurm, take the
GPU index as `GPU=<index>`, and refuse a GPU that is already in use. Run every
job from this folder.

## RPG pipeline

1. `jobs/books/prepare_semantic_ids.sh` — process the raw
   data, embed the item text with `text-embedding-3-large`
   (`scripts/rpg_embed_items.py`, resumable, about 63M tokens) and build the
   semantic IDs with OPQ. CPU only. Needs `openai_api_key` in
   `configs/rpg/local.yaml`. Add `--n_codebook 32` for 32-digit IDs.
2. `jobs/books/train.sh` — trains with `--train-only` (the
   upstream end-of-training test would build a dense n x n graph) and early
   stopping on a fixed 20,000-user validation subset (`--val-max-users`).
   About 28 min per epoch for 64 digits on an RTX A5500; use `RUN_TAG=m32`
   plus `--n_codebook 32` for the 32-digit model.
3. `jobs/books/eval.sh` — graph decoding at the configured
   budget (graph built exactly, in row blocks on the GPU:
   `--graph-backend exact_gpu`) and exhaustive scoring, on all test users.
   `GRAPH_BACKEND=hnsw` uses an approximate HNSW graph instead (CPU, slow).
4. `jobs/books/budget_sweep.sh` — graph decoding at larger
   budgets (`BUDGETS="beams:edges:steps ..."`) on a fixed 100,000-user subset.

## Analyses used in the paper

| Analysis | Script | Output |
| --- | --- | --- |
| Correctness checks and reachability | `scripts/rpg_verify_decoding.py` | `artifacts/rpg/verification/<name>.json` |
| RPG with and without past items in the ranking | `scripts/rpg_history_masking.py` | `artifacts/rpg/history_masking/<name>.json` |
| Graph construction time and memory | `scripts/rpg_graph_build_timing.py` | `artifacts/rpg/graph_build/` |
| RPG per-query latency (Sports, CDs, Books) | `jobs/latency/run_latency.sh` → `scripts/rpg_latency.py` | `artifacts/rpg/latency/` |
| SASRec per-query latency | `scripts/sasrec_latency.py` | `artifacts/sasrec/latency/` |
| Result tables | `scripts/collect_books_results.py`, `scripts/collect_sasrec_audit.py` | `results/books_scaling.csv`, `results/sasrec_audit.csv` |
| Figures | `scripts/plot_books_scaling.py` | `results/figures/` |

`rpg_verify_decoding.py` checks, on a trained checkpoint, that the row-blocked
graph equals the upstream graph, that exhaustive scoring equals the upstream
full-catalogue path, and that the traced decoder used for reachability equals
upstream `RPG.generate`; it then measures how many of the model's exhaustive
top-10 hits graph decoding reaches and keeps.

## SASRec on Books

```bash
python3 scripts/sasrec_prepare_data.py --categories Books
cd jobs/sasrec_audit
GPU=7 QUEUE="books:bce:64:val_max_users=20000 books:sampled_ce:64:val_max_users=20000" bash ./run_queue.sh
```

`sampled_ce` estimates the full-catalogue softmax with 4,096 uniform negatives
per batch and a logQ correction, since the exact softmax over 368k items does
not fit in memory; on Beauty both give the same NDCG@10 (0.0573 vs 0.0574).
Width 64 makes SASRec parameter-matched to the 64-digit RPG (23.65M vs 23.69M).
Evaluate without removing past items, as RPG does, with
`scripts/sasrec.py --eval-only --mask_history false`.

The RPG hyperparameters in `configs/rpg/books.yaml` reuse the
CDs_and_Vinyl settings and have not been tuned.
