# RPG Reproduction Notes

This note explains why the current Sports and Outdoors reproduction logs do not exactly match the numbers in `docs/pdf/2506.05781v1.md`.

## Observed Sports Run

The Sports reproduction jobs are in:

- `jobs/reproduction/sports_and_outdoors/`
- `output/reproduction/sports_and_outdoors/`

The completed training job is:

- `output/reproduction/sports_and_outdoors/rpg_sports_and_outdoors-23470485.*`

The eval-only job is:

- `output/reproduction/sports_and_outdoors/rpg_sports_eval-23472130.*`

The train job used the paper-facing Sports hyperparameters from `configs/rpg/repro/sports_and_outdoors.yaml`:

```yaml
lr: 0.003
temperature: 0.03
n_codebook: 16
num_beams: 10
n_edges: 100
propagation_steps: 2
```

Those match the Sports row in the paper's reproduction hyperparameter table.

## Dataset Count Mismatch

The paper table reports Sports as:

```text
users:        18,357
items:        35,598
interactions: 260,739
avg t:        8.32
```

The run log reports:

```text
Number of users:        35,599
Number of items:        18,358
Number of interactions: 296,337
Average item sequence length: 8.32430686255232
```

The code includes a `[PAD]` id in both user and item mappings. Excluding that padding row, the run has:

```text
users: 35,598
items: 18,357
```

That means the paper's Sports user and item columns appear to be swapped relative to the code's logged dataset object.

The interaction count difference is also explainable from the leave-last-out protocol. The run logs the full sequence interaction count before the held-out test item is removed. The paper's `260,739` is:

```text
296,337 full interactions - 35,598 non-padding users = 260,739
```

So the paper appears to count training-plus-validation interactions after one test item per user is held out, while the code logs full pre-split interactions.

## Metric Mismatch

The paper reports Sports RPG NDCG@10 as:

```text
0.0263
```

The training job's built-in final test evaluation reports:

```text
ndcg@5:         0.020510302856564522
ndcg@10:        0.025064710527658463
recall@5:       0.02932748943567276
recall@10:      0.043429404497146606
n_visited_items: 4573.0419921875
```

This is close to, but not exactly, the paper value.

The separate eval-only job reports a lower value for the same checkpoint:

```text
ndcg@5:         0.016531581059098244
ndcg@10:        0.019578667357563972
recall@5:       0.02314736694097519
recall@10:      0.032726556062698364
n_visited_items: 1667.568115234375
```

The eval-only result should not be treated as a direct deterministic re-read of the training job's final test result.

## Why Eval-Only Can Differ

RPG's graph-constrained decoding is stochastic. In `third_party/genrec/models/RPG/model.py`, graph propagation starts from randomly sampled item nodes:

```python
topk_nodes_sorted = torch.randint(
    1, self.dataset.n_items,
    (batch_size, self.num_beams),
    dtype=torch.long,
    device=token_logits.device
)
```

The training job evaluates after training has consumed a long sequence of random numbers. The eval-only job starts a fresh process and reaches graph decoding with a different RNG state. Because graph decoding is approximate, different random initial beams can produce different recommendation metrics from the same checkpoint.

This is also visible in `n_visited_items`: the training job visited about `4573` items on average, while eval-only visited about `1668`.

## Bottom Line

The reproduction is not exact for three separate reasons:

1. The paper's Sports user/item columns appear swapped relative to the logged code dataset.
2. The paper's Sports interaction count appears to be post-test-holdout, while the code logs full pre-split interactions.
3. RPG graph-constrained decoding uses random initialization, so a one-off eval-only run can differ from the training job's final test evaluation unless the evaluation seed/RNG path is controlled or multiple eval runs are averaged.

For comparison against the paper, the training job's built-in final test result is the better single-run number than the separate eval-only job. For a more stable estimate, run multiple eval passes with controlled seeds and report mean plus standard deviation.

## SASRec Reproduction (S3-Rec Path)

SASRec baselines are now implemented in-repo under `models/sasrec`, with job entrypoints:

- `jobs/reproduction/sasrec/sports_and_outdoors/train.sh`
- `jobs/reproduction/sasrec/beauty/train.sh`
- `jobs/reproduction/sasrec/toys_and_games/train.sh`

Config defaults that match the released S3-Rec `run_finetune_full.py` SASRec path:

- `hidden_size=64`, `num_hidden_layers=2`, `num_attention_heads=2`
- `hidden_act=gelu`, `attention_probs_dropout_prob=0.5`, `hidden_dropout_prob=0.5`
- `initializer_range=0.02`, `max_seq_length=50`
- `lr=0.001`, `train_batch_size=256`, `eval_batch_size=256`, `weight_decay=0.0`
- `adam_beta1=0.9`, `adam_beta2=0.999`, `rand_seed=42`, `log_freq=1`
- `patience=10`, `eval_interval=1`, `full_sort=true`, `topk=[5,10,20]`, `val_metric=ndcg@20`
- `split=leave_last_out`, `rating_score=0.0`, `user_core=5`, `item_core=5`, `attribute_core=0`

Target row metrics from the paper reproduction table are:

- Sports and Outdoors: Recall@5 `0.0233`, NDCG@5 `0.0154`, Recall@10 `0.0350`, NDCG@10 `0.0192`
- Beauty: Recall@5 `0.0387`, NDCG@5 `0.0249`, Recall@10 `0.0605`, NDCG@10 `0.0318`
- Toys and Games: Recall@5 `0.0463`, NDCG@5 `0.0306`, Recall@10 `0.0675`, NDCG@10 `0.0374`

Seed behavior:

- Use the current released S3-Rec seeding contract (including script-level NumPy seeding in preprocessing and the default `rand_seed=42` runner path), not the historical buggy seed behavior documented in older reproduction notes.

The in-repo data prep path is `scripts/sasrec_prepare_data.py`, with:

- category downloads from `https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_{category}_5.json.gz`
- chronological per-user sorting
- `np.random.seed(12345)` before k-core filtering
- PAD-reserved dense remapping of users/items (start at ID 1)
- leave-last-out split: `train=items[:-3]` labels, `val=items[-2]`, `test=items[-1]`

Contract details for evaluation/training behavior:

- BCE over one positive and one sampled negative per non-padding target position
- negative sampling rejects items present in the user's interacted-item set
- full-sort ranking over all items, masking train-history items in evaluation
- top-20 selection with `argpartition`
- checkpoint selected by `ndcg@20`
