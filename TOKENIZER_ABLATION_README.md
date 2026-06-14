# RPG Tokenizer Ablation Notes

This branch adds a small, repo-owned tokenizer ablation layer for RPG. The goal
is to test whether changing only the semantic-ID generation method can improve,
explain, or stress-test RPG, while keeping the RPG model, loss, decoding,
datasets, metrics, and training code unchanged.

The implementation deliberately avoids editing `third_party/`. The local model
package is in `models/RPGAblation/`, and it subclasses upstream RPG behavior.

## Goal

The original RPG tokenizer uses FAISS OPQ to convert item text embeddings into
semantic IDs. This branch adds alternative semantic-ID generators so we can
compare:

- the upstream OPQ baseline,
- plain PQ without OPQ rotation,
- library-backed factorized FSQ on PCA latents,
- FSQ-inspired quantile scalar bins.

The research question is:

> Can simpler or more reproducible scalar/PQ-style tokenizers produce semantic
> IDs that are competitive with OPQ for RPG recommendation quality, stability,
> code utilization, and cold-start behavior?

## What Was Added

### Local model package

New package:

```text
models/RPGAblation/
```

Important files:

- `model.py`: `RPGAblation` subclasses upstream `RPG`.
- `tokenizer.py`: `RPGAblationTokenizer` subclasses upstream `RPGTokenizer`.
- `quantizers.py`: implements the ablation semantic-ID generators.
- `config.yaml`: same RPG defaults, with `semantic_id_method: fsq`.

Only semantic-ID cache generation changes. Once IDs are generated, the upstream
RPG token offsetting, training loss, graph decoding, and evaluation path remain
the same.

### New dependency

Real FSQ scalar rounding is provided by:

```text
vector-quantize-pytorch==1.29.1
```

This was added to both:

- `requirements.txt`
- `environment.yml`

### New configs

Tokenizer ablation configs live in:

```text
configs/rpg/tokenizer_ablation/
```

Current configs:

```text
sports_fsq.yaml
sports_fsq_quantile.yaml
sports_pq.yaml
```

Use them with `--model RPGAblation`. The launcher chooses the model before the
config file is interpreted, so the model argument is required.

## Implemented Methods

### Baseline: upstream OPQ

This remains the original RPG behavior when running:

```bash
python3 scripts/rpg.py --model RPG ...
```

Upstream RPG uses a FAISS index factory like:

```text
OPQ{n_codebook},IVF1,PQ{n_codebook}x{bits}
```

OPQ first learns a global rotation, then PQ quantizes the rotated space.

### `pq`

Config:

```yaml
semantic_id_method: pq
```

Implementation:

```text
embeddings
-> existing RPG sent_emb_pca
-> FAISS IVF1,PQ without OPQ rotation
-> RPG semantic-ID digits
```

This tests how much the OPQ rotation helps. Hypothesis: PQ should usually be
slightly worse than OPQ because OPQ balances information across PQ subspaces.

### `fsq`

Config:

```yaml
semantic_id_method: fsq
```

Implementation:

```text
embeddings
-> existing RPG sent_emb_pca
-> PCA to n_codebook dimensions
-> vector-quantize-pytorch FSQ scalar bounding/rounding
-> one level ID per PCA coordinate
-> RPG semantic-ID digits
```

This is **library-backed, factorized FSQ on PCA latents**.

It is not a trained FSQ-VAE. There is no learned neural encoder or decoder.
The only fitted transformation is PCA, fitted on training-prefix items. FSQ is
then applied independently to each scalar coordinate, because RPG needs one
classification label per semantic-ID digit.

This avoids constructing a huge mixed-radix code space such as `256 ** 16`,
while still using the library's FSQ scalar quantization operation.

Hypothesis: FSQ is simple and deterministic, but it may underperform OPQ if the
fixed finite levels do not match the distribution of recommendation embeddings.
Its diagnostics should be entropy/utilization and collision rate.

### `fsq_quantile`

Config:

```yaml
semantic_id_method: fsq_quantile
```

Implementation:

```text
embeddings
-> existing RPG sent_emb_pca
-> PCA to n_codebook dimensions
-> quantile-bin each scalar coordinate into codebook_size bins
-> RPG semantic-ID digits
```

This is not FSQ from the FSQ paper. It is an FSQ-inspired scalar quantile
baseline. The useful scientific question is different:

> Does forcing high marginal code utilization per digit help RPG?

Hypothesis: this method should have strong per-digit utilization by
construction. If it still performs poorly, balanced scalar labels alone are not
enough.

## Cache Names And Stats

The ablation tokenizer writes method-specific semantic-ID caches under the
dataset processed cache directory. For Sports with `n_codebook: 16`,
`codebook_size: 256`, and `sent_emb_pca: 512`, expected names are:

```text
text-embedding-3-large_fsq_m16_k256_pca512.sem_ids
text-embedding-3-large_fsq_quantile_m16_k256_pca512.sem_ids
text-embedding-3-large_pq_m16_k256_pca512.sem_ids
```

Each cache also gets a sidecar stats file:

```text
*.sem_ids.stats.json
```

Stats include:

- `collision_rate`
- `unique_full_codes`
- `per_digit_utilization`
- `per_digit_entropy`
- `per_digit_max_bucket`

Inspect these before training. They are the fastest check for a collapsed or
unhealthy tokenizer.

## How To Run

### 1. Prepare semantic IDs

Use a compute node on Snellius unless you are sure the `.sem_ids` files already
exist. These commands may run PCA, FAISS, or embedding generation.

```bash
python3 scripts/rpg_prepare_semantic_ids.py \
  --model RPGAblation \
  --config configs/rpg/tokenizer_ablation/sports_fsq.yaml

python3 scripts/rpg_prepare_semantic_ids.py \
  --model RPGAblation \
  --config configs/rpg/tokenizer_ablation/sports_fsq_quantile.yaml

python3 scripts/rpg_prepare_semantic_ids.py \
  --model RPGAblation \
  --config configs/rpg/tokenizer_ablation/sports_pq.yaml
```

If `text-embedding-3-large.sent_emb` is missing, preparation may need to encode
item text embeddings and can take much longer. Do not run that casually on a
login node.

### 2. Inspect tokenizer stats

Find the generated stats:

```bash
find artifacts/rpg/cache/AmazonReviews2014/Sports_and_Outdoors/processed \
  -name '*.stats.json' \
  -print
```

Then inspect:

```bash
python3 -m json.tool path/to/file.sem_ids.stats.json
```

Good signs:

- low collision rate,
- high per-digit utilization,
- high per-digit entropy,
- no single bucket dominating a digit.

Warning signs:

- many full-ID collisions,
- low entropy,
- low utilization,
- very large `per_digit_max_bucket`.

### 3. Train one method first

Start with `fsq_quantile`, because it is the easiest sanity check for balanced
scalar labels:

```bash
python3 scripts/rpg.py \
  --model RPGAblation \
  --config configs/rpg/tokenizer_ablation/sports_fsq_quantile.yaml
```

Then train `fsq` and `pq` if the pipeline works:

```bash
python3 scripts/rpg.py \
  --model RPGAblation \
  --config configs/rpg/tokenizer_ablation/sports_fsq.yaml

python3 scripts/rpg.py \
  --model RPGAblation \
  --config configs/rpg/tokenizer_ablation/sports_pq.yaml
```

Training should run through Slurm on a GPU partition. Use `gpu_mig` for short
debugging and `gpu_a100` for real runs.

### 4. Evaluate

Use the existing evaluation scripts with `--model RPGAblation` and the matching
config. Suggested comparisons:

- OPQ baseline RPG,
- `pq`,
- `fsq`,
- `fsq_quantile`.

Metrics to report:

- NDCG@10,
- Recall@10,
- eval-seed mean/std,
- cold-start NDCG@10,
- collision rate,
- per-digit entropy,
- per-digit utilization,
- inference/graph decoding behavior if relevant.

### 5. Cold Start And Stability

After one checkpoint exists, run the existing cold-start and eval-seed scripts
with `--model RPGAblation`. Keep configs matched to the checkpoint/tokenizer
used for training.

## Scientific Interpretation

The safest interpretation is:

- `pq` isolates the value of OPQ's rotation.
- `fsq` tests fixed finite scalar quantization on PCA latents.
- `fsq_quantile` tests whether balanced scalar labels help RPG.

Do not describe `fsq` as a trained FSQ-VAE. It is library-backed FSQ scalar
rounding over PCA latents.

If `fsq_quantile` has excellent utilization but poor recommendation metrics,
then marginal balance alone is not enough. If `fsq` has low entropy or collapsed
digits, that explains weak performance. If either scalar method is competitive,
that is interesting because it is simpler and more reproducible than OPQ.

## Next Steps

1. Generate Sports semantic-ID caches for `fsq`, `fsq_quantile`, and `pq`.
2. Inspect the stats JSON files before training.
3. Train one `fsq_quantile` Sports checkpoint as a smoke experiment.
4. Train `fsq` and `pq`.
5. Compare against the OPQ RPG baseline.
6. Only after this, consider a larger trained tokenizer such as FSQ-AE/VQ-VAE.

## Caveats

- The ablation configs require `--model RPGAblation`.
- The current FSQ method is not trained.
- `fsq_quantile` is not paper FSQ.
- Login nodes should not be used for cache generation unless the cache already
  exists and the command exits immediately.
- No files under `third_party/` are edited by this branch.
