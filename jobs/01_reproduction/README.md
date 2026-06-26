# 01 Reproduction

This section covers the paper's baseline reproduction runs:

- `jobs/reproduction/rpg/<dataset>/train.sh`
- `jobs/reproduction/rpg/<dataset>/eval.sh`
- `jobs/reproduction/sasrec/<dataset>/prepare_data.sh`
- `jobs/reproduction/sasrec/<dataset>/train.sh`
- `jobs/reproduction/sasrec/<dataset>/eval.sh`

Paper datasets:

- `beauty`
- `cds_and_vinyl`
- `sports_and_outdoors`
- `toys_and_games`

Submit each job from its own directory.

Convenience wrapper:

```bash
cd jobs/01_reproduction
DATASET=beauty MODEL=both bash ./submit_dataset.sh
```
