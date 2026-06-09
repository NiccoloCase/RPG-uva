# SASRec Reproduction Jobs

This tree is model-first so SASRec runs stay separate from the existing RPG jobs.

Layout:

- `jobs/reproduction/sasrec/<dataset>/train.sh`
- `jobs/reproduction/sasrec/<dataset>/eval.sh`
- `output/reproduction/sasrec/<dataset>/`

Datasets currently wired:

- `beauty`
- `cds_and_vinyl`
- `sports_and_outdoors`
- `toys_and_games`

Submit from the dataset folder itself:

```bash
cd /gpfs/home6/$USER/RPG-uva/jobs/reproduction/sasrec/beauty
sbatch ./train.sh
sbatch ./eval.sh
sbatch ./eval.sh /gpfs/home6/$USER/RPG-uva/artifacts/sasrec/ckpt/sasrec_repro_beauty-Jun-09-2026_12-00-abc123.pth
```

Notes:

- `train.sh` uses `configs/sasrec/root.yaml` plus the dataset config under `configs/sasrec/repro/`.
- `eval.sh` autodiscovers the newest matching checkpoint in `artifacts/sasrec/ckpt/` when no absolute checkpoint path is passed.
