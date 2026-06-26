# New-Dataset Jobs

This tree holds runs on Amazon Reviews 2014 categories that are not part of
the original paper (Beauty/Sports/Toys/CDs), kept separate from
`jobs/reproduction/`. Categories currently present:

- `video_games`
- `pet_supplies`

The full pipeline has five stages, run in this order. Each stage's job
scripts live under `jobs/new_datasets/<stage>/<dataset>/` and must be
submitted from that directory (relative output paths depend on it):

1. **RPG semantic-ID prep** — `jobs/new_datasets/rpg/<dataset>/prepare_semantic_ids.sh`
   Downloads the raw category data, generates sentence embeddings via the
   OpenAI `text-embedding-3-large` API, and builds the semantic-ID cache.
   See `jobs/new_datasets/rpg/README.md` for details and API key setup.

2. **RPG training** — `jobs/new_datasets/rpg/<dataset>/train.sh`
   Trains the RPG model; checkpoint written to `artifacts/rpg/ckpt/`.

3. **RPG multi-seed eval** — `jobs/new_datasets/eval_seeds/<dataset>/run_<dataset>.sh`
   Runs `scripts/rpg_eval_seeds.py` over 10 seeds (2024-2033) against the
   latest matching RPG checkpoint. Results (per-seed metrics + bootstrap CI)
   are written to
   `artifacts/rpg/eval_seeds/new_datasets/<dataset>/<timestamp>_job<id>/summary.json`.

4. **SASRec data prep + training** —
   `jobs/new_datasets/sasrec/<dataset>/prepare_data.sh` then `train.sh`.
   Checkpoint written to `artifacts/sasrec/ckpt/sasrec_<dataset>.pt`.

5. **SASRec training + multi-seed eval** —
   `jobs/new_datasets/sasrec/<dataset>/train.sh`, then
   `jobs/new_datasets/sasrec/eval_seeds/<dataset>/run_<dataset>.sh`.
   Training writes `artifacts/sasrec/ckpt/sasrec_<dataset>.pt`.
   The eval_seeds runner evaluates that checkpoint over 10 seeds via
   `scripts/sasrec_eval.py` and writes
   `artifacts/sasrec/eval_seeds/new_datasets/<dataset>/<timestamp>_job<id>/summary.json`.
   (`jobs/new_datasets/sasrec/<dataset>/eval.sh` runs a single-seed eval of
   the same SASRec checkpoint instead, if needed.)

Optional cold-start follow-up jobs:

- RPG: `jobs/new_datasets/rpg/cold_start/run_cold_start_<dataset>.sh`
- SASRec: `jobs/new_datasets/sasrec/cold_start/run_cold_start_<dataset>.sh`

Example, running stage 3 for `video_games`:

```bash
cd /gpfs/home6/$USER/RPG-uva/jobs/new_datasets/eval_seeds/video_games
sbatch ./run_video_games.sh
```

Both `video_games` and `pet_supplies` have completed all five stages at
least once; see the `summary.json` paths above for results.
