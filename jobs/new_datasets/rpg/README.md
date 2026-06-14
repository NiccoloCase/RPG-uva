# RPG New-Dataset Jobs

This tree holds RPG runs on Amazon Reviews 2014 categories that are not part
of the original paper (Beauty/Sports/Toys/CDs), kept separate from
`jobs/reproduction/`:

- `jobs/new_datasets/rpg/<dataset>/...`
- `output/new_datasets/rpg/...`
- `configs/rpg/new_datasets/<dataset>.yaml`

For the full pipeline (RPG + SASRec, training + multi-seed eval) and stage
ordering, see `jobs/new_datasets/README.md`.

These datasets have no pre-existing cache, so run `prepare_semantic_ids.sh` first to
download the raw data, generate sentence embeddings, and build the
semantic-ID cache before submitting `train.sh`:

```bash
cd /gpfs/home6/$USER/RPG-uva/jobs/new_datasets/rpg/video_games
sbatch ./prepare_semantic_ids.sh
# wait for it to finish, then:
sbatch ./train.sh
```

Generating sentence embeddings calls the OpenAI `text-embedding-3-large` API
once per item, so make sure `openai_api_key` is set in `configs/rpg/local.yaml`
before running `prepare_semantic_ids.sh`.

Hyperparameters for these two presets have no paper anchor (the paper only
covers Beauty/Sports/Toys/CDs); `video_games` reuses the
`sports_and_outdoors` config (closest scale) and `pet_supplies` reuses the
`toys_and_games` config. Revisit after the first run if results look off.
