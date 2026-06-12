# RPG Reproduction Jobs

This tree is model-first so all RPG reproduction jobs live under one root:

- `jobs/reproduction/rpg/<dataset>/...`
- `jobs/reproduction/rpg/cold_start/...`
- `jobs/reproduction/rpg/perf/...`
- `output/reproduction/rpg/...`

Dataset jobs currently present:

- `beauty`
- `cds_and_vinyl`
- `sports_and_outdoors`
- `toys_and_games`

Submit from the job directory itself, for example:

```bash
cd /gpfs/home6/$USER/RPG-uva/jobs/reproduction/rpg/beauty
sbatch ./train.sh
sbatch ./eval.sh
```

For datasets not covered by the paper (e.g. Video_Games, Pet_Supplies), see
`jobs/new_datasets/rpg/`.
