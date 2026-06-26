# Jobs Layout Policy

This directory contains Slurm job definitions only.

## Paper index

For a paper-aligned entrypoint, start with:

- `jobs/01_reproduction/`
- `jobs/02_accuracy_and_fairness/`
- `jobs/03_graph_structure_and_dynamics/`
- `jobs/04_search_vs_scorer/`
- `jobs/05_efficiency/`

The runnable job scripts still live under the implementation trees such as
`jobs/reproduction/rpg/`, `jobs/reproduction/sasrec/`, and `jobs/new_datasets/`.

## Submission rule

Always submit a job from its own folder.

Before creating or changing Snellius jobs, check the local course guide:

`docs/snellius/Snellius_Practical_Guide.pdf`
or
`docs/snellius/Snellius_Practical_Guide.txt`

Snellius rules that matter for this repo:

- Use login nodes only to edit, inspect files, and submit jobs.
- Run training, evaluation, and other heavy work only through Slurm.
- Start with a short test job before submitting a long GPU run.
- Always request an explicit partition with `#SBATCH --partition=...`.
- Prefer `gpu_mig` for short GPU tests, `gpu_a100` for normal full runs, and `gpu_h100` only when necessary.
- Match `--cpus-per-task` and `--mem` to the chosen partition instead of relying on implicit defaults.

Example:

```bash
cd jobs/init/env
sbatch ./setup_env.sh
```

Do not submit from the repo root or from `$HOME`.

## Path conventions

Use real filesystem paths in job arguments and wrapped commands. Do not leave angle-bracket placeholders such as `<you>` or `<checkpoint>` in shell input, because Bash interprets them as redirections and the job fails before the workload starts.

Preferred rules:

- Submit the checked-in scripts from their own directory under `jobs/...`.
- Let the scripts derive `REPO_ROOT` from `SLURM_SUBMIT_DIR`; do not hard-code `/home/.../RPG-uva`.
- Pass checkpoint and config inputs as real absolute paths.
- On Snellius for this repo, the workspace root is `/gpfs/home6/$USER/RPG`, not `/home/$USER/...`.
- Keep shared datasets under `/projects/prjs2120/datasets`; do not duplicate them into repo or home storage.
- Keep important shared outputs under `/projects/prjs2120/groups/group_16`.
- Use `/scratch-shared/$USER` only for temporary intermediates that can be regenerated.
- Do not leave the only copy of final checkpoints or results in scratch.

Examples:

```bash
cd /gpfs/home6/$USER/RPG/jobs/reproduction/perf
sbatch ./build_graphs.sh /gpfs/work5/0/prjs2120/groups/group_16/artifacts/rpg/ckpt/model.pth
sbatch ./profile_inference.sh /gpfs/work5/0/prjs2120/groups/group_16/artifacts/rpg/ckpt/model.pth
```

## Directory contract

- `jobs/...`: job scripts and small job-local helper files.
- `output/...`: scheduler logs, mirroring the `jobs/...` path without the leading `jobs/`.
- `artifacts/...`: runtime outputs and results at repo root.

For example:

```text
jobs/init/env/setup_env.sh
output/init/env/
artifacts/
```

## Logging rule

For a job at `jobs/a/b/job.sh`, stdout and stderr belong in:

```text
output/a/b/
```

## Artifact rule

Artifacts do not live next to the job script. They belong under the repo-root `artifacts/` tree. If a job needs a dedicated area, create a stable subpath there.

If the outputs must persist outside the repo or be shared with the course group, copy or sync the selected results to `/projects/prjs2120/groups/group_16`.
