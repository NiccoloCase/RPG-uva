# Jobs Layout Policy

This directory contains Slurm job definitions only.

## Submission rule

Always submit a job from its own folder.

Before creating or changing Snellius jobs, check the UvA guide:

`https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial1/Lisa_Cluster.html`

For GPU jobs on this project, always request an explicit partition with `#SBATCH --partition=...`.
For the course setup, both `gpu_a100` and `gpu_h100` are available. Match `--cpus-per-task`
and `--mem` to the chosen partition instead of relying on implicit defaults.

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

Examples:

```bash
cd /gpfs/home6/$USER/RPG/jobs/reproduction/perf
sbatch ./build_graphs.sh /gpfs/home6/$USER/RPG/artifacts/rpg/ckpt/model.pth
sbatch ./profile_inference.sh /gpfs/home6/$USER/RPG/artifacts/rpg/ckpt/model.pth
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
