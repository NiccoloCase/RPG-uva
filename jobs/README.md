# Jobs Layout Policy

This directory contains Slurm job definitions only.

## Submission rule

Always submit a job from its own folder.

Example:

```bash
cd jobs/init/env
sbatch ./setup_env.sh
```

Do not submit from the repo root or from `$HOME`.

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
