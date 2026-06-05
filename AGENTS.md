# Agent Guidelines

## Third-Party Dependencies

The `third_party/` directory contains **Git submodules** — external repositories pinned to specific commits. These are not part of this codebase and must not be modified directly.

### Rules

- **Do not edit, create, or delete files** inside `third_party/` or any of its subdirectories.
- **Do not run `git add` or `git commit`** on anything inside `third_party/`.
- **Do not upgrade or change submodule pointers** unless explicitly instructed by a human.
- If a task requires changes to a third-party library, flag it to the user instead of modifying the submodule directly.

### How to use third-party code

You may **read** and **import** from `third_party/` freely. Only direct modification is prohibited.

```python
# OK — importing from a submodule
from third_party.some_lib import SomeClass

# NOT OK — editing files inside third_party/
# third_party/some_lib/module.py  ← do not touch
```

### Initialising submodules

If `third_party/` appears empty after cloning, run:

```bash
git submodule update --init --recursive
```

## Job Submission Policy

This repo uses a strict split between job definitions, scheduler logs, and runtime artifacts.

Snellius reference guide for job creation and partition usage:

`https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial1/Lisa_Cluster.html`

### Rules

- Submit Slurm jobs from the job's own directory under `jobs/`.
- Do not submit jobs from the repo root, `$HOME`, or any other directory.
- Keep job scripts under `jobs/`, preserving a meaningful tree such as `jobs/init/env/` or `jobs/reproduction/beauty/`.
- Write scheduler stdout/stderr logs under `output/`, mirroring the `jobs/` tree. Example:
  - job script: `jobs/init/env/setup_env.sh`
  - log dir: `output/init/env/`
- Write artifacts such as checkpoints, caches, tensorboard files, generated data, and result files under the repo-root `artifacts/` tree.
- Do not write job logs into `jobs/` or artifacts into `output/`.

### Expectations for agents

- Before changing or adding a job, read this file and keep the layout above intact.
- For Snellius GPU jobs, request an explicit partition such as `gpu_a100` or `gpu_h100` and size CPU and host-memory requests to that partition.
- Job scripts should fail early if they are launched from the wrong working directory.
- When adding a new job folder under `jobs/...`, create the matching `output/...` directory shape as needed.
