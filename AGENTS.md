# Agent Guidelines

This file defines the repo-local rules that matter most for safe changes and reproducible Snellius runs.

## 1. Third-Party Code

### 1.1 What `third_party/` is

The `third_party/` directory contains **Git submodules** pinned to external commits. They are dependencies, not repo-owned code.

### 1.2 Rules

- **Do not edit, create, or delete files** inside `third_party/` or any of its subdirectories.
- **Do not run `git add` or `git commit`** on anything inside `third_party/`.
- **Do not upgrade or change submodule pointers** unless explicitly instructed by a human.
- If a task requires changes to a third-party library, flag it to the user instead of modifying the submodule directly.

### 1.3 Allowed usage

You may **read** and **import** from `third_party/` freely. Only direct modification is prohibited.

```python
# OK — importing from a submodule
from third_party.some_lib import SomeClass

# NOT OK — editing files inside third_party/
# third_party/some_lib/module.py  ← do not touch
```

### 1.4 Submodule initialization

If `third_party/` appears empty after cloning, run:

```bash
git submodule update --init --recursive
```

## 2. Snellius

### 2.1 Reference guide

Use the local course guide whenever you create or change Snellius jobs:

- `docs/snellius/Snellius_Practical_Guide.pdf`
- `docs/snellius/Snellius_Practical_Guide.txt`

Keep only the operational rules that affect this repo:

- Login nodes are for editing, inspecting files, and submitting jobs.
- Training, evaluation, indexing, and other heavy work must run through Slurm on compute nodes.
- Test small before scaling up to long or expensive GPU jobs.

### 2.2 GPU partitions

For Snellius GPU jobs in this repo:

- Always request an explicit partition with `#SBATCH --partition=...`.
- Prefer `gpu_mig` for short GPU debugging/tests, `gpu_a100` for real runs, and `gpu_h100` only when clearly needed.
- Size `--cpus-per-task` and `--mem` to the chosen partition instead of relying on implicit defaults.

### 2.3 Workspace paths

For this repo on Snellius:

- Do not hard-code `/home/$USER/...` paths.
- Use the repo workspace under `/gpfs/home6/$USER/RPG`, or derive paths from the script location.
- Keep large shared datasets under `/projects/prjs2120/datasets`.
- Keep important shared outputs under `/projects/prjs2120/groups/group_16`.
- Use `/scratch-shared/$USER` only for temporary and reproducible intermediates.
- Do not keep important checkpoints, logs, or final results only in scratch.

## 3. Jobs

### 3.1 Submission policy

This repo uses a strict split between job definitions, scheduler logs, and runtime artifacts.

- Submit Slurm jobs from the job's own directory under `jobs/`.
- Do not submit jobs from the repo root, `$HOME`, or any other directory.
- Keep job scripts under `jobs/`, preserving a meaningful tree such as `jobs/init/env/` or `jobs/reproduction/beauty/`.
- Job scripts should fail early if they are launched from the wrong working directory.

### 3.2 Directory contract

- `jobs/...`: job scripts and small job-local helper files.
- `output/...`: scheduler logs, mirroring the `jobs/...` tree.
- `artifacts/...`: runtime outputs such as checkpoints, caches, tensorboard files, generated data, and result files.

Example:

```text
jobs/init/env/setup_env.sh
output/init/env/
artifacts/
```

### 3.3 Logging and artifacts

- Write scheduler stdout/stderr logs under `output/`, mirroring the `jobs/` tree.
- Do not write job logs into `jobs/`.
- Do not write runtime artifacts into `output/`.
- When adding a new job folder under `jobs/...`, create the matching `output/...` directory shape as needed.
- When a run also needs long-term shared storage outside the repo, copy or sync the important outputs to `/projects/prjs2120/groups/group_16`.

Example:

- job script: `jobs/init/env/setup_env.sh`
- log dir: `output/init/env/`

### 3.4 Agent expectations

- Before changing or adding a job, read this file and keep the layout above intact.
- Prefer checked-in job scripts over ad hoc `sbatch` commands when the workflow is repeated.
- For Snellius GPU jobs, request an explicit partition such as `gpu_a100` or `gpu_h100`.

## 4. Job Path Conventions

Use real filesystem paths in all job scripts, `sbatch` examples, and wrapped commands.

### 4.1 Rules

- Do not use angle-bracket placeholders such as `<you>`, `<checkpoint>`, or similar tokens in shell commands.
- Prefer deriving `REPO_ROOT` from `SLURM_SUBMIT_DIR` or `BASH_SOURCE` inside checked-in job scripts.
- When a job takes a checkpoint or config path, require a real absolute path and fail early if the file does not exist.
- Use `/projects/prjs2120/groups/group_16/...` for stable shared paths and `/scratch-shared/$USER/...` only for temporary paths.

### 4.2 Example

```bash
cd /gpfs/home6/$USER/RPG/jobs/reproduction/perf
sbatch ./build_graphs.sh /gpfs/work5/0/prjs2120/groups/group_16/artifacts/rpg/ckpt/model.pth
```
