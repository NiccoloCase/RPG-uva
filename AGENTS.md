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
