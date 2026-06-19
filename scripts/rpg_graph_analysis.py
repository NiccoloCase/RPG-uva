#!/usr/bin/env python3
"""Thin launcher for RPG graph analysis.

The implementation lives in the ``rpg_graph_analysis`` package so it can be
read, tested, and extended module by module.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rpg_graph_analysis.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
