"""Shared constants for RPG graph analysis.

The analysis code follows the RPG item-id convention used by the upstream
project: item id ``0`` is padding, and real items are in ``[1, n_items)``.
All static graph metrics exclude padding and remove self-edges before reporting
edge-level or graph-level summaries.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_K_VALUES = [10, 20, 30, 50, 100]
DEFAULT_RANDOM_SEEDS = [2024, 2025, 2026]

# Buckets used by A7 popularity-bias analysis. Values are inclusive.
POPULARITY_BUCKETS = [
    ("0-5", 0, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51+", 51, None),
]

