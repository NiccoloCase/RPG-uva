"""Shared constants for RPG graph analysis.

The analysis code follows the RPG item-id convention used by the upstream
project: item id ``0`` is padding, and real items are in ``[1, n_items)``.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RANDOM_SEEDS = [2024, 2025, 2026]

