"""Popularity-bias metrics for static graph analysis.

A7 compares graph in-degree against item frequency in the training split. This
module deliberately counts training interactions only; validation and test
labels are not used to define popularity.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats

from .settings import POPULARITY_BUCKETS


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    """Return Spearman correlation, or ``None`` values for constant inputs."""

    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return None, None
    result = stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    """Return Pearson correlation, or ``None`` values for constant inputs."""

    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return None, None
    result = stats.pearsonr(x, y)
    return float(result.statistic), float(result.pvalue)


def train_frequencies(dataset: Any) -> np.ndarray:
    """Count item occurrences in the upstream training split only."""

    frequencies = np.zeros(dataset.n_items, dtype=np.int64)
    train_split = dataset.split()["train"]
    for item_seq in train_split["item_seq"]:
        for raw_item in item_seq:
            item_id = dataset.item2id[raw_item]
            if item_id > 0:
                frequencies[item_id] += 1
    return frequencies


def bucket_for_frequency(value: int) -> str:
    """Assign an item training frequency to the configured popularity bucket."""

    for label, low, high in POPULARITY_BUCKETS:
        if value >= low and (high is None or value <= high):
            return label
    raise ValueError(f"No popularity bucket for value {value}")


def popularity_rows(
    k: int,
    frequencies: np.ndarray,
    indegree: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute A7 popularity-bias summary and bucket rows."""

    real_freq = frequencies[1:].astype(np.float64)
    real_indegree = indegree.astype(np.float64)
    spearman_r, spearman_p = safe_spearman(real_freq, real_indegree)
    pearson_r, pearson_p = safe_pearson(np.log1p(real_freq), np.log1p(real_indegree))

    n_items = real_freq.shape[0]
    top_count = max(1, int(math.ceil(n_items * 0.01)))
    top_indices = np.argsort(real_indegree)[-top_count:]
    total_interactions = float(real_freq.sum())
    top_freq = real_freq[top_indices]

    summary_row = {
        "k": k,
        "spearman_train_frequency_indegree": spearman_r,
        "spearman_pvalue": spearman_p,
        "pearson_log1p_train_frequency_log1p_indegree": pearson_r,
        "pearson_pvalue": pearson_p,
        "top_1pct_hub_count": top_count,
        "top_1pct_hub_train_frequency_mean": float(top_freq.mean()) if top_freq.size else float("nan"),
        "top_1pct_hub_train_frequency_median": (
            float(np.median(top_freq)) if top_freq.size else float("nan")
        ),
        "top_1pct_hub_training_interaction_share": (
            float(top_freq.sum() / total_interactions) if total_interactions else 0.0
        ),
        "all_items_train_frequency_mean": float(real_freq.mean()) if real_freq.size else float("nan"),
        "all_items_train_frequency_median": (
            float(np.median(real_freq)) if real_freq.size else float("nan")
        ),
    }

    bucket_rows: list[dict[str, Any]] = []
    bucket_labels = np.asarray([bucket_for_frequency(int(value)) for value in real_freq])
    for label, low, high in POPULARITY_BUCKETS:
        mask = bucket_labels == label
        values = real_indegree[mask]
        freqs = real_freq[mask]
        bucket_rows.append(
            {
                "k": k,
                "bucket": label,
                "bucket_min": low,
                "bucket_max": high,
                "n_items": int(mask.sum()),
                "train_frequency_mean": float(freqs.mean()) if freqs.size else float("nan"),
                "indegree_mean": float(values.mean()) if values.size else float("nan"),
                "indegree_median": float(np.median(values)) if values.size else float("nan"),
                "indegree_p90": float(np.percentile(values, 90)) if values.size else float("nan"),
                "indegree_max": int(values.max()) if values.size else 0,
            }
        )
    return summary_row, bucket_rows

