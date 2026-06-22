"""Popularity-bias metrics shared by the RPG and SASRec eval scripts.

Implements three metrics commonly used to characterize popularity bias in
top-K recommendation, computed per user and aggregated the same way as the
existing recall/NDCG metrics (mean over users, with bootstrap CIs):

- ARP@K (Average Recommendation Popularity): the average training-set
  popularity of the items recommended to a user.
- APLT@K (Average Percentage of Long-Tail items): the fraction of a user's
  top-K recommendations that fall in the "long tail" item set.
- Per-group NDCG: NDCG@K reported separately for user groups defined by how
  popularity-skewed each user's own interaction history is.

References:
- Abdollahpouri, Burke & Mobasher (2017), "Controlling Popularity Bias in
  Learning-to-Rank Recommendation", RecSys. Defines the short-head / long-tail
  item split used here (top fraction of items by interaction count).
- Abdollahpouri, Mansoury, Burke & Mobasher (2019), "The Unfairness of
  Popularity Bias in Recommendation", RMSE@RecSys. Defines ARP and APLT.
- Abdollahpouri, Burke & Mobasher (2019), "Managing Popularity Bias in
  Recommender Systems with Personalized Re-ranking", FLAIRS, and
  Abdollahpouri et al. (2021), "User-centered Evaluation of Popularity Bias
  in Recommender Systems", UMAP. Define user groups by the average
  popularity of items in each user's profile (here: "niche", "diverse",
  "blockbuster_focused").
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from statistics import mean
from typing import Any

ItemPopularity = dict[int, int]

POPULARITY_GROUPS: tuple[str, str, str] = ("niche", "diverse", "blockbuster_focused")


def compute_item_popularity(item_sequences: Iterable[Sequence[int]]) -> ItemPopularity:
    """Count how often each item occurs across the given (training) sequences."""
    counts: Counter[int] = Counter()
    for seq in item_sequences:
        counts.update(int(item) for item in seq)
    return dict(counts)


def compute_long_tail_items(item_popularity: ItemPopularity, short_head_fraction: float = 0.2) -> set[int]:
    """Return the set of "long tail" items, i.e. all items outside the short head.

    The short head is the top `short_head_fraction` of items by training-set
    popularity (Abdollahpouri et al., 2017).
    """
    if not 0.0 < short_head_fraction < 1.0:
        raise ValueError("short_head_fraction must be in (0, 1).")
    if not item_popularity:
        return set()

    ranked = sorted(item_popularity.items(), key=lambda kv: kv[1], reverse=True)
    n_short_head = max(1, round(len(ranked) * short_head_fraction))
    short_head = {item_id for item_id, _ in ranked[:n_short_head]}
    return set(item_popularity) - short_head


def recommendation_popularity(topk_items: Sequence[int], item_popularity: ItemPopularity) -> float:
    """ARP for a single user: mean training-set popularity of the top-K recommendations."""
    if not topk_items:
        return float("nan")
    return mean(item_popularity.get(int(item_id), 0) for item_id in topk_items)


def percentage_long_tail(topk_items: Sequence[int], long_tail_items: set[int]) -> float:
    """APLT for a single user: fraction of the top-K recommendations in the long tail."""
    if not topk_items:
        return float("nan")
    return mean(1.0 if int(item_id) in long_tail_items else 0.0 for item_id in topk_items)


def compute_profile_popularity(profile_items: Sequence[int], item_popularity: ItemPopularity) -> float:
    """Average training-set popularity of the items in a user's own history."""
    if not profile_items:
        return float("nan")
    return mean(item_popularity.get(int(item_id), 0) for item_id in profile_items)


def assign_popularity_groups(
    profile_popularity: dict[int, float],
    low_quantile: float = 0.2,
    high_quantile: float = 0.8,
) -> dict[int, str]:
    """Bucket users into popularity-profile groups via quantile cuts.

    Users whose profile popularity falls in the bottom `low_quantile` of the
    distribution are "niche" (mostly long-tail history), users in the top
    `1 - high_quantile` are "blockbuster_focused" (mostly short-head
    history), and the rest are "diverse".
    """
    if not 0.0 <= low_quantile < high_quantile <= 1.0:
        raise ValueError("Require 0 <= low_quantile < high_quantile <= 1.")

    import numpy as np

    values = [value for value in profile_popularity.values() if value == value]  # drop NaNs
    if not values:
        return {user_index: POPULARITY_GROUPS[1] for user_index in profile_popularity}

    low_cut, high_cut = np.quantile(values, [low_quantile, high_quantile])

    groups: dict[int, str] = {}
    for user_index, value in profile_popularity.items():
        if value != value:  # NaN: no profile history, default to "diverse"
            groups[user_index] = POPULARITY_GROUPS[1]
        elif value <= low_cut:
            groups[user_index] = POPULARITY_GROUPS[0]
        elif value >= high_cut:
            groups[user_index] = POPULARITY_GROUPS[2]
        else:
            groups[user_index] = POPULARITY_GROUPS[1]
    return groups


def popularity_metric_names(topk_values: Iterable[int]) -> list[str]:
    """Metric-name suffixes for ARP/APLT at each K, matching the recall@K/ndcg@K convention."""
    names: list[str] = []
    for k in topk_values:
        names.append(f"arp@{k}")
        names.append(f"aplt@{k}")
    return names


def group_metric_summary(
    rows: list[dict[str, Any]],
    user_groups: dict[int, str],
    metric_names: list[str],
    bootstrap_ci: Callable[[list[float]], tuple[float, float]],
    ci_level: float,
    group_order: Iterable[str] = POPULARITY_GROUPS,
) -> list[dict[str, Any]]:
    """Per-group counterpart of the overall metric summary.

    For each (group, metric) pair, averages each user's per-eval-seed mean
    and reports a bootstrap CI over users within that group, mirroring the
    overall `_metric_summary` aggregation used for recall/NDCG.
    """
    rows_by_group_user_metric: dict[str, dict[str, dict[int, list[float]]]] = {
        group: {metric: {} for metric in metric_names} for group in group_order
    }
    for row in rows:
        user_index = int(row["user_index"])
        group = user_groups.get(user_index)
        if group is None or group not in rows_by_group_user_metric:
            continue
        for metric in metric_names:
            rows_by_group_user_metric[group][metric].setdefault(user_index, []).append(float(row[metric]))

    summary_rows: list[dict[str, Any]] = []
    for group in group_order:
        for metric in metric_names:
            per_user_seed_means = [
                float(mean(values)) for _, values in sorted(rows_by_group_user_metric[group][metric].items())
            ]
            ci_low, ci_high = bootstrap_ci(per_user_seed_means)
            summary_rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "n_users": len(per_user_seed_means),
                    "mean": float(mean(per_user_seed_means)) if per_user_seed_means else float("nan"),
                    "ci_level": ci_level,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return summary_rows
