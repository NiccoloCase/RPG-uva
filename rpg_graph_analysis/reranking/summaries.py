"""Summary tables for RPG reranking experiments."""

from __future__ import annotations

from typing import Any

import pandas as pd


def summarize_reranking(frame: pd.DataFrame, metric_names: list[str]) -> list[dict[str, Any]]:
    """Aggregate original-vs-reranked metrics by intervention setting."""

    rows: list[dict[str, Any]] = []
    group_cols = ["n_edges", "propagation_steps", "pool_size", "rerank_method", "hybrid_alpha"]
    for keys, group in frame.groupby(group_cols, sort=True):
        n_edges, propagation_steps, pool_size, rerank_method, hybrid_alpha = keys
        row: dict[str, Any] = {
            "n_edges": int(n_edges),
            "propagation_steps": int(propagation_steps),
            "pool_size": str(pool_size),
            "rerank_method": str(rerank_method),
            "hybrid_alpha": float(hybrid_alpha),
            "n_examples": int(len(group)),
            "target_reachable_rate": float(group["target_reachable"].mean()),
            "original_selected_rate": float(group["original_target_selected"].mean()),
            "rerank_selected_rate": float(group["rerank_target_selected"].mean()),
            "reachable_but_original_not_selected_rate": float(
                (group["target_reachable"] & ~group["original_target_selected"]).mean()
            ),
            "reachable_but_rerank_not_selected_rate": float(
                (group["target_reachable"] & ~group["rerank_target_selected"]).mean()
            ),
            "mean_visited_items": float(group["n_visited_items"].mean()),
            "mean_candidate_pool_size": float(group["candidate_pool_size"].mean()),
            "mean_rerank_time_ms": float(group["rerank_time_ms"].mean()),
        }
        for metric in metric_names:
            row[f"original_{metric}"] = float(group[f"original_{metric}"].mean())
            row[f"rerank_{metric}"] = float(group[f"rerank_{metric}"].mean())
            row[f"delta_{metric}"] = row[f"rerank_{metric}"] - row[f"original_{metric}"]
        rows.append(row)
    return rows
