"""Novelty-aware graph traversal diagnostic for RPG decoding.

B8 compares upstream RPG graph propagation against a small intervention that
prefers unvisited candidates when selecting the next frontier. The goal is to
test whether the late-step saturation in new visited nodes is caused by
reselecting already-seen nodes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from tqdm import tqdm

from perf.config import checkpoint_signature

from .dynamic import _configure_dynamic_budget, _metric_names
from .dynamic_trace import BatchTrace, traced_generate, traced_generate_visited_masked
from .runtime import build_harness_from_args
from .sessions import SessionPaths, append_or_update_manifest, latest_session, write_csv
from .static import load_prepared_graph

GenerateFn = Callable[[Any, dict[str, torch.Tensor], int], tuple[torch.Tensor, torch.Tensor, BatchTrace]]


def novelty_output_paths(paths: SessionPaths) -> dict[str, Path]:
    """Return B8 output paths under the dynamic analysis directory."""

    root = paths.dynamic / "novelty"
    summaries = root / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    return {
        "per_example_parquet": root / "per_example.parquet",
        "summary_csv": summaries / "novelty_summary.csv",
        "redundancy_csv": summaries / "novelty_redundancy_summary.csv",
        "summary_json": root / "novelty_summary.json",
    }


def novelty_modes_from_config(config: dict[str, Any]) -> list[str]:
    """Resolve traversal modes for B8."""

    modes = [str(mode) for mode in config.get("graph_analysis_novelty_modes", ["original", "visited_masked"])]
    allowed = {"original", "visited_masked"}
    invalid = sorted(set(modes) - allowed)
    if invalid:
        raise ValueError(f"Unsupported graph_analysis_novelty_modes: {invalid}")
    return modes


def novelty_n_edges_from_config(config: dict[str, Any], prepared_topk: int) -> list[int]:
    """Resolve graph-width settings for the novelty diagnostic."""

    raw_values = config.get("graph_analysis_novelty_n_edges", [30, 100])
    values = sorted({int(value) for value in raw_values})
    invalid = [value for value in values if value <= 0 or value > prepared_topk]
    if invalid:
        raise ValueError(f"graph_analysis_novelty_n_edges must be in [1, {prepared_topk}], got {invalid}")
    return values


def novelty_eval_seed_from_config(config: dict[str, Any]) -> int:
    """Use one eval seed to keep B8 within the intended short runtime."""

    seeds = config.get("graph_analysis_eval_seeds", [2024])
    if not seeds:
        raise ValueError("graph_analysis_eval_seeds cannot be empty for novelty analysis.")
    return int(seeds[0])


def _generate_for_mode(mode: str) -> GenerateFn:
    """Return the traced generator for one B8 mode."""

    if mode == "original":
        return traced_generate
    if mode == "visited_masked":
        return traced_generate_visited_masked
    raise ValueError(f"Unsupported novelty mode: {mode}")


def _first_reached_step(trace: BatchTrace, batch_index: int, target: int) -> int | None:
    """Return first cumulative visited step containing ``target``."""

    visited: set[int] = set()
    for step, new_items in enumerate(trace.new_items_by_step[batch_index]):
        visited.update(int(item) for item in new_items)
        if target in visited:
            return step
    return None


def _frontier_seen_ratios(trace: BatchTrace, batch_index: int) -> tuple[list[float], list[float]]:
    """Compute how much of each selected frontier was already seen before the step."""

    visited: set[int] = set()
    seen_ratios: list[float] = []
    new_ratios: list[float] = []
    for frontier, new_items in zip(
        trace.frontier_by_step[batch_index],
        trace.new_items_by_step[batch_index],
    ):
        frontier_set = {int(item) for item in frontier}
        if frontier_set:
            seen = len(frontier_set & visited)
            seen_ratios.append(seen / len(frontier_set))
            new_ratios.append(1.0 - seen / len(frontier_set))
        else:
            seen_ratios.append(0.0)
            new_ratios.append(0.0)
        visited.update(int(item) for item in new_items)
    return seen_ratios, new_ratios


def _row_from_trace(
    *,
    trace: BatchTrace,
    batch_index: int,
    user_index: int,
    user_raw_id: str,
    mode: str,
    eval_seed: int,
    n_edges: int,
    target: int,
    predictions: list[int],
    metric_values: dict[str, list[float]],
    metric_names: list[str],
    decode_seconds_per_user: float,
) -> dict[str, Any]:
    """Build one B8 per-example diagnostics row."""

    first_reached = _first_reached_step(trace, batch_index, target)
    frontier_seen, frontier_new = _frontier_seen_ratios(trace, batch_index)
    row: dict[str, Any] = {
        "user_index": user_index,
        "user_raw_id": user_raw_id,
        "mode": mode,
        "eval_seed": eval_seed,
        "n_edges": n_edges,
        "target_item_id": target,
        "target_reachable": first_reached is not None,
        "target_first_reached_step": first_reached,
        "target_selected": target in predictions,
        "n_visited_items": len(trace.final_visited_items[batch_index]),
        "decode_seconds_per_user": decode_seconds_per_user,
    }
    for metric in metric_names:
        row[metric] = float(metric_values[metric][batch_index])
    for step, value in enumerate(trace.visited_count_by_step[batch_index]):
        row[f"visited_count_step_{step}"] = int(value)
    for step, value in enumerate(trace.new_item_count_by_step[batch_index]):
        row[f"new_item_count_step_{step}"] = int(value)
    for step, value in enumerate(trace.unique_candidate_count_by_step[batch_index]):
        row[f"unique_candidate_count_step_{step}"] = int(value)
    for step, value in enumerate(trace.novelty_ratio_by_step[batch_index]):
        row[f"candidate_novelty_ratio_step_{step}"] = float(value)
    for step, value in enumerate(frontier_seen):
        row[f"frontier_seen_before_ratio_step_{step}"] = float(value)
    for step, value in enumerate(frontier_new):
        row[f"frontier_new_ratio_step_{step}"] = float(value)
    return row


def _collect_mode_rows(
    *,
    harness: Any,
    adjacency: torch.Tensor,
    mode: str,
    n_edges: int,
    eval_seed: int,
    user_ids: list[str],
    max_users: int,
    metric_names: list[str],
) -> list[dict[str, Any]]:
    """Run one B8 mode/graph-width setting and return per-example rows."""

    from genrec.utils import init_seed

    init_seed(eval_seed, harness.config["reproducibility"])
    _configure_dynamic_budget(harness, adjacency, n_edges)
    generate = _generate_for_mode(mode)

    rows: list[dict[str, Any]] = []
    user_offset = 0
    maxk = harness.trainer.evaluator.maxk
    progress = tqdm(
        harness.test_dataloader,
        total=len(harness.test_dataloader),
        desc=f"Novelty mode={mode} n_edges={n_edges}",
    )
    with torch.no_grad():
        for batch in progress:
            if user_offset >= max_users:
                break

            batch = {key: value.to(harness.accelerator.device) for key, value in batch.items()}
            start = time.perf_counter()
            preds, visited_counts, trace = generate(harness.model, batch, maxk)
            if torch.cuda.is_available():
                torch.cuda.synchronize(harness.accelerator.device)
            elapsed = time.perf_counter() - start

            results = harness.trainer.evaluator.calculate_metrics(
                (preds, visited_counts),
                batch["labels"],
            )

            batch_size = int(batch["labels"].shape[0])
            keep = min(batch_size, max_users - user_offset)
            labels = batch["labels"].detach().cpu().view(batch_size, -1)[:, 0].tolist()
            predictions = preds.detach().cpu().squeeze(-1).numpy().tolist()
            metric_values = {
                metric: results[metric].detach().cpu().view(-1).tolist()
                for metric in metric_names
            }
            seconds_per_user = elapsed / max(batch_size, 1)

            for batch_index in range(keep):
                user_index = user_offset + batch_index
                rows.append(
                    _row_from_trace(
                        trace=trace,
                        batch_index=batch_index,
                        user_index=user_index,
                        user_raw_id=user_ids[user_index],
                        mode=mode,
                        eval_seed=eval_seed,
                        n_edges=n_edges,
                        target=int(labels[batch_index]),
                        predictions=[int(item) for item in predictions[batch_index]],
                        metric_values=metric_values,
                        metric_names=metric_names,
                        decode_seconds_per_user=seconds_per_user,
                    )
                )
            user_offset += batch_size

    return rows


def _summarize_novelty(frame: pd.DataFrame, metric_names: list[str]) -> list[dict[str, Any]]:
    """Aggregate B8 recommendation and reachability metrics."""

    rows: list[dict[str, Any]] = []
    for (mode, n_edges), group in frame.groupby(["mode", "n_edges"], sort=True):
        row: dict[str, Any] = {
            "mode": mode,
            "n_edges": int(n_edges),
            "n_examples": int(len(group)),
            "reachable_rate": float(group["target_reachable"].mean()),
            "target_selected_rate": float(group["target_selected"].mean()),
            "mean_visited_items": float(group["n_visited_items"].mean()),
            "mean_decode_seconds_per_user": float(group["decode_seconds_per_user"].mean()),
        }
        for metric in metric_names:
            row[metric] = float(group[metric].mean())
        rows.append(row)
    return rows


def _summarize_redundancy(frame: pd.DataFrame, propagation_steps: int) -> list[dict[str, Any]]:
    """Aggregate step-wise exploration metrics for B8."""

    rows: list[dict[str, Any]] = []
    for (mode, n_edges), group in frame.groupby(["mode", "n_edges"], sort=True):
        for step in range(propagation_steps + 1):
            rows.append(
                {
                    "mode": mode,
                    "n_edges": int(n_edges),
                    "step": step,
                    "visited_count_mean": float(group[f"visited_count_step_{step}"].mean()),
                    "new_item_count_mean": float(group[f"new_item_count_step_{step}"].mean()),
                    "unique_candidate_count_mean": float(
                        group[f"unique_candidate_count_step_{step}"].mean()
                    ),
                    "candidate_novelty_ratio_mean": float(
                        group[f"candidate_novelty_ratio_step_{step}"].mean()
                    ),
                    "frontier_seen_before_ratio_mean": float(
                        group[f"frontier_seen_before_ratio_step_{step}"].mean()
                    ),
                    "frontier_new_ratio_mean": float(group[f"frontier_new_ratio_step_{step}"].mean()),
                }
            )
    return rows


def run_novelty(args: Any) -> int:
    """Run B8 novelty-aware traversal analysis."""

    harness = build_harness_from_args(args)
    if harness.config["use_ddp"]:
        raise RuntimeError("Graph-analysis novelty command only supports single-process evaluation.")

    paths = latest_session(harness.config, args.session_dir)
    adjacency, metadata = load_prepared_graph(paths, harness)
    prepared_topk = int(metadata["topk"])
    modes = novelty_modes_from_config(harness.config)
    n_edges_values = novelty_n_edges_from_config(harness.config, prepared_topk)
    eval_seed = novelty_eval_seed_from_config(harness.config)
    max_users = int(harness.config.get("graph_analysis_novelty_max_users", 5000))
    if max_users <= 0:
        raise ValueError("graph_analysis_novelty_max_users must be positive.")

    metric_names = _metric_names(harness.config)
    test_split = harness.dataset.split()["test"]
    user_ids = [str(user) for user in test_split["user"]]
    max_users = min(max_users, len(user_ids))

    rows: list[dict[str, Any]] = []
    for n_edges in n_edges_values:
        for mode in modes:
            rows.extend(
                _collect_mode_rows(
                    harness=harness,
                    adjacency=adjacency,
                    mode=mode,
                    n_edges=n_edges,
                    eval_seed=eval_seed,
                    user_ids=user_ids,
                    max_users=max_users,
                    metric_names=metric_names,
                )
            )

    frame = pd.DataFrame(rows)
    outputs = novelty_output_paths(paths)
    frame.to_parquet(outputs["per_example_parquet"], index=False)
    summary_rows = _summarize_novelty(frame, metric_names)
    redundancy_rows = _summarize_redundancy(frame, int(harness.model.propagation_steps))
    write_csv(outputs["summary_csv"], summary_rows)
    write_csv(outputs["redundancy_csv"], redundancy_rows)

    summary_payload = {
        "session_root": str(paths.root),
        "checkpoint_path": str(harness.checkpoint_path),
        "checkpoint_signature": checkpoint_signature(harness.checkpoint_path),
        "prepared_graph_topk": prepared_topk,
        "modes": modes,
        "n_edges_values": n_edges_values,
        "eval_seed": eval_seed,
        "max_users": max_users,
        "num_beams": int(harness.model.num_beams),
        "propagation_steps": int(harness.model.propagation_steps),
        "temperature": float(harness.model.temperature),
        "metrics": metric_names,
        "summary": summary_rows,
    }
    outputs["summary_json"].write_text(json.dumps(summary_payload, indent=2, sort_keys=True))
    append_or_update_manifest(
        paths,
        {
            "session_root": str(paths.root),
            "novelty_outputs": {key: str(value) for key, value in outputs.items()},
        },
    )

    print(paths.root)
    return 0
