"""Lightweight beam-pruning diagnostic for RPG graph decoding.

Experiment B7 asks where reachable targets are lost:

* not reached by graph traversal;
* reached/considered but never kept in the beam;
* kept in the beam but not selected in the final top-k;
* selected.

The command runs this diagnostic on a bounded test subset while sweeping
``num_beams``. It reuses the prepared graph cache and the traced RPG decoder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from perf.config import checkpoint_signature

from .dynamic import (
    _assert_tracing_parity,
    _configure_dynamic_budget,
    _metric_names,
)
from .dynamic_trace import BatchTrace, traced_generate
from .runtime import build_harness_from_args
from .sessions import SessionPaths, append_or_update_manifest, latest_session, write_csv
from .static import load_prepared_graph


def pruning_output_paths(paths: SessionPaths) -> dict[str, Path]:
    """Return B7 output paths under the dynamic analysis directory."""

    root = paths.dynamic / "pruning"
    summaries = root / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    return {
        "per_example_parquet": root / "per_example.parquet",
        "summary_csv": summaries / "pruning_summary.csv",
        "summary_json": root / "pruning_summary.json",
    }


def pruning_num_beams_from_config(config: dict[str, Any]) -> list[int]:
    """Resolve the beam-size sweep used by the pruning diagnostic."""

    raw_values = config.get("graph_analysis_pruning_num_beams", [50, 100, 200, 500])
    values = sorted({int(value) for value in raw_values})
    invalid = [value for value in values if value <= 0]
    if invalid:
        raise ValueError(f"graph_analysis_pruning_num_beams must be positive, got {invalid}")
    return values


def pruning_eval_seed_from_config(config: dict[str, Any]) -> int:
    """Use one eval seed to keep B7 lightweight and interpretable."""

    seeds = config.get("graph_analysis_eval_seeds", [2024])
    if not seeds:
        raise ValueError("graph_analysis_eval_seeds cannot be empty for pruning analysis.")
    return int(seeds[0])


def _configure_num_beams(harness: Any, num_beams: int) -> None:
    """Set the beam width consistently on the harness and model."""

    harness.model.num_beams = int(num_beams)
    harness.model.config["num_beams"] = int(num_beams)
    harness.config["num_beams"] = int(num_beams)


def _first_step_containing(step_items: list[list[int]], target: int) -> int | None:
    """Return the first step whose item list contains ``target``."""

    for step, items in enumerate(step_items):
        if target in items:
            return step
    return None


def _classify_target(
    trace: BatchTrace,
    batch_index: int,
    target: int,
    predictions: list[int],
) -> dict[str, Any]:
    """Classify one target into the mutually exclusive B7 failure buckets."""

    first_considered = _first_step_containing(trace.unique_candidates_by_step[batch_index], target)
    first_beam = _first_step_containing(trace.frontier_by_step[batch_index], target)
    selected = target in predictions

    if selected:
        bucket = "selected"
    elif first_beam is not None:
        bucket = "in_beam_not_selected"
    elif first_considered is not None:
        bucket = "considered_never_in_beam"
    else:
        bucket = "not_reached"

    return {
        "target_considered": first_considered is not None,
        "target_first_considered_step": first_considered,
        "target_in_beam": first_beam is not None,
        "target_first_beam_step": first_beam,
        "target_selected": selected,
        "failure_bucket": bucket,
    }


def _collect_pruning_rows(
    *,
    harness: Any,
    adjacency: torch.Tensor,
    n_edges: int,
    num_beams: int,
    eval_seed: int,
    user_ids: list[str],
    max_users: int,
    metric_names: list[str],
    parity_batches: int,
) -> list[dict[str, Any]]:
    """Run one B7 beam setting and return per-example scalar rows."""

    from genrec.utils import init_seed

    init_seed(eval_seed, harness.config["reproducibility"])
    _configure_num_beams(harness, num_beams)
    _configure_dynamic_budget(harness, adjacency, n_edges)

    rows: list[dict[str, Any]] = []
    user_offset = 0
    maxk = harness.trainer.evaluator.maxk

    progress = tqdm(
        harness.test_dataloader,
        total=len(harness.test_dataloader),
        desc=f"Pruning n_edges={n_edges} beams={num_beams}",
    )
    with torch.no_grad():
        for batch_index_global, batch in enumerate(progress):
            if user_offset >= max_users:
                break

            batch = {key: value.to(harness.accelerator.device) for key, value in batch.items()}
            if batch_index_global < parity_batches:
                preds, visited_counts, trace = _assert_tracing_parity(harness, batch, maxk)
            else:
                preds, visited_counts, trace = traced_generate(harness.model, batch, maxk)

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

            for batch_index in range(keep):
                user_index = user_offset + batch_index
                target = int(labels[batch_index])
                pred_row = [int(item) for item in predictions[batch_index]]
                row = {
                    "user_index": user_index,
                    "user_raw_id": user_ids[user_index],
                    "eval_seed": eval_seed,
                    "n_edges": n_edges,
                    "num_beams": num_beams,
                    "propagation_steps": int(harness.model.propagation_steps),
                    "target_item_id": target,
                    "n_visited_items": len(trace.final_visited_items[batch_index]),
                }
                row.update(_classify_target(trace, batch_index, target, pred_row))
                for metric in metric_names:
                    row[metric] = float(metric_values[metric][batch_index])
                rows.append(row)

            user_offset += batch_size

    return rows


def _summarize_pruning(frame: pd.DataFrame, metric_names: list[str]) -> list[dict[str, Any]]:
    """Aggregate B7 buckets and recommendation metrics by ``num_beams``."""

    rows: list[dict[str, Any]] = []
    bucket_order = [
        "not_reached",
        "considered_never_in_beam",
        "in_beam_not_selected",
        "selected",
    ]
    for num_beams, group in frame.groupby("num_beams", sort=True):
        row: dict[str, Any] = {
            "num_beams": int(num_beams),
            "n_examples": int(len(group)),
            "target_considered_rate": float(group["target_considered"].mean()),
            "target_in_beam_rate": float(group["target_in_beam"].mean()),
            "target_selected_rate": float(group["target_selected"].mean()),
            "mean_visited_items": float(group["n_visited_items"].mean()),
            "mean_first_considered_step": float(
                group.loc[group["target_considered"], "target_first_considered_step"].mean()
            ),
            "mean_first_beam_step": float(
                group.loc[group["target_in_beam"], "target_first_beam_step"].mean()
            ),
        }
        for bucket in bucket_order:
            row[f"{bucket}_rate"] = float((group["failure_bucket"] == bucket).mean())
        for metric in metric_names:
            row[metric] = float(group[metric].mean())
        rows.append(row)
    return rows


def run_pruning(args: Any) -> int:
    """Run B7 lightweight beam-pruning analysis."""

    harness = build_harness_from_args(args)
    if harness.config["use_ddp"]:
        raise RuntimeError("Graph-analysis pruning command only supports single-process evaluation.")

    paths = latest_session(harness.config, args.session_dir)
    adjacency, metadata = load_prepared_graph(paths, harness)
    prepared_topk = int(metadata["topk"])
    n_edges = int(harness.config.get("graph_analysis_pruning_n_edges", prepared_topk))
    if n_edges <= 0 or n_edges > prepared_topk:
        raise ValueError(
            f"graph_analysis_pruning_n_edges must be in [1, {prepared_topk}], got {n_edges}"
        )

    num_beams_values = pruning_num_beams_from_config(harness.config)
    eval_seed = pruning_eval_seed_from_config(harness.config)
    max_users = int(harness.config.get("graph_analysis_pruning_max_users", 2000))
    if max_users <= 0:
        raise ValueError("graph_analysis_pruning_max_users must be positive.")
    parity_batches = int(harness.config.get("graph_analysis_trace_parity_batches", 1))

    metric_names = _metric_names(harness.config)
    test_split = harness.dataset.split()["test"]
    user_ids = [str(user) for user in test_split["user"]]
    max_users = min(max_users, len(user_ids))

    all_rows: list[dict[str, Any]] = []
    for num_beams in num_beams_values:
        all_rows.extend(
            _collect_pruning_rows(
                harness=harness,
                adjacency=adjacency,
                n_edges=n_edges,
                num_beams=num_beams,
                eval_seed=eval_seed,
                user_ids=user_ids,
                max_users=max_users,
                metric_names=metric_names,
                parity_batches=parity_batches,
            )
        )

    frame = pd.DataFrame(all_rows)
    outputs = pruning_output_paths(paths)
    frame.to_parquet(outputs["per_example_parquet"], index=False)
    summary_rows = _summarize_pruning(frame, metric_names)
    write_csv(outputs["summary_csv"], summary_rows)

    summary_payload = {
        "session_root": str(paths.root),
        "checkpoint_path": str(harness.checkpoint_path),
        "checkpoint_signature": checkpoint_signature(harness.checkpoint_path),
        "prepared_graph_topk": prepared_topk,
        "n_edges": n_edges,
        "num_beams_values": num_beams_values,
        "eval_seed": eval_seed,
        "max_users": max_users,
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
            "pruning_outputs": {key: str(value) for key, value in outputs.items()},
        },
    )

    print(paths.root)
    return 0
