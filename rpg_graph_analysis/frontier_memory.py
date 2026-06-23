"""Frontier-memory diagnostic for RPG graph decoding.

This command tests a narrow hypothesis raised after the novelty and visited-pool
diagnostics: repeated node re-entry may be compensating for the fact that RPG's
beam is both the exploration frontier and the memory of good candidates.

The diagnostic compares three variants on the same users and the same random
initial beams:

``original``
    Upstream RPG-style graph propagation with repeated frontier re-entry.

``prefer_unvisited``
    The existing analysis intervention that prefers unvisited candidates when
    selecting the next frontier and falls back to seen candidates only when the
    fresh frontier is too small. Final recommendations still come from the last
    frontier, so this tests whether pushing the beam toward exploration breaks
    RPG's implicit memory behavior.

``prefer_unvisited_memory``
    The same prefer-unvisited traversal, but final recommendations come from a
    bounded explicit memory of the best RPG-scored visited candidates. This is a
    lightweight post-hoc approximation of a result queue: it tests whether
    explicit candidate memory can recover quality after exploration pressure,
    but it does not yet implement a full priority-queue graph search where the
    queue also controls future expansion.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from perf.config import checkpoint_signature

from .dynamic import _configure_dynamic_budget, _metric_names, _restore_rng_state, _save_rng_state
from .dynamic_trace import (
    BatchTrace,
    compute_decoding_context,
    traced_graph_propagation,
)
from .reranking.candidates import padded_visited_candidates
from .reranking.scorers import rpg_candidate_scores
from .runtime import build_harness_from_args
from .sessions import SessionPaths, append_or_update_manifest, latest_session, write_csv
from .static import load_prepared_graph


def frontier_memory_output_paths(paths: SessionPaths, run_name: str | None = None) -> dict[str, Path]:
    """Return output paths for the frontier-memory pilot."""

    root = paths.dynamic / "frontier_memory"
    if run_name:
        root = root / run_name
    summaries = root / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    return {
        "per_example_parquet": root / "per_example.parquet",
        "summary_csv": summaries / "frontier_memory_summary.csv",
        "summary_json": root / "frontier_memory_summary.json",
    }


def frontier_memory_n_edges_from_config(config: dict[str, Any], prepared_topk: int) -> list[int]:
    """Resolve graph-width settings for the frontier-memory pilot."""

    raw_values = config.get("graph_analysis_frontier_memory_n_edges", [30, 100])
    values = sorted({int(value) for value in raw_values})
    invalid = [value for value in values if value <= 0 or value > prepared_topk]
    if invalid:
        raise ValueError(
            "graph_analysis_frontier_memory_n_edges must be in "
            f"[1, {prepared_topk}], got {invalid}"
        )
    return values


def frontier_memory_eval_seeds_from_config(config: dict[str, Any]) -> list[int]:
    """Resolve eval seeds for the frontier-memory diagnostic."""

    seeds = config.get("graph_analysis_eval_seeds", [2024])
    if not seeds:
        raise ValueError("graph_analysis_eval_seeds cannot be empty.")
    return [int(seed) for seed in seeds]


def frontier_memory_num_beams_from_config(config: dict[str, Any]) -> list[int]:
    """Resolve beam widths for the frontier-memory budget sweep."""

    raw_values = config.get("graph_analysis_frontier_memory_num_beams", [config["num_beams"]])
    values = sorted({int(value) for value in raw_values})
    invalid = [value for value in values if value <= 0]
    if invalid:
        raise ValueError(f"graph_analysis_frontier_memory_num_beams must be positive, got {invalid}")
    return values


def frontier_memory_propagation_steps_from_config(config: dict[str, Any]) -> list[int]:
    """Resolve propagation depths for the frontier-memory budget sweep."""

    raw_values = config.get(
        "graph_analysis_frontier_memory_propagation_steps",
        [config["propagation_steps"]],
    )
    values = sorted({int(value) for value in raw_values})
    invalid = [value for value in values if value <= 0]
    if invalid:
        raise ValueError(
            "graph_analysis_frontier_memory_propagation_steps must be positive, "
            f"got {invalid}"
        )
    return values


def frontier_memory_sizes_from_config(config: dict[str, Any]) -> list[int]:
    """Resolve explicit memory sizes for the frontier-memory variants."""

    raw_values = config.get("graph_analysis_frontier_memory_sizes")
    if raw_values is None:
        raw_values = [config.get("graph_analysis_frontier_memory_size", config["num_beams"])]
    values = sorted({int(value) for value in raw_values})
    invalid = [value for value in values if value <= 0]
    if invalid:
        raise ValueError(f"graph_analysis_frontier_memory_sizes must be positive, got {invalid}")
    return values


def frontier_memory_max_users_from_config(config: dict[str, Any], n_users: int) -> int:
    """Resolve the user cap, with ``null`` meaning the full test set."""

    raw_value = config.get("graph_analysis_frontier_memory_max_users", 2000)
    if raw_value is None:
        return n_users
    value = int(raw_value)
    if value <= 0:
        raise ValueError("graph_analysis_frontier_memory_max_users must be positive or null.")
    return min(value, n_users)


def _cuda_synchronize(device: torch.device) -> None:
    """Synchronize CUDA timing only when running on CUDA."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _metric_values(results: dict[str, torch.Tensor], metric_names: list[str]) -> dict[str, list[float]]:
    """Convert evaluator metric tensors into Python lists."""

    return {metric: results[metric].detach().cpu().view(-1).tolist() for metric in metric_names}


def _target_rank(predictions: list[int], target: int) -> int | None:
    """Return the 1-based rank of ``target`` in ``predictions`` if present."""

    try:
        return predictions.index(target) + 1
    except ValueError:
        return None


def _target_reachable(trace: BatchTrace, batch_index: int, target: int) -> bool:
    """Return whether ``target`` appears in the final visited set."""

    return target in {int(item) for item in trace.final_visited_items[batch_index]}


def _frontier_selection_stats(trace: BatchTrace, batch_index: int) -> tuple[int, int, float]:
    """Measure repeated frontier selections for one decoded example.

    RPG expands the selected frontier repeatedly. If the same item appears in
    multiple frontiers, it can be re-expanded or kept alive as a high-scoring
    candidate. This scalar does not prove causal waste, but it is the direct
    symptom this diagnostic targets.
    """

    selected: list[int] = []
    for step_frontier in trace.frontier_by_step[batch_index]:
        selected.extend(int(item) for item in step_frontier)
    total = len(selected)
    unique = len(set(selected))
    duplicate_rate = 1.0 - (unique / total if total else 0.0)
    return total, unique, duplicate_rate


def _topk_predictions_from_scores(
    *,
    candidate_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    scores: torch.Tensor,
    maxk: int,
    memory_size: int,
) -> torch.Tensor:
    """Return final top-k predictions from a bounded candidate memory."""

    scores = scores.masked_fill(~valid_mask, float("-inf"))
    keep_memory = min(memory_size, candidate_ids.shape[1])
    memory_indices = torch.topk(scores, k=keep_memory, dim=1).indices
    memory_ids = torch.gather(candidate_ids, dim=1, index=memory_indices)
    memory_scores = torch.gather(scores, dim=1, index=memory_indices)

    keep = min(maxk, keep_memory)
    top_indices = torch.topk(memory_scores, k=keep, dim=1).indices
    preds = torch.gather(memory_ids, dim=1, index=top_indices)
    if keep < maxk:
        padding = torch.zeros(
            (candidate_ids.shape[0], maxk - keep),
            dtype=preds.dtype,
            device=preds.device,
        )
        preds = torch.cat([preds, padding], dim=1)
    return preds.unsqueeze(-1)


def _memory_predictions(
    *,
    trace: BatchTrace,
    token_logits: torch.Tensor,
    item_id2tokens: torch.Tensor,
    device: torch.device,
    maxk: int,
    memory_size: int,
) -> tuple[torch.Tensor, list[int]]:
    """Score visited nodes and return predictions from explicit candidate memory."""

    candidate_ids, valid_mask = padded_visited_candidates(trace, device=device)
    score_ids = candidate_ids.masked_fill(~valid_mask, 1)
    scores = rpg_candidate_scores(token_logits, item_id2tokens, score_ids)
    preds = _topk_predictions_from_scores(
        candidate_ids=candidate_ids,
        valid_mask=valid_mask,
        scores=scores,
        maxk=maxk,
        memory_size=memory_size,
    )
    memory_counts = valid_mask.sum(dim=1).clamp_max(memory_size).detach().cpu().tolist()
    return preds, [int(count) for count in memory_counts]


def _assert_original_parity(
    *,
    harness: Any,
    batch: dict[str, torch.Tensor],
    original_preds: torch.Tensor,
    original_counts: torch.Tensor,
    rng_state: tuple[torch.Tensor, list[torch.Tensor] | None],
    n_return_sequences: int,
) -> None:
    """Check traced original decoding still matches upstream ``generate``."""

    _restore_rng_state(rng_state)
    upstream_preds, upstream_counts = harness.model.generate(
        batch,
        n_return_sequences=n_return_sequences,
    )
    if not torch.equal(upstream_preds.detach().cpu(), original_preds.detach().cpu()):
        raise AssertionError("Frontier-memory parity failed: original predictions differ.")
    if not torch.equal(upstream_counts.detach().cpu(), original_counts.detach().cpu()):
        raise AssertionError("Frontier-memory parity failed: original visited counts differ.")


def _variant_rows(
    *,
    variant: str,
    trace: BatchTrace,
    predictions: torch.Tensor,
    visited_counts: torch.Tensor,
    metric_values: dict[str, list[float]],
    metric_names: list[str],
    labels: list[int],
    user_ids: list[str],
    user_offset: int,
    keep: int,
    eval_seed: int,
    n_edges: int,
    num_beams: int,
    propagation_steps: int,
    memory_size: int | None,
    context_seconds_per_user: float,
    graph_seconds_per_user: float,
    memory_seconds_per_user: float,
    memory_candidate_counts: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Build per-example rows for one decoded variant."""

    rows: list[dict[str, Any]] = []
    prediction_rows = predictions.detach().cpu().squeeze(-1).numpy().tolist()
    visited_count_values = visited_counts.detach().cpu().view(-1).tolist()

    for batch_index in range(keep):
        user_index = user_offset + batch_index
        target = int(labels[batch_index])
        pred_row = [int(item) for item in prediction_rows[batch_index]]
        rank = _target_rank(pred_row, target)
        frontier_total, frontier_unique, frontier_duplicate_rate = _frontier_selection_stats(
            trace,
            batch_index,
        )
        row: dict[str, Any] = {
            "user_index": user_index,
            "user_raw_id": user_ids[user_index],
            "eval_seed": eval_seed,
            "n_edges": n_edges,
            "num_beams": num_beams,
            "propagation_steps": propagation_steps,
            "variant": variant,
            "memory_size": memory_size,
            "target_item_id": target,
            "target_reachable": _target_reachable(trace, batch_index, target),
            "target_selected": rank is not None,
            "target_rank": rank,
            "n_visited_items": int(visited_count_values[batch_index]),
            "memory_candidate_count": (
                int(memory_candidate_counts[batch_index]) if memory_candidate_counts else None
            ),
            "frontier_selection_count": frontier_total,
            "frontier_unique_selection_count": frontier_unique,
            "frontier_duplicate_selection_rate": frontier_duplicate_rate,
            "final_new_item_count": int(trace.new_item_count_by_step[batch_index][-1]),
            "final_candidate_novelty_ratio": float(trace.novelty_ratio_by_step[batch_index][-1]),
            "context_seconds_per_user": context_seconds_per_user,
            "graph_seconds_per_user": graph_seconds_per_user,
            "memory_seconds_per_user": memory_seconds_per_user,
            "predictions_json": json.dumps(pred_row),
        }
        for metric in metric_names:
            row[metric] = float(metric_values[metric][batch_index])
        rows.append(row)
    return rows


def _collect_frontier_memory_rows(
    *,
    harness: Any,
    adjacency: torch.Tensor,
    n_edges: int,
    eval_seed: int,
    user_ids: list[str],
    max_users: int,
    metric_names: list[str],
    parity_batches: int,
    memory_sizes: list[int],
    num_beams: int,
    propagation_steps: int,
) -> list[dict[str, Any]]:
    """Run all frontier-memory variants for one graph width."""

    from genrec.utils import init_seed

    init_seed(eval_seed, harness.config["reproducibility"])
    _configure_dynamic_budget(harness, adjacency, n_edges)
    harness.model.num_beams = int(num_beams)
    harness.model.propagation_steps = int(propagation_steps)
    harness.model.config["num_beams"] = int(num_beams)
    harness.model.config["propagation_steps"] = int(propagation_steps)
    harness.config["num_beams"] = int(num_beams)
    harness.config["propagation_steps"] = int(propagation_steps)

    rows: list[dict[str, Any]] = []
    user_offset = 0
    maxk = int(harness.trainer.evaluator.maxk)
    device = harness.accelerator.device
    item_id2tokens = harness.model.item_id2tokens.to(device)

    progress = tqdm(
        harness.test_dataloader,
        total=len(harness.test_dataloader),
        desc=(
            f"Frontier-memory n_edges={n_edges} beams={num_beams} "
            f"steps={propagation_steps} seed={eval_seed}"
        ),
    )
    with torch.no_grad():
        for batch_index_global, batch in enumerate(progress):
            if user_offset >= max_users:
                break

            batch = {key: value.to(device) for key, value in batch.items()}
            batch_size = int(batch["labels"].shape[0])
            keep = min(batch_size, max_users - user_offset)

            _cuda_synchronize(device)
            context_start = time.perf_counter()
            context = compute_decoding_context(harness.model, batch)
            _cuda_synchronize(device)
            context_seconds_per_user = (time.perf_counter() - context_start) / max(batch_size, 1)

            base_rng_state = _save_rng_state()

            _restore_rng_state(base_rng_state)
            _cuda_synchronize(device)
            original_start = time.perf_counter()
            original_preds, original_counts, original_trace = traced_graph_propagation(
                harness.model,
                context.token_logits,
                n_return_sequences=maxk,
                mask_visited_frontier=False,
            )
            _cuda_synchronize(device)
            original_seconds_per_user = (time.perf_counter() - original_start) / max(batch_size, 1)
            after_original_rng_state = _save_rng_state()

            if batch_index_global < parity_batches:
                _assert_original_parity(
                    harness=harness,
                    batch=batch,
                    original_preds=original_preds,
                    original_counts=original_counts,
                    rng_state=base_rng_state,
                    n_return_sequences=maxk,
                )

            _restore_rng_state(base_rng_state)
            _cuda_synchronize(device)
            no_reentry_start = time.perf_counter()
            no_reentry_preds, no_reentry_counts, no_reentry_trace = traced_graph_propagation(
                harness.model,
                context.token_logits,
                n_return_sequences=maxk,
                mask_visited_frontier=True,
            )
            _cuda_synchronize(device)
            no_reentry_seconds_per_user = (time.perf_counter() - no_reentry_start) / max(
                batch_size,
                1,
            )

            _restore_rng_state(after_original_rng_state)

            labels_tensor = batch["labels"].detach().view(batch_size, -1)[:, 0].long()
            labels = [int(item) for item in labels_tensor.cpu().tolist()]

            original_results = harness.trainer.evaluator.calculate_metrics(
                (original_preds, original_counts),
                batch["labels"],
            )
            no_reentry_results = harness.trainer.evaluator.calculate_metrics(
                (no_reentry_preds, no_reentry_counts),
                batch["labels"],
            )

            rows.extend(
                _variant_rows(
                    variant="original",
                    trace=original_trace,
                    predictions=original_preds,
                    visited_counts=original_counts,
                    metric_values=_metric_values(original_results, metric_names),
                    metric_names=metric_names,
                    labels=labels,
                    user_ids=user_ids,
                    user_offset=user_offset,
                    keep=keep,
                    eval_seed=eval_seed,
                    n_edges=n_edges,
                    num_beams=num_beams,
                    propagation_steps=propagation_steps,
                    memory_size=None,
                    context_seconds_per_user=context_seconds_per_user,
                    graph_seconds_per_user=original_seconds_per_user,
                    memory_seconds_per_user=0.0,
                )
            )
            rows.extend(
                _variant_rows(
                    variant="prefer_unvisited",
                    trace=no_reentry_trace,
                    predictions=no_reentry_preds,
                    visited_counts=no_reentry_counts,
                    metric_values=_metric_values(no_reentry_results, metric_names),
                    metric_names=metric_names,
                    labels=labels,
                    user_ids=user_ids,
                    user_offset=user_offset,
                    keep=keep,
                    eval_seed=eval_seed,
                    n_edges=n_edges,
                    num_beams=num_beams,
                    propagation_steps=propagation_steps,
                    memory_size=None,
                    context_seconds_per_user=context_seconds_per_user,
                    graph_seconds_per_user=no_reentry_seconds_per_user,
                    memory_seconds_per_user=0.0,
                )
            )
            for memory_size in memory_sizes:
                _cuda_synchronize(device)
                memory_start = time.perf_counter()
                memory_preds, memory_candidate_counts = _memory_predictions(
                    trace=no_reentry_trace,
                    token_logits=context.token_logits,
                    item_id2tokens=item_id2tokens,
                    device=device,
                    maxk=maxk,
                    memory_size=memory_size,
                )
                _cuda_synchronize(device)
                memory_seconds_per_user = (time.perf_counter() - memory_start) / max(batch_size, 1)

                memory_results = harness.trainer.evaluator.calculate_metrics(
                    (memory_preds, no_reentry_counts),
                    batch["labels"],
                )
                rows.extend(
                    _variant_rows(
                        variant="prefer_unvisited_memory",
                        trace=no_reentry_trace,
                        predictions=memory_preds,
                        visited_counts=no_reentry_counts,
                        metric_values=_metric_values(memory_results, metric_names),
                        metric_names=metric_names,
                        labels=labels,
                        user_ids=user_ids,
                        user_offset=user_offset,
                        keep=keep,
                        eval_seed=eval_seed,
                        n_edges=n_edges,
                        num_beams=num_beams,
                        propagation_steps=propagation_steps,
                        memory_size=memory_size,
                        context_seconds_per_user=context_seconds_per_user,
                        graph_seconds_per_user=no_reentry_seconds_per_user,
                        memory_seconds_per_user=memory_seconds_per_user,
                        memory_candidate_counts=memory_candidate_counts,
                    )
                )

            user_offset += batch_size

    return rows


def _summarize_frontier_memory(frame: pd.DataFrame, metric_names: list[str]) -> list[dict[str, Any]]:
    """Aggregate frontier-memory rows and paired gains versus original."""

    original = frame[frame["variant"] == "original"][
        ["eval_seed", "n_edges", "num_beams", "propagation_steps", "user_index", "target_selected"]
    ].rename(columns={"target_selected": "original_target_selected"})
    paired = frame.merge(
        original,
        on=["eval_seed", "n_edges", "num_beams", "propagation_steps", "user_index"],
        how="left",
    )

    rows: list[dict[str, Any]] = []
    group_cols = ["n_edges", "num_beams", "propagation_steps", "variant", "memory_size"]
    for keys, group in paired.groupby(group_cols, sort=True, dropna=False):
        n_edges, num_beams, propagation_steps, variant, memory_size = keys
        row: dict[str, Any] = {
            "n_edges": int(n_edges),
            "num_beams": int(num_beams),
            "propagation_steps": int(propagation_steps),
            "variant": str(variant),
            "memory_size": None if pd.isna(memory_size) else int(memory_size),
            "n_examples": int(len(group)),
            "reachable_rate": float(group["target_reachable"].mean()),
            "target_selected_rate": float(group["target_selected"].mean()),
            "gain_vs_original_rate": float(
                (group["target_selected"] & ~group["original_target_selected"]).mean()
            ),
            "loss_vs_original_rate": float(
                (~group["target_selected"] & group["original_target_selected"]).mean()
            ),
            "mean_visited_items": float(group["n_visited_items"].mean()),
            "mean_memory_candidate_count": float(group["memory_candidate_count"].mean()),
            "mean_frontier_duplicate_selection_rate": float(
                group["frontier_duplicate_selection_rate"].mean()
            ),
            "mean_final_new_item_count": float(group["final_new_item_count"].mean()),
            "mean_final_candidate_novelty_ratio": float(
                group["final_candidate_novelty_ratio"].mean()
            ),
            "mean_context_seconds_per_user": float(group["context_seconds_per_user"].mean()),
            "mean_graph_seconds_per_user": float(group["graph_seconds_per_user"].mean()),
            "mean_memory_seconds_per_user": float(group["memory_seconds_per_user"].mean()),
        }
        for metric in metric_names:
            row[metric] = float(group[metric].mean())
        rows.append(row)
    return rows


def run_frontier_memory(args: Any) -> int:
    """Run the frontier-memory pilot diagnostic."""

    harness = build_harness_from_args(args)
    if harness.config["use_ddp"]:
        raise RuntimeError("Graph-analysis frontier-memory only supports single-process evaluation.")

    paths = latest_session(harness.config, args.session_dir)
    adjacency, metadata = load_prepared_graph(paths, harness)
    prepared_topk = int(metadata["topk"])
    n_edges_values = frontier_memory_n_edges_from_config(harness.config, prepared_topk)
    eval_seeds = frontier_memory_eval_seeds_from_config(harness.config)
    num_beams_values = frontier_memory_num_beams_from_config(harness.config)
    propagation_steps_values = frontier_memory_propagation_steps_from_config(harness.config)
    memory_sizes = frontier_memory_sizes_from_config(harness.config)
    parity_batches = int(harness.config.get("graph_analysis_trace_parity_batches", 1))

    metric_names = _metric_names(harness.config)
    test_split = harness.dataset.split()["test"]
    user_ids = [str(user) for user in test_split["user"]]
    max_users = frontier_memory_max_users_from_config(harness.config, len(user_ids))

    rows: list[dict[str, Any]] = []
    for eval_seed in eval_seeds:
        for n_edges in n_edges_values:
            for num_beams in num_beams_values:
                for propagation_steps in propagation_steps_values:
                    rows.extend(
                        _collect_frontier_memory_rows(
                            harness=harness,
                            adjacency=adjacency,
                            n_edges=n_edges,
                            eval_seed=eval_seed,
                            user_ids=user_ids,
                            max_users=max_users,
                            metric_names=metric_names,
                            parity_batches=parity_batches,
                            memory_sizes=memory_sizes,
                            num_beams=num_beams,
                            propagation_steps=propagation_steps,
                        )
                    )

    frame = pd.DataFrame(rows)
    run_name = str(harness.config.get("run_id", "frontier_memory"))
    outputs = frontier_memory_output_paths(paths, run_name)
    frame.to_parquet(outputs["per_example_parquet"], index=False)
    summary_rows = _summarize_frontier_memory(frame, metric_names)
    write_csv(outputs["summary_csv"], summary_rows)

    summary_payload = {
        "session_root": str(paths.root),
        "checkpoint_path": str(harness.checkpoint_path),
        "checkpoint_signature": checkpoint_signature(harness.checkpoint_path),
        "prepared_graph_topk": prepared_topk,
        "n_edges_values": n_edges_values,
        "num_beams_values": num_beams_values,
        "propagation_steps_values": propagation_steps_values,
        "eval_seeds": eval_seeds,
        "max_users": max_users,
        "memory_sizes": memory_sizes,
        "temperature": float(harness.model.temperature),
        "variants": ["original", "prefer_unvisited", "prefer_unvisited_memory"],
        "metrics": metric_names,
        "summary": summary_rows,
    }
    outputs["summary_json"].write_text(json.dumps(summary_payload, indent=2, sort_keys=True))
    append_or_update_manifest(
        paths,
        {
            "session_root": str(paths.root),
            "frontier_memory_outputs": {key: str(value) for key, value in outputs.items()},
        },
    )

    print(paths.root)
    return 0
