"""Experiment C: RPG scoring-vs-decoding diagnostics.

This command asks whether RPG's current token scorer is already good enough
when evaluated by exhaustive scoring over all items, and whether graph decoding
recovers those high-scored items. It is intentionally diagnostic: no reranker or
new model component is introduced here.
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

from .dynamic import (
    _configure_dynamic_budget,
    _metric_names,
    _restore_rng_state,
    _save_rng_state,
    dynamic_eval_seeds_from_config,
)
from .dynamic_trace import compute_decoding_context, traced_graph_propagation
from .reranking.scorers import rpg_candidate_scores
from .runtime import build_harness_from_args
from .sessions import SessionPaths, append_or_update_manifest, latest_session, write_csv, write_json
from .static import load_prepared_graph


def scoring_output_paths(paths: SessionPaths) -> dict[str, Path]:
    """Return all files written by Experiment C."""

    summaries = paths.scoring / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    return {
        "bruteforce_per_user_parquet": paths.scoring / "bruteforce_per_user.parquet",
        "graph_per_example_parquet": paths.scoring / "graph_per_example.parquet",
        "bruteforce_summary_csv": summaries / "bruteforce_summary.csv",
        "graph_overlap_summary_csv": summaries / "graph_overlap_summary.csv",
        "runtime_summary_json": paths.scoring / "runtime_summary.json",
    }


def _optional_max_users(config: dict[str, Any], n_users: int) -> int:
    """Resolve an optional user cap used only for smoke tests."""

    raw_value = config.get("graph_analysis_scoring_max_users")
    if raw_value is None:
        return n_users
    max_users = int(raw_value)
    if max_users <= 0:
        raise ValueError("graph_analysis_scoring_max_users must be positive or null.")
    return min(max_users, n_users)


def scoring_n_edges_from_config(config: dict[str, Any], prepared_topk: int) -> list[int]:
    """Resolve graph-width settings for Experiment C.

    The C command uses a prepared top-``prepared_topk`` graph and slices it just
    like dynamic analysis. A scoring-specific config key keeps the intent clear;
    the dynamic key remains a compatibility fallback.
    """

    raw_values = config.get(
        "graph_analysis_scoring_n_edges",
        config.get("graph_analysis_dynamic_n_edges", [10, 20, 30, 50, 100]),
    )
    values = sorted({int(value) for value in raw_values})
    invalid = [value for value in values if value <= 0 or value > prepared_topk]
    if invalid:
        raise ValueError(
            f"graph_analysis_scoring_n_edges must be in [1, {prepared_topk}], got {invalid}"
        )
    return values


def _metric_values(
    results: dict[str, torch.Tensor],
    metric_names: list[str],
) -> dict[str, list[float]]:
    """Convert evaluator outputs into plain Python row lists."""

    return {
        metric: results[metric].detach().cpu().view(-1).tolist()
        for metric in metric_names
    }


def _topk_predictions_from_all_item_scores(
    *,
    scores: torch.Tensor,
    all_item_ids: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Map top-score column indices back to real item ids."""

    keep = min(topk, all_item_ids.shape[0])
    top_indices = torch.topk(scores, k=keep, dim=1).indices
    return all_item_ids[top_indices]


def _target_ranks_from_scores(scores: torch.Tensor, target_item_ids: torch.Tensor) -> torch.Tensor:
    """Compute exact target ranks without sorting every item.

    ``scores`` is ordered by item ids ``1..n_items-1``, so a target item id maps
    to column ``target - 1``. Ties are treated optimistically by counting only
    strictly higher scores.
    """

    target_columns = (target_item_ids - 1).view(-1, 1)
    target_scores = torch.gather(scores, dim=1, index=target_columns)
    return (scores > target_scores).sum(dim=1) + 1


def _cuda_synchronize(device: torch.device) -> None:
    """Synchronize CUDA timing only when running on a CUDA device."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_memory_mb(device: torch.device) -> float | None:
    """Return peak CUDA memory in MiB, or ``None`` on CPU."""

    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))


def _collect_bruteforce_rows(
    *,
    harness: Any,
    user_ids: list[str],
    max_users: int,
    metric_names: list[str],
    brute_force_topk: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score every real item for every selected test user.

    The scorer is the same semantic-token log-probability used inside RPG graph
    propagation. This produces the scoring upper bound for the current RPG
    scoring rule, not a new model or reranker.
    """

    device = harness.accelerator.device
    maxk = int(harness.trainer.evaluator.maxk)
    saved_topk = max(brute_force_topk, maxk)
    all_item_ids = torch.arange(1, harness.dataset.n_items, dtype=torch.long, device=device)
    item_id2tokens = harness.model.item_id2tokens.to(device)

    rows: list[dict[str, Any]] = []
    user_offset = 0
    total_context_seconds = 0.0
    total_item_scoring_seconds = 0.0
    peak_memory_before_reset = None
    if device.type == "cuda":
        peak_memory_before_reset = _peak_memory_mb(device)
        torch.cuda.reset_peak_memory_stats(device)

    progress = tqdm(
        harness.test_dataloader,
        total=len(harness.test_dataloader),
        desc="Experiment C brute-force scoring",
    )
    with torch.no_grad():
        for batch in progress:
            if user_offset >= max_users:
                break

            batch = {key: value.to(device) for key, value in batch.items()}
            batch_size = int(batch["labels"].shape[0])
            keep = min(batch_size, max_users - user_offset)

            _cuda_synchronize(device)
            context_start = time.perf_counter()
            context = compute_decoding_context(harness.model, batch)
            _cuda_synchronize(device)
            total_context_seconds += time.perf_counter() - context_start

            _cuda_synchronize(device)
            scoring_start = time.perf_counter()
            candidate_ids = all_item_ids.unsqueeze(0).expand(batch_size, -1)
            scores = rpg_candidate_scores(context.token_logits, item_id2tokens, candidate_ids)
            top_items = _topk_predictions_from_all_item_scores(
                scores=scores,
                all_item_ids=all_item_ids,
                topk=saved_topk,
            )
            labels = batch["labels"].view(batch_size, -1)[:, 0].long()
            target_ranks = _target_ranks_from_scores(scores, labels)
            _cuda_synchronize(device)
            total_item_scoring_seconds += time.perf_counter() - scoring_start

            metric_preds = top_items[:, :maxk].unsqueeze(-1)
            visited_counts = torch.full(
                (batch_size, 1),
                float(harness.dataset.n_items - 1),
                dtype=torch.float32,
                device=device,
            )
            results = harness.trainer.evaluator.calculate_metrics(
                (metric_preds, visited_counts),
                batch["labels"],
            )
            metric_values = _metric_values(results, metric_names)

            top_items_cpu = top_items.detach().cpu()
            labels_cpu = labels.detach().cpu().tolist()
            target_ranks_cpu = target_ranks.detach().cpu().tolist()
            target_scores_cpu = torch.gather(scores, 1, (labels - 1).view(-1, 1))
            target_scores_cpu = target_scores_cpu.detach().cpu().view(-1).tolist()
            top1_scores_cpu = scores.gather(1, top_items[:, :1] - 1)
            top1_scores_cpu = top1_scores_cpu.detach().cpu().view(-1).tolist()

            for batch_index in range(keep):
                user_index = user_offset + batch_index
                top_row = [int(item) for item in top_items_cpu[batch_index].tolist()]
                target_score = float(target_scores_cpu[batch_index])
                top1_score = float(top1_scores_cpu[batch_index])
                row: dict[str, Any] = {
                    "user_index": user_index,
                    "user_raw_id": user_ids[user_index],
                    "target_item_id": int(labels_cpu[batch_index]),
                    "bf_top10_json": json.dumps(top_row[:10]),
                    "bf_top50_json": json.dumps(top_row[:brute_force_topk]),
                    "bf_target_rank": int(target_ranks_cpu[batch_index]),
                    "bf_target_score": target_score,
                    "bf_top1_score": top1_score,
                    "bf_score_margin_top1_minus_target": top1_score - target_score,
                }
                for metric in metric_names:
                    row[f"bf_{metric}"] = float(metric_values[metric][batch_index])
                rows.append(row)

            user_offset += batch_size

    if len(rows) != max_users:
        raise RuntimeError(f"Collected {len(rows)} brute-force rows but expected {max_users}.")

    total_seconds = total_context_seconds + total_item_scoring_seconds
    runtime = {
        "n_users": int(max_users),
        "n_items_scored_per_user": int(harness.dataset.n_items - 1),
        "bruteforce_topk_saved": int(brute_force_topk),
        "total_context_seconds": float(total_context_seconds),
        "total_item_scoring_seconds": float(total_item_scoring_seconds),
        "total_context_and_scoring_seconds": float(total_seconds),
        "context_and_scoring_seconds_per_user": float(total_seconds / max(max_users, 1)),
        "peak_cuda_memory_mb": _peak_memory_mb(device),
        "peak_cuda_memory_mb_before_reset": peak_memory_before_reset,
    }
    return pd.DataFrame(rows), runtime


def _assert_context_parity(
    *,
    harness: Any,
    batch: dict[str, torch.Tensor],
    token_logits: torch.Tensor,
    n_return_sequences: int,
) -> None:
    """Check that context-based tracing still matches upstream generation."""

    rng_state = _save_rng_state()
    upstream_preds, upstream_counts = harness.model.generate(
        batch,
        n_return_sequences=n_return_sequences,
    )
    _restore_rng_state(rng_state)
    traced_preds, traced_counts, _ = traced_graph_propagation(
        harness.model,
        token_logits,
        n_return_sequences=n_return_sequences,
    )
    if not torch.equal(upstream_preds.detach().cpu(), traced_preds.detach().cpu()):
        raise AssertionError(
            "Experiment C parity failed: predictions differ from upstream generate()."
        )
    if not torch.equal(upstream_counts.detach().cpu(), traced_counts.detach().cpu()):
        raise AssertionError(
            "Experiment C parity failed: visited counts differ from upstream generate()."
        )
    _restore_rng_state(rng_state)


def _list_overlap_count(left: list[int], right: list[int]) -> int:
    """Count item overlap between two small recommendation lists."""

    return len(set(left).intersection(right))


def _graph_row(
    *,
    user_index: int,
    user_raw_id: str,
    eval_seed: int,
    n_edges: int,
    num_beams: int,
    propagation_steps: int,
    target: int,
    graph_predictions: list[int],
    visited_items: list[int],
    metric_values: dict[str, list[float]],
    metric_names: list[str],
    batch_index: int,
    bf_row: dict[str, Any],
) -> dict[str, Any]:
    """Build one graph-vs-bruteforce diagnostic row."""

    bf_top10 = [int(item) for item in json.loads(bf_row["bf_top10_json"])]
    bf_top50 = [int(item) for item in json.loads(bf_row["bf_top50_json"])]
    graph_top10 = graph_predictions[:10]
    visited = set(int(item) for item in visited_items)
    target_in_bf_top10 = target in bf_top10
    target_in_graph_top10 = target in graph_top10
    overlap_top10 = _list_overlap_count(graph_top10, bf_top10)
    overlap_denominator = max(len(bf_top10), 1)

    row: dict[str, Any] = {
        "user_index": user_index,
        "user_raw_id": user_raw_id,
        "eval_seed": eval_seed,
        "n_edges": n_edges,
        "num_beams": num_beams,
        "propagation_steps": propagation_steps,
        "target_item_id": target,
        "graph_top10_json": json.dumps(graph_top10),
        "bf_top10_json": bf_row["bf_top10_json"],
        "bf_target_rank": int(bf_row["bf_target_rank"]),
        "target_in_bf_top10": target_in_bf_top10,
        "target_in_graph_top10": target_in_graph_top10,
        "target_in_both_bf_and_graph_top10": target_in_bf_top10 and target_in_graph_top10,
        "target_bf_top10_graph_missed": target_in_bf_top10 and not target_in_graph_top10,
        "target_graph_top10_not_bf_top10": target_in_graph_top10 and not target_in_bf_top10,
        "target_in_neither_top10": not target_in_bf_top10 and not target_in_graph_top10,
        "graph_top10_overlap_bf_top10": overlap_top10,
        "graph_top10_overlap_bf_top10_frac": overlap_top10 / overlap_denominator,
        "graph_top10_contains_bf_top1": bool(bf_top10 and bf_top10[0] in graph_top10),
        "visited_bf_top1": bool(bf_top10 and bf_top10[0] in visited),
        "visited_any_bf_top10": any(item in visited for item in bf_top10),
        "visited_count_bf_top10": sum(1 for item in bf_top10 if item in visited),
        "visited_any_bf_top50": any(item in visited for item in bf_top50),
        "visited_count_bf_top50": sum(1 for item in bf_top50 if item in visited),
        "target_reachable": target in visited,
        "n_visited_items": len(visited),
    }
    for metric in metric_names:
        row[f"graph_{metric}"] = float(metric_values[metric][batch_index])
        row[f"bf_{metric}"] = float(bf_row[f"bf_{metric}"])
    return row


def _collect_graph_rows(
    *,
    harness: Any,
    adjacency: torch.Tensor,
    n_edges: int,
    eval_seed: int,
    user_ids: list[str],
    max_users: int,
    metric_names: list[str],
    parity_batches: int,
    bf_rows_by_user: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run traced graph decoding and compare each row to brute-force top items."""

    from genrec.utils import init_seed

    device = harness.accelerator.device
    init_seed(eval_seed, harness.config["reproducibility"])
    _configure_dynamic_budget(harness, adjacency, n_edges)

    rows: list[dict[str, Any]] = []
    user_offset = 0
    maxk = int(harness.trainer.evaluator.maxk)
    num_beams = int(harness.model.num_beams)
    propagation_steps = int(harness.model.propagation_steps)
    total_context_seconds = 0.0
    total_graph_propagation_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    progress = tqdm(
        harness.test_dataloader,
        total=len(harness.test_dataloader),
        desc=f"Experiment C graph n_edges={n_edges} seed={eval_seed}",
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
            total_context_seconds += time.perf_counter() - context_start

            if batch_index_global < parity_batches:
                _assert_context_parity(
                    harness=harness,
                    batch=batch,
                    token_logits=context.token_logits,
                    n_return_sequences=maxk,
                )

            _cuda_synchronize(device)
            propagation_start = time.perf_counter()
            preds, visited_counts, trace = traced_graph_propagation(
                harness.model,
                context.token_logits,
                n_return_sequences=maxk,
            )
            _cuda_synchronize(device)
            total_graph_propagation_seconds += time.perf_counter() - propagation_start

            results = harness.trainer.evaluator.calculate_metrics(
                (preds, visited_counts),
                batch["labels"],
            )
            metric_values = _metric_values(results, metric_names)
            labels = batch["labels"].detach().cpu().view(batch_size, -1)[:, 0].tolist()
            predictions = preds.detach().cpu().squeeze(-1).numpy().tolist()

            for batch_index in range(keep):
                user_index = user_offset + batch_index
                target = int(labels[batch_index])
                pred_row = [int(item) for item in predictions[batch_index]]
                rows.append(
                    _graph_row(
                        user_index=user_index,
                        user_raw_id=user_ids[user_index],
                        eval_seed=eval_seed,
                        n_edges=n_edges,
                        num_beams=num_beams,
                        propagation_steps=propagation_steps,
                        target=target,
                        graph_predictions=pred_row,
                        visited_items=trace.final_visited_items[batch_index],
                        metric_values=metric_values,
                        metric_names=metric_names,
                        batch_index=batch_index,
                        bf_row=bf_rows_by_user[user_index],
                    )
                )

            user_offset += batch_size

    if len(rows) != max_users:
        raise RuntimeError(
            f"Collected {len(rows)} graph rows for n_edges={n_edges}, seed={eval_seed}, "
            f"but expected {max_users}."
        )

    total_seconds = total_context_seconds + total_graph_propagation_seconds
    runtime = {
        "n_edges": int(n_edges),
        "eval_seed": int(eval_seed),
        "n_users": int(max_users),
        "total_context_seconds": float(total_context_seconds),
        "total_graph_propagation_seconds": float(total_graph_propagation_seconds),
        "total_context_and_graph_seconds": float(total_seconds),
        "context_and_graph_seconds_per_user": float(total_seconds / max(max_users, 1)),
        "peak_cuda_memory_mb": _peak_memory_mb(device),
    }
    return rows, runtime


def _summarize_bruteforce(frame: pd.DataFrame, metric_names: list[str]) -> list[dict[str, Any]]:
    """Aggregate brute-force RPG scorer quality."""

    row: dict[str, Any] = {
        "n_examples": int(len(frame)),
        "bf_target_rank_mean": float(frame["bf_target_rank"].mean()),
        "bf_target_rank_median": float(frame["bf_target_rank"].median()),
        "bf_target_rank_p90": float(frame["bf_target_rank"].quantile(0.9)),
        "bf_target_rank_le_10_rate": float((frame["bf_target_rank"] <= 10).mean()),
        "bf_target_rank_le_100_rate": float((frame["bf_target_rank"] <= 100).mean()),
        "bf_target_rank_le_1000_rate": float((frame["bf_target_rank"] <= 1000).mean()),
    }
    for metric in metric_names:
        row[f"bf_{metric}"] = float(frame[f"bf_{metric}"].mean())
    return [row]


def _summarize_graph_overlap(frame: pd.DataFrame, metric_names: list[str]) -> list[dict[str, Any]]:
    """Aggregate graph-vs-bruteforce diagnostics by edge budget and seed."""

    rows: list[dict[str, Any]] = []
    group_specs: list[tuple[str, list[str]]] = [
        ("seed", ["n_edges", "eval_seed"]),
        ("mean_across_seeds", ["n_edges"]),
    ]
    for aggregate, group_columns in group_specs:
        for keys, group in frame.groupby(group_columns, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row: dict[str, Any] = {"aggregate": aggregate, "n_examples": int(len(group))}
            for column, value in zip(group_columns, keys):
                row[column] = int(value)
            if "eval_seed" not in row:
                row["eval_seed"] = None
            for metric in metric_names:
                row[f"graph_{metric}"] = float(group[f"graph_{metric}"].mean())
                row[f"bf_{metric}"] = float(group[f"bf_{metric}"].mean())
            row.update(
                {
                    "target_reachable_rate": float(group["target_reachable"].mean()),
                    "target_in_bf_top10_rate": float(group["target_in_bf_top10"].mean()),
                    "target_in_graph_top10_rate": float(group["target_in_graph_top10"].mean()),
                    "target_bf_top10_graph_missed_rate": float(
                        group["target_bf_top10_graph_missed"].mean()
                    ),
                    "target_in_neither_top10_rate": float(group["target_in_neither_top10"].mean()),
                    "graph_top10_overlap_bf_top10_mean": float(
                        group["graph_top10_overlap_bf_top10"].mean()
                    ),
                    "graph_top10_overlap_bf_top10_frac_mean": float(
                        group["graph_top10_overlap_bf_top10_frac"].mean()
                    ),
                    "graph_top10_contains_bf_top1_rate": float(
                        group["graph_top10_contains_bf_top1"].mean()
                    ),
                    "visited_bf_top1_rate": float(group["visited_bf_top1"].mean()),
                    "visited_any_bf_top10_rate": float(group["visited_any_bf_top10"].mean()),
                    "visited_count_bf_top10_mean": float(group["visited_count_bf_top10"].mean()),
                    "visited_any_bf_top50_rate": float(group["visited_any_bf_top50"].mean()),
                    "visited_count_bf_top50_mean": float(group["visited_count_bf_top50"].mean()),
                    "mean_visited_items": float(group["n_visited_items"].mean()),
                }
            )
            rows.append(row)
    return rows


def run_scoring(args: Any) -> int:
    """Run Experiment C scoring-vs-decoding diagnostics."""

    harness = build_harness_from_args(args)
    if harness.config["use_ddp"]:
        raise RuntimeError(
            "Graph-analysis scoring command only supports single-process evaluation."
        )
    harness.model.eval()

    paths = latest_session(harness.config, args.session_dir)
    adjacency, metadata = load_prepared_graph(paths, harness)
    prepared_topk = int(metadata["topk"])
    n_edges_values = scoring_n_edges_from_config(harness.config, prepared_topk)
    eval_seeds = dynamic_eval_seeds_from_config(harness.config)
    metric_names = _metric_names(harness.config)
    brute_force_topk = int(harness.config.get("graph_analysis_bruteforce_topk", 50))
    if brute_force_topk < int(harness.trainer.evaluator.maxk):
        raise ValueError(
            "graph_analysis_bruteforce_topk must be at least evaluator max top-k "
            f"({harness.trainer.evaluator.maxk})."
        )
    parity_batches = int(harness.config.get("graph_analysis_trace_parity_batches", 1))

    test_split = harness.dataset.split()["test"]
    user_ids = [str(user) for user in test_split["user"]]
    if len(user_ids) != len(harness.test_dataloader.dataset):
        raise RuntimeError(
            f"Test user count ({len(user_ids)}) does not match tokenized test rows "
            f"({len(harness.test_dataloader.dataset)})."
        )
    max_users = _optional_max_users(harness.config, len(user_ids))

    outputs = scoring_output_paths(paths)
    bf_frame, bf_runtime = _collect_bruteforce_rows(
        harness=harness,
        user_ids=user_ids,
        max_users=max_users,
        metric_names=metric_names,
        brute_force_topk=brute_force_topk,
    )
    bf_frame.to_parquet(outputs["bruteforce_per_user_parquet"], index=False)
    write_csv(outputs["bruteforce_summary_csv"], _summarize_bruteforce(bf_frame, metric_names))

    bf_rows_by_user = {
        int(row["user_index"]): row
        for row in bf_frame.to_dict(orient="records")
    }

    graph_rows: list[dict[str, Any]] = []
    graph_runtime_rows: list[dict[str, Any]] = []
    for n_edges in n_edges_values:
        for eval_seed in eval_seeds:
            rows, runtime = _collect_graph_rows(
                harness=harness,
                adjacency=adjacency,
                n_edges=n_edges,
                eval_seed=eval_seed,
                user_ids=user_ids,
                max_users=max_users,
                metric_names=metric_names,
                parity_batches=parity_batches,
                bf_rows_by_user=bf_rows_by_user,
            )
            graph_rows.extend(rows)
            graph_runtime_rows.append(runtime)

    graph_frame = pd.DataFrame(graph_rows)
    graph_frame.to_parquet(outputs["graph_per_example_parquet"], index=False)
    graph_summary_rows = _summarize_graph_overlap(graph_frame, metric_names)
    write_csv(outputs["graph_overlap_summary_csv"], graph_summary_rows)

    runtime_payload = {
        "session_root": str(paths.root),
        "checkpoint_path": str(harness.checkpoint_path),
        "checkpoint_signature": checkpoint_signature(harness.checkpoint_path),
        "prepared_graph_topk": prepared_topk,
        "n_edges_values": n_edges_values,
        "eval_seeds": eval_seeds,
        "max_users": int(max_users),
        "full_test_users": int(len(user_ids)),
        "bruteforce_topk_saved": int(brute_force_topk),
        "num_beams": int(harness.model.num_beams),
        "propagation_steps": int(harness.model.propagation_steps),
        "temperature": float(harness.model.temperature),
        "metrics": metric_names,
        "bruteforce_runtime": bf_runtime,
        "graph_runtime": graph_runtime_rows,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    write_json(outputs["runtime_summary_json"], runtime_payload)
    append_or_update_manifest(
        paths,
        {
            "session_root": str(paths.root),
            "scoring_outputs": {key: str(value) for key, value in outputs.items()},
        },
    )

    print(paths.root)
    return 0
