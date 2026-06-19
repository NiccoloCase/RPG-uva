"""Visited-pool reranking diagnostic for RPG graph decoding.

B9 tests a narrow question: after standard RPG traversal has already visited a
set of items, does the final beam lose useful candidates that are still present
in the visited pool? The intervention keeps traversal unchanged, then replaces
the final-beam output with the top-k items scored over all visited nodes using
RPG's own semantic-token score.
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
from .dynamic_trace import compute_decoding_context, traced_graph_propagation
from .reranking.candidates import padded_visited_candidates
from .reranking.scorers import rpg_candidate_scores
from .runtime import build_harness_from_args
from .sessions import SessionPaths, append_or_update_manifest, latest_session, write_csv
from .static import load_prepared_graph


def pool_rerank_output_paths(paths: SessionPaths, run_name: str | None = None) -> dict[str, Path]:
    """Return all B9 output paths.

    The outputs live below ``rerank/visited_pool`` because this diagnostic is a
    final-candidate reranking intervention, even though the traversal itself is
    the unchanged RPG graph decoder.
    """

    root = paths.rerank / "visited_pool"
    if run_name:
        root = root / run_name
    summaries = root / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    return {
        "per_example_parquet": root / "per_example.parquet",
        "summary_csv": summaries / "visited_pool_summary.csv",
        "summary_json": root / "visited_pool_summary.json",
    }


def pool_n_edges_from_config(config: dict[str, Any], prepared_topk: int) -> list[int]:
    """Resolve graph-width settings for B9.

    A dedicated key keeps this diagnostic independent from broader reranking
    experiments. If absent, it uses the same small pair used by B8.
    """

    raw_values = config.get("graph_analysis_pool_n_edges", [30, 100])
    values = sorted({int(value) for value in raw_values})
    invalid = [value for value in values if value <= 0 or value > prepared_topk]
    if invalid:
        raise ValueError(f"graph_analysis_pool_n_edges must be in [1, {prepared_topk}], got {invalid}")
    return values


def pool_eval_seed_from_config(config: dict[str, Any]) -> int:
    """Use one evaluation seed by default to keep B9 lightweight."""

    seeds = config.get("graph_analysis_eval_seeds", [2024])
    if not seeds:
        raise ValueError("graph_analysis_eval_seeds cannot be empty for visited-pool reranking.")
    return int(seeds[0])


def _cuda_synchronize(device: torch.device) -> None:
    """Synchronize CUDA timing only when running on a CUDA device."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _metric_values(results: dict[str, torch.Tensor], metric_names: list[str]) -> dict[str, list[float]]:
    """Convert evaluator metric tensors into plain Python lists."""

    return {metric: results[metric].detach().cpu().view(-1).tolist() for metric in metric_names}


def _target_rank(predictions: list[int], target: int) -> int | None:
    """Return the 1-based rank of ``target`` in ``predictions`` if present."""

    try:
        return predictions.index(target) + 1
    except ValueError:
        return None


def _topk_predictions_from_pool_scores(
    *,
    pool_ids: torch.Tensor,
    pool_mask: torch.Tensor,
    scores: torch.Tensor,
    maxk: int,
) -> torch.Tensor:
    """Return top-k predictions from scored visited-pool candidates.

    ``pool_ids`` is padded with item id ``0``. Masked rows are assigned
    ``-inf`` before top-k, and the prediction tensor is padded back to ``maxk``
    if the candidate pool is smaller than the evaluator's maximum k.
    """

    scores = scores.masked_fill(~pool_mask, float("-inf"))
    keep = min(maxk, pool_ids.shape[1])
    top_indices = torch.topk(scores, k=keep, dim=1).indices
    preds = torch.gather(pool_ids, dim=1, index=top_indices)
    if keep < maxk:
        padding = torch.zeros(
            (pool_ids.shape[0], maxk - keep),
            dtype=preds.dtype,
            device=pool_ids.device,
        )
        preds = torch.cat([preds, padding], dim=1)
    return preds.unsqueeze(-1)


def _target_ranks_in_pool(
    *,
    pool_ids: torch.Tensor,
    pool_mask: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[list[int | None], list[float | None]]:
    """Compute exact target rank within each visited pool.

    Ties are handled optimistically by counting only strictly higher scores.
    ``None`` means the target was not present in the visited pool.
    """

    target_mask = (pool_ids == labels.view(-1, 1)) & pool_mask
    has_target = target_mask.any(dim=1)
    target_scores = scores.masked_fill(~target_mask, float("-inf")).max(dim=1).values
    ranks = (scores.masked_fill(~pool_mask, float("-inf")) > target_scores.view(-1, 1)).sum(dim=1) + 1

    ranks_cpu = ranks.detach().cpu().tolist()
    has_target_cpu = has_target.detach().cpu().tolist()
    target_scores_cpu = target_scores.detach().cpu().tolist()
    rank_values: list[int | None] = []
    score_values: list[float | None] = []
    for has_item, rank, score in zip(has_target_cpu, ranks_cpu, target_scores_cpu):
        rank_values.append(int(rank) if has_item else None)
        score_values.append(float(score) if has_item else None)
    return rank_values, score_values


def _assert_context_parity(
    *,
    harness: Any,
    batch: dict[str, torch.Tensor],
    token_logits: torch.Tensor,
    n_return_sequences: int,
) -> None:
    """Verify context-based tracing still matches upstream RPG generation."""

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
        raise AssertionError("Visited-pool parity failed: predictions differ from upstream generate().")
    if not torch.equal(upstream_counts.detach().cpu(), traced_counts.detach().cpu()):
        raise AssertionError("Visited-pool parity failed: visited counts differ from upstream generate().")
    _restore_rng_state(rng_state)


def _collect_pool_rows(
    *,
    harness: Any,
    adjacency: torch.Tensor,
    n_edges: int,
    eval_seed: int,
    user_ids: list[str],
    max_users: int,
    metric_names: list[str],
    parity_batches: int,
) -> list[dict[str, Any]]:
    """Run one B9 graph-width setting and return per-example rows."""

    from genrec.utils import init_seed

    init_seed(eval_seed, harness.config["reproducibility"])
    _configure_dynamic_budget(harness, adjacency, n_edges)

    rows: list[dict[str, Any]] = []
    user_offset = 0
    maxk = int(harness.trainer.evaluator.maxk)
    item_id2tokens = harness.model.item_id2tokens.to(harness.accelerator.device)

    progress = tqdm(
        harness.test_dataloader,
        total=len(harness.test_dataloader),
        desc=f"Visited-pool rerank n_edges={n_edges} seed={eval_seed}",
    )
    with torch.no_grad():
        for batch_index_global, batch in enumerate(progress):
            if user_offset >= max_users:
                break

            batch = {key: value.to(harness.accelerator.device) for key, value in batch.items()}
            batch_size = int(batch["labels"].shape[0])
            keep = min(batch_size, max_users - user_offset)

            _cuda_synchronize(harness.accelerator.device)
            context_start = time.perf_counter()
            context = compute_decoding_context(harness.model, batch)
            _cuda_synchronize(harness.accelerator.device)
            context_seconds_per_user = (time.perf_counter() - context_start) / max(batch_size, 1)

            if batch_index_global < parity_batches:
                _assert_context_parity(
                    harness=harness,
                    batch=batch,
                    token_logits=context.token_logits,
                    n_return_sequences=maxk,
                )

            _cuda_synchronize(harness.accelerator.device)
            graph_start = time.perf_counter()
            original_preds, visited_counts, trace = traced_graph_propagation(
                harness.model,
                context.token_logits,
                n_return_sequences=maxk,
            )
            _cuda_synchronize(harness.accelerator.device)
            graph_seconds_per_user = (time.perf_counter() - graph_start) / max(batch_size, 1)

            _cuda_synchronize(harness.accelerator.device)
            pool_start = time.perf_counter()
            candidate_ids, candidate_mask = padded_visited_candidates(
                trace,
                device=harness.accelerator.device,
            )
            score_ids = candidate_ids.masked_fill(~candidate_mask, 1)
            pool_scores = rpg_candidate_scores(context.token_logits, item_id2tokens, score_ids)
            pool_scores = pool_scores.masked_fill(~candidate_mask, float("-inf"))
            pool_preds = _topk_predictions_from_pool_scores(
                pool_ids=candidate_ids,
                pool_mask=candidate_mask,
                scores=pool_scores,
                maxk=maxk,
            )
            _cuda_synchronize(harness.accelerator.device)
            pool_seconds_per_user = (time.perf_counter() - pool_start) / max(batch_size, 1)

            original_results = harness.trainer.evaluator.calculate_metrics(
                (original_preds, visited_counts),
                batch["labels"],
            )
            pool_results = harness.trainer.evaluator.calculate_metrics(
                (pool_preds, visited_counts),
                batch["labels"],
            )
            original_metric_values = _metric_values(original_results, metric_names)
            pool_metric_values = _metric_values(pool_results, metric_names)

            labels = batch["labels"].detach().view(batch_size, -1)[:, 0].long()
            pool_ranks, pool_target_scores = _target_ranks_in_pool(
                pool_ids=candidate_ids,
                pool_mask=candidate_mask,
                scores=pool_scores,
                labels=labels,
            )
            labels_cpu = labels.detach().cpu().tolist()
            original_predictions = original_preds.detach().cpu().squeeze(-1).numpy().tolist()
            pool_predictions = pool_preds.detach().cpu().squeeze(-1).numpy().tolist()
            pool_counts = candidate_mask.detach().cpu().sum(dim=1).tolist()

            for batch_index in range(keep):
                user_index = user_offset + batch_index
                target = int(labels_cpu[batch_index])
                original_row = [int(item) for item in original_predictions[batch_index]]
                pool_row = [int(item) for item in pool_predictions[batch_index]]
                original_rank = _target_rank(original_row, target)
                pool_final_rank = _target_rank(pool_row, target)
                row: dict[str, Any] = {
                    "user_index": user_index,
                    "user_raw_id": user_ids[user_index],
                    "eval_seed": eval_seed,
                    "n_edges": n_edges,
                    "num_beams": int(harness.model.num_beams),
                    "propagation_steps": int(harness.model.propagation_steps),
                    "target_item_id": target,
                    "target_reachable": pool_ranks[batch_index] is not None,
                    "n_visited_items": int(pool_counts[batch_index]),
                    "original_predictions_json": json.dumps(original_row),
                    "pool_predictions_json": json.dumps(pool_row),
                    "original_target_selected": original_rank is not None,
                    "pool_target_selected": pool_final_rank is not None,
                    "original_target_rank": original_rank,
                    "pool_target_rank": pool_final_rank,
                    "target_rank_in_visited_pool": pool_ranks[batch_index],
                    "target_score_in_visited_pool": pool_target_scores[batch_index],
                    "context_seconds_per_user": float(context_seconds_per_user),
                    "graph_seconds_per_user": float(graph_seconds_per_user),
                    "pool_rerank_seconds_per_user": float(pool_seconds_per_user),
                }
                for metric in metric_names:
                    row[f"original_{metric}"] = float(original_metric_values[metric][batch_index])
                    row[f"pool_{metric}"] = float(pool_metric_values[metric][batch_index])
                rows.append(row)

            user_offset += batch_size

    return rows


def _summarize_pool_reranking(frame: pd.DataFrame, metric_names: list[str]) -> list[dict[str, Any]]:
    """Aggregate B9 rows by graph width."""

    rows: list[dict[str, Any]] = []
    for n_edges, group in frame.groupby("n_edges", sort=True):
        reachable = group[group["target_reachable"]]
        row: dict[str, Any] = {
            "n_edges": int(n_edges),
            "n_examples": int(len(group)),
            "reachable_rate": float(group["target_reachable"].mean()),
            "original_selected_rate": float(group["original_target_selected"].mean()),
            "pool_selected_rate": float(group["pool_target_selected"].mean()),
            "pool_gain_rate": float(
                (group["pool_target_selected"] & ~group["original_target_selected"]).mean()
            ),
            "pool_loss_rate": float(
                (group["original_target_selected"] & ~group["pool_target_selected"]).mean()
            ),
            "mean_visited_items": float(group["n_visited_items"].mean()),
            "mean_context_seconds_per_user": float(group["context_seconds_per_user"].mean()),
            "mean_graph_seconds_per_user": float(group["graph_seconds_per_user"].mean()),
            "mean_pool_rerank_seconds_per_user": float(group["pool_rerank_seconds_per_user"].mean()),
        }
        if reachable.empty:
            row["pool_selected_given_reachable_rate"] = float("nan")
            row["median_target_rank_in_visited_pool"] = float("nan")
        else:
            row["pool_selected_given_reachable_rate"] = float(
                reachable["pool_target_selected"].mean()
            )
            row["median_target_rank_in_visited_pool"] = float(
                reachable["target_rank_in_visited_pool"].median()
            )
        for metric in metric_names:
            row[f"original_{metric}"] = float(group[f"original_{metric}"].mean())
            row[f"pool_{metric}"] = float(group[f"pool_{metric}"].mean())
        rows.append(row)
    return rows


def run_pool_reranking(args: Any) -> int:
    """Run B9 visited-pool reranking analysis."""

    harness = build_harness_from_args(args)
    if harness.config["use_ddp"]:
        raise RuntimeError("Graph-analysis visited-pool reranking only supports single-process evaluation.")

    paths = latest_session(harness.config, args.session_dir)
    adjacency, metadata = load_prepared_graph(paths, harness)
    prepared_topk = int(metadata["topk"])
    n_edges_values = pool_n_edges_from_config(harness.config, prepared_topk)
    eval_seed = pool_eval_seed_from_config(harness.config)
    max_users = int(
        harness.config.get(
            "graph_analysis_pool_max_users",
            len(harness.dataset.split()["test"]["user"]),
        )
    )
    if max_users <= 0:
        raise ValueError("graph_analysis_pool_max_users must be positive.")
    parity_batches = int(harness.config.get("graph_analysis_trace_parity_batches", 1))

    metric_names = _metric_names(harness.config)
    test_split = harness.dataset.split()["test"]
    user_ids = [str(user) for user in test_split["user"]]
    max_users = min(max_users, len(user_ids))

    rows: list[dict[str, Any]] = []
    for n_edges in n_edges_values:
        rows.extend(
            _collect_pool_rows(
                harness=harness,
                adjacency=adjacency,
                n_edges=n_edges,
                eval_seed=eval_seed,
                user_ids=user_ids,
                max_users=max_users,
                metric_names=metric_names,
                parity_batches=parity_batches,
            )
        )

    frame = pd.DataFrame(rows)
    run_name = str(harness.config.get("run_id", "visited_pool"))
    outputs = pool_rerank_output_paths(paths, run_name)
    frame.to_parquet(outputs["per_example_parquet"], index=False)
    summary_rows = _summarize_pool_reranking(frame, metric_names)
    write_csv(outputs["summary_csv"], summary_rows)

    summary_payload = {
        "session_root": str(paths.root),
        "checkpoint_path": str(harness.checkpoint_path),
        "checkpoint_signature": checkpoint_signature(harness.checkpoint_path),
        "prepared_graph_topk": prepared_topk,
        "n_edges_values": n_edges_values,
        "eval_seed": eval_seed,
        "max_users": max_users,
        "num_beams": int(harness.model.num_beams),
        "propagation_steps": int(harness.model.propagation_steps),
        "temperature": float(harness.model.temperature),
        "scorer": "rpg_semantic_token_mean_log_score",
        "intervention": "topk_over_all_visited_nodes_after_standard_traversal",
        "metrics": metric_names,
        "summary": summary_rows,
    }
    outputs["summary_json"].write_text(json.dumps(summary_payload, indent=2, sort_keys=True))
    append_or_update_manifest(
        paths,
        {
            "session_root": str(paths.root),
            "visited_pool_rerank_outputs": {key: str(value) for key, value in outputs.items()},
        },
    )

    print(paths.root)
    return 0
