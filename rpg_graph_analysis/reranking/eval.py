"""Online reranking intervention for RPG graph-analysis sessions."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from perf.config import checkpoint_signature

from rpg_graph_analysis.dynamic import (
    _configure_dynamic_budget,
    _metric_names,
    _restore_rng_state,
    _save_rng_state,
)
from rpg_graph_analysis.dynamic_trace import compute_decoding_context, traced_graph_propagation
from rpg_graph_analysis.reranking.candidates import (
    PoolSize,
    padded_visited_candidates,
    parse_pool_sizes,
    select_candidate_pool,
)
from rpg_graph_analysis.reranking.representations import (
    normalized_user_hidden,
    token_mean_item_embeddings,
)
from rpg_graph_analysis.reranking.scorers import (
    hybrid_rpg_dot_scores,
    rpg_candidate_scores,
    score_token_mean_dot,
)
from rpg_graph_analysis.reranking.summaries import summarize_reranking
from rpg_graph_analysis.runtime import build_harness_from_args
from rpg_graph_analysis.sessions import SessionPaths, append_or_update_manifest, latest_session, write_csv
from rpg_graph_analysis.static import load_prepared_graph


def rerank_output_paths(paths: SessionPaths, run_name: str | None = None) -> dict[str, Path]:
    """Return output paths for the reranking intervention command."""

    root = paths.rerank / run_name if run_name else paths.rerank
    summaries = root / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    return {
        "per_example_parquet": root / "per_example.parquet",
        "summary_csv": summaries / "rerank_summary.csv",
        "summary_json": root / "rerank_summary.json",
    }


def rerank_n_edges_from_config(config: dict[str, Any], prepared_topk: int) -> list[int]:
    """Resolve graph-width settings for reranking interventions."""

    raw_values = config.get("graph_analysis_rerank_n_edges", [int(config.get("n_edges", 50))])
    values = sorted({int(value) for value in raw_values})
    invalid = [value for value in values if value <= 0 or value > prepared_topk]
    if invalid:
        raise ValueError(f"graph_analysis_rerank_n_edges must be in [1, {prepared_topk}], got {invalid}")
    return values


def rerank_propagation_steps_from_config(config: dict[str, Any]) -> list[int]:
    """Resolve propagation-depth settings for reranking interventions."""

    raw_values = config.get(
        "graph_analysis_rerank_propagation_steps",
        [int(config.get("propagation_steps", 3))],
    )
    values = sorted({int(value) for value in raw_values})
    invalid = [value for value in values if value < 0]
    if invalid:
        raise ValueError(f"graph_analysis_rerank_propagation_steps must be non-negative, got {invalid}")
    return values


def rerank_eval_seed_from_config(config: dict[str, Any]) -> int:
    """Use one eval seed by default to keep the first intervention cheap."""

    seeds = config.get("graph_analysis_eval_seeds", [2024])
    if not seeds:
        raise ValueError("graph_analysis_eval_seeds cannot be empty for reranking.")
    return int(seeds[0])


def rerank_hybrid_alphas_from_config(config: dict[str, Any]) -> list[float]:
    """Resolve hybrid-score weights to test.

    ``alpha`` controls how much the final rerank score trusts RPG's original
    token score after row-wise normalization. ``alpha=0`` is the pure dot
    reranker, and ``alpha=1`` is RPG-score reranking over the candidate pool.
    """

    raw_values = config.get("graph_analysis_rerank_hybrid_alphas", [0.8, 0.9])
    values = sorted({float(value) for value in raw_values})
    invalid = [value for value in values if value < 0.0 or value > 1.0]
    if invalid:
        raise ValueError(f"graph_analysis_rerank_hybrid_alphas must be in [0, 1], got {invalid}")
    if not values:
        raise ValueError("graph_analysis_rerank_hybrid_alphas cannot be empty.")
    return values


def _set_propagation_steps(harness: Any, propagation_steps: int) -> None:
    """Set propagation depth consistently on model and config objects."""

    harness.model.propagation_steps = int(propagation_steps)
    harness.model.config["propagation_steps"] = int(propagation_steps)
    harness.config["propagation_steps"] = int(propagation_steps)


def _assert_context_parity(
    *,
    harness: Any,
    batch: dict[str, torch.Tensor],
    token_logits: torch.Tensor,
    n_return_sequences: int,
) -> None:
    """Check that context-based traced decoding matches upstream generation.

    Reranking needs ``user_hidden`` from the same forward pass as token logits,
    so it calls ``compute_decoding_context`` directly. This parity check keeps
    the traced graph propagation anchored to upstream ``generate()`` without
    decoding the batch a third time.
    """

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
        raise AssertionError("Rerank parity failed: predictions differ from upstream generate().")
    if not torch.equal(upstream_counts.detach().cpu(), traced_counts.detach().cpu()):
        raise AssertionError("Rerank parity failed: visited counts differ from upstream generate().")
    _restore_rng_state(rng_state)


def _target_rank(predictions: list[int], target: int) -> int | None:
    """Return 1-based target rank, or ``None`` if the target is absent."""

    try:
        return predictions.index(target) + 1
    except ValueError:
        return None


def _metric_values(results: dict[str, torch.Tensor], metric_names: list[str]) -> dict[str, list[float]]:
    """Convert evaluator outputs into CPU row lists."""

    return {
        metric: results[metric].detach().cpu().view(-1).tolist()
        for metric in metric_names
    }


def _topk_predictions_from_scores(
    *,
    pool_ids: torch.Tensor,
    pool_mask: torch.Tensor,
    scores: torch.Tensor,
    maxk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-k predictions from precomputed candidate scores."""

    if pool_ids.is_cuda:
        torch.cuda.synchronize(pool_ids.device)
    start = time.perf_counter()
    scores = scores.masked_fill(~pool_mask, float("-inf"))
    keep = min(maxk, pool_ids.shape[1])
    top_indices = torch.topk(scores, k=keep, dim=1).indices
    preds = torch.gather(pool_ids, dim=1, index=top_indices)
    if keep < maxk:
        padding = torch.zeros(
            (pool_ids.shape[0], maxk - keep),
            dtype=preds.dtype,
            device=preds.device,
        )
        preds = torch.cat([preds, padding], dim=1)
    if pool_ids.is_cuda:
        torch.cuda.synchronize(pool_ids.device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    per_row_time = torch.full(
        (pool_ids.shape[0],),
        elapsed_ms / max(pool_ids.shape[0], 1),
        dtype=torch.float32,
    )
    return preds.unsqueeze(-1), per_row_time


def _collect_rerank_rows(
    *,
    harness: Any,
    adjacency: torch.Tensor,
    n_edges: int,
    propagation_steps: int,
    eval_seed: int,
    pool_sizes: list[PoolSize],
    hybrid_alphas: list[float],
    user_ids: list[str],
    max_users: int,
    metric_names: list[str],
    parity_batches: int,
    item_embeddings: torch.Tensor,
) -> list[dict[str, Any]]:
    """Run one reranking setting and return compact per-example rows."""

    from genrec.utils import init_seed

    init_seed(eval_seed, harness.config["reproducibility"])
    _set_propagation_steps(harness, propagation_steps)
    _configure_dynamic_budget(harness, adjacency, n_edges)

    rows: list[dict[str, Any]] = []
    user_offset = 0
    maxk = harness.trainer.evaluator.maxk
    item_id2tokens = harness.model.item_id2tokens.to(harness.accelerator.device)

    progress = tqdm(
        harness.test_dataloader,
        total=len(harness.test_dataloader),
        desc=f"Rerank n_edges={n_edges} steps={propagation_steps} seed={eval_seed}",
    )
    with torch.no_grad():
        for batch_index_global, batch in enumerate(progress):
            if user_offset >= max_users:
                break

            batch = {key: value.to(harness.accelerator.device) for key, value in batch.items()}
            context = compute_decoding_context(harness.model, batch)
            if batch_index_global < parity_batches:
                _assert_context_parity(
                    harness=harness,
                    batch=batch,
                    token_logits=context.token_logits,
                    n_return_sequences=maxk,
                )
            preds, visited_counts, trace = traced_graph_propagation(
                harness.model,
                context.token_logits,
                n_return_sequences=maxk,
            )
            original_results = harness.trainer.evaluator.calculate_metrics(
                (preds, visited_counts),
                batch["labels"],
            )
            original_metric_values = _metric_values(original_results, metric_names)

            candidate_ids, candidate_mask = padded_visited_candidates(
                trace,
                device=harness.accelerator.device,
            )
            user_hidden = normalized_user_hidden(context.user_hidden)

            batch_size = int(batch["labels"].shape[0])
            keep = min(batch_size, max_users - user_offset)
            labels = batch["labels"].detach().cpu().view(batch_size, -1)[:, 0].tolist()
            original_predictions = preds.detach().cpu().squeeze(-1).numpy().tolist()

            for pool_size in pool_sizes:
                pool_ids, pool_mask = select_candidate_pool(
                    candidate_ids=candidate_ids,
                    valid_mask=candidate_mask,
                    token_logits=context.token_logits,
                    item_id2tokens=item_id2tokens,
                    pool_size=pool_size,
                )
                safe_pool_ids = pool_ids.masked_fill(~pool_mask, 0)
                dot_scores = score_token_mean_dot(user_hidden, item_embeddings, safe_pool_ids)
                rpg_scores = rpg_candidate_scores(
                    context.token_logits,
                    item_id2tokens,
                    pool_ids.masked_fill(~pool_mask, 1),
                )
                pool_counts = pool_mask.detach().cpu().sum(dim=1).tolist()

                for alpha in hybrid_alphas:
                    hybrid_scores = hybrid_rpg_dot_scores(
                        rpg_scores=rpg_scores,
                        dot_scores=dot_scores,
                        valid_mask=pool_mask,
                        alpha=alpha,
                    )
                    rerank_preds, rerank_time_ms = _topk_predictions_from_scores(
                        pool_ids=pool_ids,
                        pool_mask=pool_mask,
                        scores=hybrid_scores,
                        maxk=maxk,
                    )
                    rerank_results = harness.trainer.evaluator.calculate_metrics(
                        (rerank_preds, visited_counts),
                        batch["labels"],
                    )
                    rerank_metric_values = _metric_values(rerank_results, metric_names)
                    rerank_predictions = rerank_preds.detach().cpu().squeeze(-1).numpy().tolist()
                    per_row_time = rerank_time_ms.detach().cpu().tolist()

                    for batch_index in range(keep):
                        user_index = user_offset + batch_index
                        target = int(labels[batch_index])
                        original_pred_row = [int(item) for item in original_predictions[batch_index]]
                        rerank_pred_row = [int(item) for item in rerank_predictions[batch_index]]
                        original_rank = _target_rank(original_pred_row, target)
                        rerank_rank = _target_rank(rerank_pred_row, target)
                        target_reachable = target in trace.final_visited_items[batch_index]
                        row: dict[str, Any] = {
                            "user_index": user_index,
                            "user_raw_id": user_ids[user_index],
                            "eval_seed": eval_seed,
                            "n_edges": n_edges,
                            "num_beams": int(harness.model.num_beams),
                            "propagation_steps": propagation_steps,
                            "pool_size": str(pool_size),
                            "rerank_method": "hybrid_rpg_token_mean_dot_zscore",
                            "hybrid_alpha": float(alpha),
                            "target_item_id": target,
                            "target_reachable": target_reachable,
                            "n_visited_items": len(trace.final_visited_items[batch_index]),
                            "candidate_pool_size": int(pool_counts[batch_index]),
                            "original_predictions_json": json.dumps(original_pred_row),
                            "rerank_predictions_json": json.dumps(rerank_pred_row),
                            "original_target_selected": original_rank is not None,
                            "rerank_target_selected": rerank_rank is not None,
                            "original_target_rank": original_rank,
                            "rerank_target_rank": rerank_rank,
                            "rerank_time_ms": float(per_row_time[batch_index]),
                        }
                        for metric in metric_names:
                            row[f"original_{metric}"] = float(
                                original_metric_values[metric][batch_index]
                            )
                            row[f"rerank_{metric}"] = float(
                                rerank_metric_values[metric][batch_index]
                            )
                        rows.append(row)

            user_offset += batch_size

    return rows


def run_reranking(args: Any) -> int:
    """Run the lightweight RPG candidate-reranking intervention."""

    harness = build_harness_from_args(args)
    if harness.config["use_ddp"]:
        raise RuntimeError("Graph-analysis reranking command only supports single-process evaluation.")

    paths = latest_session(harness.config, args.session_dir)
    adjacency, metadata = load_prepared_graph(paths, harness)
    prepared_topk = int(metadata["topk"])

    n_edges_values = rerank_n_edges_from_config(harness.config, prepared_topk)
    propagation_steps_values = rerank_propagation_steps_from_config(harness.config)
    eval_seed = rerank_eval_seed_from_config(harness.config)
    pool_sizes = parse_pool_sizes(harness.config.get("graph_analysis_rerank_pool_sizes", [500, 1000, "all"]))
    hybrid_alphas = rerank_hybrid_alphas_from_config(harness.config)
    max_users = int(harness.config.get("graph_analysis_rerank_max_users", len(harness.dataset.split()["test"]["user"])))
    if max_users <= 0:
        raise ValueError("graph_analysis_rerank_max_users must be positive.")
    parity_batches = int(harness.config.get("graph_analysis_trace_parity_batches", 1))

    metric_names = _metric_names(harness.config)
    test_split = harness.dataset.split()["test"]
    user_ids = [str(user) for user in test_split["user"]]
    max_users = min(max_users, len(user_ids))
    item_embeddings = token_mean_item_embeddings(harness.model).to(harness.accelerator.device)

    all_rows: list[dict[str, Any]] = []
    for n_edges in n_edges_values:
        for propagation_steps in propagation_steps_values:
            all_rows.extend(
                _collect_rerank_rows(
                    harness=harness,
                    adjacency=adjacency,
                    n_edges=n_edges,
                    propagation_steps=propagation_steps,
                    eval_seed=eval_seed,
                    pool_sizes=pool_sizes,
                    hybrid_alphas=hybrid_alphas,
                    user_ids=user_ids,
                    max_users=max_users,
                    metric_names=metric_names,
                    parity_batches=parity_batches,
                    item_embeddings=item_embeddings,
                )
            )

    frame = pd.DataFrame(all_rows)
    outputs = rerank_output_paths(paths, str(harness.config.get("run_id", "rerank_run")))
    frame.to_parquet(outputs["per_example_parquet"], index=False)
    summary_rows = summarize_reranking(frame, metric_names)
    write_csv(outputs["summary_csv"], summary_rows)

    summary_payload = {
        "session_root": str(paths.root),
        "checkpoint_path": str(harness.checkpoint_path),
        "checkpoint_signature": checkpoint_signature(harness.checkpoint_path),
        "prepared_graph_topk": prepared_topk,
        "n_edges_values": n_edges_values,
        "propagation_steps_values": propagation_steps_values,
        "eval_seed": eval_seed,
        "pool_sizes": [str(value) for value in pool_sizes],
        "hybrid_alphas": hybrid_alphas,
        "max_users": max_users,
        "num_beams": int(harness.model.num_beams),
        "temperature": float(harness.model.temperature),
        "scorer": "hybrid_rpg_token_mean_dot_zscore",
        "pool_selection": "top_rpg_score_or_all",
        "metrics": metric_names,
        "summary": summary_rows,
    }
    outputs["summary_json"].write_text(json.dumps(summary_payload, indent=2, sort_keys=True))
    append_or_update_manifest(
        paths,
        {
            "session_root": str(paths.root),
            "rerank_outputs": {key: str(value) for key, value in outputs.items()},
        },
    )

    print(paths.root)
    return 0
