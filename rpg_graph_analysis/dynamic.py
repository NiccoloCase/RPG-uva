"""Dynamic graph-analysis command.

This module implements Experiment Block B: query-conditioned diagnostics from
actual RPG graph decoding. It reuses a prepared exact graph-analysis adjacency
cache and slices it for each dynamic ``n_edges`` budget.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from perf.config import checkpoint_signature

from .dynamic_trace import BatchTrace, traced_generate
from .runtime import build_harness_from_args, random_seeds_from_config
from .sessions import SessionPaths, append_or_update_manifest, latest_session, write_csv
from .static import load_prepared_graph


def dynamic_output_paths(paths: SessionPaths) -> dict[str, Path]:
    """Return output paths for dynamic graph analysis.

    Dynamic outputs live below the same graph-analysis session as the prepared
    graph and static metrics. This makes B results explicitly tied to the graph
    cache used to produce them.
    """

    summaries = paths.dynamic / "summaries"
    traces = paths.dynamic / "traces"
    summaries.mkdir(parents=True, exist_ok=True)
    traces.mkdir(parents=True, exist_ok=True)
    return {
        "per_example_parquet": paths.dynamic / "per_example.parquet",
        "sample_traces_jsonl": traces / "sample_traces.jsonl",
        "reachability_csv": summaries / "dynamic_reachability_summary.csv",
        "redundancy_csv": summaries / "dynamic_redundancy_summary.csv",
        "first_hit_csv": summaries / "dynamic_first_hit_summary.csv",
        "saturation_csv": summaries / "dynamic_saturation_summary.csv",
        "summary_json": paths.dynamic / "dynamic_summary.json",
    }


def dynamic_n_edges_from_config(config: dict[str, Any], prepared_topk: int) -> list[int]:
    """Resolve dynamic ``n_edges`` budgets from config.

    The dynamic command reuses one prepared top-``prepared_topk`` adjacency and
    slices its columns. Values larger than the prepared graph width would not be
    faithful to the requested budget, so they fail early.
    """

    raw_values = config.get("graph_analysis_dynamic_n_edges", [10, 20, 30, 50, 100])
    values = sorted({int(value) for value in raw_values})
    invalid = [value for value in values if value <= 0 or value > prepared_topk]
    if invalid:
        raise ValueError(
            f"graph_analysis_dynamic_n_edges must be in [1, {prepared_topk}], got {invalid}"
        )
    return values


def dynamic_eval_seeds_from_config(config: dict[str, Any]) -> list[int]:
    """Resolve eval seeds for dynamic analysis.

    ``graph_analysis_eval_seeds`` is preferred for B because these are
    evaluation-time seeds controlling random initial beams. If absent, the
    static-analysis random seeds are reused as a convenience.
    """

    if "graph_analysis_eval_seeds" in config:
        return [int(seed) for seed in config["graph_analysis_eval_seeds"]]
    return random_seeds_from_config(config)


def _metric_names(config: dict[str, Any]) -> list[str]:
    """Return evaluator metric column names such as ``recall@10``."""

    return [f"{metric}@{k}" for metric in config["metrics"] for k in config["topk"]]


def _save_rng_state() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    """Save CPU/CUDA RNG state for parity checks."""

    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return cpu_state, cuda_state


def _restore_rng_state(state: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
    """Restore CPU/CUDA RNG state saved by ``_save_rng_state``."""

    cpu_state, cuda_state = state
    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


def _assert_tracing_parity(
    harness: Any,
    batch: dict[str, torch.Tensor],
    n_return_sequences: int,
) -> tuple[torch.Tensor, torch.Tensor, BatchTrace]:
    """Assert traced decoding matches upstream decoding and return traced output.

    The only stochastic operation in graph propagation is the random initial
    beam sample. This function saves RNG state, runs upstream ``generate()``,
    restores the RNG state, and then runs the traced implementation. Exact
    equality of predictions and visited counts is required. The traced result is
    returned so parity-checked batches are not decoded a third time, which would
    alter the RNG sequence for subsequent batches.
    """

    rng_state = _save_rng_state()
    upstream = harness.model.generate(batch, n_return_sequences=n_return_sequences)
    _restore_rng_state(rng_state)
    traced_preds, traced_counts, trace = traced_generate(
        harness.model,
        batch,
        n_return_sequences=n_return_sequences,
    )
    upstream_preds, upstream_counts = upstream
    if not torch.equal(upstream_preds.detach().cpu(), traced_preds.detach().cpu()):
        raise AssertionError("Tracing parity failed: predictions differ from upstream generate().")
    if not torch.equal(upstream_counts.detach().cpu(), traced_counts.detach().cpu()):
        raise AssertionError("Tracing parity failed: visited counts differ from upstream generate().")
    return traced_preds, traced_counts, trace


def _target_rank(predictions: list[int], target: int) -> int | None:
    """Return 1-based target rank in predictions, or ``None`` if absent."""

    try:
        return predictions.index(target) + 1
    except ValueError:
        return None


def _prefix_counts_by_step(
    final_visited_by_step: list[set[int]],
    item_tokens: np.ndarray,
    prefix_lengths: tuple[int, ...] = (1, 2, 4),
) -> dict[int, list[int]]:
    """Count unique semantic-ID prefixes in the visited set after each step.

    Prefix diversity is a cheap proxy for semantic-region diversity. For
    example, prefix length ``2`` counts how many distinct first-two-token
    semantic-ID prefixes have appeared in the cumulative visited set.
    """

    counts: dict[int, list[int]] = {length: [] for length in prefix_lengths}
    for visited in final_visited_by_step:
        visited_ids = sorted(item for item in visited if item > 0)
        for length in prefix_lengths:
            prefixes = {
                tuple(item_tokens[item_id, :length].tolist())
                for item_id in visited_ids
            }
            counts[length].append(len(prefixes))
    return counts


def _visited_sets_by_step(trace: BatchTrace, batch_index: int) -> list[set[int]]:
    """Reconstruct cumulative visited sets for one batch row.

    ``BatchTrace`` stores new items per step to keep the raw trace explicit.
    B1/B4 reachability and B2 prefix diversity need cumulative visited sets, so
    this helper rebuilds them deterministically.
    """

    cumulative: set[int] = set()
    sets_by_step: list[set[int]] = []
    for new_items in trace.new_items_by_step[batch_index]:
        cumulative.update(int(item) for item in new_items)
        sets_by_step.append(set(cumulative))
    return sets_by_step


def _row_from_trace(
    *,
    trace: BatchTrace,
    batch_index: int,
    user_index: int,
    user_raw_id: str,
    eval_seed: int,
    n_edges: int,
    num_beams: int,
    propagation_steps: int,
    target: int,
    predictions: list[int],
    metric_values: dict[str, list[float]],
    metric_names: list[str],
    item_tokens: np.ndarray,
) -> dict[str, Any]:
    """Build one per-example scalar diagnostics row.

    The resulting row is the main all-example output for B. It intentionally
    stores scalar or JSON-string values only, so the full evaluation can be kept
    compact in a Parquet table. Full item-list traces are written separately for
    a small deterministic sample of users.
    """

    rank = _target_rank(predictions, target)
    visited_sets = _visited_sets_by_step(trace, batch_index)
    prefix_counts = _prefix_counts_by_step(visited_sets, item_tokens)
    first_reached_step = None
    for step, visited in enumerate(visited_sets):
        if target in visited:
            first_reached_step = step
            break

    row: dict[str, Any] = {
        "user_index": user_index,
        "user_raw_id": user_raw_id,
        "eval_seed": eval_seed,
        "n_edges": n_edges,
        "num_beams": num_beams,
        "propagation_steps": propagation_steps,
        "target_item_id": target,
        "predictions_json": json.dumps(predictions),
        "target_selected": rank is not None,
        "target_selected_at_maxk": rank is not None,
        "target_rank": rank,
        "target_reachable": first_reached_step is not None,
        "target_first_reached_step": first_reached_step,
        "n_visited_items": len(trace.final_visited_items[batch_index]),
    }
    for metric in metric_names:
        row[metric] = float(metric_values[metric][batch_index])

    for step, value in enumerate(trace.visited_count_by_step[batch_index]):
        row[f"visited_count_step_{step}"] = int(value)
    for step, value in enumerate(trace.raw_candidate_count_by_step[batch_index]):
        row[f"raw_candidate_count_step_{step}"] = int(value)
    for step, value in enumerate(trace.unique_candidate_count_by_step[batch_index]):
        row[f"unique_candidate_count_step_{step}"] = int(value)
    for step, value in enumerate(trace.new_item_count_by_step[batch_index]):
        row[f"new_item_count_step_{step}"] = int(value)
    for step, value in enumerate(trace.duplicate_candidate_ratio_by_step[batch_index]):
        row[f"duplicate_candidate_ratio_step_{step}"] = float(value)
    for step, value in enumerate(trace.novelty_ratio_by_step[batch_index]):
        row[f"novelty_ratio_step_{step}"] = float(value)
    for prefix_length, values in prefix_counts.items():
        for step, value in enumerate(values):
            row[f"prefix{prefix_length}_count_step_{step}"] = int(value)
    return row


def _sample_trace_row(
    *,
    trace: BatchTrace,
    batch_index: int,
    user_index: int,
    user_raw_id: str,
    eval_seed: int,
    n_edges: int,
    target: int,
    predictions: list[int],
) -> dict[str, Any]:
    """Build one sampled full-trace JSON row.

    These rows are intended for debugging and later query-trace visualization.
    They are sampled because storing full visited/frontier item lists for every
    user, seed, and budget would be large and harder to inspect.
    """

    return {
        "user_index": user_index,
        "user_raw_id": user_raw_id,
        "eval_seed": eval_seed,
        "n_edges": n_edges,
        "target_item_id": target,
        "predictions": predictions,
        "initial_items": trace.initial_items[batch_index],
        "frontier_by_step": trace.frontier_by_step[batch_index],
        "unique_candidates_by_step": trace.unique_candidates_by_step[batch_index],
        "new_items_by_step": trace.new_items_by_step[batch_index],
        "final_visited_items": trace.final_visited_items[batch_index],
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as JSON lines."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _summarize_reachability(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize B1 target reachability by budget.

    ``reachable_rate`` is the fraction of examples where the ground-truth item
    appears anywhere in the cumulative visited set. ``target_selected_rate`` is
    the final-list hit indicator at the evaluator's maximum ``topk``. Metrics
    such as ``recall@10`` remain available separately in the saturation table.
    The reachability/selection gap separates graph search from final ranking.
    """

    rows = []
    for n_edges, group in frame.groupby("n_edges", sort=True):
        rows.append(
            {
                "n_edges": int(n_edges),
                "n_examples": int(len(group)),
                "reachable_rate": float(group["target_reachable"].mean()),
                "target_selected_rate": float(group["target_selected"].mean()),
                "reachable_but_not_selected_rate": float(
                    (group["target_reachable"] & ~group["target_selected"]).mean()
                ),
                "mean_visited_items": float(group["n_visited_items"].mean()),
            }
        )
    return rows


def _summarize_redundancy(frame: pd.DataFrame, propagation_steps: int) -> list[dict[str, Any]]:
    """Summarize B2 redundancy/novelty metrics by budget and step.

    Step ``0`` describes the random initial beams. Later steps describe graph
    expansions. ``duplicate_candidate_ratio`` measures how much candidate
    duplication exists within a step; ``novelty_ratio`` measures how much of the
    unique candidate set was not already visited before that step.
    """

    rows = []
    for n_edges, group in frame.groupby("n_edges", sort=True):
        for step in range(propagation_steps + 1):
            rows.append(
                {
                    "n_edges": int(n_edges),
                    "step": step,
                    "visited_count_mean": float(group[f"visited_count_step_{step}"].mean()),
                    "new_item_count_mean": float(group[f"new_item_count_step_{step}"].mean()),
                    "unique_candidate_count_mean": float(
                        group[f"unique_candidate_count_step_{step}"].mean()
                    ),
                    "duplicate_candidate_ratio_mean": float(
                        group[f"duplicate_candidate_ratio_step_{step}"].mean()
                    ),
                    "novelty_ratio_mean": float(group[f"novelty_ratio_step_{step}"].mean()),
                    "prefix1_count_mean": float(group[f"prefix1_count_step_{step}"].mean()),
                    "prefix2_count_mean": float(group[f"prefix2_count_step_{step}"].mean()),
                    "prefix4_count_mean": float(group[f"prefix4_count_step_{step}"].mean()),
                }
            )
    return rows


def _summarize_first_hit(frame: pd.DataFrame, propagation_steps: int) -> list[dict[str, Any]]:
    """Summarize B4 first-hit distribution by budget.

    ``target_first_reached_step`` is based on actual traced decoding, not a
    graph-theoretic shortest path. This matters because RPG beam pruning can
    prevent theoretically nearby targets from ever being visited.
    """

    rows = []
    for n_edges, group in frame.groupby("n_edges", sort=True):
        base = {"n_edges": int(n_edges), "n_examples": int(len(group))}
        for step in range(propagation_steps + 1):
            base[f"first_reached_step_{step}_rate"] = float(
                (group["target_first_reached_step"] == step).mean()
            )
        base["never_reached_rate"] = float(group["target_first_reached_step"].isna().mean())
        reached = group.loc[group["target_reachable"], "target_first_reached_step"].dropna()
        base["mean_first_reached_step_reachable"] = (
            float(reached.mean()) if not reached.empty else float("nan")
        )
        base["median_first_reached_step_reachable"] = (
            float(reached.median()) if not reached.empty else float("nan")
        )
        rows.append(base)
    return rows


def _summarize_saturation(
    frame: pd.DataFrame,
    metric_names: list[str],
    propagation_steps: int,
) -> list[dict[str, Any]]:
    """Summarize B6 budget saturation curves.

    This table places recommendation metrics beside dynamic diagnostics for the
    same ``n_edges`` budgets. It is the main output for checking whether
    performance, reachability, visited-set size, and semantic-prefix diversity
    saturate together.
    """

    rows = []
    for n_edges, group in frame.groupby("n_edges", sort=True):
        row = {
            "n_edges": int(n_edges),
            "n_examples": int(len(group)),
            "reachable_rate": float(group["target_reachable"].mean()),
            "target_selected_rate": float(group["target_selected"].mean()),
            "mean_visited_items": float(group["n_visited_items"].mean()),
            "final_new_item_count_mean": float(
                group[f"new_item_count_step_{propagation_steps}"].mean()
            ),
            "final_prefix1_count_mean": float(group[f"prefix1_count_step_{propagation_steps}"].mean()),
            "final_prefix2_count_mean": float(group[f"prefix2_count_step_{propagation_steps}"].mean()),
            "final_prefix4_count_mean": float(group[f"prefix4_count_step_{propagation_steps}"].mean()),
        }
        for metric in metric_names:
            row[metric] = float(group[metric].mean())
        rows.append(row)
    return rows


def _sample_user_indices(n_users: int, sample_size: int, seed: int) -> set[int]:
    """Choose deterministic users for full trace storage.

    The same sampled user indices are used for every budget and seed, making
    query-trace comparisons easier.
    """

    if sample_size <= 0:
        return set()
    if sample_size >= n_users:
        return set(range(n_users))
    rng = np.random.default_rng(seed)
    return set(int(index) for index in rng.choice(n_users, size=sample_size, replace=False))


def _configure_dynamic_budget(harness: Any, adjacency: torch.Tensor, n_edges: int) -> None:
    """Attach the sliced prepared graph to the model for one budget.

    Setting ``init_flag=True`` is deliberate: it prevents upstream ``generate``
    from rebuilding the dense graph and forces both parity checks and traced
    decoding to use the prepared graph-analysis adjacency.
    """

    harness.model.n_edges = int(n_edges)
    harness.model.config["n_edges"] = int(n_edges)
    harness.config["n_edges"] = int(n_edges)
    harness.model.adjacency = adjacency[:, :n_edges].to(harness.accelerator.device)
    harness.model.generate_w_decoding_graph = True
    harness.model.init_flag = True
    harness.model.eval()


def _collect_budget_seed_rows(
    *,
    harness: Any,
    adjacency: torch.Tensor,
    n_edges: int,
    eval_seed: int,
    user_ids: list[str],
    sample_user_indices: set[int],
    metric_names: list[str],
    item_tokens: np.ndarray,
    parity_batches: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect per-example rows and sampled full traces for one budget/seed.

    This is the inner evaluation loop. It resets the eval seed so random initial
    beams are reproducible, configures the graph slice, optionally checks parity
    with upstream generation on early batches, and then converts traced batches
    into row-oriented outputs.
    """

    from genrec.utils import init_seed

    init_seed(eval_seed, harness.config["reproducibility"])
    _configure_dynamic_budget(harness, adjacency, n_edges)

    rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    user_offset = 0
    maxk = harness.trainer.evaluator.maxk
    num_beams = int(harness.model.num_beams)
    propagation_steps = int(harness.model.propagation_steps)

    progress = tqdm(
        harness.test_dataloader,
        total=len(harness.test_dataloader),
        desc=f"Dynamic n_edges={n_edges} seed={eval_seed}",
    )
    with torch.no_grad():
        for batch_index_global, batch in enumerate(progress):
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
            labels = batch["labels"].detach().cpu().view(batch_size, -1)[:, 0].tolist()
            predictions = preds.detach().cpu().squeeze(-1).numpy().tolist()
            metric_values = {
                metric: results[metric].detach().cpu().view(-1).tolist()
                for metric in metric_names
            }

            for batch_index in range(batch_size):
                user_index = user_offset + batch_index
                target = int(labels[batch_index])
                pred_row = [int(item) for item in predictions[batch_index]]
                rows.append(
                    _row_from_trace(
                        trace=trace,
                        batch_index=batch_index,
                        user_index=user_index,
                        user_raw_id=user_ids[user_index],
                        eval_seed=eval_seed,
                        n_edges=n_edges,
                        num_beams=num_beams,
                        propagation_steps=propagation_steps,
                        target=target,
                        predictions=pred_row,
                        metric_values=metric_values,
                        metric_names=metric_names,
                        item_tokens=item_tokens,
                    )
                )
                if user_index in sample_user_indices:
                    sample_rows.append(
                        _sample_trace_row(
                            trace=trace,
                            batch_index=batch_index,
                            user_index=user_index,
                            user_raw_id=user_ids[user_index],
                            eval_seed=eval_seed,
                            n_edges=n_edges,
                            target=target,
                            predictions=pred_row,
                        )
                    )

            user_offset += batch_size

    if user_offset != len(user_ids):
        raise RuntimeError(f"Collected {user_offset} test rows but expected {len(user_ids)} users.")
    return rows, sample_rows


def run_dynamic(args: Any) -> int:
    """Run dynamic/query-conditioned graph analysis from a prepared graph.

    The command reconstructs the RPG model from the checkpoint, validates that
    the selected session graph was prepared for the same checkpoint/category,
    runs the configured ``n_edges`` x eval-seed sweep, and writes B1/B2/B4/B6
    artifacts into the session's ``dynamic`` directory.
    """

    harness = build_harness_from_args(args)
    if harness.config["use_ddp"]:
        raise RuntimeError("Graph-analysis dynamic command only supports single-process evaluation.")

    paths = latest_session(harness.config, args.session_dir)
    adjacency, metadata = load_prepared_graph(paths, harness)
    prepared_topk = int(metadata["topk"])
    n_edges_values = dynamic_n_edges_from_config(harness.config, prepared_topk)
    eval_seeds = dynamic_eval_seeds_from_config(harness.config)
    sample_size = int(harness.config.get("graph_analysis_trace_sample_size", 128))
    sample_seed = int(harness.config.get("graph_analysis_trace_sample_seed", 2024))
    parity_batches = int(harness.config.get("graph_analysis_trace_parity_batches", 1))

    metric_names = _metric_names(harness.config)
    test_split = harness.dataset.split()["test"]
    user_ids = [str(user) for user in test_split["user"]]
    if len(user_ids) != len(harness.test_dataloader.dataset):
        raise RuntimeError(
            f"Test user count ({len(user_ids)}) does not match tokenized test rows "
            f"({len(harness.test_dataloader.dataset)})."
        )

    sample_indices = _sample_user_indices(len(user_ids), sample_size, sample_seed)
    item_tokens = harness.model.item_id2tokens.detach().cpu().numpy()

    all_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for n_edges in n_edges_values:
        for eval_seed in eval_seeds:
            rows, traces = _collect_budget_seed_rows(
                harness=harness,
                adjacency=adjacency,
                n_edges=n_edges,
                eval_seed=eval_seed,
                user_ids=user_ids,
                sample_user_indices=sample_indices,
                metric_names=metric_names,
                item_tokens=item_tokens,
                parity_batches=parity_batches,
            )
            all_rows.extend(rows)
            sample_rows.extend(traces)

    frame = pd.DataFrame(all_rows)
    outputs = dynamic_output_paths(paths)
    frame.to_parquet(outputs["per_example_parquet"], index=False)
    _write_jsonl(outputs["sample_traces_jsonl"], sample_rows)

    propagation_steps = int(harness.model.propagation_steps)
    reachability_rows = _summarize_reachability(frame)
    redundancy_rows = _summarize_redundancy(frame, propagation_steps)
    first_hit_rows = _summarize_first_hit(frame, propagation_steps)
    saturation_rows = _summarize_saturation(frame, metric_names, propagation_steps)

    write_csv(outputs["reachability_csv"], reachability_rows)
    write_csv(outputs["redundancy_csv"], redundancy_rows)
    write_csv(outputs["first_hit_csv"], first_hit_rows)
    write_csv(outputs["saturation_csv"], saturation_rows)

    summary_payload = {
        "session_root": str(paths.root),
        "checkpoint_path": str(harness.checkpoint_path),
        "checkpoint_signature": checkpoint_signature(harness.checkpoint_path),
        "prepared_graph_topk": prepared_topk,
        "n_edges_values": n_edges_values,
        "eval_seeds": eval_seeds,
        "num_beams": int(harness.model.num_beams),
        "propagation_steps": int(harness.model.propagation_steps),
        "temperature": float(harness.model.temperature),
        "n_examples": int(len(frame)),
        "n_sample_traces": int(len(sample_rows)),
        "metrics": metric_names,
        "reachability_summary": reachability_rows,
        "saturation_summary": saturation_rows,
    }
    outputs["summary_json"].write_text(json.dumps(summary_payload, indent=2, sort_keys=True))

    append_or_update_manifest(
        paths,
        {
            "session_root": str(paths.root),
            "dynamic_outputs": {key: str(value) for key, value in outputs.items()},
        },
    )

    print(paths.root)
    return 0
