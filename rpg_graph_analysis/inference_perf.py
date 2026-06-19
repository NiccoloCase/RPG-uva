"""Production-style timing for RPG graph decoding vs all-item scoring.

This benchmark is deliberately separate from Experiment C. It does not trace
visited sets or write per-example diagnostics; it times full-batch inference
and aggregate evaluator metrics using the same profiling protocol as the
repo-owned performance workflow: warmup batches, repeated timed epochs, CUDA
baseline memory, peak memory, and runtime memory delta.
"""

from __future__ import annotations

import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from perf.config import checkpoint_signature

from .dynamic import _configure_dynamic_budget, _metric_names, dynamic_eval_seeds_from_config
from .dynamic_trace import compute_decoding_context
from .reranking.scorers import rpg_candidate_scores
from .runtime import build_harness_from_args
from .scoring import scoring_n_edges_from_config
from .sessions import SessionPaths, append_or_update_manifest, latest_session, write_csv, write_json
from .static import load_prepared_graph


def inference_perf_output_paths(paths: SessionPaths) -> dict[str, Path]:
    """Return output paths for the inference performance benchmark."""

    root = paths.root / "perf_inference"
    root.mkdir(parents=True, exist_ok=True)
    return {
        "raw_csv": root / "perf_raw.csv",
        "summary_csv": root / "perf_summary.csv",
        "summary_json": root / "perf_summary.json",
    }


def _cuda_synchronize(device: torch.device) -> None:
    """Synchronize CUDA timing only when the active device is CUDA."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _bytes_to_gib(value: int) -> float:
    """Convert a byte count to GiB, matching the existing perf workflow."""

    return float(value) / (1024.0**3)


def _memory_baseline(device: torch.device, measure_cuda_memory: bool) -> dict[str, float]:
    """Capture baseline CUDA memory and reset peak counters before timing."""

    if device.type != "cuda" or not measure_cuda_memory:
        return {
            "baseline_cuda_allocated_gb": float("nan"),
            "baseline_cuda_reserved_gb": float("nan"),
        }
    _cuda_synchronize(device)
    baseline_allocated = _bytes_to_gib(torch.cuda.memory_allocated(device))
    baseline_reserved = _bytes_to_gib(torch.cuda.memory_reserved(device))
    torch.cuda.reset_peak_memory_stats(device)
    return {
        "baseline_cuda_allocated_gb": baseline_allocated,
        "baseline_cuda_reserved_gb": baseline_reserved,
    }


def _memory_peak(
    device: torch.device,
    baseline: dict[str, float],
    measure_cuda_memory: bool,
) -> dict[str, float]:
    """Return peak CUDA memory and peak-minus-baseline runtime deltas."""

    if device.type != "cuda" or not measure_cuda_memory:
        return {
            "peak_cuda_allocated_gb": float("nan"),
            "peak_cuda_reserved_gb": float("nan"),
            "peak_cuda_runtime_delta_allocated_gb": float("nan"),
            "peak_cuda_runtime_delta_reserved_gb": float("nan"),
        }
    _cuda_synchronize(device)
    peak_allocated = _bytes_to_gib(torch.cuda.max_memory_allocated(device))
    peak_reserved = _bytes_to_gib(torch.cuda.max_memory_reserved(device))
    return {
        "peak_cuda_allocated_gb": peak_allocated,
        "peak_cuda_reserved_gb": peak_reserved,
        "peak_cuda_runtime_delta_allocated_gb": max(
            0.0,
            peak_allocated - baseline["baseline_cuda_allocated_gb"],
        ),
        "peak_cuda_runtime_delta_reserved_gb": max(
            0.0,
            peak_reserved - baseline["baseline_cuda_reserved_gb"],
        ),
    }


def _set_eval_seed(seed: int, reproducibility: bool) -> None:
    """Set the same CPU/CUDA seeds used by GenRec evaluation code."""

    from genrec.utils import init_seed

    init_seed(int(seed), reproducibility)


def _mean_metrics(results_sum: dict[str, float], n_users: int, metric_names: list[str]) -> dict[str, float]:
    """Normalize accumulated evaluator metric sums."""

    return {metric: results_sum[metric] / max(n_users, 1) for metric in metric_names}


def _add_metric_sums(
    results_sum: dict[str, float],
    results: dict[str, torch.Tensor],
    metric_names: list[str],
) -> None:
    """Accumulate evaluator metric sums in place."""

    for metric in metric_names:
        results_sum[metric] += float(results[metric].detach().cpu().sum().item())


def _warmup_batches(
    *,
    harness: Any,
    predict_batch: Any,
    warmup_batches: int,
) -> None:
    """Run a few untimed batches so kernel/cache setup does not pollute timing."""

    if warmup_batches <= 0:
        return
    device = harness.accelerator.device
    with torch.no_grad():
        for batch_index, batch in enumerate(harness.test_dataloader):
            if batch_index >= warmup_batches:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            predict_batch(batch)
    _cuda_synchronize(device)


def _timed_epoch(
    *,
    harness: Any,
    predict_batch: Any,
    metric_names: list[str],
    desc: str,
) -> dict[str, Any]:
    """Run one timed full-test epoch for a prepared prediction function."""

    device = harness.accelerator.device
    n_users = 0
    total_seconds = 0.0
    metrics_sum = {metric: 0.0 for metric in metric_names}

    progress = tqdm(harness.test_dataloader, total=len(harness.test_dataloader), desc=desc)
    with torch.no_grad():
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            batch_size = int(batch["labels"].shape[0])

            _cuda_synchronize(device)
            start = time.perf_counter()
            preds, visited_counts = predict_batch(batch)
            _cuda_synchronize(device)
            total_seconds += time.perf_counter() - start

            results = harness.trainer.evaluator.calculate_metrics(
                (preds, visited_counts),
                batch["labels"],
            )
            _add_metric_sums(metrics_sum, results, metric_names)
            n_users += batch_size

    row = {
        "n_users": int(n_users),
        "n_items": int(harness.dataset.n_items - 1),
        "total_seconds": float(total_seconds),
        "ms_per_user": float(total_seconds * 1000.0 / max(n_users, 1)),
        "users_per_second": float(n_users / total_seconds) if total_seconds > 0 else float("inf"),
    }
    row.update(_mean_metrics(metrics_sum, n_users, metric_names))
    return row


def _profile_repeats(
    *,
    harness: Any,
    metric_names: list[str],
    method: str,
    n_edges: int | None,
    eval_seed: int | None,
    repeat_seed_base: int,
    repeats: int,
    warmup_batches: int,
    measure_cuda_memory: bool,
    prepare_predictor: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Profile one method/configuration with warmup and repeated timed epochs."""

    raw_rows: list[dict[str, Any]] = []
    for repeat_index in range(repeats):
        repeat_seed = int(repeat_seed_base) + repeat_index
        predict_batch = prepare_predictor()

        _set_eval_seed(repeat_seed, harness.config["reproducibility"])
        _warmup_batches(
            harness=harness,
            predict_batch=predict_batch,
            warmup_batches=warmup_batches,
        )

        # Warmup consumes graph initial-beam randomness, so reset before timing.
        _set_eval_seed(repeat_seed, harness.config["reproducibility"])
        baseline = _memory_baseline(harness.accelerator.device, measure_cuda_memory)
        row = _timed_epoch(
            harness=harness,
            predict_batch=predict_batch,
            metric_names=metric_names,
            desc=f"Perf {method} repeat={repeat_index}",
        )
        row.update(_memory_peak(harness.accelerator.device, baseline, measure_cuda_memory))
        row.update(baseline)
        row.update(
            {
                "method": method,
                "n_edges": n_edges,
                "eval_seed": eval_seed,
                "repeat_index": repeat_index,
                "repeat_seed": repeat_seed,
                "num_beams": int(harness.model.num_beams) if n_edges is not None else None,
                "propagation_steps": (
                    int(harness.model.propagation_steps) if n_edges is not None else None
                ),
            }
        )
        raw_rows.append(row)

    return raw_rows, _summarize_repeat_rows(raw_rows, metric_names)


def _numeric_median(values: list[Any]) -> float:
    """Return the median of finite numeric values, or NaN if none are usable."""

    cleaned = [
        float(value)
        for value in values
        if value is not None and not (isinstance(value, float) and math.isnan(value))
    ]
    if not cleaned:
        return float("nan")
    return float(statistics.median(cleaned))


def _summarize_repeat_rows(rows: list[dict[str, Any]], metric_names: list[str]) -> dict[str, Any]:
    """Create one median summary row from raw repeat-level rows."""

    first = rows[0]
    summary: dict[str, Any] = {
        "method": first["method"],
        "n_edges": first["n_edges"],
        "eval_seed": first["eval_seed"],
        "repeats": len(rows),
        "num_beams": first["num_beams"],
        "propagation_steps": first["propagation_steps"],
        "n_users": first["n_users"],
        "n_items": first["n_items"],
    }
    for column in [
        "total_seconds",
        "ms_per_user",
        "users_per_second",
        "baseline_cuda_allocated_gb",
        "baseline_cuda_reserved_gb",
        "peak_cuda_allocated_gb",
        "peak_cuda_reserved_gb",
        "peak_cuda_runtime_delta_allocated_gb",
        "peak_cuda_runtime_delta_reserved_gb",
        *metric_names,
    ]:
        summary[f"{column}_median"] = _numeric_median([row[column] for row in rows])
    return summary


def _make_graph_predictor(harness: Any) -> Any:
    """Return a batch predictor that calls upstream graph decoding."""

    maxk = int(harness.trainer.evaluator.maxk)

    def predict_batch(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return harness.model.generate(batch, n_return_sequences=maxk)

    return predict_batch


def _make_bruteforce_predictor(harness: Any) -> Any:
    """Return a vectorized all-item RPG scorer for the current model."""

    device = harness.accelerator.device
    maxk = int(harness.trainer.evaluator.maxk)
    all_item_ids = torch.arange(1, harness.dataset.n_items, dtype=torch.long, device=device)
    item_id2tokens = harness.model.item_id2tokens.to(device)

    def predict_batch(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(batch["labels"].shape[0])
        context = compute_decoding_context(harness.model, batch)
        candidate_ids = all_item_ids.unsqueeze(0).expand(batch_size, -1)
        scores = rpg_candidate_scores(context.token_logits, item_id2tokens, candidate_ids)
        top_indices = torch.topk(scores, k=maxk, dim=1).indices
        preds = all_item_ids[top_indices].unsqueeze(-1)
        visited_counts = torch.full(
            (batch_size, 1),
            float(harness.dataset.n_items - 1),
            dtype=torch.float32,
            device=device,
        )
        return preds, visited_counts

    return predict_batch


def _perf_int(config: dict[str, Any], key: str, default: int, minimum: int = 0) -> int:
    """Resolve non-negative integer profiling config values."""

    value = int(config.get(key, default))
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}, got {value}.")
    return value


def run_inference_perf(args: Any) -> int:
    """Run the no-trace inference benchmark for graph vs all-item scoring."""

    harness = build_harness_from_args(args)
    if harness.config["use_ddp"]:
        raise RuntimeError("Inference performance benchmark supports single-process evaluation only.")
    harness.model.eval()

    paths = latest_session(harness.config, args.session_dir)
    adjacency, metadata = load_prepared_graph(paths, harness)
    prepared_topk = int(metadata["topk"])
    n_edges_values = scoring_n_edges_from_config(harness.config, prepared_topk)
    eval_seeds = dynamic_eval_seeds_from_config(harness.config)
    metric_names = _metric_names(harness.config)
    repeats = _perf_int(harness.config, "graph_analysis_perf_repeats", 3, minimum=1)
    warmup_batches = _perf_int(harness.config, "graph_analysis_perf_warmup_batches", 2)
    measure_cuda_memory = bool(harness.config.get("graph_analysis_perf_measure_cuda_memory", True))

    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    if bool(harness.config.get("graph_analysis_perf_include_bruteforce", True)):
        harness.model.generate_w_decoding_graph = False
        harness.model.adjacency = None
        brute_raw, brute_summary = _profile_repeats(
            harness=harness,
            metric_names=metric_names,
            method="bruteforce_all_items",
            n_edges=None,
            eval_seed=None,
            repeat_seed_base=int(eval_seeds[0]),
            repeats=repeats,
            warmup_batches=warmup_batches,
            measure_cuda_memory=measure_cuda_memory,
            prepare_predictor=lambda: _make_bruteforce_predictor(harness),
        )
        raw_rows.extend(brute_raw)
        summary_rows.append(brute_summary)

    for n_edges in n_edges_values:
        for eval_seed in eval_seeds:
            _configure_dynamic_budget(harness, adjacency, n_edges)
            graph_raw, graph_summary = _profile_repeats(
                harness=harness,
                metric_names=metric_names,
                method="graph_generate",
                n_edges=int(n_edges),
                eval_seed=int(eval_seed),
                repeat_seed_base=int(eval_seed),
                repeats=repeats,
                warmup_batches=warmup_batches,
                measure_cuda_memory=measure_cuda_memory,
                prepare_predictor=lambda: _make_graph_predictor(harness),
            )
            raw_rows.extend(graph_raw)
            summary_rows.append(graph_summary)

    outputs = inference_perf_output_paths(paths)
    write_csv(outputs["raw_csv"], raw_rows)
    write_csv(outputs["summary_csv"], summary_rows)
    summary_payload = {
        "session_root": str(paths.root),
        "checkpoint_path": str(harness.checkpoint_path),
        "checkpoint_signature": checkpoint_signature(harness.checkpoint_path),
        "prepared_graph_topk": prepared_topk,
        "n_edges_values": n_edges_values,
        "eval_seeds": eval_seeds,
        "repeats": repeats,
        "warmup_batches": warmup_batches,
        "measure_cuda_memory": measure_cuda_memory,
        "metrics": metric_names,
        "summary_rows": summary_rows,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    write_json(outputs["summary_json"], summary_payload)
    append_or_update_manifest(
        paths,
        {
            "session_root": str(paths.root),
            "inference_perf_outputs": {key: str(value) for key, value in outputs.items()},
        },
    )

    print(paths.root)
    return 0
