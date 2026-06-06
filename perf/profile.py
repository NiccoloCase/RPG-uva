from __future__ import annotations

import csv
import json
import math
import os
import statistics
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from .config import checkpoint_signature
from .graph import (
    build_dense_reference_adjacency,
    build_or_load_adjacency,
    build_sparse_adjacency,
    compare_adjacency_overlap,
    compare_adjacency_sets,
)
from .harness import EvaluationHarness
from .pool import augment_candidate_pool


def _session_root(config: dict[str, Any], output_root: str | None = None) -> Path:
    """Create a unique output directory for one profiling session.

    The session directory contains subfolders for raw run-level measurements,
    summary tables, and graph-build artifacts. Paths are made unique using a
    UTC timestamp and, when available, the current Slurm job ID.

    Args:
        config: Profiling config dictionary.
        output_root: Optional explicit output root. Falls back to config or to
            `artifacts/rpg/perf`.

    Returns:
        The newly created session root path.
    """
    raw_root = output_root or config.get("perf_output_dir")
    if raw_root is None:
        raw_root = "artifacts/rpg/perf"
    path = Path(raw_root)
    if not path.is_absolute():
        path = Path.cwd() / path
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    session_name = timestamp
    if slurm_job_id:
        session_name = f"{session_name}_job{slurm_job_id}"
    session_root = path.resolve() / session_name
    suffix = 1
    while session_root.exists():
        session_root = path.resolve() / f"{session_name}_{suffix:02d}"
        suffix += 1
    (session_root / "raw").mkdir(parents=True, exist_ok=True)
    (session_root / "summaries").mkdir(parents=True, exist_ok=True)
    (session_root / "graphs").mkdir(parents=True, exist_ok=True)
    return session_root


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries to a CSV file, preserving key order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries to newline-delimited JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _median_or_nan(values: list[float]) -> float:
    """Return the median of non-NaN values, or NaN when none are usable."""
    cleaned = [value for value in values if not math.isnan(value)]
    if not cleaned:
        return float("nan")
    return float(statistics.median(cleaned))


def _set_repeat_seed(base_seed: int, repeat_index: int) -> int:
    """Set deterministic CPU/CUDA seeds for one profiling repeat."""
    seed = int(base_seed) + int(repeat_index)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def _maybe_cuda_synchronize(device: torch.device) -> None:
    """Synchronize the CUDA device when profiling GPU work."""
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _bytes_to_gib(value: int) -> float:
    """Convert a byte count to gibibytes for human-readable reporting."""
    return float(value) / (1024 ** 3)


def _profile_epoch(
    harness: EvaluationHarness,
    measure_cuda_memory: bool,
) -> tuple[dict[str, float], float, float, float, float, float, float, float]:
    """Profile one evaluation pass of the harness.

    This helper runs a full evaluation over the prepared test dataloader while
    measuring wall-clock latency and, optionally, CUDA memory usage before and
    during execution.

    Args:
        harness: Ready-to-run evaluation harness.
        measure_cuda_memory: Whether to capture CUDA baseline/peak metrics when
            the active device is a GPU.

    Returns:
        A tuple containing:
        1. Evaluation metrics dictionary from `harness.evaluate()`
        2. Elapsed wall-clock seconds
        3. Peak CUDA allocated memory in GiB
        4. Peak CUDA reserved memory in GiB
        5. Baseline CUDA allocated memory in GiB
        6. Baseline CUDA reserved memory in GiB
        7. Peak runtime delta for allocated memory in GiB
        8. Peak runtime delta for reserved memory in GiB
    """
    device = harness.config["device"]

    baseline_allocated = float("nan")
    baseline_reserved = float("nan")
    if device.type == "cuda" and measure_cuda_memory:
        _maybe_cuda_synchronize(device)
        baseline_allocated = _bytes_to_gib(torch.cuda.memory_allocated(device=device))
        baseline_reserved = _bytes_to_gib(torch.cuda.memory_reserved(device=device))
        torch.cuda.reset_peak_memory_stats(device=device)

    start_time = time.perf_counter()
    results = harness.evaluate()
    _maybe_cuda_synchronize(device)
    elapsed_seconds = time.perf_counter() - start_time

    if device.type == "cuda" and measure_cuda_memory:
        peak_allocated = _bytes_to_gib(torch.cuda.max_memory_allocated(device=device))
        peak_reserved = _bytes_to_gib(torch.cuda.max_memory_reserved(device=device))
        runtime_delta_allocated = max(0.0, peak_allocated - baseline_allocated)
        runtime_delta_reserved = max(0.0, peak_reserved - baseline_reserved)
    else:
        peak_allocated = float("nan")
        peak_reserved = float("nan")
        runtime_delta_allocated = float("nan")
        runtime_delta_reserved = float("nan")

    return (
        results,
        elapsed_seconds,
        peak_allocated,
        peak_reserved,
        baseline_allocated,
        baseline_reserved,
        runtime_delta_allocated,
        runtime_delta_reserved,
    )


def _resolve_pool_sizes(config: dict[str, Any], pool_sizes_override: list[int] | None) -> list[int]:
    """Resolve which candidate-pool sizes should be profiled."""
    if pool_sizes_override:
        return pool_sizes_override
    if "pool_sizes" not in config or not config["pool_sizes"]:
        raise ValueError("No pool_sizes were provided via config or CLI override.")
    return [int(value) for value in config["pool_sizes"]]


def _resolve_graph_backend(config: dict[str, Any], override: str | None) -> str:
    """Resolve the graph backend name, preferring the CLI override."""
    return (override or config.get("graph_backend") or "hnsw").lower()


def _resolve_measure_cuda_memory(config: dict[str, Any]) -> bool:
    """Resolve whether CUDA memory metrics should be collected."""
    return bool(config.get("measure_cuda_memory", True))


def run_validate_graph_command(
    checkpoint_path: str | Path,
    config_files: list[str],
    config_overrides: dict[str, Any] | None,
    output_root: str | None = None,
) -> dict[str, Any]:
    """Validate sparse graph builders against the original dense reference.

    This command builds three graph variants on the original candidate pool:
    the released dense reference implementation, an exact FAISS flat graph, and
    an approximate HNSW graph. It then writes a JSON report describing exact and
    approximate overlap between these variants.

    Args:
        checkpoint_path: RPG checkpoint to evaluate.
        config_files: Ordered config files used to build the harness.
        config_overrides: Optional config overrides applied after file loading.
        output_root: Optional session-output root directory.

    Returns:
        A payload dictionary containing the report path, graph-build metadata,
        and comparison summaries.
    """
    harness = EvaluationHarness.build(
        checkpoint_path=checkpoint_path,
        config_files=config_files,
        config_overrides=config_overrides,
    )
    session_root = _session_root(harness.config, output_root=output_root)
    topk = int(harness.config.get("graph_topk", harness.config["n_edges"]))

    graph_records: list[dict[str, Any]] = []
    flat_start = time.perf_counter()
    exact_adjacency = build_sparse_adjacency(
        model=harness.model,
        backend="flat",
        topk=topk,
        config={**harness.config, "graph_backend": "flat"},
    )
    graph_records.append(
        {
            "backend": "flat",
            "pool_size": harness.dataset.n_items - 1,
            "topk": topk,
            "build_seconds": time.perf_counter() - flat_start,
            "cached": False,
        }
    )

    hnsw_start = time.perf_counter()
    hnsw_adjacency = build_sparse_adjacency(
        model=harness.model,
        backend="hnsw",
        topk=topk,
        config={**harness.config, "graph_backend": "hnsw"},
    )
    graph_records.append(
        {
            "backend": "hnsw",
            "pool_size": harness.dataset.n_items - 1,
            "topk": topk,
            "build_seconds": time.perf_counter() - hnsw_start,
            "cached": False,
            "graph_hnsw_m": int(harness.config.get("graph_hnsw_m", 32)),
            "graph_hnsw_ef_construction": int(
                harness.config.get("graph_hnsw_ef_construction", 200)
            ),
            "graph_hnsw_ef_search": int(
                harness.config.get("graph_hnsw_ef_search", max(256, topk * 2))
            ),
        }
    )

    reference_adjacency = build_dense_reference_adjacency(harness.model, topk=topk)
    dense_vs_flat = compare_adjacency_sets(reference_adjacency, exact_adjacency)
    comparisons = {
        "dense_vs_flat": {
            "purpose": (
                "Checks whether the FAISS exact-vector formulation reproduces "
                "the released dense graph on the original item pool."
            ),
            **compare_adjacency_overlap(reference_adjacency, exact_adjacency),
        },
        "flat_vs_hnsw": {
            "purpose": (
                "Checks the scalable HNSW backend used for enlarged-pool "
                "profiling against the exact FAISS flat backend."
            ),
            **compare_adjacency_overlap(exact_adjacency, hnsw_adjacency),
        },
        "dense_vs_hnsw": {
            "purpose": (
                "End-to-end reference check from the released dense graph to "
                "the scalable HNSW graph."
            ),
            **compare_adjacency_overlap(reference_adjacency, hnsw_adjacency),
        },
    }

    payload = {
        "topk": topk,
        "pool_size": harness.dataset.n_items - 1,
        "checkpoint_path": str(harness.checkpoint_path),
        "checkpoint_signature": checkpoint_signature(harness.checkpoint_path),
        "graph_records": graph_records,
        "comparisons": comparisons,
        "legacy_dense_vs_flat_exact": dense_vs_flat,
    }
    report_path = session_root / "graphs" / "validate_graph_report.json"
    report_path.write_text(json.dumps(payload, indent=2))

    return {"report_path": str(report_path), **payload}


def run_profile_command(
    checkpoint_path: str | Path,
    config_files: list[str],
    config_overrides: dict[str, Any] | None,
    output_root: str | None = None,
    pool_sizes_override: list[int] | None = None,
    prepare_only: bool = False,
    profile_only: bool = False,
    graph_backend_override: str | None = None,
    force_rebuild: bool = False,
) -> dict[str, str]:
    """Run the end-to-end inference profiling workflow for one checkpoint.

    For each requested candidate-pool size, the workflow optionally augments the
    dataset with dummy items, builds or loads the graph adjacency used during
    decoding, warms up the model, executes one or more timed evaluation passes,
    and writes raw plus aggregated metrics to disk.

    Args:
        checkpoint_path: RPG checkpoint to profile.
        config_files: Ordered config files used to build the harness.
        config_overrides: Optional config overrides applied after file loading.
        output_root: Optional explicit directory under which to create the
            session folder.
        pool_sizes_override: Optional CLI-provided pool sizes. Falls back to the
            merged config when omitted.
        prepare_only: If `True`, build/cache graphs only and skip evaluation.
        profile_only: If `True`, require all graph caches to exist already and
            fail instead of rebuilding them.
        graph_backend_override: Optional backend override such as `"flat"` or
            `"hnsw"`.
        force_rebuild: If `True`, rebuild graph caches even when matching cache
            files already exist.

    Returns:
        A manifest dictionary with the paths of the generated CSV/JSONL outputs.

    Raises:
        ValueError: If mutually exclusive mode flags are requested together.
        RuntimeError: If `profile_only=True` but a required graph cache is
            missing.
    """
    if prepare_only and profile_only:
        raise ValueError("prepare_only and profile_only cannot both be enabled.")

    preview_harness = EvaluationHarness.build(
        checkpoint_path=checkpoint_path,
        config_files=config_files,
        config_overrides=config_overrides,
    )
    session_root = _session_root(preview_harness.config, output_root=output_root)
    pool_sizes = _resolve_pool_sizes(preview_harness.config, pool_sizes_override)
    repeats = int(preview_harness.config.get("repeats", 1))
    graph_backend = _resolve_graph_backend(preview_harness.config, graph_backend_override)
    warmup_batches = int(preview_harness.config.get("warmup_batches", 0))
    measure_cuda_memory = _resolve_measure_cuda_memory(preview_harness.config)

    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []

    for pool_size in pool_sizes:
        harness = EvaluationHarness.build(
            checkpoint_path=checkpoint_path,
            config_files=config_files,
            config_overrides=config_overrides,
        )
        dummy_seed = int(harness.config.get("dummy_pool_seed", harness.config["rand_seed"]))
        augmentation = augment_candidate_pool(
            dataset=harness.dataset,
            tokenizer=harness.tokenizer,
            model=harness.model,
            target_pool_size=pool_size,
            seed=dummy_seed,
        )

        adjacency, graph_record = build_or_load_adjacency(
            model=harness.model,
            checkpoint_path=harness.checkpoint_path,
            config=harness.config,
            pool_size=pool_size,
            backend=graph_backend,
            force_rebuild=force_rebuild,
        )
        graph_rows.append(asdict(graph_record))

        if prepare_only:
            continue

        if profile_only and not graph_record.loaded_from_cache:
            raise RuntimeError(
                "profile_only was requested but the adjacency cache was missing."
            )

        harness.model.adjacency = adjacency.to(harness.config["device"])
        harness.model.init_flag = True
        harness.model.generate_w_decoding_graph = True

        pool_repeat_rows: list[dict[str, Any]] = []
        for repeat_index in range(repeats):
            repeat_seed = _set_repeat_seed(harness.config["rand_seed"], repeat_index)
            harness.warmup(warmup_batches)
            _set_repeat_seed(harness.config["rand_seed"], repeat_index)

            (
                eval_results,
                elapsed_seconds,
                peak_allocated,
                peak_reserved,
                baseline_allocated,
                baseline_reserved,
                runtime_delta_allocated,
                runtime_delta_reserved,
            ) = _profile_epoch(harness=harness, measure_cuda_memory=measure_cuda_memory)

            visited_items = float(eval_results["n_visited_items"])
            visited_ratio = visited_items / float(pool_size)

            row = {
                "method": harness.config["model"],
                "dataset": harness.config["dataset"],
                "category": harness.config["category"],
                "pool_size": int(pool_size),
                "graph_backend": graph_backend,
                "graph_topk": int(harness.config.get("graph_topk", harness.config["n_edges"])),
                "repeat_index": repeat_index,
                "repeat_seed": repeat_seed,
                "epoch_time_s": elapsed_seconds,
                "baseline_cuda_allocated_gb": baseline_allocated,
                "baseline_cuda_reserved_gb": baseline_reserved,
                "peak_cuda_allocated_gb": peak_allocated,
                "peak_cuda_reserved_gb": peak_reserved,
                "peak_cuda_runtime_delta_allocated_gb": runtime_delta_allocated,
                "peak_cuda_runtime_delta_reserved_gb": runtime_delta_reserved,
                "n_visited_items": visited_items,
                "visited_ratio": visited_ratio,
                "ndcg_at_10": float(eval_results.get("ndcg@10", float("nan"))),
                "graph_cache_id": graph_record.cache_id,
                "graph_loaded_from_cache": graph_record.loaded_from_cache,
                "checkpoint_path": str(harness.checkpoint_path),
                "checkpoint_signature": graph_record.checkpoint_signature,
                "dummy_seed": augmentation.seed,
                "dummy_items_added": augmentation.added_items,
            }
            raw_rows.append(row)
            pool_repeat_rows.append(row)

        if pool_repeat_rows:
            summary_rows.append(
                {
                    "method": harness.config["model"],
                    "dataset": harness.config["dataset"],
                    "category": harness.config["category"],
                    "pool_size": int(pool_size),
                    "graph_backend": graph_backend,
                    "graph_topk": int(harness.config.get("graph_topk", harness.config["n_edges"])),
                    "epoch_time_s_median": statistics.median(
                        row["epoch_time_s"] for row in pool_repeat_rows
                    ),
                    "baseline_cuda_allocated_gb_median": _median_or_nan(
                        [row["baseline_cuda_allocated_gb"] for row in pool_repeat_rows]
                    ),
                    "baseline_cuda_reserved_gb_median": _median_or_nan(
                        [row["baseline_cuda_reserved_gb"] for row in pool_repeat_rows]
                    ),
                    "peak_cuda_allocated_gb_median": _median_or_nan(
                        [row["peak_cuda_allocated_gb"] for row in pool_repeat_rows]
                    ),
                    "peak_cuda_reserved_gb_median": _median_or_nan(
                        [row["peak_cuda_reserved_gb"] for row in pool_repeat_rows]
                    ),
                    "peak_cuda_runtime_delta_allocated_gb_median": _median_or_nan(
                        [row["peak_cuda_runtime_delta_allocated_gb"] for row in pool_repeat_rows]
                    ),
                    "peak_cuda_runtime_delta_reserved_gb_median": _median_or_nan(
                        [row["peak_cuda_runtime_delta_reserved_gb"] for row in pool_repeat_rows]
                    ),
                    "n_visited_items_median": statistics.median(
                        row["n_visited_items"] for row in pool_repeat_rows
                    ),
                    "visited_ratio_median": statistics.median(
                        row["visited_ratio"] for row in pool_repeat_rows
                    ),
                    "ndcg_at_10_median": statistics.median(
                        row["ndcg_at_10"] for row in pool_repeat_rows
                    ),
                    "graph_cache_id": graph_record.cache_id,
                    "checkpoint_signature": graph_record.checkpoint_signature,
                    "dummy_items_added": augmentation.added_items,
                }
            )

    raw_csv = session_root / "raw" / "profile_runs.csv"
    raw_jsonl = session_root / "raw" / "profile_runs.jsonl"
    summary_csv = session_root / "summaries" / "profile_summary.csv"
    summary_jsonl = session_root / "summaries" / "profile_summary.jsonl"
    graph_csv = session_root / "graphs" / "graph_builds.csv"
    graph_jsonl = session_root / "graphs" / "graph_builds.jsonl"

    _write_csv(raw_csv, raw_rows)
    _write_jsonl(raw_jsonl, raw_rows)
    _write_csv(summary_csv, summary_rows)
    _write_jsonl(summary_jsonl, summary_rows)
    _write_csv(graph_csv, graph_rows)
    _write_jsonl(graph_jsonl, graph_rows)

    manifest = {
        "session_root": str(session_root),
        "raw_csv": str(raw_csv),
        "raw_jsonl": str(raw_jsonl),
        "summary_csv": str(summary_csv),
        "summary_jsonl": str(summary_jsonl),
        "graph_csv": str(graph_csv),
        "graph_jsonl": str(graph_jsonl),
    }
    (session_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
