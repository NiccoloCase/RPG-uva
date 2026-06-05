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

from .graph import (
    build_dense_reference_adjacency,
    build_or_load_adjacency,
    compare_adjacency_sets,
)
from .harness import EvaluationHarness
from .pool import augment_candidate_pool


def _session_root(config: dict[str, Any], output_root: str | None = None) -> Path:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _median_or_nan(values: list[float]) -> float:
    cleaned = [value for value in values if not math.isnan(value)]
    if not cleaned:
        return float("nan")
    return float(statistics.median(cleaned))


def _set_repeat_seed(base_seed: int, repeat_index: int) -> int:
    seed = int(base_seed) + int(repeat_index)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def _maybe_cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _profile_epoch(
    harness: EvaluationHarness,
    measure_cuda_memory: bool,
) -> tuple[dict[str, float], float, float, float]:
    device = harness.config["device"]

    if device.type == "cuda" and measure_cuda_memory:
        torch.cuda.reset_peak_memory_stats(device=device)
    _maybe_cuda_synchronize(device)

    start_time = time.perf_counter()
    results = harness.evaluate()
    _maybe_cuda_synchronize(device)
    elapsed_seconds = time.perf_counter() - start_time

    if device.type == "cuda" and measure_cuda_memory:
        peak_allocated = torch.cuda.max_memory_allocated(device=device) / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved(device=device) / (1024 ** 3)
    else:
        peak_allocated = float("nan")
        peak_reserved = float("nan")

    return results, elapsed_seconds, peak_allocated, peak_reserved


def _resolve_pool_sizes(config: dict[str, Any], pool_sizes_override: list[int] | None) -> list[int]:
    if pool_sizes_override:
        return pool_sizes_override
    if "pool_sizes" not in config or not config["pool_sizes"]:
        raise ValueError("No pool_sizes were provided via config or CLI override.")
    return [int(value) for value in config["pool_sizes"]]


def _resolve_graph_backend(config: dict[str, Any], override: str | None) -> str:
    return (override or config.get("graph_backend") or "hnsw").lower()


def _resolve_measure_cuda_memory(config: dict[str, Any]) -> bool:
    return bool(config.get("measure_cuda_memory", True))


def run_validate_graph_command(
    checkpoint_path: str | Path,
    config_files: list[str],
    config_overrides: dict[str, Any] | None,
    output_root: str | None = None,
) -> dict[str, Any]:
    harness = EvaluationHarness.build(
        checkpoint_path=checkpoint_path,
        config_files=config_files,
        config_overrides=config_overrides,
    )
    session_root = _session_root(harness.config, output_root=output_root)
    topk = int(harness.config.get("graph_topk", harness.config["n_edges"]))

    exact_adjacency, graph_record = build_or_load_adjacency(
        model=harness.model,
        checkpoint_path=harness.checkpoint_path,
        config={**harness.config, "graph_backend": "flat"},
        pool_size=harness.dataset.n_items - 1,
        backend="flat",
        force_rebuild=True,
    )
    reference_adjacency = build_dense_reference_adjacency(harness.model, topk=topk)
    comparison = compare_adjacency_sets(reference_adjacency, exact_adjacency)

    payload = {
        "topk": topk,
        "pool_size": harness.dataset.n_items - 1,
        "checkpoint_path": str(harness.checkpoint_path),
        "graph_record": asdict(graph_record),
        **comparison,
    }
    report_path = session_root / "graphs" / "validate_graph_report.json"
    report_path.write_text(json.dumps(payload, indent=2))

    if not comparison["match"]:
        raise SystemExit(
            "Exact sparse graph validation failed. See "
            f"{report_path} for mismatch details."
        )

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

            eval_results, elapsed_seconds, peak_allocated, peak_reserved = _profile_epoch(
                harness=harness,
                measure_cuda_memory=measure_cuda_memory,
            )

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
                "peak_cuda_allocated_gb": peak_allocated,
                "peak_cuda_reserved_gb": peak_reserved,
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
                    "peak_cuda_allocated_gb_median": _median_or_nan(
                        [row["peak_cuda_allocated_gb"] for row in pool_repeat_rows]
                    ),
                    "peak_cuda_reserved_gb_median": _median_or_nan(
                        [row["peak_cuda_reserved_gb"] for row in pool_repeat_rows]
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
