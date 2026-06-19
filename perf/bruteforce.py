from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from rpg_graph_analysis.dynamic_trace import compute_decoding_context
from rpg_graph_analysis.reranking.scorers import rpg_candidate_scores

from .config import checkpoint_signature
from .harness import EvaluationHarness
from .pool import augment_candidate_pool
from .profile import (
    _bytes_to_gib,
    _maybe_cuda_synchronize,
    _median_or_nan,
    _resolve_measure_cuda_memory,
    _resolve_pool_sizes,
    _session_root,
    _set_repeat_seed,
    _write_csv,
    _write_jsonl,
)


def _resolve_item_chunk_size(config: dict[str, Any]) -> int:
    """Return the item chunk size used by exact RPG brute-force scoring.

    Chunking changes only the memory schedule: every item is still scored
    exactly once. Larger chunks reduce Python-loop overhead but increase peak
    CUDA memory. The default is conservative for A100 40GB with the repo's
    default RPG eval batch size.
    """

    value = int(config.get("bruteforce_item_chunk_size", 50_000))
    if value <= 0:
        raise ValueError(f"bruteforce_item_chunk_size must be positive, got {value}.")
    return value


def _chunked_topk_predictions(
    *,
    harness: EvaluationHarness,
    batch: dict[str, torch.Tensor],
    topk: int,
    item_chunk_size: int,
) -> torch.Tensor:
    """Score all RPG items exactly while keeping only each row's best top-k.

    The upstream no-graph RPG path materializes scores for all items at once.
    This function computes the same item scores in item chunks and merges the
    per-chunk top-k lists. Since every item is evaluated, the result is an exact
    brute-force top-k, modulo unavoidable tie-order differences.
    """

    device = harness.accelerator.device
    context = compute_decoding_context(harness.model, batch)
    batch_size = int(batch["labels"].shape[0])
    predictions, _ = _chunked_topk_from_logits(
        token_logits=context.token_logits,
        item_id2tokens=harness.model.item_id2tokens.to(device),
        n_items=int(harness.dataset.n_items),
        batch_size=batch_size,
        topk=topk,
        item_chunk_size=item_chunk_size,
        device=device,
    )
    return predictions


def _chunked_topk_from_logits(
    *,
    token_logits: torch.Tensor,
    item_id2tokens: torch.Tensor,
    n_items: int,
    batch_size: int,
    topk: int,
    item_chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact top-k item IDs and scores from precomputed RPG logits."""

    all_item_ids = torch.arange(1, n_items, dtype=torch.long, device=device)

    best_scores: torch.Tensor | None = None
    best_ids: torch.Tensor | None = None

    for start in range(0, all_item_ids.numel(), item_chunk_size):
        chunk_ids = all_item_ids[start : start + item_chunk_size]
        candidate_ids = chunk_ids.unsqueeze(0).expand(batch_size, -1)
        scores = rpg_candidate_scores(token_logits, item_id2tokens, candidate_ids)

        local_k = min(topk, scores.shape[1])
        chunk_scores, chunk_positions = torch.topk(scores, k=local_k, dim=-1)
        chunk_top_ids = chunk_ids[chunk_positions]

        if best_scores is None or best_ids is None:
            best_scores = chunk_scores
            best_ids = chunk_top_ids
            continue

        merged_scores = torch.cat([best_scores, chunk_scores], dim=-1)
        merged_ids = torch.cat([best_ids, chunk_top_ids], dim=-1)
        keep_k = min(topk, merged_scores.shape[1])
        best_scores, keep_positions = torch.topk(merged_scores, k=keep_k, dim=-1)
        best_ids = torch.gather(merged_ids, dim=-1, index=keep_positions)

    if best_ids is None:
        raise RuntimeError("No candidate items were scored.")
    if best_scores is None:
        raise RuntimeError("No candidate scores were retained.")
    return best_ids[:, :topk], best_scores[:, :topk]


def _mean_metrics_from_batches(metric_sums: dict[str, float], n_examples: int) -> dict[str, float]:
    """Convert accumulated per-example metric sums into means."""

    if n_examples <= 0:
        raise RuntimeError("Cannot summarize brute-force metrics with zero examples.")
    return {key: value / float(n_examples) for key, value in metric_sums.items()}


def _evaluate_bruteforce_epoch(
    *,
    harness: EvaluationHarness,
    item_chunk_size: int,
    desc: str,
) -> dict[str, float]:
    """Evaluate exact all-item RPG scoring for one full test epoch."""

    device = harness.accelerator.device
    maxk = int(harness.trainer.evaluator.maxk)
    pool_size = int(harness.dataset.n_items - 1)
    metric_sums: dict[str, float] = {}
    n_examples = 0

    harness.model.generate_w_decoding_graph = False
    harness.model.eval()

    with torch.no_grad():
        progress = tqdm(harness.test_dataloader, total=len(harness.test_dataloader), desc=desc)
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            batch_size = int(batch["labels"].shape[0])
            predictions = _chunked_topk_predictions(
                harness=harness,
                batch=batch,
                topk=maxk,
                item_chunk_size=item_chunk_size,
            )
            visited_counts = torch.full(
                (batch_size, 1),
                float(pool_size),
                dtype=torch.float32,
                device=device,
            )
            results = harness.trainer.evaluator.calculate_metrics(
                (predictions.unsqueeze(-1), visited_counts),
                batch["labels"],
            )
            for key, value in results.items():
                if key == "n_visited_items":
                    continue
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value.float().sum())
            n_examples += batch_size

    metrics = _mean_metrics_from_batches(metric_sums, n_examples)
    metrics["n_visited_items"] = float(pool_size)
    return metrics


def _warmup_bruteforce(
    *,
    harness: EvaluationHarness,
    item_chunk_size: int,
    warmup_batches: int,
) -> None:
    """Run a few exact-scoring batches before timed profiling."""

    if warmup_batches <= 0:
        return

    device = harness.accelerator.device
    maxk = int(harness.trainer.evaluator.maxk)
    harness.model.generate_w_decoding_graph = False
    harness.model.eval()

    with torch.no_grad():
        for batch_index, batch in enumerate(harness.test_dataloader):
            if batch_index >= warmup_batches:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            _ = _chunked_topk_predictions(
                harness=harness,
                batch=batch,
                topk=maxk,
                item_chunk_size=item_chunk_size,
            )
    _maybe_cuda_synchronize(device)


def _profile_bruteforce_epoch(
    *,
    harness: EvaluationHarness,
    item_chunk_size: int,
    measure_cuda_memory: bool,
) -> tuple[dict[str, float], float, float, float, float, float, float, float]:
    """Profile one exact brute-force RPG evaluation epoch."""

    device = harness.accelerator.device
    baseline_allocated = float("nan")
    baseline_reserved = float("nan")
    if device.type == "cuda" and measure_cuda_memory:
        _maybe_cuda_synchronize(device)
        baseline_allocated = _bytes_to_gib(torch.cuda.memory_allocated(device=device))
        baseline_reserved = _bytes_to_gib(torch.cuda.memory_reserved(device=device))
        torch.cuda.reset_peak_memory_stats(device=device)

    start_time = time.perf_counter()
    results = _evaluate_bruteforce_epoch(
        harness=harness,
        item_chunk_size=item_chunk_size,
        desc="RPG brute-force scoring",
    )
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


def _validate_chunked_matches_upstream(
    *,
    harness: EvaluationHarness,
    item_chunk_size: int,
    n_batches: int,
) -> dict[str, Any]:
    """Check chunked scoring against upstream no-graph RPG on original items."""

    if n_batches <= 0:
        return {"enabled": False, "checked_batches": 0}

    device = harness.accelerator.device
    maxk = int(harness.trainer.evaluator.maxk)
    checked_batches = 0
    checked_examples = 0
    harness.model.generate_w_decoding_graph = False
    harness.model.eval()

    with torch.no_grad():
        for batch_index, batch in enumerate(harness.test_dataloader):
            if batch_index >= n_batches:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            context = compute_decoding_context(harness.model, batch)
            item_id2tokens = harness.model.item_id2tokens.to(device)
            chunked, chunked_scores = _chunked_topk_from_logits(
                token_logits=context.token_logits,
                item_id2tokens=item_id2tokens,
                n_items=int(harness.dataset.n_items),
                batch_size=int(batch["labels"].shape[0]),
                topk=maxk,
                item_chunk_size=item_chunk_size,
                device=device,
            )
            upstream = harness.model.generate(batch, n_return_sequences=maxk).squeeze(-1)
            if not torch.equal(chunked, upstream):
                upstream_scores = rpg_candidate_scores(context.token_logits, item_id2tokens, upstream)
                chunked_score_set = torch.sort(chunked_scores, dim=-1).values
                upstream_score_set = torch.sort(upstream_scores, dim=-1).values
                if torch.allclose(chunked_score_set, upstream_score_set, rtol=1e-5, atol=1e-6):
                    checked_batches += 1
                    checked_examples += int(batch["labels"].shape[0])
                    continue
                mismatch = (chunked != upstream).nonzero(as_tuple=False)[0].tolist()
                row, rank = int(mismatch[0]), int(mismatch[1])
                raise AssertionError(
                    "Chunked RPG brute-force predictions do not match upstream "
                    f"no-graph generate() at batch={batch_index}, row={row}, rank={rank}: "
                    f"chunked={int(chunked[row, rank])}, upstream={int(upstream[row, rank])}."
                )
            checked_batches += 1
            checked_examples += int(batch["labels"].shape[0])

    return {
        "enabled": True,
        "checked_batches": checked_batches,
        "checked_examples": checked_examples,
        "item_chunk_size": item_chunk_size,
    }


def run_bruteforce_profile_command(
    checkpoint_path: str | Path,
    config_files: list[str],
    config_overrides: dict[str, Any] | None,
    output_root: str | None = None,
    pool_sizes_override: list[int] | None = None,
    item_chunk_size_override: int | None = None,
    skip_parity_check: bool = False,
) -> dict[str, str]:
    """Profile exact RPG all-item scoring over enlarged candidate pools.

    This is the no-graph RPG baseline for the perf experiment. It uses the same
    deterministic dummy-item pool expansion as graph profiling, but evaluates
    every candidate item exactly with RPG's semantic-token scorer. Item chunking
    only controls peak memory; it does not approximate the result.
    """

    preview_harness = EvaluationHarness.build(
        checkpoint_path=checkpoint_path,
        config_files=config_files,
        config_overrides=config_overrides,
    )
    session_root = _session_root(preview_harness.config, output_root=output_root)
    pool_sizes = _resolve_pool_sizes(preview_harness.config, pool_sizes_override)
    repeats = int(preview_harness.config.get("repeats", 1))
    warmup_batches = int(preview_harness.config.get("warmup_batches", 0))
    measure_cuda_memory = _resolve_measure_cuda_memory(preview_harness.config)
    item_chunk_size = (
        int(item_chunk_size_override)
        if item_chunk_size_override is not None
        else _resolve_item_chunk_size(preview_harness.config)
    )
    if item_chunk_size <= 0:
        raise ValueError(f"item_chunk_size must be positive, got {item_chunk_size}.")

    parity_batches = 0 if skip_parity_check else int(
        preview_harness.config.get("bruteforce_parity_batches", 1)
    )
    parity_report = _validate_chunked_matches_upstream(
        harness=preview_harness,
        item_chunk_size=item_chunk_size,
        n_batches=parity_batches,
    )
    (session_root / "parity_report.json").write_text(json.dumps(parity_report, indent=2))

    if preview_harness.accelerator.device.type == "cuda":
        del preview_harness
        torch.cuda.empty_cache()
    else:
        del preview_harness

    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    ckpt_signature = checkpoint_signature(checkpoint_path)

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
        harness.model.generate_w_decoding_graph = False
        harness.model.init_flag = True
        if hasattr(harness.model, "adjacency"):
            harness.model.adjacency = None

        pool_repeat_rows: list[dict[str, Any]] = []
        for repeat_index in range(repeats):
            repeat_seed = _set_repeat_seed(harness.config["rand_seed"], repeat_index)
            _warmup_bruteforce(
                harness=harness,
                item_chunk_size=item_chunk_size,
                warmup_batches=warmup_batches,
            )
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
            ) = _profile_bruteforce_epoch(
                harness=harness,
                item_chunk_size=item_chunk_size,
                measure_cuda_memory=measure_cuda_memory,
            )

            visited_items = float(eval_results["n_visited_items"])
            visited_ratio = visited_items / float(pool_size)
            row = {
                "method": "RPG-BruteForce",
                "dataset": harness.config["dataset"],
                "category": harness.config["category"],
                "pool_size": int(pool_size),
                "graph_backend": "full_sort_chunked",
                "graph_topk": 0,
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
                "recall_at_10": float(eval_results.get("recall@10", float("nan"))),
                "graph_cache_id": "",
                "graph_loaded_from_cache": False,
                "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
                "checkpoint_signature": ckpt_signature,
                "dummy_seed": augmentation.seed,
                "dummy_items_added": augmentation.added_items,
                "original_pool_size": augmentation.original_pool_size,
                "bruteforce_item_chunk_size": item_chunk_size,
            }
            raw_rows.append(row)
            pool_repeat_rows.append(row)

        if pool_repeat_rows:
            summary_rows.append(
                {
                    "method": "RPG-BruteForce",
                    "dataset": harness.config["dataset"],
                    "category": harness.config["category"],
                    "pool_size": int(pool_size),
                    "graph_backend": "full_sort_chunked",
                    "graph_topk": 0,
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
                    "recall_at_10_median": statistics.median(
                        row["recall_at_10"] for row in pool_repeat_rows
                    ),
                    "graph_cache_id": "",
                    "checkpoint_signature": ckpt_signature,
                    "dummy_items_added": augmentation.added_items,
                    "original_pool_size": augmentation.original_pool_size,
                    "bruteforce_item_chunk_size": item_chunk_size,
                }
            )

        if harness.accelerator.device.type == "cuda":
            del harness
            torch.cuda.empty_cache()
        else:
            del harness

    raw_csv = session_root / "raw" / "profile_runs.csv"
    raw_jsonl = session_root / "raw" / "profile_runs.jsonl"
    summary_csv = session_root / "summaries" / "profile_summary.csv"
    summary_jsonl = session_root / "summaries" / "profile_summary.jsonl"

    _write_csv(raw_csv, raw_rows)
    _write_jsonl(raw_jsonl, raw_rows)
    _write_csv(summary_csv, summary_rows)
    _write_jsonl(summary_jsonl, summary_rows)

    manifest = {
        "session_root": str(session_root),
        "raw_csv": str(raw_csv),
        "raw_jsonl": str(raw_jsonl),
        "summary_csv": str(summary_csv),
        "summary_jsonl": str(summary_jsonl),
        "parity_report": str(session_root / "parity_report.json"),
        "bruteforce_item_chunk_size": item_chunk_size,
    }
    (session_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
