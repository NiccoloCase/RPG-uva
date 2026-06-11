#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.sasrec_modernized import SASRecModernizedDataset, SASRecModernizedModel  # noqa: E402
from models.sasrec_modernized.utils import get_user_seqs, set_seed  # noqa: E402
from sasrec_modernized import (  # noqa: E402
    PRESET_CONFIGS,
    build_config_files,
    load_config,
    normalize_config,
    parse_override_args,
)


DEFAULT_POOL_SIZES = [20000, 50000, 100000, 200000, 500000]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile SASRec inference over enlarged candidate pools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile_parser = subparsers.add_parser(
        "profile",
        help="Profile full-sort SASRec inference over enlarged candidate pools.",
    )
    profile_parser.add_argument("--checkpoint", required=True, help="Path to a trained SASRec checkpoint.")
    profile_parser.add_argument("--preset", choices=sorted(PRESET_CONFIGS), help="Named SASRec preset to apply.")
    profile_parser.add_argument("--dataset", default=None, help="Dataset/category override.")
    profile_parser.add_argument("--config", action="append", default=[], help="Additional YAML config file.")
    profile_parser.add_argument("--no-root-config", action="store_true", help="Skip the default SASRec root config.")
    profile_parser.add_argument("--no-local-config", action="store_true", help="Skip the local SASRec config.")
    profile_parser.add_argument("--output-dir", default=None, help="Optional session root override.")
    profile_parser.add_argument("--pool-sizes", default=None, help="Comma-separated pool sizes.")

    plot_parser = subparsers.add_parser(
        "plot",
        help="Render a two-panel plot from a summary CSV or profiling session directory.",
    )
    plot_parser.add_argument("--input", required=True, help="Summary CSV path or profiling session directory.")
    plot_parser.add_argument("--output", required=True, help="Output image path.")

    return parser


def _session_root(output_root: str | None = None) -> Path:
    raw_root = output_root or "artifacts/sasrec/perf/sports"
    path = Path(raw_root).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    session_name = timestamp if not slurm_job_id else f"{timestamp}_job{slurm_job_id}"
    session_root = path.resolve() / session_name
    suffix = 1
    while session_root.exists():
        session_root = path.resolve() / f"{session_name}_{suffix:02d}"
        suffix += 1

    (session_root / "raw").mkdir(parents=True, exist_ok=True)
    (session_root / "summaries").mkdir(parents=True, exist_ok=True)
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


def _parse_int_list(raw_value: str | None, default: list[int]) -> list[int]:
    if raw_value is None:
        return default
    values = [int(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one pool size must be provided.")
    return values


def _median_or_nan(values: list[float]) -> float:
    cleaned = [value for value in values if not np.isnan(value)]
    if not cleaned:
        return float("nan")
    return float(statistics.median(cleaned))


def _maybe_cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _bytes_to_gib(value: int) -> float:
    return float(value) / (1024 ** 3)


def _checkpoint_signature(checkpoint_path: str | Path) -> str:
    path = Path(checkpoint_path).expanduser().resolve()
    stat = path.stat()
    payload = f"{path}:{stat.st_size}:{int(stat.st_mtime)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _set_repeat_seed(base_seed: int, repeat_index: int) -> int:
    seed = int(base_seed) + int(repeat_index)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def _load_args(parsed_args: argparse.Namespace, override_tokens: list[str]) -> SimpleNamespace:
    overrides = parse_override_args(override_tokens)
    if parsed_args.dataset is not None:
        overrides["dataset"] = parsed_args.dataset
    config_files = build_config_files(parsed_args)
    merged_config = load_config(config_files, overrides)
    return normalize_config(merged_config, parsed_args.checkpoint)


def _build_base_model(args: SimpleNamespace, checkpoint_path: str, device: torch.device):
    model = SASRecModernizedModel(args).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _expand_item_embeddings(
    model: SASRecModernizedModel,
    target_item_size: int,
    original_candidate_count: int,
    seed: int,
) -> int:
    """Expand item embeddings by deterministic cloning of existing real item rows."""
    current_size = model.item_embeddings.num_embeddings
    if target_item_size < current_size:
        raise ValueError(
            f"Target item size {target_item_size} is smaller than checkpoint item size {current_size}."
        )
    if target_item_size == current_size:
        return 0

    current_weight = model.item_embeddings.weight.data
    extra_count = target_item_size - current_size
    source_count = original_candidate_count
    source_offset = seed % max(source_count, 1)
    source_ids = ((torch.arange(extra_count, device=current_weight.device) + source_offset) % source_count) + 1
    extra_weight = current_weight[source_ids].clone()
    expanded_weight = torch.cat([current_weight, extra_weight], dim=0)
    model.item_embeddings = torch.nn.Embedding.from_pretrained(
        expanded_weight,
        freeze=False,
        padding_idx=0,
    )
    return extra_count


def _build_test_dataloader(args: SimpleNamespace, user_seq: list[list[int]]) -> DataLoader:
    test_dataset = SASRecModernizedDataset(args, user_seq, data_type="test")
    return DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.eval_batch_size,
    )


def _mask_invalid_and_seen_items(
    rating_pred: np.ndarray,
    args: SimpleNamespace,
    batch_user_index: np.ndarray,
) -> None:
    rating_pred[:, 0] = -np.inf
    if 0 <= args.mask_id < rating_pred.shape[1]:
        rating_pred[:, args.mask_id] = -np.inf
    seen = args.train_matrix[batch_user_index].toarray() > 0
    rating_pred[:, : seen.shape[1]][seen] = -np.inf


def _evaluate_epoch(
    model: SASRecModernizedModel,
    dataloader: DataLoader,
    args: SimpleNamespace,
    device: torch.device,
) -> dict[str, float]:
    topk_max = max(args.topk)
    metric_values = {metric: [] for metric in ("recall@5", "ndcg@5", "recall@10", "ndcg@10")}
    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            user_ids, input_ids, _, _, answers = batch
            sequence_output = model(input_ids)
            recommend_output = sequence_output[:, -1, :]
            rating_pred = torch.matmul(recommend_output, model.item_embeddings.weight.transpose(0, 1))
            rating_pred = rating_pred.cpu().numpy().copy()
            batch_user_index = user_ids.cpu().numpy()
            _mask_invalid_and_seen_items(rating_pred, args, batch_user_index)

            ind = np.argpartition(rating_pred, -topk_max)[:, -topk_max:]
            arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
            arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
            pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
            targets = answers.cpu().numpy().reshape(-1)

            for target, predictions in zip(targets, pred_list):
                for k in (5, 10):
                    topk = predictions[:k].tolist()
                    if int(target) in topk:
                        rank = topk.index(int(target))
                        metric_values[f"recall@{k}"].append(1.0)
                        metric_values[f"ndcg@{k}"].append(float(1.0 / np.log2(rank + 2)))
                    else:
                        metric_values[f"recall@{k}"].append(0.0)
                        metric_values[f"ndcg@{k}"].append(0.0)

    return {key: float(statistics.mean(values)) for key, values in metric_values.items()}


def _warmup(model, dataloader, args, device: torch.device, warmup_batches: int) -> None:
    if warmup_batches <= 0:
        return
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            if index >= warmup_batches:
                break
            batch = tuple(t.to(device) for t in batch)
            user_ids, input_ids, _, _, _ = batch
            sequence_output = model(input_ids)
            recommend_output = sequence_output[:, -1, :]
            rating_pred = torch.matmul(recommend_output, model.item_embeddings.weight.transpose(0, 1))
            rating_pred = rating_pred.cpu().numpy().copy()
            _mask_invalid_and_seen_items(rating_pred, args, user_ids.cpu().numpy())
            np.argpartition(rating_pred, -max(args.topk))[:, -max(args.topk):]
    _maybe_cuda_synchronize(device)


def _profile_epoch(
    model: SASRecModernizedModel,
    dataloader: DataLoader,
    args: SimpleNamespace,
    device: torch.device,
    measure_cuda_memory: bool,
) -> tuple[dict[str, float], float, float, float, float, float, float, float]:
    baseline_allocated = float("nan")
    baseline_reserved = float("nan")
    if device.type == "cuda" and measure_cuda_memory:
        _maybe_cuda_synchronize(device)
        baseline_allocated = _bytes_to_gib(torch.cuda.memory_allocated(device=device))
        baseline_reserved = _bytes_to_gib(torch.cuda.memory_reserved(device=device))
        torch.cuda.reset_peak_memory_stats(device=device)

    start_time = time.perf_counter()
    results = _evaluate_epoch(model, dataloader, args, device)
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


def run_profile(parsed_args: argparse.Namespace, override_tokens: list[str]) -> dict[str, str]:
    args = _load_args(parsed_args, override_tokens)
    if not Path(args.data_file).is_file():
        raise FileNotFoundError(f"Missing SASRec data file: {args.data_file}")
    if not Path(args.checkpoint_path).is_file():
        raise FileNotFoundError(f"SASRec checkpoint not found: {args.checkpoint_path}")

    set_seed(args.seed)
    user_seq, max_item, _, test_rating_matrix = get_user_seqs(args.data_file)
    original_item_size = max_item + 2
    original_pool_size = max_item
    args.item_size = original_item_size
    args.mask_id = max_item + 1
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    args.train_matrix = test_rating_matrix

    config_pool_sizes = getattr(args, "pool_sizes", DEFAULT_POOL_SIZES)
    pool_sizes = _parse_int_list(parsed_args.pool_sizes, [int(value) for value in config_pool_sizes])
    repeats = int(getattr(args, "repeats", 1))
    warmup_batches = int(getattr(args, "warmup_batches", 0))
    measure_cuda_memory = bool(getattr(args, "measure_cuda_memory", True))
    dummy_seed = int(getattr(args, "dummy_pool_seed", args.seed))
    ckpt_signature = _checkpoint_signature(args.checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    dataloader = _build_test_dataloader(args, user_seq)
    session_root = _session_root(parsed_args.output_dir or getattr(args, "perf_output_dir", None))

    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for pool_size in pool_sizes:
        target_item_size = int(pool_size) + 2
        model = _build_base_model(args, args.checkpoint_path, device)
        dummy_items_added = _expand_item_embeddings(
            model,
            target_item_size,
            original_candidate_count=original_pool_size,
            seed=dummy_seed,
        )
        model.to(device)
        model.eval()

        pool_repeat_rows: list[dict[str, Any]] = []
        for repeat_index in range(repeats):
            repeat_seed = _set_repeat_seed(args.seed, repeat_index)
            _warmup(model, dataloader, args, device, warmup_batches)
            _set_repeat_seed(args.seed, repeat_index)
            (
                eval_results,
                elapsed_seconds,
                peak_allocated,
                peak_reserved,
                baseline_allocated,
                baseline_reserved,
                runtime_delta_allocated,
                runtime_delta_reserved,
            ) = _profile_epoch(model, dataloader, args, device, measure_cuda_memory)

            row = {
                "method": "SASRec",
                "dataset": "AmazonReviews2014",
                "category": args.data_name,
                "pool_size": int(pool_size),
                "graph_backend": "full_sort",
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
                "n_visited_items": float(pool_size),
                "visited_ratio": 1.0,
                "ndcg_at_10": float(eval_results.get("ndcg@10", float("nan"))),
                "graph_cache_id": "",
                "graph_loaded_from_cache": False,
                "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
                "checkpoint_signature": ckpt_signature,
                "dummy_seed": dummy_seed,
                "dummy_items_added": dummy_items_added,
                "original_pool_size": original_pool_size,
            }
            raw_rows.append(row)
            pool_repeat_rows.append(row)

        if pool_repeat_rows:
            summary_rows.append(
                {
                    "method": "SASRec",
                    "dataset": "AmazonReviews2014",
                    "category": args.data_name,
                    "pool_size": int(pool_size),
                    "graph_backend": "full_sort",
                    "graph_topk": 0,
                    "epoch_time_s_median": statistics.median(row["epoch_time_s"] for row in pool_repeat_rows),
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
                    "n_visited_items_median": statistics.median(row["n_visited_items"] for row in pool_repeat_rows),
                    "visited_ratio_median": statistics.median(row["visited_ratio"] for row in pool_repeat_rows),
                    "ndcg_at_10_median": statistics.median(row["ndcg_at_10"] for row in pool_repeat_rows),
                    "graph_cache_id": "",
                    "checkpoint_signature": ckpt_signature,
                    "dummy_items_added": dummy_items_added,
                    "original_pool_size": original_pool_size,
                }
            )

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
    }
    (session_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _plot_summary_csv(input_path: str | Path, output_path: str | Path) -> Path:
    path = Path(input_path).expanduser().resolve()
    if path.is_dir():
        path = path / "summaries" / "profile_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {path}")

    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("matplotlib and pandas are required for plotting.") from exc

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("No rows found in the summary CSV.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    frame = frame.sort_values("pool_size")
    axes[0].plot(frame["pool_size"], frame["epoch_time_s_median"], marker="o", label="SASRec")
    axes[0].set_title("Inference Time vs Item Pool Size")
    axes[0].set_xlabel("Item pool size")
    axes[0].set_ylabel("Epoch time (s)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        frame["pool_size"],
        frame["peak_cuda_runtime_delta_allocated_gb_median"],
        marker="o",
        label="SASRec",
    )
    axes[1].set_title("Peak CUDA Runtime Memory vs Item Pool Size")
    axes[1].set_xlabel("Item pool size")
    axes[1].set_ylabel("Peak CUDA runtime delta (GB)")
    axes[1].grid(True, alpha=0.3)
    for axis in axes:
        axis.legend()

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, override_tokens = parser.parse_known_args(argv)

    if args.command == "plot":
        output = _plot_summary_csv(args.input, args.output)
        print(output)
        return 0

    if args.command == "profile":
        manifest = run_profile(args, override_tokens)
        print(manifest["session_root"])
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
