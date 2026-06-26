#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SASREC_SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SASREC_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SASREC_SCRIPT_DIR))

from models.sasrec import SASRecDataset, SASRecModel  # noqa: E402
from models.sasrec.utils import get_user_seqs, set_seed  # noqa: E402
from popularity_metrics import (  # noqa: E402
    ItemPopularity,
    assign_popularity_groups,
    compute_item_popularity,
    compute_long_tail_items,
    compute_profile_popularity,
    group_metric_summary,
    percentage_long_tail,
    popularity_metric_names,
    recommendation_popularity,
)
from sasrec import (  # noqa: E402
    PRESET_CONFIGS,
    build_config_files,
    load_config,
    normalize_config,
    parse_override_args,
)

DEFAULT_EVAL_SEEDS = "2024,2025,2026,2027,2028,2029,2030,2031,2032,2033"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a SASRec checkpoint with either the normal single-seed "
            "protocol or the RPG-style multi-eval-seed user-level aggregation."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a trained SASRec checkpoint.",
    )
    parser.add_argument(
        "--eval-mode",
        choices=("normal", "eval_seeds"),
        default="normal",
        help="normal runs one eval seed; eval_seeds runs the comma-separated seed list.",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="Single evaluation seed for --eval-mode normal. Defaults to the config seed.",
    )
    parser.add_argument(
        "--eval-seeds",
        default=DEFAULT_EVAL_SEEDS,
        help="Comma-separated evaluation seeds for --eval-mode eval_seeds.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_CONFIGS),
        help="Named SASRec preset to apply.",
    )
    parser.add_argument("--dataset", default=None, help="Dataset/category override.")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Additional YAML config file. May be provided multiple times.",
    )
    parser.add_argument(
        "--no-root-config",
        action="store_true",
        help="Skip configs/sasrec/root.yaml.",
    )
    parser.add_argument(
        "--no-local-config",
        action="store_true",
        help="Skip configs/sasrec/local.yaml even if it exists.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/sasrec/eval_seeds",
        help="Directory under which a timestamped eval session will be written.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=5000,
        help="Number of user-bootstrap resamples for confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=2024,
        help="Random seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Two-sided bootstrap confidence level for metric summaries.",
    )
    parser.add_argument(
        "--short-head-fraction",
        type=float,
        default=0.2,
        help="Fraction of items (by training-set popularity) treated as the short head for APLT.",
    )
    parser.add_argument(
        "--popularity-low-quantile",
        type=float,
        default=0.2,
        help="Users below this quantile of profile popularity are grouped as 'niche'.",
    )
    parser.add_argument(
        "--popularity-high-quantile",
        type=float,
        default=0.8,
        help="Users above this quantile of profile popularity are grouped as 'blockbuster_focused'.",
    )
    return parser


def _parse_seeds(raw_value: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("At least one eval seed must be provided.")
    return seeds


def _load_args(parsed_args: argparse.Namespace, override_tokens: list[str]) -> SimpleNamespace:
    overrides = parse_override_args(override_tokens)
    if parsed_args.dataset is not None:
        overrides["dataset"] = parsed_args.dataset
    config_files = build_config_files(parsed_args)
    merged_config = load_config(config_files, overrides)
    return normalize_config(merged_config, parsed_args.checkpoint)


def _metric_names(args: SimpleNamespace) -> list[str]:
    return [f"{metric}@{k}" for k in args.topk for metric in ("recall", "ndcg")]


def _all_metric_names(args: SimpleNamespace) -> list[str]:
    return _metric_names(args) + popularity_metric_names(sorted(int(k) for k in args.topk))


def _session_root(output_root: str | Path) -> Path:
    path = Path(output_root).expanduser()
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
    session_root.mkdir(parents=True, exist_ok=True)
    return session_root


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _nan_if_empty(values: list[float]) -> float:
    return float("nan") if not values else float(mean(values))


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(stdev(values))


def _bootstrap_ci(
    values: list[float],
    n_samples: int,
    seed: int,
    ci_level: float,
) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if n_samples <= 0:
        estimate = float(mean(values))
        return estimate, estimate
    if not 0 < ci_level < 1:
        raise ValueError("--ci-level must be between 0 and 1.")

    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_samples, dtype=np.float64)
    for index in range(n_samples):
        sample_indices = rng.integers(0, array.shape[0], size=array.shape[0])
        estimates[index] = array[sample_indices].mean()

    alpha = 1.0 - ci_level
    low, high = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def _read_user_ids(data_file: str | Path) -> list[str]:
    user_ids: list[str] = []
    with Path(data_file).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            user_ids.append(stripped.split(" ", 1)[0])
    return user_ids


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


def _single_target_metrics(
    target: int,
    predictions: np.ndarray,
    topk_values: list[int],
    item_popularity: ItemPopularity,
    long_tail_items: set[int],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    prediction_list = predictions.tolist()
    for k in topk_values:
        topk = prediction_list[:k]
        if target in topk:
            rank = topk.index(target)
            metrics[f"recall@{k}"] = 1.0
            metrics[f"ndcg@{k}"] = float(1.0 / np.log2(rank + 2))
        else:
            metrics[f"recall@{k}"] = 0.0
            metrics[f"ndcg@{k}"] = 0.0
        metrics[f"arp@{k}"] = recommendation_popularity(topk, item_popularity)
        metrics[f"aplt@{k}"] = percentage_long_tail(topk, long_tail_items)
    return metrics


def _build_model(args: SimpleNamespace, device: torch.device) -> SASRecModel:
    model = SASRecModel(args).to(device)
    state_dict = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _collect_seed_rows(
    model: SASRecModel,
    dataloader: DataLoader,
    args: SimpleNamespace,
    user_ids: list[str],
    eval_seed: int,
    device: torch.device,
    item_popularity: ItemPopularity,
    long_tail_items: set[int],
    user_groups: dict[int, str],
) -> list[dict[str, Any]]:
    set_seed(eval_seed)
    model.eval()
    rows: list[dict[str, Any]] = []
    user_offset = 0
    topk_values = sorted(int(value) for value in args.topk)
    topk_max = min(max(topk_values), args.item_size)

    progress = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"SASRec eval seed {eval_seed}",
        bar_format="{l_bar}{r_bar}",
    )
    with torch.no_grad():
        for batch in progress:
            batch = tuple(t.to(device) for t in batch)
            batch_user_ids, input_ids, _, _, answers = batch
            sequence_output = model(input_ids)
            recommend_output = sequence_output[:, -1, :]
            rating_pred = torch.matmul(recommend_output, model.item_embeddings.weight.transpose(0, 1))
            rating_pred = rating_pred.cpu().numpy().copy()
            batch_user_index = batch_user_ids.cpu().numpy()
            _mask_invalid_and_seen_items(rating_pred, args, batch_user_index)

            ind = np.argpartition(rating_pred, -topk_max)[:, -topk_max:]
            arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
            arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
            pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
            targets = answers.cpu().numpy().reshape(-1)

            for batch_index, (target, predictions) in enumerate(zip(targets, pred_list)):
                user_index = user_offset + batch_index
                per_user = _single_target_metrics(
                    int(target), predictions, topk_values, item_popularity, long_tail_items
                )
                row = {
                    "user_index": user_index,
                    "user_raw_id": user_ids[user_index],
                    "eval_seed": eval_seed,
                    "label_item_id": int(target),
                    "n_visited_items": float(args.mask_id - 1),
                    "pop_group": user_groups[user_index],
                }
                row.update(per_user)
                rows.append(row)

            user_offset += len(targets)

    if user_offset != len(user_ids):
        raise RuntimeError(f"Collected {user_offset} test rows but expected {len(user_ids)} users.")
    return rows


def _per_seed_summary(
    rows: list[dict[str, Any]],
    metric_names: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["eval_seed"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for seed in sorted(grouped):
        seed_rows = grouped[seed]
        summary = {
            "eval_seed": seed,
            "n_users": len(seed_rows),
            "n_visited_items": _nan_if_empty([float(row["n_visited_items"]) for row in seed_rows]),
        }
        for metric in metric_names:
            summary[metric] = _nan_if_empty([float(row[metric]) for row in seed_rows])
        summary_rows.append(summary)
    return summary_rows


def _metric_summary(
    rows: list[dict[str, Any]],
    per_seed_rows: list[dict[str, Any]],
    metric_names: list[str],
    bootstrap_samples: int,
    bootstrap_seed: int,
    ci_level: float,
) -> list[dict[str, Any]]:
    rows_by_user_metric: dict[str, dict[int, list[float]]] = {
        metric: defaultdict(list) for metric in metric_names
    }
    for row in rows:
        user_index = int(row["user_index"])
        for metric in metric_names:
            rows_by_user_metric[metric][user_index].append(float(row[metric]))

    metric_rows: list[dict[str, Any]] = []
    for metric in metric_names:
        per_user_seed_means = [
            float(mean(values))
            for _, values in sorted(rows_by_user_metric[metric].items())
        ]
        per_seed_values = [float(row[metric]) for row in per_seed_rows]
        ci_low, ci_high = _bootstrap_ci(
            values=per_user_seed_means,
            n_samples=bootstrap_samples,
            seed=bootstrap_seed,
            ci_level=ci_level,
        )
        metric_rows.append(
            {
                "metric": metric,
                "n_users": len(per_user_seed_means),
                "n_eval_seeds": len(per_seed_values),
                "final_user_avg": _nan_if_empty(per_user_seed_means),
                "user_bootstrap_ci_level": ci_level,
                "user_bootstrap_ci_low": ci_low,
                "user_bootstrap_ci_high": ci_high,
                "eval_seed_mean": _nan_if_empty(per_seed_values),
                "eval_seed_std": _sample_std(per_seed_values),
                "eval_seed_min": min(per_seed_values) if per_seed_values else float("nan"),
                "eval_seed_max": max(per_seed_values) if per_seed_values else float("nan"),
            }
        )
    return metric_rows


def _eval_seeds_for_mode(parsed_args: argparse.Namespace, args: SimpleNamespace) -> list[int]:
    if parsed_args.eval_mode == "normal":
        return [int(args.seed if parsed_args.eval_seed is None else parsed_args.eval_seed)]
    return _parse_seeds(parsed_args.eval_seeds)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parsed_args, override_tokens = parser.parse_known_args(argv)
    args = _load_args(parsed_args, override_tokens)
    if not Path(args.data_file).is_file():
        raise FileNotFoundError(
            f"Missing SASRec data file: {args.data_file}. Run scripts/sasrec_prepare_data.py first."
        )
    if not Path(args.checkpoint_path).is_file():
        raise FileNotFoundError(f"SASRec checkpoint not found: {args.checkpoint_path}")

    user_seq, max_item, _, test_rating_matrix = get_user_seqs(args.data_file)
    user_ids = _read_user_ids(args.data_file)
    if len(user_ids) != len(user_seq):
        raise RuntimeError(f"Read {len(user_ids)} user ids but {len(user_seq)} sequences.")

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    args.train_matrix = test_rating_matrix

    item_popularity = compute_item_popularity(seq[:-2] for seq in user_seq)
    long_tail_items = compute_long_tail_items(item_popularity, short_head_fraction=parsed_args.short_head_fraction)
    profile_popularity = {
        user_index: compute_profile_popularity(seq[:-1], item_popularity)
        for user_index, seq in enumerate(user_seq)
    }
    user_groups = assign_popularity_groups(
        profile_popularity,
        low_quantile=parsed_args.popularity_low_quantile,
        high_quantile=parsed_args.popularity_high_quantile,
    )

    eval_seeds = _eval_seeds_for_mode(parsed_args, args)
    set_seed(eval_seeds[0])
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    test_dataset = SASRecDataset(args, user_seq, data_type="test")
    dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.eval_batch_size,
    )
    model = _build_model(args, device)
    metric_names = _all_metric_names(args)

    all_rows: list[dict[str, Any]] = []
    for eval_seed in eval_seeds:
        all_rows.extend(
            _collect_seed_rows(
                model=model,
                dataloader=dataloader,
                args=args,
                user_ids=user_ids,
                eval_seed=eval_seed,
                device=device,
                item_popularity=item_popularity,
                long_tail_items=long_tail_items,
                user_groups=user_groups,
            )
        )

    per_seed_rows = _per_seed_summary(all_rows, metric_names)
    metric_rows = _metric_summary(
        rows=all_rows,
        per_seed_rows=per_seed_rows,
        metric_names=metric_names,
        bootstrap_samples=parsed_args.bootstrap_samples,
        bootstrap_seed=parsed_args.bootstrap_seed,
        ci_level=parsed_args.ci_level,
    )
    group_rows = group_metric_summary(
        rows=all_rows,
        user_groups=user_groups,
        metric_names=metric_names,
        bootstrap_ci=lambda values: _bootstrap_ci(
            values=values,
            n_samples=parsed_args.bootstrap_samples,
            seed=parsed_args.bootstrap_seed,
            ci_level=parsed_args.ci_level,
        ),
        ci_level=parsed_args.ci_level,
    )

    session_root = _session_root(parsed_args.output_dir)
    per_user_csv = session_root / "per_user_metrics.csv"
    per_user_jsonl = session_root / "per_user_metrics.jsonl"
    per_seed_csv = session_root / "per_seed_summary.csv"
    summary_csv = session_root / "summary.csv"
    group_summary_csv = session_root / "group_summary.csv"
    summary_json = session_root / "summary.json"
    manifest_path = session_root / "manifest.json"

    _write_csv(per_user_csv, all_rows)
    _write_jsonl(per_user_jsonl, all_rows)
    _write_csv(per_seed_csv, per_seed_rows)
    _write_csv(summary_csv, metric_rows)
    _write_csv(group_summary_csv, group_rows)

    summary_payload = {
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "dataset": args.data_name,
        "category": getattr(args, "category", args.data_name),
        "model": "SASRec",
        "eval_mode": parsed_args.eval_mode,
        "eval_seeds": eval_seeds,
        "metrics": metric_names,
        "aggregation": "mean_per_user_over_eval_seeds_then_mean_users",
        "bootstrap_samples": parsed_args.bootstrap_samples,
        "bootstrap_seed": parsed_args.bootstrap_seed,
        "ci_level": parsed_args.ci_level,
        "popularity": {
            "short_head_fraction": parsed_args.short_head_fraction,
            "n_items_with_train_interactions": len(item_popularity),
            "n_long_tail_items": len(long_tail_items),
            "popularity_low_quantile": parsed_args.popularity_low_quantile,
            "popularity_high_quantile": parsed_args.popularity_high_quantile,
            "group_sizes": {
                group: sum(1 for value in user_groups.values() if value == group)
                for group in sorted(set(user_groups.values()))
            },
        },
        "per_seed_summary": per_seed_rows,
        "metric_summary": metric_rows,
        "group_metric_summary": group_rows,
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2))

    manifest = {
        "session_root": str(session_root),
        "per_user_csv": str(per_user_csv),
        "per_user_jsonl": str(per_user_jsonl),
        "per_seed_csv": str(per_seed_csv),
        "summary_csv": str(summary_csv),
        "group_summary_csv": str(group_summary_csv),
        "summary_json": str(summary_json),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(session_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
