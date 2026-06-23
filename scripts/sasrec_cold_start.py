#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.sasrec import SASRecDataset, SASRecModel  # noqa: E402
from models.sasrec.utils import get_user_seqs as get_user_seqs_sasrec, set_seed as set_seed_sasrec  # noqa: E402
from models.sasrec_modernized import SASRecModernizedDataset, SASRecModernizedModel  # noqa: E402
from models.sasrec_modernized.utils import get_user_seqs as get_user_seqs_modernized, set_seed as set_seed_modernized  # noqa: E402
from sasrec import (  # noqa: E402
    PRESET_CONFIGS as SASREC_PRESET_CONFIGS,
    build_config_files as build_sasrec_config_files,
    load_config as load_sasrec_config,
    normalize_config as normalize_sasrec_config,
    parse_override_args as parse_sasrec_override_args,
)
from sasrec_modernized import (  # noqa: E402
    PRESET_CONFIGS as MODERNIZED_PRESET_CONFIGS,
    build_config_files as build_modernized_config_files,
    load_config as load_modernized_config,
    normalize_config as normalize_modernized_config,
    parse_override_args as parse_modernized_override_args,
)


DEFAULT_BUCKETS = "0-5,6-10,11-15,16-20"
METRIC_NAMES = ("recall@5", "ndcg@5", "recall@10", "ndcg@10")
MODEL_FAMILY_CHOICES = ("auto", "sasrec", "sasrec_modernized")
PRESET_CHOICES = sorted(set(SASREC_PRESET_CONFIGS) | set(MODERNIZED_PRESET_CONFIGS))


@dataclass(frozen=True)
class ColdStartBucket:
    label: str
    min_count: int
    max_count: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a SASRec checkpoint by cold-start frequency buckets and "
            "render a figure from the grouped results."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_inputs(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--checkpoint", required=True, help="Path to a trained SASRec checkpoint.")
        subparser.add_argument("--preset", choices=PRESET_CHOICES, help="Named SASRec preset to apply.")
        subparser.add_argument("--dataset", default=None, help="Dataset/category override.")
        subparser.add_argument(
            "--model-family",
            choices=MODEL_FAMILY_CHOICES,
            default="auto",
            help="Which SASRec implementation to evaluate. Defaults to auto-detect from the checkpoint path.",
        )
        subparser.add_argument("--config", action="append", default=[], help="Additional YAML config file.")
        subparser.add_argument("--no-root-config", action="store_true", help="Skip the default SASRec root config.")
        subparser.add_argument("--no-local-config", action="store_true", help="Skip the local SASRec config even if it exists.")

    run_parser = subparsers.add_parser("run", help="Run cold-start evaluation and render a figure in one session folder.")
    add_common_inputs(run_parser)
    run_parser.add_argument("--output-dir", default=None, help="Optional session-root override. Defaults to artifacts/sasrec/cold_start.")
    run_parser.add_argument("--buckets", default=DEFAULT_BUCKETS, help=f"Comma-separated inclusive train-frequency ranges, for example '{DEFAULT_BUCKETS}'.")
    run_parser.add_argument("--plot-metric", default="ndcg@10", help="Metric to visualize in the generated figure.")
    run_parser.add_argument("--plot-title", default="SASRec Cold-Start Analysis", help="Title used in the generated plot.")

    plot_parser = subparsers.add_parser("plot", help="Render a figure from a previously generated cold-start summary JSON.")
    plot_parser.add_argument("--input", required=True, help="Path to cold_start_summary.json or a session directory containing it.")
    plot_parser.add_argument("--output", required=True, help="Destination image path.")
    plot_parser.add_argument("--metric", default="ndcg@10", help="Metric to visualize from the summary JSON.")
    plot_parser.add_argument("--title", default="SASRec Cold-Start Analysis", help="Title used in the generated plot.")
    return parser


def _session_root(output_root: str | None = None) -> Path:
    raw_root = output_root or "artifacts/sasrec/cold_start"
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

    (session_root / "tables").mkdir(parents=True, exist_ok=True)
    (session_root / "figures").mkdir(parents=True, exist_ok=True)
    return session_root


def _parse_bucket_spec(raw_spec: str) -> list[ColdStartBucket]:
    buckets: list[ColdStartBucket] = []
    for index, raw_bucket in enumerate(part.strip() for part in raw_spec.split(",") if part.strip()):
        if "-" not in raw_bucket:
            raise ValueError(f"Invalid bucket '{raw_bucket}'. Use inclusive ranges like '0-5,6-10'.")
        start_raw, end_raw = raw_bucket.split("-", 1)
        start = int(start_raw)
        end = int(end_raw)
        if start < 0 or end < start:
            raise ValueError(f"Invalid bucket bounds '{raw_bucket}'.")
        buckets.append(ColdStartBucket(label=str(index), min_count=start, max_count=end))
    if not buckets:
        raise ValueError("At least one cold-start bucket must be provided.")
    return buckets


def _bucket_label_for_count(count: int, buckets: list[ColdStartBucket]) -> str | None:
    for bucket in buckets:
        if bucket.min_count <= count <= bucket.max_count:
            return bucket.label
    return None


def _build_item_frequency(user_seq: list[list[int]]) -> dict[int, int]:
    frequencies: defaultdict[int, int] = defaultdict(int)
    for items in user_seq:
        for item_id in items[:-2]:
            frequencies[item_id] += 1
    return dict(frequencies)


def _build_item_group_mapping(
    user_seq: list[list[int]],
    real_item_count: int,
    buckets: list[ColdStartBucket],
) -> dict[int, str]:
    frequencies = _build_item_frequency(user_seq)
    item2group: dict[int, str] = {}
    for item_id in range(1, real_item_count + 1):
        label = _bucket_label_for_count(frequencies.get(item_id, 0), buckets)
        if label is not None:
            item2group[item_id] = label
    return item2group


def _mask_invalid_and_seen_items(
    rating_pred: np.ndarray,
    args: SimpleNamespace,
    batch_user_index: np.ndarray,
) -> None:
    rating_pred[:, 0] = -np.inf
    seen = args.train_matrix[batch_user_index].toarray() > 0
    rating_pred[:, : seen.shape[1]][seen] = -np.inf


def _single_target_metrics(target: int, predictions: np.ndarray) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in (5, 10):
        topk = predictions[:k].tolist()
        if target in topk:
            rank = topk.index(target)
            metrics[f"recall@{k}"] = 1.0
            metrics[f"ndcg@{k}"] = float(1.0 / np.log2(rank + 2))
        else:
            metrics[f"recall@{k}"] = 0.0
            metrics[f"ndcg@{k}"] = 0.0
    return metrics


def _aggregate_group_results(
    group2results: dict[str, dict[str, list[float]]],
    buckets: list[ColdStartBucket],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    rows: list[dict[str, Any]] = []
    grouped_means: dict[str, dict[str, float]] = {}

    for bucket in buckets:
        metric_lists = group2results.get(bucket.label, {})
        counts = [len(values) for values in metric_lists.values() if values]
        n_examples = counts[0] if counts else 0

        metric_means: dict[str, float] = {}
        for metric_name, values in metric_lists.items():
            if values:
                metric_means[metric_name] = float(mean(values))

        grouped_means[bucket.label] = metric_means
        row = {
            "bucket_label": bucket.label,
            "bucket_range": f"[{bucket.min_count}, {bucket.max_count}]",
            "bucket_min_count": bucket.min_count,
            "bucket_max_count": bucket.max_count,
            "n_examples": n_examples,
        }
        row.update(metric_means)
        rows.append(row)

    return rows, grouped_means


def _write_group_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_summary_path(raw_input: str | Path) -> Path:
    path = Path(raw_input).expanduser().resolve()
    if path.is_dir():
        path = path / "tables" / "cold_start_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Cold-start summary JSON not found: {path}")
    return path


def _plot_summary(summary_path: str | Path, output_path: str | Path, metric: str, title: str) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting.") from exc

    summary_file = _resolve_summary_path(summary_path)
    payload = json.loads(summary_file.read_text())
    rows = payload["group_rows"]
    bucket_labels = [row["bucket_range"] for row in rows]
    values = [float(row.get(metric, float("nan"))) for row in rows]
    counts = [int(row.get("n_examples", 0)) for row in rows]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(bucket_labels, values, color="#355C7D")
    ax.set_title(title)
    ax.set_xlabel("Training frequency bucket")
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    for index, (value, count) in enumerate(zip(values, counts)):
        if value == value:
            ax.text(index, value, f"{value:.4f}\n(n={count})", ha="center", va="bottom")
        else:
            ax.text(index, 0, f"nan\n(n={count})", ha="center", va="bottom")

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def _resolve_model_family(parsed_args: argparse.Namespace) -> str:
    if parsed_args.model_family != "auto":
        return parsed_args.model_family
    checkpoint_name = Path(parsed_args.checkpoint).name.lower()
    if "modernized" in checkpoint_name:
        return "sasrec_modernized"
    return "sasrec"


def _load_args(parsed_args: argparse.Namespace, override_tokens: list[str]) -> tuple[SimpleNamespace, str]:
    model_family = _resolve_model_family(parsed_args)
    if model_family == "sasrec_modernized":
        overrides = parse_modernized_override_args(override_tokens)
        build_config_files = build_modernized_config_files
        load_config = load_modernized_config
        normalize_config = normalize_modernized_config
    else:
        overrides = parse_sasrec_override_args(override_tokens)
        build_config_files = build_sasrec_config_files
        load_config = load_sasrec_config
        normalize_config = normalize_sasrec_config

    if parsed_args.dataset is not None:
        overrides["dataset"] = parsed_args.dataset

    config_files = build_config_files(parsed_args)
    merged_config = load_config(config_files, overrides)
    return normalize_config(merged_config, parsed_args.checkpoint), model_family


def _evaluate_cold_start(
    args: SimpleNamespace,
    model_family: str,
    user_seq: list[list[int]],
    item2group: dict[int, str],
    buckets: list[ColdStartBucket],
):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    if model_family == "sasrec_modernized":
        dataset_cls = SASRecModernizedDataset
        model_cls = SASRecModernizedModel
        n_visited_items = float(args.mask_id - 1)
    else:
        dataset_cls = SASRecDataset
        model_cls = SASRecModel
        n_visited_items = float(args.item_size - 2)

    test_dataset = dataset_cls(args, user_seq, data_type="test")
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.eval_batch_size,
    )
    model = model_cls(args).to(device)
    state_dict = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    all_results: dict[str, list[float]] = {metric: [] for metric in METRIC_NAMES}
    all_results["n_visited_items"] = []
    group2results: dict[str, dict[str, list[float]]] = {
        bucket.label: defaultdict(list) for bucket in buckets
    }
    topk_max = max(args.topk)

    progress = tqdm(
        enumerate(test_dataloader),
        total=len(test_dataloader),
        desc="SASRec cold-start eval",
        bar_format="{l_bar}{r_bar}",
    )
    with torch.no_grad():
        for _, batch in progress:
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
            batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
            batch_answers = answers.cpu().numpy().reshape(-1)

            for target, predictions in zip(batch_answers, batch_pred_list):
                per_user = _single_target_metrics(int(target), predictions)
                for metric_name, value in per_user.items():
                    all_results[metric_name].append(value)
                all_results["n_visited_items"].append(n_visited_items)

                group = item2group.get(int(target))
                if group is None:
                    continue
                for metric_name, value in per_user.items():
                    group2results[group][metric_name].append(value)
                group2results[group]["n_visited_items"].append(n_visited_items)

    overall_results = {
        metric_name: float(mean(values)) if values else float("nan")
        for metric_name, values in all_results.items()
    }
    return overall_results, group2results


def _run_cold_start(parsed_args: argparse.Namespace, override_tokens: list[str]) -> int:
    args, model_family = _load_args(parsed_args, override_tokens)
    if not Path(args.data_file).is_file():
        raise FileNotFoundError(f"Missing SASRec data file: {args.data_file}. Run scripts/sasrec_prepare_data.py first.")
    if not Path(args.checkpoint_path).is_file():
        raise FileNotFoundError(f"SASRec checkpoint not found: {args.checkpoint_path}")

    if model_family == "sasrec_modernized":
        set_seed_modernized(args.seed)
        user_seq, max_item, _, test_rating_matrix = get_user_seqs_modernized(args.data_file)
        args.item_size = max_item + 2
        args.mask_id = max_item + 1
    else:
        set_seed_sasrec(args.seed)
        user_seq, max_item, _, test_rating_matrix = get_user_seqs_sasrec(args.data_file)
        args.item_size = max_item + 1
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    args.train_matrix = test_rating_matrix

    session_root = _session_root(parsed_args.output_dir)
    buckets = _parse_bucket_spec(parsed_args.buckets)
    item2group = _build_item_group_mapping(user_seq, max_item, buckets)
    overall_results, group2results = _evaluate_cold_start(args, model_family, user_seq, item2group, buckets)
    group_rows, grouped_means = _aggregate_group_results(group2results, buckets)

    summary_payload = {
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "dataset": args.data_name,
        "category": getattr(args, "category", args.data_name),
        "model": "SASRecModernized" if model_family == "sasrec_modernized" else "SASRec",
        "model_family": model_family,
        "plot_metric": parsed_args.plot_metric,
        "bucket_spec": parsed_args.buckets,
        "overall_results": overall_results,
        "group_rows": group_rows,
        "group_metric_means": grouped_means,
    }

    summary_json = session_root / "tables" / "cold_start_summary.json"
    summary_csv = session_root / "tables" / "cold_start_summary.csv"
    figure_path = session_root / "figures" / f"{parsed_args.plot_metric.replace('@', '_at_')}.png"
    manifest_path = session_root / "manifest.json"

    summary_json.write_text(json.dumps(summary_payload, indent=2))
    _write_group_csv(summary_csv, group_rows)
    plotted_figure = _plot_summary(
        summary_path=summary_json,
        output_path=figure_path,
        metric=parsed_args.plot_metric,
        title=parsed_args.plot_title,
    )

    manifest = {
        "session_root": str(session_root),
        "summary_json": str(summary_json),
        "summary_csv": str(summary_csv),
        "figure_path": str(plotted_figure),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(manifest["session_root"])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, override_tokens = parser.parse_known_args(argv)

    if args.command == "plot":
        output = _plot_summary(
            summary_path=args.input,
            output_path=args.output,
            metric=args.metric,
            title=args.title,
        )
        print(output)
        return 0

    if args.command == "run":
        return _run_cold_start(args, override_tokens)

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
