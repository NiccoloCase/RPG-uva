#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perf.config import build_repo_config_files, ensure_submodule_available, parse_override_args
from perf.harness import EvaluationHarness


DEFAULT_BUCKETS = "0-5,6-10,11-15,16-20"


@dataclass(frozen=True)
class ColdStartBucket:
    """Inclusive train-frequency bucket used for cold-start evaluation."""

    label: str
    min_count: int
    max_count: int


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for cold-start evaluation and plotting."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an RPG checkpoint by cold-start frequency buckets and "
            "render a figure from the grouped results."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_inputs(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--checkpoint",
            required=True,
            help="Path to a trained RPG checkpoint.",
        )
        subparser.add_argument(
            "--config",
            action="append",
            default=[],
            help="Additional YAML config file. May be provided multiple times.",
        )
        subparser.add_argument(
            "--no-root-config",
            action="store_true",
            help="Skip configs/rpg/root.yaml.",
        )
        subparser.add_argument(
            "--no-local-config",
            action="store_true",
            help="Skip configs/rpg/local.yaml even if it exists.",
        )

    run_parser = subparsers.add_parser(
        "run",
        help="Run cold-start evaluation and render a figure in one session folder.",
    )
    add_common_inputs(run_parser)
    run_parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional session-root override. Defaults to artifacts/rpg/cold_start "
            "under the repository root."
        ),
    )
    run_parser.add_argument(
        "--buckets",
        default=DEFAULT_BUCKETS,
        help=(
            "Comma-separated inclusive train-frequency ranges, for example "
            f"'{DEFAULT_BUCKETS}'."
        ),
    )
    run_parser.add_argument(
        "--plot-metric",
        default="ndcg@10",
        help="Metric to visualize in the rendered figure.",
    )
    run_parser.add_argument(
        "--plot-title",
        default="RPG Cold-Start Analysis",
        help="Title used in the generated plot.",
    )

    plot_parser = subparsers.add_parser(
        "plot",
        help="Render a figure from a previously generated cold-start summary JSON.",
    )
    plot_parser.add_argument(
        "--input",
        required=True,
        help="Path to cold_start_summary.json or a session directory containing it.",
    )
    plot_parser.add_argument(
        "--output",
        required=True,
        help="Destination image path, for example artifacts/.../cold_start_ndcg10.png.",
    )
    plot_parser.add_argument(
        "--metric",
        default="ndcg@10",
        help="Metric to visualize from the summary JSON.",
    )
    plot_parser.add_argument(
        "--title",
        default="RPG Cold-Start Analysis",
        help="Title used in the generated plot.",
    )

    return parser


def _session_root(output_root: str | None = None) -> Path:
    """Create a unique artifact directory for one cold-start run."""
    raw_root = output_root or "artifacts/rpg/cold_start"
    path = Path(raw_root)
    if not path.is_absolute():
        path = REPO_ROOT / path

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
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
    """Parse the CLI bucket specification into ordered inclusive ranges."""
    buckets: list[ColdStartBucket] = []
    for index, raw_bucket in enumerate(part.strip() for part in raw_spec.split(",") if part.strip()):
        if "-" not in raw_bucket:
            raise ValueError(
                f"Invalid bucket '{raw_bucket}'. Use inclusive ranges like '0-5,6-10'."
            )
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
    """Map one train-frequency count to the configured bucket label."""
    for bucket in buckets:
        if bucket.min_count <= count <= bucket.max_count:
            return bucket.label
    return None


def _build_item_frequency(dataset: Any) -> Counter[str]:
    """Count how many times each raw item appears in the training split."""
    train_split = dataset.split()["train"]
    frequencies: Counter[str] = Counter()
    for item_seq in train_split["item_seq"]:
        for item in item_seq:
            frequencies[item] += 1
    return frequencies


def _build_item_group_mapping(dataset: Any, buckets: list[ColdStartBucket]) -> dict[int, str]:
    """Assign each relevant target item ID to its cold-start bucket label."""
    frequencies = _build_item_frequency(dataset)
    item2group: dict[int, str] = {}

    for raw_item, item_id in dataset.item2id.items():
        if item_id == 0:
            continue
        label = _bucket_label_for_count(frequencies.get(raw_item, 0), buckets)
        if label is not None:
            item2group[item_id] = label

    return item2group


def _build_token2item_identity(dataset: Any) -> dict[str, int]:
    """Build the label-to-item mapping expected by `evaluate_cold_start`.

    For RPG test batches the label is already the next-item ID, so the mapping
    is effectively identity on the string form of the item ID.
    """
    return {str(item_id): item_id for item_id in range(1, dataset.n_items)}


def _aggregate_group_results(
    group2results: dict[str, dict[str, list[float]]],
    buckets: list[ColdStartBucket],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """Compute mean metrics and example counts per cold-start bucket."""
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
    """Write per-bucket cold-start metrics to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_summary_path(raw_input: str | Path) -> Path:
    """Resolve a summary JSON path from either a file or a session directory."""
    path = Path(raw_input).expanduser().resolve()
    if path.is_dir():
        path = path / "tables" / "cold_start_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Cold-start summary JSON not found: {path}")
    return path


def _plot_summary(summary_path: str | Path, output_path: str | Path, metric: str, title: str) -> Path:
    """Render a single-metric bar chart from a cold-start summary JSON file."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Update the project environment "
            "before running the cold-start plot command."
        ) from exc

    summary_file = _resolve_summary_path(summary_path)
    payload = json.loads(summary_file.read_text())
    rows = payload["group_rows"]
    bucket_labels = [row["bucket_range"] for row in rows]
    values = [float(row.get(metric, float("nan"))) for row in rows]
    counts = [int(row.get("n_examples", 0)) for row in rows]

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax.bar(bucket_labels, values, color="#355C7D")
    ax.set_title(title)
    ax.set_xlabel("Training frequency bucket")
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    for index, (value, count) in enumerate(zip(values, counts)):
        if value == value:  # NaN check
            ax.text(index, value, f"{value:.4f}\n(n={count})", ha="center", va="bottom")
        else:
            ax.text(index, 0, f"nan\n(n={count})", ha="center", va="bottom")

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def _run_cold_start(args: argparse.Namespace, override_tokens: list[str]) -> int:
    """Execute cold-start evaluation, write tables, and render a figure."""
    ensure_submodule_available()
    config_files = build_repo_config_files(
        extra_configs=args.config,
        include_root_config=not args.no_root_config,
        include_local_config=not args.no_local_config,
    )
    config_overrides = parse_override_args(override_tokens)
    session_root = _session_root(args.output_dir)
    buckets = _parse_bucket_spec(args.buckets)

    harness = EvaluationHarness.build(
        checkpoint_path=args.checkpoint,
        config_files=config_files,
        config_overrides=config_overrides,
    )

    token2item = _build_token2item_identity(harness.dataset)
    item2group = _build_item_group_mapping(harness.dataset, buckets)
    overall_results, group2results = harness.trainer.evaluate_cold_start(
        harness.test_dataloader,
        token2item=token2item,
        item2group=item2group,
        split="test",
    )

    group_rows, grouped_means = _aggregate_group_results(group2results, buckets)

    summary_payload = {
        "checkpoint_path": str(harness.checkpoint_path),
        "dataset": harness.config["dataset"],
        "category": harness.config.get("category"),
        "model": harness.config["model"],
        "plot_metric": args.plot_metric,
        "bucket_spec": args.buckets,
        "overall_results": overall_results,
        "group_rows": group_rows,
        "group_metric_means": grouped_means,
    }

    summary_json = session_root / "tables" / "cold_start_summary.json"
    summary_csv = session_root / "tables" / "cold_start_summary.csv"
    figure_path = session_root / "figures" / f"{args.plot_metric.replace('@', '_at_')}.png"
    manifest_path = session_root / "manifest.json"

    summary_json.write_text(json.dumps(summary_payload, indent=2))
    _write_group_csv(summary_csv, group_rows)
    plotted_figure = _plot_summary(
        summary_path=summary_json,
        output_path=figure_path,
        metric=args.plot_metric,
        title=args.plot_title,
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
    """Entry point for the repo-owned cold-start reproduction script."""
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
