#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perf.config import build_repo_config_files, ensure_submodule_available, parse_override_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run eval-only RPG graph decoding across multiple eval seeds and "
            "write per-user metrics plus bootstrap summaries."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a trained RPG checkpoint.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Additional YAML config file. May be provided multiple times.",
    )
    parser.add_argument(
        "--no-root-config",
        action="store_true",
        help="Skip configs/rpg/root.yaml.",
    )
    parser.add_argument(
        "--no-local-config",
        action="store_true",
        help="Skip configs/rpg/local.yaml even if it exists.",
    )
    parser.add_argument(
        "--eval-seeds",
        required=True,
        help="Comma-separated eval seeds, for example '2024,2025,2026'.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/rpg/eval_seeds",
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
        "--paper-metric",
        default=None,
        help="Optional metric name to compare against the paper, e.g. 'ndcg@10'.",
    )
    parser.add_argument(
        "--paper-value",
        type=float,
        default=None,
        help="Optional paper metric value used for equivalence testing.",
    )
    parser.add_argument(
        "--equivalence-margin",
        type=float,
        default=None,
        help="Optional absolute equivalence margin around the paper value.",
    )
    return parser


def _parse_seeds(raw_value: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("At least one eval seed must be provided.")
    return seeds


def _metric_names(config: dict[str, Any]) -> list[str]:
    return [
        f"{metric}@{k}"
        for metric in config["metrics"]
        for k in config["topk"]
    ]


def _session_root(output_root: str | Path) -> Path:
    path = Path(output_root).expanduser()
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

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
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
    offset: float = 0.0,
) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if n_samples <= 0:
        estimate = float(mean(values)) - offset
        return estimate, estimate
    if not 0 < ci_level < 1:
        raise ValueError("--ci-level must be between 0 and 1.")

    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_samples, dtype=np.float64)
    for index in range(n_samples):
        sample_indices = rng.integers(0, array.shape[0], size=array.shape[0])
        estimates[index] = array[sample_indices].mean() - offset

    alpha = 1.0 - ci_level
    low, high = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def _build_graph_once(harness: EvaluationHarness) -> None:
    """Initialize the upstream decoding graph once and reuse it across eval seeds."""
    harness.model.generate_w_decoding_graph = True
    if not harness.model.init_flag:
        harness.model.init_graph()
        harness.model.init_flag = True


def _collect_seed_rows(
    harness: EvaluationHarness,
    eval_seed: int,
    user_ids: list[str],
    metric_names: list[str],
) -> list[dict[str, Any]]:
    from genrec.utils import init_seed

    init_seed(eval_seed, harness.config["reproducibility"])
    harness.model.generate_w_decoding_graph = True
    harness.model.init_flag = True
    harness.model.eval()

    rows: list[dict[str, Any]] = []
    user_offset = 0
    maxk = harness.trainer.evaluator.maxk

    progress = tqdm(
        harness.test_dataloader,
        total=len(harness.test_dataloader),
        desc=f"Eval seed {eval_seed}",
    )
    with torch.no_grad():
        for batch in progress:
            batch = {key: value.to(harness.accelerator.device) for key, value in batch.items()}
            preds = harness.model.generate(batch, n_return_sequences=maxk)
            results = harness.trainer.evaluator.calculate_metrics(preds, batch["labels"])

            batch_size = int(batch["labels"].shape[0])
            labels = batch["labels"].detach().cpu().view(batch_size, -1)[:, 0].tolist()
            metric_values = {
                metric: results[metric].detach().cpu().view(-1).tolist()
                for metric in metric_names
            }
            visited_values = results["n_visited_items"].detach().cpu().view(-1).tolist()

            for batch_index in range(batch_size):
                user_index = user_offset + batch_index
                row = {
                    "user_index": user_index,
                    "user_raw_id": user_ids[user_index],
                    "eval_seed": eval_seed,
                    "label_item_id": int(labels[batch_index]),
                    "n_visited_items": float(visited_values[batch_index]),
                }
                for metric in metric_names:
                    row[metric] = float(metric_values[metric][batch_index])
                rows.append(row)

            user_offset += batch_size

    if user_offset != len(user_ids):
        raise RuntimeError(
            f"Collected {user_offset} test rows but expected {len(user_ids)} users."
        )
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
    paper_metric: str | None,
    paper_value: float | None,
    equivalence_margin: float | None,
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

        summary = {
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

        if paper_metric == metric:
            assert paper_value is not None
            assert equivalence_margin is not None
            diff_low, diff_high = _bootstrap_ci(
                values=per_user_seed_means,
                n_samples=bootstrap_samples,
                seed=bootstrap_seed,
                ci_level=0.90,
                offset=paper_value,
            )
            observed_diff = summary["final_user_avg"] - paper_value
            summary.update(
                {
                    "paper_value": paper_value,
                    "paper_difference": observed_diff,
                    "equivalence_margin": equivalence_margin,
                    "equivalence_ci_level": 0.90,
                    "equivalence_diff_ci_low": diff_low,
                    "equivalence_diff_ci_high": diff_high,
                    "equivalent_to_paper": (
                        diff_low >= -equivalence_margin
                        and diff_high <= equivalence_margin
                    ),
                }
            )
        metric_rows.append(summary)

    return metric_rows


def _validate_paper_args(
    paper_metric: str | None,
    paper_value: float | None,
    equivalence_margin: float | None,
    metric_names: list[str],
) -> None:
    provided = [
        paper_metric is not None,
        paper_value is not None,
        equivalence_margin is not None,
    ]
    if any(provided) and not all(provided):
        raise ValueError(
            "--paper-metric, --paper-value, and --equivalence-margin must be provided together."
        )
    if paper_metric is not None and paper_metric not in metric_names:
        raise ValueError(
            f"Paper metric {paper_metric!r} is not in active metrics: {metric_names}."
        )
    if equivalence_margin is not None and equivalence_margin <= 0:
        raise ValueError("--equivalence-margin must be positive.")


def _resolve_paper_comparison(args: argparse.Namespace, config: dict[str, Any]) -> tuple[str | None, float | None, float | None]:
    paper_metric = args.paper_metric
    paper_value = args.paper_value
    equivalence_margin = args.equivalence_margin

    if paper_metric is None:
        paper_metric = config.get("paper_metric")
    if paper_value is None and config.get("paper_value") is not None:
        paper_value = float(config["paper_value"])
    if equivalence_margin is None and config.get("equivalence_margin") is not None:
        equivalence_margin = float(config["equivalence_margin"])

    return paper_metric, paper_value, equivalence_margin


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, override_tokens = parser.parse_known_args(argv)

    ensure_submodule_available()
    from perf.harness import EvaluationHarness

    eval_seeds = _parse_seeds(args.eval_seeds)
    config_files = build_repo_config_files(
        extra_configs=args.config,
        include_root_config=not args.no_root_config,
        include_local_config=not args.no_local_config,
    )
    config_overrides = parse_override_args(override_tokens)

    harness = EvaluationHarness.build(
        checkpoint_path=args.checkpoint,
        config_files=config_files,
        config_overrides=config_overrides,
    )
    if harness.config["use_ddp"]:
        raise RuntimeError("scripts/rpg_eval_seeds.py only supports single-process evaluation.")

    metric_names = _metric_names(harness.config)
    paper_metric, paper_value, equivalence_margin = _resolve_paper_comparison(args, harness.config)
    _validate_paper_args(
        paper_metric=paper_metric,
        paper_value=paper_value,
        equivalence_margin=equivalence_margin,
        metric_names=metric_names,
    )

    test_split = harness.dataset.split()["test"]
    user_ids = [str(user) for user in test_split["user"]]
    if len(user_ids) != len(harness.test_dataloader.dataset):
        raise RuntimeError(
            f"Test user count ({len(user_ids)}) does not match tokenized test rows "
            f"({len(harness.test_dataloader.dataset)})."
        )

    _build_graph_once(harness)

    all_rows: list[dict[str, Any]] = []
    for eval_seed in eval_seeds:
        all_rows.extend(
            _collect_seed_rows(
                harness=harness,
                eval_seed=eval_seed,
                user_ids=user_ids,
                metric_names=metric_names,
            )
        )

    per_seed_rows = _per_seed_summary(all_rows, metric_names)
    metric_rows = _metric_summary(
        rows=all_rows,
        per_seed_rows=per_seed_rows,
        metric_names=metric_names,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        ci_level=args.ci_level,
        paper_metric=paper_metric,
        paper_value=paper_value,
        equivalence_margin=equivalence_margin,
    )

    session_root = _session_root(args.output_dir)
    per_user_csv = session_root / "per_user_metrics.csv"
    per_user_jsonl = session_root / "per_user_metrics.jsonl"
    per_seed_csv = session_root / "per_seed_summary.csv"
    metric_summary_csv = session_root / "summary.csv"
    summary_json = session_root / "summary.json"
    manifest_path = session_root / "manifest.json"

    _write_csv(per_user_csv, all_rows)
    _write_jsonl(per_user_jsonl, all_rows)
    _write_csv(per_seed_csv, per_seed_rows)
    _write_csv(metric_summary_csv, metric_rows)

    summary_payload = {
        "checkpoint_path": str(harness.checkpoint_path),
        "dataset": harness.config["dataset"],
        "category": harness.config.get("category"),
        "model": harness.config["model"],
        "eval_seeds": eval_seeds,
        "metrics": metric_names,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "ci_level": args.ci_level,
        "paper_comparison": {
            "paper_metric": paper_metric,
            "paper_value": paper_value,
            "equivalence_margin": equivalence_margin,
        },
        "per_seed_summary": per_seed_rows,
        "metric_summary": metric_rows,
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2))

    manifest = {
        "session_root": str(session_root),
        "per_user_csv": str(per_user_csv),
        "per_user_jsonl": str(per_user_jsonl),
        "per_seed_csv": str(per_seed_csv),
        "summary_csv": str(metric_summary_csv),
        "summary_json": str(summary_json),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(session_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
