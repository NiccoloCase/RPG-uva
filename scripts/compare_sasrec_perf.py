#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine two SASRec perf sessions and compute per-pool comparison deltas.",
    )
    parser.add_argument("--baseline-session", required=True, help="Baseline session directory.")
    parser.add_argument("--candidate-session", required=True, help="Candidate session directory.")
    parser.add_argument("--baseline-label", default="baseline", help="Baseline label.")
    parser.add_argument("--candidate-label", default="candidate", help="Candidate label.")
    parser.add_argument("--output-dir", required=True, help="Directory for merged comparison files.")
    return parser.parse_args()


def _summary_csv_path(session: str) -> Path:
    path = Path(session).expanduser().resolve()
    if path.is_dir():
        path = path / "summaries" / "profile_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {path}")
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _to_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _labeled_rows(rows: list[dict[str, str]], label: str, session_dir: str) -> list[dict[str, object]]:
    labeled: list[dict[str, object]] = []
    for row in rows:
        payload: dict[str, object] = dict(row)
        payload["model_label"] = label
        payload["session_dir"] = session_dir
        labeled.append(payload)
    return labeled


def _build_delta_rows(
    baseline_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    baseline_label: str,
    candidate_label: str,
) -> list[dict[str, object]]:
    baseline_by_pool = {int(row["pool_size"]): row for row in baseline_rows}
    candidate_by_pool = {int(row["pool_size"]): row for row in candidate_rows}
    pool_sizes = sorted(set(baseline_by_pool) & set(candidate_by_pool))
    deltas: list[dict[str, object]] = []

    for pool_size in pool_sizes:
        baseline = baseline_by_pool[pool_size]
        candidate = candidate_by_pool[pool_size]
        deltas.append(
            {
                "pool_size": pool_size,
                "baseline_label": baseline_label,
                "candidate_label": candidate_label,
                "baseline_epoch_time_s_median": _to_float(baseline, "epoch_time_s_median"),
                "candidate_epoch_time_s_median": _to_float(candidate, "epoch_time_s_median"),
                "epoch_time_delta_s": _to_float(candidate, "epoch_time_s_median")
                - _to_float(baseline, "epoch_time_s_median"),
                "baseline_peak_cuda_runtime_delta_allocated_gb_median": _to_float(
                    baseline, "peak_cuda_runtime_delta_allocated_gb_median"
                ),
                "candidate_peak_cuda_runtime_delta_allocated_gb_median": _to_float(
                    candidate, "peak_cuda_runtime_delta_allocated_gb_median"
                ),
                "peak_cuda_runtime_delta_allocated_gb_delta": _to_float(
                    candidate, "peak_cuda_runtime_delta_allocated_gb_median"
                )
                - _to_float(baseline, "peak_cuda_runtime_delta_allocated_gb_median"),
                "baseline_ndcg_at_10_median": _to_float(baseline, "ndcg_at_10_median"),
                "candidate_ndcg_at_10_median": _to_float(candidate, "ndcg_at_10_median"),
                "ndcg_at_10_delta": _to_float(candidate, "ndcg_at_10_median")
                - _to_float(baseline, "ndcg_at_10_median"),
            }
        )
    return deltas


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_csv = _summary_csv_path(args.baseline_session)
    candidate_csv = _summary_csv_path(args.candidate_session)
    baseline_rows = _read_csv(baseline_csv)
    candidate_rows = _read_csv(candidate_csv)

    combined_rows = _labeled_rows(
        baseline_rows, args.baseline_label, str(Path(args.baseline_session).expanduser().resolve())
    )
    combined_rows.extend(
        _labeled_rows(candidate_rows, args.candidate_label, str(Path(args.candidate_session).expanduser().resolve()))
    )
    delta_rows = _build_delta_rows(baseline_rows, candidate_rows, args.baseline_label, args.candidate_label)

    combined_csv = output_dir / "combined_profile_summary.csv"
    delta_csv = output_dir / "comparison_deltas.csv"
    manifest_json = output_dir / "manifest.json"

    _write_csv(combined_csv, combined_rows)
    _write_csv(delta_csv, delta_rows)
    manifest = {
        "baseline_session": str(Path(args.baseline_session).expanduser().resolve()),
        "candidate_session": str(Path(args.candidate_session).expanduser().resolve()),
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "combined_csv": str(combined_csv),
        "delta_csv": str(delta_csv),
    }
    manifest_json.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
