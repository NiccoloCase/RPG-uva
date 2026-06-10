#!/usr/bin/env python3
"""Parse SLURM output logs from the m-sweep and print a results table.

Usage:
    python scripts/collect_sweep_m_results.py
    python scripts/collect_sweep_m_results.py --log-dir output/reproduction/train/sweep_m
    python scripts/collect_sweep_m_results.py --csv results_sweep_m.csv
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = REPO_ROOT / "output" / "reproduction" / "train" / "sweep_m"

SWEEP_START_RE = re.compile(
    r"SWEEP_START\s+dataset=(\S+)\s+n_codebook=(\d+)\s+run_id=(\S+)"
)
TEST_RESULTS_RE = re.compile(r"Test Results:\s*(\{.*\})")


def parse_log(path: Path) -> dict | None:
    dataset = n_codebook = run_id = None
    metrics: dict | None = None

    out = path.with_suffix(".out")
    text = path.read_text(errors="replace") + (out.read_text(errors="replace") if out.exists() else "")
    for line in text.splitlines():
        m = SWEEP_START_RE.search(line)
        if m:
            dataset, n_codebook, run_id = m.group(1), int(m.group(2)), m.group(3)

        m = TEST_RESULTS_RE.search(line)
        if m:
            try:
                metrics = ast.literal_eval(m.group(1))
            except (ValueError, SyntaxError):
                pass

    if dataset is None or metrics is None:
        return None

    return {
        "dataset": dataset,
        "n_codebook": n_codebook,
        "run_id": run_id,
        **metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        default=str(DEFAULT_LOG_DIR),
        help="Directory containing SLURM .out files (default: %(default)s)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="If given, also write results to this CSV path.",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"ERROR: log directory not found: {log_dir}", file=sys.stderr)
        return 1

    rows = []
    for path in sorted(log_dir.glob("*.err")):
        row = parse_log(path)
        if row:
            rows.append(row)
        else:
            print(f"  skipped (incomplete): {path.name}", file=sys.stderr)

    if not rows:
        print("No completed runs found.", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: (r["dataset"], r["n_codebook"]))

    metric_keys = ["recall@5", "recall@10", "ndcg@5", "ndcg@10"]
    present_keys = [k for k in metric_keys if any(k in r for r in rows)]

    header = ["dataset", "n_codebook"] + present_keys
    col_w = max(len(h) for h in header) + 2

    print("  ".join(h.ljust(col_w) for h in header))
    print("  ".join("-" * col_w for _ in header))
    for row in rows:
        vals = [row.get("dataset", ""), str(row.get("n_codebook", ""))]
        for k in present_keys:
            v = row.get(k)
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v) if v is not None else "—")
        print("  ".join(v.ljust(col_w) for v in vals))

    if args.csv:
        csv_path = Path(args.csv)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
