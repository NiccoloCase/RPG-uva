#!/usr/bin/env python3
"""Collect RPG grid-search results into CSVs.

Two independent grids, two CSVs:

  inference grid  (run_infer_grid.sh):
    output/reproduction/rpg/grid/infer/{ds}/b{B}_k{K}_q{Q}/{session}/summary.json
    -> long rows: dataset,num_beams,n_edges,propagation_steps,metric,mean,std,n_seeds

  training grid   (run_train_grid.sh):
    output/reproduction/rpg/grid/train/{ds}/*.err  (single-seed pipeline eval)
    -> long rows: dataset,lr,temperature,metric,value
    

stdlib only -- runs anywhere, no conda needed.

Usage:
  python scripts/collect_grid_results.py \
      [--infer-root output/reproduction/rpg/grid/infer] \
      [--train-root output/reproduction/rpg/grid/train] \
      [--infer-csv infer_grid.csv] [--train-csv train_grid.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CELL_RE = re.compile(r"^b(?P<b>\d+)_k(?P<k>\d+)_q(?P<q>\d+)$")
TEST_RESULTS_RE = re.compile(r"Test Results:")
KV_RE = re.compile(r"\('([^']+)',\s*([-\d.eE+]+)\)")
TRAIN_START_RE = re.compile(
    r"TRAIN_GRID_START\s+dataset=(?P<ds>\S+)\s+m=(?P<m>\S+)\s+"
    r"lr=(?P<lr>\S+)\s+temperature=(?P<temp>\S+)"
)


def _newest_summary(cell_dir: Path) -> Path | None:
    sessions = sorted(cell_dir.glob("*/summary.json"))
    return sessions[-1] if sessions else None


def collect_infer(infer_root: Path) -> list[dict]:
    rows: list[dict] = []
    if not infer_root.exists():
        return rows
    for ds_dir in sorted(p for p in infer_root.iterdir() if p.is_dir()):
        for cell_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
            match = CELL_RE.match(cell_dir.name)
            if not match:
                continue
            summary = _newest_summary(cell_dir)
            if summary is None:
                print(f"  skip {ds_dir.name}/{cell_dir.name}: no summary.json yet")
                continue
            payload = json.loads(summary.read_text())
            for metric_row in payload.get("metric_summary", []):
                rows.append(
                    {
                        "dataset": ds_dir.name,
                        "num_beams": int(match["b"]),
                        "n_edges": int(match["k"]),
                        "propagation_steps": int(match["q"]),
                        "metric": metric_row["metric"],
                        "mean": metric_row.get("eval_seed_mean"),
                        "std": metric_row.get("eval_seed_std"),
                        "n_seeds": metric_row.get("n_eval_seeds"),
                    }
                )
    rows.sort(key=lambda r: (r["dataset"], r["num_beams"], r["n_edges"],
                             r["propagation_steps"], r["metric"]))
    return rows


def collect_train(train_root: Path) -> list[dict]:
    rows: list[dict] = []
    if not train_root.exists():
        return rows
    for ds_dir in sorted(p for p in train_root.iterdir() if p.is_dir()):
        for err in sorted(ds_dir.glob("*.err")):
            text = err.read_text(errors="replace")
            start = TRAIN_START_RE.search(text)
            if not start:
                continue
            last_pairs: list[tuple[str, str]] = []
            for line in text.splitlines():
                if TEST_RESULTS_RE.search(line):
                    pairs = KV_RE.findall(line)
                    if pairs:
                        last_pairs = pairs  # keep the final (test) eval
            if not last_pairs:
                print(f"  skip {ds_dir.name}/{err.name}: no Test Results yet")
                continue
            for metric, value in last_pairs:
                rows.append(
                    {
                        "dataset": start["ds"],
                        "lr": float(start["lr"]),
                        "temperature": float(start["temp"]),
                        "metric": metric,
                        "value": float(value),
                    }
                )
    rows.sort(key=lambda r: (r["dataset"], r["lr"], r["temperature"], r["metric"]))
    return rows


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--infer-root",
                        default="output/reproduction/rpg/grid/infer")
    parser.add_argument("--train-root",
                        default="output/reproduction/rpg/grid/train")
    parser.add_argument("--infer-csv", default="infer_grid.csv")
    parser.add_argument("--train-csv", default="train_grid.csv")
    args = parser.parse_args()

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else REPO_ROOT / path

    infer_rows = collect_infer(_resolve(args.infer_root))
    train_rows = collect_train(_resolve(args.train_root))

    _write_csv(_resolve(args.infer_csv), infer_rows,
               ["dataset", "num_beams", "n_edges", "propagation_steps",
                "metric", "mean", "std", "n_seeds"])
    _write_csv(_resolve(args.train_csv), train_rows,
               ["dataset", "lr", "temperature", "metric", "value"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
