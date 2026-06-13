#!/usr/bin/env python3
"""Collect RPG grid-search results into CSVs.

Three grids, three CSVs:

  inference grid  (run_infer_grid.sh)
    output/reproduction/rpg/grid/infer/{ds}/b{B}_k{K}_q{Q}/{session}/summary.json
    -> long rows: dataset,num_beams,n_edges,propagation_steps,metric,mean,std,n_seeds

    output/reproduction/rpg/grid/fig6/{ds}/b{B}_k{K}_q{Q}/{session}/summary.json
    -> long rows: dataset,num_beams,n_edges,propagation_steps,metric,mean,std,n_seeds

  training grid   (run_train_grid.sh) -- seeded retrain over lr x temperature:
    output/reproduction/rpg/grid/train/*.err  (one retrain per lr/temp/seed cell)
    -> long rows aggregated over seeds: dataset,lr,temperature,metric,mean,std,n_seeds

The training grid is seeded: each (dataset,lr,temperature) cell is retrained over
several rand_seeds. Logs that predate the seed dimension (no "seed=" token) are
treated as the genrec-default seed 2024. For each (dataset,lr,temperature,seed) the
NEWEST .err wins (by mtime), then we report mean/std over the distinct seeds.

stdlib only -- runs anywhere, no conda needed.

Usage:
  python scripts/collect_grid_results.py \
      [--infer-root output/reproduction/rpg/grid/infer] \
      [--train-root output/reproduction/rpg/grid/train] \
      [--fig6-root  output/reproduction/rpg/grid/fig6] \
      [--infer-csv infer_grid.csv] [--train-csv train_grid.csv] [--fig6-csv fig6_grid.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CELL_RE = re.compile(r"^b(?P<b>\d+)_k(?P<k>\d+)_q(?P<q>\d+)$")
TEST_RESULTS_RE = re.compile(r"Test Results:")
KV_RE = re.compile(r"\('([^']+)',\s*([-\d.eE+]+)\)")
TRAIN_START_RE = re.compile(
    r"TRAIN_GRID_START\s+dataset=(?P<ds>\S+)\s+m=(?P<m>\S+)\s+"
    r"lr=(?P<lr>\S+)\s+temperature=(?P<temp>\S+)"
    r"(?:\s+seed=(?P<seed>\S+))?"
)
DEFAULT_SEED = 2024  # genrec/default.yaml rand_seed; pre-seed-dimension logs == this
RPG_VAL_RE = re.compile(r"Best val score:\s*([\-\d.eE+]+)")

SASREC_START_RE = re.compile(
    r"SASREC_GRID_START\s+dataset=(?P<ds>\S+)\s+lr=(?P<lr>\S+)\s+"
    r"dropout=(?P<drop>\S+)\s+blocks=(?P<blocks>\S+)\s+seed=(?P<seed>\S+)"
)
SASREC_VAL_RE = re.compile(r"Validation score improved to \[([\d.eE+-]+)\]")
SASREC_METRIC_RE = re.compile(r"\('(RECALL|NDCG)@(\d+)',\s*'([\d.eE+-]+)'\)")


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
    if not train_root.exists():
        return []

    # (ds, lr, temp, seed) -> {metric: value}, newest .err wins (by mtime).
    by_cell_seed: dict[tuple, dict[str, float]] = {}
    for err in sorted(train_root.rglob("*.err"), key=lambda p: p.stat().st_mtime):
        text = err.read_text(errors="replace")
        start = TRAIN_START_RE.search(text)
        if not start:
            out = err.with_suffix(".out")
            if out.exists():
                start = TRAIN_START_RE.search(out.read_text(errors="replace"))
        if not start:
            continue
        last_pairs: list[tuple[str, str]] = []
        for line in text.splitlines():
            if TEST_RESULTS_RE.search(line):
                pairs = KV_RE.findall(line)
                if pairs:
                    last_pairs = pairs
        if not last_pairs:
            print(f"  skip {err.name}: no Test Results yet")
            continue
        val_hit = RPG_VAL_RE.search(text)
        seed = int(start["seed"]) if start["seed"] is not None else DEFAULT_SEED
        cell = (start["ds"], float(start["lr"]), float(start["temp"]), seed)
        by_cell_seed[cell] = {
            "test": {m: float(v) for m, v in last_pairs},
            "val": float(val_hit.group(1)) if val_hit else None,
        }  # newest wins

    test_agg: dict[tuple, list[float]] = {}
    val_agg: dict[tuple, list[float]] = {}
    for (ds, lr, temp, _seed), payload in by_cell_seed.items():
        if payload["val"] is not None:
            val_agg.setdefault((ds, lr, temp), []).append(payload["val"])
        for metric, value in payload["test"].items():
            test_agg.setdefault((ds, lr, temp, metric), []).append(value)

    rows: list[dict] = []
    for (ds, lr, temp, metric), values in test_agg.items():
        vals = val_agg.get((ds, lr, temp), [])
        rows.append(
            {
                "dataset": ds,
                "lr": lr,
                "temperature": temp,
                "metric": metric,
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "n_seeds": len(values),
                "val_ndcg10_mean": statistics.fmean(vals) if vals else "",
                "val_ndcg10_std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            }
        )
    rows.sort(key=lambda r: (r["dataset"], r["lr"], r["temperature"], r["metric"]))
    return rows


def collect_sasrec_train(train_root: Path) -> list[dict]:
    if not train_root.exists():
        return []

    by_cell_seed: dict[tuple, dict] = {}
    for out in sorted(train_root.rglob("*.out"), key=lambda p: p.stat().st_mtime):
        text = out.read_text(errors="replace")
        start = SASREC_START_RE.search(text)
        if not start:
            continue
        val_hits = SASREC_VAL_RE.findall(text)
        run_id = (f"sasrec_grid_{start['ds']}_lr{start['lr']}_d{start['drop']}"
                  f"_b{start['blocks']}_s{start['seed']}")
        test_pairs = []
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == run_id and i + 1 < len(lines):
                test_pairs = SASREC_METRIC_RE.findall(lines[i + 1])
        if not test_pairs and not val_hits:
            print(f"  skip {out.name}: no results yet")
            continue
        cell = (start["ds"], float(start["lr"]), float(start["drop"]),
                int(start["blocks"]), int(start["seed"]))
        by_cell_seed[cell] = {
            "val": float(val_hits[-1]) if val_hits else None,
            "test": {f"{name.lower()}@{k}": float(v) for name, k, v in test_pairs},
        }

    test_agg: dict[tuple, list[float]] = {}
    val_agg: dict[tuple, list[float]] = {}
    for (ds, lr, drop, blocks, _seed), payload in by_cell_seed.items():
        if payload["val"] is not None:
            val_agg.setdefault((ds, lr, drop, blocks), []).append(payload["val"])
        for metric, value in payload["test"].items():
            test_agg.setdefault((ds, lr, drop, blocks, metric), []).append(value)

    def _mean(xs):
        return statistics.fmean(xs) if xs else ""

    def _std(xs):
        return statistics.stdev(xs) if len(xs) > 1 else 0.0

    rows: list[dict] = []
    for (ds, lr, drop, blocks, metric), values in test_agg.items():
        vals = val_agg.get((ds, lr, drop, blocks), [])
        rows.append(
            {
                "dataset": ds,
                "lr": lr,
                "dropout": drop,
                "n_blocks": blocks,
                "metric": metric,
                "test_mean": statistics.fmean(values),
                "test_std": _std(values),
                "n_seeds": len(values),
                "val_ndcg20_mean": _mean(vals),
                "val_ndcg20_std": _std(vals),
            }
        )
    rows.sort(key=lambda r: (r["dataset"], r["lr"], r["dropout"], r["n_blocks"], r["metric"]))
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
    parser.add_argument("--fig6-root",
                        default="output/reproduction/rpg/grid/fig6")
    parser.add_argument("--sasrec-train-root",
                        default="output/reproduction/sasrec/grid/train")
    parser.add_argument("--infer-csv", default="infer_grid.csv")
    parser.add_argument("--train-csv", default="train_grid.csv")
    parser.add_argument("--fig6-csv", default="fig6_grid.csv")
    parser.add_argument("--sasrec-train-csv", default="sasrec_train_grid.csv")
    args = parser.parse_args()

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else REPO_ROOT / path

    infer_rows = collect_infer(_resolve(args.infer_root))
    fig6_rows = collect_infer(_resolve(args.fig6_root))  # same summary.json layout
    train_rows = collect_train(_resolve(args.train_root))
    sasrec_rows = collect_sasrec_train(_resolve(args.sasrec_train_root))

    decode_fields = ["dataset", "num_beams", "n_edges", "propagation_steps",
                     "metric", "mean", "std", "n_seeds"]
    _write_csv(_resolve(args.infer_csv), infer_rows, decode_fields)
    _write_csv(_resolve(args.fig6_csv), fig6_rows, decode_fields)
    _write_csv(_resolve(args.train_csv), train_rows,
               ["dataset", "lr", "temperature", "metric", "mean", "std", "n_seeds",
                "val_ndcg10_mean", "val_ndcg10_std"])
    _write_csv(_resolve(args.sasrec_train_csv), sasrec_rows,
               ["dataset", "lr", "dropout", "n_blocks", "metric", "test_mean",
                "test_std", "n_seeds", "val_ndcg20_mean", "val_ndcg20_std"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
