#!/usr/bin/env python3
"""Collect RPG grid-search results into CSVs.

Three grids, three CSVs:

  inference grid  (run_infer_grid.sh)
    output/reproduction/rpg/grid/infer/{ds}/b{B}_k{K}_q{Q}/{session}/summary.json
    -> long rows: dataset,num_beams,n_edges,propagation_steps,metric,mean,std,n_seeds

    output/reproduction/rpg/grid/fig6/{ds}/b{B}_k{K}_q{Q}/{session}/summary.json
    -> long rows: dataset,num_beams,n_edges,propagation_steps,metric,mean,std,n_seeds

    output/reproduction/rpg/grid/decode_test_confirm/{ds}/b{B}_k{K}_q{Q}/{session}/summary.json
    -> long rows (TEST @10 seeds, the val-selected cluster re-decoded):
       dataset,num_beams,n_edges,propagation_steps,metric,mean,std,n_seeds
       plus a per-dataset reporting-rule summary (decode_test_selected.csv):
       the val-argmax cell's TEST number (zero test-peeking) alongside the
       best-of-cluster TEST number (a secondary ceiling that does peek at test).

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
        cell = (start["ds"], int(start["m"]), float(start["lr"]), float(start["temp"]), seed)
        by_cell_seed[cell] = {
            "test": {mt: float(v) for mt, v in last_pairs},
            "val": float(val_hit.group(1)) if val_hit else None,
        }  # newest wins

    test_agg: dict[tuple, list[float]] = {}
    val_agg: dict[tuple, list[float]] = {}
    for (ds, m, lr, temp, _seed), payload in by_cell_seed.items():
        if payload["val"] is not None:
            val_agg.setdefault((ds, m, lr, temp), []).append(payload["val"])
        for metric, value in payload["test"].items():
            test_agg.setdefault((ds, m, lr, temp, metric), []).append(value)

    rows: list[dict] = []
    for (ds, m, lr, temp, metric), values in test_agg.items():
        vals = val_agg.get((ds, m, lr, temp), [])
        rows.append(
            {
                "dataset": ds,
                "m": m,
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
    rows.sort(key=lambda r: (r["dataset"], r["m"], r["lr"], r["temperature"], r["metric"]))
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


README_CONFIG = {
    "sports_and_outdoors": (100, 30, 5),
    "beauty": (20, 200, 3),
    "toys_and_games": (200, 20, 3),
    "cds_and_vinyl": (20, 500, 5),
}
APPENDIX_CONFIG = {
    "sports_and_outdoors": (10, 100, 2),
    "beauty": (10, 100, 3),
    "toys_and_games": (10, 50, 5),
    "cds_and_vinyl": (10, 500, 3),
}

_AXIS_SPAN = (200 - 10, 500 - 20, 5 - 1)


def _normalised_distance(a: tuple, b: tuple) -> float:
    return sum(abs(x - y) / s for x, y, s in zip(a, b, _AXIS_SPAN))


def select_val_cluster(decode_rows: list[dict], metric: str = "ndcg@10") -> tuple[list[dict], list[dict]]:
    """Per dataset, find the val argmax and the cluster of cells statistically
    tied with it (gap <= combined 1-SE band, SE = std / sqrt(n_seeds)).

    Returns (cluster_rows, summary_rows):
      cluster_rows  -- every (dataset, b, k, q) to re-decode on test at 10 seeds.
      summary_rows  -- one row/dataset: argmax, cluster size, and the README /
                       appendix coords + val scores + which source is nearest.
    """
    by_dataset: dict[str, list[dict]] = {}
    for row in decode_rows:
        if row["metric"] == metric and row["mean"] is not None:
            by_dataset.setdefault(row["dataset"], []).append(row)

    def _se(row: dict) -> float:
        n = row.get("n_seeds") or 1
        return (row["std"] or 0.0) / (float(n) ** 0.5)

    def _cell_mean(rows: list[dict], cfg: tuple) -> float | str:
        for r in rows:
            if (r["num_beams"], r["n_edges"], r["propagation_steps"]) == cfg:
                return r["mean"]
        return ""  

    cluster_rows: list[dict] = []
    summary_rows: list[dict] = []
    for dataset in sorted(by_dataset):
        rows = by_dataset[dataset]
        best = max(rows, key=lambda r: r["mean"])
        best_se = _se(best)
        cluster = [r for r in rows if (best["mean"] - r["mean"]) <= (best_se + _se(r))]
        cluster.sort(key=lambda r: -r["mean"])
        for r in cluster:
            cluster_rows.append({
                "dataset": dataset,
                "num_beams": r["num_beams"], "n_edges": r["n_edges"],
                "propagation_steps": r["propagation_steps"],
                "val_mean": r["mean"], "val_std": r["std"], "n_seeds": r["n_seeds"],
                "is_argmax": int(r is best),
            })

        sel = (best["num_beams"], best["n_edges"], best["propagation_steps"])
        readme, appendix = README_CONFIG.get(dataset), APPENDIX_CONFIG.get(dataset)
        readme_mean = _cell_mean(rows, readme) if readme else ""
        appendix_mean = _cell_mean(rows, appendix) if appendix else ""
        nearest = ""
        if isinstance(readme_mean, (int, float)) and isinstance(appendix_mean, (int, float)):
            dr, da = abs(best["mean"] - readme_mean), abs(best["mean"] - appendix_mean)
            nearest = "readme" if dr < da else ("appendix" if da < dr else "tie")
        summary_rows.append({
            "dataset": dataset, "metric": metric,
            "sel_b": sel[0], "sel_k": sel[1], "sel_q": sel[2],
            "sel_val_mean": best["mean"], "sel_val_std": best["std"],
            "cluster_size": len(cluster), "n_seeds": best["n_seeds"],
            "readme_b": readme[0] if readme else "", "readme_k": readme[1] if readme else "",
            "readme_q": readme[2] if readme else "",
            "readme_val_mean": readme_mean,
            "appendix_b": appendix[0] if appendix else "", "appendix_k": appendix[1] if appendix else "",
            "appendix_q": appendix[2] if appendix else "",
            "appendix_val_mean": appendix_mean,
            "nearest_source": nearest,
        })
    return cluster_rows, summary_rows


def select_test_confirm(
    test_rows: list[dict],
    val_selected_rows: list[dict],
    metric: str = "ndcg@10",
) -> list[dict]:
    """Per dataset, summarise the TEST confirmation of the val-selected cluster.

    Reporting rule (mirrors the authors' A3: select on val, report on test):
      - val_argmax_test_*   -- TEST score of the cell that won on VAL. Headline,
                               zero-test-peeking number.
      - best_cluster_test_* -- best TEST score among the confirmed cluster cells.
                               This DOES peek at test; report only as a secondary
                               "ceiling", never as the primary result.
    README / appendix TEST scores are filled in only when those exact configs
    happened to fall inside the confirmed cluster (otherwise blank -- they were
    not re-decoded on test).
    """
    by_ds: dict[str, dict[tuple, dict]] = {}
    for r in test_rows:
        if r["metric"] == metric and r["mean"] is not None:
            coord = (r["num_beams"], r["n_edges"], r["propagation_steps"])
            by_ds.setdefault(r["dataset"], {})[coord] = r

    val_sel = {row["dataset"]: row for row in val_selected_rows}

    rows: list[dict] = []
    for dataset in sorted(by_ds):
        cells = by_ds[dataset]
        vs = val_sel.get(dataset)
        sel_coord = (vs["sel_b"], vs["sel_k"], vs["sel_q"]) if vs else None
        argmax = cells.get(sel_coord) if sel_coord else None
        best = max(cells.values(), key=lambda r: r["mean"])
        best_coord = (best["num_beams"], best["n_edges"], best["propagation_steps"])

        readme, appendix = README_CONFIG.get(dataset), APPENDIX_CONFIG.get(dataset)
        readme_cell = cells.get(readme) if readme else None
        appendix_cell = cells.get(appendix) if appendix else None

        rows.append({
            "dataset": dataset, "metric": metric,
            "sel_b": sel_coord[0] if sel_coord else "",
            "sel_k": sel_coord[1] if sel_coord else "",
            "sel_q": sel_coord[2] if sel_coord else "",
            "val_argmax_test_mean": argmax["mean"] if argmax else "",
            "val_argmax_test_std": argmax["std"] if argmax else "",
            "n_seeds": (argmax or best)["n_seeds"],
            "best_b": best_coord[0], "best_k": best_coord[1], "best_q": best_coord[2],
            "best_cluster_test_mean": best["mean"], "best_cluster_test_std": best["std"],
            "argmax_is_best": int(argmax is not None and best_coord == sel_coord),
            "readme_test_mean": readme_cell["mean"] if readme_cell else "",
            "appendix_test_mean": appendix_cell["mean"] if appendix_cell else "",
        })
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
    parser.add_argument("--decode-val-root",
                        default="output/reproduction/rpg/grid/decode_val")
    parser.add_argument("--decode-test-confirm-root",
                        default="output/reproduction/rpg/grid/decode_test_confirm")
    parser.add_argument("--sasrec-train-root",
                        default="output/reproduction/sasrec/grid/train")
    parser.add_argument("--infer-csv", default="infer_grid.csv")
    parser.add_argument("--train-csv", default="train_grid.csv")
    parser.add_argument("--fig6-csv", default="fig6_grid.csv")
    parser.add_argument("--decode-val-csv", default="results/decode_val_grid.csv")
    parser.add_argument("--decode-val-cluster-csv", default="results/decode_val_cluster.csv")
    parser.add_argument("--decode-val-selected-csv", default="results/decode_val_selected.csv")
    parser.add_argument("--decode-test-confirm-csv", default="results/decode_test_confirm.csv")
    parser.add_argument("--decode-test-selected-csv", default="results/decode_test_selected.csv")
    parser.add_argument("--sasrec-train-csv", default="sasrec_train_grid.csv")
    args = parser.parse_args()

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else REPO_ROOT / path

    infer_rows = collect_infer(_resolve(args.infer_root))
    fig6_rows = collect_infer(_resolve(args.fig6_root))  # same summary.json layout
    decode_val_rows = collect_infer(_resolve(args.decode_val_root))  # same layout
    decode_test_rows = collect_infer(_resolve(args.decode_test_confirm_root))  # same layout
    train_rows = collect_train(_resolve(args.train_root))
    sasrec_rows = collect_sasrec_train(_resolve(args.sasrec_train_root))

    decode_fields = ["dataset", "num_beams", "n_edges", "propagation_steps",
                     "metric", "mean", "std", "n_seeds"]
    _write_csv(_resolve(args.infer_csv), infer_rows, decode_fields)
    _write_csv(_resolve(args.fig6_csv), fig6_rows, decode_fields)
    _write_csv(_resolve(args.decode_val_csv), decode_val_rows, decode_fields)

    cluster_rows, selected_rows = select_val_cluster(decode_val_rows)
    _write_csv(_resolve(args.decode_val_cluster_csv), cluster_rows,
               ["dataset", "num_beams", "n_edges", "propagation_steps",
                "val_mean", "val_std", "n_seeds", "is_argmax"])
    _write_csv(_resolve(args.decode_val_selected_csv), selected_rows,
               ["dataset", "metric", "sel_b", "sel_k", "sel_q",
                "sel_val_mean", "sel_val_std", "cluster_size", "n_seeds",
                "readme_b", "readme_k", "readme_q", "readme_val_mean",
                "appendix_b", "appendix_k", "appendix_q", "appendix_val_mean",
                "nearest_source"])

    _write_csv(_resolve(args.decode_test_confirm_csv), decode_test_rows, decode_fields)
    test_selected_rows = select_test_confirm(decode_test_rows, selected_rows)
    _write_csv(_resolve(args.decode_test_selected_csv), test_selected_rows,
               ["dataset", "metric", "sel_b", "sel_k", "sel_q",
                "val_argmax_test_mean", "val_argmax_test_std", "n_seeds",
                "best_b", "best_k", "best_q",
                "best_cluster_test_mean", "best_cluster_test_std", "argmax_is_best",
                "readme_test_mean", "appendix_test_mean"])

    _write_csv(_resolve(args.train_csv), train_rows,
               ["dataset", "m", "lr", "temperature", "metric", "mean", "std", "n_seeds",
                "val_ndcg10_mean", "val_ndcg10_std"])
    _write_csv(_resolve(args.sasrec_train_csv), sasrec_rows,
               ["dataset", "lr", "dropout", "n_blocks", "metric", "test_mean",
                "test_std", "n_seeds", "val_ndcg20_mean", "val_ndcg20_std"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
