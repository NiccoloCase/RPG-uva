#!/usr/bin/env python3
"""Build the paper / appendix / repo comparison for the m-sweep.

Pulls three sources together for each (dataset, m):

  * repo (original)   - parsed from the original m-sweep SLURM .err logs
                        (the numbers you already reported).
  * repo (harness)    - re-decoded repo config via redecode_configs.sh.
                        Used ONLY to validate that the eval-harness path
                        reproduces the training-pipeline numbers.
  * appendix (harness)- re-decoded paper_appendix (Table 6) config.

It then validates repo(harness) against repo(original), and prints a
per-dataset table against the paper's reported anchors.

Usage:
    python scripts/collect_redecode_results.py
    python scripts/collect_redecode_results.py --metric ndcg@10 --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP_LOG_DIR = REPO_ROOT / "output" / "reproduction" / "train" / "sweep_m"
DEFAULT_REDECODE_DIR = REPO_ROOT / "output" / "reproduction" / "train" / "sweep_m_redecode"

M_GRID = [4, 8, 16, 32, 64]
DATASETS = ["Sports_and_Outdoors", "Beauty", "Toys_and_Games", "CDs_and_Vinyl"]

PAPER_NDCG10 = {
    "Sports_and_Outdoors": {4: 0.0117, 16: 0.0263},
    "Beauty": {4: 0.0235, 32: 0.0464},
    "Toys_and_Games": {4: 0.0275, 16: 0.0490},
    "CDs_and_Vinyl": {4: 0.0175, 64: 0.0415},
}

SWEEP_START_RE = re.compile(r"SWEEP_START\s+dataset=(\S+)\s+n_codebook=(\d+)")
KV_RE = re.compile(r"\('([^']+)',\s*([\d.eE+-]+)\)")


def parse_sweep_logs(log_dir: Path) -> dict[tuple[str, int], dict[str, float]]:
    """(dataset, m) -> {metric: value} from the original sweep stdout/stderr."""
    out: dict[tuple[str, int], dict[str, float]] = {}
    if not log_dir.exists():
        return out
    for err in sorted(log_dir.glob("*.err")):
        text = err.read_text(errors="replace")
        companion = err.with_suffix(".out")
        if companion.exists():
            text += companion.read_text(errors="replace")
        dataset = m = metrics = None
        for line in text.splitlines():
            s = SWEEP_START_RE.search(line)
            if s:
                dataset, m = s.group(1), int(s.group(2))
            if "Test Results:" in line:
                pairs = KV_RE.findall(line)
                if pairs:
                    metrics = {k: float(v) for k, v in pairs}
        if dataset is not None and metrics is not None:
            out[(dataset, m)] = metrics
    return out


TAG_M_RE = re.compile(r"(repo|appendix)_m(\d+)")


def parse_redecode(redecode_dir: Path) -> dict[tuple[str, str, int], dict[str, tuple[float, float]]]:
    """(category, tag, m) -> {metric: (mean_over_seeds, std_over_seeds)}.

    Keeps the most recently modified session when a config was run more than once.
    """
    best_session: dict[tuple[str, str, int], Path] = {}
    if not redecode_dir.exists():
        return {}
    for summary in redecode_dir.rglob("summary.json"):
        tag_m = None
        for part in summary.parts:
            match = TAG_M_RE.fullmatch(part)
            if match:
                tag_m = (match.group(1), int(match.group(2)))
                break
        if tag_m is None:
            continue
        try:
            payload = json.loads(summary.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        category = payload.get("category")
        if category is None:
            continue
        key = (category, tag_m[0], tag_m[1])
        prev = best_session.get(key)
        if prev is None or summary.stat().st_mtime > prev.stat().st_mtime:
            best_session[key] = summary

    results: dict[tuple[str, str, int], dict[str, tuple[float, float]]] = {}
    for key, summary in best_session.items():
        payload = json.loads(summary.read_text())
        per_metric: dict[str, tuple[float, float]] = {}
        for row in payload.get("metric_summary", []):
            per_metric[row["metric"]] = (
                float(row.get("eval_seed_mean", float("nan"))),
                float(row.get("eval_seed_std", 0.0)),
            )
        results[key] = per_metric
    return results


def fmt(value: float | None, std: float | None = None) -> str:
    if value is None:
        return "   -   "
    if std is None:
        return f"{value:.4f} "
    return f"{value:.4f}±{std:.4f}"


def validate(sweep, redecode, metric) -> list[str]:
    lines = []
    for ds in DATASETS:
        for m in M_GRID:
            orig = sweep.get((ds, m), {}).get(metric)
            har = redecode.get((ds, "repo", m), {}).get(metric)
            if orig is None or har is None:
                continue
            har_mean, har_std = har
            delta = har_mean - orig
            tol = max(2 * har_std, 0.0015)
            verdict = "PASS" if abs(delta) <= tol else "WARN"
            lines.append(
                f"  [{verdict}] {ds:<20} m={m:<2}  orig={orig:.4f}  "
                f"harness={har_mean:.4f}±{har_std:.4f}  Δ={delta:+.4f}  (tol={tol:.4f})"
            )
    if not lines:
        lines.append("  (no overlapping repo runs to validate yet)")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep-log-dir", default=str(DEFAULT_SWEEP_LOG_DIR))
    parser.add_argument("--redecode-dir", default=str(DEFAULT_REDECODE_DIR))
    parser.add_argument("--metric", default="ndcg@10", help="Metric for the printed table (default: %(default)s).")
    parser.add_argument("--csv", default=None, help="Optional path to dump every metric for every cell.")
    args = parser.parse_args()

    sweep = parse_sweep_logs(Path(args.sweep_log_dir))
    redecode = parse_redecode(Path(args.redecode_dir))
    metric = args.metric

    print("=" * 78)
    print(f"VALIDATION  repo(harness) vs repo(original)   metric={metric}")
    print("  PASS => eval-harness path reproduces the training-pipeline numbers,")
    print("          the appendix re-decodes are directly comparable.")
    print("=" * 78)
    for line in validate(sweep, redecode, metric):
        print(line)

    print()
    print("=" * 78)
    print(f"COMPARISON   metric={metric}   (harness cells are mean±std over eval seeds)")
    print("=" * 78)
    header = f"{'dataset':<20} {'m':>3}  {'paper':>8}  {'repo(orig)':>10}  {'repo(harn)':>14}  {'appendix(harn)':>14}"
    for ds in DATASETS:
        print(f"\n{ds}")
        print(header)
        print("-" * len(header))
        for m in M_GRID:
            paper = PAPER_NDCG10.get(ds, {}).get(m) if metric == "ndcg@10" else None
            orig = sweep.get((ds, m), {}).get(metric)
            har_repo = redecode.get((ds, "repo", m), {}).get(metric)
            har_app = redecode.get((ds, "appendix", m), {}).get(metric)
            print(
                f"{'':<20} {m:>3}  "
                f"{fmt(paper):>8}  "
                f"{fmt(orig):>10}  "
                f"{(fmt(*har_repo) if har_repo else fmt(None)):>14}  "
                f"{(fmt(*har_app) if har_app else fmt(None)):>14}"
            )

    if args.csv:
        all_metrics = sorted({m for d in redecode.values() for m in d} | {m for d in sweep.values() for m in d})
        rows = []
        for ds in DATASETS:
            for m in M_GRID:
                row = {"dataset": ds, "m": m}
                for mt in all_metrics:
                    o = sweep.get((ds, m), {}).get(mt)
                    rp = redecode.get((ds, "repo", m), {}).get(mt)
                    ap = redecode.get((ds, "appendix", m), {}).get(mt)
                    row[f"repo_orig_{mt}"] = o
                    row[f"repo_harness_mean_{mt}"] = rp[0] if rp else None
                    row[f"repo_harness_std_{mt}"] = rp[1] if rp else None
                    row[f"appendix_harness_mean_{mt}"] = ap[0] if ap else None
                    row[f"appendix_harness_std_{mt}"] = ap[1] if ap else None
                rows.append(row)
        csv_path = Path(args.csv)
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote per-metric CSV to {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
