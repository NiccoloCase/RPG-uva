#!/usr/bin/env python3
"""Collect the RPG decoding-variant results of the second code base into one CSV.

The second code base ran upstream RPG with two decoders on the four paper
datasets: the original dense graph ("original", directory RPG_original) and an
approximate HNSW graph over mean token embeddings ("hnsw", directory RPG_HNSW).
Each run wrote one upstream log whose file name carries the category and the
decoding budget and whose last lines report the test metrics. This script reads
every log that contains a test line, keeps the paper datasets, and writes one
row per run plus a per-(variant, dataset) summary.

Usage:
  python scripts/results/collect_memgen_rpg_logs.py --logs-root /path/to/recsys_group9
Writes results/decoding_variants_logs.csv (one row per run) and prints the summary.
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VARIANT_DIRS = {
    "original": "RPG_original/logs/AmazonReviews2014/RPG",
    "hnsw": "RPG_HNSW/RPG_KDD2025/logs/AmazonReviews2014/RPG",
}
PAPER_CATEGORIES = ("Sports_and_Outdoors", "Beauty", "Toys_and_Games", "CDs_and_Vinyl")
METRICS = ("ndcg@5", "ndcg@10", "recall@5", "recall@10")
TEST_LINE = re.compile(r"Test Results\s*:\s*(.*)$")
ARG_RE = re.compile(r"--(category|lr|temperature|n_codebook|num_beams|n_edges|propagation_steps|rand_seed)=([A-Za-z0-9_.]+)")
DATE_RE = re.compile(r"-([A-Z][a-z]{2}-\d{2}-\d{4}_\d{2}-\d{2})-[0-9a-f]{6}\.log$")


def parse_test_line(text: str) -> dict[str, float] | None:
    """Return the test metrics from a log, from either the upstream or the reformatted line."""
    match = None
    for line in text.splitlines():
        m = TEST_LINE.search(line)
        if m:
            match = m.group(1).strip()
    if match is None:
        return None
    if match.startswith("OrderedDict("):
        pairs = ast.literal_eval(match[len("OrderedDict(") : -1])
        values = dict(pairs)
    else:
        values = ast.literal_eval(match)
    return {k: float(values[k]) for k in METRICS if k in values}


def collect(logs_root: Path) -> list[dict]:
    rows = []
    for variant, rel in VARIANT_DIRS.items():
        for path in sorted((logs_root / rel).glob("*.log")):
            args = {key: value.rstrip("_") for key, value in ARG_RE.findall(path.name)}  # values end in "_" before the next flag
            if args.get("category") not in PAPER_CATEGORIES:
                continue
            metrics = parse_test_line(path.read_text(errors="replace"))
            if metrics is None:
                continue
            date = DATE_RE.search(path.name)
            rows.append({
                "variant": variant,
                "category": args["category"],
                "n_codebook": int(args.get("n_codebook", 0)),
                "lr": float(args.get("lr", 0)),
                "temperature": float(args.get("temperature", 0)),
                "num_beams": int(args.get("num_beams", 0)),
                "n_edges": int(args.get("n_edges", 0)),
                "propagation_steps": int(args.get("propagation_steps", 0)),
                "rand_seed": int(args["rand_seed"]) if "rand_seed" in args else "",
                **{m: round(metrics[m], 5) for m in METRICS},
                "run_date": date.group(1) if date else "",
                "log_file": path.name,
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs-root", required=True, help="Checkout of the second code base (contains RPG_original/ and RPG_HNSW/).")
    parser.add_argument("--csv", default="results/decoding_variants_logs.csv")
    args = parser.parse_args()
    rows = collect(Path(args.logs_root).expanduser().resolve())
    if not rows:
        raise SystemExit("no logs with test results found")
    out = REPO_ROOT / args.csv
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}\n")
    print(f"{'variant':9s} {'dataset':20s} {'runs':>4s} {'ndcg@10':>8s} {'sd':>7s} {'recall@10':>9s}")
    for variant in VARIANT_DIRS:
        for category in PAPER_CATEGORIES:
            sel = [r for r in rows if r["variant"] == variant and r["category"] == category]
            if not sel:
                continue
            n = [r["ndcg@10"] for r in sel]
            sd = statistics.stdev(n) if len(n) > 1 else 0.0
            print(f"{variant:9s} {category:20s} {len(sel):4d} {statistics.mean(n):8.4f} {sd:7.4f} {statistics.mean(r['recall@10'] for r in sel):9.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
