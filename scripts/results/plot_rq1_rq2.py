#!/usr/bin/env python3
"""Appendix figures for RQ1 and RQ2.

  results/figures/semantic_id_length.pdf  NDCG@10 of RPG against the semantic ID
      length m, one panel per dataset, with the optimum reported by Hou et al.
      (claim C2). Source: results/redecode_compare.csv.
  results/figures/cold_start.pdf  NDCG@10 by training frequency of the test item,
      for RPG, SASRec and parameter-matched SASRec, mean over three training
      seeds, past items kept (claim C3). Source: the cold-start summaries in
      artifacts/rpg/cold_start/ and artifacts/sasrec/cold_start/.

Usage:
  python scripts/results/plot_rq1_rq2.py
"""

from __future__ import annotations

import csv
import glob
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "results" / "figures"
DATASETS = [("sports_and_outdoors", "Sports_and_Outdoors", "Sports"), ("beauty", "Beauty", "Beauty"),
            ("toys_and_games", "Toys_and_Games", "Toys"), ("cds_and_vinyl", "CDs_and_Vinyl", "CDs")]
PAPER_OPTIMUM = {"Sports": 16, "Beauty": 32, "Toys": 16, "CDs": 64}
# Categorical slots 1-3 of the validated default palette; each series also has
# its own marker, since slots 3 and 4 are below 3:1 contrast on white.
SERIES = [("RPG", "#2a78d6", "o"), ("SASRec, parameter-matched", "#eb6834", "s"), ("SASRec", "#1baf7a", "^")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5,
    "pdf.fonttype": 42,
})


def _style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_length():
    rows = list(csv.DictReader(open(REPO_ROOT / "results/redecode_compare.csv")))
    fig, axes = plt.subplots(1, 4, figsize=(4.8, 1.45))
    for ax, (_, category, name) in zip(axes, DATASETS):
        pts = sorted((int(r["m"]), float(r["repo_harness_mean_ndcg@10"])) for r in rows if r["dataset"] == category)
        ms, ys = zip(*pts)
        ax.plot(ms, ys, "-o", color=SERIES[0][1], markersize=3.5, markeredgecolor="white", markeredgewidth=0.6)
        opt = PAPER_OPTIMUM[name]
        ax.axvline(opt, color=MUTED, linewidth=0.8, linestyle=":")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ms)
        ax.set_xticklabels([str(m) for m in ms])
        ax.set_title(name, color=INK)
        ax.set_xlabel("$m$")
        _style(ax)
    axes[0].set_ylabel("NDCG@10")
    fig.tight_layout(w_pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"semantic_id_length.{ext}", dpi=200)
    plt.close(fig)


def _bucket_means(pattern):
    runs = []
    for path in glob.glob(pattern):
        groups = json.load(open(path))["group_rows"]
        runs.append({g["bucket_range"]: g["ndcg@10"] for g in groups})
    if not runs:
        return None, 0
    buckets = list(runs[0])
    return {b: statistics.mean(r[b] for r in runs) for b in buckets}, len(runs)


def plot_cold_start():
    fig, axes = plt.subplots(1, 4, figsize=(4.8, 1.6))
    for ax, (slug, _, name) in zip(axes, DATASETS):
        sources = [
            f"artifacts/rpg/cold_start/{slug}_s*/*/tables/cold_start_summary.json",
            f"artifacts/sasrec/cold_start/{slug}_pm_s*/*/tables/cold_start_summary.json",
            f"artifacts/sasrec/cold_start/{slug}_h64_s*/*/tables/cold_start_summary.json",
        ]
        for (label, color, marker), pattern in zip(SERIES, sources):
            means, n = _bucket_means(str(REPO_ROOT / pattern))
            if not means:
                continue
            xs = list(range(len(means)))
            ax.plot(xs, list(means.values()), "-" + marker, color=color, markersize=3.5,
                    markeredgecolor="white", markeredgewidth=0.6, label=label)
            ax.set_xticks(xs)
            ax.set_xticklabels([b.strip("[]").replace("0, 5", "0-5").replace(", ", "-") for b in means], fontsize=5.5)
        ax.set_title(name, color=INK)
        ax.set_xlabel("training interactions")
        _style(ax)
    axes[0].set_ylabel("NDCG@10")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"cold_start.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_length()
    plot_cold_start()
    print(OUT / "semantic_id_length.pdf")
    print(OUT / "cold_start.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
