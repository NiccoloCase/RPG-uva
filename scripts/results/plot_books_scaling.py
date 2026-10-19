#!/usr/bin/env python3
"""Figures for RQ3 and RQ4 (graph decoding, Amazon Books, latency).

Reads the collected results and writes three figures to results/figures/,
as PDF for the paper and PNG for previews:

                     decoding scores and the share of the exhaustive top-10
                     hits it returns, at the released decoding settings.
                     Source: artifacts/rpg/verification/*.json
                     (scripts/rpg/verify_decoding.py).
  books_budget.pdf   on Books, NDCG@10 of graph decoding against the share of
                     the catalogue it scores, for both semantic ID lengths,
                     with exhaustive scoring as the reference.
                     Source: results/books_scaling.csv (scripts/results/collect_books.py).
  latency.pdf        ms per user against catalogue size for RPG graph decoding,
                     RPG exhaustive scoring, SASRec and TIGER, on GPU and CPU at
                     batch size 1 and 128.
                     Source: results/latency.csv (scripts/results/collect_main.py).
  graph_memory.pdf   peak GPU memory of building the decoding graph with the
                     released dense builder and the row-blocked exact builder.
                     Source: artifacts/rpg/graph_build/*.json (scripts/rpg/graph_build_timing.py).

Usage:
  python scripts/results/plot_books_scaling.py [--books-digits 64]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
BOOKS_ITEMS = 367_982
ITEMS = {"Sports_and_Outdoors": 18_357, "CDs_and_Vinyl": 64_443, "Books": BOOKS_ITEMS}
# Categorical slots of the validated default palette, each paired with a marker.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, CONTEXT = "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5,
    "pdf.fonttype": 42,
})
MARK = dict(markersize=3.8, markeredgecolor="white", markeredgewidth=0.6)


def _grid(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _shared_xlabel(fig, axes, text):
    """One x label for a row of panels, placed just below the lowest tick labels."""
    fig.canvas.draw()
    bottom = min(ax.get_tightbbox(fig.canvas.get_renderer()).y0 for ax in axes)
    y = fig.transFigure.inverted().transform((0, bottom))[1]
    fig.text(0.5, y - 0.02, text, ha="center", va="top", fontsize=7, color=INK)


def _save(fig, out_dir, name, fixed=False):
    """fixed=True keeps the canvas size, so paired panels come out the same size."""
    if fixed:
        fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=200, bbox_inches=None if fixed else "tight")
    plt.close(fig)


def plot_budget(out_dir: Path) -> None:
    rows = list(csv.DictReader((REPO_ROOT / "results/books_scaling.csv").open()))
    fig, ax = plt.subplots(figsize=(3.0, 1.9))
    for run, label, color, marker in (("books", "$m$=64", BLUE, "o"), ("books_m32", "$m$=32", ORANGE, "s")):
        # Graph-decoding points with 5 propagation steps; more steps change nothing (see text).
        graph = sorted(
            (int(r["visited_items_per_user"]) / BOOKS_ITEMS, float(r["ndcg@10"]))
            for r in rows
            if r["run"] == run and r["decoding"] == "graph" and r["graph_backend"] == "exact_gpu"
            and (r["budget"] == "" or r["budget"].endswith("_q5"))
        )
        exhaustive = [float(r["ndcg@10"]) for r in rows if r["run"] == run and r["decoding"] == "exhaustive"][0]
        ax.plot([x for x, _ in graph], [y for _, y in graph], "-" + marker, color=color, label=f"graph decoding, {label}", **MARK)
        ax.axhline(exhaustive, color=color, linestyle="--", linewidth=1.0)
        ax.text(0.01, exhaustive, f"exhaustive, {label}", color=INK, fontsize=6, va="bottom", ha="left")
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of the catalogue scored per user")
    ax.set_ylabel("NDCG@10")
    ax.legend(frameon=False, loc="lower right")
    _grid(ax)
    _save(fig, out_dir, "books_budget", fixed=True)


def plot_latency(out_dir: Path, books_digits: int) -> None:
    rows = list(csv.DictReader((REPO_ROOT / "results/latency.csv").open()))

    def series(model, device, batch, decoding=""):
        points = []
        for r in rows:
            if r["model"] != model or r["device"] != device or int(r["batch_size"]) != batch:
                continue
            if model == "RPG":
                if r["decoding"] != decoding:
                    continue
                if r["dataset"] == "Books" and int(r["n_codebook"]) != books_digits:
                    continue
                if decoding == "graph" and r["n_edges"] == "2000":  # the large-budget point, reported in the text
                    continue
            points.append((ITEMS[r["dataset"]], float(r["ms_per_user"])))
        return sorted(points)

    fig, axes = plt.subplots(1, 4, figsize=(4.8, 1.55), sharex=True)
    panels = [("cuda", 1), ("cuda", 128), ("cpu", 1), ("cpu", 128)]
    lines = [
        (("RPG", "graph"), "RPG, graph decoding", BLUE, "o"),
        (("RPG", "exhaustive"), "RPG, exhaustive scoring", ORANGE, "s"),
        (("TIGER", ""), "TIGER", YELLOW, "D"),
        (("SASRec", ""), "SASRec", AQUA, "^"),
    ]
    for ax, (device, batch) in zip(axes, panels):
        for (model, decoding), label, color, marker in lines:
            pts = series(model, device, batch, decoding)
            ax.plot([x for x, _ in pts], [y for _, y in pts], "-" + marker, color=color, label=label, **MARK)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([2e4, 1e5, 4e5])
        ax.set_xticklabels(["20k", "100k", "400k"])
        ax.minorticks_off()
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_title(f"{'GPU' if device == 'cuda' else 'CPU'}, batch {batch}", color=INK)
        _grid(ax)
    axes[0].set_ylabel("ms per user")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.5)
    _shared_xlabel(fig, axes, "catalogue size")
    _save(fig, out_dir, "latency")


def plot_graph_memory(out_dir: Path) -> None:
    """Peak GPU memory of building the decoding graph, dense vs row-blocked."""
    runs = [json.loads(path.read_text()) for path in sorted((REPO_ROOT / "artifacts/rpg/graph_build").glob("*.json"))]
    dense = sorted((r["n_items"], r["builders"]["dense"]["peak_gpu_gib"] * 2**30 / 1e9) for r in runs if "peak_gpu_gib" in r["builders"].get("dense", {}))
    blocked = {}
    for r in runs:
        blocked[r["n_items"]] = max(blocked.get(r["n_items"], 0), r["builders"]["exact_gpu"]["peak_gpu_gib"] * 2**30 / 1e9)
    blocked = sorted(blocked.items())
    fig, ax = plt.subplots(figsize=(3.0, 1.9))
    ns = [1e4 * 1.2 ** i for i in range(0, 60) if 1e4 * 1.2 ** i <= 1.2e6]
    ax.plot(ns, [4 * n * n / 1e9 for n in ns], "--", color=ORANGE, linewidth=1.0, label="dense, $4N^2$ bytes")
    ax.plot([n for n, _ in dense], [g for _, g in dense], "s", color=ORANGE, label="dense, measured", **MARK)
    ax.plot([n for n, _ in blocked], [g for _, g in blocked], "-o", color=BLUE, label="row-blocked, measured", **MARK)
    ax.axhspan(24, 40, color=GRID, alpha=0.6, linewidth=0)
    ax.text(1.4e6, 31, "GPU memory\n24-40 GB", color=MUTED, fontsize=6, va="center", ha="right")
    ax.axvline(BOOKS_ITEMS, color=CONTEXT, linewidth=0.8)
    ax.text(BOOKS_ITEMS * 1.05, 0.03, "Books", color=MUTED, fontsize=6, rotation=90, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1e4, 1e5, 1e6])
    ax.set_xticklabels(["10k", "100k", "1M"])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("catalogue size $N$")
    ax.set_ylabel("peak GPU memory (GB)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    _grid(ax)
    _save(fig, out_dir, "graph_memory", fixed=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--books-digits", type=int, default=64, help="Semantic ID length of the Books model in the latency figure.")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_budget(out_dir)
    plot_latency(out_dir, args.books_digits)
    plot_graph_memory(out_dir)
    for name in ("books_budget.pdf", "latency.pdf", "graph_memory.pdf"):
        print(out_dir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
