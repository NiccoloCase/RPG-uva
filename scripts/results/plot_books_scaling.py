#!/usr/bin/env python3
"""Figures for the large-catalogue (Amazon Books) analysis.

Reads the collected results and writes three figures to results/figures/,
as PDF for the paper and PNG for previews:

  books_budget.pdf       NDCG@10 of graph decoding against the share of the
                         catalogue it scores, for both semantic-ID lengths,
                         with exhaustive scoring as the reference.
                         Source: results/books_scaling.csv
                         (scripts/results/collect_books.py).
  reachability.pdf       share of the catalogue scored and share of the
                         exhaustive top-10 hits that graph decoding keeps, per
                         dataset, at the published decoding budget.
                         Source: artifacts/rpg/verification/*.json
                         (scripts/rpg/verify_decoding.py).
  latency.pdf            per-user latency against catalogue size for RPG
                         graph decoding, RPG exhaustive scoring and SASRec,
                         on GPU and CPU at batch size 1 and 16.
                         Sources: artifacts/rpg/latency/*/summary.json
                         (scripts/rpg/latency.py) and
                         artifacts/sasrec/latency/*/summary.json
                         (scripts/sasrec/latency.py); the newest session per
                         dataset and semantic-ID length is used.

Usage:
  python scripts/results/plot_books_scaling.py [--output-dir results/figures] [--books-digits 64]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
BOOKS_ITEMS = 367_982
DATASET_LABELS = {"Sports_and_Outdoors": "Sports", "CDs_and_Vinyl": "CDs", "Books": "Books"}


def plot_budget(out_dir: Path) -> None:
    rows = list(csv.DictReader((REPO_ROOT / "results/books_scaling.csv").open()))
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for run, label, color in (("books", "64 digits", "tab:blue"), ("books_m32", "32 digits", "tab:orange")):
        # Graph-decoding points with 5 propagation steps; the steps sweep is reported separately.
        graph = sorted(
            (int(r["visited_items_per_user"]) / BOOKS_ITEMS, float(r["ndcg@10"]))
            for r in rows
            if r["run"] == run
            and r["decoding"] == "graph"
            and r["graph_backend"] == "exact_gpu"
            and (r["budget"] == "" or r["budget"].endswith("_q5"))
        )
        exhaustive = [float(r["ndcg@10"]) for r in rows if r["run"] == run and r["decoding"] == "exhaustive"]
        ax.plot([x for x, _ in graph], [y for _, y in graph], marker="o", color=color, label=f"graph decoding, {label}")
        if exhaustive:
            ax.axhline(exhaustive[0], color=color, linestyle="--", linewidth=1, label=f"exhaustive scoring, {label}")
    ax.set_xlabel("Share of catalogue scored per user")
    ax.set_ylabel("NDCG@10")
    ax.set_xlim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "books_budget.pdf")
    fig.savefig(out_dir / "books_budget.png", dpi=150)
    plt.close(fig)


def plot_reachability(out_dir: Path) -> None:
    order = [("sports", "Sports\n18k"), ("cds", "CDs\n64k"), ("books_m32", "Books\n368k, 32 dig."), ("books_m64", "Books\n368k, 64 dig.")]
    data = []
    for name, label in order:
        path = REPO_ROOT / "artifacts/rpg/verification" / f"{name}.json"
        if path.is_file():
            d = json.loads(path.read_text())
            data.append((label, d["D_share_of_catalogue_visited"], d["D_of_exhaustive_hits_kept_by_graph"]))
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    positions = range(len(data))
    width = 0.38
    ax.bar([p - width / 2 for p in positions], [v for _, v, _ in data], width, label="catalogue scored", color="tab:gray")
    ax.bar([p + width / 2 for p in positions], [k for _, _, k in data], width, label="exhaustive top-10 hits kept", color="tab:blue")
    ax.set_xticks(list(positions))
    ax.set_xticklabels([label for label, _, _ in data], fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "reachability.pdf")
    fig.savefig(out_dir / "reachability.png", dpi=150)
    plt.close(fig)


def newest_sessions(root: Path, key_fields: tuple[str, ...]) -> list[dict]:
    newest: dict[tuple, tuple[str, dict]] = {}
    for path in root.glob("*/summary.json"):
        d = json.loads(path.read_text())
        key = tuple(d.get(field) for field in key_fields)
        if key not in newest or path.parent.name > newest[key][0]:
            newest[key] = (path.parent.name, d)
    return [d for _, d in newest.values()]


def plot_latency(out_dir: Path, books_digits: int) -> None:
    rpg = [d for d in newest_sessions(REPO_ROOT / "artifacts/rpg/latency", ("category", "n_codebook"))
           if d["category"] != "Books" or d["n_codebook"] == books_digits]
    sasrec = newest_sessions(REPO_ROOT / "artifacts/sasrec/latency", ("dataset",))

    def series(sessions, size_key, device, batch, decoding=None):
        points = []
        for d in sessions:
            for r in d["rows"]:
                if r["device"] == device and r["batch_size"] == batch and (decoding is None or r["decoding"] == decoding):
                    if decoding == "graph" and r["num_beams"] == 200 and r["n_edges"] == 2000:
                        continue
                    points.append((d[size_key], r["ms_per_user"]))
        return sorted(points)

    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8), sharex=True)
    panels = [("cuda", 1), ("cuda", 16), ("cpu", 1), ("cpu", 16)]
    for ax, (device, batch) in zip(axes, panels):
        for points, label, style in (
            (series(rpg, "n_items", device, batch, "graph"), "RPG graph decoding", "o-"),
            (series(rpg, "n_items", device, batch, "exhaustive"), "RPG exhaustive scoring", "s-"),
            (series(sasrec, "n_items", device, batch), "SASRec", "^-"),
        ):
            ax.plot([x for x, _ in points], [y for _, y in points], style, label=label, markersize=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{'GPU' if device == 'cuda' else 'CPU'}, batch {batch}", fontsize=9)
        ax.set_xlabel("Catalogue size (items)")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("ms per user")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "latency.pdf")
    fig.savefig(out_dir / "latency.png", dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--books-digits", type=int, default=64, help="Semantic-ID length of the Books model in the latency figure.")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_budget(out_dir)
    plot_reachability(out_dir)
    plot_latency(out_dir, args.books_digits)
    for name in ("books_budget.pdf", "reachability.pdf", "latency.pdf"):
        print(out_dir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
