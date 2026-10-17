#!/usr/bin/env python3
"""Collect the main results of the paper into four CSV files.

  results/main_results.csv     one row per model, dataset, training seed and
                               evaluation protocol (Tables 2 and 3): RPG (graph
                               decoding and exhaustive scoring), SASRec and
                               parameter-matched SASRec (S3-Rec recipe),
                               full-softmax SASRec and TIGER (MemGen-GR)
  results/decoding_variants.csv  exhaustive scoring, exact graph and HNSW graph
                               on the same RPG checkpoints (Table 4)
  results/latency.csv          ms per user for every model, device and batch size
                               (latest session per model and dataset)

Sources (written by the jobs listed in README.md):
  artifacts/rpg/eval_seeds/seeds/<dataset>/s<seed>/<session>/summary.json
  artifacts/rpg/history_masking/<dataset>_s<seed>.json
  artifacts/sasrec/logs/sasrec_<preset>_h<hidden>_L<layers>_e300_lr0p0003_s<seed>_{kept,removed}.log
  artifacts/memgen/logs/AmazonReviews2014/<model>/memgen_<model>_<preset>_seed<seed>[_removed]-*.log
  artifacts/rpg/eval_seeds/decoding_variants/<dataset>/<variant>/<session>/summary.json
  artifacts/{rpg,sasrec,memgen}/latency/<session>/summary.json

Usage:
  python scripts/results/collect_main.py
"""

from __future__ import annotations

import ast
import csv
import glob
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ART = REPO_ROOT / "artifacts"
DATASETS = {  # slug -> (MemGen-GR preset, display name)
    "sports_and_outdoors": ("sports", "Sports"),
    "beauty": ("beauty", "Beauty"),
    "toys_and_games": ("toys", "Toys"),
    "cds_and_vinyl": ("cds", "CDs"),
    "books": ("books", "Books"),
}
MATCHED_HIDDEN = {"sports_and_outdoors": 326, "beauty": 540, "toys_and_games": 396, "cds_and_vinyl": 328}
METRICS = ("ndcg@5", "ndcg@10", "recall@5", "recall@10")
# Runs that did not converge (validation NDCG@10 stuck far below the other seeds
# from the first epochs on); kept in the CSV, flagged, and left out of the means.
NOT_CONVERGED = {("TIGER", "sports_and_outdoors", 2025), ("TIGER", "cds_and_vinyl", 2025)}
SASREC_LINE = re.compile(r"OrderedDict\(\[\('Epoch', 0\), (.*)\]\)")


def _row(model, slug, seed, protocol, decoding, metrics, source):
    return {
        "model": model,
        "dataset": DATASETS[slug][1],
        "seed": seed,
        "protocol": protocol,
        "decoding": decoding,
        **{m: round(float(metrics[m]), 5) if m in metrics else "" for m in METRICS},
        "converged": (model, slug, seed) not in NOT_CONVERGED,
        "source": str(Path(source).relative_to(REPO_ROOT)),
    }


def rpg_rows():
    rows = []
    for slug in DATASETS:
        for seed in (2024, 2025, 2026):
            sessions = sorted(glob.glob(str(ART / f"rpg/eval_seeds/seeds/{slug}/s{seed}/*/summary.json")))
            if sessions:
                summary = json.load(open(sessions[-1]))
                metrics = {r["metric"]: r["final_user_avg"] for r in summary["metric_summary"]}
                rows.append(_row("RPG", slug, seed, "kept", "graph", metrics, sessions[-1]))
            masking = ART / f"rpg/history_masking/{slug}_s{seed}.json"
            if masking.is_file():
                data = json.load(open(masking))
                rows.append(_row("RPG", slug, seed, "kept", "exhaustive", data["past_items_kept"], masking))
                rows.append(_row("RPG", slug, seed, "removed", "exhaustive", data["past_items_removed"], masking))
    return rows


def sasrec_rows():
    rows = []
    for slug in DATASETS:
        variants = [("SASRec", 64, 2)]
        if slug in MATCHED_HIDDEN:
            variants.append(("SASRec (parameter-matched)", MATCHED_HIDDEN[slug], 3 if slug == "beauty" else 2))
        for model, hidden, layers in variants:
            for seed in (2024, 2025, 2026):
                for protocol in ("kept", "removed"):
                    log = ART / f"sasrec/logs/sasrec_{slug}_h{hidden}_L{layers}_e300_lr0p0003_s{seed}_{protocol}.log"
                    if not log.is_file():
                        continue
                    lines = [m.group(1) for m in map(SASREC_LINE.search, log.read_text().splitlines()) if m]
                    if not lines:
                        continue
                    values = dict(ast.literal_eval("[" + lines[-1] + "]"))
                    metrics = {k.lower(): v for k, v in values.items()}
                    rows.append(_row(model, slug, seed, protocol, "", metrics, log))
    return rows


def memgen_rows():
    rows = []
    for slug, (preset, _) in DATASETS.items():
        for model in ("SASRec", "TIGER"):
            name = "SASRec (full softmax)" if model == "SASRec" else "TIGER"
            for seed in (2024, 2025, 2026, 2027):
                for protocol, tag in (("kept", ""), ("removed", "_removed")):
                    logs = sorted(glob.glob(str(ART / f"memgen/logs/AmazonReviews2014/{model}/memgen_{model.lower()}_{preset}_seed{seed}{tag}-*.log")))
                    for log in reversed(logs):  # newest log that has a test line
                        text = Path(log).read_text(errors="replace")
                        hits = re.findall(r"Test Results\s*:\s*(OrderedDict\(.*\)|\{.*\})", text)
                        if not hits:
                            continue
                        raw = hits[-1]
                        values = dict(ast.literal_eval(raw[len("OrderedDict("):-1])) if raw.startswith("OrderedDict(") else ast.literal_eval(raw)
                        rows.append(_row(name, slug, seed, protocol, "", values, log))
                        break
    return rows


def decoding_variant_rows():
    rows = []
    for slug in DATASETS:
        for variant in ("exhaustive", "exact_gpu", "hnsw"):
            sessions = sorted(glob.glob(str(ART / f"rpg/eval_seeds/decoding_variants/{slug}/{variant}/*/summary.json")))
            if not sessions:
                continue
            summary = json.load(open(sessions[-1]))
            metrics = {r["metric"]: r["final_user_avg"] for r in summary["metric_summary"]}
            rows.append({
                "dataset": DATASETS[slug][1],
                "decoding": variant,
                "eval_seeds": len(summary.get("eval_seeds", [])),
                **{m: round(metrics[m], 5) for m in METRICS},
                "visited_items_per_user": round(metrics.get("n_visited_items", 0)),
                "source": str(Path(sessions[-1]).relative_to(REPO_ROOT)),
            })
    return rows


def latency_rows():
    latest = {}
    for root, model in (("rpg", "RPG"), ("sasrec", "SASRec"), ("memgen", None)):
        for path in glob.glob(str(ART / f"{root}/latency/*/summary.json")):
            summary = json.load(open(path))
            name = model or ("SASRec (full softmax)" if summary.get("model") == "SASRec" else summary.get("model"))
            category = summary.get("category") or summary.get("dataset")
            key = (name, category, summary.get("n_codebook", ""))
            session = Path(path).parent.name
            if key not in latest or session > latest[key][0]:
                latest[key] = (session, summary, path)
    rows = []
    for (name, category, n_codebook), (_, summary, path) in sorted(latest.items(), key=lambda x: str(x[0])):
        for r in summary["rows"]:
            rows.append({
                "model": name, "dataset": category, "n_codebook": n_codebook,
                "device": r["device"], "batch_size": r["batch_size"], "decoding": r.get("decoding", ""),
                "num_beams": r.get("num_beams", ""), "n_edges": r.get("n_edges", ""), "propagation_steps": r.get("propagation_steps", ""),
                "ms_per_user": r["ms_per_user"], "ndcg@10": r.get("ndcg@10", ""),
                "source": str(Path(path).relative_to(REPO_ROOT)),
            })
    return rows


def write(path, rows):
    path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows):4d} rows to {path.relative_to(REPO_ROOT)}")


def print_summary(rows):
    groups = defaultdict(list)
    for r in rows:
        if r["converged"]:
            groups[(r["model"], r["decoding"], r["protocol"], r["dataset"])].append(r["ndcg@10"])
    print("\nNDCG@10, mean over converged seeds (number of seeds):")
    datasets = [d for _, d in DATASETS.values()]
    keys = sorted({k[:3] for k in groups})
    for model, decoding, protocol in keys:
        cells = []
        for d in datasets:
            v = groups.get((model, decoding, protocol, d))
            cells.append(f"{statistics.mean(v):.4f} ({len(v)})" if v else "      -    ")
        label = f"{model}{' ' + decoding if decoding else ''}, {protocol}"
        print(f"  {label:45s} " + "  ".join(cells))


def main() -> int:
    main_rows = rpg_rows() + sasrec_rows() + memgen_rows()
    write("results/main_results.csv", main_rows)
    write("results/decoding_variants.csv", decoding_variant_rows())
    write("results/latency.csv", latency_rows())
    print_summary(main_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
