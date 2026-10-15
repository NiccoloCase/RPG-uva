#!/usr/bin/env python3
"""Collect the Books (large-catalogue) RPG results into one table.

Reads what jobs/books/eval.sh and budget_sweep.sh write:

  artifacts/rpg/eval_seeds/new_datasets/books[_TAG]/graph/<session>/summary.json
  artifacts/rpg/eval_seeds/new_datasets/books[_TAG]/exhaustive/<session>/summary.json
  artifacts/rpg/eval_seeds/new_datasets/books[_TAG]/budget/b<b>_k<k>_q<q>/<session>/summary.json

and the decoding time per eval seed from the matching logs in
output/new_datasets/rpg/books/ (the tqdm elapsed time of "Eval seed <s>").

Writes results/books_scaling.csv with one row per (run, decoding, budget,
session): ndcg/recall at 5 and 10, visited items per user, graph
backend, number of eval seeds, and seconds per user where a log was found.

stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "artifacts/rpg/eval_seeds/new_datasets"
LOG_ROOT = REPO_ROOT / "output/new_datasets/rpg/books"
METRICS = ("ndcg@5", "ndcg@10", "recall@5", "recall@10")
SEED_LINE = re.compile(r"Eval seed (\d+): 100%\|[^|]*\| (\d+)/(\d+) \[(\d+):(\d+)(?::(\d+))?<")


def _elapsed_seconds(hh_or_mm: str, mm_or_ss: str, ss: str | None) -> float:
    if ss is None:
        return int(hh_or_mm) * 60 + int(mm_or_ss)
    return int(hh_or_mm) * 3600 + int(mm_or_ss) * 60 + int(ss)


def seed_times_from_logs() -> dict[tuple[str, str], float]:
    """Map (session timestamp prefix, eval seed) -> seconds for the full pass.

    Logs do not name the session, so the match is by the budget/eval log that
    contains the session's output directory, printed by rpg_eval_seeds.py.
    """
    times: dict[tuple[str, str], float] = {}
    for log in sorted(LOG_ROOT.glob("*.log")):
        text = log.read_text(errors="replace").replace("\r", "\n")
        current_session = None
        pending: list[tuple[str, float]] = []
        for line in text.splitlines():
            match = SEED_LINE.search(line)
            if match:
                seed, done, total = match.group(1), match.group(2), match.group(3)
                if done == total:
                    pending.append((seed, _elapsed_seconds(match.group(4), match.group(5), match.group(6))))
                continue
            if "/eval_seeds/new_datasets/" in line and line.strip().endswith(tuple("0123456789Z")):
                current_session = Path(line.strip()).name
                for seed, seconds in pending:
                    times[(current_session, seed)] = seconds
                pending = []
    return times


def collect() -> list[dict]:
    rows: list[dict] = []
    times = seed_times_from_logs()
    for run_dir in sorted(RESULT_ROOT.glob("books*")):
        summaries = list(run_dir.glob("graph/*/summary.json")) + list(run_dir.glob("exhaustive/*/summary.json")) + list(run_dir.glob("budget/*/*/summary.json"))
        for summary_path in sorted(summaries):
            d = json.loads(summary_path.read_text())
            session = summary_path.parent.name
            budget = summary_path.parent.parent.name if summary_path.parent.parent.name.startswith("b") else ""
            metrics = {r["metric"]: r for r in d["metric_summary"]}
            seeds = [str(s) for s in d["eval_seeds"]]
            seed_seconds = [times[(session, s)] for s in seeds if (session, s) in times]
            row = {
                "run": run_dir.name,
                "decoding": d.get("decoding", "graph"),
                "graph_backend": d.get("graph_backend") or "",
                "budget": budget,
                "num_beams": d.get("num_beams", ""),
                "session": session,
                "n_eval_seeds": len(seeds),
                "n_users": metrics["ndcg@10"]["n_users"],
                "visited_items_per_user": round(d["per_seed_summary"][0]["n_visited_items"]),
            }
            for m in METRICS:
                row[m] = round(metrics[m]["final_user_avg"], 5)
                row[m + "_ci_low"] = round(metrics[m]["user_bootstrap_ci_low"], 5)
                row[m + "_ci_high"] = round(metrics[m]["user_bootstrap_ci_high"], 5)
            row["seconds_per_user"] = round(sum(seed_seconds) / len(seed_seconds) / row["n_users"], 6) if seed_seconds else ""
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="results/books_scaling.csv")
    args = parser.parse_args()
    rows = collect()
    out = REPO_ROOT / args.csv
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit("no Books summaries found")
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}\n")
    print(f"{'run':12s} {'decoding':10s} {'budget':14s} {'backend':9s} {'seeds':>5s} {'visited':>8s} {'ndcg@10':>8s} {'recall@10':>9s} {'ms/user':>8s}")
    for r in rows:
        ms = f"{1000*r['seconds_per_user']:.2f}" if r["seconds_per_user"] != "" else "-"
        print(f"{r['run']:12s} {r['decoding']:10s} {r['budget'] or '-':14s} {r['graph_backend'] or '-':9s} {r['n_eval_seeds']:5d} {r['visited_items_per_user']:8d} {r['ndcg@10']:8.4f} {r['recall@10']:9.4f} {ms:>8s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
