#!/usr/bin/env python3
"""Collect the SASRec audit runs into one CSV.

Reads the test metrics that scripts/sasrec.py appends to the end of
artifacts/sasrec/logs/<run_id>.log for every run started by
jobs/sasrec_audit/run_queue.sh (run ids "sasrec_audit_<dataset>_<loss>_h<hidden>_lr<lr>"),
and the best validation epoch from the per-epoch lines.

Writes results/sasrec_audit.csv: dataset, loss, hidden_size, lr, recall/ndcg at
5, 10, 20, number of epochs run, and prints an NDCG@10 table.

stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = re.compile(r"^sasrec_audit_(?P<dataset>.+?)_(?P<loss>bce|sampled_ce|ce)_h(?P<hidden>\d+)_lr(?P<lr>[0-9p]+)$")
EPOCH_LINE = re.compile(r"\('Epoch', (\d+)\)")


def parse_log(path: Path) -> dict | None:
    match = RUN_ID.match(path.stem)
    if match is None:
        return None
    text = path.read_text(errors="replace")
    start = text.rfind("\n{")
    if start < 0 or path.stem not in text[:start]:
        return None
    try:
        metrics = json.loads(text[start:])
    except json.JSONDecodeError:
        return None
    epochs = [int(value) for value in EPOCH_LINE.findall(text)]
    row = {
        "dataset": match["dataset"],
        "loss": match["loss"],
        "hidden_size": int(match["hidden"]),
        "lr": float(match["lr"].replace("p", ".")),
        "epochs_run": max(epochs) + 1 if epochs else "",
    }
    for key in sorted(metrics):
        row[key] = round(float(metrics[key]), 5)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-dir", default="artifacts/sasrec/logs")
    parser.add_argument("--csv", default="results/sasrec_audit.csv")
    args = parser.parse_args()

    rows = [row for path in sorted((REPO_ROOT / args.log_dir).glob("sasrec_audit_*.log")) if (row := parse_log(path))]
    if not rows:
        raise SystemExit("no finished SASRec audit runs found")
    out = REPO_ROOT / args.csv
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}\n")
    print(f"{'dataset':22s} {'loss':5s} {'hidden':>6s} {'ndcg@10':>8s} {'recall@10':>9s} {'epochs':>6s}")
    for row in sorted(rows, key=lambda r: (r["dataset"], r["hidden_size"], r["loss"])):
        print(f"{row['dataset']:22s} {row['loss']:5s} {row['hidden_size']:6d} {row.get('ndcg@10', float('nan')):8.4f} {row.get('recall@10', float('nan')):9.4f} {row['epochs_run']!s:>6s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
