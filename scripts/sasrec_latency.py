#!/usr/bin/env python3
"""Per-query latency of SASRec full-sort inference.

The item-ID counterpart of scripts/rpg_latency.py: same users per dataset
(a fixed random subset of the test split), same batch sizes, same devices,
same timing rules. Each batch is timed end to end, from the model forward
pass to the top-k item ids on the device, after a few warm-up batches.
Previously seen items are not removed from the ranking, matching RPG's
evaluation protocol.

Usage:
  python scripts/sasrec_latency.py --preset books \\
      --checkpoint artifacts/sasrec/ckpt/<run_id>.pt --hidden_size 64 \\
      --devices cuda,cpu --batch-sizes 1,16,128 --max-users 1000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import median
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import sasrec as sasrec_cli  # noqa: E402
from models.sasrec import SASRecDataset, SASRecModel  # noqa: E402
from models.sasrec.utils import get_user_seqs  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", required=True, choices=sorted(sasrec_cli.PRESET_CONFIGS), help="SASRec dataset preset.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained SASRec checkpoint.")
    parser.add_argument("--config", action="append", default=[], help="Additional YAML config file.")
    parser.add_argument("--devices", default="cuda,cpu", help="Comma-separated devices to time on.")
    parser.add_argument("--batch-sizes", default="1,16,128", help="Comma-separated batch sizes.")
    parser.add_argument("--max-users", type=int, default=1000, help="Number of test users timed per setting (fixed random subset).")
    parser.add_argument("--warmup-batches", type=int, default=2, help="Untimed batches before each measurement.")
    parser.add_argument("--cpu-threads", type=int, default=16, help="torch threads for CPU timing.")
    parser.add_argument("--seed", type=int, default=2024, help="Seed for the user subset.")
    parser.add_argument("--output-dir", default="artifacts/sasrec/latency", help="Session root for the result files.")
    return parser


def build_args(cli: argparse.Namespace, override_tokens: list[str]) -> SimpleNamespace:
    namespace = SimpleNamespace(no_root_config=False, no_local_config=False, preset=cli.preset, config=cli.config)
    config = sasrec_cli.load_config(sasrec_cli.build_config_files(namespace), sasrec_cli.parse_override_args(override_tokens))
    return sasrec_cli.normalize_config(config, cli.checkpoint)


def time_setting(model, dataset, device: torch.device, batch_size: int, warmup: int, topk: int) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    item_weights = model.item_embeddings.weight
    batch_seconds: list[float] = []
    n_users = hits = 0
    ndcg = 0.0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            _, input_ids, _, _, answers = (t.to(device) for t in batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            hidden = model(input_ids)[:, -1, :]
            scores = hidden @ item_weights.T
            scores[:, 0] = float("-inf")
            scores[:, -1] = float("-inf")  # unused mask id
            top = scores.topk(topk, dim=-1).indices
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            if batch_index < warmup:
                continue
            batch_seconds.append(elapsed)
            match = top == answers[:, :1]
            found = match.any(-1)
            rank = match.float().argmax(-1)
            hits += int(found.sum())
            ndcg += float((found.float() / torch.log2(rank.float() + 2)).sum())
            n_users += int(input_ids.shape[0])
    total = sum(batch_seconds)
    return {
        "device": device.type,
        "batch_size": batch_size,
        "n_users_timed": n_users,
        "total_seconds": round(total, 4),
        "ms_per_user": round(1000 * total / max(n_users, 1), 4),
        "batch_ms_p50": round(1000 * median(batch_seconds), 4) if batch_seconds else "",
        "batch_ms_p95": round(1000 * float(np.percentile(batch_seconds, 95)), 4) if batch_seconds else "",
        "recall@10": round(hits / max(n_users, 1), 5),
        "ndcg@10": round(ndcg / max(n_users, 1), 5),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    cli, override_tokens = parser.parse_known_args(argv)
    args = build_args(cli, override_tokens)

    user_seq, max_item, _, _ = get_user_seqs(args.data_file)
    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    test_dataset = SASRecDataset(args, user_seq, data_type="test")
    subset = np.sort(np.random.default_rng(cli.seed).choice(len(test_dataset), min(cli.max_users, len(test_dataset)), replace=False))
    timed_dataset = Subset(test_dataset, subset.tolist())

    model = SASRecModel(args)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location="cpu"))
    model.eval()
    torch.set_num_threads(cli.cpu_threads)

    rows = []
    for device_name in cli.devices.split(","):
        device = torch.device(device_name)
        model.to(device)
        for batch_size in (int(value) for value in cli.batch_sizes.split(",")):
            row = time_setting(model, timed_dataset, device, batch_size, cli.warmup_batches, topk=10)
            rows.append(row)
            print(json.dumps(row), flush=True)

    root = Path(cli.output_dir)
    if not root.is_absolute():
        root = REPO_ROOT / root
    session = root / f"{args.data_name}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    session.mkdir(parents=True, exist_ok=True)
    with (session / "latency.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (session / "summary.json").write_text(json.dumps({
        "checkpoint_path": str(args.checkpoint_path),
        "dataset": args.data_name,
        "n_items": max_item,
        "hidden_size": args.hidden_size,
        "loss_type": args.loss_type,
        "cpu_threads": cli.cpu_threads,
        "max_users": cli.max_users,
        "rows": rows,
    }, indent=2))
    print(session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
