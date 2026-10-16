#!/usr/bin/env python3
"""Per-query latency of the MemGen-GR baselines (SASRec with softmax loss, TIGER).

The baseline counterpart of scripts/rpg/latency.py, with the same protocol:
a fixed random subset of test users, the same batch sizes and devices, each
batch timed end to end (model forward pass to top-k item ids) after a few
warm-up batches. Recall@10 and NDCG@10 on the timed users are reported as a
sanity check. For TIGER the timed step is its beam search over semantic IDs.

Usage:
  python scripts/memgen/latency.py --model SASRec --preset sports \\
      --checkpoint artifacts/memgen/ckpt/<run>.pth \\
      --devices cuda,cpu --batch-sizes 1,16,128 --max-users 2000
Writes artifacts/memgen/latency/<model>_<category>_<timestamp>/{latency.csv,summary.json}.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import median

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train import DEFAULT_MEMGEN_ROOT, MODELS, PRESETS, parse_overrides  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=MODELS, help="MemGen-GR model name.")
    parser.add_argument("--preset", required=True, choices=PRESETS, help="Dataset preset under configs/memgen/<model>/.")
    parser.add_argument("--checkpoint", required=True, help="Trained MemGen-GR checkpoint (.pth).")
    parser.add_argument("--memgen-root", default=os.environ.get("MEMGEN_ROOT", str(DEFAULT_MEMGEN_ROOT)), help="MemGen-GR checkout.")
    parser.add_argument("--artifacts-dir", default=str(REPO_ROOT / "artifacts"), help="Root for the shared data cache and the outputs.")
    parser.add_argument("--devices", default="cuda,cpu", help="Comma-separated devices to time on.")
    parser.add_argument("--batch-sizes", default="1,16,128", help="Comma-separated batch sizes.")
    parser.add_argument("--max-users", type=int, default=2000, help="Number of test users timed per setting (fixed random subset).")
    parser.add_argument("--warmup-batches", type=int, default=3, help="Untimed batches before each measurement.")
    parser.add_argument("--cpu-threads", type=int, default=16, help="torch threads for CPU timing.")
    parser.add_argument("--seed", type=int, default=2024, help="Seed for the user subset.")
    parser.add_argument("--output-dir", default="artifacts/memgen/latency", help="Session root for the result files.")
    return parser


def time_setting(pipeline, dataset, device: torch.device, batch_size: int, warmup: int) -> dict:
    from torch.utils.data import DataLoader

    model = pipeline.model.to(device)
    model.eval()
    pipeline.config["device"] = device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pipeline.tokenizer.collate_fn["test"])
    batch_seconds: list[float] = []
    sums = {"recall@10": 0.0, "ndcg@10": 0.0}
    n_users = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            preds = model.generate(batch, n_return_sequences=10)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            if batch_index < warmup:
                continue
            batch_seconds.append(elapsed)
            if preds.dim() == 2:
                preds = preds.view(-1, 10, preds.size(-1))
            results = pipeline.trainer.evaluator.calculate_metrics(preds, batch["labels"])
            for key in sums:
                sums[key] += float(results[key].sum())
            n_users += int(batch["labels"].shape[0])
    total = sum(batch_seconds)
    return {
        "device": device.type,
        "batch_size": batch_size,
        "n_users_timed": n_users,
        "total_seconds": round(total, 4),
        "ms_per_user": round(1000 * total / max(n_users, 1), 4),
        "batch_ms_p50": round(1000 * median(batch_seconds), 4) if batch_seconds else "",
        "batch_ms_p95": round(1000 * float(np.percentile(batch_seconds, 95)), 4) if batch_seconds else "",
        "recall@10": round(sums["recall@10"] / max(n_users, 1), 5),
        "ndcg@10": round(sums["ndcg@10"] / max(n_users, 1), 5),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, override_tokens = parser.parse_known_args(argv)
    memgen_root = Path(args.memgen_root).resolve()
    if not (memgen_root / "genrec" / "default.yaml").is_file():
        raise SystemExit(f"MemGen-GR not found at {memgen_root}. Run `git submodule update --init` or set --memgen-root.")
    preset_file = REPO_ROOT / "configs" / "memgen" / args.model.lower() / f"{args.preset}.yaml"
    checkpoint = str(Path(args.checkpoint).resolve())  # before changing into the MemGen-GR folder
    artifacts = Path(args.artifacts_dir).resolve()
    config = {
        "cache_dir": str(artifacts / "rpg" / "cache"),
        "ckpt_dir": str(artifacts / "memgen" / "ckpt"),
        "log_dir": str(artifacts / "memgen" / "logs"),
        "tensorboard_log_dir": str(artifacts / "memgen" / "tensorboard"),
        "logging": False,
        "eval_fine_grained": False,
        "epochs": 0,
        "run_id": f"memgen_latency_{args.model.lower()}_{args.preset}",
    }
    config.update(parse_overrides(override_tokens))

    os.environ.setdefault("WANDB_MODE", "disabled")
    sys.path.insert(0, str(memgen_root))
    os.chdir(memgen_root)
    from genrec.utils import get_pipeline

    pipeline = get_pipeline(args.model)(
        model_name=args.model,
        dataset_name="AmazonReviews2014",
        checkpoint_path=checkpoint,
        config_file=str(preset_file),
        config_dict=config,
    )
    test_dataset = pipeline.tokenized_datasets["test"]
    subset = np.sort(np.random.default_rng(args.seed).choice(len(test_dataset), min(args.max_users, len(test_dataset)), replace=False))
    timed_dataset = test_dataset.select(subset.tolist())
    torch.set_num_threads(args.cpu_threads)

    rows = []
    for device_name in args.devices.split(","):
        device = torch.device(device_name)
        for batch_size in (int(v) for v in args.batch_sizes.split(",")):
            row = time_setting(pipeline, timed_dataset, device, batch_size, args.warmup_batches)
            rows.append(row)
            print(json.dumps(row), flush=True)

    root = Path(args.output_dir)
    if not root.is_absolute():
        root = REPO_ROOT / root
    session = root / f"{args.model}_{pipeline.config.get('category', 'dataset')}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    session.mkdir(parents=True, exist_ok=True)
    with (session / "latency.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (session / "summary.json").write_text(json.dumps({
        "model": args.model,
        "checkpoint_path": checkpoint,
        "category": pipeline.config.get("category"),
        "n_items": int(pipeline.raw_dataset.n_items) - 1,
        "cpu_threads": args.cpu_threads,
        "max_users": args.max_users,
        "rows": rows,
    }, indent=2))
    print(session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
