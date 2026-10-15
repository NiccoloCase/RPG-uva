#!/usr/bin/env python3
"""Per-query latency of RPG graph decoding versus exhaustive scoring.

Answers a narrower question than the throughput numbers in rpg_eval_seeds.py:
how long does one recommendation request take, on GPU and on CPU, at small
batch sizes. Both decoders run on the same users, the same checkpoint and the
same model forward pass; only the step from token scores to the top-k items
differs.

  graph       upstream ``RPG.generate`` with graph-constrained decoding. The
              graph is built once with the exact row-blocked builder
              (perf.graph.build_chunked_exact_adjacency) and is not timed.
  exhaustive  every item scored from the token log-probabilities, catalogue
              processed in chunks, exact top-k (perf.bruteforce).

For every (device, batch size, decoding, budget) the script times each batch
end to end, including the model forward pass, after a few warm-up batches, and
reports milliseconds per user plus median and 95th-percentile batch latency.
Recall@10 and NDCG@10 on the timed users are reported as a sanity check.

Usage:
  python scripts/rpg_latency.py \\
      --checkpoint artifacts/rpg/ckpt/<run>.pth \\
      --config configs/rpg/books.yaml \\
      --devices cuda,cpu --batch-sizes 1,16,128 --max-users 2000 \\
      --budgets 20:500:5,200:2000:5 --n_codebook 32
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
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perf.config import build_repo_config_files, ensure_submodule_available, parse_override_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to a trained RPG checkpoint.")
    parser.add_argument("--config", action="append", default=[], help="Additional YAML config file. May be provided multiple times.")
    parser.add_argument("--no-root-config", action="store_true", help="Skip configs/rpg/root.yaml.")
    parser.add_argument("--no-local-config", action="store_true", help="Skip configs/rpg/local.yaml even if it exists.")
    parser.add_argument("--devices", default="cuda,cpu", help="Comma-separated devices to time on.")
    parser.add_argument("--batch-sizes", default="1,16,128", help="Comma-separated batch sizes.")
    parser.add_argument("--max-users", type=int, default=2000, help="Number of test users timed per setting (fixed random subset).")
    parser.add_argument("--budgets", default=None, help="Comma-separated graph budgets 'beams:edges:steps'. Default: the config's values.")
    parser.add_argument("--warmup-batches", type=int, default=3, help="Untimed batches before each measurement.")
    parser.add_argument("--cpu-threads", type=int, default=16, help="torch threads for CPU timing.")
    parser.add_argument("--item-chunk-size", type=int, default=50_000, help="Catalogue chunk size for exhaustive scoring.")
    parser.add_argument("--seed", type=int, default=2024, help="Seed for the user subset and the graph's random start.")
    parser.add_argument("--output-dir", default="artifacts/rpg/latency", help="Session root for the result files.")
    return parser


def _parse_budgets(raw: str | None, config: dict[str, Any]) -> list[tuple[int, int, int]]:
    if raw is None:
        return [(int(config["num_beams"]), int(config["n_edges"]), int(config["propagation_steps"]))]
    budgets = []
    for token in raw.split(","):
        beams, edges, steps = (int(value) for value in token.split(":"))
        budgets.append((beams, edges, steps))
    return budgets


def _move_model(model: Any, device: torch.device, adjacency: torch.Tensor) -> None:
    """Move the model and its plain-attribute tensors to ``device``."""
    model.to(device)
    model.item_id2tokens = model.item_id2tokens.to(device)
    model.adjacency = adjacency.to(device)
    model.init_flag = True


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _exhaustive_topk(model: Any, batch: dict[str, torch.Tensor], topk: int, chunk_size: int) -> torch.Tensor:
    from perf.bruteforce import _chunked_topk_from_logits
    from rpg_graph_analysis.dynamic_trace import compute_decoding_context

    context = compute_decoding_context(model, batch)
    item_ids, _ = _chunked_topk_from_logits(
        token_logits=context.token_logits,
        item_id2tokens=model.item_id2tokens,
        n_items=int(model.dataset.n_items),
        batch_size=int(batch["labels"].shape[0]),
        topk=topk,
        item_chunk_size=chunk_size,
        device=context.token_logits.device,
    )
    return item_ids.unsqueeze(-1)


def _time_setting(
    *,
    model: Any,
    evaluator: Any,
    dataset: Any,
    collate_fn: Any,
    device: torch.device,
    batch_size: int,
    decoding: str,
    budget: tuple[int, int, int],
    adjacency: torch.Tensor,
    warmup_batches: int,
    chunk_size: int,
    seed: int,
) -> dict[str, Any]:
    from genrec.utils import init_seed
    from torch.utils.data import DataLoader

    beams, edges, steps = budget
    model.num_beams, model.n_edges, model.propagation_steps = beams, edges, steps
    model.adjacency = adjacency[:, :edges].to(device)
    model.generate_w_decoding_graph = decoding == "graph"
    maxk = evaluator.maxk

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    init_seed(seed, True)
    batch_seconds: list[float] = []
    n_users = 0
    metric_sums = {"recall@10": 0.0, "ndcg@10": 0.0}
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            _synchronize(device)
            start = time.perf_counter()
            if decoding == "graph":
                preds = model.generate(batch, n_return_sequences=maxk)
            else:
                preds = _exhaustive_topk(model, batch, maxk, chunk_size)
            _synchronize(device)
            elapsed = time.perf_counter() - start
            if batch_index < warmup_batches:
                continue
            batch_seconds.append(elapsed)
            results = evaluator.calculate_metrics(preds, batch["labels"])
            size = int(batch["labels"].shape[0])
            n_users += size
            for metric in metric_sums:
                metric_sums[metric] += float(results[metric].sum())

    total = sum(batch_seconds)
    return {
        "device": device.type,
        "batch_size": batch_size,
        "decoding": decoding,
        "num_beams": beams if decoding == "graph" else "",
        "n_edges": edges if decoding == "graph" else "",
        "propagation_steps": steps if decoding == "graph" else "",
        "n_users_timed": n_users,
        "total_seconds": round(total, 4),
        "ms_per_user": round(1000 * total / max(n_users, 1), 4),
        "batch_ms_p50": round(1000 * median(batch_seconds), 4) if batch_seconds else "",
        "batch_ms_p95": round(1000 * float(np.percentile(batch_seconds, 95)), 4) if batch_seconds else "",
        "recall@10": round(metric_sums["recall@10"] / max(n_users, 1), 5),
        "ndcg@10": round(metric_sums["ndcg@10"] / max(n_users, 1), 5),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, override_tokens = parser.parse_known_args(argv)
    ensure_submodule_available()
    from perf.graph import build_chunked_exact_adjacency
    from perf.harness import EvaluationHarness

    harness = EvaluationHarness.build(
        checkpoint_path=args.checkpoint,
        config_files=build_repo_config_files(
            extra_configs=args.config,
            include_root_config=not args.no_root_config,
            include_local_config=not args.no_local_config,
        ),
        config_overrides=parse_override_args(override_tokens),
    )
    model = harness.model
    model.eval()
    budgets = _parse_budgets(args.budgets, harness.config)
    devices = [torch.device(name) for name in args.devices.split(",")]
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]

    test_dataset = harness.test_dataloader.dataset
    rng = np.random.default_rng(args.seed)
    subset = np.sort(rng.choice(len(test_dataset), min(args.max_users, len(test_dataset)), replace=False))
    timed_dataset = test_dataset.select(subset.tolist())
    collate_fn = harness.test_dataloader.collate_fn

    build_device = devices[0] if devices[0].type == "cuda" else torch.device("cpu")
    model.to(build_device)
    model.item_id2tokens = model.item_id2tokens.to(build_device)
    max_edges = max(edges for _, edges, _ in budgets)
    start = time.perf_counter()
    adjacency = build_chunked_exact_adjacency(model, topk=max_edges)
    graph_build_seconds = time.perf_counter() - start

    torch.set_num_threads(args.cpu_threads)
    rows: list[dict[str, Any]] = []
    for device in devices:
        _move_model(model, device, adjacency)
        for batch_size in batch_sizes:
            settings = [("exhaustive", budgets[0])] + [("graph", budget) for budget in budgets]
            for decoding, budget in settings:
                row = _time_setting(
                    model=model,
                    evaluator=harness.trainer.evaluator,
                    dataset=timed_dataset,
                    collate_fn=collate_fn,
                    device=device,
                    batch_size=batch_size,
                    decoding=decoding,
                    budget=budget,
                    adjacency=adjacency,
                    warmup_batches=args.warmup_batches,
                    chunk_size=args.item_chunk_size,
                    seed=args.seed,
                )
                rows.append(row)
                print(json.dumps(row), flush=True)

    root = Path(args.output_dir)
    if not root.is_absolute():
        root = REPO_ROOT / root
    session = root / f"{harness.config.get('category', 'dataset')}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    session.mkdir(parents=True, exist_ok=True)
    with (session / "latency.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (session / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint_path": str(args.checkpoint),
                "category": harness.config.get("category"),
                "n_items": int(harness.dataset.n_items) - 1,
                "n_codebook": int(harness.config["n_codebook"]),
                "graph_build_seconds": round(graph_build_seconds, 2),
                "graph_build_device": build_device.type,
                "cpu_threads": args.cpu_threads,
                "max_users": args.max_users,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
