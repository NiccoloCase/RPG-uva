#!/usr/bin/env python3
"""Time the construction of RPG's decoding graph.

Builds the top-``n_edges`` item-item adjacency used by graph-constrained
decoding and reports wall-clock time and peak GPU memory for each builder:

  dense      upstream ``RPG.build_ii_sim_mat`` + ``build_adjacency_list``:
             the full n x n similarity matrix on the GPU. Skipped when that
             matrix would not fit in GPU memory; its size is still reported.
  exact_gpu  perf.graph.build_chunked_exact_adjacency: the same similarity
             and neighbours, computed in row blocks so the full matrix is
             never stored.

The model is loaded without tokenizing user sequences, which the graph does
not need, so the timing covers graph construction only.

Usage:
  python scripts/rpg_graph_build_timing.py \\
      --checkpoint artifacts/rpg/ckpt/<run>.pth \\
      --config configs/rpg/new_datasets/books.yaml --repeats 3 [--n_codebook 32]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_ROOT = REPO_ROOT / "third_party"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perf.config import build_repo_config_files, ensure_submodule_available, parse_override_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to a trained RPG checkpoint.")
    parser.add_argument("--config", action="append", default=[], help="Additional YAML config file. May be provided multiple times.")
    parser.add_argument("--no-root-config", action="store_true", help="Skip configs/rpg/root.yaml.")
    parser.add_argument("--no-local-config", action="store_true", help="Skip configs/rpg/local.yaml even if it exists.")
    parser.add_argument("--builders", default="dense,exact_gpu", help="Comma-separated builders to time.")
    parser.add_argument("--repeats", type=int, default=3, help="Timed builds per builder.")
    parser.add_argument("--output-dir", default="artifacts/rpg/graph_build", help="Directory for the result JSON.")
    return parser


def load_model(checkpoint: str, config_files: list[str], overrides: dict):
    """Load config, dataset, tokenizer and RPG weights, without tokenizing sequences."""
    if str(THIRD_PARTY_ROOT) not in sys.path:
        sys.path.insert(0, str(THIRD_PARTY_ROOT))
    from accelerate import Accelerator
    from genrec.utils import get_config, get_dataset, get_model, get_tokenizer, init_logger, init_seed

    config = get_config(model_name="RPG", dataset_name="AmazonReviews2014", config_file=config_files or None, config_dict=overrides or None)
    accelerator = Accelerator()
    config["accelerator"] = accelerator
    config["device"] = accelerator.device
    config["use_ddp"] = False
    init_seed(config["rand_seed"], config["reproducibility"])
    init_logger(config)

    dataset = get_dataset(config["dataset"])(config)
    tokenizer = get_tokenizer(config["model"])(config, dataset)
    model = get_model(config["model"])(config, dataset, tokenizer)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.to(config["device"])
    model.item_id2tokens = model.item_id2tokens.to(config["device"])
    model.eval()
    return config, model


def time_builder(model, builder: str, topk: int) -> tuple[float, float]:
    """Return (seconds, peak GPU memory in GiB above the starting allocation)."""
    from perf.graph import build_chunked_exact_adjacency

    device = model.gpt2.device
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    start = time.perf_counter()
    with torch.no_grad():
        if builder == "dense":
            similarity = model.build_ii_sim_mat()
            adjacency = model.build_adjacency_list(similarity)
            del similarity
        else:
            adjacency = build_chunked_exact_adjacency(model, topk=topk)
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    peak_gib = (torch.cuda.max_memory_allocated(device) - baseline) / 1024**3
    del adjacency
    torch.cuda.empty_cache()
    return seconds, peak_gib


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, override_tokens = parser.parse_known_args(argv)
    ensure_submodule_available()
    if not torch.cuda.is_available():
        raise SystemExit("A GPU is required.")

    config_files = build_repo_config_files(
        extra_configs=args.config,
        include_root_config=not args.no_root_config,
        include_local_config=not args.no_local_config,
    )
    config, model = load_model(args.checkpoint, config_files, parse_override_args(override_tokens))
    n_items = int(model.dataset.n_items)
    topk = int(config["n_edges"])
    dense_bytes = n_items * n_items * 4
    free_bytes, _ = torch.cuda.mem_get_info(model.gpt2.device)

    result = {
        "checkpoint": str(args.checkpoint),
        "category": config.get("category"),
        "n_items": n_items - 1,
        "n_codebook": int(config["n_codebook"]),
        "n_edges": topk,
        "gpu": torch.cuda.get_device_name(model.gpt2.device),
        "dense_matrix_gib": round(dense_bytes / 1024**3, 2),
        "builders": {},
    }
    for builder in args.builders.split(","):
        if builder == "dense" and dense_bytes > 0.8 * free_bytes:
            result["builders"][builder] = {"skipped": f"needs {dense_bytes / 1024**3:.1f} GiB, {free_bytes / 1024**3:.1f} GiB free"}
            print(f"{builder}: skipped ({result['builders'][builder]['skipped']})", flush=True)
            continue
        runs = [time_builder(model, builder, topk) for _ in range(args.repeats)]
        result["builders"][builder] = {
            "seconds": [round(seconds, 2) for seconds, _ in runs],
            "median_seconds": round(sorted(seconds for seconds, _ in runs)[len(runs) // 2], 2),
            "peak_gpu_gib": round(max(peak for _, peak in runs), 3),
        }
        print(f"{builder}: {result['builders'][builder]}", flush=True)

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{config.get('category')}_m{config['n_codebook']}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
