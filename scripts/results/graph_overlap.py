#!/usr/bin/env python3
"""Neighbour overlap between the exact decoding graph and the HNSW graph.

For one RPG checkpoint, builds both graphs at the released width k and reports
the mean share of an item's k neighbours that the two graphs have in common,
and the share of items whose exact neighbour list contains the item itself.

Usage:
  python scripts/results/graph_overlap.py --name sports --checkpoint <ckpt> \\
      --config configs/rpg/repro/sports_and_outdoors.yaml
Writes artifacts/rpg/graph_overlap/<name>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from decoding.config import build_repo_config_files  # noqa: E402
from decoding.graph import build_sparse_adjacency, compare_adjacency_overlap  # noqa: E402
from decoding.harness import EvaluationHarness  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", action="append", default=[])
    args = parser.parse_args()
    h = EvaluationHarness.build(
        checkpoint_path=args.checkpoint,
        config_files=build_repo_config_files(extra_configs=args.config, include_root_config=True, include_local_config=False),
        config_overrides={},
    )
    k = int(h.config["n_edges"])
    exact = build_sparse_adjacency(h.model, "exact_gpu", k, h.config)
    hnsw = build_sparse_adjacency(h.model, "hnsw", k, h.config)
    overlap = compare_adjacency_overlap(exact, hnsw)
    ids = torch.arange(1, exact.shape[0])
    result = {
        "name": args.name, "n_edges": k, "n_items": int(exact.shape[0] - 1),
        "mean_neighbour_overlap": overlap["mean_overlap_rate"],
        "exact_rows_with_self": float((exact[1:] == ids[:, None]).any(-1).float().mean()),
        "hnsw_rows_with_self": float((hnsw[1:] == ids[:, None]).any(-1).float().mean()),
    }
    out = REPO_ROOT / "artifacts/rpg/graph_overlap"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
