"""Graph preparation command.

``prepare-graph`` reconstructs the RPG model from a checkpoint, builds one exact
item-item graph with ``graph_topk`` neighbors per item, and saves it in a
graph-analysis session. The analysis commands then slice this saved graph to
smaller decoding widths ``n_edges`` without rebuilding it.
"""

from __future__ import annotations

import json
import time
from typing import Any

import torch

from perf.config import checkpoint_signature
from perf.graph import build_sparse_adjacency
from perf.harness import EvaluationHarness

from .runtime import build_harness_from_args, topk_from_config
from .sessions import (
    SessionPaths,
    adjacency_path,
    append_or_update_manifest,
    create_session,
    graph_metadata_path,
    write_json,
)


def prepare_graph(args: Any) -> int:
    """Build and save a fresh exact graph for the decoding analyses.

    Args:
        args: Parsed CLI namespace. Must include checkpoint/config/session fields.

    Returns:
        Process exit code. The session root is printed to stdout for shell use.
    """

    harness = build_harness_from_args(args)
    topk = topk_from_config(harness.config)
    backend = str(harness.config.get("graph_backend", "exact_gpu")).lower()
    if backend not in ("exact_gpu", "dense"):
        raise ValueError("Graph analysis requires an exact graph: graph_backend exact_gpu or dense.")

    paths = create_session(harness.config, args.session_dir)

    start = time.perf_counter()
    adjacency = build_sparse_adjacency(
        model=harness.model,
        backend=backend,
        topk=topk,
        config=harness.config,
    )
    build_seconds = time.perf_counter() - start

    adjacency_file = adjacency_path(paths, topk)
    torch.save(adjacency.cpu(), adjacency_file)

    metadata = {
        "command": "prepare-graph",
        "dataset": harness.config["dataset"],
        "category": harness.config.get("category"),
        "model": harness.config["model"],
        "checkpoint_path": str(harness.checkpoint_path),
        "checkpoint_signature": checkpoint_signature(harness.checkpoint_path),
        "backend": backend,
        "topk": topk,
        "n_items": int(harness.dataset.n_items),
        "n_real_items": int(harness.dataset.n_items - 1),
        "n_digit": int(harness.tokenizer.n_digit),
        "codebook_size": int(harness.tokenizer.codebook_size),
        "build_seconds": build_seconds,
        "adjacency_path": str(adjacency_file),
    }
    write_json(graph_metadata_path(paths), metadata)
    append_or_update_manifest(
        paths,
        {
            "session_root": str(paths.root),
            "graph_metadata": str(graph_metadata_path(paths)),
            "adjacency": str(adjacency_file),
        },
    )

    print(paths.root)
    return 0


def load_prepared_graph(
    paths: SessionPaths,
    harness: EvaluationHarness,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load and validate a graph prepared for the same checkpoint/category.

    The check uses the cheap checkpoint signature from ``perf.config``. This is
    enough to prevent accidentally analyzing a graph generated from a different
    checkpoint or dataset category.
    """

    metadata_path = graph_metadata_path(paths)
    metadata = json.loads(metadata_path.read_text())

    expected_signature = checkpoint_signature(harness.checkpoint_path)
    if metadata.get("checkpoint_signature") != expected_signature:
        raise ValueError(
            "Prepared graph checkpoint signature mismatch: "
            f"expected {expected_signature}, found {metadata.get('checkpoint_signature')}"
        )
    if metadata.get("category") != harness.config.get("category"):
        raise ValueError(
            f"Prepared graph category mismatch: expected {harness.config.get('category')}, "
            f"found {metadata.get('category')}"
        )

    topk = int(metadata["topk"])
    adjacency = torch.load(adjacency_path(paths, topk), map_location="cpu")
    expected_shape = (harness.dataset.n_items, topk)
    if tuple(adjacency.shape) != expected_shape:
        raise ValueError(f"Expected adjacency shape {expected_shape}, found {tuple(adjacency.shape)}")
    return adjacency, metadata
