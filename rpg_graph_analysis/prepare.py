"""Graph preparation command.

``prepare-graph`` reconstructs the RPG model from a checkpoint, builds one exact
flat item-item graph with ``graph_topk`` neighbors per item, and saves it in a
graph-analysis session. Static experiments then slice this saved graph to
smaller effective ``k`` values without rebuilding it.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from perf.config import checkpoint_signature
from perf.graph import build_sparse_adjacency

from .runtime import build_harness_from_args, topk_from_config
from .sessions import (
    adjacency_path,
    append_or_update_manifest,
    create_session,
    graph_metadata_path,
    write_json,
)


def prepare_graph(args: Any) -> int:
    """Build and save a fresh exact flat graph for later static analysis.

    Args:
        args: Parsed CLI namespace. Must include checkpoint/config/session fields.

    Returns:
        Process exit code. The session root is printed to stdout for shell use.
    """

    harness = build_harness_from_args(args)
    topk = topk_from_config(harness.config)
    backend = str(harness.config.get("graph_backend", "flat")).lower()
    if backend != "flat":
        raise ValueError("Graph analysis v1 requires graph_backend: flat.")

    paths = create_session(harness.config, args.session_dir)

    start = time.perf_counter()
    adjacency = build_sparse_adjacency(
        model=harness.model,
        backend="flat",
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
        "backend": "flat",
        "topk": topk,
        "n_items": int(harness.dataset.n_items),
        "n_real_items": int(harness.dataset.n_items - 1),
        "n_digit": int(harness.tokenizer.n_digit),
        "codebook_size": int(harness.tokenizer.codebook_size),
        "graph_vector_batch_size": int(harness.config.get("graph_vector_batch_size", 1024)),
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

