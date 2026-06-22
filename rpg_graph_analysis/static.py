"""Static graph-analysis command.

This module implements Experiment Block A. It assumes a prepared top-``K`` RPG
item graph already exists in the session and computes static graph summaries
for smaller effective ``k`` slices.

The static command never runs RPG recommendation decoding. Inference parameters
such as ``num_beams`` and ``propagation_steps`` therefore do not affect these
metrics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from perf.config import checkpoint_signature
from perf.graph import _build_token_embedding_table
from perf.harness import EvaluationHarness

from .edge_metrics import (
    compute_hamming,
    compute_shifted_similarity,
    edge_arrays,
    histogram_rows,
    integer_histogram_rows,
    random_pairs,
    summary,
)
from .popularity import popularity_rows, train_frequencies
from .structural_metrics import (
    clustering_summary,
    component_summary,
    gini,
    indegree_histogram_rows,
    random_indegree_summaries,
    reciprocity,
    undirected_graph,
)
from .runtime import build_harness_from_args, k_values_from_config, random_seeds_from_config
from .sessions import (
    SessionPaths,
    adjacency_path,
    append_or_update_manifest,
    graph_metadata_path,
    latest_session,
    write_csv,
    write_json,
)


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


def static_output_paths(paths: SessionPaths) -> dict[str, Path]:
    """Return all CSV/JSON outputs produced by the static command."""

    return {
        "static_summary_csv": paths.static / "static_summary.csv",
        "static_summary_json": paths.static / "static_summary.json",
        "edge_similarity_summary_csv": paths.static / "edge_similarity_summary.csv",
        "edge_similarity_histogram_csv": paths.static / "edge_similarity_histogram.csv",
        "hamming_summary_csv": paths.static / "hamming_summary.csv",
        "hamming_histogram_csv": paths.static / "hamming_histogram.csv",
        "reciprocity_summary_csv": paths.static / "reciprocity_summary.csv",
        "hubness_summary_csv": paths.static / "hubness_summary.csv",
        "indegree_histogram_csv": paths.static / "indegree_histogram.csv",
        "components_summary_csv": paths.static / "components_summary.csv",
        "clustering_summary_csv": paths.static / "clustering_summary.csv",
        "popularity_summary_csv": paths.static / "popularity_summary.csv",
        "popularity_buckets_csv": paths.static / "popularity_buckets.csv",
        "item_metrics_by_k_csv": paths.static / "item_metrics_by_k.csv",
    }


def write_static_outputs(
    paths: SessionPaths,
    outputs: dict[str, Path],
    checkpoint_path: Path,
    k_values: list[int],
    seeds: list[int],
    tables: dict[str, list[dict[str, Any]]],
) -> None:
    """Persist all static-analysis tables and update the session manifest."""

    write_csv(outputs["static_summary_csv"], tables["static"])
    write_json(
        outputs["static_summary_json"],
        {
            "session_root": str(paths.root),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_signature": checkpoint_signature(checkpoint_path),
            "k_values": k_values,
            "random_seeds": seeds,
            "rows": tables["static"],
        },
    )
    write_csv(outputs["edge_similarity_summary_csv"], tables["similarity"])
    write_csv(outputs["edge_similarity_histogram_csv"], tables["similarity_hist"])
    write_csv(outputs["hamming_summary_csv"], tables["hamming"])
    write_csv(outputs["hamming_histogram_csv"], tables["hamming_hist"])
    write_csv(outputs["reciprocity_summary_csv"], tables["reciprocity"])
    write_csv(outputs["hubness_summary_csv"], tables["hubness"])
    write_csv(outputs["indegree_histogram_csv"], tables["indegree_hist"])
    write_csv(outputs["components_summary_csv"], tables["components"])
    write_csv(outputs["clustering_summary_csv"], tables["clustering"])
    write_csv(outputs["popularity_summary_csv"], tables["popularity"])
    write_csv(outputs["popularity_buckets_csv"], tables["popularity_buckets"])
    write_csv(outputs["item_metrics_by_k_csv"], tables["item_metrics"])

    append_or_update_manifest(
        paths,
        {
            "session_root": str(paths.root),
            "graph_metadata": str(graph_metadata_path(paths)),
            "static_outputs": {key: str(value) for key, value in outputs.items()},
        },
    )


def _empty_tables() -> dict[str, list[dict[str, Any]]]:
    """Create named row buffers for every output table."""

    return {
        "static": [],
        "similarity": [],
        "similarity_hist": [],
        "hamming": [],
        "hamming_hist": [],
        "reciprocity": [],
        "hubness": [],
        "indegree_hist": [],
        "components": [],
        "clustering": [],
        "popularity": [],
        "popularity_buckets": [],
        "item_metrics": [],
    }


def run_static(args: Any) -> int:
    """Compute all static graph-analysis outputs for a prepared graph.

    The command:

    1. Reconstructs the RPG checkpoint and tokenizer.
    2. Loads the prepared top-``K`` adjacency tensor.
    3. Precomputes edge similarity/Hamming metrics for the largest requested
       ``k`` to avoid duplicate work.
    4. Slices those arrays for each effective ``k`` and writes CSV/JSON tables.
    """

    harness = build_harness_from_args(args)
    paths = latest_session(harness.config, args.session_dir)
    adjacency_tensor, metadata = load_prepared_graph(paths, harness)

    topk = int(metadata["topk"])
    k_values = k_values_from_config(harness.config, topk)
    seeds = random_seeds_from_config(harness.config)
    batch_size = int(harness.config.get("graph_analysis_similarity_batch_size", 32768))

    adjacency = adjacency_tensor.numpy()
    item_tokens = harness.model.item_id2tokens.detach().cpu().numpy()
    token_table = _build_token_embedding_table(harness.model)
    n_items = int(harness.dataset.n_items)
    n_real = n_items - 1
    n_digit = int(harness.tokenizer.n_digit)
    codebook_size = int(harness.tokenizer.codebook_size)

    # Compute once at max_k, then select the appropriate raw neighbor ranks for
    # each smaller k. Self-edges were removed, but the raw adjacency rank is
    # preserved in max_ranks.
    max_k = max(k_values)
    max_sources, max_destinations, max_ranks = edge_arrays(adjacency, max_k)
    max_edge_count = int(max_sources.shape[0])

    edge_similarity = compute_shifted_similarity(
        sources=max_sources,
        destinations=max_destinations,
        item_tokens=item_tokens,
        token_table=token_table,
        codebook_size=codebook_size,
        batch_size=batch_size,
    )
    edge_hamming = compute_hamming(max_sources, max_destinations, item_tokens, batch_size)

    random_pairs_by_seed = {
        seed: random_pairs(n_items=n_items, n_pairs=max_edge_count, seed=seed) for seed in seeds
    }
    random_similarity_by_seed = {}
    random_hamming_by_seed = {}
    for seed, (sources, destinations) in random_pairs_by_seed.items():
        random_similarity_by_seed[seed] = compute_shifted_similarity(
            sources=sources,
            destinations=destinations,
            item_tokens=item_tokens,
            token_table=token_table,
            codebook_size=codebook_size,
            batch_size=batch_size,
        )
        random_hamming_by_seed[seed] = compute_hamming(sources, destinations, item_tokens, batch_size)

    train_frequency = train_frequencies(harness.dataset)
    tables = _empty_tables()
    similarity_bins = np.linspace(0.0, 1.0, 51)

    for k in k_values:
        edge_mask = max_ranks < k
        sources = max_sources[edge_mask]
        destinations = max_destinations[edge_mask]
        edge_count = int(sources.shape[0])
        similarity_values = edge_similarity[edge_mask]
        hamming_values = edge_hamming[edge_mask]

        random_similarity = np.concatenate(
            [random_similarity_by_seed[seed][:edge_count] for seed in seeds]
        )
        random_hamming = np.concatenate(
            [random_hamming_by_seed[seed][:edge_count] for seed in seeds]
        )

        sim_summary = {
            "k": k,
            "n_edges": edge_count,
            **summary(similarity_values, "edge_similarity"),
            **summary(random_similarity, "random_similarity"),
        }
        tables["similarity"].append(sim_summary)
        tables["similarity_hist"].extend(
            histogram_rows(similarity_values, similarity_bins, "similarity", k, "graph")
        )
        tables["similarity_hist"].extend(
            histogram_rows(random_similarity, similarity_bins, "similarity", k, "random")
        )

        hamming_summary = {
            "k": k,
            "n_edges": edge_count,
            **summary(hamming_values.astype(np.float64), "hamming"),
            **summary(hamming_values.astype(np.float64) / n_digit, "normalized_hamming"),
            **summary(random_hamming.astype(np.float64), "random_hamming"),
            **summary(random_hamming.astype(np.float64) / n_digit, "random_normalized_hamming"),
        }
        tables["hamming"].append(hamming_summary)
        tables["hamming_hist"].extend(integer_histogram_rows(hamming_values, n_digit, "hamming", k, "graph"))
        tables["hamming_hist"].extend(integer_histogram_rows(random_hamming, n_digit, "hamming", k, "random"))

        reciprocity_summary, node_reciprocity = reciprocity(sources, destinations, n_items)
        reciprocity_summary = {"k": k, "n_edges": edge_count, **reciprocity_summary}
        tables["reciprocity"].append(reciprocity_summary)

        indegree = np.bincount(destinations - 1, minlength=n_real)
        top_count = max(1, int(math.ceil(n_real * 0.01)))
        hub_summary = {
            "k": k,
            "n_edges": edge_count,
            "directed_density": float(edge_count / max(n_real * (n_real - 1), 1)),
            "indegree_mean": float(indegree.mean()),
            "indegree_std": float(indegree.std()),
            "indegree_max": int(indegree.max()),
            "indegree_gini": gini(indegree),
            "top_1pct_node_count": top_count,
            "top_1pct_incoming_edge_share": float(np.sort(indegree)[-top_count:].sum() / max(edge_count, 1)),
            **random_indegree_summaries(n_items, edge_count, seeds, random_pairs_by_seed),
        }
        tables["hubness"].append(hub_summary)
        tables["indegree_hist"].extend(indegree_histogram_rows(indegree, k, "graph"))

        graph = undirected_graph(sources, destinations, n_real)
        component_row, component_sizes_by_node, in_largest_component_by_node = component_summary(graph, k)
        tables["components"].append(component_row)
        clustering_row, clustering_by_node = clustering_summary(graph, k, seeds)
        tables["clustering"].append(clustering_row)

        pop_summary, bucket_rows = popularity_rows(k, train_frequency, indegree)
        tables["popularity"].append(pop_summary)
        tables["popularity_buckets"].extend(bucket_rows)

        for item_index in range(n_real):
            tables["item_metrics"].append(
                {
                    "k": k,
                    "item_id": item_index + 1,
                    "train_frequency": int(train_frequency[item_index + 1]),
                    "indegree": int(indegree[item_index]),
                    "node_reciprocity": float(node_reciprocity[item_index]),
                    "component_size": int(component_sizes_by_node[item_index]),
                    "in_largest_component": bool(in_largest_component_by_node[item_index]),
                    "clustering": float(clustering_by_node[item_index]),
                }
            )

        tables["static"].append(
            {
                "k": k,
                "n_nodes": n_real,
                "n_edges": edge_count,
                **{key: value for key, value in sim_summary.items() if key not in {"k", "n_edges"}},
                **{key: value for key, value in hamming_summary.items() if key not in {"k", "n_edges"}},
                **{key: value for key, value in reciprocity_summary.items() if key not in {"k", "n_edges"}},
                **{key: value for key, value in hub_summary.items() if key not in {"k", "n_edges"}},
                **{key: value for key, value in component_row.items() if key != "k"},
                **{key: value for key, value in clustering_row.items() if key != "k"},
                **{key: value for key, value in pop_summary.items() if key != "k"},
            }
        )

    outputs = static_output_paths(paths)
    write_static_outputs(
        paths=paths,
        outputs=outputs,
        checkpoint_path=harness.checkpoint_path,
        k_values=k_values,
        seeds=seeds,
        tables=tables,
    )

    print(paths.root)
    return 0
