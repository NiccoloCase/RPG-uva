"""Structural metrics for static graph analysis.

This module covers A3 reciprocity, A4 hubness, A5 connected components, and A6
clustering. Directed metrics operate on flattened source/destination arrays;
component and clustering metrics use a symmetrized undirected igraph graph.
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np


def gini(values: np.ndarray) -> float:
    """Compute the Gini coefficient for a non-negative vector."""

    if values.size == 0:
        return float("nan")
    sorted_values = np.sort(values.astype(np.float64))
    total = sorted_values.sum()
    if total <= 0:
        return 0.0
    n = sorted_values.shape[0]
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * sorted_values) / (n * total)) - ((n + 1.0) / n))


def reciprocity(
    sources: np.ndarray,
    destinations: np.ndarray,
    n_items: int,
) -> tuple[dict[str, Any], np.ndarray]:
    """Measure reciprocal directed edges globally and per source node."""

    n_real = n_items - 1
    if sources.size == 0:
        return {
            "reciprocity": 0.0,
            "node_reciprocity_mean": float("nan"),
            "node_reciprocity_median": float("nan"),
            "node_reciprocity_p10": float("nan"),
            "node_reciprocity_p90": float("nan"),
        }, np.zeros(n_real, dtype=np.float32)

    edge_codes = sources * np.int64(n_items) + destinations
    reverse_codes = destinations * np.int64(n_items) + sources
    reciprocal_mask = np.isin(reverse_codes, edge_codes, assume_unique=False)
    reciprocal_counts = np.bincount(sources[reciprocal_mask] - 1, minlength=n_real).astype(
        np.float32
    )
    out_counts = np.bincount(sources - 1, minlength=n_real).astype(np.float32)
    node_values = np.divide(
        reciprocal_counts,
        out_counts,
        out=np.zeros_like(reciprocal_counts),
        where=out_counts > 0,
    )
    return {
        "reciprocity": float(reciprocal_mask.mean()),
        "node_reciprocity_mean": float(np.mean(node_values)),
        "node_reciprocity_median": float(np.median(node_values)),
        "node_reciprocity_p10": float(np.percentile(node_values, 10)),
        "node_reciprocity_p90": float(np.percentile(node_values, 90)),
    }, node_values


def import_igraph():
    """Import igraph lazily and fail explicitly if the declared dependency is missing."""

    try:
        import igraph as ig
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The graph-analysis static command requires igraph. "
            "Install the declared dependency from environment.yml or requirements.txt."
        ) from exc
    return ig


def undirected_graph(sources: np.ndarray, destinations: np.ndarray, n_real: int):
    """Build the undirected symmetrized graph used by components and clustering."""

    ig = import_igraph()
    edges = list(zip((sources - 1).tolist(), (destinations - 1).tolist()))
    graph = ig.Graph(n=n_real, edges=edges, directed=False)
    graph.simplify(multiple=True, loops=True)
    return graph


def component_summary(graph, k: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Summarize undirected connected components.

    Returns:
        A tuple ``(summary, node_component_sizes, in_largest_component)`` where
        node-level arrays are indexed by ``item_id - 1``.
    """

    components = graph.connected_components(mode="weak")
    membership = np.asarray(components.membership, dtype=np.int64)
    sizes = np.asarray(components.sizes(), dtype=np.int64)
    node_component_sizes = sizes[membership] if sizes.size else np.zeros(graph.vcount(), dtype=np.int64)
    sorted_sizes = np.sort(sizes)[::-1] if sizes.size else np.array([], dtype=np.int64)
    largest = int(sorted_sizes[0]) if sorted_sizes.size else 0
    largest_component_id = int(np.argmax(sizes)) if sizes.size else -1
    in_largest_component = (
        membership == largest_component_id if sizes.size else np.zeros(graph.vcount(), dtype=bool)
    )
    second = int(sorted_sizes[1]) if sorted_sizes.size > 1 else 0
    return {
        "k": k,
        "undirected_edges": int(graph.ecount()),
        "undirected_density": float(graph.density(loops=False)) if graph.vcount() > 1 else 0.0,
        "n_components": int(sizes.size),
        "largest_component_size": largest,
        "largest_component_fraction": largest / max(graph.vcount(), 1),
        "second_largest_component_size": second,
        "n_isolates": int(np.sum(sizes == 1)),
        "component_size_median": float(np.median(sizes)) if sizes.size else float("nan"),
        "component_size_p90": float(np.percentile(sizes, 90)) if sizes.size else float("nan"),
        "nodes_in_components_lt10_fraction": (
            float(sizes[sizes < 10].sum() / max(graph.vcount(), 1)) if sizes.size else 0.0
        ),
    }, node_component_sizes, in_largest_component


def random_graph_clustering(n_vertices: int, n_edges: int, seed: int) -> float:
    """Average local clustering of an Erdos-Renyi graph with matched edge count."""

    ig = import_igraph()
    ig.set_random_number_generator(random.Random(seed))
    graph = ig.Graph.Erdos_Renyi(n=n_vertices, m=n_edges, directed=False, loops=False)
    values = np.asarray(graph.transitivity_local_undirected(mode="zero"), dtype=np.float64)
    return float(values.mean()) if values.size else float("nan")


def clustering_summary(graph, k: int, seeds: list[int]) -> tuple[dict[str, Any], np.ndarray]:
    """Summarize local clustering and compare to random matched-edge baselines."""

    values = np.asarray(graph.transitivity_local_undirected(mode="zero"), dtype=np.float64)
    random_values = np.asarray(
        [random_graph_clustering(graph.vcount(), graph.ecount(), seed) for seed in seeds],
        dtype=np.float64,
    )
    random_mean = float(np.nanmean(random_values)) if random_values.size else float("nan")
    mean_value = float(np.mean(values)) if values.size else float("nan")
    lift = mean_value / random_mean if random_mean > 1e-12 else None
    return {
        "k": k,
        "clustering_mean": mean_value,
        "clustering_median": float(np.median(values)) if values.size else float("nan"),
        "clustering_p10": float(np.percentile(values, 10)) if values.size else float("nan"),
        "clustering_p90": float(np.percentile(values, 90)) if values.size else float("nan"),
        "clustering_zero_fraction": float(np.mean(values == 0.0)) if values.size else float("nan"),
        "random_clustering_mean": random_mean,
        "random_clustering_std": float(np.nanstd(random_values)) if random_values.size else float("nan"),
        "clustering_lift_over_random": lift,
    }, values


def indegree_histogram_rows(indegree: np.ndarray, k: int, source: str) -> list[dict[str, Any]]:
    """Return one row per non-empty in-degree value."""

    counts = np.bincount(indegree.astype(np.int64))
    return [
        {"k": k, "source": source, "indegree": int(value), "count": int(count)}
        for value, count in enumerate(counts)
        if count
    ]


def random_indegree_summaries(
    n_items: int,
    edge_count: int,
    seeds: list[int],
    max_pairs_by_seed: dict[int, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Summarize random directed-graph in-degree baselines."""

    n_real = n_items - 1
    rows = []
    for seed in seeds:
        _, random_destinations = max_pairs_by_seed[seed]
        indegree = np.bincount(random_destinations[:edge_count] - 1, minlength=n_real)
        top_count = max(1, int(math.ceil(n_real * 0.01)))
        rows.append(
            {
                "mean": float(indegree.mean()),
                "std": float(indegree.std()),
                "max": float(indegree.max()),
                "gini": gini(indegree),
                "top_share": float(np.sort(indegree)[-top_count:].sum() / max(indegree.sum(), 1)),
            }
        )
    return {
        "random_indegree_mean": float(np.mean([row["mean"] for row in rows])),
        "random_indegree_std": float(np.mean([row["std"] for row in rows])),
        "random_indegree_max_mean": float(np.mean([row["max"] for row in rows])),
        "random_indegree_gini_mean": float(np.mean([row["gini"] for row in rows])),
        "random_top_1pct_incoming_edge_share_mean": float(np.mean([row["top_share"] for row in rows])),
    }

