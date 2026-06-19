"""Edge-level metrics for static graph analysis.

This module covers A1 neighbor similarity and A2 semantic-ID Hamming distance,
plus small CSV-summary helpers used by those experiments.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def edge_arrays(adjacency: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten the first ``k`` adjacency columns into directed edge arrays.

    Padding item ``0`` and self-edges are removed. The returned rank array keeps
    the original raw adjacency-column rank, so slicing by ``rank < k`` later
    still means "first k graph neighbors before self-edge removal".
    """

    n_items = adjacency.shape[0]
    source_rows = np.arange(1, n_items, dtype=np.int64)
    neighbors = adjacency[1:, :k].astype(np.int64, copy=False)
    sources = np.repeat(source_rows, k)
    destinations = neighbors.reshape(-1)
    ranks = np.tile(np.arange(k, dtype=np.int64), source_rows.shape[0])
    mask = (destinations > 0) & (destinations != sources)
    return sources[mask], destinations[mask], ranks[mask]


def digit_offsets(n_digit: int, codebook_size: int) -> np.ndarray:
    """Return offsets that convert global RPG token ids to codebook-local ids."""

    return np.arange(n_digit, dtype=np.int64) * codebook_size + 1


def compute_shifted_similarity(
    sources: np.ndarray,
    destinations: np.ndarray,
    item_tokens: np.ndarray,
    token_table: np.ndarray,
    codebook_size: int,
    batch_size: int,
) -> np.ndarray:
    """Compute the RPG graph-construction similarity for item pairs.

    RPG represents each item as one token per semantic-ID digit. For a pair
    ``(i, j)``, this metric averages cosine similarities between the trained
    token embeddings at each digit, then shifts the value from ``[-1, 1]`` to
    ``[0, 1]``. This mirrors the similarity used by the graph builder.
    """

    n_edges = sources.shape[0]
    n_digit = token_table.shape[0]
    offsets = digit_offsets(n_digit, codebook_size)
    scores = np.empty(n_edges, dtype=np.float32)

    for start in range(0, n_edges, batch_size):
        end = min(start + batch_size, n_edges)
        src_tokens = item_tokens[sources[start:end]] - offsets
        dst_tokens = item_tokens[destinations[start:end]] - offsets
        batch_scores = np.zeros(end - start, dtype=np.float32)
        for digit in range(n_digit):
            src_emb = token_table[digit, src_tokens[:, digit]]
            dst_emb = token_table[digit, dst_tokens[:, digit]]
            batch_scores += np.einsum("ij,ij->i", src_emb, dst_emb, optimize=True)
        batch_scores /= np.float32(n_digit)
        scores[start:end] = 0.5 * (batch_scores + 1.0)
    return scores


def compute_hamming(
    sources: np.ndarray,
    destinations: np.ndarray,
    item_tokens: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Compute semantic-ID Hamming distance for item pairs."""

    distances = np.empty(sources.shape[0], dtype=np.int16)
    for start in range(0, sources.shape[0], batch_size):
        end = min(start + batch_size, sources.shape[0])
        distances[start:end] = np.count_nonzero(
            item_tokens[sources[start:end]] != item_tokens[destinations[start:end]],
            axis=1,
        )
    return distances


def random_pairs(n_items: int, n_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample random directed item pairs with ``i != j`` and no padding item."""

    rng = np.random.default_rng(seed)
    sources = rng.integers(1, n_items, size=n_pairs, dtype=np.int64)
    destinations = rng.integers(1, n_items, size=n_pairs, dtype=np.int64)
    equal = sources == destinations
    while np.any(equal):
        destinations[equal] = rng.integers(1, n_items, size=int(equal.sum()), dtype=np.int64)
        equal = sources == destinations
    return sources, destinations


def summary(values: np.ndarray, prefix: str) -> dict[str, Any]:
    """Return mean, median, p10, and p90 with names prefixed by ``prefix``."""

    if values.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_p10": float("nan"),
            f"{prefix}_p90": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p10": float(np.percentile(values, 10)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
    }


def histogram_rows(
    values: np.ndarray,
    bins: np.ndarray,
    value_name: str,
    k: int,
    source: str,
) -> list[dict[str, Any]]:
    """Convert a NumPy histogram into long-form CSV rows."""

    counts, edges = np.histogram(values, bins=bins)
    rows = []
    for index, count in enumerate(counts):
        rows.append(
            {
                "k": k,
                "source": source,
                "metric": value_name,
                "bin_left": float(edges[index]),
                "bin_right": float(edges[index + 1]),
                "count": int(count),
                "fraction": float(count / max(values.shape[0], 1)),
            }
        )
    return rows


def integer_histogram_rows(
    values: np.ndarray,
    max_value: int,
    value_name: str,
    k: int,
    source: str,
) -> list[dict[str, Any]]:
    """Return histogram rows for integer-valued metrics such as Hamming distance."""

    counts = np.bincount(values.astype(np.int64), minlength=max_value + 1)
    return [
        {
            "k": k,
            "source": source,
            "metric": value_name,
            "value": int(value),
            "count": int(count),
            "fraction": float(count / max(values.shape[0], 1)),
        }
        for value, count in enumerate(counts)
    ]

