from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import checkpoint_signature


@dataclass
class GraphBuildRecord:
    cache_id: str
    adjacency_path: str
    metadata_path: str
    backend: str
    pool_size: int
    topk: int
    vector_batch_size: int
    build_seconds: float
    loaded_from_cache: bool
    checkpoint_signature: str


def _import_faiss():
    """Import FAISS lazily so profiling utilities can fail only when needed."""
    import faiss

    return faiss


def _valid_item_ids(model: Any) -> np.ndarray:
    """Return the non-padding item IDs that participate in retrieval."""
    return np.arange(1, model.dataset.n_items, dtype=np.int64)


def _graph_topk(config: dict[str, Any]) -> int:
    """Resolve the requested graph neighborhood width from config."""
    if "graph_topk" in config and config["graph_topk"] is not None:
        return int(config["graph_topk"])
    return int(config["n_edges"])


def _vector_batch_size(config: dict[str, Any]) -> int:
    """Resolve the batch size used while materializing item graph vectors."""
    return int(config.get("graph_vector_batch_size", 1024))


def _build_token_embedding_table(model: Any) -> np.ndarray:
    """Build the normalized per-digit token embedding table used for graphing.

    The upstream RPG model stores one embedding vector per semantic-token value
    in `gpt2.wte.weight`. This helper removes special tokens, reshapes the
    tensor into `[n_digit, codebook_size, hidden_dim]`, normalizes each token
    embedding, and returns a NumPy copy for FAISS indexing.

    Args:
        model: Loaded RPG model.

    Returns:
        A float32 NumPy array of shape
        `(n_digit, codebook_size, hidden_dim_per_token)`.
    """
    token_embs = model.gpt2.wte.weight[1:-1].detach().float().cpu()
    token_embs = token_embs.view(model.tokenizer.n_digit, model.tokenizer.codebook_size, -1)
    token_embs = torch.nn.functional.normalize(token_embs, dim=-1)
    return token_embs.numpy().astype(np.float32)


def _iter_graph_vectors(
    model: Any,
    token_table: np.ndarray,
    batch_size: int,
):
    """Yield dense item vectors used to build approximate item-item graphs.

    Each item in RPG is represented by `n_digit` semantic tokens. For graph
    construction we look up the normalized embedding of every digit token,
    concatenate those digit embeddings into one long vector, and scale by
    `1/sqrt(n_digit)` so vector norms stay comparable as the number of digits
    changes.

    Args:
        model: Loaded RPG model containing `item_id2tokens`.
        token_table: Normalized per-digit token embedding table produced by
            `_build_token_embedding_table`.
        batch_size: Number of items to materialize per yielded batch.

    Yields:
        Tuples `(batch_item_ids, batch_vectors)` where `batch_item_ids` is a
        NumPy array of dataset item IDs and `batch_vectors` is a float32 array
        of concatenated item vectors ready for FAISS indexing/search.
    """
    item_ids = _valid_item_ids(model)
    item_tokens = model.item_id2tokens.detach().cpu().numpy()
    n_digit = model.tokenizer.n_digit
    codebook_size = model.tokenizer.codebook_size
    digit_indices = np.arange(n_digit, dtype=np.int64)
    digit_offsets = digit_indices * codebook_size + 1
    scale = np.float32(1.0 / math.sqrt(n_digit))

    for start in range(0, item_ids.shape[0], batch_size):
        end = min(start + batch_size, item_ids.shape[0])
        batch_item_ids = item_ids[start:end]
        batch_tokens = item_tokens[batch_item_ids] - digit_offsets
        batch_vectors = token_table[digit_indices[None, :], batch_tokens]
        batch_vectors = batch_vectors.reshape(batch_item_ids.shape[0], -1)
        batch_vectors = batch_vectors * scale
        yield batch_item_ids, batch_vectors.astype(np.float32, copy=False)


def _build_index(
    dim: int,
    backend: str,
    topk: int,
    config: dict[str, Any],
):
    """Instantiate the FAISS index used for sparse graph construction.

    Args:
        dim: Dimensionality of each item vector.
        backend: Retrieval backend name. Supported values are `"flat"` for
            exact inner-product search and `"hnsw"` for approximate search.
        topk: Desired graph neighborhood size. Used to derive default HNSW
            search parameters.
        config: Profiling config dictionary containing backend-specific knobs.

    Returns:
        A configured FAISS index instance.

    Raises:
        ValueError: If the requested backend is not supported.
    """
    faiss = _import_faiss()
    if backend == "flat":
        return faiss.IndexFlatIP(dim)
    if backend == "hnsw":
        hnsw_m = int(config.get("graph_hnsw_m", 32))
        ef_construction = int(config.get("graph_hnsw_ef_construction", 200))
        ef_search = int(config.get("graph_hnsw_ef_search", max(256, topk * 2)))
        index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
        return index
    raise ValueError(f"Unsupported graph backend: {backend}")


def _enforce_self_neighbors(
    search_result_ids: np.ndarray,
    batch_item_ids: np.ndarray,
    topk: int,
) -> np.ndarray:
    """Normalize raw nearest-neighbor results into RPG adjacency rows.

    The graph-decoding code assumes that every adjacency row starts with the
    item itself, followed by unique valid neighbors. FAISS search results may
    omit self-neighbors or include duplicates/invalid entries, so this helper
    repairs each row and pads with the item itself if the candidate list is too
    short.

    Args:
        search_result_ids: Retrieved neighbor item IDs per query item.
        batch_item_ids: Item IDs corresponding to the query rows.
        topk: Final number of neighbors required per row.

    Returns:
        An integer array of shape `(len(batch_item_ids), topk)` suitable for
        direct insertion into the adjacency tensor.
    """
    adjusted = np.zeros((batch_item_ids.shape[0], topk), dtype=np.int64)

    for row_index, item_id in enumerate(batch_item_ids):
        row = [int(item_id)]
        for neighbor_id in search_result_ids[row_index]:
            neighbor = int(neighbor_id)
            if neighbor <= 0 or neighbor == item_id:
                continue
            if neighbor in row:
                continue
            row.append(neighbor)
            if len(row) == topk:
                break
        while len(row) < topk:
            row.append(int(item_id))
        adjusted[row_index] = np.array(row, dtype=np.int64)

    return adjusted


def _candidate_search_k(valid_pool_size: int, topk: int, backend: str) -> int:
    """Choose how many FAISS neighbors to request before post-processing."""
    if backend == "flat":
        return min(valid_pool_size, topk + 1)
    return min(valid_pool_size, max(topk * 4, topk + 1))


def build_sparse_adjacency(
    model: Any,
    backend: str,
    topk: int,
    config: dict[str, Any],
) -> torch.Tensor:
    """Build an item-item adjacency tensor using vector search over item codes.

    This is the scalable replacement for the original dense
    `build_ii_sim_mat()` path. It converts every item into a concatenated vector
    of normalized digit-token embeddings, indexes those vectors with FAISS, and
    searches the resulting index to obtain the top-`k` neighbors for every
    non-padding item.

    Args:
        model: Loaded RPG model.
        backend: FAISS backend to use, typically `"flat"` or `"hnsw"`.
        topk: Number of neighbors to keep per item.
        config: Profiling config dictionary containing vectorization and backend
            parameters.

    Returns:
        A CPU `torch.LongTensor` of shape `(n_items, topk)` where row `0` is
        zeroed for padding and every other row contains item IDs.
    """
    token_table = _build_token_embedding_table(model)
    dim = model.tokenizer.n_digit * token_table.shape[-1]
    index = _build_index(dim=dim, backend=backend, topk=topk, config=config)

    batch_size = _vector_batch_size(config)
    item_ids = _valid_item_ids(model)
    valid_pool_size = item_ids.shape[0]
    search_k = _candidate_search_k(valid_pool_size=valid_pool_size, topk=topk, backend=backend)

    for _, batch_vectors in _iter_graph_vectors(model, token_table, batch_size=batch_size):
        index.add(batch_vectors)

    adjacency = torch.zeros((model.dataset.n_items, topk), dtype=torch.long)
    for batch_item_ids, batch_vectors in _iter_graph_vectors(model, token_table, batch_size=batch_size):
        _, raw_neighbors = index.search(batch_vectors, search_k)
        neighbor_item_ids = raw_neighbors.astype(np.int64) + 1
        adjusted_neighbors = _enforce_self_neighbors(
            search_result_ids=neighbor_item_ids,
            batch_item_ids=batch_item_ids,
            topk=topk,
        )
        adjacency[batch_item_ids] = torch.from_numpy(adjusted_neighbors)

    adjacency[0] = 0
    return adjacency


def build_dense_reference_adjacency(model: Any, topk: int) -> torch.Tensor:
    """Build the original dense adjacency as a correctness reference.

    The upstream RPG release computes an all-pairs item similarity matrix and
    then keeps the top neighbors per row. This helper temporarily overrides the
    model's `n_edges` setting so the dense path returns the requested `topk`.

    Args:
        model: Loaded RPG model exposing `build_ii_sim_mat()`.
        topk: Number of neighbors to extract per item.

    Returns:
        A CPU `torch.LongTensor` adjacency tensor built by the original dense
        implementation.
    """
    original_n_edges = model.n_edges
    try:
        model.n_edges = topk
        similarity = model.build_ii_sim_mat()
        adjacency = torch.topk(similarity, k=topk, dim=-1).indices.detach().cpu()
        adjacency[0] = 0
        return adjacency
    finally:
        model.n_edges = original_n_edges


def compare_adjacency_sets(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    valid_item_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """Check exact set equality between two adjacency tensors.

    Args:
        reference: Gold/reference adjacency tensor.
        candidate: Adjacency tensor to validate against the reference.
        valid_item_ids: Optional subset of item IDs to inspect. Defaults to all
            non-padding items.

    Returns:
        A dictionary summarizing whether all checked rows match exactly and
        including up to 20 mismatch examples.
    """
    if valid_item_ids is None:
        valid_item_ids = np.arange(1, reference.shape[0], dtype=np.int64)

    mismatches: list[dict[str, Any]] = []
    mismatch_count = 0
    for item_id in valid_item_ids:
        reference_set = set(reference[item_id].tolist())
        candidate_set = set(candidate[item_id].tolist())
        if reference_set != candidate_set:
            mismatch_count += 1
            if len(mismatches) < 20:
                mismatches.append(
                    {
                        "item_id": int(item_id),
                        "reference": sorted(reference_set),
                        "candidate": sorted(candidate_set),
                    }
                )

    return {
        "match": mismatch_count == 0,
        "checked_items": int(len(valid_item_ids)),
        "mismatch_count": mismatch_count,
        "mismatch_examples": mismatches,
    }


def compare_adjacency_overlap(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    valid_item_ids: np.ndarray | None = None,
    max_examples: int = 20,
) -> dict[str, Any]:
    """Summarize overlap quality between two adjacency tensors.

    Unlike `compare_adjacency_sets`, this helper reports graded overlap
    statistics, which are more informative when evaluating approximate graph
    backends such as HNSW.

    Args:
        reference: Gold/reference adjacency tensor.
        candidate: Adjacency tensor to compare against the reference.
        valid_item_ids: Optional subset of item IDs to inspect. Defaults to all
            non-padding items.
        max_examples: Maximum number of mismatch examples to include.

    Returns:
        A dictionary containing overlap-rate statistics and representative
        mismatch rows.
    """
    if valid_item_ids is None:
        valid_item_ids = np.arange(1, reference.shape[0], dtype=np.int64)

    overlap_counts: list[int] = []
    overlap_rates: list[float] = []
    mismatch_examples: list[dict[str, Any]] = []
    exact_match_count = 0

    for item_id in valid_item_ids:
        reference_set = set(reference[item_id].tolist())
        candidate_set = set(candidate[item_id].tolist())
        overlap = len(reference_set & candidate_set)
        denominator = max(len(reference_set), 1)
        overlap_counts.append(overlap)
        overlap_rates.append(overlap / denominator)

        if reference_set == candidate_set:
            exact_match_count += 1
        elif len(mismatch_examples) < max_examples:
            mismatch_examples.append(
                {
                    "item_id": int(item_id),
                    "overlap_count": int(overlap),
                    "overlap_rate": overlap / denominator,
                    "missing_from_candidate": sorted(reference_set - candidate_set),
                    "extra_in_candidate": sorted(candidate_set - reference_set),
                }
            )

    rates = np.array(overlap_rates, dtype=np.float64)
    counts = np.array(overlap_counts, dtype=np.float64)
    checked_items = int(len(valid_item_ids))
    mismatch_count = checked_items - exact_match_count

    return {
        "checked_items": checked_items,
        "topk": int(reference.shape[1]),
        "exact_match": mismatch_count == 0,
        "exact_match_count": int(exact_match_count),
        "exact_match_rate": exact_match_count / max(checked_items, 1),
        "mismatch_count": int(mismatch_count),
        "mean_overlap_count": float(counts.mean()) if checked_items else 0.0,
        "mean_overlap_rate": float(rates.mean()) if checked_items else 0.0,
        "min_overlap_rate": float(rates.min()) if checked_items else 0.0,
        "p5_overlap_rate": float(np.percentile(rates, 5)) if checked_items else 0.0,
        "p50_overlap_rate": float(np.percentile(rates, 50)) if checked_items else 0.0,
        "p95_overlap_rate": float(np.percentile(rates, 95)) if checked_items else 0.0,
        "mismatch_examples": mismatch_examples,
    }


def _graph_cache_dir(config: dict[str, Any]) -> Path:
    """Resolve the directory used to cache built adjacency tensors."""
    raw_path = config.get("graph_cache_dir")
    if raw_path is None:
        raw_path = Path(config["cache_dir"]).resolve().parents[0] / "perf" / "graphs"
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _graph_cache_id(
    checkpoint_path: Path,
    config: dict[str, Any],
    pool_size: int,
    backend: str,
    topk: int,
) -> str:
    """Build a stable cache identifier for one graph-construction setting.

    The cache ID encodes all knobs that affect the adjacency contents so cached
    graph files can be reused safely across profiling runs.
    """
    category = str(config["category"]).lower()
    model_name = str(config["model"]).lower()
    signature = checkpoint_signature(checkpoint_path)
    dummy_seed = int(config.get("dummy_pool_seed", config.get("rand_seed", 0)))
    hnsw_suffix = ""
    if backend == "hnsw":
        hnsw_suffix = (
            f"_m-{int(config.get('graph_hnsw_m', 32))}"
            f"_efc-{int(config.get('graph_hnsw_ef_construction', 200))}"
            f"_efs-{int(config.get('graph_hnsw_ef_search', max(256, topk * 2)))}"
        )
    return (
        f"{model_name}_{category}_pool{pool_size}_backend-{backend}_"
        f"topk-{topk}{hnsw_suffix}_seed-{dummy_seed}_{signature}"
    )


def _expected_cache_metadata(
    checkpoint_path: Path,
    config: dict[str, Any],
    pool_size: int,
    backend: str,
    topk: int,
) -> dict[str, Any]:
    """Create the metadata dictionary that a valid cache entry must satisfy."""
    metadata = {
        "cache_id": _graph_cache_id(
            checkpoint_path=checkpoint_path,
            config=config,
            pool_size=pool_size,
            backend=backend,
            topk=topk,
        ),
        "backend": backend,
        "pool_size": int(pool_size),
        "topk": int(topk),
        "vector_batch_size": _vector_batch_size(config),
        "checkpoint_signature": checkpoint_signature(checkpoint_path),
    }
    if backend == "hnsw":
        metadata["graph_hnsw_m"] = int(config.get("graph_hnsw_m", 32))
        metadata["graph_hnsw_ef_construction"] = int(
            config.get("graph_hnsw_ef_construction", 200)
        )
        metadata["graph_hnsw_ef_search"] = int(
            config.get("graph_hnsw_ef_search", max(256, topk * 2))
        )
    return metadata


def _validate_cached_adjacency(
    adjacency: torch.Tensor,
    metadata: dict[str, Any],
    expected_metadata: dict[str, Any],
    model: Any,
) -> None:
    """Validate that a cached adjacency tensor matches the current request.

    Args:
        adjacency: Cached adjacency tensor loaded from disk.
        metadata: Metadata stored alongside the cached tensor.
        expected_metadata: Metadata values implied by the current run.
        model: Loaded model, used to validate the expected tensor shape.

    Returns:
        None.

    Raises:
        ValueError: If metadata fields or tensor shape do not match the current
            request.
    """
    for key, expected_value in expected_metadata.items():
        actual_value = metadata.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"Cached adjacency metadata mismatch for '{key}': "
                f"expected {expected_value!r}, found {actual_value!r}."
            )

    expected_shape = (model.dataset.n_items, int(expected_metadata["topk"]))
    if tuple(adjacency.shape) != expected_shape:
        raise ValueError(
            f"Cached adjacency shape mismatch: expected {expected_shape}, "
            f"found {tuple(adjacency.shape)}."
        )


def _graph_cache_paths(
    checkpoint_path: Path,
    config: dict[str, Any],
    pool_size: int,
    backend: str,
    topk: int,
) -> tuple[str, Path, Path]:
    """Return the cache ID plus the data and metadata paths for one graph run."""
    cache_dir = _graph_cache_dir(config)
    cache_id = _graph_cache_id(
        checkpoint_path=checkpoint_path,
        config=config,
        pool_size=pool_size,
        backend=backend,
        topk=topk,
    )
    adjacency_path = cache_dir / f"{cache_id}.pt"
    metadata_path = cache_dir / f"{cache_id}.json"
    return cache_id, adjacency_path, metadata_path


def build_or_load_adjacency(
    model: Any,
    checkpoint_path: Path,
    config: dict[str, Any],
    pool_size: int,
    backend: str,
    force_rebuild: bool = False,
) -> tuple[torch.Tensor, GraphBuildRecord]:
    """Load a cached graph adjacency or build and cache it on demand.

    Args:
        model: Loaded RPG model used to build the adjacency when needed.
        checkpoint_path: Checkpoint path whose signature becomes part of the
            cache key.
        config: Profiling config dictionary.
        pool_size: Candidate pool size being profiled.
        backend: Graph backend name, typically `"flat"` or `"hnsw"`.
        force_rebuild: If `True`, ignore any existing cache files and rebuild.

    Returns:
        A tuple `(adjacency, record)` where `adjacency` is the CPU adjacency
        tensor and `record` summarizes how it was produced.
    """
    topk = _graph_topk(config)
    expected_metadata = _expected_cache_metadata(
        checkpoint_path=checkpoint_path,
        config=config,
        pool_size=pool_size,
        backend=backend,
        topk=topk,
    )
    cache_id, adjacency_path, metadata_path = _graph_cache_paths(
        checkpoint_path=checkpoint_path,
        config=config,
        pool_size=pool_size,
        backend=backend,
        topk=topk,
    )
    adjacency_path.parent.mkdir(parents=True, exist_ok=True)

    if adjacency_path.is_file() and metadata_path.is_file() and not force_rebuild:
        adjacency = torch.load(adjacency_path, map_location="cpu")
        metadata = json.loads(metadata_path.read_text())
        _validate_cached_adjacency(
            adjacency=adjacency,
            metadata=metadata,
            expected_metadata=expected_metadata,
            model=model,
        )
        return adjacency, GraphBuildRecord(
            cache_id=cache_id,
            adjacency_path=str(adjacency_path),
            metadata_path=str(metadata_path),
            backend=backend,
            pool_size=pool_size,
            topk=topk,
            vector_batch_size=_vector_batch_size(config),
            build_seconds=float(metadata["build_seconds"]),
            loaded_from_cache=True,
            checkpoint_signature=metadata["checkpoint_signature"],
        )

    start_time = time.perf_counter()
    adjacency = build_sparse_adjacency(
        model=model,
        backend=backend,
        topk=topk,
        config=config,
    )
    build_seconds = time.perf_counter() - start_time

    metadata = {
        **expected_metadata,
        "build_seconds": build_seconds,
    }
    torch.save(adjacency, adjacency_path)
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return adjacency, GraphBuildRecord(
        cache_id=cache_id,
        adjacency_path=str(adjacency_path),
        metadata_path=str(metadata_path),
        backend=backend,
        pool_size=pool_size,
        topk=topk,
        vector_batch_size=_vector_batch_size(config),
        build_seconds=build_seconds,
        loaded_from_cache=False,
        checkpoint_signature=metadata["checkpoint_signature"],
    )
