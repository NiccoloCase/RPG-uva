from __future__ import annotations

import json
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


def _hnsw_params(config: dict[str, Any], topk: int) -> tuple[int, int, int]:
    """HNSW settings: graph degree M, efConstruction and efSearch.

    Defaults are the ones used for the HNSW results in the paper
    (``IndexHNSWFlat(d, 32)``, efConstruction 200, efSearch max(k+1, 50)).
    """
    hnsw_m = int(config.get("graph_hnsw_m", 32))
    ef_construction = int(config.get("graph_hnsw_ef_construction", 200))
    ef_search = int(config.get("graph_hnsw_ef_search") or max(topk + 1, 50))
    return hnsw_m, ef_construction, ef_search


def build_hnsw_adjacency(
    model: Any,
    topk: int,
    hnsw_m: int = 32,
    ef_construction: int = 200,
    ef_search: int | None = None,
) -> torch.Tensor:
    """Build an approximate decoding graph with a FAISS HNSW index.

    Each item is represented by the L2-normalised mean of its token embeddings
    (one vector of the backbone width per item), so the index holds n vectors of
    dimension d and construction needs O(n * d) memory instead of the dense
    n x n similarity matrix. This similarity differs from the upstream graph
    (mean over digits of the per-digit cosine), and the item itself is not kept
    as a neighbour.

    Returns:
        A CPU ``torch.LongTensor`` of shape ``(n_items, topk)``; row 0 (padding)
        is zero.
    """
    faiss = _import_faiss()
    n_items = int(model.dataset.n_items)
    if ef_search is None:
        ef_search = max(topk + 1, 50)
    with torch.no_grad():
        token_embs = model.gpt2.wte.weight[1:-1]
        token_ids = model.item_id2tokens[1:] - 1
        item_embs = token_embs[token_ids.to(token_embs.device)].mean(dim=1)
        item_embs = torch.nn.functional.normalize(item_embs, dim=-1)
    item_embs_np = item_embs.detach().cpu().float().numpy()

    index = faiss.IndexHNSWFlat(item_embs_np.shape[1], hnsw_m)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(item_embs_np)
    _, neighbours = index.search(item_embs_np, topk + 1)
    neighbours = neighbours[:, 1:]  # drop the query item itself

    adjacency = torch.zeros((n_items, topk), dtype=torch.long)
    adjacency[1:] = torch.from_numpy(neighbours.astype(np.int64) + 1)
    return adjacency


def build_chunked_exact_adjacency(
    model: Any,
    topk: int,
    row_chunk_size: int = 512,
) -> torch.Tensor:
    """Build the exact upstream decoding graph without the dense n x n matrix.

    Same arithmetic as the upstream ``build_ii_sim_mat`` + ``torch.topk``:
    the similarity of two items is the mean over digits of the cosine between
    their digit tokens, mapped to [0, 1]. The difference is only the memory
    schedule: rows are scored one block at a time on the model's device and
    only each row's top-``topk`` neighbours are kept, so memory is
    O(row_chunk_size * n_items) instead of O(n_items^2). Neighbour sets match
    the dense builder exactly, up to ties.

    Returns:
        A CPU ``torch.LongTensor`` of shape ``(n_items, topk)``; row 0 (padding)
        is zero and every other row starts with the item itself.
    """
    device = model.gpt2.device
    n_items = int(model.dataset.n_items)
    n_digit = int(model.tokenizer.n_digit)
    codebook_size = int(model.tokenizer.codebook_size)
    if topk > n_items - 1:
        raise ValueError(f"topk={topk} exceeds the number of real items ({n_items - 1}).")

    with torch.no_grad():
        token_embs = model.gpt2.wte.weight[1:-1].view(n_digit, codebook_size, -1)
        token_embs = torch.nn.functional.normalize(token_embs, dim=-1)
        token_sims = 0.5 * (torch.bmm(token_embs, token_embs.transpose(1, 2)) + 1.0)
        offsets = torch.arange(n_digit, device=device) * codebook_size + 1
        codes = (model.item_id2tokens.to(device)[1:] - offsets).contiguous()  # (n_real, n_digit)

        adjacency = torch.zeros((n_items, topk), dtype=torch.long)
        for start in range(0, n_items - 1, row_chunk_size):
            end = min(start + row_chunk_size, n_items - 1)
            similarity = torch.zeros((end - start, n_items - 1), device=device)
            for digit in range(n_digit):
                digit_rows = token_sims[digit].index_select(0, codes[start:end, digit])
                similarity += digit_rows.index_select(1, codes[:, digit])
            similarity /= n_digit
            neighbours = torch.topk(similarity, k=topk, dim=-1).indices + 1
            adjacency[start + 1 : end + 1] = neighbours.cpu()
            del similarity
    return adjacency


def build_sparse_adjacency(
    model: Any,
    backend: str,
    topk: int,
    config: dict[str, Any],
) -> torch.Tensor:
    """Build the item-item adjacency used by graph-constrained decoding.

    Args:
        model: Loaded RPG model.
        backend: ``"exact_gpu"`` (the upstream similarity computed in row
            blocks on the model's device, see ``build_chunked_exact_adjacency``),
            ``"hnsw"`` (approximate graph over mean token embeddings, see
            ``build_hnsw_adjacency``) or ``"dense"`` (the upstream builder with
            the full n x n matrix).
        topk: Number of neighbours to keep per item.
        config: Config dictionary with the backend knobs
            (``graph_row_chunk_size``, ``graph_hnsw_*``).

    Returns:
        A CPU ``torch.LongTensor`` of shape ``(n_items, topk)`` where row ``0``
        is zero for padding and every other row contains item IDs.
    """
    if backend == "exact_gpu":
        return build_chunked_exact_adjacency(
            model, topk=topk, row_chunk_size=int(config.get("graph_row_chunk_size", 512))
        )
    if backend == "hnsw":
        hnsw_m, ef_construction, ef_search = _hnsw_params(config, topk)
        return build_hnsw_adjacency(
            model, topk=topk, hnsw_m=hnsw_m, ef_construction=ef_construction, ef_search=ef_search
        )
    if backend == "dense":
        return build_dense_reference_adjacency(model, topk=topk)
    raise ValueError(f"Unsupported graph backend: {backend} (expected exact_gpu, hnsw or dense).")


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
        hnsw_m, ef_construction, ef_search = _hnsw_params(config, topk)
        hnsw_suffix = f"_m-{hnsw_m}_efc-{ef_construction}_efs-{ef_search}"
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
        "checkpoint_signature": checkpoint_signature(checkpoint_path),
    }
    if backend == "hnsw":
        hnsw_m, ef_construction, ef_search = _hnsw_params(config, topk)
        metadata["graph_hnsw_m"] = hnsw_m
        metadata["graph_hnsw_ef_construction"] = ef_construction
        metadata["graph_hnsw_ef_search"] = ef_search
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
        backend: Graph backend name, `"exact_gpu"` or `"hnsw"`.
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
        build_seconds=build_seconds,
        loaded_from_cache=False,
        checkpoint_signature=metadata["checkpoint_signature"],
    )
