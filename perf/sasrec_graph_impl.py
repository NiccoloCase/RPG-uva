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
    pool_size: int
    topk: int
    build_block_size: int
    build_seconds: float
    loaded_from_cache: bool
    checkpoint_signature: str


def _graph_topk(config: dict[str, Any]) -> int:
    return int(config.get("graph_topk", 100))


def _graph_build_block_size(config: dict[str, Any]) -> int:
    return int(config.get("graph_build_block_size", 2048))


def _graph_cache_dir(config: dict[str, Any]) -> Path:
    raw_path = config.get("graph_cache_dir")
    if raw_path is None:
        raw_path = "artifacts/sasrec/perf/graphs"
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _graph_cache_id(
    checkpoint_path: Path,
    config: dict[str, Any],
    pool_size: int,
    topk: int,
) -> str:
    category = str(config.get("data_name", config.get("dataset", "dataset"))).lower()
    signature = checkpoint_signature(checkpoint_path)
    dummy_seed = int(config.get("dummy_pool_seed", config.get("seed", 0)))
    return (
        f"sasrec_{category}_pool{pool_size}_"
        f"topk-{topk}_seed-{dummy_seed}_{signature}"
    )


def _graph_cache_paths(
    checkpoint_path: Path,
    config: dict[str, Any],
    pool_size: int,
    topk: int,
) -> tuple[str, Path, Path]:
    cache_dir = _graph_cache_dir(config)
    cache_id = _graph_cache_id(
        checkpoint_path=checkpoint_path,
        config=config,
        pool_size=pool_size,
        topk=topk,
    )
    adjacency_path = cache_dir / f"{cache_id}.pt"
    metadata_path = cache_dir / f"{cache_id}.json"
    return cache_id, adjacency_path, metadata_path


def _expected_cache_metadata(
    checkpoint_path: Path,
    config: dict[str, Any],
    pool_size: int,
    topk: int,
    original_pool_size: int,
) -> dict[str, Any]:
    return {
        "cache_id": _graph_cache_id(
            checkpoint_path=checkpoint_path,
            config=config,
            pool_size=pool_size,
            topk=topk,
        ),
        "pool_size": int(pool_size),
        "topk": int(topk),
        "build_block_size": _graph_build_block_size(config),
        "original_pool_size": int(original_pool_size),
        "checkpoint_signature": checkpoint_signature(checkpoint_path),
        "dummy_pool_seed": int(config.get("dummy_pool_seed", config.get("seed", 0))),
    }


def _validate_cached_adjacency(
    adjacency: torch.Tensor,
    metadata: dict[str, Any],
    expected_metadata: dict[str, Any],
    candidate_item_ids: np.ndarray,
) -> None:
    for key, expected_value in expected_metadata.items():
        actual_value = metadata.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"Cached adjacency metadata mismatch for '{key}': "
                f"expected {expected_value!r}, found {actual_value!r}."
            )

    expected_shape = (int(expected_metadata["pool_size"]) + 2, int(expected_metadata["topk"]))
    if tuple(adjacency.shape) != expected_shape:
        raise ValueError(
            f"Cached adjacency shape mismatch: expected {expected_shape}, found {tuple(adjacency.shape)}."
        )

    if adjacency[0].any():
        raise ValueError("Cached adjacency row 0 must be zero for padding.")
    mask_id = int(expected_metadata["original_pool_size"]) + 1
    if mask_id < adjacency.shape[0] and adjacency[mask_id].any():
        raise ValueError("Cached adjacency mask row must be zero.")

    candidate_id_set = set(int(item_id) for item_id in candidate_item_ids.tolist())
    for item_id in candidate_item_ids[: min(32, len(candidate_item_ids))]:
        row = adjacency[int(item_id)].tolist()
        if not row or row[0] != int(item_id):
            raise ValueError(f"Cached adjacency row {int(item_id)} does not start with itself.")
        if any(neighbor_id not in candidate_id_set for neighbor_id in row):
            raise ValueError(f"Cached adjacency row {int(item_id)} contains invalid candidate IDs.")


def _build_source_membership(
    candidate_item_ids: np.ndarray,
    expanded_to_source: np.ndarray,
    original_pool_size: int,
) -> list[list[int]]:
    source_members: list[list[int]] = [[] for _ in range(original_pool_size + 1)]
    for item_id in candidate_item_ids.tolist():
        source_id = int(expanded_to_source[int(item_id)])
        if source_id <= 0:
            continue
        source_members[source_id].append(int(item_id))
    return source_members


def _candidate_item_ids(item_size: int, mask_id: int) -> np.ndarray:
    item_ids = np.arange(1, item_size, dtype=np.int64)
    return item_ids[item_ids != mask_id]


def _enforce_self_neighbors(
    search_result_ids: torch.Tensor,
    batch_item_ids: np.ndarray,
    topk: int,
) -> torch.Tensor:
    adjusted = torch.zeros((len(batch_item_ids), topk), dtype=torch.long)
    rows = search_result_ids.detach().cpu().tolist()
    for row_index, item_id in enumerate(batch_item_ids.tolist()):
        row = [int(item_id)]
        for neighbor_id in rows[row_index]:
            neighbor = int(neighbor_id)
            if neighbor <= 0 or neighbor == item_id or neighbor in row:
                continue
            row.append(neighbor)
            if len(row) == topk:
                break
        while len(row) < topk:
            row.append(int(item_id))
        adjusted[row_index] = torch.tensor(row, dtype=torch.long)
    return adjusted


def _build_source_adjacency(
    model: Any,
    original_pool_size: int,
    topk: int,
    block_size: int,
) -> torch.Tensor:
    item_vectors = model.item_embeddings.weight.detach().float().cpu()[1 : original_pool_size + 1]
    item_vectors = torch.nn.functional.normalize(item_vectors, dim=-1)
    adjacency = torch.zeros((original_pool_size + 2, topk), dtype=torch.long)

    for start in range(0, original_pool_size, block_size):
        end = min(start + block_size, original_pool_size)
        block = item_vectors[start:end]
        similarities = torch.matmul(block, item_vectors.transpose(0, 1))
        raw_neighbor_ids = torch.topk(similarities, k=min(topk, original_pool_size), dim=-1).indices + 1
        batch_item_ids = np.arange(start + 1, end + 1, dtype=np.int64)
        adjusted = _enforce_self_neighbors(raw_neighbor_ids, batch_item_ids, topk=topk)
        adjacency[start + 1 : end + 1] = adjusted

    adjacency[0] = 0
    adjacency[original_pool_size + 1] = 0
    return adjacency


def _expand_source_adjacency(
    source_adjacency: torch.Tensor,
    candidate_item_ids: np.ndarray,
    expanded_to_source: np.ndarray,
    original_pool_size: int,
    topk: int,
) -> torch.Tensor:
    item_size = int(expanded_to_source.shape[0])
    adjacency = torch.zeros((item_size, topk), dtype=torch.long)
    source_members = _build_source_membership(candidate_item_ids, expanded_to_source, original_pool_size)

    for item_id in candidate_item_ids.tolist():
        source_id = int(expanded_to_source[item_id])
        row: list[int] = [int(item_id)]
        row_set = {int(item_id)}

        for sibling_id in source_members[source_id]:
            if sibling_id in row_set:
                continue
            row.append(int(sibling_id))
            row_set.add(int(sibling_id))
            if len(row) == topk:
                break

        if len(row) < topk:
            for neighbor_source_id in source_adjacency[source_id].tolist():
                if neighbor_source_id <= 0:
                    continue
                for neighbor_item_id in source_members[int(neighbor_source_id)]:
                    if neighbor_item_id in row_set:
                        continue
                    row.append(int(neighbor_item_id))
                    row_set.add(int(neighbor_item_id))
                    if len(row) == topk:
                        break
                if len(row) == topk:
                    break

        while len(row) < topk:
            row.append(int(item_id))
        adjacency[item_id] = torch.tensor(row, dtype=torch.long)

    adjacency[0] = 0
    mask_id = original_pool_size + 1
    if mask_id < adjacency.shape[0]:
        adjacency[mask_id] = 0
    return adjacency


def build_sparse_adjacency(
    model: Any,
    original_pool_size: int,
    expanded_to_source: np.ndarray,
    topk: int,
    config: dict[str, Any],
) -> torch.Tensor:
    block_size = _graph_build_block_size(config)
    source_adjacency = _build_source_adjacency(
        model=model,
        original_pool_size=original_pool_size,
        topk=topk,
        block_size=block_size,
    )
    candidate_item_ids = _candidate_item_ids(model.item_embeddings.num_embeddings, mask_id=original_pool_size + 1)
    return _expand_source_adjacency(
        source_adjacency=source_adjacency,
        candidate_item_ids=candidate_item_ids,
        expanded_to_source=expanded_to_source,
        original_pool_size=original_pool_size,
        topk=topk,
    )


def build_or_load_adjacency(
    model: Any,
    checkpoint_path: Path,
    config: dict[str, Any],
    pool_size: int,
    original_pool_size: int,
    expanded_to_source: np.ndarray,
    force_rebuild: bool = False,
) -> tuple[torch.Tensor, GraphBuildRecord]:
    topk = _graph_topk(config)
    expected_metadata = _expected_cache_metadata(
        checkpoint_path=checkpoint_path,
        config=config,
        pool_size=pool_size,
        topk=topk,
        original_pool_size=original_pool_size,
    )
    cache_id, adjacency_path, metadata_path = _graph_cache_paths(
        checkpoint_path=checkpoint_path,
        config=config,
        pool_size=pool_size,
        topk=topk,
    )
    adjacency_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_item_ids = _candidate_item_ids(model.item_embeddings.num_embeddings, mask_id=original_pool_size + 1)

    if adjacency_path.is_file() and metadata_path.is_file() and not force_rebuild:
        adjacency = torch.load(adjacency_path, map_location="cpu")
        metadata = json.loads(metadata_path.read_text())
        _validate_cached_adjacency(
            adjacency=adjacency,
            metadata=metadata,
            expected_metadata=expected_metadata,
            candidate_item_ids=candidate_item_ids,
        )
        return adjacency, GraphBuildRecord(
            cache_id=cache_id,
            adjacency_path=str(adjacency_path),
            metadata_path=str(metadata_path),
            pool_size=pool_size,
            topk=topk,
            build_block_size=_graph_build_block_size(config),
            build_seconds=float(metadata["build_seconds"]),
            loaded_from_cache=True,
            checkpoint_signature=metadata["checkpoint_signature"],
        )

    start_time = time.perf_counter()
    adjacency = build_sparse_adjacency(
        model=model,
        original_pool_size=original_pool_size,
        expanded_to_source=expanded_to_source,
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
        pool_size=pool_size,
        topk=topk,
        build_block_size=_graph_build_block_size(config),
        build_seconds=build_seconds,
        loaded_from_cache=False,
        checkpoint_signature=metadata["checkpoint_signature"],
    )


def _candidate_tensor(
    item_size: int,
    mask_id: int,
    device: torch.device,
) -> torch.Tensor:
    item_ids = torch.arange(1, item_size, dtype=torch.long, device=device)
    return item_ids[item_ids != mask_id]


def _filter_neighbors(
    neighbors: torch.Tensor,
    seen_item_ids: np.ndarray,
    mask_id: int,
) -> torch.Tensor:
    if neighbors.numel() == 0:
        return neighbors
    filtered = neighbors[neighbors > 0]
    if filtered.numel() == 0:
        return filtered
    filtered = filtered[filtered != mask_id]
    if filtered.numel() == 0:
        return filtered
    if seen_item_ids.size:
        seen_tensor = torch.as_tensor(seen_item_ids, dtype=torch.long, device=filtered.device)
        keep_mask = ~torch.isin(filtered, seen_tensor)
        filtered = filtered[keep_mask]
    return filtered


def graph_propagation(
    user_vectors: torch.Tensor,
    item_embeddings: torch.Tensor,
    adjacency: torch.Tensor,
    num_beams: int,
    propagation_steps: int,
    n_return_sequences: int,
    mask_id: int,
    seen_item_ids_per_user: list[np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = user_vectors.shape[0]
    device = user_vectors.device
    candidate_ids = _candidate_tensor(item_embeddings.shape[0], mask_id=mask_id, device=device)
    sample_indices = torch.randint(
        low=0,
        high=candidate_ids.shape[0],
        size=(batch_size, num_beams),
        device=device,
    )
    topk_nodes_sorted = candidate_ids[sample_indices]

    visited_nodes: dict[int, set[int]] = {}
    for batch_id in range(batch_size):
        visited_nodes[batch_id] = set(int(node) for node in topk_nodes_sorted[batch_id].detach().cpu().tolist())

    for _ in range(propagation_steps):
        all_neighbors = adjacency[topk_nodes_sorted].reshape(batch_size, -1)
        next_nodes: list[torch.Tensor] = []
        for batch_id in range(batch_size):
            neighbors_in_batch = torch.unique(all_neighbors[batch_id])
            for node in neighbors_in_batch.detach().cpu().tolist():
                if int(node) > 0:
                    visited_nodes[batch_id].add(int(node))

            filtered_neighbors = _filter_neighbors(
                neighbors=neighbors_in_batch,
                seen_item_ids=seen_item_ids_per_user[batch_id],
                mask_id=mask_id,
            )
            if filtered_neighbors.numel() == 0:
                filtered_neighbors = _filter_neighbors(
                    neighbors=topk_nodes_sorted[batch_id],
                    seen_item_ids=seen_item_ids_per_user[batch_id],
                    mask_id=mask_id,
                )
            if filtered_neighbors.numel() == 0:
                filtered_neighbors = topk_nodes_sorted[batch_id][:1]

            candidate_embeddings = item_embeddings[filtered_neighbors]
            scores = torch.matmul(candidate_embeddings, user_vectors[batch_id])
            beam_width = min(num_beams, filtered_neighbors.shape[0])
            idxs = torch.topk(scores, beam_width).indices
            selected = filtered_neighbors[idxs]
            if selected.shape[0] < num_beams:
                pad = selected[0].repeat(num_beams - selected.shape[0])
                selected = torch.cat([selected, pad], dim=0)
            next_nodes.append(selected)
        topk_nodes_sorted = torch.stack(next_nodes, dim=0)

    visited_counts = torch.tensor(
        [[float(len(visited_nodes[batch_id]))] for batch_id in range(batch_size)],
        dtype=torch.float32,
        device=device,
    )
    return topk_nodes_sorted[:, :n_return_sequences], visited_counts
