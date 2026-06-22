"""Candidate-pool construction for RPG reranking experiments."""

from __future__ import annotations

from typing import Literal

import torch

from rpg_graph_analysis.dynamic_trace import BatchTrace
from rpg_graph_analysis.reranking.scorers import rpg_candidate_scores

PoolSize = int | Literal["all"]


def parse_pool_sizes(raw_values: list[object]) -> list[PoolSize]:
    """Parse configured candidate-pool sizes.

    Integer values keep the top-``M`` visited candidates by RPG token score.
    The string ``"all"`` keeps the complete visited set. Values are deduped
    while preserving the conventional order: numeric pools ascending, then
    ``"all"`` if present.
    """

    numeric: set[int] = set()
    include_all = False
    for value in raw_values:
        if isinstance(value, str) and value.lower() == "all":
            include_all = True
            continue
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"Candidate pool sizes must be positive or 'all', got {value!r}.")
        numeric.add(parsed)
    ordered: list[PoolSize] = sorted(numeric)
    if include_all:
        ordered.append("all")
    if not ordered:
        raise ValueError("At least one reranking candidate pool size is required.")
    return ordered


def padded_visited_candidates(trace: BatchTrace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return padded final visited candidates and a validity mask.

    ``BatchTrace`` stores visited candidates as ragged Python lists. Reranking is
    easier and faster when these are padded into tensors. Padding uses item id
    ``0`` and is masked out by downstream scoring.
    """

    max_candidates = max(len(items) for items in trace.final_visited_items)
    candidates = torch.zeros(
        (len(trace.final_visited_items), max_candidates),
        dtype=torch.long,
        device=device,
    )
    mask = torch.zeros_like(candidates, dtype=torch.bool)
    for row_index, items in enumerate(trace.final_visited_items):
        real_items = [int(item) for item in items if int(item) > 0]
        if not real_items:
            continue
        row = torch.tensor(real_items, dtype=torch.long, device=device)
        candidates[row_index, : row.numel()] = row
        mask[row_index, : row.numel()] = True
    return candidates, mask


def select_candidate_pool(
    *,
    candidate_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    token_logits: torch.Tensor,
    item_id2tokens: torch.Tensor,
    pool_size: PoolSize,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select a bounded pool from visited candidates using RPG token scores.

    The complete visited set can be large and ragged. For numeric pool sizes,
    this keeps each row's best candidates under the current RPG semantic-token
    score before the separate reranker is applied. This keeps the intervention
    cheap while preserving the original traversal/candidate distribution.
    """

    if pool_size == "all":
        return candidate_ids, valid_mask

    score_ids = candidate_ids.masked_fill(~valid_mask, 1)
    scores = rpg_candidate_scores(token_logits, item_id2tokens, score_ids)
    scores = scores.masked_fill(~valid_mask, float("-inf"))
    keep = min(int(pool_size), candidate_ids.shape[1])
    top_indices = torch.topk(scores, k=keep, dim=1).indices
    selected_ids = torch.gather(candidate_ids, dim=1, index=top_indices)
    selected_mask = torch.gather(valid_mask, dim=1, index=top_indices)
    return selected_ids, selected_mask

