"""Candidate pools and RPG's item score, shared by the decoding analyses."""

from __future__ import annotations

import torch

from .dynamic_trace import BatchTrace


def padded_visited_candidates(trace: BatchTrace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return padded final visited candidates and a validity mask.

    ``BatchTrace`` stores visited candidates as ragged Python lists. Scoring is
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


def rpg_candidate_scores(
    token_logits: torch.Tensor,
    item_id2tokens: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> torch.Tensor:
    """Score candidate items with RPG's own item score.

    This reproduces the item scoring rule used inside RPG graph propagation:
    gather one log-probability per semantic-ID digit and average over digits.
    """

    batch_size, n_candidates = candidate_ids.shape
    candidate_tokens = item_id2tokens[candidate_ids] - 1
    expanded_logits = token_logits.unsqueeze(1).expand(-1, n_candidates, -1)
    return torch.gather(expanded_logits, dim=-1, index=candidate_tokens).mean(dim=-1)
