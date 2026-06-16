"""Scoring functions used by reranking interventions."""

from __future__ import annotations

import torch


def masked_rowwise_zscore(scores: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Normalize scores within each user's valid candidate pool.

    RPG token log-scores and representation dot-products live on different
    scales. Z-scoring per row makes a simple hybrid score meaningful without
    introducing global calibration assumptions.
    """

    valid = valid_mask.to(scores.dtype)
    count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    safe_scores = scores.masked_fill(~valid_mask, 0.0)
    mean = safe_scores.sum(dim=1, keepdim=True) / count
    centered = (scores - mean).masked_fill(~valid_mask, 0.0)
    variance = (centered.square().sum(dim=1, keepdim=True) / count).clamp_min(1e-12)
    normalized = (scores - mean) / torch.sqrt(variance)
    return normalized.masked_fill(~valid_mask, float("-inf"))


def score_token_mean_dot(
    user_hidden: torch.Tensor,
    item_embeddings: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> torch.Tensor:
    """Score candidates by dot product with token-mean item embeddings.

    Args:
        user_hidden: Normalized user vectors with shape ``[batch, hidden]``.
        item_embeddings: Normalized item vectors indexed by item id, with shape
            ``[n_items, hidden]``.
        candidate_ids: Candidate item ids with shape ``[batch, n_candidates]``.

    Returns:
        Candidate scores with shape ``[batch, n_candidates]``. Because both
        sides are normalized by the caller, the dot product is cosine
        similarity.
    """

    candidate_embeddings = item_embeddings[candidate_ids]
    return torch.einsum("bd,bnd->bn", user_hidden, candidate_embeddings)


def rpg_candidate_scores(
    token_logits: torch.Tensor,
    item_id2tokens: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> torch.Tensor:
    """Score candidate items with RPG's existing semantic-token log score.

    This reproduces the item scoring rule used inside RPG graph propagation:
    gather one log-probability per semantic-ID digit and average over digits.
    It is used only to form bounded candidate pools before applying a separate
    reranker.
    """

    batch_size, n_candidates = candidate_ids.shape
    candidate_tokens = item_id2tokens[candidate_ids] - 1
    expanded_logits = token_logits.unsqueeze(1).expand(-1, n_candidates, -1)
    return torch.gather(expanded_logits, dim=-1, index=candidate_tokens).mean(dim=-1)


def hybrid_rpg_dot_scores(
    *,
    rpg_scores: torch.Tensor,
    dot_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Blend normalized RPG token scores and token-mean-dot scores.

    ``alpha=1`` recovers RPG-score reranking over the candidate pool, while
    ``alpha=0`` recovers the pure dot-product reranker. Intermediate values test
    whether the representation score can help without discarding RPG's own
    semantic-token probability.
    """

    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"Hybrid alpha must be in [0, 1], got {alpha}.")
    rpg_norm = masked_rowwise_zscore(rpg_scores, valid_mask)
    dot_norm = masked_rowwise_zscore(dot_scores, valid_mask)
    return (alpha * rpg_norm + (1.0 - alpha) * dot_norm).masked_fill(
        ~valid_mask,
        float("-inf"),
    )
