"""Representation helpers for lightweight RPG candidate reranking."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def token_mean_item_embeddings(model: Any) -> torch.Tensor:
    """Build one item vector by averaging each item's semantic-ID token vectors.

    RPG does not keep a separate learned item-embedding table. Its input item
    representation is the mean of the semantic-ID token embeddings, so this
    helper reconstructs that same representation for every item id. The padding
    row ``0`` is zeroed explicitly because graph/reranking candidates should be
    real item ids only.
    """

    item_tokens = model.item_id2tokens.to(model.gpt2.wte.weight.device)
    with torch.no_grad():
        item_embeddings = model.gpt2.wte(item_tokens).mean(dim=-2)
        item_embeddings = F.normalize(item_embeddings, dim=-1)
    item_embeddings[0].zero_()
    return item_embeddings.detach()


def normalized_user_hidden(user_hidden: torch.Tensor) -> torch.Tensor:
    """Normalize last-position GPT user states for cosine-style scoring."""

    return F.normalize(user_hidden, dim=-1)
