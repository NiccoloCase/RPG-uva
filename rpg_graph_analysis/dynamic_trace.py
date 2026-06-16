"""Traced RPG graph decoding.

The functions in this module mirror the upstream RPG ``generate`` and
``graph_propagation`` logic, but return enough trace data for dynamic graph
analysis. They are repo-owned wrappers; the ``third_party`` model is not
modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class BatchTrace:
    """Step-wise trace data for one decoded batch.

    Every field is indexed first by batch row and then by decoding step. Step
    ``0`` is the random initial beam set sampled by RPG before any graph
    expansion. Steps ``1..propagation_steps`` correspond to the neighbor sets
    observed during graph propagation.

    ``frontier_by_step`` stores the selected beam after scoring for each step.
    ``unique_candidates_by_step`` stores the unique candidates considered before
    scoring at each propagation step. ``new_items_by_step`` stores only items
    that were not already in the cumulative visited set before that step.
    """

    initial_items: list[list[int]]
    frontier_by_step: list[list[list[int]]]
    unique_candidates_by_step: list[list[list[int]]]
    new_items_by_step: list[list[list[int]]]
    final_visited_items: list[list[int]]
    visited_count_by_step: list[list[int]]
    raw_candidate_count_by_step: list[list[int]]
    unique_candidate_count_by_step: list[list[int]]
    new_item_count_by_step: list[list[int]]
    duplicate_candidate_ratio_by_step: list[list[float]]
    novelty_ratio_by_step: list[list[float]]


@dataclass
class DecodingContext:
    """Model states needed by traced decoding and optional downstream rerankers.

    ``token_logits`` is exactly the tensor used by upstream RPG graph
    propagation. ``user_hidden`` is the raw GPT hidden state at the last real
    input item in each sequence. Keeping both in one object lets intervention
    experiments reuse the same forward pass instead of recomputing the model.
    """

    token_logits: torch.Tensor
    user_hidden: torch.Tensor


def compute_decoding_context(model: Any, batch: dict[str, torch.Tensor]) -> DecodingContext:
    """Compute RPG token log-probabilities and the final user hidden state.

    This function intentionally duplicates the upstream sequence-state gather,
    token-embedding normalization, temperature scaling, and per-head log-softmax
    logic. Keeping this separate from tracing makes parity failures easier to
    localize: if predictions differ from upstream, either this logit path or the
    propagation path changed.
    """

    outputs = model.forward(batch, return_loss=False)
    user_hidden = outputs.last_hidden_state.gather(
        dim=1,
        index=(batch["seq_lens"] - 1)
        .view(-1, 1, 1)
        .expand(-1, 1, model.config["n_embd"]),
    ).squeeze(1)
    states = outputs.final_states.gather(
        dim=1,
        index=(batch["seq_lens"] - 1)
        .view(-1, 1, 1, 1)
        .expand(-1, 1, model.n_pred_head, model.config["n_embd"]),
    )
    states = F.normalize(states, dim=-1)

    token_emb = model.gpt2.wte.weight[1:-1]
    token_emb = F.normalize(token_emb, dim=-1)
    token_embs = torch.chunk(token_emb, model.n_pred_head, dim=0)
    logits = [
        torch.matmul(states[:, 0, index, :], token_embs[index].T) / model.temperature
        for index in range(model.n_pred_head)
    ]
    logits = [F.log_softmax(logit, dim=-1) for logit in logits]
    return DecodingContext(
        token_logits=torch.cat(logits, dim=-1),
        user_hidden=user_hidden,
    )


def _compute_token_logits(model: Any, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute token log-probabilities exactly as upstream ``RPG.generate``."""

    return compute_decoding_context(model, batch).token_logits


def traced_graph_propagation(
    model: Any,
    token_logits: torch.Tensor,
    n_return_sequences: int,
) -> tuple[torch.Tensor, torch.Tensor, BatchTrace]:
    """Run graph propagation and collect per-step trace data.

    This intentionally follows the upstream implementation line by line for
    random initialization, neighbor expansion, scoring, and beam pruning.
    Additional bookkeeping is done with Python sets after each operation.

    Args:
        model: Loaded RPG model with ``adjacency``, ``item_id2tokens``,
            ``num_beams``, and ``propagation_steps`` already configured.
        token_logits: Concatenated per-digit item-token log-probabilities with
            the same layout expected by upstream ``graph_propagation``.
        n_return_sequences: Number of final recommendations to return.

    Returns:
        ``(predictions, visited_counts, trace)``. The first two entries match
        upstream graph decoding: predictions have shape
        ``[batch, n_return_sequences, 1]`` and visited counts have shape
        ``[batch, 1]``. The trace is extra analysis metadata and must not affect
        beam selection.
    """

    batch_size = token_logits.shape[0]
    visited_nodes: dict[int, set[int]] = {batch_id: set() for batch_id in range(batch_size)}

    topk_nodes_sorted = torch.randint(
        1,
        model.dataset.n_items,
        (batch_size, model.num_beams),
        dtype=torch.long,
        device=token_logits.device,
    )

    initial_items: list[list[int]] = []
    frontier_by_step: list[list[list[int]]] = [[] for _ in range(batch_size)]
    unique_candidates_by_step: list[list[list[int]]] = [[] for _ in range(batch_size)]
    new_items_by_step: list[list[list[int]]] = [[] for _ in range(batch_size)]
    visited_count_by_step: list[list[int]] = [[] for _ in range(batch_size)]
    raw_candidate_count_by_step: list[list[int]] = [[] for _ in range(batch_size)]
    unique_candidate_count_by_step: list[list[int]] = [[] for _ in range(batch_size)]
    new_item_count_by_step: list[list[int]] = [[] for _ in range(batch_size)]
    duplicate_candidate_ratio_by_step: list[list[float]] = [[] for _ in range(batch_size)]
    novelty_ratio_by_step: list[list[float]] = [[] for _ in range(batch_size)]

    for batch_id in range(batch_size):
        initial = topk_nodes_sorted[batch_id].detach().cpu().numpy().tolist()
        initial_items.append([int(node) for node in initial])
        previous = set(visited_nodes[batch_id])
        for node in initial:
            visited_nodes[batch_id].add(int(node))
        new_items = sorted(visited_nodes[batch_id] - previous)
        unique_count = len(set(initial))
        raw_count = len(initial)
        frontier_by_step[batch_id].append([int(node) for node in initial])
        unique_candidates_by_step[batch_id].append(sorted({int(node) for node in initial}))
        new_items_by_step[batch_id].append(new_items)
        visited_count_by_step[batch_id].append(len(visited_nodes[batch_id]))
        raw_candidate_count_by_step[batch_id].append(raw_count)
        unique_candidate_count_by_step[batch_id].append(unique_count)
        new_item_count_by_step[batch_id].append(len(new_items))
        duplicate_candidate_ratio_by_step[batch_id].append(
            1.0 - (unique_count / raw_count if raw_count else 0.0)
        )
        novelty_ratio_by_step[batch_id].append(len(new_items) / unique_count if unique_count else 0.0)

    for _ in range(model.propagation_steps):
        all_neighbors = model.adjacency[topk_nodes_sorted].view(batch_size, -1)

        next_nodes = []
        for batch_id in range(batch_size):
            raw_neighbors = all_neighbors[batch_id].detach().cpu().numpy().tolist()
            neighbors_in_batch = torch.unique(all_neighbors[batch_id])
            unique_neighbors = neighbors_in_batch.detach().cpu().numpy().tolist()

            previous = set(visited_nodes[batch_id])
            for node in unique_neighbors:
                visited_nodes[batch_id].add(int(node))
            new_items = sorted(visited_nodes[batch_id] - previous)

            scores = torch.gather(
                input=token_logits[batch_id].unsqueeze(0).expand(neighbors_in_batch.shape[0], -1),
                dim=-1,
                index=(model.item_id2tokens[neighbors_in_batch] - 1),
            ).mean(dim=-1)

            idxs = torch.topk(scores, model.num_beams).indices
            selected_next = neighbors_in_batch[idxs]
            next_nodes.append(selected_next)

            raw_count = len(raw_neighbors)
            unique_count = len(unique_neighbors)
            frontier_by_step[batch_id].append(
                [int(node) for node in selected_next.detach().cpu().numpy().tolist()]
            )
            unique_candidates_by_step[batch_id].append([int(node) for node in unique_neighbors])
            new_items_by_step[batch_id].append(new_items)
            visited_count_by_step[batch_id].append(len(visited_nodes[batch_id]))
            raw_candidate_count_by_step[batch_id].append(raw_count)
            unique_candidate_count_by_step[batch_id].append(unique_count)
            new_item_count_by_step[batch_id].append(len(new_items))
            duplicate_candidate_ratio_by_step[batch_id].append(
                1.0 - (unique_count / raw_count if raw_count else 0.0)
            )
            novelty_ratio_by_step[batch_id].append(
                len(new_items) / unique_count if unique_count else 0.0
            )

        topk_nodes_sorted = torch.stack(next_nodes, dim=0)

    visited_counts = torch.FloatTensor(
        [[len(visited_nodes[batch_id])] for batch_id in range(batch_size)]
    )
    trace = BatchTrace(
        initial_items=initial_items,
        frontier_by_step=frontier_by_step,
        unique_candidates_by_step=unique_candidates_by_step,
        new_items_by_step=new_items_by_step,
        final_visited_items=[sorted(visited_nodes[batch_id]) for batch_id in range(batch_size)],
        visited_count_by_step=visited_count_by_step,
        raw_candidate_count_by_step=raw_candidate_count_by_step,
        unique_candidate_count_by_step=unique_candidate_count_by_step,
        new_item_count_by_step=new_item_count_by_step,
        duplicate_candidate_ratio_by_step=duplicate_candidate_ratio_by_step,
        novelty_ratio_by_step=novelty_ratio_by_step,
    )
    return topk_nodes_sorted[:, :n_return_sequences].unsqueeze(-1), visited_counts, trace


def traced_generate(
    model: Any,
    batch: dict[str, torch.Tensor],
    n_return_sequences: int,
) -> tuple[torch.Tensor, torch.Tensor, BatchTrace]:
    """Generate graph-decoding recommendations plus trace metadata.

    This is the dynamic-analysis equivalent of upstream ``RPG.generate`` when
    ``generate_w_decoding_graph`` is enabled. It does not initialize or build a
    graph; callers must attach the desired adjacency to ``model.adjacency``
    before calling this function.
    """

    token_logits = _compute_token_logits(model, batch)
    return traced_graph_propagation(
        model=model,
        token_logits=token_logits,
        n_return_sequences=n_return_sequences,
    )
