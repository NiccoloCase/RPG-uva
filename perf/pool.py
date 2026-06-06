from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DUMMY_ITEM_PREFIX = "__dummy_candidate__"


@dataclass
class PoolAugmentationResult:
    original_pool_size: int
    target_pool_size: int
    added_items: int
    seed: int
    source_cycle_offset: int


def augment_candidate_pool(
    dataset: Any,
    tokenizer: Any,
    model: Any,
    target_pool_size: int,
    seed: int,
) -> PoolAugmentationResult:
    """Expand the candidate item pool by cloning existing items as dummies.

    The released RPG checkpoint was trained on the original dataset size, so
    the perf tooling cannot simply invent brand-new items with unseen semantic
    IDs. Instead, it duplicates existing items under synthetic item names and
    reuses their semantic-token codes. This preserves the model's assumptions
    while allowing inference to be stress-tested on larger candidate pools.

    Args:
        dataset: Loaded dataset instance from the upstream RPG code.
        tokenizer: Tokenizer instance that owns the `item2tokens` mapping.
        model: RPG model instance whose `item_id2tokens` lookup table must stay
            synchronized with the dataset/tokenizer state.
        target_pool_size: Desired number of non-padding items after
            augmentation.
        seed: Seed used to choose where the source-item cycling starts.

    Returns:
        A `PoolAugmentationResult` describing the original pool size, requested
        pool size, number of added dummy items, and deterministic cycling
        details.

    Raises:
        ValueError: If the requested target pool is smaller than the original
            pool or if a dummy item name collision occurs.
        RuntimeError: If the dataset size after augmentation does not match the
            expected target.
    """
    original_pool_size = dataset.n_items - 1
    if target_pool_size < original_pool_size:
        raise ValueError(
            f"Target pool size {target_pool_size} is smaller than the original pool "
            f"size {original_pool_size}."
        )
    if target_pool_size == original_pool_size:
        model.item_id2tokens = model._map_item_tokens().to(model.config["device"])
        model.generate_w_decoding_graph = True
        model.init_flag = False
        return PoolAugmentationResult(
            original_pool_size=original_pool_size,
            target_pool_size=target_pool_size,
            added_items=0,
            seed=seed,
            source_cycle_offset=seed % max(original_pool_size, 1),
        )

    existing_item_names = dataset.id_mapping["id2item"][1 : original_pool_size + 1]
    cycle_offset = seed % len(existing_item_names)
    items_to_add = target_pool_size - original_pool_size

    for offset in range(items_to_add):
        source_name = existing_item_names[(cycle_offset + offset) % len(existing_item_names)]
        dummy_name = f"{DUMMY_ITEM_PREFIX}{target_pool_size:08d}_{offset + 1:08d}"
        if dummy_name in dataset.item2id:
            raise ValueError(f"Dummy item collision detected: {dummy_name}")

        dataset.item2id[dummy_name] = len(dataset.id_mapping["id2item"])
        dataset.id_mapping["id2item"].append(dummy_name)
        tokenizer.item2tokens[dummy_name] = tuple(tokenizer.item2tokens[source_name])

        if dataset.item2meta is not None and source_name in dataset.item2meta:
            dataset.item2meta[dummy_name] = dataset.item2meta[source_name]

    model.item_id2tokens = model._map_item_tokens().to(model.config["device"])
    model.generate_w_decoding_graph = True
    model.init_flag = False
    if hasattr(model, "adjacency"):
        model.adjacency = None

    expected_size = original_pool_size + items_to_add
    if dataset.n_items - 1 != expected_size:
        raise RuntimeError(
            f"Candidate-pool augmentation failed: expected {expected_size} items, "
            f"found {dataset.n_items - 1}."
        )

    return PoolAugmentationResult(
        original_pool_size=original_pool_size,
        target_pool_size=target_pool_size,
        added_items=items_to_add,
        seed=seed,
        source_cycle_offset=cycle_offset,
    )
