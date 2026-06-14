from __future__ import annotations

"""Small semantic-ID quantizers for the RPG tokenizer ablation.

The upstream RPG tokenizer uses FAISS OPQ and then offsets each semantic digit
into a separate token range. This module keeps that downstream contract intact:
every quantizer returns a dense integer array with shape
`(n_items, n_codebook)`, where each value is in `[0, codebook_size - 1]`.

Keeping the contract this small is intentional. It lets us compare tokenizer
choices without changing RPG's MTP loss, per-digit prediction heads, graph
decoding, or evaluator.
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def generate_codes(
    method: str,
    sent_embs: np.ndarray,
    train_mask: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """Generate un-offset semantic ID digits for all non-padding items.

    Args:
        method: Quantizer name from `semantic_id_method`; currently `fsq`,
            `fsq_quantile`, or `pq`.
        sent_embs: Item metadata embeddings after any tokenizer-level PCA. Row
            `i` corresponds to item id `i + 1` in the upstream dataset mapping.
        train_mask: Boolean mask over `sent_embs` selecting items seen in the
            training prefixes. Quantizers fit only on these items to match the
            upstream OPQ protocol.
        config: Merged GenRec config. Uses `n_codebook`, `codebook_size`,
            `rand_seed`, and FAISS thread settings where relevant.

    Returns:
        Integer semantic IDs with shape `(len(sent_embs), n_codebook)`. Values
        are not offset into RPG token ids yet; the tokenizer applies the offset
        later using the upstream `_sem_ids_to_tokens` helper.
    """
    method = method.lower()
    if method == "fsq":
        return _generate_fsq_codes(sent_embs, train_mask, config)
    if method == "fsq_quantile":
        return _generate_fsq_quantile_codes(sent_embs, train_mask, config)
    if method == "pq":
        return _generate_pq_codes(sent_embs, train_mask, config)
    raise ValueError(f"Unsupported semantic_id_method: {method}")


def write_stats(path: str | Path, codes: np.ndarray, config: dict[str, Any]) -> None:
    """Write a simple sidecar JSON next to a `.sem_ids` cache.

    The stats are meant for quick experiment diagnostics: they make it easy to
    see whether a tokenizer collapsed, left many codes unused, or produced many
    full-ID collisions before spending GPU hours on training.
    """
    stats_path = Path(f"{path}.stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(build_stats(codes, config), indent=2))


def build_stats(codes: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    """Summarize code utilization and collision behavior.

    `collision_rate` is the fraction of items that do not have a unique full
    semantic ID. Per-digit utilization and entropy measure whether individual
    prediction heads receive reasonably balanced classification targets.
    """
    codebook_size = int(config["codebook_size"])
    n_items = int(codes.shape[0])
    unique_full_codes = len({tuple(row.tolist()) for row in codes})
    collision_rate = 0.0 if n_items == 0 else 1.0 - (unique_full_codes / n_items)

    per_digit_utilization = []
    per_digit_entropy = []
    per_digit_max_bucket = []
    for digit in range(codes.shape[1]):
        counts = np.bincount(codes[:, digit], minlength=codebook_size).astype(np.float64)
        used = int(np.count_nonzero(counts))
        per_digit_utilization.append(used / codebook_size)
        per_digit_max_bucket.append(int(counts.max()) if counts.size else 0)

        probs = counts[counts > 0] / max(float(counts.sum()), 1.0)
        entropy = float(-(probs * np.log2(probs)).sum()) if probs.size else 0.0
        per_digit_entropy.append(entropy)

    return {
        "semantic_id_method": str(config.get("semantic_id_method", "")),
        "n_items": n_items,
        "n_codebook": int(config["n_codebook"]),
        "codebook_size": codebook_size,
        "unique_full_codes": unique_full_codes,
        "collision_rate": collision_rate,
        "per_digit_utilization": per_digit_utilization,
        "per_digit_entropy": per_digit_entropy,
        "per_digit_max_bucket": per_digit_max_bucket,
    }


def _generate_fsq_codes(
    sent_embs: np.ndarray,
    train_mask: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """Build library-backed, factorized FSQ codes.

    FSQ discretizes each scalar coordinate into a fixed set of finite levels by
    bounding and rounding the coordinate. Unlike `fsq_quantile`, the bin edges
    are not fitted from empirical quantiles. The only fitted part here is the
    deterministic PCA projection that reduces item embeddings to one scalar per
    RPG digit before the library FSQ quantizer is applied.

    The FSQ paper often packs several scalar levels into one mixed-radix code
    and is commonly used inside a trained autoencoder. This ablation does not
    train an encoder/decoder. RPG needs independent per-digit classification
    labels, so we apply a one-dimensional FSQ quantizer to every PCA coordinate
    and keep the resulting per-coordinate level IDs. This avoids constructing an
    enormous mixed-radix code space such as `256 ** 16`, while still using the
    library's FSQ bounding-and-rounding operation.
    """
    import torch
    from sklearn.decomposition import PCA
    from vector_quantize_pytorch import FSQ

    n_digit = int(config["n_codebook"])
    codebook_size = int(config["codebook_size"])
    train_embs = sent_embs[train_mask]
    if min(train_embs.shape) < n_digit:
        raise ValueError(
            f"FSQ PCA needs at least n_codebook={n_digit} training items and "
            f"features, got shape {train_embs.shape}."
        )
    if codebook_size <= 1:
        raise ValueError(f"FSQ needs codebook_size > 1, got {codebook_size}.")

    pca = PCA(n_components=n_digit, whiten=True, random_state=int(config["rand_seed"]))
    pca.fit(train_embs)
    projected = pca.transform(sent_embs).astype(np.float32)

    quantizer = FSQ(
        levels=[codebook_size],
        dim=1,
        return_indices=False,
        preserve_symmetry=codebook_size == 2,
    )
    quantizer.eval()

    with torch.no_grad():
        projected_tensor = torch.from_numpy(projected.reshape(1, -1, 1))
        quantized_tensor, _ = quantizer(projected_tensor)
        level_tensor = _fsq_level_indices(
            quantized_tensor,
            codebook_size=codebook_size,
            preserve_symmetry=codebook_size == 2,
        )

    codes = level_tensor.reshape(sent_embs.shape[0], n_digit).cpu().numpy().astype(np.int64)
    return np.clip(codes, 0, codebook_size - 1)


def _fsq_level_indices(quantized_tensor, codebook_size: int, preserve_symmetry: bool):
    """Convert normalized FSQ scalar outputs to integer level IDs.

    `vector-quantize-pytorch` exposes mixed-radix indices for whole FSQ tuples.
    RPG needs factorized labels instead, so this mirrors the library's
    scale-and-shift conversion for each scalar coordinate separately.
    """
    if preserve_symmetry:
        return ((quantized_tensor + 1.0) / (2.0 / (codebook_size - 1))).round().long()

    half_width = codebook_size // 2
    return (quantized_tensor * half_width + half_width).round().long()


def _generate_fsq_quantile_codes(
    sent_embs: np.ndarray,
    train_mask: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """Build FSQ-inspired scalar codes with PCA plus quantile bins.

    This is not the learned finite-scalar quantizer from the FSQ paper. It is a
    deliberately simple scalar-tokenizer ablation: project the item embedding
    space down to one scalar coordinate per RPG digit, then discretize each
    coordinate into `codebook_size` quantile bins. Thresholds are fitted on
    training-prefix items so marginal utilization is expected to be high on the
    training distribution.

    Unlike VQ/PQ, the tokens are scalar intervals rather than learned vector
    prototypes. That makes this a clean test of whether balanced scalar labels
    are enough for RPG's independent prediction heads.
    """
    from sklearn.decomposition import PCA

    n_digit = int(config["n_codebook"])
    codebook_size = int(config["codebook_size"])
    train_embs = sent_embs[train_mask]
    if min(train_embs.shape) < n_digit:
        raise ValueError(
            f"FSQ-quantile PCA needs at least n_codebook={n_digit} training "
            f"items and features, got shape {train_embs.shape}."
        )

    pca = PCA(n_components=n_digit, whiten=True, random_state=int(config["rand_seed"]))
    pca.fit(train_embs)
    projected = pca.transform(sent_embs).astype(np.float32)
    train_projected = projected[train_mask]

    codes = np.zeros((sent_embs.shape[0], n_digit), dtype=np.int64)
    if codebook_size <= 1:
        return codes

    quantiles = np.linspace(0.0, 1.0, codebook_size + 1, dtype=np.float64)[1:-1]
    for digit in range(n_digit):
        thresholds = np.quantile(train_projected[:, digit], quantiles)
        codes[:, digit] = np.searchsorted(thresholds, projected[:, digit], side="right")

    return np.clip(codes, 0, codebook_size - 1)


def _generate_pq_codes(
    sent_embs: np.ndarray,
    train_mask: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """
    Build plain FAISS PQ codes without OPQ's learned rotation.
    """
    import faiss

    n_digit = int(config["n_codebook"])
    codebook_size = int(config["codebook_size"])
    n_bits = _get_codebook_bits(codebook_size)
    faiss.omp_set_num_threads(int(config["faiss_omp_num_threads"]))

    index = faiss.index_factory(
        sent_embs.shape[1],
        f"IVF1,PQ{n_digit}x{n_bits}",
        faiss.METRIC_INNER_PRODUCT,
    )
    index.train(sent_embs[train_mask].astype(np.float32))
    index.add(sent_embs.astype(np.float32))

    ivf_index = faiss.downcast_index(index.index)
    invlists = faiss.extract_index_ivf(ivf_index).invlists
    list_size = invlists.list_size(0)
    pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), list_size * invlists.code_size)
    pq_codes = pq_codes.reshape(-1, invlists.code_size)

    decoded_codes = []
    for packed_code in pq_codes:
        reader = faiss.BitstringReader(faiss.swig_ptr(packed_code), pq_codes.shape[1])
        decoded_codes.append([reader.read(n_bits) for _ in range(n_digit)])

    return np.asarray(decoded_codes, dtype=np.int64)


def _get_codebook_bits(codebook_size: int) -> int:
    """Return the number of bits needed by FAISS packed PQ codes."""
    bits = math.log2(codebook_size)
    if not bits.is_integer() or bits < 0:
        raise ValueError(f"codebook_size must be a power of two, got {codebook_size}.")
    return int(bits)
