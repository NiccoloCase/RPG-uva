from __future__ import annotations

"""Repo-owned tokenizer for RPG semantic-ID ablations.

`RPGAblationTokenizer` subclasses the released RPG tokenizer and changes only
semantic-ID cache generation. All later behavior stays upstream-compatible:
items are mapped to fixed-length semantic IDs, those digits are offset into
per-position token ranges, and the RPG model still trains with the same MTP
loss and graph-constrained decoding code.
"""

import json
import os
from pathlib import Path

import numpy as np

from genrec.dataset import AbstractDataset
from genrec.models.RPG.tokenizer import RPGTokenizer

from .quantizers import generate_codes, write_stats


class RPGAblationTokenizer(RPGTokenizer):
    """RPG tokenizer that swaps only semantic-ID quantization.

    The class exists to keep tokenizer ablations out of `third_party/`. It
    reuses upstream sentence embedding, item split, token offset, and collation
    logic, but routes semantic-ID generation through the small quantizer module
    in this package.
    """

    def __init__(self, config: dict, dataset: AbstractDataset):
        """Store the requested ablation method before upstream initialization.

        `RPGTokenizer.__init__` immediately calls `_init_tokenizer`, so the
        method name must be available before `super().__init__`.
        """
        self.semantic_id_method = str(config.get("semantic_id_method", "fsq")).lower()
        super().__init__(config, dataset)

    @classmethod
    def semantic_id_cache_path(cls, config: dict, dataset: AbstractDataset) -> Path:
        """Return the method-specific semantic-ID cache path.

        The filename includes only the config pieces that define the generated
        code table in v1: embedding model, quantizer method, ID length,
        per-digit codebook size, and upstream PCA dimension. This prevents
        library FSQ, FSQ-quantile, PQ, and OPQ caches from being mixed
        accidentally.
        """
        method = str(config.get("semantic_id_method", "fsq")).lower()
        filename = (
            f'{os.path.basename(config["sent_emb_model"])}_'
            f'{method}_m{config["n_codebook"]}_'
            f'k{config["codebook_size"]}_'
            f'pca{config.get("sent_emb_pca", 0)}.sem_ids'
        )
        return Path(os.path.join(dataset.cache_dir, "processed", filename))

    def _init_tokenizer(self, dataset: AbstractDataset):
        """Load or create ablation semantic IDs, then apply RPG token offsets.

        This mirrors upstream `_init_tokenizer`: reuse cached sentence
        embeddings when present, optionally apply the existing `sent_emb_pca`,
        fit the tokenizer only on training-prefix items, and finally load the
        JSON cache into `item2tokens`.
        """
        sem_ids_path = self.semantic_id_cache_path(self.config, dataset)

        if not sem_ids_path.exists():
            sent_emb_path = os.path.join(
                dataset.cache_dir,
                "processed",
                f'{os.path.basename(self.config["sent_emb_model"])}.sent_emb',
            )
            if os.path.exists(sent_emb_path):
                self.log(f"[TOKENIZER] Loading sentence embeddings from {sent_emb_path}...")
                sent_embs = np.fromfile(sent_emb_path, dtype=np.float32).reshape(
                    -1,
                    self.config["sent_emb_dim"],
                )
            else:
                self.log("[TOKENIZER] Encoding sentence embeddings...")
                sent_embs = self._encode_sent_emb(dataset, sent_emb_path)

            if self.config["sent_emb_pca"] > 0:
                self.log("[TOKENIZER] Applying PCA to sentence embeddings...")
                from sklearn.decomposition import PCA

                pca = PCA(n_components=self.config["sent_emb_pca"], whiten=True)
                sent_embs = pca.fit_transform(sent_embs)
            self.log(f"[TOKENIZER] Sentence embeddings shape: {sent_embs.shape}")

            training_item_mask = self._get_items_for_training(dataset)
            self._generate_semantic_ids(sent_embs, sem_ids_path, training_item_mask)

        self.log(f"[TOKENIZER] Loading semantic IDs from {sem_ids_path}...")
        item2sem_ids = json.load(open(sem_ids_path, "r"))
        return self._sem_ids_to_tokens(item2sem_ids)

    def _generate_semantic_ids(
        self,
        sent_embs: np.ndarray,
        sem_ids_path: str | Path,
        train_mask: np.ndarray,
    ) -> None:
        """Generate, validate, and cache un-offset semantic IDs.

        The quantizer returns raw digit values in `[0, codebook_size - 1]`.
        This method validates that shape/range contract, writes the upstream
        `.sem_ids` JSON mapping, and writes diagnostic stats in a sidecar JSON.
        """
        self.log(f"[TOKENIZER] Generating {self.semantic_id_method} semantic IDs...")
        codes = generate_codes(
            method=self.semantic_id_method,
            sent_embs=sent_embs,
            train_mask=train_mask,
            config=self.config,
        )
        if codes.shape != (sent_embs.shape[0], self.n_digit):
            raise ValueError(
                f"Expected semantic IDs with shape {(sent_embs.shape[0], self.n_digit)}, "
                f"got {codes.shape}."
            )
        if codes.min() < 0 or codes.max() >= self.codebook_size:
            raise ValueError(
                f"{self.semantic_id_method} generated codes outside "
                f"[0, {self.codebook_size - 1}]."
            )

        item2sem_ids = {}
        for index in range(codes.shape[0]):
            item = self.id2item[index + 1]
            item2sem_ids[item] = tuple(int(value) for value in codes[index].tolist())

        sem_ids_path = Path(sem_ids_path)
        self.log(f"[TOKENIZER] Saving semantic IDs to {sem_ids_path}...")
        sem_ids_path.parent.mkdir(parents=True, exist_ok=True)
        sem_ids_path.write_text(json.dumps(item2sem_ids))
        write_stats(sem_ids_path, codes, self.config)
