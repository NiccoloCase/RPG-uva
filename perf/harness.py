from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from .config import THIRD_PARTY_ROOT

SCRIPTS_ROOT = THIRD_PARTY_ROOT.parent / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from genrec_repo_support import prepare_genrec_runtime


@dataclass
class EvaluationHarness:
    config: dict[str, Any]
    accelerator: Accelerator
    dataset: Any
    tokenizer: Any
    model: Any
    trainer: Any
    test_dataloader: DataLoader
    checkpoint_path: Path

    @staticmethod
    def _semantic_id_cache_path(
        config: dict[str, Any],
        dataset: Any,
        tokenizer_cls: type | None = None,
    ) -> Path:
        """Reconstruct the semantic-ID cache path used by the upstream tokenizer.

        Args:
            config: Fully merged RPG config dictionary.
            dataset: Upstream dataset instance exposing `cache_dir`.

        Returns:
            Absolute or repository-relative path to the `.sem_ids` cache file
            expected by `RPGTokenizer`.
        """
        if tokenizer_cls is not None and hasattr(tokenizer_cls, "semantic_id_cache_path"):
            return Path(tokenizer_cls.semantic_id_cache_path(config, dataset))

        n_codebook_bits = int(math.log2(config["codebook_size"]))
        index_factory = (
            f'OPQ{config["n_codebook"]},IVF1,PQ{config["n_codebook"]}x{n_codebook_bits}'
        )
        return Path(
            os.path.join(
                dataset.cache_dir,
                "processed",
                f'{os.path.basename(config["sent_emb_model"])}_{index_factory}.sem_ids',
            )
        )

    @classmethod
    def build(
        cls,
        checkpoint_path: str | Path,
        config_files: list[str],
        config_overrides: dict[str, Any] | None = None,
        model_name: str = "RPG",
    ) -> "EvaluationHarness":
        """Construct a ready-to-evaluate RPG stack from a checkpoint.

        This helper mirrors the minimum subset of the upstream pipeline needed
        for offline inference profiling: load configs, dataset, tokenizer,
        tokenized test split, model weights, trainer, and test dataloader.

        Args:
            checkpoint_path: Path to the trained RPG checkpoint to profile.
            config_files: Ordered config files to merge before evaluation.
            config_overrides: Optional in-memory overrides applied after the
                config files.

        Returns:
            An `EvaluationHarness` containing the assembled config, model,
            tokenizer, trainer, and test dataloader.

        Raises:
            FileNotFoundError: If the checkpoint path does not exist.
            RuntimeError: If semantic-ID caches are required but cannot be
                generated safely with the active embedding configuration.
        """
        prepare_genrec_runtime(model_name)

        from genrec.utils import (
            get_config,
            get_dataset,
            get_model,
            get_tokenizer,
            get_trainer,
            init_logger,
            init_seed,
        )

        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        config = get_config(
            model_name=model_name,
            dataset_name="AmazonReviews2014",
            config_file=config_files or None,
            config_dict=config_overrides or None,
        )
        accelerator = Accelerator()
        config["accelerator"] = accelerator
        config["device"] = accelerator.device
        config["use_ddp"] = accelerator.num_processes > 1

        init_seed(config["rand_seed"], config["reproducibility"])
        init_logger(config)

        dataset = get_dataset(config["dataset"])(config)
        split_datasets = dataset.split()

        tokenizer_cls = get_tokenizer(config["model"])
        semantic_id_cache = cls._semantic_id_cache_path(config, dataset, tokenizer_cls)
        if (
            config.get("metadata") == "sentence"
            and "text-embedding-3" in str(config.get("sent_emb_model", ""))
            and not semantic_id_cache.is_file()
            and not config.get("openai_api_key")
        ):
            raise RuntimeError(
                "Semantic-ID cache is missing and the active tokenizer config uses "
                "OpenAI embeddings without an API key. Either populate the dataset "
                f"cache first ({semantic_id_cache}), set openai_api_key in "
                "configs/rpg/local.yaml, or override sent_emb_model to a local "
                "sentence-transformers encoder before profiling."
            )

        tokenizer = tokenizer_cls(config, dataset)
        tokenized_test = tokenizer.tokenize({"test": split_datasets["test"]})["test"]

        model = get_model(config["model"])(config, dataset, tokenizer)
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(config["device"])
        model.eval()

        trainer = get_trainer(config["model"])(config, model, tokenizer)
        test_dataloader = DataLoader(
            tokenized_test,
            batch_size=config["eval_batch_size"],
            shuffle=False,
            collate_fn=tokenizer.collate_fn["test"],
        )

        return cls(
            config=config,
            accelerator=accelerator,
            dataset=dataset,
            tokenizer=tokenizer,
            model=model,
            trainer=trainer,
            test_dataloader=test_dataloader,
            checkpoint_path=checkpoint,
        )

    def warmup(self, num_batches: int) -> None:
        """Run a small number of inference batches to warm caches and kernels.

        Args:
            num_batches: Number of test batches to execute. Non-positive values
                disable warmup.

        Returns:
            None.
        """
        if num_batches <= 0:
            return

        maxk = self.trainer.evaluator.maxk
        self.model.eval()

        with torch.no_grad():
            for batch_index, batch in enumerate(self.test_dataloader):
                if batch_index >= num_batches:
                    break
                batch = {key: value.to(self.accelerator.device) for key, value in batch.items()}
                _ = self.model.generate(batch, n_return_sequences=maxk)

    def evaluate(self) -> dict[str, float]:
        """Evaluate the checkpoint on the test split with graph decoding enabled.

        Returns:
            The metrics dictionary returned by the upstream trainer, typically
            containing ranking metrics such as `recall@k`, `ndcg@k`, and the
            average number of visited items during graph-constrained decoding.
        """
        self.model.generate_w_decoding_graph = True
        self.model.eval()
        return self.trainer.evaluate(self.test_dataloader, split="test")
