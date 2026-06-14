#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

from accelerate import Accelerator

from perf.config import build_repo_config_files, ensure_submodule_available, parse_override_args
from genrec_repo_support import prepare_genrec_runtime


REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_ROOT = REPO_ROOT / "third_party"
PRESET_CONFIGS = {
    "sports_and_outdoors": REPO_ROOT / "configs" / "rpg" / "repro" / "sports_and_outdoors.yaml",
    "beauty": REPO_ROOT / "configs" / "rpg" / "repro" / "beauty.yaml",
    "toys_and_games": REPO_ROOT / "configs" / "rpg" / "repro" / "toys_and_games.yaml",
    "cds_and_vinyl": REPO_ROOT / "configs" / "rpg" / "repro" / "cds_and_vinyl.yaml",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare RPG semantic-ID caches without training or profiling.",
    )
    parser.add_argument("--model", default="RPG", help="Model name exposed by third_party/genrec.")
    parser.add_argument(
        "--dataset",
        default="AmazonReviews2014",
        help="Dataset name exposed by third_party/genrec.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_CONFIGS),
        help="Apply one of the paper reproduction presets.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Additional YAML config file. May be provided multiple times.",
    )
    parser.add_argument(
        "--no-root-config",
        action="store_true",
        help="Skip configs/rpg/root.yaml.",
    )
    parser.add_argument(
        "--no-local-config",
        action="store_true",
        help="Skip configs/rpg/local.yaml even if it exists.",
    )
    return parser


def semantic_id_cache_path(config: dict, dataset) -> Path:
    from genrec.utils import get_tokenizer

    tokenizer_cls = get_tokenizer(config["model"])
    if hasattr(tokenizer_cls, "semantic_id_cache_path"):
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, override_tokens = parser.parse_known_args(argv)

    ensure_submodule_available()
    prepare_genrec_runtime(args.model)

    config_files = build_repo_config_files(
        extra_configs=args.config,
        include_root_config=not args.no_root_config,
        include_local_config=not args.no_local_config,
    )
    if args.preset:
        config_files.append(str(PRESET_CONFIGS[args.preset]))
    config_overrides = parse_override_args(override_tokens)

    from genrec.utils import get_config, get_dataset, get_tokenizer, init_logger, init_seed

    config = get_config(
        model_name=args.model,
        dataset_name=args.dataset,
        config_file=config_files or None,
        config_dict=config_overrides or None,
    )
    accelerator = Accelerator()
    config["accelerator"] = accelerator
    config["device"] = accelerator.device
    config["use_ddp"] = accelerator.num_processes > 1

    init_seed(config["rand_seed"], config["reproducibility"])
    init_logger(config)

    dataset = get_dataset(args.dataset)(config)
    dataset.split()
    cache_path = semantic_id_cache_path(config, dataset)
    print(f"semantic_id_cache={cache_path}")

    if cache_path.is_file():
        print("status=exists")
        return 0

    get_tokenizer(args.model)(config, dataset)

    if not cache_path.is_file():
        raise SystemExit(f"Semantic-ID cache was not created: {cache_path}")

    print("status=created")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
