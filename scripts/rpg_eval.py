#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys

from rpg import (
    build_config_files,
    parse_override_args,
)
from genrec_repo_support import THIRD_PARTY_ROOT, prepare_genrec_runtime


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Evaluate a GenRec checkpoint from the repository root without retraining.",
        epilog=(
            "Config precedence matches scripts/rpg.py. Forwarded config overrides "
            "accept both '--key=value' and '--key value'."
        ),
    )
    parser.add_argument("--model", default="RPG", help="Model name exposed by the vendored or repo-owned GenRec registry.")
    parser.add_argument("--dataset", default="AmazonReviews2014", help="Dataset name exposed by third_party/genrec.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path to evaluate.")
    parser.add_argument(
        "--preset",
        choices=["beauty", "cds_and_vinyl", "sports_and_outdoors", "toys_and_games"],
        help="Apply one of the paper reproduction presets.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Additional YAML config file. May be provided multiple times.",
    )
    parser.add_argument("--no-root-config", action="store_true", help="Skip configs/rpg/root.yaml.")
    parser.add_argument(
        "--no-local-config",
        action="store_true",
        help="Skip configs/rpg/local.yaml even if it exists.",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="Seed to reset immediately before evaluation. Defaults to config rand_seed.",
    )
    return parser.parse_known_args()


def main() -> int:
    args, override_tokens = parse_args()
    prepare_genrec_runtime(args.model)

    sys.path.insert(0, str(THIRD_PARTY_ROOT))

    from genrec.pipeline import Pipeline
    from genrec.utils import init_seed
    from torch.utils.data import DataLoader

    pipeline = Pipeline(
        model_name=args.model,
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        config_file=build_config_files(args) or None,
        config_dict=parse_override_args(override_tokens) or None,
    )

    test_dataloader = DataLoader(
        pipeline.tokenized_datasets["test"],
        batch_size=pipeline.config["eval_batch_size"],
        shuffle=False,
        collate_fn=pipeline.tokenizer.collate_fn["test"],
    )

    pipeline.model = pipeline.accelerator.unwrap_model(pipeline.model)
    pipeline.model, test_dataloader = pipeline.accelerator.prepare(
        pipeline.model,
        test_dataloader,
    )
    pipeline.trainer.model = pipeline.model
    pipeline.trainer.model.generate_w_decoding_graph = True

    base_seed = pipeline.config["rand_seed"] if args.eval_seed is None else args.eval_seed
    init_seed(base_seed, pipeline.config["reproducibility"])
    pipeline.log(f"Eval seed: {base_seed}")
    test_results = pipeline.trainer.evaluate(test_dataloader)
    pipeline.log(f"Eval-only checkpoint: {args.checkpoint}")
    pipeline.log(f"Test Results: {test_results}")
    pipeline.trainer.end()

    serializable_results = {
        key: float(value) if isinstance(value, (float, int)) else value
        for key, value in test_results.items()
    }
    payload = {
        "eval_seed": base_seed,
        "test_results": serializable_results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
