"""Runtime configuration helpers.

The upstream GenRec code uses a merged YAML configuration rather than Hydra.
These helpers centralize the pieces needed by graph preparation and the
decoding analyses: resolving config files, constructing the RPG evaluation
harness, and normalizing analysis-specific settings.
"""

from __future__ import annotations

import argparse
from typing import Any

from decoding.config import build_repo_config_files, ensure_submodule_available, parse_override_args
from decoding.harness import EvaluationHarness

from .settings import DEFAULT_RANDOM_SEEDS


def config_files_from_args(args: argparse.Namespace) -> list[str]:
    """Build the ordered config-file list consumed by the RPG harness."""

    return build_repo_config_files(
        extra_configs=args.config,
        include_root_config=not args.no_root_config,
        include_local_config=not args.no_local_config,
    )


def build_harness_from_args(args: argparse.Namespace) -> EvaluationHarness:
    """Build the RPG evaluation harness for a checkpoint and config overlay.

    The harness loads the dataset, tokenizer, model weights, and test dataloader.
    Graph preparation only needs the dataset/tokenizer/model, but using the
    same harness keeps checkpoint reconstruction consistent with evaluation code.
    """

    ensure_submodule_available()
    config_files = config_files_from_args(args)
    config_overrides = parse_override_args(getattr(args, "override_tokens", []))
    return EvaluationHarness.build(
        checkpoint_path=args.checkpoint,
        config_files=config_files,
        config_overrides=config_overrides,
    )


def topk_from_config(config: dict[str, Any]) -> int:
    """Resolve the width of the prepared graph cache.

    ``graph_topk`` is intentionally separate from RPG decoding ``n_edges``.
    Graph preparation should fail loudly if the graph cache width is not
    configured, rather than silently reusing a decoding parameter.
    """

    if "graph_topk" not in config or config["graph_topk"] is None:
        raise ValueError("Graph preparation requires graph_topk in the graph-analysis config.")
    return int(config["graph_topk"])


def random_seeds_from_config(config: dict[str, Any]) -> list[int]:
    """Resolve the fixed seeds used when a config gives no evaluation seeds."""

    return [int(seed) for seed in config.get("graph_analysis_random_seeds", DEFAULT_RANDOM_SEEDS)]
