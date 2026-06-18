"""Command-line interface for RPG graph analysis."""

from __future__ import annotations

import argparse

from .dynamic import run_dynamic
from .prepare import prepare_graph
from .pruning import run_pruning
from .reranking.eval import run_reranking
from .scoring import run_scoring
from .static import run_static


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser used by the thin launcher script."""

    parser = argparse.ArgumentParser(
        description="Prepare and analyze RPG decoding graphs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_inputs(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--checkpoint",
            required=True,
            help="Path to a trained RPG checkpoint.",
        )
        subparser.add_argument(
            "--config",
            action="append",
            default=[],
            help="Additional YAML config file. May be provided multiple times.",
        )
        subparser.add_argument(
            "--session-dir",
            default=None,
            help=(
                "Optional graph-analysis session directory. prepare-graph creates it; "
                "static and dynamic read/write it. If omitted, prepare-graph creates a "
                "timestamped session and analysis commands use the latest session with "
                "graph metadata."
            ),
        )
        subparser.add_argument(
            "--no-root-config",
            action="store_true",
            help="Skip configs/rpg/root.yaml.",
        )
        subparser.add_argument(
            "--no-local-config",
            action="store_true",
            help="Skip configs/rpg/local.yaml even if it exists.",
        )

    prepare_parser = subparsers.add_parser(
        "prepare-graph",
        help="Build a fresh exact flat graph into graph-analysis artifacts.",
    )
    add_common_inputs(prepare_parser)

    static_parser = subparsers.add_parser(
        "static",
        help="Compute static graph-analysis CSV/JSON outputs from a prepared graph.",
    )
    add_common_inputs(static_parser)

    dynamic_parser = subparsers.add_parser(
        "dynamic",
        help="Run dynamic/query-conditioned graph analysis from a prepared graph.",
    )
    add_common_inputs(dynamic_parser)

    pruning_parser = subparsers.add_parser(
        "pruning",
        help="Run a lightweight beam-budget diagnostic from a prepared graph.",
    )
    add_common_inputs(pruning_parser)

    rerank_parser = subparsers.add_parser(
        "rerank",
        help="Run a lightweight candidate-reranking intervention from a prepared graph.",
    )
    add_common_inputs(rerank_parser)

    scoring_parser = subparsers.add_parser(
        "scoring",
        help="Run Experiment C: brute-force RPG scoring vs graph decoding.",
    )
    add_common_inputs(scoring_parser)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and dispatch to the selected subcommand."""

    parser = build_parser()
    args, override_tokens = parser.parse_known_args(argv)
    args.override_tokens = override_tokens

    if args.command == "prepare-graph":
        return prepare_graph(args)
    if args.command == "static":
        return run_static(args)
    if args.command == "dynamic":
        return run_dynamic(args)
    if args.command == "pruning":
        return run_pruning(args)
    if args.command == "rerank":
        return run_reranking(args)
    if args.command == "scoring":
        return run_scoring(args)
    parser.error(f"Unsupported command: {args.command}")
    return 2
