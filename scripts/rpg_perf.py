#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perf.config import (
    build_repo_config_files,
    ensure_submodule_available,
    parse_int_list,
    parse_override_args,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile RPG inference without modifying third_party/.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_arguments(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--model",
            default="RPG",
            help="Model name exposed by the vendored or repo-owned GenRec registry.",
        )
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
            "--no-root-config",
            action="store_true",
            help="Skip configs/rpg/root.yaml.",
        )
        subparser.add_argument(
            "--no-local-config",
            action="store_true",
            help="Skip configs/rpg/local.yaml even if it exists.",
        )
        subparser.add_argument(
            "--output-dir",
            default=None,
            help="Optional session root override for profiling artifacts.",
        )

    validate_parser = subparsers.add_parser(
        "validate-graph",
        help="Compare the repo-owned exact sparse graph against the upstream dense graph.",
    )
    add_common_arguments(validate_parser)

    profile_parser = subparsers.add_parser(
        "profile",
        help="Prepare adjacency caches and/or profile inference over enlarged candidate pools.",
    )
    add_common_arguments(profile_parser)
    profile_parser.add_argument(
        "--pool-sizes",
        default=None,
        help="Comma-separated pool sizes to override config.pool_sizes.",
    )
    profile_parser.add_argument(
        "--graph-backend",
        default=None,
        choices=["flat", "hnsw"],
        help="Optional graph backend override.",
    )
    profile_parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build or refresh adjacency caches without running timed inference.",
    )
    profile_parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Require existing adjacency caches and skip cache builds.",
    )
    profile_parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild adjacency caches even if matching cache files already exist.",
    )

    plot_parser = subparsers.add_parser(
        "plot",
        help="Render a two-panel plot from a summary CSV or profiling session directory.",
    )
    plot_parser.add_argument(
        "--input",
        required=True,
        help="Summary CSV path or profiling session directory.",
    )
    plot_parser.add_argument(
        "--output",
        required=True,
        help="Output image path, for example artifacts/.../perf_rpg.png.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, override_tokens = parser.parse_known_args(argv)

    if args.command == "plot":
        from perf.plotting import plot_summary_csv

        output_path = plot_summary_csv(args.input, args.output)
        print(output_path)
        return 0

    ensure_submodule_available()
    config_files = build_repo_config_files(
        extra_configs=args.config,
        include_root_config=not args.no_root_config,
        include_local_config=not args.no_local_config,
    )
    config_overrides = parse_override_args(override_tokens)

    if args.command == "validate-graph":
        from perf.profile import run_validate_graph_command

        result = run_validate_graph_command(
            checkpoint_path=args.checkpoint,
            config_files=config_files,
            config_overrides=config_overrides,
            model_name=args.model,
            output_root=args.output_dir,
        )
        print(result["report_path"])
        return 0

    if args.command == "profile":
        from perf.profile import run_profile_command

        manifest = run_profile_command(
            checkpoint_path=args.checkpoint,
            config_files=config_files,
            config_overrides=config_overrides,
            model_name=args.model,
            output_root=args.output_dir,
            pool_sizes_override=parse_int_list(args.pool_sizes),
            prepare_only=args.prepare_only,
            profile_only=args.profile_only,
            graph_backend_override=args.graph_backend,
            force_rebuild=args.force_rebuild,
        )
        print(manifest["session_root"])
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
