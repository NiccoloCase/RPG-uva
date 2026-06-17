#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

# Determine REPO_ROOT dynamically based on script location
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Paths to the DRPG-specific configurations
ROOT_CONFIG = REPO_ROOT / "configs" / "drpg" / "root.yaml"
LOCAL_CONFIG = REPO_ROOT / "configs" / "drpg" / "local.yaml"
PRESET_CONFIGS = {
    "sports_and_outdoors": REPO_ROOT / "configs" / "drpg" / "repro" / "sports_and_outdoors.yaml",
    "beauty": REPO_ROOT / "configs" / "drpg" / "repro" / "beauty.yaml",
    "toys_and_games": REPO_ROOT / "configs" / "drpg" / "repro" / "toys_and_games.yaml",
    "cds_and_vinyl": REPO_ROOT / "configs" / "drpg" / "repro" / "cds_and_vinyl.yaml",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run the DRPG model from the diffusion/genrec pipeline.",
        epilog=(
            "Config precedence: CLI overrides > --config files > --preset file > "
            "configs/drpg/local.yaml > configs/drpg/root.yaml > genrec defaults. "
            "Forwarded config overrides accept both '--key=value' and '--key value'."
        ),
    )
    parser.add_argument("--model", default="DRPG", help="Model name exposed by the genrec registry.")
    parser.add_argument("--dataset", default="AmazonReviews2014", help="Dataset name exposed by genrec.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path.")
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
        help="Skip configs/drpg/root.yaml.",
    )
    parser.add_argument(
        "--no-local-config",
        action="store_true",
        help="Skip configs/drpg/local.yaml even if it exists.",
    )
    return parser.parse_known_args()


def parse_override_value(raw_value: str):
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null", "~"}:
        return None

    try:
        return ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        return raw_value


def parse_override_args(tokens: list[str]) -> dict:
    overrides = {}
    index = 0

    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected argument: {token}")

        body = token[2:]
        if not body:
            raise ValueError("Encountered an empty override flag.")

        if "=" in body:
            key, raw_value = body.split("=", 1)
            index += 1
        else:
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
                raise ValueError(
                    f"Invalid override '{token}'. Use '--key=value' or '--key value'."
                )
            key = body
            raw_value = tokens[index + 1]
            index += 2

        overrides[key.replace("-", "_")] = parse_override_value(raw_value)

    return overrides


def resolve_user_config(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    return path


def build_config_files(args: argparse.Namespace) -> list[str]:
    config_files: list[Path] = []

    if not args.no_root_config and ROOT_CONFIG.is_file():
        config_files.append(ROOT_CONFIG)
    if not args.no_local_config and LOCAL_CONFIG.is_file():
        config_files.append(LOCAL_CONFIG)
    if args.preset:
        config_files.append(PRESET_CONFIGS[args.preset])

    for raw_path in args.config:
        config_files.append(resolve_user_config(raw_path))

    return [str(path) for path in config_files]


def main() -> int:
    args, override_tokens = parse_args()

    # Inject the new diffusion path so 'genrec' resolves to diffusion/genrec
    # instead of third_party/genrec
    diffusion_path = str(REPO_ROOT / "diffusion")
    if diffusion_path not in sys.path:
        sys.path.insert(0, diffusion_path)

    config_files = build_config_files(args)
    overrides = parse_override_args(override_tokens)

    # Import pipeline from the modified framework copy
    from genrec.pipeline import Pipeline

    pipeline = Pipeline(
        model_name=args.model,
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        config_file=config_files or None,
        config_dict=overrides or None,
    )
    pipeline.run()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
