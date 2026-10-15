#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from decoding.config import THIRD_PARTY_ROOT, ensure_submodule_available  # noqa: E402
ROOT_CONFIG = REPO_ROOT / "configs" / "rpg" / "root.yaml"
LOCAL_CONFIG = REPO_ROOT / "configs" / "rpg" / "local.yaml"
PRESET_CONFIGS = {
    "sports_and_outdoors": REPO_ROOT / "configs" / "rpg" / "repro" / "sports_and_outdoors.yaml",
    "beauty": REPO_ROOT / "configs" / "rpg" / "repro" / "beauty.yaml",
    "toys_and_games": REPO_ROOT / "configs" / "rpg" / "repro" / "toys_and_games.yaml",
    "cds_and_vinyl": REPO_ROOT / "configs" / "rpg" / "repro" / "cds_and_vinyl.yaml",
    "books": REPO_ROOT / "configs" / "rpg" / "books.yaml",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Train and test RPG from the repository root without modifying the upstream code.",
        epilog=(
            "Config precedence: CLI overrides > --config files > --preset file > "
            "configs/rpg/local.yaml > configs/rpg/root.yaml > upstream defaults. "
            "Forwarded config overrides accept both '--key=value' and '--key value'."
        ),
    )
    parser.add_argument("--model", default="RPG", help="Model name exposed by the upstream RPG code (genrec).")
    parser.add_argument("--dataset", default="AmazonReviews2014", help="Dataset name exposed by the upstream RPG code (genrec).")
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
        help="Skip configs/rpg/root.yaml.",
    )
    parser.add_argument(
        "--no-local-config",
        action="store_true",
        help="Skip configs/rpg/local.yaml even if it exists.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help=(
            "Train and save the best checkpoint, then stop. Skips the final "
            "graph-decoded test, whose dense item-item matrix does not fit in "
            "memory for large catalogues. Evaluate afterwards with "
            "scripts/rpg/eval_seeds.py."
        ),
    )
    parser.add_argument(
        "--val-max-users",
        type=int,
        default=None,
        help=(
            "With --train-only: validate on a fixed random subset of this many "
            "users instead of all of them. Validation scores every item for "
            "every user, so this bounds its cost on large datasets."
        ),
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


def run_train_only(pipeline, val_max_users: int | None) -> None:
    """Run the training half of ``Pipeline.run`` and stop before the test.

    The dataloaders are built exactly as in ``Pipeline.run``. The only optional
    difference is the validation subset used for early stopping.
    """

    from torch.utils.data import DataLoader

    config = pipeline.config
    val_dataset = pipeline.tokenized_datasets["val"]
    if val_max_users is not None and val_max_users < len(val_dataset):
        val_dataset = val_dataset.shuffle(seed=config["rand_seed"]).select(range(val_max_users))
        pipeline.log(f"Validating on a fixed subset of {val_max_users} users.")

    train_dataloader = DataLoader(
        pipeline.tokenized_datasets["train"],
        batch_size=config["train_batch_size"],
        shuffle=True,
        collate_fn=pipeline.tokenizer.collate_fn["train"],
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["eval_batch_size"],
        shuffle=False,
        collate_fn=pipeline.tokenizer.collate_fn["val"],
    )

    best_epoch, best_val_score = pipeline.trainer.fit(train_dataloader, val_dataloader)
    pipeline.log(
        f"Train-only run finished. best_epoch={best_epoch} "
        f"best_val_score={best_val_score} "
        f"checkpoint={pipeline.trainer.saved_model_ckpt}"
    )
    pipeline.trainer.end()


MAX_FILE_NAME_LENGTH = 255


def shorten_argv_for_file_names(overrides: dict) -> None:
    """Keep GenRec's checkpoint and log file names under the filesystem limit.

    GenRec names both files ``<run_id>-<full command line>-<time>-<md5>``.
    Long preset names plus a few overrides push that past 255 characters, and
    the run then fails at its first checkpoint save. When that would happen,
    the command line part is reduced to the run id; the md5 of the full config
    still identifies the settings.
    """

    run_id = str(overrides.get("run_id", ""))
    command_line = "_".join(sys.argv).replace("/", "|")
    projected = len(run_id) + len(command_line) + len("-Mon-00-0000_00-00-abcdef.pth") + 2
    if projected > MAX_FILE_NAME_LENGTH:
        sys.argv = [sys.argv[0], "--run_id", run_id]


def main() -> int:
    args, override_tokens = parse_args()
    if args.val_max_users is not None and not args.train_only:
        raise SystemExit("--val-max-users requires --train-only.")
    ensure_submodule_available()
    if str(THIRD_PARTY_ROOT) not in sys.path:
        sys.path.insert(0, str(THIRD_PARTY_ROOT))

    config_files = build_config_files(args)
    overrides = parse_override_args(override_tokens)

    from genrec.pipeline import Pipeline

    shorten_argv_for_file_names(overrides)

    pipeline = Pipeline(
        model_name=args.model,
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        config_file=config_files or None,
        config_dict=overrides or None,
    )
    if args.train_only:
        run_train_only(pipeline, args.val_max_users)
    else:
        pipeline.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
