#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.sasrec import SASRecDataset, SASRecModel, SASRecTrainer
from models.sasrec.utils import EarlyStopping, check_path, get_user_seqs, set_seed

MODEL_CONFIG = REPO_ROOT / "models" / "sasrec" / "config.yaml"
ROOT_CONFIG = REPO_ROOT / "configs" / "sasrec" / "root.yaml"
LOCAL_CONFIG = REPO_ROOT / "configs" / "sasrec" / "local.yaml"
PRESET_CONFIGS = {
    "beauty": REPO_ROOT / "configs" / "sasrec" / "beauty.yaml",
    "cds_and_vinyl": REPO_ROOT / "configs" / "sasrec" / "cds_and_vinyl.yaml",
    "sports_and_outdoors": REPO_ROOT / "configs" / "sasrec" / "sports_and_outdoors.yaml",
    "toys_and_games": REPO_ROOT / "configs" / "sasrec" / "toys_and_games.yaml",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run the standalone SASRec reproduction path migrated from CIKM2020-S3Rec.",
        epilog=(
            "Config precedence: CLI overrides > --config files > --preset file > "
            "configs/sasrec/local.yaml > configs/sasrec/root.yaml > models/sasrec/config.yaml."
        ),
    )
    parser.add_argument("--dataset", default=None, help="Dataset/category override, for example Beauty.")
    parser.add_argument("--preset", choices=sorted(PRESET_CONFIGS), help="Named SASRec preset to apply.")
    parser.add_argument("--config", action="append", default=[], help="Additional YAML config file.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path override.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run test evaluation only.")
    parser.add_argument("--no-root-config", action="store_true", help="Skip configs/sasrec/root.yaml.")
    parser.add_argument("--no-local-config", action="store_true", help="Skip configs/sasrec/local.yaml.")
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


def parse_override_args(tokens: list[str]) -> dict[str, object]:
    overrides: dict[str, object] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected argument: {token}")

        body = token[2:]
        if "=" in body:
            key, raw_value = body.split("=", 1)
            index += 1
        else:
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
                raise ValueError(f"Invalid override '{token}'. Use '--key=value' or '--key value'.")
            key = body
            raw_value = tokens[index + 1]
            index += 2
        overrides[key.replace("-", "_")] = parse_override_value(raw_value)
    return overrides


def resolve_user_config(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    return path


def build_config_files(args: argparse.Namespace) -> list[Path]:
    config_files = [MODEL_CONFIG]
    if not args.no_root_config and ROOT_CONFIG.is_file():
        config_files.append(ROOT_CONFIG)
    if not args.no_local_config and LOCAL_CONFIG.is_file():
        config_files.append(LOCAL_CONFIG)
    if args.preset:
        config_files.append(PRESET_CONFIGS[args.preset])
    for raw_path in args.config:
        config_files.append(resolve_user_config(raw_path))
    return config_files


def load_config(config_files: list[Path], overrides: dict[str, object]) -> dict[str, object]:
    merged: dict[str, object] = {}
    for path in config_files:
        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        merged.update(payload)
    merged.update(overrides)
    return merged


def resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def normalize_config(config: dict[str, object], checkpoint_override: str | None) -> SimpleNamespace:
    normalized = dict(config)
    if normalized.get("dataset") and not normalized.get("category"):
        normalized["category"] = normalized["dataset"]
    if normalized.get("category") and not normalized.get("data_name"):
        normalized["data_name"] = normalized["category"]
    if "seed" not in normalized and "rand_seed" in normalized:
        normalized["seed"] = normalized["rand_seed"]

    normalized["topk"] = sorted({int(k) for k in normalized.get("topk", [5, 10, 20])})
    normalized["seed"] = int(normalized.get("seed", 42))
    normalized["epochs"] = int(normalized.get("epochs", 200))
    normalized["train_batch_size"] = int(normalized.get("train_batch_size", 256))
    normalized["eval_batch_size"] = int(normalized.get("eval_batch_size", 256))
    normalized["log_freq"] = int(normalized.get("log_freq", 1))
    normalized["patience"] = int(normalized.get("patience", 10))
    normalized["full_sort"] = bool(normalized.get("full_sort", True))
    normalized["no_cuda"] = bool(normalized.get("no_cuda", False))
    normalized["val_metric"] = str(normalized.get("val_metric", "ndcg@20")).lower()

    normalized["data_dir"] = str(resolve_repo_path(normalized.get("data_dir", "artifacts/sasrec/data")))
    normalized["log_dir"] = str(resolve_repo_path(normalized.get("log_dir", "artifacts/sasrec/logs")))
    normalized["ckpt_dir"] = str(resolve_repo_path(normalized.get("ckpt_dir", "artifacts/sasrec/ckpt")))
    normalized["output_dir"] = str(
        resolve_repo_path(normalized.get("output_dir", "output/reproduction/sasrec"))
    )

    data_name = str(normalized["data_name"])
    run_id = str(normalized.get("run_id", f"sasrec_{data_name.lower()}"))
    normalized["run_id"] = run_id
    dataset_root = Path(normalized["data_dir"]) / data_name
    normalized["dataset_root"] = str(dataset_root)
    normalized["data_file"] = str(dataset_root / f"{data_name}.txt")
    normalized["log_file"] = str(Path(normalized["log_dir"]) / f"{run_id}.log")
    normalized["checkpoint_path"] = (
        str(resolve_repo_path(checkpoint_override))
        if checkpoint_override is not None
        else str(Path(normalized["ckpt_dir"]) / f"{run_id}.pt")
    )
    return SimpleNamespace(**normalized)


def build_dataloaders(args: SimpleNamespace, user_seq: list[list[int]]):
    train_dataset = SASRecDataset(args, user_seq, data_type="train")
    valid_dataset = SASRecDataset(args, user_seq, data_type="valid")
    test_dataset = SASRecDataset(args, user_seq, data_type="test")

    train_sampler = RandomSampler(train_dataset)
    valid_sampler = SequentialSampler(valid_dataset)
    test_sampler = SequentialSampler(test_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        sampler=valid_sampler,
        batch_size=args.eval_batch_size,
    )
    test_dataloader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=args.eval_batch_size,
    )
    return train_dataloader, valid_dataloader, test_dataloader


def log_run_header(args: SimpleNamespace) -> None:
    payload = {
        key: getattr(args, key)
        for key in sorted(vars(args))
        if key not in {"train_matrix"}
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    with open(args.log_file, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True))
        handle.write("\n")


def main() -> int:
    parsed_args, override_tokens = parse_args()
    overrides = parse_override_args(override_tokens)
    if parsed_args.dataset is not None:
        overrides["dataset"] = parsed_args.dataset

    config_files = build_config_files(parsed_args)
    merged_config = load_config(config_files, overrides)
    args = normalize_config(merged_config, parsed_args.checkpoint)

    check_path(args.log_dir)
    check_path(args.ckpt_dir)
    check_path(args.output_dir)

    if not Path(args.data_file).is_file():
        raise FileNotFoundError(
            f"Missing SASRec data file: {args.data_file}. Run scripts/sasrec_prepare_data.py first."
        )

    set_seed(args.seed)
    user_seq, max_item, valid_rating_matrix, test_rating_matrix = get_user_seqs(args.data_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    args.train_matrix = valid_rating_matrix

    log_run_header(args)
    train_dataloader, valid_dataloader, test_dataloader = build_dataloaders(args, user_seq)
    model = SASRecModel(args)
    trainer = SASRecTrainer(model, train_dataloader, valid_dataloader, test_dataloader, args)

    if parsed_args.eval_only:
        trainer.args.train_matrix = test_rating_matrix
        trainer.load(args.checkpoint_path)
        print(f"Loaded checkpoint for evaluation: {args.checkpoint_path}")
        test_metrics, result_info = trainer.test(0, full_sort=args.full_sort)
    else:
        early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            valid_metrics, _ = trainer.valid(epoch, full_sort=args.full_sort)
            early_stopping(np.array([valid_metrics[args.val_metric]]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        trainer.args.train_matrix = test_rating_matrix
        trainer.load(args.checkpoint_path)
        print(f"Loaded best checkpoint for test: {args.checkpoint_path}")
        test_metrics, result_info = trainer.test(0, full_sort=args.full_sort)

    print(args.run_id)
    print(result_info)
    with open(args.log_file, "a", encoding="utf-8") as handle:
        handle.write(f"{args.run_id}\n")
        handle.write(f"{result_info}\n")
        handle.write(json.dumps(test_metrics, indent=2, sort_keys=True))
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
