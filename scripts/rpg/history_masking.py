#!/usr/bin/env python3
"""RPG test metrics with and without removing the user's past items.

RPG's evaluation ranks the whole catalogue, including items the user has
already interacted with. The S3-Rec / SASRec evaluation code removes those
items before ranking. Because no test target in these datasets appears in the
user's history, removing past items can only raise the metrics. This script
scores RPG both ways, with exhaustive scoring, so RPG and SASRec can be
compared under the same protocol (see scripts/sasrec/train.py --mask_history).

Usage:
  python scripts/rpg/history_masking.py --name beauty \\
      --checkpoint artifacts/rpg/ckpt/<run>.pth --config configs/rpg/repro/beauty.yaml
Writes artifacts/rpg/history_masking/<name>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from decoding.config import build_repo_config_files, ensure_submodule_available, parse_override_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="Label for the output file, e.g. beauty.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained RPG checkpoint.")
    parser.add_argument("--config", action="append", default=[], help="Additional YAML config file.")
    parser.add_argument("--output-dir", default="artifacts/rpg/history_masking", help="Directory for the result JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, override_tokens = parser.parse_known_args(argv)
    ensure_submodule_available()
    from decoding.harness import EvaluationHarness
    from decoding.trace import compute_decoding_context

    harness = EvaluationHarness.build(
        checkpoint_path=args.checkpoint,
        config_files=build_repo_config_files(extra_configs=args.config, include_root_config=True, include_local_config=False),
        config_overrides=parse_override_args(override_tokens),
    )
    model = harness.model.eval()
    device = harness.accelerator.device
    item_tokens = model.item_id2tokens.to(device)[1:] - 1  # (n_real_items, n_digit); column i is item id i + 1

    sums = {False: {"recall@10": 0.0, "ndcg@10": 0.0}, True: {"recall@10": 0.0, "ndcg@10": 0.0}}
    past_items_in_top10 = 0.0
    n_users = 0
    with torch.no_grad():
        for batch in harness.test_dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            token_logits = compute_decoding_context(model, batch).token_logits
            scores = token_logits[:, item_tokens].mean(-1)
            labels = batch["labels"][:, 0]

            history = torch.zeros_like(scores, dtype=torch.bool)
            input_ids = batch["input_ids"]
            valid = input_ids > 0
            rows = torch.arange(input_ids.shape[0], device=device).unsqueeze(1).expand_as(input_ids)
            history[rows[valid], input_ids[valid] - 1] = True

            for remove_past in (False, True):
                ranked = scores.masked_fill(history, float("-inf")) if remove_past else scores
                top = ranked.topk(10, dim=-1).indices + 1
                if not remove_past:
                    past_items_in_top10 += history.gather(1, top - 1).float().mean(-1).sum().item()
                hit = top == labels.unsqueeze(1)
                found = hit.any(-1).float()
                rank = hit.float().argmax(-1).float()
                sums[remove_past]["recall@10"] += found.sum().item()
                sums[remove_past]["ndcg@10"] += (found / torch.log2(rank + 2)).sum().item()
            n_users += labels.shape[0]

    result = {
        "name": args.name,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "category": harness.config.get("category"),
        "n_users": n_users,
        "past_items_kept": {k: round(v / n_users, 5) for k, v in sums[False].items()},
        "past_items_removed": {k: round(v / n_users, 5) for k, v in sums[True].items()},
        "share_of_top10_that_are_past_items": round(past_items_in_top10 / n_users, 4),
    }
    print(json.dumps(result, indent=2))
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.name}.json").write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
