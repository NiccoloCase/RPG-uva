#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from genrec_repo_support import prepare_genrec_runtime  # noqa: E402


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "artifacts/rpg/ckpt/drpg_beauty_lr003_s6_h100-scripts|rpg.py_--model_DRPG_--preset_beauty_--run_id_drpg_beauty_lr003_s6_h100_--diffusion_mask_counts_32,24,16,8,4,1_--lr_0.003-Jun-16-2026_17-12-d0ed4f.pth"
)
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/rpg/diagnostics/drpg_beauty_pregraph_diagnostics.json"
ROOT_CONFIG = REPO_ROOT / "configs/rpg/root.yaml"
BEAUTY_CONFIG = REPO_ROOT / "configs/rpg/repro/beauty.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run checkpoint-only pre-graph DRPG diagnostics.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="DRPG checkpoint to diagnose.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="JSON output path.")
    parser.add_argument("--max-examples", type=int, default=256, help="Number of validation examples to inspect.")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Override device.")
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def item_preds_from_token_logprobs(model, token_logprobs: torch.Tensor, k: int) -> torch.Tensor:
    item_logits = torch.gather(
        input=token_logprobs.unsqueeze(-2).expand(-1, model.dataset.n_items, -1),
        dim=-1,
        index=(model.item_id2tokens[1:, :] - 1).unsqueeze(0).expand(token_logprobs.shape[0], -1, -1),
    ).mean(dim=-1)
    return item_logits.topk(k, dim=-1).indices + 1


def one_shot_token_logprobs(model, memory_context: torch.Tensor, memory_padding_mask: torch.Tensor) -> torch.Tensor:
    current_targets = torch.full(
        (memory_context.size(0), model.n_digit),
        model.mask_token_id,
        dtype=torch.long,
        device=memory_context.device,
    )
    logits = model.forward_denoiser_only(
        {
            "target_tokens": current_targets,
            "memory_context": memory_context,
            "memory_padding_mask": memory_padding_mask,
        }
    )["logits"]
    return F.log_softmax(logits, dim=-1).reshape(memory_context.size(0), -1)


def oracle_mask_logits(
    model,
    memory_context: torch.Tensor,
    memory_padding_mask: torch.Tensor,
    target_tokens: torch.Tensor,
    mask_count: int,
) -> torch.Tensor:
    target_tokens = target_tokens.to(memory_context.device)
    current_targets = target_tokens.clone()
    if mask_count > 0:
        current_targets[:, -mask_count:] = model.mask_token_id
    logits = model.forward_denoiser_only(
        {
            "target_tokens": current_targets,
            "memory_context": memory_context,
            "memory_padding_mask": memory_padding_mask,
        }
    )["logits"]
    return logits


def update_digit_stats(stats: dict, logits: torch.Tensor, target_tokens: torch.Tensor, codebook_size: int) -> None:
    offsets = torch.arange(target_tokens.size(1), device=target_tokens.device) * codebook_size + 1
    local_targets = target_tokens - offsets.unsqueeze(0)
    top5 = logits.topk(5, dim=-1).indices
    top1 = top5[:, :, 0]

    for digit in range(target_tokens.size(1)):
        cur_target = local_targets[:, digit]
        stats[digit]["n"] += int(cur_target.numel())
        stats[digit]["top1"] += int((top1[:, digit] == cur_target).sum().item())
        stats[digit]["top5"] += int((top5[:, digit, :] == cur_target.unsqueeze(-1)).any(dim=-1).sum().item())
        stats[digit]["pred_counter"].update(top1[:, digit].detach().cpu().tolist())


def summarize_digit_stats(stats: dict, codebook_size: int, top_codes: int = 5) -> list[dict]:
    rows = []
    for digit in sorted(stats):
        n = stats[digit]["n"]
        most_common = stats[digit]["pred_counter"].most_common(top_codes)
        top_code_share = most_common[0][1] / n if n else float("nan")
        rows.append(
            {
                "digit": digit,
                "n": n,
                "top1_accuracy": stats[digit]["top1"] / n if n else float("nan"),
                "top5_accuracy": stats[digit]["top5"] / n if n else float("nan"),
                "unique_top1_codes": len(stats[digit]["pred_counter"]),
                "unique_top1_code_share": len(stats[digit]["pred_counter"]) / codebook_size,
                "most_common_top1_codes": [{"code": int(code), "count": int(count)} for code, count in most_common],
                "most_common_top1_share": top_code_share,
            }
        )
    return rows


def summarize_metrics(metric_rows: list[dict]) -> dict:
    keys = sorted(metric_rows[0]) if metric_rows else []
    return {key: mean([row[key] for row in metric_rows]) for key in keys}


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    prepare_genrec_runtime("DRPG")

    from genrec.utils import get_config, get_dataset, get_model, get_tokenizer, get_trainer, init_device, init_seed

    config_dict = {
        "eval_batch_size": args.batch_size,
        "num_proc": 1,
        "run_id": "drpg_pregraph_diagnostics",
        "diffusion_mask_counts": "32,24,16,8,4,1",
        "lr": 0.003,
    }
    if args.device:
        config_dict["device"] = torch.device(args.device)

    config = get_config(
        model_name="DRPG",
        dataset_name="AmazonReviews2014",
        config_file=[str(ROOT_CONFIG), str(BEAUTY_CONFIG)],
        config_dict=config_dict,
    )
    config["device"], config["use_ddp"] = init_device()
    if args.device:
        config["device"] = torch.device(args.device)
    config["accelerator"] = Accelerator()
    init_seed(config["rand_seed"], config["reproducibility"])

    raw_dataset = get_dataset("AmazonReviews2014")(config)
    split_datasets = raw_dataset.split()
    tokenizer = get_tokenizer("DRPG")(config, raw_dataset)
    tokenized_val = tokenizer.tokenize({"val": split_datasets["val"]})["val"]

    model = get_model("DRPG")(config, raw_dataset, tokenizer)
    model.load_state_dict(torch.load(checkpoint, map_location=config["device"]))
    trainer = get_trainer("DRPG")(config, model, tokenizer)

    device = config["device"]
    model = model.to(device)
    model.eval()
    model.generate_w_decoding_graph = False
    model.init_flag = False

    n_examples = min(args.max_examples, len(tokenized_val))
    val_subset = Subset(tokenized_val, range(n_examples))
    dataloader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=tokenizer.collate_fn["val"],
    )

    raw_rows = []
    one_shot_rows = []
    digit_stats = defaultdict(lambda: {"n": 0, "top1": 0, "top5": 0, "pred_counter": Counter()})
    oracle_stats = {
        mask_count: defaultdict(lambda: {"n": 0, "top1": 0, "top5": 0, "pred_counter": Counter()})
        for mask_count in model.mask_counts
    }
    confidence_rows = []

    for batch in tqdm(dataloader, desc="Pre-graph diagnostics"):
        batch = {key: value.to(device) for key, value in batch.items()}
        labels = batch["labels"]
        target_tokens = model.item_id2tokens[labels.squeeze(-1)].to(device)

        with torch.no_grad():
            model.generate_w_decoding_graph = False
            raw_preds = model.generate(batch, n_return_sequences=10)
            raw_results = trainer.evaluator.calculate_metrics(raw_preds, labels)
            raw_rows.append({key: value.float().mean().item() for key, value in raw_results.items()})

            outputs = model.forward(batch, return_loss=False)
            one_shot_logprobs = one_shot_token_logprobs(model, outputs.memory_context, outputs.memory_padding_mask)
            one_shot_preds = item_preds_from_token_logprobs(model, one_shot_logprobs, k=10).unsqueeze(-1)
            one_shot_results = trainer.evaluator.calculate_metrics(one_shot_preds, labels)
            one_shot_rows.append({key: value.float().mean().item() for key, value in one_shot_results.items()})

            one_shot_logits = one_shot_logprobs.reshape(labels.size(0), model.n_digit, model.codebook_size)
            update_digit_stats(digit_stats, one_shot_logits, target_tokens, model.codebook_size)

            probs = one_shot_logits.softmax(dim=-1)
            confidence_rows.append(
                {
                    "mean_top1_confidence": probs.max(dim=-1).values.mean().item(),
                    "mean_entropy": (-(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)).mean().item(),
                }
            )

            for mask_count in model.mask_counts:
                logits = oracle_mask_logits(
                    model,
                    outputs.memory_context,
                    outputs.memory_padding_mask,
                    target_tokens,
                    mask_count,
                )
                if mask_count == 0:
                    continue
                masked_positions = torch.zeros_like(target_tokens, dtype=torch.bool)
                masked_positions[:, -mask_count:] = True
                masked_logits = logits[masked_positions].reshape(labels.size(0), mask_count, model.codebook_size)
                masked_targets = target_tokens[masked_positions].reshape(labels.size(0), mask_count)
                start_digit = model.n_digit - mask_count
                offsets = torch.arange(start_digit, model.n_digit, device=device) * model.codebook_size + 1
                local_targets = masked_targets - offsets.unsqueeze(0)
                top5 = masked_logits.topk(5, dim=-1).indices
                top1 = top5[:, :, 0]
                for local_idx, digit in enumerate(range(start_digit, model.n_digit)):
                    cur_target = local_targets[:, local_idx]
                    oracle_stats[mask_count][digit]["n"] += int(cur_target.numel())
                    oracle_stats[mask_count][digit]["top1"] += int((top1[:, local_idx] == cur_target).sum().item())
                    oracle_stats[mask_count][digit]["top5"] += int(
                        (top5[:, local_idx, :] == cur_target.unsqueeze(-1)).any(dim=-1).sum().item()
                    )
                    oracle_stats[mask_count][digit]["pred_counter"].update(top1[:, local_idx].detach().cpu().tolist())

    raw_summary = summarize_metrics(raw_rows)
    one_shot_summary = summarize_metrics(one_shot_rows)
    digit_rows = summarize_digit_stats(digit_stats, model.codebook_size)

    oracle_rows = []
    for mask_count, stats in oracle_stats.items():
        rows = summarize_digit_stats(stats, model.codebook_size)
        if not rows:
            continue
        oracle_rows.append(
            {
                "mask_count": mask_count,
                "revealed_true_digits": model.n_digit - mask_count,
                "evaluated_digits": f"{model.n_digit - mask_count}-{model.n_digit - 1}",
                "mean_top1_accuracy": mean([row["top1_accuracy"] for row in rows]),
                "mean_top5_accuracy": mean([row["top5_accuracy"] for row in rows]),
                "mean_unique_top1_code_share": mean([row["unique_top1_code_share"] for row in rows]),
            }
        )

    output = {
        "metadata": {
            "checkpoint": str(checkpoint),
            "output_path": str(output_path),
            "max_examples_requested": args.max_examples,
            "n_examples": n_examples,
            "batch_size": args.batch_size,
            "device": str(device),
            "mask_counts": model.mask_counts,
            "n_digit": model.n_digit,
            "codebook_size": model.codebook_size,
            "n_items": model.dataset.n_items,
        },
        "raw_iterative_no_graph_metrics": raw_summary,
        "one_shot_all_masked_no_graph_metrics": one_shot_summary,
        "one_shot_digit_accuracy": digit_rows,
        "oracle_suffix_mask_accuracy": oracle_rows,
        "one_shot_confidence": summarize_metrics(confidence_rows),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["metadata"], indent=2))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
