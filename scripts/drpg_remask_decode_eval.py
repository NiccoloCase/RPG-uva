#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/rpg/diagnostics/drpg_beauty_remask_decode_eval.json"
ROOT_CONFIG = REPO_ROOT / "configs/rpg/root.yaml"
BEAUTY_CONFIG = REPO_ROOT / "configs/rpg/repro/beauty.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DRPG remasking and prefix-constrained decoding policies from one checkpoint."
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="DRPG checkpoint to evaluate.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="JSON output path.")
    parser.add_argument("--max-examples", type=int, default=1024, help="Number of validation examples to evaluate.")
    parser.add_argument("--batch-size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Override device.")
    parser.add_argument(
        "--reveal-counts",
        default="8,4,2",
        help="Comma-separated counts of digits to reveal before suffix denoising.",
    )
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--random-oracle-seed",
        type=int,
        default=17,
        help="Seed for the oracle variant that reveals random true digit positions.",
    )
    parser.add_argument("--recent-history", type=int, default=3, help="How many recent history items seed graph candidates.")
    parser.add_argument("--graph-candidates-per-seed", type=int, default=200, help="Neighbors retained per history seed.")
    parser.add_argument("--include-graph", action="store_true", help="Build DRPG/RPG adjacency and run graph-prefix eval.")
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def parse_reveal_counts(raw_value: str, n_digit: int) -> list[int]:
    counts = []
    for part in raw_value.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0 or value >= n_digit:
            raise ValueError(f"Reveal count must be in [1, {n_digit - 1}], got {value}")
        counts.append(value)
    if not counts:
        raise ValueError("At least one reveal count is required")
    return sorted(set(counts), reverse=True)


def summarize_metric_rows(rows: list[dict]) -> dict:
    keys = sorted(rows[0]) if rows else []
    return {key: mean([row[key] for row in rows]) for key in keys}


def to_metric_row(results: dict) -> dict:
    return {key: value.float().mean().item() for key, value in results.items()}


def forward_logits(model, current_targets: torch.Tensor, memory_context: torch.Tensor, memory_padding_mask: torch.Tensor):
    return model.forward_denoiser_only(
        {
            "target_tokens": current_targets,
            "memory_context": memory_context,
            "memory_padding_mask": memory_padding_mask,
        }
    )["logits"]


def logits_to_flat_logprobs(logits: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits, dim=-1).reshape(logits.size(0), -1)


def fill_predictions(model, current_targets: torch.Tensor, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.arange(model.n_digit, device=current_targets.device) * model.codebook_size + 1
    probs = F.softmax(logits, dim=-1)
    confidence, pred_ids = probs.max(dim=-1)
    global_pred_ids = pred_ids + offsets.unsqueeze(0)
    filled = torch.where(current_targets == model.mask_token_id, global_pred_ids, current_targets)
    return filled, confidence


def confidence_for_current_tokens(model, current_targets: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    offsets = torch.arange(model.n_digit, device=current_targets.device) * model.codebook_size + 1
    probs = F.softmax(logits, dim=-1)
    safe_local = (current_targets - offsets.unsqueeze(0)).clamp(min=0, max=model.codebook_size - 1)
    token_confidence = probs.gather(dim=-1, index=safe_local.unsqueeze(-1)).squeeze(-1)
    max_confidence = probs.max(dim=-1).values
    return torch.where(current_targets == model.mask_token_id, max_confidence, token_confidence)


def step_monotonic(model, current_targets: torch.Tensor, logits: torch.Tensor, next_count: int) -> torch.Tensor:
    is_masked = current_targets == model.mask_token_id
    filled, confidence = fill_predictions(model, current_targets, logits)
    confidence = confidence.masked_fill(~is_masked, 1e9)
    if next_count > 0:
        keep_masked = torch.topk(confidence, k=next_count, dim=-1, largest=False).indices
        filled.scatter_(1, keep_masked, model.mask_token_id)
    return filled


def step_confidence_reselect(model, current_targets: torch.Tensor, logits: torch.Tensor, next_count: int) -> torch.Tensor:
    filled, _ = fill_predictions(model, current_targets, logits)
    token_confidence = confidence_for_current_tokens(model, filled, logits)
    if next_count > 0:
        keep_masked = torch.topk(token_confidence, k=next_count, dim=-1, largest=False).indices
        filled.scatter_(1, keep_masked, model.mask_token_id)
    return filled


def run_schedule(
    model,
    memory_context: torch.Tensor,
    memory_padding_mask: torch.Tensor,
    current_targets: torch.Tensor | None = None,
    mask_schedule: list[int] | None = None,
    remask: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = memory_context.size(0)
    if current_targets is None:
        current_targets = torch.full(
            (batch_size, model.n_digit),
            model.mask_token_id,
            dtype=torch.long,
            device=memory_context.device,
        )

    if mask_schedule is None:
        mask_schedule = model.mask_counts

    masked_counts = (current_targets == model.mask_token_id).sum(dim=1)
    if not torch.equal(masked_counts, masked_counts[:1].expand_as(masked_counts)):
        raise ValueError("All examples in the batch must share the same masked-count schedule")
    start_mask_count = int(masked_counts[0].item())
    start_index = mask_schedule.index(start_mask_count)

    final_logits = None
    for index in range(start_index, len(mask_schedule)):
        logits = forward_logits(model, current_targets, memory_context, memory_padding_mask)
        if index == len(mask_schedule) - 1:
            final_logits = logits
            break
        next_count = mask_schedule[index + 1]
        if remask:
            current_targets = step_confidence_reselect(model, current_targets, logits, next_count)
        else:
            current_targets = step_monotonic(model, current_targets, logits, next_count)

    final_state, _ = fill_predictions(model, current_targets, final_logits)
    return logits_to_flat_logprobs(final_logits), final_state


def score_items(model, token_logprobs: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
    token_index = model.item_id2tokens[item_ids] - 1
    return torch.gather(
        input=token_logprobs.unsqueeze(-2).expand(-1, item_ids.numel(), -1),
        dim=-1,
        index=token_index.unsqueeze(0).expand(token_logprobs.size(0), -1, -1),
    ).mean(dim=-1)


def topk_all_items(model, token_logprobs: torch.Tensor, k: int) -> torch.Tensor:
    item_ids = torch.arange(1, model.dataset.n_items, device=token_logprobs.device)
    scores = score_items(model, token_logprobs, item_ids)
    return scores.topk(k, dim=-1).indices + 1


def one_shot_logprobs(model, memory_context: torch.Tensor, memory_padding_mask: torch.Tensor) -> torch.Tensor:
    current_targets = torch.full(
        (memory_context.size(0), model.n_digit),
        model.mask_token_id,
        dtype=torch.long,
        device=memory_context.device,
    )
    logits = forward_logits(model, current_targets, memory_context, memory_padding_mask)
    return logits_to_flat_logprobs(logits)


def prefix_item_from_all_catalog(
    model,
    token_logprobs: torch.Tensor,
    prefix_len: int,
) -> torch.Tensor:
    item_ids = torch.arange(1, model.dataset.n_items, device=token_logprobs.device)
    token_index = model.item_id2tokens[item_ids, :prefix_len] - 1
    prefix_scores = torch.gather(
        input=token_logprobs.unsqueeze(-2).expand(-1, item_ids.numel(), -1),
        dim=-1,
        index=token_index.unsqueeze(0).expand(token_logprobs.size(0), -1, -1),
    ).mean(dim=-1)
    return prefix_scores.argmax(dim=-1) + 1


def build_token_tuple_to_item(model) -> dict[tuple[int, ...], int]:
    mapping = {}
    item_tokens = model.item_id2tokens.detach().cpu()
    for item_id in range(1, model.dataset.n_items):
        mapping.setdefault(tuple(int(value) for value in item_tokens[item_id].tolist()), item_id)
    return mapping


def history_item_ids_from_batch(batch: dict, token_tuple_to_item: dict[tuple[int, ...], int]) -> list[list[int]]:
    histories = []
    history_sid = batch["history_sid"].detach().cpu()
    history_mask = batch["history_mask"].detach().cpu()
    for row_tokens, row_mask in zip(history_sid, history_mask, strict=True):
        cur = []
        for tokens, is_valid in zip(row_tokens, row_mask, strict=True):
            if bool(is_valid):
                item_id = token_tuple_to_item.get(tuple(int(value) for value in tokens.tolist()))
                if item_id is not None:
                    cur.append(item_id)
        histories.append(cur)
    return histories


def graph_candidate_pools(
    model,
    histories: list[list[int]],
    recent_history: int,
    candidates_per_seed: int,
) -> tuple[list[torch.Tensor], list[int]]:
    pools = []
    sizes = []
    for history in histories:
        seen = set(history)
        seeds = history[-recent_history:]
        candidates = set()
        for seed in seeds:
            neighbors = model.adjacency[seed][:candidates_per_seed].detach().cpu().tolist()
            candidates.update(int(item_id) for item_id in neighbors if int(item_id) > 0)
        candidates.difference_update(seen)
        if not candidates:
            candidates = set(range(1, model.dataset.n_items))
        ordered = sorted(candidates)
        sizes.append(len(ordered))
        pools.append(torch.tensor(ordered, dtype=torch.long, device=model.item_id2tokens.device))
    return pools, sizes


def prefix_item_from_candidate_pools(
    model,
    token_logprobs: torch.Tensor,
    pools: list[torch.Tensor],
    prefix_len: int,
) -> torch.Tensor:
    chosen = []
    for batch_id, item_ids in enumerate(pools):
        token_index = model.item_id2tokens[item_ids, :prefix_len] - 1
        scores = torch.gather(
            input=token_logprobs[batch_id].unsqueeze(0).expand(item_ids.numel(), -1),
            dim=-1,
            index=token_index,
        ).mean(dim=-1)
        chosen.append(item_ids[scores.argmax()])
    return torch.stack(chosen)


def topk_from_candidate_pools(model, token_logprobs: torch.Tensor, pools: list[torch.Tensor], k: int) -> torch.Tensor:
    rows = []
    for batch_id, item_ids in enumerate(pools):
        token_index = model.item_id2tokens[item_ids] - 1
        scores = torch.gather(
            input=token_logprobs[batch_id].unsqueeze(0).expand(item_ids.numel(), -1),
            dim=-1,
            index=token_index,
        ).mean(dim=-1)
        top_count = min(k, item_ids.numel())
        selected = item_ids[scores.topk(top_count).indices]
        if top_count < k:
            pad = selected[-1:].expand(k - top_count)
            selected = torch.cat([selected, pad], dim=0)
        rows.append(selected)
    return torch.stack(rows, dim=0)


def masked_mean(matches: torch.Tensor, mask: torch.Tensor) -> float:
    count = int(mask.sum().item())
    if count == 0:
        return float("nan")
    return matches[mask].float().mean().item()


def target_item_top1_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return (preds[:, 0] == labels.squeeze(-1)).float().mean().item()


def completion_diagnostics(
    initial_state: torch.Tensor,
    final_state: torch.Tensor,
    target_tokens: torch.Tensor,
    reveal_mask: torch.Tensor,
    preds: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    initial_matches = initial_state == target_tokens
    final_matches = final_state == target_tokens
    return {
        "input_revealed_accuracy": masked_mean(initial_matches, reveal_mask),
        "completed_unrevealed_accuracy": masked_mean(final_matches, ~reveal_mask),
        "final_digit_accuracy": final_matches.float().mean().item(),
        "final_sid_exact_match": final_matches.all(dim=1).float().mean().item(),
        "target_item_top1_accuracy": target_item_top1_accuracy(preds, labels),
    }


def reveal_random_true_digits(
    model,
    target_tokens: torch.Tensor,
    reveal_count: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = torch.full_like(target_tokens, model.mask_token_id)
    random_scores = torch.rand(target_tokens.shape, device=target_tokens.device, generator=generator)
    reveal_cols = torch.topk(random_scores, k=reveal_count, dim=1, largest=False).indices
    state.scatter_(1, reveal_cols, torch.gather(target_tokens, dim=1, index=reveal_cols))
    return state, reveal_cols


def reveal_mask_for_prefix(target_tokens: torch.Tensor, reveal_count: int) -> torch.Tensor:
    reveal_mask = torch.zeros_like(target_tokens, dtype=torch.bool)
    reveal_mask[:, :reveal_count] = True
    return reveal_mask


def reveal_mask_for_columns(target_tokens: torch.Tensor, reveal_cols: torch.Tensor) -> torch.Tensor:
    reveal_mask = torch.zeros_like(target_tokens, dtype=torch.bool)
    reveal_mask.scatter_(1, reveal_cols, True)
    return reveal_mask


def build_progressive_mask_schedule(model, reveal_count: int) -> list[int]:
    start_mask_count = model.n_digit - reveal_count
    schedule = [start_mask_count]
    current = start_mask_count
    while current > reveal_count:
        current = max(reveal_count, current - reveal_count)
        if current != schedule[-1]:
            schedule.append(current)
    tail = [count for count in model.mask_counts if count < schedule[-1]]
    for count in tail:
        if count != schedule[-1]:
            schedule.append(count)
    return schedule


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
        "run_id": "drpg_remask_decode_eval",
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

    if args.include_graph:
        model.init_graph()
        model.init_flag = True

    token_tuple_to_item = build_token_tuple_to_item(model) if args.include_graph else {}

    n_examples = min(args.max_examples, len(tokenized_val))
    val_subset = Subset(tokenized_val, range(n_examples))
    dataloader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=tokenizer.collate_fn["val"],
    )

    metric_rows = defaultdict(list)
    diagnostic_rows = defaultdict(list)
    strategy_meta = {}
    candidate_rows = []
    maxk = trainer.evaluator.maxk
    if args.prefix_len is not None:
        reveal_counts = [args.prefix_len]
    else:
        reveal_counts = parse_reveal_counts(args.reveal_counts, model.n_digit)
    reveal_schedules = {reveal_count: build_progressive_mask_schedule(model, reveal_count) for reveal_count in reveal_counts}
    random_oracle_generator = torch.Generator(device=device)
    random_oracle_generator.manual_seed(args.random_oracle_seed)

    for batch in tqdm(dataloader, desc="Remask decode eval"):
        batch = {key: value.to(device) for key, value in batch.items()}
        labels = batch["labels"]
        target_tokens = model.item_id2tokens[labels.squeeze(-1)].to(device)

        with torch.no_grad():
            outputs = model.forward(batch, return_loss=False)
            one_shot = one_shot_logprobs(model, outputs.memory_context, outputs.memory_padding_mask)

            monotonic_logprobs, monotonic_state = run_schedule(
                model,
                outputs.memory_context,
                outputs.memory_padding_mask,
                remask=False,
            )
            preds = topk_all_items(model, monotonic_logprobs, maxk).unsqueeze(-1)
            strategy = "monotonic_current"
            strategy_meta[strategy] = {"reveal_count": 0, "reveal_policy": "current_monotonic"}
            metric_rows[strategy].append(to_metric_row(trainer.evaluator.calculate_metrics(preds, labels)))
            diagnostic_rows[strategy].append(
                completion_diagnostics(
                    initial_state=torch.full_like(target_tokens, model.mask_token_id),
                    final_state=monotonic_state,
                    target_tokens=target_tokens,
                    reveal_mask=torch.zeros_like(target_tokens, dtype=torch.bool),
                    preds=preds.squeeze(-1),
                    labels=labels,
                )
            )

            remask_logprobs, remask_state = run_schedule(
                model,
                outputs.memory_context,
                outputs.memory_padding_mask,
                remask=True,
            )
            preds = topk_all_items(model, remask_logprobs, maxk).unsqueeze(-1)
            strategy = "confidence_reselect_remask"
            strategy_meta[strategy] = {"reveal_count": 0, "reveal_policy": "confidence_remask"}
            metric_rows[strategy].append(to_metric_row(trainer.evaluator.calculate_metrics(preds, labels)))
            diagnostic_rows[strategy].append(
                completion_diagnostics(
                    initial_state=torch.full_like(target_tokens, model.mask_token_id),
                    final_state=remask_state,
                    target_tokens=target_tokens,
                    reveal_mask=torch.zeros_like(target_tokens, dtype=torch.bool),
                    preds=preds.squeeze(-1),
                    labels=labels,
                )
            )

            first_logits = forward_logits(
                model,
                torch.full_like(target_tokens, model.mask_token_id),
                outputs.memory_context,
                outputs.memory_padding_mask,
            )
            first_filled, _ = fill_predictions(model, torch.full_like(target_tokens, model.mask_token_id), first_logits)
            if args.include_graph:
                histories = history_item_ids_from_batch(batch, token_tuple_to_item)
                pools, pool_sizes = graph_candidate_pools(
                    model,
                    histories,
                    recent_history=args.recent_history,
                    candidates_per_seed=args.graph_candidates_per_seed,
                )

                graph_rerank_preds = topk_from_candidate_pools(model, one_shot, pools, maxk).unsqueeze(-1)
                graph_n_visited = torch.tensor(pool_sizes, dtype=torch.float32, device=labels.device)
                strategy = "graph_history_candidate_rerank"
                strategy_meta[strategy] = {"reveal_count": 0, "reveal_policy": "graph_candidate_rerank"}
                metric_rows[strategy].append(
                    to_metric_row(trainer.evaluator.calculate_metrics((graph_rerank_preds, graph_n_visited), labels))
                )
                label_list = labels.squeeze(-1).detach().cpu().tolist()
                candidate_rows.extend(
                    {
                        "pool_size": int(pool_size),
                        "true_item_in_pool": int(int(label) in set(pool.detach().cpu().tolist())),
                    }
                    for label, pool, pool_size in zip(label_list, pools, pool_sizes, strict=True)
                )

            for reveal_count in reveal_counts:
                mask_schedule = reveal_schedules[reveal_count]
                reveal_mask = reveal_mask_for_prefix(target_tokens, reveal_count)

                prefix_pred_state = torch.full_like(target_tokens, model.mask_token_id)
                prefix_pred_state[:, :reveal_count] = first_filled[:, :reveal_count]
                prefix_pred_logprobs, prefix_pred_final_state = run_schedule(
                    model,
                    outputs.memory_context,
                    outputs.memory_padding_mask,
                    current_targets=prefix_pred_state,
                    mask_schedule=mask_schedule,
                    remask=False,
                )
                preds = topk_all_items(model, prefix_pred_logprobs, maxk).unsqueeze(-1)
                strategy = f"predicted_prefix{reveal_count}_suffix"
                strategy_meta[strategy] = {"reveal_count": reveal_count, "reveal_policy": "predicted_prefix"}
                metric_rows[strategy].append(to_metric_row(trainer.evaluator.calculate_metrics(preds, labels)))
                diagnostic_rows[strategy].append(
                    completion_diagnostics(
                        prefix_pred_state,
                        prefix_pred_final_state,
                        target_tokens,
                        reveal_mask,
                        preds.squeeze(-1),
                        labels,
                    )
                )

                catalog_items = prefix_item_from_all_catalog(model, one_shot, reveal_count)
                catalog_state = torch.full_like(target_tokens, model.mask_token_id)
                catalog_state[:, :reveal_count] = model.item_id2tokens[catalog_items, :reveal_count]
                catalog_logprobs, catalog_final_state = run_schedule(
                    model,
                    outputs.memory_context,
                    outputs.memory_padding_mask,
                    current_targets=catalog_state,
                    mask_schedule=mask_schedule,
                    remask=False,
                )
                preds = topk_all_items(model, catalog_logprobs, maxk).unsqueeze(-1)
                strategy = f"catalog_prefix{reveal_count}_suffix"
                strategy_meta[strategy] = {"reveal_count": reveal_count, "reveal_policy": "catalog_prefix"}
                metric_rows[strategy].append(to_metric_row(trainer.evaluator.calculate_metrics(preds, labels)))
                diagnostic_rows[strategy].append(
                    completion_diagnostics(
                        catalog_state,
                        catalog_final_state,
                        target_tokens,
                        reveal_mask,
                        preds.squeeze(-1),
                        labels,
                    )
                )

                oracle_state = torch.full_like(target_tokens, model.mask_token_id)
                oracle_state[:, :reveal_count] = target_tokens[:, :reveal_count]
                oracle_logprobs, oracle_final_state = run_schedule(
                    model,
                    outputs.memory_context,
                    outputs.memory_padding_mask,
                    current_targets=oracle_state,
                    mask_schedule=mask_schedule,
                    remask=False,
                )
                preds = topk_all_items(model, oracle_logprobs, maxk).unsqueeze(-1)
                strategy = f"oracle_prefix{reveal_count}_suffix"
                strategy_meta[strategy] = {"reveal_count": reveal_count, "reveal_policy": "oracle_prefix"}
                metric_rows[strategy].append(to_metric_row(trainer.evaluator.calculate_metrics(preds, labels)))
                diagnostic_rows[strategy].append(
                    completion_diagnostics(
                        oracle_state,
                        oracle_final_state,
                        target_tokens,
                        reveal_mask,
                        preds.squeeze(-1),
                        labels,
                    )
                )

                random_oracle_state, random_reveal_cols = reveal_random_true_digits(
                    model,
                    target_tokens,
                    reveal_count,
                    random_oracle_generator,
                )
                random_oracle_reveal_mask = reveal_mask_for_columns(target_tokens, random_reveal_cols)
                random_oracle_logprobs, random_oracle_final_state = run_schedule(
                    model,
                    outputs.memory_context,
                    outputs.memory_padding_mask,
                    current_targets=random_oracle_state,
                    mask_schedule=mask_schedule,
                    remask=False,
                )
                preds = topk_all_items(model, random_oracle_logprobs, maxk).unsqueeze(-1)
                strategy = f"oracle_random{reveal_count}_suffix"
                strategy_meta[strategy] = {"reveal_count": reveal_count, "reveal_policy": "oracle_random"}
                metric_rows[strategy].append(to_metric_row(trainer.evaluator.calculate_metrics(preds, labels)))
                diagnostic_rows[strategy].append(
                    completion_diagnostics(
                        random_oracle_state,
                        random_oracle_final_state,
                        target_tokens,
                        random_oracle_reveal_mask,
                        preds.squeeze(-1),
                        labels,
                    )
                )

                if args.include_graph:
                    graph_prefix_items = prefix_item_from_candidate_pools(model, one_shot, pools, reveal_count)
                    graph_state = torch.full_like(target_tokens, model.mask_token_id)
                    graph_state[:, :reveal_count] = model.item_id2tokens[graph_prefix_items, :reveal_count]
                    graph_logprobs, graph_final_state = run_schedule(
                        model,
                        outputs.memory_context,
                        outputs.memory_padding_mask,
                        current_targets=graph_state,
                        mask_schedule=mask_schedule,
                        remask=False,
                    )
                    preds = topk_all_items(model, graph_logprobs, maxk).unsqueeze(-1)
                    strategy = f"graph_history_prefix{reveal_count}_suffix"
                    strategy_meta[strategy] = {"reveal_count": reveal_count, "reveal_policy": "graph_prefix"}
                    metric_rows[strategy].append(to_metric_row(trainer.evaluator.calculate_metrics(preds, labels)))
                    diagnostic_rows[strategy].append(
                        completion_diagnostics(
                            graph_state,
                            graph_final_state,
                            target_tokens,
                            reveal_mask,
                            preds.squeeze(-1),
                            labels,
                        )
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
            "reveal_counts": reveal_counts,
            "reveal_schedules": reveal_schedules,
            "random_oracle_seed": args.random_oracle_seed,
            "include_graph": args.include_graph,
            "recent_history": args.recent_history,
            "graph_candidates_per_seed": args.graph_candidates_per_seed,
            "n_digit": model.n_digit,
            "codebook_size": model.codebook_size,
            "n_items": model.dataset.n_items,
        },
        "strategy_metrics": [
            {"strategy": strategy, **strategy_meta.get(strategy, {}), **summarize_metric_rows(rows)}
            for strategy, rows in sorted(metric_rows.items())
        ],
        "decode_diagnostics": [
            {"strategy": strategy, **strategy_meta.get(strategy, {}), **summarize_metric_rows(rows)}
            for strategy, rows in sorted(diagnostic_rows.items())
        ],
        "prefix_accuracy": [
            {
                "strategy": strategy,
                **strategy_meta.get(strategy, {}),
                "revealed_digit_accuracy": summarize_metric_rows(rows).get("input_revealed_accuracy", float("nan")),
            }
            for strategy, rows in sorted(diagnostic_rows.items())
            if strategy_meta.get(strategy, {}).get("reveal_count", 0) > 0
        ],
        "graph_candidate_pool": {
            "mean_pool_size": mean([row["pool_size"] for row in candidate_rows]),
            "true_item_recall": mean([row["true_item_in_pool"] for row in candidate_rows]),
            "n_examples": len(candidate_rows),
        }
        if candidate_rows
        else None,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["metadata"], indent=2))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
