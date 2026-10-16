#!/usr/bin/env python3
"""Train and test a MemGen-GR baseline (SASRec with softmax loss, or TIGER).

Thin launcher around the pinned MemGen-GR checkout in third_party/MemGen-GR
(Ding et al.). It does not modify that code. It only

  * loads the dataset preset from configs/memgen/<model>/<preset>.yaml on top of
    the MemGen-GR defaults,
  * points cache, checkpoint, log and tensorboard directories into artifacts/,
    and turns the Weights & Biases tracker off,
  * sets the random seed and a readable run id,
  * by default replaces the trainer's evaluation loop with a plain one that
    computes the same Recall/NDCG/ERR metrics without MemGen-GR's per-prediction
    memorisation/generalisation bookkeeping (pure Python, 10 lookups per user
    per validation pass; prohibitive on Books). Pass --fine-grained-eval to
    keep MemGen-GR's original evaluation.

The raw and processed Amazon data are shared with the RPG runs: MemGen-GR and
RPG use the same genrec dataset code and write the same files.

Usage:
  python scripts/memgen/train.py --model SASRec --preset sports
  python scripts/memgen/train.py --model TIGER --preset beauty --seed 2025
  python scripts/memgen/train.py --model SASRec --preset books --train_batch_size=32

  python scripts/memgen/train.py --model SASRec --preset sports --eval-only \
      --checkpoint artifacts/memgen/ckpt/<run>.pth --mask-history --run-tag removed

Extra ``--key=value`` tokens are forwarded to MemGen-GR as config overrides.
Test metrics are the ``Test Results`` line at the end of
artifacts/memgen/logs/AmazonReviews2014/<model>/<run_id>-*.log.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEMGEN_ROOT = REPO_ROOT / "third_party" / "MemGen-GR"
PRESETS = ("sports", "beauty", "toys", "cds", "books")
MODELS = ("SASRec", "TIGER")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=MODELS, help="MemGen-GR model name.")
    parser.add_argument("--preset", required=True, choices=PRESETS, help="Dataset preset under configs/memgen/<model>/.")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed (MemGen-GR rand_seed).")
    parser.add_argument("--run-tag", default="", help="Optional suffix for the run id.")
    parser.add_argument("--checkpoint", default=None, help="Evaluate this checkpoint instead of training.")
    parser.add_argument("--memgen-root", default=os.environ.get("MEMGEN_ROOT", str(DEFAULT_MEMGEN_ROOT)), help="MemGen-GR checkout.")
    parser.add_argument("--artifacts-dir", default=str(REPO_ROOT / "artifacts"), help="Root for cache/, memgen/ckpt, memgen/logs.")
    parser.add_argument("--fine-grained-eval", action="store_true", help="Keep MemGen-GR's memorisation/generalisation evaluation.")
    parser.add_argument("--eval-only", action="store_true", help="Test --checkpoint without training (MemGen-GR always trains otherwise).")
    parser.add_argument("--mask-history", action="store_true",
                        help="SASRec only: remove the items of the input sequence from the ranking before the top 10 "
                             "(the S3-Rec protocol). Default keeps them, as MemGen-GR does.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved configuration and exit.")
    return parser


def parse_overrides(tokens: list[str]) -> dict:
    """``--key=value`` tokens, with the same literal parsing MemGen-GR applies."""
    overrides = {}
    for token in tokens:
        if not token.startswith("--") or "=" not in token:
            raise SystemExit(f"Unknown argument {token!r}; config overrides look like --key=value.")
        key, value = token[2:].split("=", 1)
        try:
            value = eval(value)  # noqa: S307 - mirrors genrec.utils.parse_command_line_args
        except Exception:
            pass
        overrides[key] = value
    return overrides


def _masked_sasrec_topk(model, batch, k):
    """SASRec ``generate`` with the input items removed from the ranking.

    Same logits as ``SASRec.generate`` (last position of the GPT-2 output);
    the item ids present in the input sequence are set to -inf before the top-k.
    """
    import torch

    model = getattr(model, "module", model)
    outputs = model.gpt2(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = model.gather_index(outputs.logits, batch["seq_lens"] - 1)
    seen = batch["input_ids"].masked_fill(batch["attention_mask"] == 0, 0)
    logits = logits.scatter(1, seen, float("-inf"))
    return logits.topk(k, dim=-1).indices.unsqueeze(-1)


def plain_evaluate(self, dataloader, split="test", step=None, epoch=None):
    """MemGen-GR ``Trainer.evaluate`` without the fine-grained pattern statistics.

    Same predictions (``generate`` with 10 return sequences), same metric code
    (``Evaluator.calculate_metrics``), same CSV log of the training curve.
    """
    import torch
    from tqdm import tqdm

    self.model.eval()
    all_results = defaultdict(list)
    for batch in tqdm(dataloader, total=len(dataloader), desc=f"Eval - {split}"):
        with torch.no_grad():
            batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
            if self.config.get("mask_history"):
                all_preds, all_labels = _masked_sasrec_topk(self.model, batch, 10), batch["labels"]
            elif self.config["use_ddp"]:
                preds = self.model.module.generate(batch, n_return_sequences=10)
                all_preds, all_labels = self.accelerator.gather_for_metrics((preds, batch["labels"]))
            else:
                all_preds = self.model.generate(batch, n_return_sequences=10)
                all_labels = batch["labels"]
            if all_preds.dim() == 2:
                all_preds = all_preds.view(-1, 10, all_preds.size(-1))
            for key, value in self.evaluator.calculate_metrics(all_preds, all_labels).items():
                all_results[key].append(value)

    output_results = OrderedDict()
    for metric in self.config["metrics"]:
        for k in self.config["topk"]:
            key = f"{metric}@{k}"
            output_results[key] = torch.cat(all_results[key]).mean().item()
    if step is not None and epoch is not None and self.accelerator.is_main_process:
        self._append_eval_results_to_csv(output_results, step, epoch, split)
    return output_results


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, override_tokens = parser.parse_known_args(argv)
    memgen_root = Path(args.memgen_root).resolve()
    if not (memgen_root / "genrec" / "default.yaml").is_file():
        raise SystemExit(f"MemGen-GR not found at {memgen_root}. Run `git submodule update --init` or set --memgen-root.")

    preset_file = REPO_ROOT / "configs" / "memgen" / args.model.lower() / f"{args.preset}.yaml"
    if not preset_file.is_file():
        raise SystemExit(f"Missing preset {preset_file}")

    artifacts = Path(args.artifacts_dir).resolve()
    checkpoint = str(Path(args.checkpoint).resolve()) if args.checkpoint else None  # before changing directory
    run_id = f"memgen_{args.model.lower()}_{args.preset}_seed{args.seed}" + (f"_{args.run_tag}" if args.run_tag else "")
    config = {
        "cache_dir": str(artifacts / "rpg" / "cache"),
        "ckpt_dir": str(artifacts / "memgen" / "ckpt"),
        "log_dir": str(artifacts / "memgen" / "logs"),
        "tensorboard_log_dir": str(artifacts / "memgen" / "tensorboard"),
        "logging": False,  # no Weights & Biases; file logs are still written
        "eval_fine_grained": args.fine_grained_eval,
        "rand_seed": args.seed,
        "run_id": run_id,
    }
    config.update(parse_overrides(override_tokens))
    if args.mask_history:
        if args.model != "SASRec":
            raise SystemExit("--mask-history is only defined for SASRec.")
        if args.fine_grained_eval:
            raise SystemExit("--mask-history needs the plain evaluation loop.")
        config["mask_history"] = True
    if args.eval_only and not args.checkpoint:
        raise SystemExit("--eval-only needs --checkpoint.")

    if args.dry_run:
        print(f"memgen_root: {memgen_root}\npreset: {preset_file}\nconfig_dict: {config}")
        return 0

    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(memgen_root))  # `genrec` and `mem_gen_categorizer` are imported from the checkout root
    os.chdir(memgen_root)  # the dataset code resolves a few relative paths against the working directory

    from genrec.utils import get_pipeline

    if not args.fine_grained_eval:
        import genrec.trainer

        genrec.trainer.Trainer.evaluate = plain_evaluate

    pipeline = get_pipeline(args.model)(
        model_name=args.model,
        dataset_name="AmazonReviews2014",
        checkpoint_path=checkpoint,
        config_file=str(preset_file),
        config_dict=config,
    )
    if args.eval_only:
        from torch.utils.data import DataLoader

        loader = DataLoader(
            pipeline.tokenized_datasets["test"],
            batch_size=pipeline.config["eval_batch_size"],
            shuffle=False,
            collate_fn=pipeline.tokenizer.collate_fn["test"],
        )
        pipeline.model, loader = pipeline.accelerator.prepare(pipeline.model, loader)
        results = pipeline.trainer.evaluate(loader, split="test")
        pipeline.log(f"Test Results: {results}")
    else:
        pipeline.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
