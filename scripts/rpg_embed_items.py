#!/usr/bin/env python3
"""Embed item text with the OpenAI API, resumably.

Writes the same file the upstream RPG tokenizer writes and later reads:

    <cache>/AmazonReviews2014/<Category>/processed/<sent_emb_model>.sent_emb

a raw float32 array with one row per item, in item-id order (item 1 first).
Once this file exists, the tokenizer skips its own embedding step and goes
straight to PCA and OPQ.

Why not let the tokenizer embed: it keeps every embedding in memory and writes
the file only after the last request, so a single failed request loses all the
work already paid for. This script instead:

  - saves each batch to a part file as soon as it returns, and skips finished
    batches when restarted;
  - builds batches under the API's per-request token limit;
  - retries failed requests with a growing delay;
  - can report the token count and cost without calling the API (--dry-run).

Text handling matches the tokenizer: the same item sentences in the same order,
and the same character-based shortening for sentences above the model's
8192-token input limit.

Usage:
  python scripts/rpg_embed_items.py --preset books --dry-run
  python scripts/rpg_embed_items.py --preset books
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_ROOT = REPO_ROOT / "third_party" / "RPG_KDD2025"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from accelerate import Accelerator
from perf.config import build_repo_config_files, ensure_submodule_available, parse_override_args
from rpg_prepare_semantic_ids import PRESET_CONFIGS

MAX_INPUT_TOKENS = 8192
USD_PER_MILLION_TOKENS = {
    "text-embedding-3-large": 0.13,
    "text-embedding-3-small": 0.02,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed item text with the OpenAI API and write the RPG .sent_emb cache file.",
    )
    parser.add_argument("--preset", choices=sorted(PRESET_CONFIGS), help="Apply one of the checked-in dataset presets.")
    parser.add_argument("--config", action="append", default=[], help="Additional YAML config file. May be provided multiple times.")
    parser.add_argument("--no-root-config", action="store_true", help="Skip configs/rpg/root.yaml.")
    parser.add_argument("--no-local-config", action="store_true", help="Skip configs/rpg/local.yaml even if it exists.")
    parser.add_argument("--dry-run", action="store_true", help="Count tokens and estimate the cost. Makes no API call and writes nothing.")
    parser.add_argument("--max-batch-items", type=int, default=512, help="Maximum number of items per request.")
    parser.add_argument("--max-batch-tokens", type=int, default=250_000, help="Maximum total tokens per request (the API limit is 300,000).")
    parser.add_argument("--max-retries", type=int, default=8, help="Attempts per request before giving up.")
    parser.add_argument("--tokens-per-minute", type=int, default=None, help="Pace requests to stay under this many tokens per minute (the account's rate limit).")
    parser.add_argument("--empty-text", default="unknown", help="Text sent for items whose metadata sentence is missing or empty; the API rejects empty input.")
    return parser


def load_item_sentences(dataset) -> tuple[list[str], int]:
    """Item sentences in item-id order, as the tokenizer builds them.

    Returns the sentences and how many of them were missing or empty.
    """
    id2item = dataset.id_mapping["id2item"]
    sentences: list[str] = []
    n_missing = 0
    for item_id in range(1, dataset.n_items):
        sentence = dataset.item2meta.get(id2item[item_id], "")
        if not sentence.strip():
            n_missing += 1
        sentences.append(sentence)
    return sentences, n_missing


def shorten_long_sentences(sentences: list[str], token_counts: np.ndarray) -> list[str]:
    """Apply the tokenizer's rule for sentences at or above the input limit.

    The tokenizer cuts by characters, in proportion to the token overshoot.
    That is only an estimate, so a cut sentence can still be over the limit and
    the API then rejects the whole request. Any sentence still too long after
    the character cut is cut again by exact token count.
    """
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    shortened = list(sentences)
    for index in np.flatnonzero(token_counts >= MAX_INPUT_TOKENS):
        sentence = sentences[index]
        n_chars = MAX_INPUT_TOKENS / token_counts[index] * len(sentence) - 100
        candidate = sentence[: int(n_chars)]
        tokens = encoding.encode(candidate, disallowed_special=())
        if len(tokens) >= MAX_INPUT_TOKENS:
            candidate = encoding.decode(tokens[: MAX_INPUT_TOKENS - 1])
        shortened[index] = candidate
    return shortened


def count_tokens(sentences: list[str], num_threads: int = 16) -> np.ndarray:
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    encoded = encoding.encode_batch(sentences, num_threads=num_threads, disallowed_special=())
    return np.array([len(tokens) for tokens in encoded], dtype=np.int64)


def plan_batches(token_counts: np.ndarray, max_items: int, max_tokens: int) -> list[tuple[int, int]]:
    """Split items into consecutive ``[start, end)`` batches under both limits."""
    batches: list[tuple[int, int]] = []
    start = 0
    running_tokens = 0
    for index, n_tokens in enumerate(token_counts):
        too_many_items = index - start >= max_items
        too_many_tokens = running_tokens + int(n_tokens) > max_tokens
        if index > start and (too_many_items or too_many_tokens):
            batches.append((start, index))
            start = index
            running_tokens = 0
        running_tokens += int(n_tokens)
    if start < len(token_counts):
        batches.append((start, len(token_counts)))
    return batches


def embed_batch(client, model: str, texts: list[str], max_retries: int) -> np.ndarray:
    delay_seconds = 5.0
    for attempt in range(1, max_retries + 1):
        try:
            response = client.embeddings.create(input=texts, model=model)
            ordered = sorted(response.data, key=lambda row: row.index)
            return np.array([row.embedding for row in ordered], dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 - any API failure is retried
            # A rejected request (HTTP 400) fails the same way every time.
            if attempt == max_retries or type(exc).__name__ == "BadRequestError":
                raise
            print(f"request failed ({type(exc).__name__}: {exc}); retry {attempt}/{max_retries - 1} in {delay_seconds:.0f}s", flush=True)
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 2, 120.0)
    raise RuntimeError("unreachable")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, override_tokens = parser.parse_known_args(argv)

    ensure_submodule_available()
    config_files = build_repo_config_files(
        extra_configs=args.config,
        include_root_config=not args.no_root_config,
        include_local_config=not args.no_local_config,
    )
    if args.preset:
        config_files.append(str(PRESET_CONFIGS[args.preset]))
    config_overrides = parse_override_args(override_tokens)

    if str(THIRD_PARTY_ROOT) not in sys.path:
        sys.path.insert(0, str(THIRD_PARTY_ROOT))
    from genrec.utils import get_config, get_dataset, init_logger, init_seed

    config = get_config(
        model_name="RPG",
        dataset_name="AmazonReviews2014",
        config_file=config_files or None,
        config_dict=config_overrides or None,
    )
    accelerator = Accelerator(cpu=True)
    config["accelerator"] = accelerator
    config["device"] = accelerator.device
    config["use_ddp"] = False
    init_seed(config["rand_seed"], config["reproducibility"])
    init_logger(config)

    model_name = config["sent_emb_model"]
    if "text-embedding-3" not in model_name:
        raise SystemExit(f"This script only handles OpenAI text-embedding-3 models, got '{model_name}'.")

    dataset = get_dataset("AmazonReviews2014")(config)
    processed_dir = Path(dataset.cache_dir) / "processed"
    output_path = processed_dir / f"{Path(model_name).name}.sent_emb"
    parts_dir = processed_dir / f"{Path(model_name).name}.sent_emb.parts"
    print(f"sent_emb_cache={output_path}")
    if output_path.is_file():
        print("status=exists")
        return 0

    sentences, n_missing = load_item_sentences(dataset)
    sentences = [sentence if sentence.strip() else args.empty_text for sentence in sentences]
    token_counts = count_tokens(sentences)
    sentences = shorten_long_sentences(sentences, token_counts)
    n_shortened = int((token_counts >= MAX_INPUT_TOKENS).sum())
    billed_tokens = np.minimum(token_counts, MAX_INPUT_TOKENS)
    batches = plan_batches(billed_tokens, args.max_batch_items, args.max_batch_tokens)

    total_tokens = int(billed_tokens.sum())
    price = USD_PER_MILLION_TOKENS.get(Path(model_name).name)
    print(f"n_items={len(sentences)}")
    print(f"items_without_text={n_missing} (sent as '{args.empty_text}')")
    print(f"items_shortened_to_input_limit={n_shortened}")
    print(f"total_tokens={total_tokens}")
    print(f"n_requests={len(batches)}")
    if price is not None:
        print(f"estimated_cost_usd={total_tokens / 1e6 * price:.2f}")
    if args.dry_run:
        print("status=dry_run")
        return 0

    if not config.get("openai_api_key"):
        raise SystemExit("openai_api_key is not set. Put it in configs/rpg/local.yaml.")

    from openai import OpenAI

    client = OpenAI(api_key=config["openai_api_key"])
    parts_dir.mkdir(parents=True, exist_ok=True)
    plan_path = parts_dir / "plan.json"
    plan = {"n_items": len(sentences), "batches": batches, "model": model_name, "empty_text": args.empty_text}
    if plan_path.is_file():
        saved_plan = json.loads(plan_path.read_text())
        saved_plan["batches"] = [tuple(batch) for batch in saved_plan["batches"]]
        if saved_plan != plan:
            raise SystemExit(
                f"Existing part files in {parts_dir} were made with a different batch plan. "
                "Rerun with the same settings, or delete that folder to start over."
            )
    else:
        plan_path.write_text(json.dumps(plan))

    run_started = time.monotonic()
    tokens_sent = 0
    for batch_index, (start, end) in enumerate(batches):
        part_path = parts_dir / f"part_{start:08d}_{end:08d}.npy"
        if part_path.is_file():
            continue
        embeddings = embed_batch(client, model_name, sentences[start:end], args.max_retries)
        if args.tokens_per_minute:
            # Wait until the average rate since the start is back under the limit.
            tokens_sent += int(billed_tokens[start:end].sum())
            earliest_next = run_started + tokens_sent / args.tokens_per_minute * 60.0
            time.sleep(max(0.0, earliest_next - time.monotonic()))
        if embeddings.shape != (end - start, config["sent_emb_dim"]):
            raise RuntimeError(f"Unexpected embedding shape {embeddings.shape} for items [{start}, {end}).")
        temporary_path = part_path.with_suffix(".tmp.npy")
        np.save(temporary_path, embeddings)
        temporary_path.rename(part_path)
        if (batch_index + 1) % 20 == 0 or batch_index + 1 == len(batches):
            print(f"embedded {end}/{len(sentences)} items ({batch_index + 1}/{len(batches)} requests)", flush=True)

    all_embeddings = np.concatenate(
        [np.load(parts_dir / f"part_{start:08d}_{end:08d}.npy") for start, end in batches],
        axis=0,
    )
    if all_embeddings.shape != (len(sentences), config["sent_emb_dim"]):
        raise RuntimeError(f"Merged embeddings have shape {all_embeddings.shape}, expected ({len(sentences)}, {config['sent_emb_dim']}).")
    temporary_output = output_path.with_suffix(".sent_emb.tmp")
    all_embeddings.tofile(temporary_output)
    temporary_output.rename(output_path)
    print(f"status=created rows={all_embeddings.shape[0]} dim={all_embeddings.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
