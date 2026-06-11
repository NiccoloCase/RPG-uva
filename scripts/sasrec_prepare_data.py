#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gzip
import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


AMAZON_REVIEWS_URL = (
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_{category}_5.json.gz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare SASRec-ready Amazon user sequences in the CIKM2020-S3Rec text format.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["Beauty", "CDs_and_Vinyl", "Sports_and_Outdoors", "Toys_and_Games"],
        help="Amazon category names to preprocess.",
    )
    parser.add_argument("--data-root", default="artifacts/sasrec/data", help="Base directory for SASRec data.")
    parser.add_argument("--rating-score", type=float, default=0.0, help="Drop reviews with score <= this value.")
    parser.add_argument("--user-core", type=int, default=5, help="Minimum user interaction count.")
    parser.add_argument("--item-core", type=int, default=5, help="Minimum item interaction count.")
    parser.add_argument("--force", action="store_true", help="Regenerate processed outputs.")
    return parser.parse_args()


def download_if_missing(url: str, output_path: Path) -> None:
    if output_path.exists():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[sasrec_prepare_data] Downloading {url}")
    urllib.request.urlretrieve(url, output_path)


def load_reviews(raw_path: Path, min_rating: float) -> list[tuple[str, str, int]]:
    reviews: list[tuple[str, str, int]] = []
    with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if float(record.get("overall", 0.0)) <= min_rating:
                continue
            reviews.append(
                (
                    str(record["reviewerID"]),
                    str(record["asin"]),
                    int(record["unixReviewTime"]),
                )
            )
    return reviews


def build_user_sequences(reviews: list[tuple[str, str, int]]) -> dict[str, list[str]]:
    user_seq: dict[str, list[tuple[str, int]]] = {}
    for user_id, item_id, timestamp in reviews:
        user_seq.setdefault(user_id, []).append((item_id, timestamp))

    ordered: dict[str, list[str]] = {}
    for user_id, item_time in user_seq.items():
        item_time.sort(key=lambda x: x[1])
        ordered[user_id] = [item for item, _ in item_time]
    return ordered


def check_kcore(user_items: dict[str, list[str]], user_core: int, item_core: int):
    user_count = Counter()
    item_count = Counter()
    for user_id, items in user_items.items():
        user_count[user_id] += len(items)
        item_count.update(items)
    is_kcore = all(count >= user_core for count in user_count.values()) and all(
        count >= item_core for count in item_count.values()
    )
    return user_count, item_count, is_kcore


def filter_kcore(user_items: dict[str, list[str]], user_core: int, item_core: int) -> dict[str, list[str]]:
    filtered = {user_id: list(items) for user_id, items in user_items.items()}
    _, _, is_kcore = check_kcore(filtered, user_core, item_core)
    while not is_kcore:
        user_count, item_count, _ = check_kcore(filtered, user_core, item_core)
        next_filtered: dict[str, list[str]] = {}
        for user_id, items in filtered.items():
            if user_count[user_id] < user_core:
                continue
            kept_items = [item for item in items if item_count[item] >= item_core]
            if kept_items:
                next_filtered[user_id] = kept_items
        filtered = next_filtered
        _, _, is_kcore = check_kcore(filtered, user_core, item_core)
    return filtered


def remap_ids(user_items: dict[str, list[str]]):
    user2id: dict[str, int] = {}
    item2id: dict[str, int] = {}
    id2user: dict[int, str] = {}
    id2item: dict[int, str] = {}
    all_item_seqs: dict[int, list[int]] = {}

    next_user_id = 1
    next_item_id = 1
    for raw_user, items in user_items.items():
        if raw_user not in user2id:
            user2id[raw_user] = next_user_id
            id2user[next_user_id] = raw_user
            next_user_id += 1

        remapped_items: list[int] = []
        for raw_item in items:
            if raw_item not in item2id:
                item2id[raw_item] = next_item_id
                id2item[next_item_id] = raw_item
                next_item_id += 1
            remapped_items.append(item2id[raw_item])
        all_item_seqs[user2id[raw_user]] = remapped_items

    return all_item_seqs, {
        "user2id": user2id,
        "item2id": item2id,
        "id2user": id2user,
        "id2item": id2item,
    }


def write_dataset_file(output_path: Path, all_item_seqs: dict[int, list[int]]) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        for user_id in sorted(all_item_seqs):
            item_str = " ".join(str(item) for item in all_item_seqs[user_id])
            handle.write(f"{user_id} {item_str}\n")


def process_category(
    category: str,
    data_root: Path,
    rating_score: float,
    user_core: int,
    item_core: int,
    force: bool,
) -> None:
    dataset_root = data_root / category
    raw_dir = dataset_root / "raw"
    raw_path = raw_dir / f"reviews_{category}_5.json.gz"
    txt_path = dataset_root / f"{category}.txt"
    mapping_path = dataset_root / "id_mapping.json"
    summary_path = dataset_root / "summary.json"

    if force:
        for path in (txt_path, mapping_path, summary_path):
            if path.exists():
                path.unlink()

    download_if_missing(AMAZON_REVIEWS_URL.format(category=category), raw_path)

    reviews = load_reviews(raw_path, rating_score)
    user_items = build_user_sequences(reviews)
    filtered = filter_kcore(user_items, user_core, item_core)
    all_item_seqs, id_mapping = remap_ids(filtered)

    dataset_root.mkdir(parents=True, exist_ok=True)
    write_dataset_file(txt_path, all_item_seqs)
    with open(mapping_path, "w", encoding="utf-8") as handle:
        json.dump(id_mapping, handle, indent=2, sort_keys=True)

    interaction_lengths = [len(items) for items in all_item_seqs.values()]
    n_users = len(all_item_seqs)
    n_items = len(id_mapping["item2id"])
    n_interactions = sum(interaction_lengths)
    summary = {
        "category": category,
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,
        "avg_seq_len": (n_interactions / n_users) if n_users else 0.0,
        "min_seq_len": min(interaction_lengths) if interaction_lengths else 0,
        "max_seq_len": max(interaction_lengths) if interaction_lengths else 0,
        "rating_score": rating_score,
        "user_core": user_core,
        "item_core": item_core,
        "raw_reviews_path": str(raw_path),
        "dataset_file": str(txt_path),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(
        f"[sasrec_prepare_data] {category}: users={n_users}, items={n_items}, "
        f"interactions={n_interactions}, avg_len={summary['avg_seq_len']:.6f}"
    )
    print(f"[sasrec_prepare_data] wrote {txt_path}")


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    for category in args.categories:
        process_category(
            category=category,
            data_root=data_root,
            rating_score=args.rating_score,
            user_core=args.user_core,
            item_core=args.item_core,
            force=args.force,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
