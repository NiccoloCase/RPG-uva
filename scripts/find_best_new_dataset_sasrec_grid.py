#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GRID_RESULTS_PATH = REPO_ROOT / "results" / "sasrec_new_datasets_grid_results.md"
DATASET_CHOICES = ("video_games", "pet_supplies")
DATASET_TITLES = {
    "video_games": "Video Games",
    "pet_supplies": "Pet Supplies",
}
DATASET_CATEGORIES = {
    "video_games": "Video_Games",
    "pet_supplies": "Pet_Supplies",
}


@dataclass(frozen=True)
class GridBestResult:
    dataset_slug: str
    dataset_title: str
    dataset_category: str
    lr: float
    dropout: float
    blocks: int
    best_epoch: int
    ndcg10: float
    ndcg20: float
    recall10: float
    recall20: float
    source_log: str
    run_id: str
    checkpoint_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve the best SASRec grid setting for the new datasets from the checked-in markdown summary.",
    )
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True, help="Dataset slug to resolve.")
    parser.add_argument(
        "--format",
        choices=("json", "shell", "text"),
        default="text",
        help="Output format.",
    )
    return parser.parse_args()


def _extract_markdown_table_row(dataset_slug: str) -> GridBestResult:
    if not GRID_RESULTS_PATH.is_file():
        raise FileNotFoundError(f"Grid result summary not found: {GRID_RESULTS_PATH}")

    lines = GRID_RESULTS_PATH.read_text(encoding="utf-8").splitlines()
    dataset_title = DATASET_TITLES[dataset_slug]
    section_header = f"## {dataset_title}"

    in_section = False
    data_rows: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            in_section = stripped == section_header
            continue
        if not in_section or not stripped.startswith("|"):
            continue
        if stripped.startswith("| Rank ") or stripped.startswith("| --- "):
            continue
        data_rows.append(stripped)

    if not data_rows:
        raise FileNotFoundError(f"No grid table found for dataset '{dataset_slug}' in {GRID_RESULTS_PATH}")

    first = [cell.strip() for cell in data_rows[0].strip("|").split("|")]
    if len(first) != 10:
        raise ValueError(f"Unexpected markdown row format for dataset '{dataset_slug}': {data_rows[0]}")

    lr = float(first[1])
    dropout = float(first[2])
    blocks = int(first[3])
    best_epoch = int(first[4])
    ndcg10 = float(first[5])
    ndcg20 = float(first[6])
    recall10 = float(first[7])
    recall20 = float(first[8])
    source_log = first[9].strip("`")
    lr_raw = first[1]
    drop_raw = first[2]
    run_id = f"sasrec_grid_{dataset_slug}_lr{lr_raw}_d{drop_raw}_b{blocks}_s2024"
    checkpoint_path = str((REPO_ROOT / "artifacts" / "sasrec" / "ckpt" / f"{run_id}.pt").resolve())

    return GridBestResult(
        dataset_slug=dataset_slug,
        dataset_title=dataset_title,
        dataset_category=DATASET_CATEGORIES[dataset_slug],
        lr=lr,
        dropout=dropout,
        blocks=blocks,
        best_epoch=best_epoch,
        ndcg10=ndcg10,
        ndcg20=ndcg20,
        recall10=recall10,
        recall20=recall20,
        source_log=source_log,
        run_id=run_id,
        checkpoint_path=checkpoint_path,
    )


def emit_shell(result: GridBestResult) -> None:
    payload = {
        "DATASET_SLUG": result.dataset_slug,
        "DATASET_TITLE": result.dataset_title,
        "DATASET_CATEGORY": result.dataset_category,
        "LR": str(result.lr),
        "DROPOUT": str(result.dropout),
        "BLOCKS": str(result.blocks),
        "BEST_EPOCH": str(result.best_epoch),
        "NDCG10": str(result.ndcg10),
        "NDCG20": str(result.ndcg20),
        "RECALL10": str(result.recall10),
        "RECALL20": str(result.recall20),
        "SOURCE_LOG": result.source_log,
        "RUN_ID": result.run_id,
        "CHECKPOINT_PATH": result.checkpoint_path,
    }
    for key, value in payload.items():
        print(f"{key}={shlex.quote(value)}")


def main() -> int:
    args = parse_args()
    try:
        result = _extract_markdown_table_row(args.dataset)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(json.dumps(asdict(result), indent=2))
    elif args.format == "shell":
        emit_shell(result)
    else:
        print(
            "\t".join(
                (
                    result.dataset_slug,
                    result.run_id,
                    result.checkpoint_path,
                    f"{result.ndcg10:.4f}",
                    f"{result.ndcg20:.4f}",
                    result.source_log,
                )
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
