#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ABLATION_OUTPUT_ROOT = REPO_ROOT / "output" / "reproduction" / "sasrec_modernized" / "ablation_size"
DATASET_CHOICES = ("beauty", "cds_and_vinyl", "sports_and_outdoors", "toys_and_games")
RUN_ID_RE = re.compile(r"^(sasrec_modernized_[A-Za-z0-9_]+)$")
METRIC_RE = re.compile(
    r"OrderedDict\(\[\('Epoch', 0\), .*?\('NDCG@10', '([0-9.]+)'\).*?\('NDCG@20', '([0-9.]+)'\)\]\)"
)


@dataclass(frozen=True)
class AblationResult:
    dataset_slug: str
    run_id: str
    checkpoint_path: str
    test_ndcg10: float
    test_ndcg20: float
    source_log: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve the best completed SASRec Modernized ablation checkpoint from finished sweep logs.",
    )
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True, help="Dataset slug to resolve.")
    parser.add_argument(
        "--format",
        choices=("json", "shell", "text"),
        default="text",
        help="Output format.",
    )
    return parser.parse_args()


def _extract_result(path: Path) -> AblationResult | None:
    text = path.read_text(errors="ignore")
    if "Loaded best checkpoint for test:" not in text:
        return None

    checkpoint_path = None
    for line in text.splitlines():
        if line.startswith("Loaded best checkpoint for test: "):
            checkpoint_path = line.split(": ", 1)[1].strip()

    if not checkpoint_path:
        return None

    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = RUN_ID_RE.match(line.strip())
        if not match:
            continue
        run_id = match.group(1)
        dataset_slug = next((slug for slug in DATASET_CHOICES if slug in run_id), None)
        if dataset_slug is None or index + 1 >= len(lines):
            continue
        metric_match = METRIC_RE.search(lines[index + 1])
        if not metric_match:
            continue
        return AblationResult(
            dataset_slug=dataset_slug,
            run_id=run_id,
            checkpoint_path=checkpoint_path,
            test_ndcg10=float(metric_match.group(1)),
            test_ndcg20=float(metric_match.group(2)),
            source_log=str(path.resolve()),
        )
    return None


def find_best_result(dataset_slug: str) -> AblationResult:
    candidates: list[AblationResult] = []
    for path in sorted(ABLATION_OUTPUT_ROOT.rglob("*.out")):
        result = _extract_result(path)
        if result is None or result.dataset_slug != dataset_slug:
            continue
        candidates.append(result)

    if not candidates:
        raise FileNotFoundError(
            f"No completed SASRec Modernized ablation logs found for dataset '{dataset_slug}' under {ABLATION_OUTPUT_ROOT}."
        )

    candidates.sort(
        key=lambda row: (
            -row.test_ndcg20,
            -row.test_ndcg10,
            row.run_id,
        )
    )
    return candidates[0]


def emit_shell(result: AblationResult) -> None:
    payload = {
        "DATASET_SLUG": result.dataset_slug,
        "RUN_ID": result.run_id,
        "CHECKPOINT_PATH": result.checkpoint_path,
        "TEST_NDCG10": str(result.test_ndcg10),
        "TEST_NDCG20": str(result.test_ndcg20),
        "SOURCE_LOG": result.source_log,
    }
    for key, value in payload.items():
        print(f"{key}={shlex.quote(value)}")


def main() -> int:
    args = parse_args()
    try:
        result = find_best_result(args.dataset)
    except FileNotFoundError as exc:
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
                    f"{result.test_ndcg10:.4f}",
                    f"{result.test_ndcg20:.4f}",
                    result.source_log,
                )
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
