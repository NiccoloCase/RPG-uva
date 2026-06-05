from __future__ import annotations

import csv
from pathlib import Path


def _load_summary_rows(input_path: str | Path) -> list[dict[str, str]]:
    path = Path(input_path).expanduser().resolve()
    if path.is_dir():
        path = path / "summaries" / "profile_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {path}")

    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def plot_summary_csv(input_path: str | Path, output_path: str | Path) -> Path:
    rows = _load_summary_rows(input_path)
    if not rows:
        raise ValueError("No rows found in the summary CSV.")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Update the project environment "
            "from environment.yml before running the plot command."
        ) from exc

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        method = row["method"]
        grouped.setdefault(method, []).append(row)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for method, method_rows in grouped.items():
        method_rows.sort(key=lambda row: int(row["pool_size"]))
        pool_sizes = [int(row["pool_size"]) for row in method_rows]
        epoch_time = [float(row["epoch_time_s_median"]) for row in method_rows]
        runtime_memory_key = "peak_cuda_runtime_delta_allocated_gb_median"
        runtime_memory = [
            float(row[runtime_memory_key]) if runtime_memory_key in row else float("nan")
            for row in method_rows
        ]
        total_peak_allocated = [
            float(row["peak_cuda_allocated_gb_median"]) for row in method_rows
        ]

        axes[0].plot(pool_sizes, epoch_time, marker="o", label=method)
        axes[1].plot(pool_sizes, runtime_memory, marker="o", label=method)
        axes[2].plot(pool_sizes, total_peak_allocated, marker="o", label=method)

    axes[0].set_title("Inference Time vs Item Pool Size")
    axes[0].set_xlabel("Item pool size")
    axes[0].set_ylabel("Epoch time (s)")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Peak CUDA Runtime Memory vs Item Pool Size")
    axes[1].set_xlabel("Item pool size")
    axes[1].set_ylabel("Peak CUDA runtime delta (GB)")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Peak CUDA Memory vs Item Pool Size")
    axes[2].set_xlabel("Item pool size")
    axes[2].set_ylabel("Peak CUDA allocated (GB)")
    axes[2].grid(True, alpha=0.3)

    for axis in axes:
        axis.legend()

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    return output
