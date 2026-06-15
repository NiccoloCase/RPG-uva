"""Session and artifact-path helpers for graph-analysis runs.

Every run writes into a single session directory:

```
artifacts/rpg/graph_analysis/sports/<session>/
  graphs/
  static/
  manifest.json
```

Keeping graph caches and static outputs under the same session makes it easy to
inspect, rerun, and later attach dynamic/query-conditioned outputs without
mixing unrelated runs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .settings import REPO_ROOT


@dataclass(frozen=True)
class SessionPaths:
    """Concrete filesystem locations for one graph-analysis session."""

    root: Path
    graphs: Path
    static: Path
    manifest: Path


def resolve_repo_path(raw_path: str | Path) -> Path:
    """Resolve a path relative to the repository root.

    Args:
        raw_path: Absolute path, user-relative path, or repository-relative path.

    Returns:
        Absolute resolved path.
    """

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def graph_output_root(config: dict[str, Any]) -> Path:
    """Return the root directory that stores graph-analysis sessions."""

    return resolve_repo_path(
        config.get("graph_analysis_output_dir", "artifacts/rpg/graph_analysis/sports")
    )


def timestamped_session_name() -> str:
    """Create a sortable UTC session name, including the Slurm job id if present."""

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    return timestamp if not slurm_job_id else f"{timestamp}_job{slurm_job_id}"


def make_session_paths(root: Path) -> SessionPaths:
    """Create the standard subdirectories for a session and return their paths."""

    paths = SessionPaths(
        root=root,
        graphs=root / "graphs",
        static=root / "static",
        manifest=root / "manifest.json",
    )
    paths.graphs.mkdir(parents=True, exist_ok=True)
    paths.static.mkdir(parents=True, exist_ok=True)
    return paths


def create_session(config: dict[str, Any], raw_session_dir: str | None) -> SessionPaths:
    """Create a new session, or use an explicitly provided session directory."""

    if raw_session_dir:
        return make_session_paths(resolve_repo_path(raw_session_dir))

    output_root = graph_output_root(config)
    base_root = output_root / timestamped_session_name()
    session_root = base_root
    suffix = 1
    while session_root.exists():
        session_root = output_root / f"{base_root.name}_{suffix:02d}"
        suffix += 1
    return make_session_paths(session_root)


def latest_session(config: dict[str, Any], raw_session_dir: str | None) -> SessionPaths:
    """Resolve the session that should be used by the static command.

    If ``raw_session_dir`` is provided, that exact session must already contain
    graph metadata. Otherwise, the newest session under the configured output
    root that contains ``graphs/graph_metadata.json`` is selected.
    """

    if raw_session_dir:
        paths = make_session_paths(resolve_repo_path(raw_session_dir))
        if not graph_metadata_path(paths).is_file():
            raise FileNotFoundError(f"Graph metadata not found: {graph_metadata_path(paths)}")
        return paths

    output_root = graph_output_root(config)
    candidates = (
        sorted(
            path
            for path in output_root.iterdir()
            if path.is_dir() and (path / "graphs" / "graph_metadata.json").is_file()
        )
        if output_root.is_dir()
        else []
    )
    if not candidates:
        raise FileNotFoundError(
            f"No prepared graph sessions found under {output_root}. Run prepare-graph first."
        )
    return make_session_paths(candidates[-1])


def adjacency_path(paths: SessionPaths, topk: int) -> Path:
    """Return the saved adjacency tensor path for a given graph width."""

    return paths.graphs / f"adjacency_top{topk}.pt"


def graph_metadata_path(paths: SessionPaths) -> Path:
    """Return the metadata JSON path for a prepared graph."""

    return paths.graphs / "graph_metadata.json"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a deterministic, indented JSON payload."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write tabular rows to CSV, creating parent directories if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def append_or_update_manifest(paths: SessionPaths, updates: dict[str, Any]) -> dict[str, Any]:
    """Merge new fields into ``manifest.json`` and return the updated manifest."""

    manifest = json.loads(paths.manifest.read_text()) if paths.manifest.is_file() else {}
    manifest.update(updates)
    write_json(paths.manifest, manifest)
    return manifest

