from __future__ import annotations

import ast
import hashlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_ROOT = REPO_ROOT / "third_party"
ROOT_CONFIG = REPO_ROOT / "configs" / "rpg" / "root.yaml"
LOCAL_CONFIG = REPO_ROOT / "configs" / "rpg" / "local.yaml"


def ensure_submodule_available() -> None:
    expected_paths = [
        THIRD_PARTY_ROOT / "main.py",
        THIRD_PARTY_ROOT / "genrec" / "pipeline.py",
    ]
    missing_paths = [path for path in expected_paths if not path.exists()]
    if missing_paths:
        missing_str = ", ".join(str(path.relative_to(REPO_ROOT)) for path in missing_paths)
        raise SystemExit(
            f"Missing third_party sources: {missing_str}. "
            "Run 'git submodule update --init --recursive'."
        )


def parse_override_value(raw_value: str):
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null", "~"}:
        return None

    try:
        return ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        return raw_value


def parse_override_args(tokens: list[str]) -> dict:
    overrides = {}
    index = 0

    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected argument: {token}")

        body = token[2:]
        if not body:
            raise ValueError("Encountered an empty override flag.")

        if "=" in body:
            key, raw_value = body.split("=", 1)
            index += 1
        else:
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
                raise ValueError(
                    f"Invalid override '{token}'. Use '--key=value' or '--key value'."
                )
            key = body
            raw_value = tokens[index + 1]
            index += 2

        overrides[key.replace("-", "_")] = parse_override_value(raw_value)

    return overrides


def resolve_user_config(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    return path


def build_repo_config_files(
    extra_configs: list[str],
    include_root_config: bool = True,
    include_local_config: bool = True,
) -> list[str]:
    config_files: list[Path] = []

    if include_root_config and ROOT_CONFIG.is_file():
        config_files.append(ROOT_CONFIG)
    if include_local_config and LOCAL_CONFIG.is_file():
        config_files.append(LOCAL_CONFIG)

    for raw_path in extra_configs:
        config_files.append(resolve_user_config(raw_path))

    return [str(path) for path in config_files]


def parse_int_list(raw_value: str | None) -> list[int] | None:
    if raw_value is None:
        return None
    values = [part.strip() for part in raw_value.split(",") if part.strip()]
    return [int(value) for value in values]


def checkpoint_signature(checkpoint_path: str | Path) -> str:
    path = Path(checkpoint_path).expanduser().resolve()
    stat = path.stat()
    payload = f"{path}:{stat.st_size}:{int(stat.st_mtime)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
