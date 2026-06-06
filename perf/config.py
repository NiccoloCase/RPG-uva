from __future__ import annotations

import ast
import hashlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_ROOT = REPO_ROOT / "third_party"
ROOT_CONFIG = REPO_ROOT / "configs" / "rpg" / "root.yaml"
LOCAL_CONFIG = REPO_ROOT / "configs" / "rpg" / "local.yaml"


def ensure_submodule_available() -> None:
    """Validate that the upstream `third_party` checkout is present.

    The perf utilities import and instantiate the original RPG code from the
    vendored `third_party/genrec` tree. This helper checks for a couple of
    representative files that should exist when the submodule has been
    initialized correctly.

    Returns:
        None.

    Raises:
        SystemExit: If the expected third-party files are missing, with a hint
            to run `git submodule update --init --recursive`.
    """
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
    """Parse a CLI override value into a Python object when possible.

    The perf commands accept free-form `--key value` / `--key=value` overrides.
    This helper keeps common scalar values convenient by recognizing booleans,
    null-like strings, and Python literals such as integers, floats, lists, and
    dictionaries.

    Args:
        raw_value: Raw string value parsed from the CLI.

    Returns:
        A best-effort Python value. Strings remain strings when literal parsing
        is not appropriate.
    """
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
    """Convert unknown CLI tokens into a config override dictionary.

    Supported formats are `--key=value` and `--key value`. Hyphens in keys are
    normalized to underscores so they match the config keys used by the RPG
    codebase.

    Args:
        tokens: CLI tokens that were not consumed by the top-level argument
            parser.

    Returns:
        A dictionary mapping normalized override keys to parsed Python values.

    Raises:
        ValueError: If the override syntax is malformed.
    """
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
    """Resolve a user-supplied config path relative to the repository root.

    Args:
        raw_path: Absolute path or repository-relative path provided by the
            caller.

    Returns:
        An absolute, resolved `Path` pointing to an existing config file.

    Raises:
        FileNotFoundError: If the resolved path does not exist or is not a
            regular file.
    """
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
    """Build the ordered list of config files consumed by the perf harness.

    The helper mirrors the repository convention used by the CLI scripts:
    include the shared root config, optionally overlay the untracked local
    config, and finally append any explicit config files requested by the user.

    Args:
        extra_configs: Additional config files to append after the default repo
            configs.
        include_root_config: Whether to include `configs/rpg/root.yaml` when it
            exists.
        include_local_config: Whether to include `configs/rpg/local.yaml` when
            it exists.

    Returns:
        A list of absolute config file paths in merge order.
    """
    config_files: list[Path] = []

    if include_root_config and ROOT_CONFIG.is_file():
        config_files.append(ROOT_CONFIG)
    if include_local_config and LOCAL_CONFIG.is_file():
        config_files.append(LOCAL_CONFIG)

    for raw_path in extra_configs:
        config_files.append(resolve_user_config(raw_path))

    return [str(path) for path in config_files]


def parse_int_list(raw_value: str | None) -> list[int] | None:
    """Parse a comma-separated list of integers from CLI input.

    Args:
        raw_value: Raw comma-separated string, or `None` to indicate that the
            caller did not provide an override.

    Returns:
        A list of integers, or `None` when `raw_value` is `None`.
    """
    if raw_value is None:
        return None
    values = [part.strip() for part in raw_value.split(",") if part.strip()]
    return [int(value) for value in values]


def checkpoint_signature(checkpoint_path: str | Path) -> str:
    """Create a short cache key fragment for a model checkpoint file.

    The signature intentionally depends on file identity and file metadata
    rather than the full checkpoint contents so it is cheap to compute during
    profiling runs.

    Args:
        checkpoint_path: Path to a model checkpoint file.

    Returns:
        A short SHA1-based signature string suitable for filenames and cache
        metadata.
    """
    path = Path(checkpoint_path).expanduser().resolve()
    stat = path.stat()
    payload = f"{path}:{stat.st_size}:{int(stat.st_mtime)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
