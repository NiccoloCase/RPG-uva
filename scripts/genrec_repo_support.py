#!/usr/bin/env python3

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Optional, Union

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_ROOT = REPO_ROOT / "third_party"
LOCAL_MODELS_ROOT = REPO_ROOT / "models"


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


def _resolve_model_config_path(model_name: Union[str, object], genrec_root: Path) -> Optional[Path]:
    if not isinstance(model_name, str):
        return None

    vendored_config = genrec_root / "models" / model_name / "config.yaml"
    if vendored_config.is_file():
        return vendored_config

    local_config = LOCAL_MODELS_ROOT / model_name / "config.yaml"
    if local_config.is_file():
        return local_config

    return None


def _build_get_config(genrec_utils_module):
    def get_config_with_repo_models(
        model_name,
        dataset_name,
        config_file,
        config_dict,
    ) -> dict:
        final_config = {}
        logger = genrec_utils_module.getLogger()
        genrec_root = Path(genrec_utils_module.__file__).resolve().parent

        config_file_list = [genrec_root / "default.yaml"]

        if isinstance(dataset_name, str):
            config_file_list.append(genrec_root / "datasets" / dataset_name / "config.yaml")
            final_config["dataset"] = dataset_name
        else:
            logger.info(
                'Custom dataset, whose config should be manually loaded and passed '
                'via "config_file" or "config_dict".'
            )
            final_config["dataset"] = dataset_name.__class__.__name__

        if isinstance(model_name, str):
            model_config = _resolve_model_config_path(model_name, genrec_root)
            if model_config is None:
                raise FileNotFoundError(f"Config file not found for model: {model_name}")
            config_file_list.append(model_config)
            final_config["model"] = model_name
        else:
            logger.info(
                'Custom model, whose config should be manually loaded and passed '
                'via "config_file" or "config_dict".'
            )
            final_config["model"] = model_name.__class__.__name__

        if config_file:
            if isinstance(config_file, str):
                config_file = [config_file]
            config_file_list.extend(Path(path) for path in config_file)

        for file in config_file_list:
            with open(file, "r", encoding="utf-8") as handle:
                cur_config = yaml.safe_load(handle)
            if cur_config is not None:
                final_config.update(cur_config)

        if config_dict:
            final_config.update(config_dict)

        final_config["run_local_time"] = genrec_utils_module.get_local_time()
        final_config = genrec_utils_module.convert_config_dict(final_config)
        return final_config

    return get_config_with_repo_models


def prepare_genrec_runtime(model_name: str) -> None:
    ensure_submodule_available()

    if str(THIRD_PARTY_ROOT) not in sys.path:
        sys.path.insert(0, str(THIRD_PARTY_ROOT))

    import genrec.models as genrec_models
    import genrec.utils as genrec_utils

    local_models_root_str = str(LOCAL_MODELS_ROOT)
    if local_models_root_str not in genrec_models.__path__:
        genrec_models.__path__.append(local_models_root_str)

    genrec_utils.get_config = _build_get_config(genrec_utils)

    local_model_dir = LOCAL_MODELS_ROOT / model_name
    if not local_model_dir.is_dir():
        return

    model_module = importlib.import_module(f"genrec.models.{model_name}.model")
    setattr(genrec_models, model_name, getattr(model_module, model_name))
