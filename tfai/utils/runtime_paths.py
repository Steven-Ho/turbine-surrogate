import copy
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from .path_utils import is_cross_platform_abs_path, normalize_relative_path, resolve_path_from_base


RUNTIME_CONTEXT_KEY = "_runtime"
PROJECT_ROOT_KEY = "project_root"
PROJECT_ROOT_ENV_VAR = "TFAI_PROJECT_ROOT"

TRAIN_PATH_FIELDS: Tuple[Tuple[str, ...], ...] = (
    ("general", "experiment"),
    ("general", "train_samples"),
    ("general", "train_processed_dir"),
    ("model", "checkpoint_path"),
    ("training", "resume", "path"),
)

PREDICT_PATH_FIELDS: Tuple[Tuple[str, ...], ...] = (
    ("prediction", "predict_samples"),
    ("prediction", "processed_dir"),
    ("prediction", "output_dir"),
)

EVALUATE_PATH_FIELDS: Tuple[Tuple[str, ...], ...] = (
    ("evaluate", "checkpoint_path"),
    ("evaluate", "evaluate_samples"),
    ("evaluate", "output_dir"),
)

RUNTIME_PATH_FIELDS: Tuple[Tuple[str, ...], ...] = (
    TRAIN_PATH_FIELDS + PREDICT_PATH_FIELDS + EVALUATE_PATH_FIELDS
)


def get_project_root() -> str:
    env_root = os.environ.get(PROJECT_ROOT_ENV_VAR)
    if env_root and str(env_root).strip():
        return os.path.abspath(str(env_root).strip())
    return str(Path(__file__).resolve().parents[2])


def attach_runtime_context(config: Dict, project_root: Optional[str] = None) -> Dict:
    runtime_cfg = copy.deepcopy(config)
    runtime_ctx = runtime_cfg.setdefault(RUNTIME_CONTEXT_KEY, {})
    runtime_ctx[PROJECT_ROOT_KEY] = os.path.abspath(project_root or get_project_root())
    return runtime_cfg


def resolve_runtime_paths(config: Dict, project_root: Optional[str] = None) -> Dict:
    runtime_cfg = attach_runtime_context(config, project_root=project_root)
    resolved_root = runtime_cfg[RUNTIME_CONTEXT_KEY][PROJECT_ROOT_KEY]

    for path_keys in RUNTIME_PATH_FIELDS:
        value = _get_nested_value(runtime_cfg, path_keys)
        if isinstance(value, str) and value.strip():
            _set_nested_value(
                runtime_cfg,
                path_keys,
                resolve_path_from_base(resolved_root, value),
            )

    return runtime_cfg


def sanitize_config_for_checkpoint(config: Dict) -> Dict:
    checkpoint_cfg = copy.deepcopy(config)
    runtime_ctx = checkpoint_cfg.pop(RUNTIME_CONTEXT_KEY, {})
    project_root = runtime_ctx.get(PROJECT_ROOT_KEY)

    if project_root:
        for path_keys in RUNTIME_PATH_FIELDS:
            value = _get_nested_value(checkpoint_cfg, path_keys)
            if isinstance(value, str) and value.strip():
                _set_nested_value(
                    checkpoint_cfg,
                    path_keys,
                    _to_project_relative_path(project_root, value),
                )

    return checkpoint_cfg


def _to_project_relative_path(project_root: str, path_value: str) -> str:
    raw = str(path_value).strip()
    if not raw or not is_cross_platform_abs_path(raw):
        return raw

    try:
        rel_path = os.path.relpath(raw, project_root)
    except ValueError:
        return raw

    if rel_path.startswith(".."):
        return raw

    return normalize_relative_path(rel_path)


def _get_nested_value(payload: Dict, path_keys: Iterable[str]):
    current = payload
    for key in path_keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _set_nested_value(payload: Dict, path_keys: Iterable[str], value) -> None:
    keys = list(path_keys)
    current = payload
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value
