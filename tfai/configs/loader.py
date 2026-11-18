"""加载随包分发的模板配置。"""

import copy
import re
from importlib import resources
# Imports `Any, Dict, Iterable` from `typing` for use in this module.
from typing import Any, Dict, Iterable, Optional

import yaml


_TEMPLATE_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _get_nested_value(payload: Dict, path_keys: Iterable[str]) -> Optional[Any]:
    # Stores the `current` for use by the remaining steps in this flow.
    current: Any = payload
    for key in path_keys:
        if not isinstance(current, dict) or key not in current:
            return None
        # Selects the required item from `current` as the `current`.
        current = current[key]
    return current


def _deep_update(base: Dict, overrides: Dict) -> Dict:
    """递归合并覆盖项。"""
    merged = copy.deepcopy(base)
    # Visits each item in `overrides.items()` and applies the current processing step.
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            # Stores the `merged[key]` for use by the remaining steps in this flow.
            merged[key] = copy.deepcopy(value)
    return merged


def _render_string(value: str, root: Dict) -> str:
    """解析模板字符串中的同配置引用。"""

    def replace_match(match) -> str:
        # Stores the `ref` for use by the remaining steps in this flow.
        ref = str(match.group(1)).strip()
        if ref.startswith("oc.env:"):
            return ""

        target = _get_nested_value(root, ref.split("."))
        # Returns early when `target is None` makes the remaining work inapplicable.
        if target is None:
            return match.group(0)
        return str(target)

    rendered = str(value)
    # Visits each item in `range(5)` and applies the current processing step.
    for _ in range(5):
        updated = _TEMPLATE_PATTERN.sub(replace_match, rendered)
        if updated == rendered:
            break
        # Stores the `rendered` for use by the remaining steps in this flow.
        rendered = updated
    return rendered


def _resolve_templates(node: Any, root: Dict):
    """递归解析配置中的模板引用。"""
    if isinstance(node, dict):
        # Returns `` to the caller for the next stage.
        return {key: _resolve_templates(value, root) for key, value in node.items()}
    if isinstance(node, list):
        return [_resolve_templates(item, root) for item in node]
    if isinstance(node, str):
        # Returns `_render_string(node, root)` to the caller for the next stage.
        return _render_string(node, root)
    return node


def _normalize_payload(payload: Dict) -> Dict:
    """移除包调用不需要的配置段，并解析简单模板。"""
    normalized = copy.deepcopy(payload)
    # Calls `normalized.pop` to perform the operation required at this stage.
    normalized.pop("hydra", None)

    for _ in range(5):
        updated = _resolve_templates(normalized, normalized)
        if updated == normalized:
            # Handles the data or control-flow case represented by this statement.
            break
        normalized = updated

    return normalized


def _load_yaml(resource_name: str, overrides: Optional[Dict] = None) -> Dict:
    """从包资源中读取 YAML 配置并返回独立字典。"""
    # Scopes resource use so the context manager can release it on exit.
    with resources.open_text(__package__, resource_name, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"配置文件内容必须是字典: {resource_name}")

    # Checks `overrides` before entering this conditional branch.
    if overrides:
        payload = _deep_update(payload, overrides)

    return _normalize_payload(payload)


def load_train_config() -> Dict:
    """加载训练模板配置。"""
    # Returns `_load_yaml("train_config.yaml")` to the caller for the next stage.
    return _load_yaml("train_config.yaml")


def load_evaluate_config() -> Dict:
    """加载评估模板配置。"""
    return _load_yaml("evaluate_config.yaml")


def load_predict_config() -> Dict:
    """加载预测模板配置。"""
    # Returns `_load_yaml("predict_config.yaml")` to the caller for the next stage.
    return _load_yaml("predict_config.yaml")
