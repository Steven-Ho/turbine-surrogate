from typing import Any, Dict, List


def _deduplicate_names(names) -> List[str]:
    return list(dict.fromkeys(str(name) for name in (names or [])))


def resolve_scalar_target_variables(task_cfg: Dict[str, Any]) -> List[str]:
    """解析 scalar 任务的目标变量列表，兼容 variables/output_variables/target_variables。"""
    if not isinstance(task_cfg, dict):
        return []

    for key in ("target_variables", "output_variables", "variables"):
        resolved = _deduplicate_names(task_cfg.get(key) or [])
        if resolved:
            return resolved
    return []


def normalize_scalar_task_variable_aliases(config: Dict[str, Any]) -> Dict[str, Any]:
    """将 scalar 任务的变量配置规范化到统一键名。"""
    if not isinstance(config, dict):
        return config

    task_cfg = config.setdefault("task", {})
    if str(task_cfg.get("type", "field")).lower() != "scalar":
        return config

    target_variables = resolve_scalar_target_variables(task_cfg)
    if not target_variables:
        return config

    task_cfg["target_variables"] = list(target_variables)
    task_cfg["output_variables"] = list(target_variables)
    task_cfg["variables"] = list(target_variables)
    return config
