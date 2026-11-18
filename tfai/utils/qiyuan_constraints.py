"""启元发行包的任务与模型边界。"""

from typing import Dict, Tuple


SUPPORTED_STEADY_MODELS: Tuple[str, ...] = (
    "POD",
    "UNet",
    "GCN",
    "Graph_UNet",
    "Graph_Transformer",
    "Transolver_3",
    "GeoTransolver",
)


def validate_qiyuan_config(config: Dict, model_type: str = "") -> str:
    """校验启元发行包仅允许稳态场任务和白名单模型。"""
    if not isinstance(config, dict):
        raise TypeError("启元配置必须是字典")

    task_cfg = config.get("task", {}) or {}
    task_type = str(task_cfg.get("type", "")).strip().lower()
    mode = str(task_cfg.get("mode", "")).strip().lower()
    if task_type != "field":
        raise ValueError("启元发行包仅支持 field 场任务")
    if mode != "steady":
        raise ValueError("启元发行包仅支持 steady 稳态场景")

    resolved_model_type = str(
        model_type or (config.get("model", {}) or {}).get("type", "")
    ).strip()
    if resolved_model_type not in SUPPORTED_STEADY_MODELS:
        supported = ", ".join(SUPPORTED_STEADY_MODELS)
        raise ValueError(
            f"模型 {resolved_model_type!r} 不在启元发行包白名单中，仅支持: {supported}"
        )

    training_cfg = config.get("training", {}) or {}
    multi_stage_cfg = training_cfg.get("multi_stage", {}) or {}
    if bool(multi_stage_cfg.get("enabled", False)):
        raise ValueError("启元发行包不支持多阶段残差训练")

    return resolved_model_type
