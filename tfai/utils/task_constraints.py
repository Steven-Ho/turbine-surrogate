from typing import Dict


def validate_matrix_pod_fixed_geometry_only(config: Dict, stage: str) -> None:
    """校验 Matrix + POD 链路仅支持固定几何。"""
    task_cfg = config.get("task", {})
    model_cfg = config.get("model", {})

    model_type = str(model_cfg.get("type", "")).strip().upper()
    is_geometry_varying = bool(task_cfg.get("is_geometry_varying", False))

    if model_type != "POD":
        return
    if not is_geometry_varying:
        return

    raise NotImplementedError(
        f"Matrix + POD {stage} only supports fixed geometry inputs."
    )
