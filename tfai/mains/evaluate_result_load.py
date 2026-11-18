"""评估结果读取接口。"""

import copy
import os
from typing import Dict, Optional

# Imports `load_torch` from `tfai.utils.torch_utils` for use in this module.
from tfai.utils.torch_utils import load_torch


def _normalize_checkpoint_evaluate_result(
    payload: Optional[Dict],
    field_name: str,
# Continues the surrounding statement or collection definition.
) -> Optional[Dict]:
    """将 checkpoint 中的评估结果字段归一化为稳定结构。"""
    if payload is None:
        return None
    if not isinstance(payload, dict):
        # Raises `ValueError` to report why this path cannot continue.
        raise ValueError(f"检查点中的 {field_name} 必须为 dict")

    return {
        "metrics": dict(payload.get("metrics") or {}),
        "scatter_data": copy.deepcopy(
            # Completes the multi-line `payload.get` arguments or elements.
            payload.get("scatter_data")
            or {
                "overall": {"predictions": [], "targets": []},
                "by_variable": {},
            # Completes the multi-line `collection` arguments or elements.
            }
        ),
        "writeback_report": copy.deepcopy(payload.get("writeback_report") or {}),
        "samples_evaluated": list(payload.get("samples_evaluated") or []),
    # Completes the multi-line `collection` arguments or elements.
    }


def load_evaluate_results(
    model_path: str,
) -> Dict:
    """直接从 checkpoint 读取评估结果字段。"""
    # Normalizes the checkpoint path before subsequent file operations.
    checkpoint_path = os.path.abspath(str(model_path))
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型检查点不存在: {checkpoint_path}")

    checkpoint = load_torch(checkpoint_path, map_location="cpu", weights_only=False)
    # Validates `not isinstance(checkpoint, dict)` and raises `ValueError` when the input cannot be processed safely.
    if not isinstance(checkpoint, dict):
        raise ValueError("检查点内容不是合法的 dict，无法读取评估结果")

    return {
        "model_path": checkpoint_path,
        # Completes the multi-line `_normalize_checkpoint_evaluate_result` arguments or elements.
        "evaluate_result_train": _normalize_checkpoint_evaluate_result(
            checkpoint.get("evaluate_result_train"),
            field_name="evaluate_result_train",
        ),
        # Completes the multi-line `_normalize_checkpoint_evaluate_result` arguments or elements.
        "evaluate_result_valid": _normalize_checkpoint_evaluate_result(
            checkpoint.get("evaluate_result_valid"),
            field_name="evaluate_result_valid",
        ),
        # Completes the multi-line `_normalize_checkpoint_evaluate_result` arguments or elements.
        "evaluate_result_test": _normalize_checkpoint_evaluate_result(
            checkpoint.get("evaluate_result_test", checkpoint.get("evaluate_result")),
            field_name="evaluate_result_test",
        ),
    # Completes the multi-line `collection` arguments or elements.
    }
