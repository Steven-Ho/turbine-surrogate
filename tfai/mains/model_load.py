"""模型加载接口。"""

import copy
import os
from typing import Dict

# Imports `Predictor` from `tfai.prediction.predictor` for use in this module.
from tfai.prediction.predictor import Predictor
from tfai.utils.runtime_paths import resolve_runtime_paths
from tfai.utils.sample_utils import resolve_packaged_default_source_file
from tfai.utils.torch_utils import load_torch

# Imports `PROJECT_ROOT, clear_prediction_default_paths, ensure_prediction_runtime_paths` from `_shared` for use in this module.
from ._shared import (
    PROJECT_ROOT,
    clear_prediction_default_paths,
    ensure_prediction_runtime_paths,
# Continues the surrounding statement or collection definition.
)


def _resolve_checkpoint_path(model_path: str) -> str:
    """解析模型路径，兼容直接传 checkpoint 文件或模型目录。"""
    normalized_path = os.path.abspath(str(model_path))
    if os.path.isdir(normalized_path):
        # Stores the `candidate` for use by the remaining steps in this flow.
        candidate = os.path.join(normalized_path, "best_model.pt")
        if os.path.exists(candidate):
            return candidate
    return normalized_path


# 规范化 checkpoint 中保存的预测参数 schema。
def _normalize_parameter_schema(schema) -> list:
    """规范化 checkpoint 中保存的预测参数 schema。"""
    if not isinstance(schema, list):
        return []

    normalized = []
    # Visits each item in `schema` and applies the current processing step.
    for item in schema:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            if not name:
                # Handles the data or control-flow case represented by this statement.
                continue
            value_type = str(item.get("type") or "float").strip() or "float"
            normalized.append({"name": name, "type": value_type})
        else:
            # Stores the name for use by the remaining steps in this flow.
            name = str(item or "").strip()
            if name:
                normalized.append({"name": name, "type": "float"})
    return normalized


# Defines `_deduplicate_names` for  deduplicate names logic.
def _deduplicate_names(names) -> list:
    return list(dict.fromkeys(str(name) for name in (names or []) if str(name).strip()))


def _resolve_parameter_schema_from_metadata(task_cfg: Dict) -> list:
    """从模型自带元数据解析预测参数 schema，不读取训练样本表。"""
    for schema_candidate in (
        # Completes the multi-line `task_cfg.get` arguments or elements.
        task_cfg.get("parameter_schema"),
        task_cfg.get("input_parameter_schema"),
    ):
        schema = _normalize_parameter_schema(schema_candidate)
        # Returns early when `schema` makes the remaining work inapplicable.
        if schema:
            return schema

    input_variables = _deduplicate_names(task_cfg.get("input_variables") or [])
    return [{"name": name, "type": "float"} for name in input_variables]


# 加载预测所需的模型与上下文信息。
def load_model_for_prediction(
    model_path: str,
    config: Dict,
) -> Dict:
    """加载预测所需的模型与上下文信息。"""
    # Stores the checkpoint path for use by the remaining steps in this flow.
    checkpoint_path = _resolve_checkpoint_path(model_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型检查点不存在: {checkpoint_path}")

    predict_cfg = copy.deepcopy(config)
    # Stores the `predict_cfg` for use by the remaining steps in this flow.
    predict_cfg = clear_prediction_default_paths(predict_cfg)
    prediction_cfg = predict_cfg.setdefault("prediction", {})
    prediction_cfg["checkpoint_path"] = checkpoint_path

    checkpoint = load_torch(checkpoint_path, map_location="cpu", weights_only=False)
    # Reads `config` from the mapping into the `checkpoint_config` with its declared fallback.
    checkpoint_config = checkpoint.get("config", {})
    if not checkpoint_config:
        raise ValueError("检查点缺少 config，无法加载预测模型")

    extra_info = checkpoint.get("extra_info")
    # Checks `not isinstance(extra_info, dict)` before entering this conditional branch.
    if not isinstance(extra_info, dict):
        extra_info = {}

    runtime_config = copy.deepcopy(checkpoint_config)
    # 预测设备由本次预测配置控制，不依赖检查点保存的训练设备。
    predict_general_cfg = predict_cfg.get("general", {})
    if "device" in predict_general_cfg:
        runtime_config.setdefault("general", {})["device"] = predict_general_cfg["device"]
    runtime_config["prediction"] = {
        # Completes the multi-line `checkpoint_config.get` arguments or elements.
        **checkpoint_config.get("prediction", {}),
        **prediction_cfg,
    }
    runtime_config = resolve_runtime_paths(runtime_config, project_root=PROJECT_ROOT)
    # Stores the `runtime_config` for use by the remaining steps in this flow.
    runtime_config = ensure_prediction_runtime_paths(runtime_config, checkpoint_path)

    task_cfg = runtime_config.setdefault("task", {})
    parameter_schema = _resolve_parameter_schema_from_metadata(task_cfg)
    if parameter_schema and not task_cfg.get("input_variables"):
        # Builds the `task_cfg['input_variables']` using the comprehension's filter and mapping rules.
        task_cfg["input_variables"] = [item["name"] for item in parameter_schema]
    if not task_cfg.get("output_variables"):
        output_variables = (
            task_cfg.get("variables")
            # Completes the multi-line `task_cfg.get` arguments or elements.
            or task_cfg.get("logical_variables")
            or task_cfg.get("effective_variables")
            or []
        )
        # Stores the `task_cfg['output_variables']` for use by the remaining steps in this flow.
        task_cfg["output_variables"] = list(output_variables)

    predictor = Predictor(
        checkpoint_path=checkpoint_path,
        predict_config=runtime_config,
        # Completes the multi-line `Predictor` arguments or elements.
        logger=None,
        defer_reader_initialization=True,
    )
    predictor.config.setdefault("task", {}).update(
        # Completes the multi-line `collection` arguments or elements.
        {
            "input_variables": copy.deepcopy(task_cfg.get("input_variables", [])),
            "output_variables": copy.deepcopy(task_cfg.get("output_variables", [])),
            "parameter_schema": copy.deepcopy(parameter_schema),
        # Completes the multi-line `collection` arguments or elements.
        }
    )

    checkpoint_dir = os.path.dirname(checkpoint_path)
    reference_file = ""
    # Checks `not bool(getattr(predictor, "is_geometry_varying", False))` before entering this conditional branch.
    if not bool(getattr(predictor, "is_geometry_varying", False)):
        try:
            reference_file = predictor._resolve_reference_file()
        except Exception:
            # Stores the `reference_file` for use by the remaining steps in this flow.
            reference_file = ""
    default_source_file = resolve_packaged_default_source_file(checkpoint_dir)

    return {
        "model_path": checkpoint_path,
        # Completes the multi-line `collection` arguments or elements.
        "model_instance": predictor,
        "model_bundle": predictor,
        "extra_info": copy.deepcopy(extra_info),
        "parameter_schema": parameter_schema,
        # Completes the multi-line `collection` arguments or elements.
        "prediction_context": {
            "experiment": runtime_config.get("general", {}).get("experiment", ""),
            "reference_file": reference_file,
            "default_source_file": default_source_file,
            # Completes the multi-line `bool` arguments or elements.
            "is_geometry_varying": bool(getattr(predictor, "is_geometry_varying", False)),
        },
    }
