"""模型包预测能力校验接口。"""

import json
import os
import tempfile
# Imports `Path` from `pathlib` for use in this module.
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tfai.utils.torch_utils import load_torch

from .model_load import (
    # Continues the surrounding statement or collection definition.
    _resolve_parameter_schema_from_metadata,
    load_model_for_prediction,
)
from .predict import run_predict


# Initializes the `_VALID_SOURCE_MODES` as a collection for values from this processing stage.
_VALID_SOURCE_MODES = {"reference", "default_source", "external_source"}


def _new_item(
    name: str,
    status: str,
    # Continues the surrounding statement or collection definition.
    detail: str,
    **extra: Any,
) -> Dict[str, Any]:
    item = {"name": name, "status": status, "detail": detail}
    # Adds the current value to `item` for later aggregation.
    item.update({key: value for key, value in extra.items() if value is not None})
    return item


def _new_issue(
    code: str,
    # Continues the surrounding statement or collection definition.
    title: str,
    message: str,
    expected: str = "",
    actual: str = "",
    # Continues the surrounding statement or collection definition.
    suggestion: str = "",
    related_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        # Completes the multi-line `collection` arguments or elements.
        "code": code,
        "title": title,
        "message": message,
        "expected": expected,
        # Completes the multi-line `collection` arguments or elements.
        "actual": actual,
        "suggestion": suggestion,
        "related_paths": list(related_paths or []),
    }


# Defines `_section_status` for  section status logic.
def _section_status(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "skipped"
    statuses = {str(item.get("status", "")) for item in items}
    # Returns early when `"failed" in statuses` makes the remaining work inapplicable.
    if "failed" in statuses:
        return "failed"
    if "warning" in statuses or "missing" in statuses:
        return "warning"
    # Returns early when `"skipped" in statuses` makes the remaining work inapplicable.
    if "skipped" in statuses:
        return "skipped"
    return "passed"


def _section(message: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Returns `` to the caller for the next stage.
    return {
        "status": _section_status(items),
        "message": message,
        "items": items,
    # Completes the multi-line `collection` arguments or elements.
    }


def _empty_report(pt_path: str) -> Dict[str, Any]:
    normalized_path = os.path.abspath(str(pt_path or "")) if pt_path else ""
    return {
        # Completes the multi-line `collection` arguments or elements.
        "summary": {
            "status": "failed",
            "message": "模型包校验尚未完成。",
            "can_load": False,
            # Completes the multi-line `collection` arguments or elements.
            "can_predict_now": False,
            "next_action": "请查看校验详情。",
        },
        "model": {
            # Completes the multi-line `collection` arguments or elements.
            "pt_path": normalized_path,
            "model_dir": os.path.dirname(normalized_path) if normalized_path else "",
            "task": "",
            "geometry": "",
            # Completes the multi-line `collection` arguments or elements.
            "mode": "",
            "strategy": "",
            "processor": "",
            "model_type": "",
            # Completes the multi-line `collection` arguments or elements.
            "input_parameters": [],
            "output_variables": [],
        },
        "validation": {
            # Completes the multi-line `_section` arguments or elements.
            "checkpoint": _section("尚未检查 checkpoint。", []),
            "resources": _section("尚未检查模型包资源。", []),
            "smoke_test": _section("尚未执行预测冒烟测试。", []),
        },
        # Completes the multi-line `collection` arguments or elements.
        "errors": [],
        "warnings": [],
    }


def _append_error(
    # Completes the multi-line `collection` arguments or elements.
    report: Dict[str, Any],
    code: str,
    title: str,
    message: str,
    # Continues the surrounding statement or collection definition.
    **kwargs: Any,
) -> None:
    report["errors"].append(_new_issue(code, title, message, **kwargs))


def _append_warning(
    # Completes the multi-line `collection` arguments or elements.
    report: Dict[str, Any],
    code: str,
    title: str,
    message: str,
    # Continues the surrounding statement or collection definition.
    **kwargs: Any,
) -> None:
    report["warnings"].append(_new_issue(code, title, message, **kwargs))


def _load_json(path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    # Keeps the operation and its recovery path within the same exception boundary.
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        # Returns `None, str(exc)` to the caller for the next stage.
        return None, str(exc)
    if not isinstance(payload, dict):
        return None, "JSON 根节点必须为对象"
    return payload, None


# Defines `_derive_source_mode` for  derive source mode logic.
def _derive_source_mode(task_cfg: Dict[str, Any]) -> str:
    if bool(task_cfg.get("is_geometry_varying", False)):
        return "default_source"

    mode = str(task_cfg.get("mode", "steady")).strip().lower()
    # Stores the `strategy` for use by the remaining steps in this flow.
    strategy = str(task_cfg.get("temporal_strategy", "direct")).strip().lower()
    try:
        input_length = int(task_cfg.get("input_length", 1) or 1)
    except (TypeError, ValueError):
        # Stores the `input_length` for use by the remaining steps in this flow.
        input_length = 1
    if mode == "temporal" and strategy == "autoregressive" and input_length > 1:
        return "default_source"
    return "reference"


# Defines `_resolve_meta_default_source` for  resolve meta default source logic.
def _resolve_meta_default_source(meta: Dict[str, Any]) -> Dict[str, Any]:
    nested = meta.get("default_source")
    if isinstance(nested, dict):
        return nested
    # Returns `` to the caller for the next stage.
    return {
        key: meta.get(key)
        for key in (
            "entry_type",
            # Completes the multi-line `collection` arguments or elements.
            "entry_relpath",
            "original_source_path",
            "sample_order",
            "is_temporal",
        # Completes the multi-line `collection` arguments or elements.
        )
        if key in meta
    }


def _infer_parameter_schema_from_values(params: Dict[str, Any]) -> List[Dict[str, str]]:
    """从冒烟样本参数推断旧模型缺失的输入参数 schema。"""
    # Initializes the `schema` as a collection for values from this processing stage.
    schema: List[Dict[str, str]] = []
    for name, value in params.items():
        if isinstance(value, bool):
            value_type = "bool"
        # Checks `isinstance(value, int)` before entering this conditional branch.
        elif isinstance(value, int):
            value_type = "int"
        elif isinstance(value, float):
            value_type = "float"
        # Continues the surrounding statement or collection definition.
        else:
            value_type = "str"
        schema.append({"name": str(name), "type": value_type})
    return schema


# Defines `_resolve_package_path` for  resolve package path logic.
def _resolve_package_path(model_dir: str, relative_path: str) -> Optional[str]:
    relative = str(relative_path or "").strip()
    if not relative or os.path.isabs(relative):
        return None
    # Stores the `resolved_dir` for use by the remaining steps in this flow.
    resolved_dir = os.path.realpath(model_dir)
    candidate = os.path.realpath(os.path.join(resolved_dir, relative))
    try:
        if os.path.commonpath([resolved_dir, candidate]) != resolved_dir:
            # Returns `None` to the caller for the next stage.
            return None
    except ValueError:
        return None
    return candidate


# Defines `_find_reference_files` for  find reference files logic.
def _find_reference_files(model_dir: str) -> List[str]:
    try:
        return sorted(
            str(path)
            # Completes the multi-line `Path(model_dir).iterdir` arguments or elements.
            for path in Path(model_dir).iterdir()
            if path.is_file() and path.name.startswith("reference")
        )
    except OSError:
        # Returns `[]` to the caller for the next stage.
        return []


def _validate_smoke_params(
    parameter_schema: List[Dict[str, Any]],
    meta: Dict[str, Any],
# Completes the multi-line `collection` arguments or elements.
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    smoke_test = meta.get("smoke_test") if isinstance(meta, dict) else None
    raw_params = smoke_test.get("params") if isinstance(smoke_test, dict) else None
    params = dict(raw_params) if isinstance(raw_params, dict) else {}
    # Initializes the `items` as a collection for values from this processing stage.
    items: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    if parameter_schema and not isinstance(raw_params, dict):
        items.append(_new_item("冒烟测试参数", "failed", "缺少 smoke_test.params。"))
        # Adds the current value to `errors` for later aggregation.
        errors.append(
            _new_issue(
                "SMOKE_PARAMS_MISSING",
                "缺少冒烟测试参数",
                # Completes the multi-line `_new_issue` arguments or elements.
                "模型定义了输入参数，但元数据未提供 smoke_test.params。",
                expected="default_source_meta.json 中包含 smoke_test.params",
                actual="未找到参数对象",
                suggestion="重新训练并生成 v2 default_source_meta.json。",
            # Completes the multi-line `_new_issue` arguments or elements.
            )
        )
        return {}, items, errors

    # 旧 checkpoint 可能未持久化 parameter_schema，但 GraphProcessor 仍会将
    # 元数据中的全部参数拼入节点特征，因此仅在缺少 schema 时保留全量参数。
    normalized: Dict[str, Any] = {} if parameter_schema else dict(params)
    # Checks `not parameter_schema and isinstance(raw_params, dict) and params` before entering this conditional branch.
    if not parameter_schema and isinstance(raw_params, dict) and params:
        items.append(
            _new_item(
                "参数 schema",
                # Completes the multi-line `_new_item` arguments or elements.
                "warning",
                "Checkpoint 未声明输入参数 schema，冒烟测试已使用元数据中的全部参数。",
            )
        )
    # Visits each item in `parameter_schema` and applies the current processing step.
    for field in parameter_schema:
        name = str(field.get("name") or "").strip()
        value_type = str(field.get("type") or "float").strip().lower()
        if not name:
            # Handles the data or control-flow case represented by this statement.
            continue
        if name not in params:
            items.append(_new_item(name, "failed", "缺少必填预测参数。", expected=value_type))
            errors.append(
                # Completes the multi-line `_new_issue` arguments or elements.
                _new_issue(
                    "SMOKE_PARAMS_MISSING",
                    "缺少冒烟测试参数",
                    f"参数 {name} 未在 smoke_test.params 中提供。",
                    # Completes the multi-line `_new_issue` arguments or elements.
                    expected=f"提供类型为 {value_type} 的参数 {name}",
                    actual="参数缺失",
                    suggestion="补充 default_source_meta.json 中的 smoke_test.params。",
                )
            # Completes the multi-line `errors.append` arguments or elements.
            )
            continue
        value = params[name]
        try:
            # Checks `value_type in {"float", "double", "number"}` before entering this conditional branch.
            if value_type in {"float", "double", "number"}:
                normalized[name] = float(value)
            elif value_type in {"int", "integer"}:
                normalized[name] = int(value)
            # Checks `value_type in {"bool", "boolean"}` before entering this conditional branch.
            elif value_type in {"bool", "boolean"}:
                if isinstance(value, str) and value.strip().lower() not in {
                    "true",
                    "false",
                    # Completes the multi-line `collection` arguments or elements.
                    "1",
                    "0",
                }:
                    raise ValueError("布尔值仅支持 true/false/1/0")
                # Stores the `normalized[name]` for use by the remaining steps in this flow.
                normalized[name] = (
                    str(value).strip().lower() in {"true", "1"}
                    if isinstance(value, str)
                    else bool(value)
                # Continues the surrounding statement or collection definition.
                )
            else:
                normalized[name] = str(value)
        except (TypeError, ValueError) as exc:
            # Adds the current value to `items` for later aggregation.
            items.append(
                _new_item(name, "failed", "参数类型不合法。", expected=value_type, actual=value)
            )
            errors.append(
                # Completes the multi-line `_new_issue` arguments or elements.
                _new_issue(
                    "SMOKE_PARAMS_INVALID",
                    "冒烟测试参数类型错误",
                    f"参数 {name} 无法转换为 {value_type}: {exc}",
                    # Completes the multi-line `_new_issue` arguments or elements.
                    expected=value_type,
                    actual=str(value),
                    suggestion="修正 default_source_meta.json 中对应参数的值。",
                )
            # Completes the multi-line `errors.append` arguments or elements.
            )

    schema_names = {
        str(item.get("name") or "").strip()
        for item in parameter_schema
        # Completes the multi-line `str(item.get('name') or '').strip` arguments or elements.
        if str(item.get("name") or "").strip()
    }
    extra_names = sorted(set(params) - schema_names)
    if parameter_schema and extra_names:
        # Adds the current value to `items` for later aggregation.
        items.append(
            _new_item(
                "冗余参数",
                "failed",
                # Completes the multi-line `_new_item` arguments or elements.
                f"未在模型参数 schema 中定义: {extra_names}",
            )
        )
        errors.append(
            # Completes the multi-line `_new_issue` arguments or elements.
            _new_issue(
                "SMOKE_PARAMS_EXTRA",
                "冒烟测试包含冗余参数",
                f"smoke_test.params 包含未在模型输入 schema 中声明的参数: {extra_names}",
                # Completes the multi-line `sorted` arguments or elements.
                expected=f"仅提供模型声明的参数: {sorted(schema_names)}",
                actual=f"包含冗余参数: {extra_names}",
                suggestion="重新训练模型或修正 default_source_meta.json 中的 smoke_test.params。",
            )
        # Completes the multi-line `errors.append` arguments or elements.
        )
    if not items:
        items.append(_new_item("冒烟测试参数", "passed", "输入参数与模型 schema 一致。"))
    return normalized, items, errors


# Defines `_build_model_info` for  build model info logic.
def _build_model_info(
    checkpoint_path: str, checkpoint: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    config = checkpoint.get("config") if isinstance(checkpoint, dict) else {}
    # Stores the task settings for use by the remaining steps in this flow.
    task_cfg = config.get("task", {}) if isinstance(config, dict) else {}
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    schema = (
        _resolve_parameter_schema_from_metadata(task_cfg)
        # Completes the multi-line `isinstance` arguments or elements.
        if isinstance(task_cfg, dict)
        else []
    )
    output_variables = (
        # Completes the multi-line `list` arguments or elements.
        list(
            task_cfg.get("output_variables")
            or task_cfg.get("variables")
            or task_cfg.get("logical_variables")
            # Completes the multi-line `task_cfg.get` arguments or elements.
            or task_cfg.get("effective_variables")
            or []
        )
        if isinstance(task_cfg, dict)
        # Completes the multi-line `collection` arguments or elements.
        else []
    )
    info = {
        "pt_path": checkpoint_path,
        # Completes the multi-line `os.path.dirname` arguments or elements.
        "model_dir": os.path.dirname(checkpoint_path),
        "task": str(task_cfg.get("type") or ""),
        "geometry": "varying"
        if bool(task_cfg.get("is_geometry_varying", False))
        # Completes the multi-line `collection` arguments or elements.
        else "fixed",
        "mode": str(task_cfg.get("mode") or ""),
        "strategy": str(task_cfg.get("temporal_strategy") or ""),
        "processor": str(config.get("processor", {}).get("type") or "")
        # Completes the multi-line `isinstance` arguments or elements.
        if isinstance(config, dict)
        else "",
        "model_type": str(model_cfg.get("type") or "")
        if isinstance(model_cfg, dict)
        # Completes the multi-line `collection` arguments or elements.
        else "",
        "input_parameters": schema,
        "output_variables": output_variables,
    }
    # Returns `info, schema` to the caller for the next stage.
    return info, schema


def _finalize_summary(report: Dict[str, Any], smoke_executed: bool) -> Dict[str, Any]:
    if report["errors"]:
        report["summary"] = {
            # Completes the multi-line `collection` arguments or elements.
            "status": "failed",
            "message": "模型文件校验失败",
            "can_load": bool(report["validation"]["checkpoint"]["status"] == "passed"),
            "can_predict_now": False,
            # Completes the multi-line `report['errors'][0].get` arguments or elements.
            "next_action": report["errors"][0].get("suggestion") or "请修复校验失败项。",
        }
    elif smoke_executed:
        report["summary"] = {
            # Completes the multi-line `collection` arguments or elements.
            "status": "passed",
            "message": "模型文件校验成功",
            "can_load": True,
            "can_predict_now": True,
            # Completes the multi-line `collection` arguments or elements.
            "next_action": "可创建预测任务。",
        }
    else:
        report["summary"] = {
            # Completes the multi-line `collection` arguments or elements.
            "status": "external_source_required",
            "message": "模型文件校验通过，但预测时需要上传源文件或历史时序。",
            "can_load": True,
            "can_predict_now": False,
            # Completes the multi-line `collection` arguments or elements.
            "next_action": "预测请求中必须提供 source_file。",
        }
    return report


def validate_model_package(pt_path: str) -> Dict[str, Any]:
    """校验 field 模型包是否可使用同目录资源执行预测。"""
    # Stores the `report` for use by the remaining steps in this flow.
    report = _empty_report(pt_path)
    checkpoint_items: List[Dict[str, Any]] = []
    resource_items: List[Dict[str, Any]] = []
    smoke_items: List[Dict[str, Any]] = []
    # Selects the required item from `report["model"]` as the checkpoint path.
    checkpoint_path = report["model"]["pt_path"]

    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        checkpoint_items.append(
            _new_item("PT 文件", "failed", "指定 PT 文件不存在。", path=checkpoint_path)
        # Completes the multi-line `checkpoint_items.append` arguments or elements.
        )
        _append_error(
            report,
            "PT_NOT_FOUND",
            # Completes the multi-line `_append_error` arguments or elements.
            "模型文件不存在",
            "指定的 PT 文件不存在或不是普通文件。",
            expected="存在一个可读取的 .pt 文件",
            actual=checkpoint_path or "空路径",
            # Completes the multi-line `_append_error` arguments or elements.
            suggestion="确认上传解压目录和 PT 文件路径。",
            related_paths=[checkpoint_path] if checkpoint_path else [],
        )
        report["validation"]["checkpoint"] = _section(
            # Completes the multi-line `_section` arguments or elements.
            "Checkpoint 校验失败。", checkpoint_items
        )
        report["validation"]["resources"] = _section("未执行资源校验。", resource_items)
        report["validation"]["smoke_test"] = _section("未执行预测冒烟测试。", smoke_items)
        # Returns `_finalize_summary(report, smoke_executed=False)` to the caller for the next stage.
        return _finalize_summary(report, smoke_executed=False)

    if Path(checkpoint_path).suffix.lower() != ".pt":
        checkpoint_items.append(
            _new_item("PT 文件", "failed", "文件扩展名不是 .pt。", path=checkpoint_path)
        # Completes the multi-line `checkpoint_items.append` arguments or elements.
        )
        _append_error(
            report,
            "PT_EXTENSION_INVALID",
            # Completes the multi-line `_append_error` arguments or elements.
            "模型文件扩展名无效",
            "校验接口仅接受 .pt checkpoint 文件。",
            expected=".pt 文件",
            actual=Path(checkpoint_path).suffix or "无扩展名",
            # Completes the multi-line `_append_error` arguments or elements.
            suggestion="传入模型 checkpoint 的 .pt 文件路径。",
            related_paths=[checkpoint_path],
        )
        report["validation"]["checkpoint"] = _section(
            # Completes the multi-line `_section` arguments or elements.
            "Checkpoint 校验失败。", checkpoint_items
        )
        report["validation"]["resources"] = _section("未执行资源校验。", resource_items)
        report["validation"]["smoke_test"] = _section("未执行预测冒烟测试。", smoke_items)
        # Returns `_finalize_summary(report, smoke_executed=False)` to the caller for the next stage.
        return _finalize_summary(report, smoke_executed=False)

    try:
        checkpoint = load_torch(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        # Adds the current value to `checkpoint_items` for later aggregation.
        checkpoint_items.append(
            _new_item("PT 文件", "failed", "Checkpoint 无法读取。", error=str(exc))
        )
        _append_error(
            # Completes the multi-line `_append_error` arguments or elements.
            report,
            "PT_LOAD_FAILED",
            "模型文件无法读取",
            f"加载 checkpoint 失败: {exc}",
            # Completes the multi-line `_append_error` arguments or elements.
            suggestion="确认 PT 文件未损坏且与当前 PyTorch 版本兼容。",
            related_paths=[checkpoint_path],
        )
        report["validation"]["checkpoint"] = _section(
            # Completes the multi-line `_section` arguments or elements.
            "Checkpoint 校验失败。", checkpoint_items
        )
        report["validation"]["resources"] = _section("未执行资源校验。", resource_items)
        report["validation"]["smoke_test"] = _section("未执行预测冒烟测试。", smoke_items)
        # Returns `_finalize_summary(report, smoke_executed=False)` to the caller for the next stage.
        return _finalize_summary(report, smoke_executed=False)

    if not isinstance(checkpoint, dict):
        checkpoint_items.append(
            _new_item("Checkpoint 格式", "failed", "Checkpoint 顶层不是字典。")
        # Completes the multi-line `checkpoint_items.append` arguments or elements.
        )
        _append_error(
            report,
            "CHECKPOINT_FORMAT_INVALID",
            # Completes the multi-line `_append_error` arguments or elements.
            "Checkpoint 格式无效",
            "PT 文件内容不是平台支持的 checkpoint 字典。",
            expected="包含 config 和 model_state_dict 的字典",
            actual=type(checkpoint).__name__,
            # Completes the multi-line `_append_error` arguments or elements.
            suggestion="上传由当前平台训练流程导出的 best_model.pt。",
            related_paths=[checkpoint_path],
        )
        report["validation"]["checkpoint"] = _section(
            # Completes the multi-line `_section` arguments or elements.
            "Checkpoint 校验失败。", checkpoint_items
        )
        report["validation"]["resources"] = _section("未执行资源校验。", resource_items)
        report["validation"]["smoke_test"] = _section("未执行预测冒烟测试。", smoke_items)
        # Returns `_finalize_summary(report, smoke_executed=False)` to the caller for the next stage.
        return _finalize_summary(report, smoke_executed=False)

    report["model"], parameter_schema = _build_model_info(checkpoint_path, checkpoint)
    config = checkpoint.get("config")
    task_cfg = config.get("task") if isinstance(config, dict) else None
    # Checks `not isinstance(config, dict) or not isinstance(task_cfg, dict)` before entering this conditional branch.
    if not isinstance(config, dict) or not isinstance(task_cfg, dict):
        checkpoint_items.append(_new_item("预测配置", "failed", "Checkpoint 缺少 config。"))
        _append_error(
            report,
            # Completes the multi-line `_append_error` arguments or elements.
            "CHECKPOINT_CONFIG_MISSING",
            "缺少预测配置",
            "Checkpoint 缺少有效的 config.task，无法恢复预测链路。",
            suggestion="重新导出完整 best_model.pt。",
        # Completes the multi-line `_append_error` arguments or elements.
        )
    elif str(task_cfg.get("type") or "").strip().lower() != "field":
        checkpoint_items.append(
            _new_item(
                # Completes the multi-line `task_cfg.get` arguments or elements.
                "任务类型", "failed", "当前校验接口仅支持 field 任务。", actual=task_cfg.get("type")
            )
        )
        _append_error(
            # Completes the multi-line `_append_error` arguments or elements.
            report,
            "TASK_TYPE_UNSUPPORTED",
            "任务类型不受支持",
            "当前校验接口仅支持 field 任务。",
            # Completes the multi-line `_append_error` arguments or elements.
            expected="field",
            actual=str(task_cfg.get("type") or ""),
            suggestion="请使用 field 模型校验接口。",
        )
    # Continues the surrounding statement or collection definition.
    else:
        checkpoint_items.append(_new_item("预测配置", "passed", "已读取 field 任务配置。"))

    if "model_state_dict" not in checkpoint:
        checkpoint_items.append(
            # Completes the multi-line `_new_item` arguments or elements.
            _new_item("模型权重", "failed", "Checkpoint 缺少 model_state_dict。")
        )
        _append_error(
            report,
            # Completes the multi-line `_append_error` arguments or elements.
            "MODEL_STATE_MISSING",
            "缺少模型权重",
            "Checkpoint 缺少 model_state_dict，无法恢复模型。",
            suggestion="重新导出完整 best_model.pt。",
        # Completes the multi-line `_append_error` arguments or elements.
        )
    else:
        checkpoint_items.append(_new_item("模型权重", "passed", "Checkpoint 包含模型权重。"))

    if not checkpoint.get("norm_params"):
        # Adds the current value to `checkpoint_items` for later aggregation.
        checkpoint_items.append(
            _new_item("归一化参数", "failed", "Checkpoint 缺少 norm_params。")
        )
        _append_error(
            # Completes the multi-line `_append_error` arguments or elements.
            report,
            "NORM_PARAMS_MISSING",
            "缺少归一化参数",
            "预测预处理和反归一化需要 norm_params。",
            # Completes the multi-line `_append_error` arguments or elements.
            suggestion="重新导出训练完成后的 best_model.pt。",
        )
    else:
        checkpoint_items.append(_new_item("归一化参数", "passed", "Checkpoint 包含归一化参数。"))

    # Selects the required item from `report["model"]` as the `processor_type`.
    processor_type = report["model"]["processor"]
    if processor_type == "MatrixProcessor" and not checkpoint.get("modes"):
        checkpoint_items.append(
            _new_item("POD 模态", "failed", "MatrixProcessor checkpoint 缺少 modes。")
        # Completes the multi-line `checkpoint_items.append` arguments or elements.
        )
        _append_error(
            report,
            "POD_MODES_MISSING",
            # Completes the multi-line `_append_error` arguments or elements.
            "缺少 POD 模态",
            "MatrixProcessor 预测需要 checkpoint 中的 modes。",
            suggestion="重新导出包含 POD 模态的 best_model.pt。",
        )
    # Checks `processor_type == "MatrixProcessor"` before entering this conditional branch.
    elif processor_type == "MatrixProcessor":
        checkpoint_items.append(_new_item("POD 模态", "passed", "Checkpoint 包含 POD 模态。"))

    model_bundle = None
    if not report["errors"]:
        # Keeps the operation and its recovery path within the same exception boundary.
        try:
            model_bundle = load_model_for_prediction(
                checkpoint_path, config={"prediction": {}}
            )
            # Stores the `report['model']['processor']` for use by the remaining steps in this flow.
            report["model"]["processor"] = type(
                model_bundle["model_instance"].processor
            ).__name__
            checkpoint_items.append(_new_item("模型恢复", "passed", "模型实例和处理器已成功恢复。"))
        # Continues the surrounding statement or collection definition.
        except Exception as exc:
            checkpoint_items.append(
                _new_item("模型恢复", "failed", "模型实例恢复失败。", error=str(exc))
            )
            # Calls `_append_error` to perform the operation required at this stage.
            _append_error(
                report,
                "MODEL_BUILD_FAILED",
                "模型恢复失败",
                # Completes the multi-line `_append_error` arguments or elements.
                f"无法依据 checkpoint 恢复模型: {exc}",
                suggestion="确认模型类型、配置和权重与当前平台版本兼容。",
                related_paths=[checkpoint_path],
            )

    # Stores the `report['validation']['checkpoint']` for use by the remaining steps in this flow.
    report["validation"]["checkpoint"] = _section(
        "Checkpoint、配置与模型恢复校验。", checkpoint_items
    )
    if report["errors"]:
        # Stores the `report['validation']['resources']` for use by the remaining steps in this flow.
        report["validation"]["resources"] = _section(
            "因 checkpoint 校验失败，未执行资源校验。", resource_items
        )
        report["validation"]["smoke_test"] = _section(
            # Completes the multi-line `_section` arguments or elements.
            "因 checkpoint 校验失败，未执行预测冒烟测试。", smoke_items
        )
        return _finalize_summary(report, smoke_executed=False)

    model_dir = report["model"]["model_dir"]
    # Stores the `meta_path` for use by the remaining steps in this flow.
    meta_path = os.path.join(model_dir, "default_source_meta.json")
    meta: Dict[str, Any] = {}
    if os.path.isfile(meta_path):
        loaded_meta, meta_error = _load_json(meta_path)
        # Checks `meta_error` before entering this conditional branch.
        if meta_error:
            resource_items.append(
                _new_item(
                    "default_source_meta.json",
                    # Completes the multi-line `_new_item` arguments or elements.
                    "failed",
                    "元数据文件无效。",
                    path=meta_path,
                    error=meta_error,
                # Completes the multi-line `_new_item` arguments or elements.
                )
            )
            _append_error(
                report,
                # Completes the multi-line `_append_error` arguments or elements.
                "META_INVALID",
                "默认源元数据无效",
                f"default_source_meta.json 无法解析: {meta_error}",
                suggestion="修正或重新生成 default_source_meta.json。",
                # Completes the multi-line `collection` arguments or elements.
                related_paths=[meta_path],
            )
        else:
            meta = loaded_meta or {}
            # Adds the current value to `resource_items` for later aggregation.
            resource_items.append(
                _new_item(
                    "default_source_meta.json",
                    "passed",
                    # Completes the multi-line `_new_item` arguments or elements.
                    "元数据文件可读取。",
                    path=meta_path,
                    schema_version=meta.get("schema_version"),
                )
            # Completes the multi-line `resource_items.append` arguments or elements.
            )
            smoke_test = meta.get("smoke_test")
            smoke_params = (
                smoke_test.get("params") if isinstance(smoke_test, dict) else None
            # Continues the surrounding statement or collection definition.
            )
            if not parameter_schema and isinstance(smoke_params, dict) and smoke_params:
                report["model"][
                    "input_parameters"
                # Completes the multi-line `_infer_parameter_schema_from_values` arguments or elements.
                ] = _infer_parameter_schema_from_values(smoke_params)
                resource_items.append(
                    _new_item(
                        "输入参数 schema",
                        # Completes the multi-line `_new_item` arguments or elements.
                        "warning",
                        "Checkpoint 未保存参数 schema，已由冒烟样本参数推断。",
                    )
                )
                # Calls `_append_warning` to perform the operation required at this stage.
                _append_warning(
                    report,
                    "PARAMETER_SCHEMA_MISSING",
                    "Checkpoint 缺少输入参数 schema",
                    # Completes the multi-line `_append_warning` arguments or elements.
                    "前端参数定义已由 smoke_test.params 临时推断。",
                    suggestion="重新训练模型以将 parameter_schema 固化到 checkpoint。",
                    related_paths=[checkpoint_path, meta_path],
                )
    # Continues the surrounding statement or collection definition.
    else:
        resource_items.append(
            _new_item(
                "default_source_meta.json",
                # Completes the multi-line `_new_item` arguments or elements.
                "warning",
                "未找到默认源元数据，将依据 checkpoint 推导资源要求。",
                path=meta_path,
            )
        # Completes the multi-line `resource_items.append` arguments or elements.
        )
        _append_warning(
            report,
            "DEFAULT_SOURCE_META_MISSING",
            # Completes the multi-line `_append_warning` arguments or elements.
            "缺少默认源元数据",
            "未找到 default_source_meta.json，无法获得包内冒烟参数。",
            suggestion="重新训练并生成 v2 default_source_meta.json。",
            related_paths=[meta_path],
        # Completes the multi-line `_append_warning` arguments or elements.
        )

    source_mode = str(meta.get("source_mode") or "").strip().lower()
    if source_mode and source_mode not in _VALID_SOURCE_MODES:
        resource_items.append(
            # Completes the multi-line `_new_item` arguments or elements.
            _new_item("预测源模式", "failed", "source_mode 不受支持。", actual=source_mode)
        )
        _append_error(
            report,
            # Completes the multi-line `_append_error` arguments or elements.
            "SOURCE_MODE_INVALID",
            "预测源模式无效",
            f"source_mode={source_mode!r} 不受支持。",
            expected="reference、default_source 或 external_source",
            # Completes the multi-line `_append_error` arguments or elements.
            actual=source_mode,
            suggestion="修正 default_source_meta.json。",
            related_paths=[meta_path],
        )
    # Checks `not source_mode` before entering this conditional branch.
    if not source_mode:
        source_mode = _derive_source_mode(task_cfg)
        resource_items.append(
            _new_item("预测源模式", "warning", f"元数据未声明 source_mode，已推导为 {source_mode}。")
        # Completes the multi-line `resource_items.append` arguments or elements.
        )
    else:
        resource_items.append(_new_item("预测源模式", "passed", f"模型包声明使用 {source_mode}。"))

    default_source_path = ""
    # Checks `source_mode == "reference"` before entering this conditional branch.
    if source_mode == "reference":
        reference_files = _find_reference_files(model_dir)
        if len(reference_files) != 1:
            resource_items.append(
                # Completes the multi-line `_new_item` arguments or elements.
                _new_item("reference", "failed", "参考文件数量不符合要求。", actual=reference_files)
            )
            _append_error(
                report,
                # Completes the multi-line `_append_error` arguments or elements.
                "REFERENCE_MISSING" if not reference_files else "REFERENCE_MULTIPLE",
                "参考几何文件无效",
                "固定几何预测需要且只能有一个 reference.* 文件。",
                expected="一个可读的 reference.* 文件",
                # Completes the multi-line `len` arguments or elements.
                actual=f"找到 {len(reference_files)} 个",
                suggestion="保留一个与模型匹配的 reference.* 文件。",
                related_paths=reference_files or [model_dir],
            )
        # Continues the surrounding statement or collection definition.
        else:
            resource_items.append(
                _new_item(
                    "reference", "passed", "已找到唯一参考几何文件。", path=reference_files[0]
                # Completes the multi-line `_new_item` arguments or elements.
                )
            )
    elif source_mode == "default_source":
        entry = _resolve_meta_default_source(meta)
        # Stores the `entry_relpath` for use by the remaining steps in this flow.
        entry_relpath = str(entry.get("entry_relpath") or "").strip()
        resolved_entry = _resolve_package_path(model_dir, entry_relpath)
        if not resolved_entry or not os.path.exists(resolved_entry):
            resource_items.append(
                # Completes the multi-line `_new_item` arguments or elements.
                _new_item(
                    "default_source",
                    "failed",
                    "默认源入口不存在或不在模型目录内。",
                    # Completes the multi-line `_new_item` arguments or elements.
                    entry_relpath=entry_relpath,
                )
            )
            _append_error(
                # Completes the multi-line `_append_error` arguments or elements.
                report,
                "DEFAULT_SOURCE_ENTRY_INVALID",
                "默认预测源无效",
                "default_source 的入口路径不存在、为空或越出模型目录。",
                # Completes the multi-line `_append_error` arguments or elements.
                expected="模型目录内存在 meta.entry_relpath 指向的文件或目录",
                actual=entry_relpath or "空路径",
                suggestion="重新打包 default_source 并修正 default_source_meta.json。",
                related_paths=[meta_path],
            # Completes the multi-line `_append_error` arguments or elements.
            )
        else:
            expected_type = str(entry.get("entry_type") or "").strip().lower()
            type_valid = (
                # Continues the surrounding statement or collection definition.
                not expected_type
                or (expected_type in {"directory"} and os.path.isdir(resolved_entry))
                or (
                    expected_type in {"file", "manifest"}
                    # Completes the multi-line `os.path.isfile` arguments or elements.
                    and os.path.isfile(resolved_entry)
                )
            )
            if not type_valid:
                # Adds the current value to `resource_items` for later aggregation.
                resource_items.append(
                    _new_item(
                        "default_source",
                        "failed",
                        # Completes the multi-line `_new_item` arguments or elements.
                        "默认源入口类型与元数据不一致。",
                        path=resolved_entry,
                        expected=expected_type,
                    )
                # Completes the multi-line `resource_items.append` arguments or elements.
                )
                _append_error(
                    report,
                    "DEFAULT_SOURCE_ENTRY_INVALID",
                    # Completes the multi-line `_append_error` arguments or elements.
                    "默认预测源类型不匹配",
                    "默认源实际类型与 entry_type 不一致。",
                    expected=expected_type,
                    actual="directory" if os.path.isdir(resolved_entry) else "file",
                    # Completes the multi-line `_append_error` arguments or elements.
                    suggestion="重新打包 default_source。",
                    related_paths=[resolved_entry, meta_path],
                )
            else:
                # Stores the `default_source_path` for use by the remaining steps in this flow.
                default_source_path = resolved_entry
                resource_items.append(
                    _new_item(
                        "default_source",
                        # Completes the multi-line `_new_item` arguments or elements.
                        "passed",
                        "默认预测源入口有效。",
                        path=resolved_entry,
                        entry_type=expected_type,
                    # Completes the multi-line `_new_item` arguments or elements.
                    )
                )
    else:
        resource_items.append(
            # Completes the multi-line `_new_item` arguments or elements.
            _new_item("default_source", "not_required", "模型要求在实际预测请求中上传 source_file。")
        )

    report["validation"]["resources"] = _section(
        "模型目录中的 reference、default_source 和元数据校验。", resource_items
    # Completes the multi-line `_section` arguments or elements.
    )
    if report["errors"]:
        report["validation"]["smoke_test"] = _section("因资源校验失败，未执行预测冒烟测试。", smoke_items)
        return _finalize_summary(report, smoke_executed=False)

    # Checks `source_mode == "external_source"` before entering this conditional branch.
    if source_mode == "external_source":
        smoke_items.append(
            _new_item("完整预测", "skipped", "模型要求调用方上传 source_file，无法仅凭模型包执行冒烟预测。")
        )
        # Calls `_append_warning` to perform the operation required at this stage.
        _append_warning(
            report,
            "EXTERNAL_SOURCE_REQUIRED",
            "预测需要外部源文件",
            # Completes the multi-line `_append_warning` arguments or elements.
            "模型文件校验通过，但预测时需要调用方上传 source_file 或历史时序。",
            expected="预测请求提供 source_file 或满足模型要求的历史时序",
            actual="模型包未包含可用于当前预测的默认源文件",
            suggestion="创建预测任务时上传与该模型匹配的源文件或历史时序。",
            # Completes the multi-line `collection` arguments or elements.
            related_paths=[checkpoint_path],
        )
        report["validation"]["smoke_test"] = _section("预测冒烟测试被跳过。", smoke_items)
        return _finalize_summary(report, smoke_executed=False)

    # Stores the `(normalized_params, param_items, param_errors)` for use by the remaining steps in this flow.
    normalized_params, param_items, param_errors = _validate_smoke_params(
        parameter_schema, meta
    )
    smoke_items.extend(param_items)
    # Adds the current value to `report['errors']` for later aggregation.
    report["errors"].extend(param_errors)
    if param_errors:
        report["validation"]["smoke_test"] = _section("冒烟测试参数校验失败。", smoke_items)
        return _finalize_summary(report, smoke_executed=False)

    # Keeps the operation and its recovery path within the same exception boundary.
    try:
        with tempfile.TemporaryDirectory(prefix="tfai_model_validation_") as run_dir:
            samples_predict = [{"order": 1, "params": normalized_params}]
            if source_mode == "default_source":
                # Stores the `samples_predict[0]['source_file']` for use by the remaining steps in this flow.
                samples_predict[0]["source_file"] = default_source_path
            result = run_predict(
                config={"prediction": {"strict_writeback": True}},
                model_bundle=model_bundle,
                # Completes the multi-line `run_predict` arguments or elements.
                run_dir=run_dir,
                samples_predict=samples_predict,
                default_source_file=default_source_path or None,
            )
            # Stores the `result_files` for use by the remaining steps in this flow.
            result_files = list(result.get("result_files") or [])
            missing_output_files = [
                path for path in result_files if not os.path.isfile(path)
            ]
            # Stores the `writeback_report` for use by the remaining steps in this flow.
            writeback_report = result.get("writeback_report") or {}
            if (
                not result_files
                or missing_output_files
                # Completes the multi-line `int` arguments or elements.
                or int(writeback_report.get("failed_count", 0) or 0) > 0
            ):
                raise RuntimeError(
                    "预测未生成完整输出: "
                    # Completes the multi-line `len` arguments or elements.
                    f"result_files={len(result_files)}, missing={len(missing_output_files)}, "
                    f"writeback={writeback_report}"
                )
            smoke_items.append(
                # Completes the multi-line `_new_item` arguments or elements.
                _new_item(
                    "完整预测", "passed", "预处理、推理和结果写回均成功。", output_count=len(result_files)
                )
            )
            # Stores the `report['validation']['smoke_test']` for use by the remaining steps in this flow.
            report["validation"]["smoke_test"] = _section("完整预测冒烟测试通过。", smoke_items)
            return _finalize_summary(report, smoke_executed=True)
    except Exception as exc:
        smoke_items.append(_new_item("完整预测", "failed", "预测冒烟测试失败。", error=str(exc)))
        # Calls `_append_error` to perform the operation required at this stage.
        _append_error(
            report,
            "SMOKE_PREDICT_FAILED",
            "预测冒烟测试失败",
            # Completes the multi-line `_append_error` arguments or elements.
            f"模型无法使用包内资源完成预测: {exc}",
            suggestion="检查 reference/default_source、冒烟参数和模型运行时兼容性。",
            related_paths=[checkpoint_path],
        )
        # Stores the `report['validation']['smoke_test']` for use by the remaining steps in this flow.
        report["validation"]["smoke_test"] = _section("完整预测冒烟测试失败。", smoke_items)
        return _finalize_summary(report, smoke_executed=False)
