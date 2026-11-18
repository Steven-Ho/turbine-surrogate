"""训练样本读取接口。"""

import inspect
import os
from typing import Any, Dict, List, Tuple

# Imports `pandas` for reuse in this module.
import pandas as pd

from tfai.factories.reader_factory import ReaderFactory
from tfai.readers.sample_map import SampleMap
from tfai.utils.path_utils import resolve_path_from_anchor
# Imports `resolve_sample_source_path` from `tfai.utils.sample_utils` for use in this module.
from tfai.utils.sample_utils import resolve_sample_source_path
from tfai.utils.task_variable_config import (
    normalize_scalar_task_variable_aliases,
    resolve_scalar_target_variables,
# Continues the surrounding statement or collection definition.
)
from tfai.utils.variable_resolver import infer_component_map_from_field_names


def _build_samples_all(sample_map: SampleMap, sample_table_path: str) -> List[Dict[str, Any]]:
    """将样本表转换为后端调用约定的样本列表。"""
    records: List[Dict[str, Any]] = []
    # Visits each item in `sample_map.get_all_samples()` and applies the current processing step.
    for sample_order in sample_map.get_all_samples():
        info = dict(sample_map.get_sample_info(sample_order) or {})
        file_path = str(info.pop("file_path", "") or "")
        resolved_path = resolve_path_from_anchor(sample_table_path, file_path) if file_path else ""
        # Adds the current value to `records` for later aggregation.
        records.append(
            {
                "order": int(sample_order),
                "file_path": file_path,
                # Completes the multi-line `collection` arguments or elements.
                "resolved_path": resolved_path,
                "source_file": resolve_sample_source_path({"resolved_path": resolved_path}),
                "path_type": "directory" if (resolved_path and os.path.isdir(resolved_path)) else "file",
                "exists": bool(resolved_path and os.path.exists(resolved_path)),
                # Completes the multi-line `collection` arguments or elements.
                "params": info,
            }
        )
    return records


# 按出现顺序收集样本参数字段。
def _collect_available_input_variables(samples_all: List[Dict[str, Any]]) -> List[str]:
    """按出现顺序收集样本参数字段。"""
    fields: List[str] = []
    for item in samples_all:
        params = item.get("params") or {}
        # Checks `not isinstance(params, dict)` before entering this conditional branch.
        if not isinstance(params, dict):
            continue
        for key in params.keys():
            key_str = str(key)
            # Checks `key_str not in fields` before entering this conditional branch.
            if key_str not in fields:
                fields.append(key_str)
    return fields


def _build_scalar_samples_all_from_dataframe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """将标量表按行转换为统一样本列表。"""
    # Initializes the `records` as a collection for values from this processing stage.
    records: List[Dict[str, Any]] = []
    for row_index, row in df.reset_index(drop=True).iterrows():
        records.append(
            {
                # Completes the multi-line `int` arguments or elements.
                "order": int(row_index),
                "file_path": "",
                "resolved_path": "",
                "source_file": "",
                # Completes the multi-line `collection` arguments or elements.
                "path_type": "file",
                "exists": True,
                "params": {
                    str(key): value
                    # Completes the multi-line `collection` arguments or elements.
                    for key, value in row.to_dict().items()
                },
            }
        )
    # Returns `records` to the caller for the next stage.
    return records


def _resolve_scalar_available_input_variables(
    columns: List[str],
    task_cfg: Dict[str, Any],
# Continues the surrounding statement or collection definition.
) -> List[str]:
    """推断 scalar 任务可选输入列。"""
    configured_inputs = [str(name) for name in task_cfg.get("input_variables") or []]
    if configured_inputs:
        return [name for name in configured_inputs if name in columns]

    # Stores the `target_variables` for use by the remaining steps in this flow.
    target_variables = resolve_scalar_target_variables(task_cfg)
    if target_variables:
        return [name for name in columns if name not in target_variables]

    return list(columns)


# 尽量以兼容方式调用 reader.read，探测可用输出变量。
def _call_reader_read(reader, file_path: str, sample_order: int):
    """尽量以兼容方式调用 reader.read，探测可用输出变量。"""
    read_fn = getattr(reader, "read")
    signature = inspect.signature(read_fn)
    kwargs: Dict[str, Any] = {"file_path": file_path}
    # Checks `"variables" in signature.parameters` before entering this conditional branch.
    if "variables" in signature.parameters:
        kwargs["variables"] = []
    if "sample_order" in signature.parameters:
        kwargs["sample_order"] = sample_order
    # Checks `"allow_missing_fields" in signature.parameters` before entering this conditional branch.
    if "allow_missing_fields" in signature.parameters:
        kwargs["allow_missing_fields"] = True
    return read_fn(**kwargs)


def _extract_available_output_variables_from_sample_data(sample_data, task_type: str) -> List[str]:
    """从已成功读取的样本中提取可用输出变量。"""
    # Checks `task_type == "scalar"` before entering this conditional branch.
    if task_type == "scalar":
        scalar_data = sample_data.get_scalar_data()
        if hasattr(scalar_data, "columns"):
            return [str(column) for column in scalar_data.columns]
        # Returns `[]` to the caller for the next stage.
        return []

    payload = sample_data.get_data() or {}
    variable_catalog = payload.get("VariableCatalog") or {}
    if isinstance(variable_catalog, dict) and variable_catalog:
        # Returns `[str(name) for name in variable_catalog.keys()]` to the caller for the next stage.
        return [str(name) for name in variable_catalog.keys()]

    component_map = payload.get("FieldComponentMap") or {}
    if isinstance(component_map, dict) and component_map:
        return [str(name) for name in component_map.keys()]

    # Stores the `effective_variables` for use by the remaining steps in this flow.
    effective_variables = list(payload.get("EffectiveVariables") or [])
    if effective_variables:
        return list(infer_component_map_from_field_names(effective_variables).keys())

    return []


# 从可成功读取的样本中推断可供前端选择的输出变量。
def _collect_available_output_variables(
    reader,
    sample_map: SampleMap,
    sample_table_path: str,
    # Continues the surrounding statement or collection definition.
    task_type: str,
    fallback_variables: List[str],
) -> List[str]:
    """从可成功读取的样本中推断可供前端选择的输出变量。"""
    sample_ids = sample_map.get_all_samples()
    # Returns early when `not sample_ids` makes the remaining work inapplicable.
    if not sample_ids:
        return list(fallback_variables)

    for raw_sample_id in sample_ids:
        sample_id = int(raw_sample_id)
        # Stores the `sample_file` for use by the remaining steps in this flow.
        sample_file = sample_map.get_file_path(sample_id) or ""
        if not sample_file:
            continue

        sample_abs_path = resolve_path_from_anchor(sample_table_path, sample_file)
        # Keeps the operation and its recovery path within the same exception boundary.
        try:
            sample_data = _call_reader_read(
                reader,
                file_path=sample_abs_path,
                # Completes the multi-line `_call_reader_read` arguments or elements.
                sample_order=sample_id,
            )
        except Exception:
            continue

        # Stores the `resolved_variables` for use by the remaining steps in this flow.
        resolved_variables = _extract_available_output_variables_from_sample_data(
            sample_data=sample_data,
            task_type=task_type,
        )
        # Returns early when `resolved_variables` makes the remaining work inapplicable.
        if resolved_variables:
            return resolved_variables

    return list(fallback_variables)


def read_train_samples(
    # Continues the surrounding statement or collection definition.
    train_samples: str,
    is_geometry_varying: bool,
    mode: str,
    config: Dict,
# Completes the multi-line `collection` arguments or elements.
) -> Tuple[List[Dict], Dict]:
    """读取训练样本并返回样本列表与变量信息。"""
    sample_table_path = os.path.abspath(str(train_samples))
    if not os.path.exists(sample_table_path):
        raise FileNotFoundError(f"训练样本表不存在: {sample_table_path}")

    # Stores the `general_cfg` for use by the remaining steps in this flow.
    general_cfg = config.setdefault("general", {})
    task_cfg = config.setdefault("task", {})
    task_cfg.setdefault("variables", [])
    normalize_scalar_task_variable_aliases(config)

    # Stores the `general_cfg['train_samples']` for use by the remaining steps in this flow.
    general_cfg["train_samples"] = sample_table_path
    task_cfg["is_geometry_varying"] = bool(is_geometry_varying)
    task_cfg["mode"] = str(mode)
    task_type = str(task_cfg.get("type", "field")).lower()

    # Stores the `sample_map` for use by the remaining steps in this flow.
    sample_map = SampleMap(sample_table_path)
    reader = ReaderFactory.get_reader(
        config=config,
        flag="train",
        # Completes the multi-line `ReaderFactory.get_reader` arguments or elements.
        sample_table_path_override=sample_table_path,
    )

    if task_type == "scalar":
        sample_data = _call_reader_read(
            # Completes the multi-line `_call_reader_read` arguments or elements.
            reader,
            file_path=sample_table_path,
            sample_order=0,
        )
        # Stores the `scalar_data` for use by the remaining steps in this flow.
        scalar_data = sample_data.get_scalar_data()
        if not hasattr(scalar_data, "columns"):
            raise ValueError("scalar 样本表读取后未返回可识别的表格数据")

        if sample_map.get_all_samples():
            # Stores the `samples_all` for use by the remaining steps in this flow.
            samples_all = _build_samples_all(sample_map=sample_map, sample_table_path=sample_table_path)
        else:
            samples_all = _build_scalar_samples_all_from_dataframe(pd.DataFrame(scalar_data))

        available_output_variables = _extract_available_output_variables_from_sample_data(
            # Completes the multi-line `_extract_available_output_variables_from_sample_data` arguments or elements.
            sample_data=sample_data,
            task_type=task_type,
        )
        available_input_variables = _resolve_scalar_available_input_variables(
            # Completes the multi-line `_resolve_scalar_available_input_variables` arguments or elements.
            columns=available_output_variables,
            task_cfg=task_cfg,
        )
    else:
        # Stores the `samples_all` for use by the remaining steps in this flow.
        samples_all = _build_samples_all(sample_map=sample_map, sample_table_path=sample_table_path)
        available_input_variables = _collect_available_input_variables(samples_all)
        available_output_variables = _collect_available_output_variables(
            reader=reader,
            # Completes the multi-line `_collect_available_output_variables` arguments or elements.
            sample_map=sample_map,
            sample_table_path=sample_table_path,
            task_type=task_type,
            fallback_variables=[str(name) for name in task_cfg.get("variables") or []],
        # Completes the multi-line `_collect_available_output_variables` arguments or elements.
        )

    variable_info = {
        "available_input_variables": available_input_variables,
        "available_output_variables": available_output_variables,
    # Completes the multi-line `collection` arguments or elements.
    }
    return samples_all, variable_info
