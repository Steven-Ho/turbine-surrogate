"""输入输出变量设定接口。"""

import copy
from numbers import Number
from typing import Any, Dict, List, Optional

# Imports `normalize_scalar_task_variable_aliases` from `tfai.utils.task_variable_config` for use in this module.
from tfai.utils.task_variable_config import normalize_scalar_task_variable_aliases


def apply_variable_selection(
    config: Dict,
    input_variables: List[str],
    # Continues the surrounding statement or collection definition.
    output_variables: List[str],
    samples_all: Optional[List[Dict[str, Any]]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """将变量选择写入配置，并可返回按选择裁剪后的训练样本。"""
    task_cfg = config.setdefault("task", {})

    # Builds the `normalized_input` using the comprehension's filter and mapping rules.
    normalized_input = [str(name) for name in input_variables or []]
    normalized_output = [str(name) for name in output_variables or []]
    if not normalized_output:
        raise ValueError("output_variables 不能为空")

    # Stores the `dedup_input` for use by the remaining steps in this flow.
    dedup_input = list(dict.fromkeys(normalized_input))
    dedup_output = list(dict.fromkeys(normalized_output))

    task_cfg["input_variables"] = dedup_input
    task_cfg["output_variables"] = dedup_output
    # Stores the `task_cfg['variables']` for use by the remaining steps in this flow.
    task_cfg["variables"] = dedup_output
    normalize_scalar_task_variable_aliases(config)

    for key in [
        "logical_variables",
        # Completes the multi-line `collection` arguments or elements.
        "effective_variables",
        "field_component_map",
        "variable_catalog",
        "effective_variable_sources",
        # Completes the multi-line `collection` arguments or elements.
        "point_or_cell",
        "field_location_map",
        "projected_point_or_cell",
        "mixed_field_policy",
        # Completes the multi-line `collection` arguments or elements.
        "projection_method",
        "mesh_dimension",
        "geometry_regime",
        "geometry_motion",
        # Completes the multi-line `collection` arguments or elements.
        "geometry_representation",
        "point_feature_fields",
        "point_target_fields",
        "displacement_logical_variable",
        # Completes the multi-line `collection` arguments or elements.
        "displacement_effective_variables",
        "parameter_schema",
    ]:
        task_cfg.pop(key, None)

    # Returns early when `samples_all is None` makes the remaining work inapplicable.
    if samples_all is None:
        return None
    if not isinstance(samples_all, list):
        raise TypeError("samples_all 必须为样本字典列表或 None")

    # Stores the `task_type` for use by the remaining steps in this flow.
    task_type = str(task_cfg.get("type", "field")).lower()
    retained_variables = list(dict.fromkeys(
        dedup_input + dedup_output if task_type == "scalar" else dedup_input
    ))
    # Initializes the `selected_samples` as a collection for values from this processing stage.
    selected_samples: List[Dict[str, Any]] = []

    for index, item in enumerate(samples_all):
        if not isinstance(item, dict):
            raise TypeError(f"samples_all[{index}] 必须为字典")

        # Stores the `raw_params` for use by the remaining steps in this flow.
        raw_params = item.get("params") or {}
        if not isinstance(raw_params, dict):
            raise TypeError(f"samples_all[{index}].params 必须为字典")

        sample_order = item.get("order", item.get("sample_id", index))
        # Builds the `missing_variables` using the comprehension's filter and mapping rules.
        missing_variables = [
            name for name in retained_variables if name not in raw_params
        ]
        if missing_variables:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(
                f"样本 {sample_order} 缺少用户选择的变量: {missing_variables}"
            )

        if task_type != "scalar":
            # Builds the `non_numeric_variables` using the comprehension's filter and mapping rules.
            non_numeric_variables = [
                name
                for name in retained_variables
                if not isinstance(raw_params[name], Number)
            # Continues the surrounding statement or collection definition.
            ]
            if non_numeric_variables:
                raise ValueError(
                    f"field 样本 {sample_order} 的输入变量必须为数值类型: "
                    # Completes the multi-line `ValueError` arguments or elements.
                    f"{non_numeric_variables}"
                )

        selected_item = copy.deepcopy(item)
        selected_item["params"] = {
            # Completes the multi-line `copy.deepcopy` arguments or elements.
            name: copy.deepcopy(raw_params[name])
            for name in retained_variables
        }
        selected_samples.append(selected_item)

    # Returns `selected_samples` to the caller for the next stage.
    return selected_samples
