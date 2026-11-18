import os
from typing import Dict, Tuple

import torch

# Imports `SampleMap` from `tfai.readers.sample_map` for use in this module.
from tfai.readers.sample_map import SampleMap
from tfai.utils.path_utils import resolve_path_from_anchor


def get_first_sample_abs_path(sample_map: SampleMap, samples_csv: str) -> Tuple[object, str]:
    sample_ids = sample_map.get_all_samples()
    # Validates `not sample_ids` and raises `ValueError` when the input cannot be processed safely.
    if not sample_ids:
        raise ValueError("样本表为空，无法解析变量")

    first_id = sample_ids[0]
    first_file = sample_map.get_file_path(first_id)
    # Validates `not first_file` and raises `ValueError` when the input cannot be processed safely.
    if not first_file:
        raise ValueError("样本表首个样本缺少file_path，无法解析变量")

    first_abs = resolve_path_from_anchor(samples_csv, first_file)
    return first_id, first_abs


# 从首个可读样本中解析变量清单，容忍前置坏样本。
def resolve_first_valid_variable_manifest(
    reader,
    sample_map: SampleMap,
    samples_csv: str,
    # Continues the surrounding statement or collection definition.
    logical_variables,
    logger,
):
    """从首个可读样本中解析变量清单，容忍前置坏样本。"""
    sample_ids = sample_map.get_all_samples()
    # Validates `not sample_ids` and raises `ValueError` when the input cannot be processed safely.
    if not sample_ids:
        raise ValueError("样本表为空，无法解析变量")

    last_exc = None
    for sample_id in sample_ids:
        # Stores the `sample_file` for use by the remaining steps in this flow.
        sample_file = sample_map.get_file_path(sample_id)
        if not sample_file:
            logger.warning("样本 %s 缺少 file_path，跳过变量解析探测", sample_id)
            continue

        # Stores the `sample_abs` for use by the remaining steps in this flow.
        sample_abs = resolve_path_from_anchor(samples_csv, sample_file)
        try:
            manifest = reader.resolve_variable_manifest(
                file_path=sample_abs,
                # Completes the multi-line `reader.resolve_variable_manifest` arguments or elements.
                logical_variables=logical_variables,
                sample_order=sample_id,
            )
            return sample_id, sample_abs, manifest
        # Continues the surrounding statement or collection definition.
        except Exception as exc:
            last_exc = exc
            logger.warning("样本 %s 变量解析失败，继续尝试后续样本: %s", sample_id, str(exc))

    if last_exc is not None:
        # Raises `ValueError` to report why this path cannot continue.
        raise ValueError("所有样本都无法完成变量解析，无法继续训练预处理") from last_exc
    raise ValueError("样本表中没有可用于变量解析的有效样本")


def resolve_training_variables(
    config: Dict,
    # Continues the surrounding statement or collection definition.
    reader,
    sample_map: SampleMap,
    samples_csv: str,
    processed_dir: str,
    # Continues the surrounding statement or collection definition.
    logger,
) -> None:
    """Resolve and persist variable manifest for field tasks."""
    task_cfg = config.get('task', {})
    if task_cfg.get('type', 'field') != 'field':
        # Records the current execution state to make progress and failures traceable.
        logger.info("标量任务跳过变量展开，保留读取器默认变量语义")
        return

    logical_vars = list(task_cfg.get('variables') or [])
    if not logical_vars:
        # Raises `ValueError` to report why this path cannot continue.
        raise ValueError("field任务必须配置task.variables（逻辑变量名）")

    first_id, first_abs, manifest = resolve_first_valid_variable_manifest(
        reader=reader,
        sample_map=sample_map,
        # Completes the multi-line `resolve_first_valid_variable_manifest` arguments or elements.
        samples_csv=samples_csv,
        logical_variables=logical_vars,
        logger=logger,
    )

    # Stores the `resolved_logical` for use by the remaining steps in this flow.
    resolved_logical = list(manifest.get('logical_variables') or logical_vars)
    effective_vars = list(manifest.get('effective_variables') or [])
    field_component_map = manifest.get('field_component_map') or {}
    variable_catalog = manifest.get('variable_catalog') or {}
    # Stores the `effective_variable_sources` for use by the remaining steps in this flow.
    effective_variable_sources = manifest.get('effective_variable_sources') or {}
    point_or_cell = manifest.get('point_or_cell')
    field_location_map = manifest.get('field_location_map') or {}
    projected_point_or_cell = manifest.get('projected_point_or_cell') or point_or_cell
    # Converts the `mixed_field_policy` to `str` so later checks use a consistent type.
    mixed_field_policy = str(manifest.get('mixed_field_policy') or 'majority_side')
    projection_method = str(manifest.get('projection_method') or 'mean')
    mesh_dimension = manifest.get('mesh_dimension')
    geometry_regime = str(manifest.get('geometry_regime') or 'static_geometry')
    # Stores the `geometry_motion` for use by the remaining steps in this flow.
    geometry_motion = dict(manifest.get('geometry_motion') or {})
    geometry_representation = str(manifest.get('geometry_representation') or '').strip().lower()
    point_feature_fields = list(manifest.get('point_feature_fields') or [])
    point_target_fields = list(manifest.get('point_target_fields') or [])
    # Reads `displacement_logical_variable` from the mapping into the `displacement_logical_variable`.
    displacement_logical_variable = manifest.get('displacement_logical_variable')
    displacement_effective_variables = list(manifest.get('displacement_effective_variables') or [])
    if geometry_motion:
        displacement_logical_variable = geometry_motion.get(
            # Completes the multi-line `geometry_motion.get` arguments or elements.
            'displacement_logical_name',
            displacement_logical_variable,
        )
        displacement_effective_variables = list(
            # Completes the multi-line `geometry_motion.get` arguments or elements.
            geometry_motion.get('displacement_effective_variables')
            or displacement_effective_variables
        )
    elif geometry_regime == 'displaced_static_topology':
        # Initializes the `geometry_motion` as a collection for values from this processing stage.
        geometry_motion = {
            'geometry_regime': geometry_regime,
            'displacement_logical_name': displacement_logical_variable,
            'displacement_effective_variables': list(displacement_effective_variables),
        # Completes the multi-line `collection` arguments or elements.
        }
    if not effective_vars:
        raise ValueError("变量解析失败：effective_variables为空")
    if not isinstance(field_component_map, dict) or not field_component_map:
        # Raises `ValueError` to report why this path cannot continue.
        raise ValueError("变量解析失败：field_component_map为空")
    if variable_catalog and not isinstance(variable_catalog, dict):
        raise ValueError("变量解析失败：variable_catalog格式非法")
    if effective_variable_sources and not isinstance(effective_variable_sources, dict):
        # Raises `ValueError` to report why this path cannot continue.
        raise ValueError("变量解析失败：effective_variable_sources格式非法")
    if point_or_cell not in {'point', 'cell'}:
        raise ValueError(f"变量解析失败：point_or_cell非法: {point_or_cell}")
    if mesh_dimension is not None:
        # Keeps the operation and its recovery path within the same exception boundary.
        try:
            mesh_dimension = int(mesh_dimension)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"变量解析失败：mesh_dimension非法: {mesh_dimension}") from exc
        # Validates `mesh_dimension not in {2, 3}` and raises `ValueError` when the input cannot be processed safely.
        if mesh_dimension not in {2, 3}:
            raise ValueError(f"变量解析失败：mesh_dimension超出支持范围: {mesh_dimension}")

    if geometry_regime == 'displaced_static_topology':
        if not displacement_effective_variables:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("检测到运动网格，但未解析出位移训练通道")
        missing_displacement = [
            name for name in displacement_effective_variables
            if name not in effective_vars
        # Continues the surrounding statement or collection definition.
        ]
        if missing_displacement:
            raise ValueError(
                "检测到运动网格时，task.variables 必须显式选择全部位移字段或数组: "
                # Completes the multi-line `ValueError` arguments or elements.
                f"missing={missing_displacement}"
            )
        if not displacement_logical_variable:
            for selection_key, components in field_component_map.items():
                # Checks `displacement_effective_variables[0] in list(components or [])` before entering this conditional branch.
                if displacement_effective_variables[0] in list(components or []):
                    displacement_logical_variable = selection_key
                    break
        geometry_motion['geometry_regime'] = geometry_regime
        # Stores the `geometry_motion['displacement_effective_variables']` for use by the remaining steps in this flow.
        geometry_motion['displacement_effective_variables'] = list(
            displacement_effective_variables
        )
        geometry_motion.setdefault(
            # Completes the multi-line `geometry_motion.setdefault` arguments or elements.
            'displacement_logical_name',
            manifest.get('displacement_logical_variable'),
        )
        if point_or_cell != 'point':
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("检测到运动网格情形 2 时，当前仅支持点场位移写回，请保证 point_or_cell=point")
        if not displacement_logical_variable:
            raise ValueError("检测到运动网格情形 2，但样本中未识别到位移逻辑变量")
        normalized_requested = {str(name).casefold() for name in logical_vars}
        # Validates `` and raises `ValueError` when the input cannot be processed safely.
        if str(displacement_logical_variable).casefold() not in normalized_requested:
            raise ValueError(
                "检测到运动网格情形 2，task.variables 必须包含位移逻辑变量。"
                f"当前缺少: {displacement_logical_variable}"
            # Completes the multi-line `ValueError` arguments or elements.
            )

    task_cfg['logical_variables'] = resolved_logical
    task_cfg['effective_variables'] = effective_vars
    task_cfg['field_component_map'] = field_component_map
    # Stores the `task_cfg['variable_catalog']` for use by the remaining steps in this flow.
    task_cfg['variable_catalog'] = variable_catalog
    task_cfg['effective_variable_sources'] = effective_variable_sources
    task_cfg['point_or_cell'] = point_or_cell
    task_cfg['field_location_map'] = field_location_map
    # Stores the `task_cfg['projected_point_or_cell']` for use by the remaining steps in this flow.
    task_cfg['projected_point_or_cell'] = projected_point_or_cell
    task_cfg['mixed_field_policy'] = mixed_field_policy
    task_cfg['projection_method'] = projection_method
    task_cfg['geometry_regime'] = geometry_regime
    # Stores the `task_cfg['geometry_motion']` for use by the remaining steps in this flow.
    task_cfg['geometry_motion'] = geometry_motion
    if geometry_representation:
        task_cfg['geometry_representation'] = geometry_representation
    task_cfg['point_feature_fields'] = point_feature_fields
    # Stores the `task_cfg['point_target_fields']` for use by the remaining steps in this flow.
    task_cfg['point_target_fields'] = point_target_fields
    task_cfg['displacement_logical_variable'] = displacement_logical_variable
    task_cfg['displacement_effective_variables'] = displacement_effective_variables
    if mesh_dimension is not None:
        # Stores the `task_cfg['mesh_dimension']` for use by the remaining steps in this flow.
        task_cfg['mesh_dimension'] = mesh_dimension

    os.makedirs(processed_dir, exist_ok=True)
    manifest_path = os.path.join(processed_dir, 'variable_manifest.pt')
    torch.save(
        # Completes the multi-line `collection` arguments or elements.
        {
            'logical_variables': resolved_logical,
            'effective_variables': effective_vars,
            'field_component_map': field_component_map,
            # Completes the multi-line `collection` arguments or elements.
            'variable_catalog': variable_catalog,
            'effective_variable_sources': effective_variable_sources,
            'point_or_cell': point_or_cell,
            'field_location_map': field_location_map,
            # Completes the multi-line `collection` arguments or elements.
            'projected_point_or_cell': projected_point_or_cell,
            'mixed_field_policy': mixed_field_policy,
            'projection_method': projection_method,
            'mesh_dimension': mesh_dimension,
            # Completes the multi-line `collection` arguments or elements.
            'geometry_regime': geometry_regime,
            'geometry_motion': geometry_motion,
            'geometry_representation': geometry_representation,
            'point_feature_fields': point_feature_fields,
            # Completes the multi-line `collection` arguments or elements.
            'point_target_fields': point_target_fields,
            'displacement_logical_variable': displacement_logical_variable,
            'displacement_effective_variables': displacement_effective_variables,
        },
        # Completes the multi-line `torch.save` arguments or elements.
        manifest_path,
    )
    logger.info(
        f"变量解析完成: logical={resolved_logical}, effective={effective_vars}, "
        # Completes the multi-line `logger.info` arguments or elements.
        f"point_or_cell={point_or_cell}, mesh_dimension={mesh_dimension}, "
        f"geometry_regime={geometry_regime}, displacement={displacement_logical_variable}, "
        f"projected_point_or_cell={projected_point_or_cell}"
    )
    # Records the current execution state to make progress and failures traceable.
    logger.info(f"变量清单已保存: {manifest_path}")
