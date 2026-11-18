from __future__ import annotations

import logging
import os
# Imports `Dict, List, Optional` from `typing` for use in this module.
from typing import Dict, List, Optional, Tuple

import numpy as np

from tfai.utils.field_projection import normalize_field_location, project_field_values
from tfai.utils.logger import Logger


# 负责将预测结果写回 VTK/VTU 文件。
class VTKPredictionWriter:
    """负责将预测结果写回 VTK/VTU 文件。"""

    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger

    # Defines `_log` for  log logic.
    def _log(self, level: str, message: str) -> None:
        logger = self.logger
        if logger is None:
            logger = Logger.get_first_configured_logger(
                # Completes the multi-line `collection` arguments or elements.
                ["predict", "evaluate", "train"],
                fallback_name=__name__,
            )
        log_fn = getattr(logger, level, logger.info)
        # Calls `log_fn` to perform the operation required at this stage.
        log_fn(message)

    def _get_write_location(
        self,
        sample_id: int,
        # Completes the multi-line `collection` arguments or elements.
        write_fields: Dict[str, np.ndarray],
        num_points: int,
        num_cells: int,
        location: Optional[str] = None,
    # Continues the surrounding statement or collection definition.
    ) -> str:
        """兼容旧调用方的写回位置校验接口。"""
        task_cfg = self.config.get("task", {}) if isinstance(self.config, dict) else {}
        normalized_location = normalize_field_location(
            location or task_cfg.get("projected_point_or_cell") or task_cfg.get("point_or_cell") or "point"
        # Completes the multi-line `normalize_field_location` arguments or elements.
        )
        expected = num_points if normalized_location == "point" else num_cells
        self._validate_entity_count(
            sample_id=sample_id,
            # Completes the multi-line `self._validate_entity_count` arguments or elements.
            fields=write_fields,
            expected=expected,
            location=normalized_location,
        )
        # Returns `normalized_location` to the caller for the next stage.
        return normalized_location

    @staticmethod
    def _normalize_component_map(raw_map: Dict[str, List[str]]) -> Dict[str, List[str]]:
        component_map: Dict[str, List[str]] = {}
        # Visits each item in `(raw_map or {}).items()` and applies the current processing step.
        for raw_name, split_names in (raw_map or {}).items():
            if not isinstance(raw_name, str) or not isinstance(split_names, list):
                continue
            component_map[raw_name] = [name for name in split_names if isinstance(name, str)]
        # Returns `component_map` to the caller for the next stage.
        return component_map

    @staticmethod
    def _normalize_location_map(raw_map: Dict[str, str]) -> Dict[str, str]:
        location_map: Dict[str, str] = {}
        # Visits each item in `(raw_map or {}).items()` and applies the current processing step.
        for name, location in (raw_map or {}).items():
            if not isinstance(name, str):
                continue
            location_map[name] = normalize_field_location(location)
        # Returns `location_map` to the caller for the next stage.
        return location_map

    @staticmethod
    def _reassemble_vtk_fields(
        processed_fields: Dict[str, np.ndarray],
        # Completes the multi-line `collection` arguments or elements.
        component_map: Dict[str, List[str]],
    ) -> Dict[str, np.ndarray]:
        if not processed_fields:
            return {}
        # Returns early when `not component_map` makes the remaining work inapplicable.
        if not component_map:
            return dict(processed_fields)

        out: Dict[str, np.ndarray] = dict(processed_fields)
        to_remove = set()

        # Visits each item in `component_map.items()` and applies the current processing step.
        for raw_name, split_names in component_map.items():
            if not split_names:
                continue
            if not all(name in processed_fields for name in split_names):
                # Handles the data or control-flow case represented by this statement.
                continue

            comp_arrays = [np.asarray(processed_fields[name]) for name in split_names]
            ref_shape = comp_arrays[0].shape
            if any(arr.shape != ref_shape for arr in comp_arrays[1:]):
                # Handles the data or control-flow case represented by this statement.
                continue

            out[raw_name] = np.stack(comp_arrays, axis=-1)
            to_remove.update(name for name in split_names if name != raw_name)

        for name in to_remove:
            # Calls `out.pop` to perform the operation required at this stage.
            out.pop(name, None)

        return out

    @staticmethod
    def _normalize_effective_sources(raw_sources: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, object]]:
        # Initializes the `sources` as a collection for values from this processing stage.
        sources: Dict[str, Dict[str, object]] = {}
        for runtime_key, source in (raw_sources or {}).items():
            if not isinstance(runtime_key, str) or not isinstance(source, dict):
                continue
            # Checks `str(source.get("format") or "vtk").lower() != "vtk"` before entering this conditional branch.
            if str(source.get("format") or "vtk").lower() != "vtk":
                continue
            array_name = str(source.get("array_name") or "").strip()
            location = str(source.get("location") or "").strip().lower()
            # Checks `not array_name or location not in {"point", "cell"}` before entering this conditional branch.
            if not array_name or location not in {"point", "cell"}:
                continue
            try:
                component_index = int(source.get("component_index", 0))
                # Converts the `num_components` to `int` so later checks use a consistent type.
                num_components = int(source.get("num_components", 1))
                occurrence = int(source.get("occurrence", 0))
            except (TypeError, ValueError):
                continue
            # Checks `component_index < 0 or num_components <= 0` before entering this conditional branch.
            if component_index < 0 or num_components <= 0:
                continue
            normalized = dict(source)
            normalized["array_name"] = array_name
            # Stores the `normalized['location']` for use by the remaining steps in this flow.
            normalized["location"] = location
            normalized["component_index"] = component_index
            normalized["num_components"] = num_components
            normalized["occurrence"] = occurrence
            # Stores the `sources[runtime_key]` for use by the remaining steps in this flow.
            sources[runtime_key] = normalized
        return sources

    @staticmethod
    def _reassemble_vtk_fields_by_sources(
        # Completes the multi-line `collection` arguments or elements.
        processed_fields: Dict[str, np.ndarray],
        effective_sources: Dict[str, Dict[str, object]],
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """按来源描述完整重组 VTK 原始数组，避免依赖显示后缀。"""
        grouped_sources: Dict[
            # Completes the multi-line `collection` arguments or elements.
            Tuple[str, str, int],
            List[Tuple[str, Dict[str, object]]],
        ] = {}
        for runtime_key, source in effective_sources.items():
            # Checks `runtime_key not in processed_fields` before entering this conditional branch.
            if runtime_key not in processed_fields:
                continue
            group_key = (
                str(source["location"]),
                # Completes the multi-line `str` arguments or elements.
                str(source["array_name"]),
                int(source["occurrence"]),
            )
            grouped_sources.setdefault(group_key, []).append((runtime_key, source))

        # Initializes the `grouped_fields` as a collection for values from this processing stage.
        grouped_fields: Dict[str, Dict[str, np.ndarray]] = {"point": {}, "cell": {}}
        for (location, array_name, occurrence), components in grouped_sources.items():
            components.sort(key=lambda item: int(item[1]["component_index"]))
            component_count = int(components[0][1]["num_components"])
            # Builds the `indices` using the comprehension's filter and mapping rules.
            indices = [int(source["component_index"]) for _, source in components]
            if indices != list(range(component_count)):
                raise ValueError(
                    "VTK 多分量数组缺少预测分量，无法按 clean_display 写回: "
                    # Completes the multi-line `ValueError` arguments or elements.
                    f"array={array_name}, occurrence={occurrence}, indices={indices}"
                )
            if any(int(source["num_components"]) != component_count for _, source in components):
                raise ValueError(f"VTK 数组分量元数据不一致: {array_name}")

            # Builds the `component_values` using the comprehension's filter and mapping rules.
            component_values = [np.asarray(processed_fields[runtime_key]) for runtime_key, _ in components]
            reference_shape = component_values[0].shape
            if any(values.shape != reference_shape for values in component_values[1:]):
                raise ValueError(f"VTK 数组分量形状不一致: {array_name}")
            # Stores the `values` for use by the remaining steps in this flow.
            values = component_values[0] if component_count == 1 else np.stack(component_values, axis=-1)

            if array_name in grouped_fields[location]:
                raise ValueError(
                    "同一位置存在重复 VTK 原始数组名，当前 writer 无法无损写回: "
                    # Completes the multi-line `ValueError` arguments or elements.
                    f"location={location}, array={array_name}"
                )
            grouped_fields[location][array_name] = values

        return grouped_fields

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _infer_dataset_field_locations(raw_dataset, component_map: Dict[str, List[str]]) -> Dict[str, str]:
        locations: Dict[str, str] = {}
        if raw_dataset is None:
            # Returns `locations` to the caller for the next stage.
            return locations

        def collect(data_attr, location: str) -> None:
            if data_attr is None:
                return
            # Stores the `array_count` for use by the remaining steps in this flow.
            array_count = data_attr.GetNumberOfArrays()
            for index in range(array_count):
                arr = data_attr.GetArray(index)
                if arr is None:
                    # Handles the data or control-flow case represented by this statement.
                    continue
                raw_name = arr.GetName() or f"array_{index}"
                locations[raw_name] = location
                for split_name in component_map.get(raw_name, []):
                    # Stores the `locations[split_name]` for use by the remaining steps in this flow.
                    locations[split_name] = location

        collect(getattr(raw_dataset, "GetPointData", lambda: None)(), "point")
        collect(getattr(raw_dataset, "GetCellData", lambda: None)(), "cell")
        return locations

    # Defines `_resolve_projected_location` for  resolve projected location logic.
    def _resolve_projected_location(self) -> str:
        task_cfg = self.config.get("task", {}) if isinstance(self.config, dict) else {}
        location = (
            task_cfg.get("projected_point_or_cell")
            # Completes the multi-line `task_cfg.get` arguments or elements.
            or task_cfg.get("point_or_cell")
            or "point"
        )
        return normalize_field_location(location)

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _validate_entity_count(
        sample_id: int,
        fields: Dict[str, np.ndarray],
        # Continues the surrounding statement or collection definition.
        expected: int,
        location: str,
    ) -> None:
        for name, values in fields.items():
            # Stores the `arr` for use by the remaining steps in this flow.
            arr = np.asarray(values)
            if arr.ndim == 0:
                raise ValueError(f"样本 {sample_id} 的字段 {name} 维度非法")
            if arr.shape[0] != expected:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(
                    f"样本 {sample_id} 的字段 {name} 写回长度不匹配: "
                    f"location={location}, actual={arr.shape[0]}, expected={expected}"
                )

    # Defines `_resolve_output_field_locations` for  resolve output field locations logic.
    def _resolve_output_field_locations(
        self,
        component_map: Dict[str, List[str]],
        configured_locations: Dict[str, str],
        # Completes the multi-line `collection` arguments or elements.
        dataset_locations: Dict[str, str],
    ) -> Dict[str, str]:
        resolved = dict(dataset_locations)
        resolved.update(configured_locations)

        # Visits each item in `component_map.items()` and applies the current processing step.
        for raw_name, split_names in component_map.items():
            raw_location = resolved.get(raw_name)
            if raw_location is None:
                for split_name in split_names:
                    # Stores the `split_location` for use by the remaining steps in this flow.
                    split_location = resolved.get(split_name)
                    if split_location in {"point", "cell"}:
                        raw_location = split_location
                        break
            # Checks `raw_location in {"point", "cell"}` before entering this conditional branch.
            if raw_location in {"point", "cell"}:
                resolved[raw_name] = raw_location
                for split_name in split_names:
                    resolved.setdefault(split_name, raw_location)

        # Returns `resolved` to the caller for the next stage.
        return resolved

    def _project_fields_back_to_original_locations(
        self,
        sample_id: int,
        # Completes the multi-line `collection` arguments or elements.
        write_fields: Dict[str, np.ndarray],
        field_locations: Dict[str, str],
        projected_location: str,
        connectivity,
        # Continues the surrounding statement or collection definition.
        num_points: int,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        grouped: Dict[str, Dict[str, np.ndarray]] = {"point": {}, "cell": {}}
        task_cfg = self.config.get("task", {}) if isinstance(self.config, dict) else {}
        # Converts the `projection_method` to `str` so later checks use a consistent type.
        projection_method = str(task_cfg.get("projection_method") or "mean")

        for field_name, values in write_fields.items():
            target_location = field_locations.get(field_name, projected_location)
            target_location = normalize_field_location(target_location)
            # Stores the `grouped[target_location][field_name]` for use by the remaining steps in this flow.
            grouped[target_location][field_name] = project_field_values(
                values=np.asarray(values),
                source_location=projected_location,
                target_location=target_location,
                # Completes the multi-line `project_field_values` arguments or elements.
                connectivity=connectivity,
                num_points=num_points,
                projection_method=projection_method,
            )

        # Returns `grouped` to the caller for the next stage.
        return grouped

    def _project_source_fields_back_to_original_locations(
        self,
        source_grouped_fields: Dict[str, Dict[str, np.ndarray]],
        # Continues the surrounding statement or collection definition.
        projected_location: str,
        connectivity,
        num_points: int,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """将按来源重组的数组投影回各自的 point/cell 位置。"""
        # Initializes the `grouped` as a collection for values from this processing stage.
        grouped: Dict[str, Dict[str, np.ndarray]] = {"point": {}, "cell": {}}
        task_cfg = self.config.get("task", {}) if isinstance(self.config, dict) else {}
        projection_method = str(task_cfg.get("projection_method") or "mean")
        for target_location, fields in source_grouped_fields.items():
            # Visits each item in `fields.items()` and applies the current processing step.
            for field_name, values in fields.items():
                grouped[target_location][field_name] = project_field_values(
                    values=np.asarray(values),
                    source_location=projected_location,
                    # Completes the multi-line `project_field_values` arguments or elements.
                    target_location=target_location,
                    connectivity=connectivity,
                    num_points=num_points,
                    projection_method=projection_method,
                # Completes the multi-line `project_field_values` arguments or elements.
                )
        return grouped

    @staticmethod
    def _write_dataset_with_grouped_fields(
        # Continues the surrounding statement or collection definition.
        dataset,
        output_file: str,
        grouped_fields: Dict[str, Dict[str, np.ndarray]],
        point_coordinates: Optional[np.ndarray] = None,
    # Continues the surrounding statement or collection definition.
    ) -> None:
        from tfai.io_utils.io_vtk import VTKIOWriter

        if point_coordinates is not None:
            VTKIOWriter.update_dataset_points(dataset, point_coordinates)
        # Calls `VTKIOWriter.clear_dataset_field_arrays` to perform the operation required at this stage.
        VTKIOWriter.clear_dataset_field_arrays(dataset)
        if grouped_fields.get("point"):
            VTKIOWriter._attach_fields(dataset, grouped_fields["point"], "point")
        if grouped_fields.get("cell"):
            # Calls `VTKIOWriter._attach_fields` to perform the operation required at this stage.
            VTKIOWriter._attach_fields(dataset, grouped_fields["cell"], "cell")
        VTKIOWriter._write_dataset(dataset, output_file)

    def write(
        self,
        # Continues the surrounding statement or collection definition.
        sample_id: int,
        sample_file: str,
        processed_fields: Dict[str, np.ndarray],
        output_dir: str,
        # Continues the surrounding statement or collection definition.
        output_basename: Optional[str] = None,
        point_coordinates: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """按模板类型安全写回 VTK 结果。"""
        try:
            # Imports `VTKIOReader, VTKIOWriter` from `tfai.io_utils.io_vtk` for use in this module.
            from tfai.io_utils.io_vtk import VTKIOReader, VTKIOWriter

            if not processed_fields:
                self._log("warning", f"样本 {sample_id} 没有可写回的 VTK 字段")
                return None

            # Stores the task settings for use by the remaining steps in this flow.
            task_cfg = self.config.get("task", {}) if isinstance(self.config, dict) else {}
            raw_map = task_cfg.get("field_component_map", {})
            effective_sources = self._normalize_effective_sources(
                task_cfg.get("effective_variable_sources", {})
            # Completes the multi-line `self._normalize_effective_sources` arguments or elements.
            )
            if not effective_sources and (not isinstance(raw_map, dict) or not raw_map):
                raise ValueError("VTK 写回时必须提供 task.field_component_map")

            raw_dataset = VTKIOReader.read_dataset(sample_file)
            # Stores the `projected_location` for use by the remaining steps in this flow.
            projected_location = self._resolve_projected_location()

            if effective_sources:
                source_grouped_fields = self._reassemble_vtk_fields_by_sources(
                    processed_fields,
                    # Completes the multi-line `self._reassemble_vtk_fields_by_sources` arguments or elements.
                    effective_sources,
                )
                write_fields = {
                    f"{location}:{field_name}": values
                    # Completes the multi-line `collection` arguments or elements.
                    for location, fields in source_grouped_fields.items()
                    for field_name, values in fields.items()
                }
            else:
                # Stores the `component_map` for use by the remaining steps in this flow.
                component_map = self._normalize_component_map(raw_map)
                configured_locations = self._normalize_location_map(
                    task_cfg.get("field_location_map", {})
                )
                # Stores the `dataset_locations` for use by the remaining steps in this flow.
                dataset_locations = self._infer_dataset_field_locations(
                    raw_dataset,
                    component_map,
                )
                # Stores the `field_locations` for use by the remaining steps in this flow.
                field_locations = self._resolve_output_field_locations(
                    component_map=component_map,
                    configured_locations=configured_locations,
                    dataset_locations=dataset_locations,
                # Completes the multi-line `self._resolve_output_field_locations` arguments or elements.
                )
                write_fields = self._reassemble_vtk_fields(processed_fields, component_map)
            if not write_fields:
                self._log("warning", f"样本 {sample_id} 在字段重组后没有可写回的 VTK 字段")
                # Returns `None` to the caller for the next stage.
                return None

            source_ext = os.path.splitext(sample_file)[1].lower()
            write_ext, downgraded = VTKIOWriter.resolve_output_extension_for_dataset(
                raw_dataset,
                # Completes the multi-line `VTKIOWriter.resolve_output_extension_for_dataset` arguments or elements.
                source_ext,
            )
            target_dataset = raw_dataset if not downgraded else VTKIOReader.read_canonical_dataset(sample_file)
            if downgraded:
                # Calls `self._log` to perform the operation required at this stage.
                self._log("info", f"样本 {sample_id} 降级写出为 {write_ext} 以保证兼容性")
            else:
                self._log("info", f"样本 {sample_id} 保持原始 VTK 类型写回")

            num_points = target_dataset.GetNumberOfPoints()
            # Stores the `connectivity` for use by the remaining steps in this flow.
            connectivity = VTKIOReader.extract_connectivity(target_dataset, one_based=True)
            expected_count = num_points if projected_location == "point" else len(connectivity)
            self._validate_entity_count(
                sample_id=sample_id,
                # Completes the multi-line `self._validate_entity_count` arguments or elements.
                fields=write_fields,
                expected=expected_count,
                location=projected_location,
            )

            # Checks `effective_sources` before entering this conditional branch.
            if effective_sources:
                grouped_fields = self._project_source_fields_back_to_original_locations(
                    source_grouped_fields=source_grouped_fields,
                    projected_location=projected_location,
                    # Completes the multi-line `self._project_source_fields_back_to_original_locations` arguments or elements.
                    connectivity=connectivity,
                    num_points=num_points,
                )
            else:
                # Stores the `grouped_fields` for use by the remaining steps in this flow.
                grouped_fields = self._project_fields_back_to_original_locations(
                    sample_id=sample_id,
                    write_fields=write_fields,
                    field_locations=field_locations,
                    # Completes the multi-line `self._project_fields_back_to_original_locations` arguments or elements.
                    projected_location=projected_location,
                    connectivity=connectivity,
                    num_points=num_points,
                )

            # Stores the `basename` for use by the remaining steps in this flow.
            basename = output_basename or f"predicted_{sample_id}"
            output_file = os.path.join(output_dir, f"{basename}{write_ext}")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            if not downgraded:
                # Calls `self._write_dataset_with_grouped_fields` to perform the operation required at this stage.
                self._write_dataset_with_grouped_fields(
                    dataset=target_dataset,
                    output_file=output_file,
                    grouped_fields=grouped_fields,
                    # Completes the multi-line `self._write_dataset_with_grouped_fields` arguments or elements.
                    point_coordinates=point_coordinates,
                )
            else:
                downgraded_points = (
                    # Completes the multi-line `np.asarray` arguments or elements.
                    np.asarray(point_coordinates, dtype=np.float64)
                    if point_coordinates is not None
                    else VTKIOReader.extract_points(target_dataset)
                )
                # Stores the `downgraded_grid` for use by the remaining steps in this flow.
                downgraded_grid = VTKIOWriter.build_unstructured_grid_with_cell_types(
                    points=downgraded_points,
                    typed_cells=VTKIOReader.extract_cells_with_types(target_dataset, one_based=True),
                    one_based=True,
                # Completes the multi-line `VTKIOWriter.build_unstructured_grid_with_cell_types` arguments or elements.
                )
                self._write_dataset_with_grouped_fields(
                    dataset=downgraded_grid,
                    output_file=output_file,
                    # Completes the multi-line `self._write_dataset_with_grouped_fields` arguments or elements.
                    grouped_fields=grouped_fields,
                    point_coordinates=None,
                )

            written_count = len(grouped_fields.get("point", {})) + len(grouped_fields.get("cell", {}))
            # Calls `self._log` to perform the operation required at this stage.
            self._log("info", f"样本 {sample_id} 已写回 {written_count} 个字段到 {write_ext}")
            return output_file

        except Exception as exc:
            self._log("error", f"样本 {sample_id} VTK 写回失败: {str(exc)}")
            # Returns `None` to the caller for the next stage.
            return None
