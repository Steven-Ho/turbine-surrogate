from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Imports `BaseReader` from `base_reader` for use in this module.
from .base_reader import BaseReader
from ..data.base_data import BaseData
from ..data.temporal_utils import (
    aggregate_field_temporal_sequence,
    # Continues the surrounding statement or collection definition.
    infer_displacement_effective_variables,
    infer_displacement_logical_name,
    is_temporal_manifest_path,
    list_temporal_sequence_files,
# Continues the surrounding statement or collection definition.
)
from ..io_utils.io_vtk import (
    VTKIOReader,
)
# Imports `SampleMap` from `sample_map` for use in this module.
from .sample_map import SampleMap
from ..utils.field_projection import (
    build_component_location_map,
    choose_projected_point_or_cell,
    # Continues the surrounding statement or collection definition.
    merge_field_location_maps,
    project_fields_to_location,
)
from ..utils.field_variable_catalog import (
    # Continues the surrounding statement or collection definition.
    build_field_variable_catalog,
    catalog_component_map,
    catalog_sources_by_runtime_key,
    select_catalog_variables,
# Continues the surrounding statement or collection definition.
)
from ..utils.variable_resolver import (
    resolve_effective_variables,
)
# Imports `normalize_field_bc_names` from `utils.boundary_feature_encoding` for use in this module.
from ..utils.boundary_feature_encoding import normalize_field_bc_names
from ..utils.path_utils import path_endswith


class VTKReader(BaseReader):
    """VTK文件读取器，基于io_utils.io_vtk模块。"""

    SUPPORTED_EXTENSIONS = tuple(VTKIOReader.supported_extensions)

    # Defines `__init__` for   init   logic.
    def __init__(self, config: Dict):
        super().__init__(config)
        self.task_type = config['task']['type']
        self.mode = config['task']['mode']
        # Stores the `variables` on the instance so other methods can reuse it.
        self.variables = config['task']['variables']
        self.point_or_cell = None
        self.geometry_policy = config.get('task', {}).get('vtk_geometry_policy', '3d_coords')
        self.supported_extensions = self.SUPPORTED_EXTENSIONS

        # Reads `train_samples` from the mapping into the `sample_table_path`.
        sample_table_path = config.get('general', {}).get('train_samples')
        self.sample_map = SampleMap(sample_table_path) if sample_table_path else None

    @staticmethod
    def _build_variable_catalog_from_dataset(dataset) -> Dict[str, Dict[str, object]]:
        """按 VTK 原始数组构建可选变量目录，不依赖数组分量名称。"""
        # Initializes the `entries` as a collection for values from this processing stage.
        entries = []

        def collect_from_attr(data_attr, location: str) -> None:
            if data_attr is None:
                return
            # Visits each item in `range(data_attr.GetNumberOfArrays())` and applies the current processing step.
            for index in range(data_attr.GetNumberOfArrays()):
                array = data_attr.GetArray(index)
                if array is None:
                    continue
                # Adds the current value to `entries` for later aggregation.
                entries.append(
                    {
                        "format": "vtk",
                        "location": location,
                        # Completes the multi-line `array.GetName` arguments or elements.
                        "array_name": array.GetName() or f"array_{index}",
                        "num_components": int(array.GetNumberOfComponents() or 1),
                    }
                )

        # Calls `collect_from_attr` to perform the operation required at this stage.
        collect_from_attr(dataset.GetPointData(), "point")
        collect_from_attr(dataset.GetCellData(), "cell")
        return build_field_variable_catalog(entries)

    def _build_component_map_from_dataset(self, dataset) -> Dict[str, List[str]]:
        # Returns `` to the caller for the next stage.
        return catalog_component_map(self._build_variable_catalog_from_dataset(dataset))

    def _resolve_sample_info(self, file_path: str, sample_order: Optional[int]) -> Dict:
        sample_info = {'sample_order': sample_order}
        if not self.sample_map:
            # Returns `sample_info` to the caller for the next stage.
            return sample_info

        if sample_order is not None:
            info = self.sample_map.get_sample_info(sample_order)
            if info:
                # Adds the current value to `sample_info` for later aggregation.
                sample_info.update(info)
            return sample_info

        for order in self.sample_map.get_all_samples():
            mapped = self.sample_map.get_file_path(order)
            # Checks `not mapped` before entering this conditional branch.
            if not mapped:
                continue
            if path_endswith(file_path, mapped):
                info = self.sample_map.get_sample_info(order)
                # Checks `info` before entering this conditional branch.
                if info:
                    sample_info.update(info)
                break
        return sample_info

    # 返回需在目标字段筛选后保留的 VTK 局部特征字段。
    def _resolve_local_feature_field_names(self) -> List[str]:
        """返回需在目标字段筛选后保留的 VTK 局部特征字段。"""
        task_cfg = self.config.get("task", {})
        if str(task_cfg.get("local_feature_encoding", "")).strip().lower() != "vtk_based":
            return []
        # Returns `normalize_field_bc_names(task_cfg.get("field_bc"))` to the caller for the next stage.
        return normalize_field_bc_names(task_cfg.get("field_bc"))

    def _resolve_raw_fields_for_loading(
        self,
        dataset,
        # Continues the surrounding statement or collection definition.
        logical_variables: List[str],
        read_all_fields: bool,
    ) -> Optional[List[str]]:
        """根据数组元数据解析实际需转换的原始 VTK 数组名。"""
        if read_all_fields:
            # Returns `None` to the caller for the next stage.
            return None

        variable_catalog = self._build_variable_catalog_from_dataset(dataset)
        requested_names = list(logical_variables)
        for field_name in self._resolve_local_feature_field_names():
            # Checks `field_name not in requested_names` before entering this conditional branch.
            if field_name not in requested_names:
                requested_names.append(field_name)
        try:
            (
                # Completes the multi-line `collection` arguments or elements.
                _,
                effective_names,
                effective_sources,
                _,
                # Completes the multi-line `collection` arguments or elements.
                _,
            ) = select_catalog_variables(
                variable_catalog,
                requested_names,
                # Completes the multi-line `select_catalog_variables` arguments or elements.
                strict_missing=False,
            )
        except ValueError:
            # 保持缺失字段预测路径的原有兼容行为，交由后续逻辑构造占位目标。
            return None
        # Returns early when `not effective_names` makes the remaining work inapplicable.
        if not effective_names:
            return None

        raw_names = []
        for runtime_key in effective_names:
            # Stores the `source` for use by the remaining steps in this flow.
            source = effective_sources.get(runtime_key) or {}
            raw_name = str(source.get("array_name") or "")
            if raw_name and raw_name not in raw_names:
                raw_names.append(raw_name)
        # Returns `raw_names or None` to the caller for the next stage.
        return raw_names or None

    def _resolve_temporal_raw_field_names(
        self,
        dataset,
        # Continues the surrounding statement or collection definition.
        logical_variables: List[str],
        read_all_fields: bool,
    ) -> List[str]:
        """确定时序后续帧按名称直接读取的原始字段列表。"""
        raw_names = self._resolve_raw_fields_for_loading(
            # Completes the multi-line `self._resolve_raw_fields_for_loading` arguments or elements.
            dataset,
            logical_variables,
            read_all_fields,
        )
        # Returns early when `raw_names is not None` makes the remaining work inapplicable.
        if raw_names is not None:
            return list(raw_names)
        raw_names = []
        for source in catalog_sources_by_runtime_key(
            # Completes the multi-line `self._build_variable_catalog_from_dataset` arguments or elements.
            self._build_variable_catalog_from_dataset(dataset)
        ).values():
            raw_name = str(source.get("array_name") or "")
            if raw_name and raw_name not in raw_names:
                # Adds the current value to `raw_names` for later aggregation.
                raw_names.append(raw_name)
        return raw_names

    @staticmethod
    def _can_direct_lookup_raw_fields(raw_field_names: List[str]) -> bool:
        """匿名 VTK 数组无法按名称获取，保留遍历式读取作为兼容回退。"""
        # Returns `` to the caller for the next stage.
        return all(
            not str(field_name).startswith("array_")
            for field_name in raw_field_names
        )

    # 仅读取 VTK 数组元数据，保留全部字段的位置映射。
    def _build_field_location_map_from_dataset(self, dataset) -> Dict[str, str]:
        """仅读取 VTK 数组元数据，保留全部字段的位置映射。"""
        return {
            runtime_key: str(source.get("location"))
            for runtime_key, source in catalog_sources_by_runtime_key(
                # Completes the multi-line `self._build_variable_catalog_from_dataset` arguments or elements.
                self._build_variable_catalog_from_dataset(dataset)
            ).items()
        }

    @staticmethod
    # Defines `_infer_mesh_dimension_from_points` for  infer mesh dimension from points logic.
    def _infer_mesh_dimension_from_points(points: np.ndarray) -> Optional[int]:
        coords = np.asarray(points, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[0] == 0 or coords.shape[1] == 0:
            return None

        # VTK 通常始终使用三列坐标，这里按有效坐标跨度推断实际几何维度。
        # Stores the `coord_ranges` for use by the remaining steps in this flow.
        coord_ranges = np.ptp(coords, axis=0)
        active_axes = int(np.count_nonzero(coord_ranges > 1e-12))
        if active_axes <= 0:
            return 0
        # Returns `active_axes` to the caller for the next stage.
        return active_axes

    @staticmethod
    def _infer_mesh_dimension(dataset, points: np.ndarray) -> Optional[int]:
        if dataset is not None and dataset.GetNumberOfCells() > 0:
            # Stores the `max_dim` for use by the remaining steps in this flow.
            max_dim = 0
            for cell_idx in range(dataset.GetNumberOfCells()):
                cell_dim = int(dataset.GetCell(cell_idx).GetCellDimension())
                if cell_dim > max_dim:
                    # Stores the `max_dim` for use by the remaining steps in this flow.
                    max_dim = cell_dim
            if max_dim > 0:
                return max_dim

        return VTKReader._infer_mesh_dimension_from_points(points)

    # Defines `_resolve_mixed_fields` for  resolve mixed fields logic.
    def _resolve_mixed_fields(
        self,
        point_fields: Dict[str, np.ndarray],
        cell_fields: Dict[str, np.ndarray],
        # Completes the multi-line `collection` arguments or elements.
        point_meta: Dict[str, Dict[str, object]],
        cell_meta: Dict[str, Dict[str, object]],
        point_component_map: Dict[str, List[str]],
        cell_component_map: Dict[str, List[str]],
        # Continues the surrounding statement or collection definition.
        connectivity,
        num_points: int,
        requested_vars: Optional[List[str]],
    ) -> Tuple[
        # Completes the multi-line `collection` arguments or elements.
        Dict[str, np.ndarray],
        Dict[str, Dict[str, object]],
        Dict[str, List[str]],
        Dict[str, str],
        # Completes the multi-line `collection` arguments or elements.
        str,
    ]:
        def infer_meta(values: np.ndarray) -> Dict[str, object]:
            arr = np.asarray(values)
            # Checks `arr.ndim == 1` before entering this conditional branch.
            if arr.ndim == 1:
                n_comp = 1
                comp_shape = ()
            else:
                # Converts the `n_comp` to `int` so later checks use a consistent type.
                n_comp = int(np.prod(arr.shape[1:]))
                comp_shape = tuple(arr.shape[1:])

            if n_comp == 1:
                attr = 'scalar'
            # Checks `n_comp == 3` before entering this conditional branch.
            elif n_comp == 3:
                attr = 'vector'
            elif n_comp in (6, 9):
                attr = 'tensor'
            # Continues the surrounding statement or collection definition.
            else:
                attr = 'multi_component'

            return {
                'num_components': n_comp,
                # Completes the multi-line `collection` arguments or elements.
                'attribute': attr,
                'component_shape': comp_shape,
            }

        field_location_map = merge_field_location_maps(
            # Completes the multi-line `build_component_location_map` arguments or elements.
            build_component_location_map(point_component_map, "point"),
            build_component_location_map(cell_component_map, "cell"),
        )
        projected_point_or_cell = choose_projected_point_or_cell(
            # Completes the multi-line `choose_projected_point_or_cell` arguments or elements.
            requested_vars=requested_vars,
            point_component_map=point_component_map,
            cell_component_map=cell_component_map,
            default_location="point",
        # Completes the multi-line `choose_projected_point_or_cell` arguments or elements.
        )

        merged_source_fields = dict(point_fields)
        merged_source_fields.update(cell_fields)
        projected_fields = project_fields_to_location(
            # Completes the multi-line `project_fields_to_location` arguments or elements.
            fields=merged_source_fields,
            field_location_map=field_location_map,
            target_location=projected_point_or_cell,
            connectivity=connectivity,
            # Completes the multi-line `project_fields_to_location` arguments or elements.
            num_points=num_points,
            projection_method="mean",
        )

        merged_meta = {}
        # Visits each item in `projected_fields.items()` and applies the current processing step.
        for name, values in projected_fields.items():
            if name in point_meta:
                merged_meta[name] = dict(point_meta[name])
            elif name in cell_meta:
                # Stores the `merged_meta[name]` for use by the remaining steps in this flow.
                merged_meta[name] = dict(cell_meta[name])
            else:
                merged_meta[name] = infer_meta(values)

        merged_component_map = {k: list(v) for k, v in point_component_map.items()}
        # Visits each item in `cell_component_map.items()` and applies the current processing step.
        for raw_name, split_names in cell_component_map.items():
            merged_component_map.setdefault(raw_name, list(split_names))

        self.point_or_cell = projected_point_or_cell
        return (
            # Completes the multi-line `collection` arguments or elements.
            projected_fields,
            merged_meta,
            merged_component_map,
            field_location_map,
            # Completes the multi-line `collection` arguments or elements.
            projected_point_or_cell,
        )

    @staticmethod
    def _component_suffixes(num_components: int) -> List[str]:
        # Returns `[f"_{i}" for i in range(num_components)]` to the caller for the next stage.
        return [f"_{i}" for i in range(num_components)]

    def _split_multicomponent_fields(
        self,
        fields: Dict[str, np.ndarray],
        # Completes the multi-line `collection` arguments or elements.
        field_meta: Dict[str, Dict[str, object]],
        variable_catalog: Optional[Dict[str, Dict[str, object]]] = None,
        location: Optional[str] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, object]], Dict[str, List[str]]]:
        # Initializes the `split_fields` as a collection for values from this processing stage.
        split_fields: Dict[str, np.ndarray] = {}
        split_meta: Dict[str, Dict[str, object]] = {}
        component_map: Dict[str, List[str]] = {}

        sources_by_array: Dict[str, List[Dict[str, object]]] = {}
        # Checks `variable_catalog and location in {"point", "cell"}` before entering this conditional branch.
        if variable_catalog and location in {"point", "cell"}:
            for entry in variable_catalog.values():
                if entry.get("location") != location:
                    continue
                # Converts the `raw_name` to `str` so later checks use a consistent type.
                raw_name = str(entry.get("array_name") or "")
                sources = [dict(source) for source in (entry.get("sources") or [])]
                if raw_name and sources:
                    sources_by_array.setdefault(raw_name, []).append(
                        # Completes the multi-line `collection` arguments or elements.
                        {
                            "selection_key": str(entry.get("selection_key") or raw_name),
                            "sources": sources,
                        }
                    # Completes the multi-line `sources_by_array.setdefault(raw_name, []).append` arguments or elements.
                    )

        for name, values in fields.items():
            arr = np.asarray(values)
            meta = field_meta.get(name, {})
            # Converts the `n_comp` to `int` so later checks use a consistent type.
            n_comp = int(meta.get('num_components', 1 if arr.ndim == 1 else np.prod(arr.shape[1:])))
            source_entries = sources_by_array.get(name) or []
            source_entry = source_entries[0] if len(source_entries) == 1 else None
            source_components = list(source_entry.get("sources") or []) if source_entry else []
            # Stores the `selection_key` for use by the remaining steps in this flow.
            selection_key = str(source_entry.get("selection_key") or name) if source_entry else name

            if source_components and len(source_components) != n_comp:
                raise ValueError(
                    f"VTK 数组分量数与变量目录不一致: array={name}, "
                    # Completes the multi-line `len` arguments or elements.
                    f"catalog={len(source_components)}, actual={n_comp}"
                )

            if arr.ndim == 1 or n_comp == 1:
                runtime_key = (
                    # Completes the multi-line `str` arguments or elements.
                    str(source_components[0].get("runtime_key"))
                    if source_components
                    else name
                )
                # Stores the `split_fields[runtime_key]` for use by the remaining steps in this flow.
                split_fields[runtime_key] = arr
                split_meta[runtime_key] = {
                    'num_components': 1,
                    'attribute': 'scalar',
                    # Completes the multi-line `collection` arguments or elements.
                    'component_shape': (),
                    'source_field': name,
                    'component_index': 0,
                }
                # Initializes the `component_map[selection_key]` as a collection for values from this processing stage.
                component_map[selection_key] = [runtime_key]
                continue

            arr_2d = arr.reshape(arr.shape[0], -1)
            suffixes = self._component_suffixes(arr_2d.shape[1])
            # Initializes the `split_names` as a collection for values from this processing stage.
            split_names: List[str] = []

            for ci in range(arr_2d.shape[1]):
                comp_name = (
                    str(source_components[ci].get("runtime_key"))
                    # Continues the surrounding statement or collection definition.
                    if source_components
                    else f"{name}{suffixes[ci]}"
                )
                split_fields[comp_name] = arr_2d[:, ci]
                # Initializes the `split_meta[comp_name]` as a collection for values from this processing stage.
                split_meta[comp_name] = {
                    'num_components': 1,
                    'attribute': 'scalar',
                    'component_shape': (),
                    # Completes the multi-line `collection` arguments or elements.
                    'source_field': name,
                    'component_index': ci,
                }
                split_names.append(comp_name)

            # Stores the `component_map[selection_key]` for use by the remaining steps in this flow.
            component_map[selection_key] = split_names

        return split_fields, split_meta, component_map

    @staticmethod
    def _extract_catalog_fields_from_dataset(
        # Continues the surrounding statement or collection definition.
        dataset,
        variable_catalog: Dict[str, Dict[str, object]],
        location: str,
        raw_field_names: Optional[List[str]],
    # Completes the multi-line `collection` arguments or elements.
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, object]], Dict[str, List[str]]]:
        """按 VTK 数组索引提取字段，避免同名数组在字典中被覆盖。"""
        from vtk.util import numpy_support

        data_attr = dataset.GetPointData() if location == "point" else dataset.GetCellData()
        expected_size = dataset.GetNumberOfPoints() if location == "point" else dataset.GetNumberOfCells()
        # Stores the `allowed_names` for use by the remaining steps in this flow.
        allowed_names = set(raw_field_names) if raw_field_names else None
        catalog_entries: Dict[Tuple[str, int], Dict[str, object]] = {}
        for entry in variable_catalog.values():
            if entry.get("location") != location:
                # Handles the data or control-flow case represented by this statement.
                continue
            catalog_entries[(
                str(entry.get("array_name") or "").casefold(),
                int(entry.get("occurrence", 0)),
            # Completes the multi-line `collection` arguments or elements.
            )] = entry

        fields: Dict[str, np.ndarray] = {}
        field_meta: Dict[str, Dict[str, object]] = {}
        component_map: Dict[str, List[str]] = {}
        # Initializes the `occurrences` as a collection for values from this processing stage.
        occurrences: Dict[str, int] = {}
        if data_attr is None:
            return fields, field_meta, component_map

        for index in range(data_attr.GetNumberOfArrays()):
            # Stores the `array` for use by the remaining steps in this flow.
            array = data_attr.GetArray(index)
            if array is None:
                continue
            raw_name = array.GetName() or f"array_{index}"
            # Stores the `normalized_raw_name` for use by the remaining steps in this flow.
            normalized_raw_name = str(raw_name).casefold()
            occurrence = occurrences.get(normalized_raw_name, 0)
            occurrences[normalized_raw_name] = occurrence + 1
            if allowed_names is not None and raw_name not in allowed_names:
                # Handles the data or control-flow case represented by this statement.
                continue

            entry = catalog_entries.get((normalized_raw_name, occurrence))
            if entry is None:
                continue
            # Stores the `sources` for use by the remaining steps in this flow.
            sources = list(entry.get("sources") or [])
            component_count = int(array.GetNumberOfComponents() or 1)
            if len(sources) != component_count:
                raise ValueError(
                    # Completes the multi-line `ValueError` arguments or elements.
                    f"VTK 数组分量数与变量目录不一致: array={raw_name}, "
                    f"catalog={len(sources)}, actual={component_count}"
                )

            values = VTKIOReader._normalize_extracted_array(
                # Completes the multi-line `numpy_support.vtk_to_numpy` arguments or elements.
                numpy_support.vtk_to_numpy(array),
                expected_size,
            )
            if values is None:
                # Handles the data or control-flow case represented by this statement.
                continue
            values_2d = np.asarray(values).reshape(expected_size, -1)
            runtime_keys = []
            for component_index, source in enumerate(sources):
                # Converts the `runtime_key` to `str` so later checks use a consistent type.
                runtime_key = str(source.get("runtime_key") or "")
                if not runtime_key:
                    raise ValueError(f"VTK 变量目录缺少运行时分量键: {raw_name}")
                fields[runtime_key] = values_2d[:, component_index]
                # Initializes the `field_meta[runtime_key]` as a collection for values from this processing stage.
                field_meta[runtime_key] = {
                    "num_components": 1,
                    "attribute": "scalar",
                    "component_shape": (),
                    # Completes the multi-line `collection` arguments or elements.
                    "source_field": raw_name,
                    "component_index": component_index,
                }
                runtime_keys.append(runtime_key)
            # Stores the `get('selection_key') or raw_name)]` for use by the remaining steps in this flow.
            component_map[str(entry.get("selection_key") or raw_name)] = runtime_keys

        return fields, field_meta, component_map

    @staticmethod
    def _requires_indexed_catalog_extraction(
        # Completes the multi-line `collection` arguments or elements.
        variable_catalog: Dict[str, Dict[str, object]],
    ) -> bool:
        """仅在同一位置重复原始数组名时绕过常规字典式提取。"""
        seen = set()
        for entry in variable_catalog.values():
            # Initializes the `key` as a collection for values from this processing stage.
            key = (str(entry.get("location")), str(entry.get("array_name")))
            if key in seen:
                return True
            seen.add(key)
        # Returns `False` to the caller for the next stage.
        return False

    @staticmethod
    def _filter_fields(
        fields: Dict[str, np.ndarray],
        # Completes the multi-line `collection` arguments or elements.
        field_meta: Dict[str, Dict[str, object]],
        keep_names: List[str],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, object]]]:
        keep_set = set(keep_names)
        # Builds the `filtered_fields` using the comprehension's filter and mapping rules.
        filtered_fields = {k: v for k, v in fields.items() if k in keep_set}
        filtered_meta = {k: v for k, v in field_meta.items() if k in keep_set}
        return filtered_fields, filtered_meta

    @staticmethod
    # Defines `_resolve_requested_names` for  resolve requested names logic.
    def _resolve_requested_names(
        requested_vars: Optional[List[str]],
        available_fields: Dict[str, np.ndarray],
        component_map: Dict[str, List[str]],
        # Continues the surrounding statement or collection definition.
        strict_missing: bool = False,
    ) -> List[str]:
        if not requested_vars:
            return list(available_fields.keys())

        # Initializes the `resolved` as a collection for values from this processing stage.
        resolved: List[str] = []
        missing: List[str] = []

        for var in requested_vars:
            if var in available_fields:
                # Adds the current value to `resolved` for later aggregation.
                resolved.append(var)
                continue

            if var in component_map:
                resolved.extend(component_map[var])
                # Handles the data or control-flow case represented by this statement.
                continue

            missing.append(var)

        if missing and (strict_missing or not resolved):
            raise ValueError(f"Requested variables not found after VTK component expansion: {missing}")

        # preserve order and remove duplicates
        # Returns `list(dict.fromkeys(resolved))` to the caller for the next stage.
        return list(dict.fromkeys(resolved))

    def _resolve_fieldless_point_or_cell(self) -> str:
        configured = str(self.config.get('task', {}).get('point_or_cell', '')).strip().lower()
        if configured in {'point', 'cell'}:
            # Returns `configured` to the caller for the next stage.
            return configured
        if self.point_or_cell in {'point', 'cell'}:
            return self.point_or_cell
        return 'point'

    # 仅扫描时序 VTK 的坐标与拓扑，避免变量清单阶段重复转换全部场数组。
    def _inspect_temporal_geometry(self, file_path: str) -> Dict[str, object]:
        """仅扫描时序 VTK 的坐标与拓扑，避免变量清单阶段重复转换全部场数组。"""
        sequence_files = list_temporal_sequence_files(self, file_path)
        if not sequence_files:
            raise ValueError(f"未从时序 VTK 源读取到任何时间步: {file_path}")

        # Stores the `first_dataset` for use by the remaining steps in this flow.
        first_dataset = VTKIOReader.read_canonical_dataset_without_meta(
            sequence_files[0],
            geometry_policy=self.geometry_policy,
        )
        # Stores the `reference_points` for use by the remaining steps in this flow.
        reference_points = VTKIOReader.extract_points(first_dataset)
        reference_signature = VTKIOReader.extract_connectivity_signature(first_dataset)
        mesh_dimension = self._infer_mesh_dimension(first_dataset, reference_points)
        has_motion = False

        # Visits each item in `sequence_files[1:]` and applies the current processing step.
        for step_file in sequence_files[1:]:
            dataset = VTKIOReader.read_canonical_dataset_without_meta(
                step_file,
                geometry_policy=self.geometry_policy,
            # Completes the multi-line `VTKIOReader.read_canonical_dataset_without_meta` arguments or elements.
            )
            if not VTKIOReader.connectivity_signatures_equal(
                reference_signature,
                VTKIOReader.extract_connectivity_signature(dataset),
            # Completes the multi-line `VTKIOReader.connectivity_signatures_equal` arguments or elements.
            ):
                raise ValueError(f"时序步骤之间的单元连接关系发生变化: {step_file}")

            points = VTKIOReader.extract_points(dataset)
            if points.shape != reference_points.shape:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(
                    "时序步骤之间的点坐标形状发生变化: "
                    f"{step_file}, expected={reference_points.shape}, actual={points.shape}"
                )
            # Checks `` before entering this conditional branch.
            if not np.array_equal(points, reference_points) and not np.allclose(
                points,
                reference_points,
            ):
                # Stores the `has_motion` for use by the remaining steps in this flow.
                has_motion = True

        return {
            "geometry_regime": (
                "displaced_static_topology" if has_motion else "static_geometry"
            # Completes the multi-line `collection` arguments or elements.
            ),
            "mesh_dimension": mesh_dimension,
        }

    def resolve_variable_manifest(
        # Continues the surrounding statement or collection definition.
        self,
        file_path: str,
        logical_variables: List[str],
        sample_order: Optional[int] = None,
    # Completes the multi-line `collection` arguments or elements.
    ) -> Dict[str, object]:
        """读取全字段样本并解析训练所需的变量清单与运动网格元信息。"""
        max_time_steps = (
            1
            if self.mode == "temporal"
            # Completes the multi-line `Path(file_path).is_dir` arguments or elements.
            and (Path(file_path).is_dir() or is_temporal_manifest_path(file_path))
            else None
        )
        data = self.read(
            # Completes the multi-line `self.read` arguments or elements.
            file_path,
            variables=[],
            sample_order=sample_order,
            allow_missing_fields=True,
            # Completes the multi-line `self.read` arguments or elements.
            max_time_steps=max_time_steps,
        )
        data_dict = data.get_data()
        field_component_map = {
            # Completes the multi-line `str` arguments or elements.
            str(name): list(components or [])
            for name, components in (data_dict.get('FieldComponentMap') or {}).items()
        }
        if not isinstance(field_component_map, dict) or not field_component_map:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("样本中未解析出 FieldComponentMap，无法解析变量")

        geometry_info: Dict[str, object] = {}
        if max_time_steps is not None:
            geometry_info = self._inspect_temporal_geometry(file_path)

        # Stores the `mesh_dimension` for use by the remaining steps in this flow.
        mesh_dimension = geometry_info.get("mesh_dimension") or data.get_mesh_dimension()
        point_coordinates = np.asarray(data_dict.get("PointCoordinates"))
        coordinate_dim = int(point_coordinates.shape[1]) if point_coordinates.ndim == 2 else 3
        detected_geometry_regime = (
            # Completes the multi-line `geometry_info.get` arguments or elements.
            geometry_info.get("geometry_regime")
            or data.get_geometry_regime()
            or "static_geometry"
        )
        # Stores the `variable_catalog` for use by the remaining steps in this flow.
        variable_catalog = data_dict.get('VariableCatalog') or {}
        if isinstance(variable_catalog, dict) and variable_catalog:
            (
                field_component_map,
                # Completes the multi-line `collection` arguments or elements.
                effective_variables,
                effective_variable_sources,
                field_location_map,
                _,
            # Completes the multi-line `select_catalog_variables` arguments or elements.
            ) = select_catalog_variables(variable_catalog, logical_variables)
        else:
            effective_variables = resolve_effective_variables(
                logical_variables,
                # Completes the multi-line `resolve_effective_variables` arguments or elements.
                field_component_map,
                strict_missing=True,
            )
            effective_variable_sources = {
                # Completes the multi-line `str` arguments or elements.
                str(name): dict(source)
                for name, source in (data_dict.get('EffectiveVariableSources') or {}).items()
                if str(name) in effective_variables
            }
            # Builds the `field_location_map` using the comprehension's filter and mapping rules.
            field_location_map = {
                str(k): str(v)
                for k, v in (data_dict.get('FieldLocationMap') or {}).items()
                if str(k) in effective_variables
            # Continues the surrounding statement or collection definition.
            }
        if not effective_variables:
            raise ValueError("样本中未解析出有效变量，无法解析变量")

        coord_dim = (
            # Continues the surrounding statement or collection definition.
            coordinate_dim
            if detected_geometry_regime == "displaced_static_topology"
            else (int(mesh_dimension) if mesh_dimension is not None else 3)
        )
        # Stores the `displacement_logical_name` for use by the remaining steps in this flow.
        displacement_logical_name = infer_displacement_logical_name(
            field_component_map,
            coord_dim=coord_dim,
        )
        # Stores the `displacement_effective_variables` for use by the remaining steps in this flow.
        displacement_effective_variables = infer_displacement_effective_variables(
            field_component_map,
            displacement_logical_name,
            coord_dim=coord_dim,
        # Completes the multi-line `infer_displacement_effective_variables` arguments or elements.
        )
        geometry_motion = dict(data_dict.get("GeometryMotion") or {})
        if geometry_motion:
            displacement_logical_name = geometry_motion.get(
                # Completes the multi-line `geometry_motion.get` arguments or elements.
                "displacement_logical_name",
                displacement_logical_name,
            )
            displacement_effective_variables = list(
                # Completes the multi-line `geometry_motion.get` arguments or elements.
                geometry_motion.get("displacement_effective_variables")
                or displacement_effective_variables
            )
            geometry_motion["displacement_logical_name"] = displacement_logical_name
            # Stores the `geometry_motion['displacement_effective_variables']` for use by the remaining steps in this flow.
            geometry_motion["displacement_effective_variables"] = list(
                displacement_effective_variables
            )
        return {
            # Completes the multi-line `list` arguments or elements.
            'logical_variables': list(logical_variables),
            'effective_variables': list(effective_variables),
            'field_component_map': {k: list(v) for k, v in field_component_map.items()},
            'variable_catalog': variable_catalog,
            # Completes the multi-line `collection` arguments or elements.
            'effective_variable_sources': effective_variable_sources,
            'point_or_cell': data.point_or_cell,
            'field_location_map': field_location_map,
            'projected_point_or_cell': data_dict.get('ProjectedPointOrCell') or data.point_or_cell,
            # Completes the multi-line `data_dict.get` arguments or elements.
            'mixed_field_policy': data_dict.get('MixedFieldPolicy') or 'majority_side',
            'projection_method': data_dict.get('ProjectionMethod') or 'mean',
            'mesh_dimension': mesh_dimension,
            'geometry_regime': (
                # Completes the multi-line `collection` arguments or elements.
                detected_geometry_regime
            ),
            'geometry_motion': geometry_motion,
            'displacement_logical_variable': displacement_logical_name,
            # Completes the multi-line `list` arguments or elements.
            'displacement_effective_variables': list(displacement_effective_variables),
        }

    def _extract_fields_from_dataset(
        self,
        # Continues the surrounding statement or collection definition.
        dataset,
        logical_variables: List[str],
        read_all_fields: bool,
        allow_missing_fields: bool,
        # Continues the surrounding statement or collection definition.
        connectivity,
        num_points: int,
        raw_field_names: Optional[List[str]] = None,
        all_field_location_map: Optional[Dict[str, str]] = None,
        # Continues the surrounding statement or collection definition.
        direct_field_lookup: bool = False,
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, Dict[str, object]],
        # Completes the multi-line `collection` arguments or elements.
        Dict[str, List[str]],
        Dict[str, str],
        str,
        List[str],
        # Completes the multi-line `collection` arguments or elements.
        Dict[str, Dict[str, object]],
        Dict[str, Dict[str, object]],
    ]:
        """从单个 VTK 数据集解析字段，供首帧和后续时间步复用。"""
        variable_catalog = self._build_variable_catalog_from_dataset(dataset)
        # Checks `raw_field_names is None` before entering this conditional branch.
        if raw_field_names is None:
            raw_field_names = self._resolve_raw_fields_for_loading(
                dataset,
                logical_variables,
                # Completes the multi-line `self._resolve_raw_fields_for_loading` arguments or elements.
                read_all_fields,
            )
        if self._requires_indexed_catalog_extraction(variable_catalog):
            point_fields, point_meta, point_component_map = self._extract_catalog_fields_from_dataset(
                # Completes the multi-line `self._extract_catalog_fields_from_dataset` arguments or elements.
                dataset,
                variable_catalog,
                "point",
                raw_field_names,
            # Completes the multi-line `self._extract_catalog_fields_from_dataset` arguments or elements.
            )
            cell_fields, cell_meta, cell_component_map = self._extract_catalog_fields_from_dataset(
                dataset,
                variable_catalog,
                # Completes the multi-line `self._extract_catalog_fields_from_dataset` arguments or elements.
                "cell",
                raw_field_names,
            )
        else:
            # Checks `direct_field_lookup and raw_field_names is not None` before entering this conditional branch.
            if direct_field_lookup and raw_field_names is not None:
                point_fields_raw, point_meta_raw = VTKIOReader.extract_named_point_fields_with_meta(
                    dataset,
                    raw_field_names,
                # Completes the multi-line `VTKIOReader.extract_named_point_fields_with_meta` arguments or elements.
                )
                cell_fields_raw, cell_meta_raw = VTKIOReader.extract_named_cell_fields_with_meta(
                    dataset,
                    raw_field_names,
                # Completes the multi-line `VTKIOReader.extract_named_cell_fields_with_meta` arguments or elements.
                )
            else:
                point_fields_raw, point_meta_raw = VTKIOReader.extract_point_fields_with_meta(
                    dataset,
                    # Completes the multi-line `VTKIOReader.extract_point_fields_with_meta` arguments or elements.
                    raw_field_names,
                )
                cell_fields_raw, cell_meta_raw = VTKIOReader.extract_cell_fields_with_meta(
                    dataset,
                    # Completes the multi-line `VTKIOReader.extract_cell_fields_with_meta` arguments or elements.
                    raw_field_names,
                )
            point_fields, point_meta, point_component_map = self._split_multicomponent_fields(
                point_fields_raw,
                # Completes the multi-line `self._split_multicomponent_fields` arguments or elements.
                point_meta_raw,
                variable_catalog=variable_catalog,
                location="point",
            )
            # Stores the `(cell_fields, cell_meta, cell_component_map)` for use by the remaining steps in this flow.
            cell_fields, cell_meta, cell_component_map = self._split_multicomponent_fields(
                cell_fields_raw,
                cell_meta_raw,
                variable_catalog=variable_catalog,
                # Completes the multi-line `self._split_multicomponent_fields` arguments or elements.
                location="cell",
            )
        if all_field_location_map is None:
            all_field_location_map = self._build_field_location_map_from_dataset(dataset)
        # Continues the surrounding statement or collection definition.
        else:
            all_field_location_map = dict(all_field_location_map)
        field_location_map = dict(all_field_location_map)
        projected_point_or_cell = None
        # Checks `not point_fields and not cell_fields` before entering this conditional branch.
        if not point_fields and not cell_fields:
            if not allow_missing_fields:
                raise ValueError("No readable PointData/CellData fields found in VTK dataset")
            fields = {}
            # Initializes the `field_attributes` as a collection for values from this processing stage.
            field_attributes = {}
            variable_catalog = (
                variable_catalog
                or self.config.get('task', {}).get('variable_catalog', {})
                # Completes the multi-line `collection` arguments or elements.
                or {}
            )
            field_component_map = self.config.get('task', {}).get('field_component_map', {})
            field_location_map = self.config.get('task', {}).get('field_location_map', {})
            # Stores the `point_or_cell` on the instance so other methods can reuse it.
            self.point_or_cell = self._resolve_fieldless_point_or_cell()
            projected_point_or_cell = self.config.get('task', {}).get(
                'projected_point_or_cell',
                self.point_or_cell,
            # Completes the multi-line `self.config.get('task', {}).get` arguments or elements.
            )
            effective_variables = list(fields.keys()) if read_all_fields else list(
                self.config.get('task', {}).get('effective_variables', []) or logical_variables
            )
            # Builds the `effective_variable_sources` using the comprehension's filter and mapping rules.
            effective_variable_sources = {
                str(name): dict(source)
                for name, source in (
                    self.config.get('task', {}).get('effective_variable_sources', {}) or {}
                # Completes the multi-line `(self.config.get('task', {}).get('effective_variable_sources', {}) or {}).items` arguments or elements.
                ).items()
                if str(name) in effective_variables
            }
        else:
            # Stores the `location` for use by the remaining steps in this flow.
            location = VTKIOReader.detect_field_location(point_fields, cell_fields)

            if location == 'point':
                fields = point_fields
                field_attributes = point_meta
                # Stores the `field_component_map` for use by the remaining steps in this flow.
                field_component_map = point_component_map
                self.point_or_cell = 'point'
                projected_point_or_cell = 'point'
            elif location == 'cell':
                # Stores the `fields` for use by the remaining steps in this flow.
                fields = cell_fields
                field_attributes = cell_meta
                field_component_map = cell_component_map
                self.point_or_cell = 'cell'
                # Stores the `projected_point_or_cell` for use by the remaining steps in this flow.
                projected_point_or_cell = 'cell'
            elif location == 'mixed':
                (
                    fields,
                    # Completes the multi-line `collection` arguments or elements.
                    field_attributes,
                    field_component_map,
                    field_location_map,
                    projected_point_or_cell,
                # Completes the multi-line `collection` arguments or elements.
                ) = self._resolve_mixed_fields(
                    point_fields=point_fields,
                    cell_fields=cell_fields,
                    point_meta=point_meta,
                    # Completes the multi-line `self._resolve_mixed_fields` arguments or elements.
                    cell_meta=cell_meta,
                    point_component_map=point_component_map,
                    cell_component_map=cell_component_map,
                    connectivity=connectivity,
                    # Completes the multi-line `self._resolve_mixed_fields` arguments or elements.
                    num_points=num_points,
                    requested_vars=logical_variables,
                )
            else:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError("No readable PointData/CellData fields found in VTK dataset")

            if read_all_fields:
                field_component_map = catalog_component_map(variable_catalog)
                effective_variable_sources = catalog_sources_by_runtime_key(variable_catalog)
                # Builds the `effective_variables` using the comprehension's filter and mapping rules.
                effective_variables = [
                    runtime_key
                    for runtime_key in effective_variable_sources
                    if runtime_key in fields
                # Continues the surrounding statement or collection definition.
                ]
                effective_variable_sources = {
                    runtime_key: effective_variable_sources[runtime_key]
                    for runtime_key in effective_variables
                # Continues the surrounding statement or collection definition.
                }
            else:
                try:
                    (
                        # Completes the multi-line `collection` arguments or elements.
                        field_component_map,
                        effective_variables,
                        effective_variable_sources,
                        _,
                        # Completes the multi-line `collection` arguments or elements.
                        _,
                    ) = select_catalog_variables(
                        variable_catalog,
                        logical_variables,
                        # Completes the multi-line `select_catalog_variables` arguments or elements.
                        strict_missing=not allow_missing_fields,
                    )
                except ValueError:
                    if not allow_missing_fields:
                        # Raises `the active exception` to report why this path cannot continue.
                        raise
                    # 预测输入可仅包含几何与局部 BC，目标物理场由后续处理器构造虚拟占位。
                    field_component_map = dict(
                        self.config.get('task', {}).get('field_component_map', {}) or {}
                    )
                    # Stores the `effective_variables` for use by the remaining steps in this flow.
                    effective_variables = list(
                        self.config.get('task', {}).get('effective_variables', [])
                        or logical_variables
                    )
                    # Builds the `effective_variable_sources` using the comprehension's filter and mapping rules.
                    effective_variable_sources = {
                        str(name): dict(source)
                        for name, source in (
                            self.config.get('task', {}).get(
                                # Completes the multi-line `collection` arguments or elements.
                                'effective_variable_sources', {}
                            )
                            or {}
                        ).items()
                        # Completes the multi-line `str` arguments or elements.
                        if str(name) in effective_variables
                    }
                missing_effective = [
                    runtime_key
                    # Continues the surrounding statement or collection definition.
                    for runtime_key in effective_variables
                    if runtime_key not in fields
                ]
                if missing_effective and not allow_missing_fields:
                    # Raises `ValueError` to report why this path cannot continue.
                    raise ValueError(
                        f"VTK 请求数组缺少训练分量: {missing_effective}"
                    )

                retained_field_names = list(effective_variables)
                # Stores the `local_feature_names` for use by the remaining steps in this flow.
                local_feature_names = self._resolve_local_feature_field_names()
                if local_feature_names:
                    _, local_effective, _, _, _ = select_catalog_variables(
                        variable_catalog,
                        # Completes the multi-line `select_catalog_variables` arguments or elements.
                        local_feature_names,
                        strict_missing=False,
                    )
                    for field_name in local_effective:
                        # Checks `field_name not in retained_field_names` before entering this conditional branch.
                        if field_name not in retained_field_names:
                            retained_field_names.append(field_name)
                fields, field_attributes = self._filter_fields(
                    fields,
                    # Completes the multi-line `self._filter_fields` arguments or elements.
                    field_attributes,
                    retained_field_names,
                )
            all_field_location_map = merge_field_location_maps(
                # Completes the multi-line `merge_field_location_maps` arguments or elements.
                all_field_location_map,
                field_location_map,
            )
            field_location_map = {
                # Continues the surrounding statement or collection definition.
                field_name: all_field_location_map[field_name]
                for field_name in fields
                if field_name in all_field_location_map
            }

        # Returns `` to the caller for the next stage.
        return (
            fields,
            field_attributes,
            field_component_map,
            # Completes the multi-line `collection` arguments or elements.
            field_location_map,
            projected_point_or_cell or self.point_or_cell,
            list(effective_variables),
            variable_catalog,
            # Completes the multi-line `collection` arguments or elements.
            effective_variable_sources,
        )

    def _build_base_data(
        self,
        # Continues the surrounding statement or collection definition.
        *,
        file_path: str,
        sample_order: Optional[int],
        logical_variables: List[str],
        # Continues the surrounding statement or collection definition.
        points: np.ndarray,
        cell_centers: np.ndarray,
        connectivity,
        mesh_dimension: Optional[int],
        # Completes the multi-line `collection` arguments or elements.
        dataset_meta: Dict[str, object],
        fields: Dict[str, np.ndarray],
        field_attributes: Dict[str, Dict[str, object]],
        field_component_map: Dict[str, List[str]],
        # Completes the multi-line `collection` arguments or elements.
        variable_catalog: Dict[str, Dict[str, object]],
        effective_variable_sources: Dict[str, Dict[str, object]],
        field_location_map: Dict[str, str],
        projected_point_or_cell: str,
        # Continues the surrounding statement or collection definition.
        effective_variables: List[str],
        topology_prevalidated: bool = False,
        temporal_coordinates_prevalidated: bool = False,
        point_coordinates_match_reference: Optional[bool] = None,
        # Continues the surrounding statement or collection definition.
        cell_coordinates_match_reference: Optional[bool] = None,
    ) -> BaseData:
        field_data = {
            'PointCoordinates': points,
            # Completes the multi-line `collection` arguments or elements.
            'CellCoordinates': cell_centers,
            'ElementConnectivity': np.asarray(connectivity, dtype=object),
            'Fields': fields,
            'LogicalVariables': list(logical_variables),
            # Completes the multi-line `list` arguments or elements.
            'EffectiveVariables': list(effective_variables),
            'FieldAttributes': field_attributes,
            'FieldComponentMap': field_component_map,
            'VariableCatalog': variable_catalog,
            # Completes the multi-line `collection` arguments or elements.
            'EffectiveVariableSources': effective_variable_sources,
            'FieldLocationMap': field_location_map,
            'ProjectedPointOrCell': projected_point_or_cell or self.point_or_cell,
            'MixedFieldPolicy': 'majority_side',
            # Completes the multi-line `collection` arguments or elements.
            'ProjectionMethod': 'mean',
            'BoundaryConditions': {},
            'MaterialIDs': None,
            'MeshDimension': mesh_dimension,
            # Completes the multi-line `collection` arguments or elements.
            'SourceVTKMeta': dataset_meta,
            'CanonicalVTKType': dataset_meta.get('canonical_dataset_type'),
            'OriginalVTKType': dataset_meta.get('source_dataset_type'),
        }
        # Checks `topology_prevalidated` before entering this conditional branch.
        if topology_prevalidated:
            field_data['TemporalTopologyPrevalidated'] = True
        if temporal_coordinates_prevalidated:
            field_data['TemporalCoordinatesPrevalidated'] = True
            # Converts the `field_data['TemporalPointCoordinatesMatchReference']` to `bool` so later checks use a consistent type.
            field_data['TemporalPointCoordinatesMatchReference'] = bool(
                point_coordinates_match_reference
            )
            field_data['TemporalCellCoordinatesMatchReference'] = bool(
                # Completes the multi-line `bool` arguments or elements.
                cell_coordinates_match_reference
            )

        sample_info = self._resolve_sample_info(file_path=file_path, sample_order=sample_order)
        return BaseData(
            # Completes the multi-line `BaseData` arguments or elements.
            data=field_data,
            task_type=self.task_type,
            mode=self.mode,
            point_or_cell=self.point_or_cell,
            # Completes the multi-line `BaseData` arguments or elements.
            sample_map=sample_info,
        )

    @staticmethod
    def _connectivity_signature_from_rows(
        # Continues the surrounding statement or collection definition.
        connectivity,
        *,
        num_points: int,
        cell_types: Optional[np.ndarray],
    # Completes the multi-line `collection` arguments or elements.
    ) -> Dict[str, np.ndarray]:
        offsets = [0]
        values: List[int] = []
        for cell in connectivity:
            # Stores the `row` for use by the remaining steps in this flow.
            row = np.asarray(cell, dtype=np.int64).reshape(-1)
            # VTKReader 对外保存 one-based 连接关系，签名统一为 VTK 原始 zero-based。
            values.extend((row - 1).tolist())
            offsets.append(len(values))
        return {
            # Completes the multi-line `np.asarray` arguments or elements.
            "num_points": np.asarray([int(num_points)], dtype=np.int64),
            "num_cells": np.asarray([len(offsets) - 1], dtype=np.int64),
            "offsets": np.asarray(offsets, dtype=np.int64),
            "connectivity": np.asarray(values, dtype=np.int64),
            # Completes the multi-line `np.asarray([] if cell_types is None else cell_types, dtype=np.int32).reshape` arguments or elements.
            "cell_types": np.asarray(
                [] if cell_types is None else cell_types,
                dtype=np.int32,
            ).reshape(-1),
        # Completes the multi-line `collection` arguments or elements.
        }

    def _read_temporal_step_fast(
        self,
        *,
        # Continues the surrounding statement or collection definition.
        file_path: str,
        sample_order: Optional[int],
        logical_variables: List[str],
        read_all_fields: bool,
        # Continues the surrounding statement or collection definition.
        allow_missing_fields: bool,
        template: Dict[str, object],
    ) -> BaseData:
        """读取后续 VTK 时间步，复用首帧连通关系与元数据。"""
        dataset = VTKIOReader.read_canonical_dataset_without_meta(
            # Completes the multi-line `VTKIOReader.read_canonical_dataset_without_meta` arguments or elements.
            file_path,
            geometry_policy=self.geometry_policy,
        )
        points = VTKIOReader.extract_points(dataset)
        # Stores the `actual_signature` for use by the remaining steps in this flow.
        actual_signature = VTKIOReader.extract_connectivity_signature(dataset)
        if not VTKIOReader.connectivity_signatures_equal(
            template["topology_signature"],
            actual_signature,
        # Completes the multi-line `VTKIOReader.connectivity_signatures_equal` arguments or elements.
        ):
            raise ValueError(f"时序步骤之间的单元连接关系发生变化: {file_path}")

        connectivity = template["connectivity"]
        reference_points = np.asarray(template["reference_points"])
        # Validates `points.shape != reference_points.shape` and raises `ValueError` when the input cannot be processed safely.
        if points.shape != reference_points.shape:
            raise ValueError(
                "时序步骤之间的点坐标形状发生变化: "
                f"{file_path}, expected={reference_points.shape}, actual={points.shape}"
            # Completes the multi-line `ValueError` arguments or elements.
            )
        points_match_reference = bool(
            np.array_equal(points, reference_points)
            or np.allclose(points, reference_points)
        # Completes the multi-line `bool` arguments or elements.
        )
        if points_match_reference:
            cell_centers = np.asarray(template["reference_cell_centers"])
        else:
            # Checks `not bool(template.get("cell_center_weights_resolved", False))` before entering this conditional branch.
            if not bool(template.get("cell_center_weights_resolved", False)):
                template["cell_center_weights"] = (
                    VTKIOReader.extract_cell_center_interpolation_weights(dataset)
                )
                # Stores the `template['cell_center_weights_resolved']` for use by the remaining steps in this flow.
                template["cell_center_weights_resolved"] = True
            cell_center_weights = template.get("cell_center_weights")
            if cell_center_weights is None:
                cell_centers = VTKIOReader.extract_cell_centers(dataset)
            # Continues the surrounding statement or collection definition.
            else:
                cell_centers = VTKIOReader.compute_cell_centers_from_connectivity(
                    points=points,
                    connectivity=connectivity,
                    # Completes the multi-line `VTKIOReader.compute_cell_centers_from_connectivity` arguments or elements.
                    one_based=True,
                    cell_center_weights=cell_center_weights,
                )
        cell_centers_match_reference = bool(
            # Completes the multi-line `np.array_equal` arguments or elements.
            np.array_equal(cell_centers, template["reference_cell_centers"])
            or np.allclose(cell_centers, template["reference_cell_centers"])
        )
        (
            # Completes the multi-line `collection` arguments or elements.
            fields,
            field_attributes,
            field_component_map,
            field_location_map,
            # Completes the multi-line `collection` arguments or elements.
            projected_point_or_cell,
            effective_variables,
            variable_catalog,
            effective_variable_sources,
        # Completes the multi-line `collection` arguments or elements.
        ) = self._extract_fields_from_dataset(
            dataset=dataset,
            logical_variables=logical_variables,
            read_all_fields=read_all_fields,
            # Completes the multi-line `self._extract_fields_from_dataset` arguments or elements.
            allow_missing_fields=allow_missing_fields,
            connectivity=connectivity,
            num_points=int(points.shape[0]),
            raw_field_names=template.get("raw_field_names"),
            # Completes the multi-line `template.get` arguments or elements.
            all_field_location_map=template.get("field_location_map"),
            direct_field_lookup=bool(template.get("direct_field_lookup", False)),
        )
        return self._build_base_data(
            # Completes the multi-line `self._build_base_data` arguments or elements.
            file_path=file_path,
            sample_order=sample_order,
            logical_variables=logical_variables,
            points=points,
            # Completes the multi-line `self._build_base_data` arguments or elements.
            cell_centers=cell_centers,
            connectivity=connectivity,
            mesh_dimension=template["mesh_dimension"],
            dataset_meta=template["dataset_meta"],
            # Completes the multi-line `self._build_base_data` arguments or elements.
            fields=fields,
            field_attributes=field_attributes,
            field_component_map=field_component_map,
            variable_catalog=variable_catalog,
            # Completes the multi-line `self._build_base_data` arguments or elements.
            effective_variable_sources=effective_variable_sources,
            field_location_map=field_location_map,
            projected_point_or_cell=projected_point_or_cell,
            effective_variables=effective_variables,
            # Completes the multi-line `self._build_base_data` arguments or elements.
            topology_prevalidated=True,
            temporal_coordinates_prevalidated=True,
            point_coordinates_match_reference=points_match_reference,
            cell_coordinates_match_reference=cell_centers_match_reference,
        # Completes the multi-line `self._build_base_data` arguments or elements.
        )

    def read_temporal_sequence(
        self,
        file_path: str,
        # Continues the surrounding statement or collection definition.
        variables: Optional[List[str]] = None,
        sample_order: Optional[int] = None,
        allow_missing_fields: bool = False,
        max_time_steps: Optional[int] = None,
    # Continues the surrounding statement or collection definition.
    ) -> BaseData:
        """以共享拓扑路径读取 VTK 时序目录或清单。"""
        logical_variables = self.variables if variables is None else variables
        read_all_fields = variables is not None and len(variables) == 0

        sequence_files = list_temporal_sequence_files(
            # Completes the multi-line `list_temporal_sequence_files` arguments or elements.
            self,
            file_path,
            max_time_steps=max_time_steps,
        )
        # Validates `not sequence_files` and raises `ValueError` when the input cannot be processed safely.
        if not sequence_files:
            raise ValueError(f"未从时序 VTK 源读取到任何时间步: {file_path}")

        first = self.read(
            sequence_files[0],
            # Completes the multi-line `self.read` arguments or elements.
            variables=variables,
            sample_order=sample_order,
            allow_missing_fields=allow_missing_fields,
            _capture_temporal_topology=True,
        # Completes the multi-line `self.read` arguments or elements.
        )
        first_data = first.get_data()
        connectivity = first_data["ElementConnectivity"]
        template = {
            # Completes the multi-line `collection` arguments or elements.
            "connectivity": connectivity,
            "topology_signature": self._connectivity_signature_from_rows(
                connectivity,
                num_points=int(np.asarray(first_data["PointCoordinates"]).shape[0]),
                # Completes the multi-line `first_data.pop` arguments or elements.
                cell_types=first_data.pop("_TemporalCellTypes", None),
            ),
            "mesh_dimension": first_data.get("MeshDimension"),
            "dataset_meta": dict(first_data.get("SourceVTKMeta") or {}),
            # Completes the multi-line `np.asarray` arguments or elements.
            "reference_points": np.asarray(first_data["PointCoordinates"]),
            "reference_cell_centers": np.asarray(first_data["CellCoordinates"]),
            "cell_center_weights": None,
            "cell_center_weights_resolved": False,
            # Completes the multi-line `first_data.pop` arguments or elements.
            "raw_field_names": first_data.pop("_TemporalRawFieldNames", []),
            "direct_field_lookup": bool(
                first_data.pop("_TemporalDirectFieldLookup", False)
            ),
            # Completes the multi-line `dict` arguments or elements.
            "field_location_map": dict(first_data.get("FieldLocationMap") or {}),
        }

        def data_steps():
            yield first
            # Visits each item in `sequence_files[1:]` and applies the current processing step.
            for step_file in sequence_files[1:]:
                yield self._read_temporal_step_fast(
                    file_path=step_file,
                    sample_order=sample_order,
                    # Completes the multi-line `self._read_temporal_step_fast` arguments or elements.
                    logical_variables=logical_variables,
                    read_all_fields=read_all_fields,
                    allow_missing_fields=allow_missing_fields,
                    template=template,
                # Completes the multi-line `self._read_temporal_step_fast` arguments or elements.
                )

        return aggregate_field_temporal_sequence(
            data_steps(),
            file_path,
            # Completes the multi-line `self._resolve_local_feature_field_names` arguments or elements.
            static_field_names=self._resolve_local_feature_field_names(),
            expected_time_steps=len(sequence_files),
        )

    def read(
        # Continues the surrounding statement or collection definition.
        self,
        file_path: str,
        variables: Optional[List[str]] = None,
        sample_order: Optional[int] = None,
        # Continues the surrounding statement or collection definition.
        allow_missing_fields: bool = False,
        max_time_steps: Optional[int] = None,
        _capture_temporal_topology: bool = False,
    ) -> BaseData:
        """读取VTK文件并转换为BaseData。"""
        # Stores the `logical_variables` for use by the remaining steps in this flow.
        logical_variables = self.variables if variables is None else variables
        read_all_fields = variables is not None and len(variables) == 0

        if self.mode == "temporal" and (
            Path(file_path).is_dir() or is_temporal_manifest_path(file_path)
        # Continues the surrounding statement or collection definition.
        ):
            return self.read_temporal_sequence(
                file_path=file_path,
                variables=variables,
                # Completes the multi-line `self.read_temporal_sequence` arguments or elements.
                sample_order=sample_order,
                allow_missing_fields=allow_missing_fields,
                max_time_steps=max_time_steps,
            )

        # Stores the `(dataset, dataset_meta)` for use by the remaining steps in this flow.
        dataset, dataset_meta = VTKIOReader.read_canonical_dataset_with_meta(
            file_path,
            geometry_policy=self.geometry_policy,
        )

        # Stores the `points` for use by the remaining steps in this flow.
        points = VTKIOReader.extract_points(dataset)
        connectivity = VTKIOReader.extract_connectivity(dataset, one_based=True)
        cell_centers = VTKIOReader.extract_cell_centers(dataset)
        mesh_dimension = self._infer_mesh_dimension(dataset, points)

        # Stores the `(fields, field_attributes, field_component_map, field_location_map, projected_point_or_cell, effective_variables, variable_catalog, effective_variable_sources)` for use by the remaining steps in this flow.
        (
            fields,
            field_attributes,
            field_component_map,
            # Completes the multi-line `collection` arguments or elements.
            field_location_map,
            projected_point_or_cell,
            effective_variables,
            variable_catalog,
            # Completes the multi-line `collection` arguments or elements.
            effective_variable_sources,
        ) = self._extract_fields_from_dataset(
            dataset=dataset,
            logical_variables=logical_variables,
            # Completes the multi-line `self._extract_fields_from_dataset` arguments or elements.
            read_all_fields=read_all_fields,
            allow_missing_fields=allow_missing_fields,
            connectivity=connectivity,
            num_points=int(points.shape[0]),
        # Completes the multi-line `self._extract_fields_from_dataset` arguments or elements.
        )
        base_data = self._build_base_data(
            file_path=file_path,
            sample_order=sample_order,
            # Completes the multi-line `self._build_base_data` arguments or elements.
            logical_variables=logical_variables,
            points=points,
            cell_centers=cell_centers,
            connectivity=connectivity,
            # Completes the multi-line `self._build_base_data` arguments or elements.
            mesh_dimension=mesh_dimension,
            dataset_meta=dataset_meta,
            fields=fields,
            field_attributes=field_attributes,
            # Completes the multi-line `self._build_base_data` arguments or elements.
            field_component_map=field_component_map,
            variable_catalog=variable_catalog,
            effective_variable_sources=effective_variable_sources,
            field_location_map=field_location_map,
            # Completes the multi-line `self._build_base_data` arguments or elements.
            projected_point_or_cell=projected_point_or_cell,
            effective_variables=effective_variables,
        )
        if _capture_temporal_topology:
            # Stores the `payload` for use by the remaining steps in this flow.
            payload = base_data.get_data()
            payload["_TemporalCellTypes"] = VTKIOReader._extract_cell_types_fast(dataset)
            raw_field_names = self._resolve_temporal_raw_field_names(
                dataset,
                # Completes the multi-line `self._resolve_temporal_raw_field_names` arguments or elements.
                logical_variables,
                read_all_fields,
            )
            payload["_TemporalRawFieldNames"] = raw_field_names
            # Stores the `payload['_TemporalDirectFieldLookup']` for use by the remaining steps in this flow.
            payload["_TemporalDirectFieldLookup"] = self._can_direct_lookup_raw_fields(
                raw_field_names
            )
        return base_data
