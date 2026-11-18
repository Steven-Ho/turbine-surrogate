import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
# Imports `h5py` for reuse in this module.
import h5py
import numpy as np

from ..data.temporal_utils import (
    aggregate_field_temporal_sequence,
    # Continues the surrounding statement or collection definition.
    infer_displacement_effective_variables,
    infer_displacement_logical_name,
    is_temporal_manifest_path,
    list_temporal_sequence_files,
    # Continues the surrounding statement or collection definition.
    read_single_sample,
)
from ..data.base_data import BaseData
from ..io_utils import cgnsmesh
# Imports `cgnskeywords` from `io_utils` for use in this module.
from ..io_utils import cgnskeywords
from ..io_utils.io_cgns import CGNSReader as IOCGNSReader
from ..utils.field_projection import (
    build_component_location_map,
    # Continues the surrounding statement or collection definition.
    choose_projected_point_or_cell,
    project_fields_to_location,
)
from ..utils.path_utils import path_endswith
# Imports `Logger` from `utils.logger` for use in this module.
from ..utils.logger import Logger
from ..utils.field_variable_catalog import (
    build_field_variable_catalog,
    catalog_component_map,
    # Continues the surrounding statement or collection definition.
    catalog_sources_by_runtime_key,
    select_catalog_variables,
)
from ..utils.variable_resolver import (
    # Continues the surrounding statement or collection definition.
    resolve_effective_variables,
)
from .base_reader import BaseReader
from .sample_map import SampleMap


# Defines `_get_logger` for  get logger logic.
def _get_logger() -> logging.Logger:
    return Logger.get_first_configured_logger(
        ["train", "predict", "evaluate"],
        fallback_name=__name__,
    # Completes the multi-line `Logger.get_first_configured_logger` arguments or elements.
    )


class CGNSReader(BaseReader):
    """CGNS文件读取器，基于io_utils.io_cgns模块"""
    SUPPORTED_EXTENSIONS = (".cgns",)
    
    def __init__(self, config: Dict):
        """初始化CGNS读取器
        
        Args:
            config: 读取器配置参数
        """
        # Calls `super().__init__` to perform the operation required at this stage.
        super().__init__(config)
        self.task_type = config['task']['type']
        self.mode = config['task']['mode']
        self.variables = config['task']['variables']
        # Stores the `point_or_cell` on the instance so other methods can reuse it.
        self.point_or_cell = None
        self.supported_extensions = self.SUPPORTED_EXTENSIONS
        self._family_bc_types: Dict[str, str] = {}
        # 初始化SampleMap
        sample_table_path = config['general']['train_samples']
        # Stores the `sample_map` on the instance so other methods can reuse it.
        self.sample_map = SampleMap(sample_table_path) if sample_table_path else None

    @staticmethod
    def _build_cgns_variable_catalog(
        field_location_map: Dict[str, str],
    # Completes the multi-line `collection` arguments or elements.
    ) -> Dict[str, Dict[str, object]]:
        """将 CGNS 的每个原始字段作为独立可选变量。"""
        entries = [
            {
                "format": "cgns",
                # Completes the multi-line `collection` arguments or elements.
                "location": location,
                "array_name": field_name,
                "num_components": 1,
            }
            # Completes the multi-line `collection` arguments or elements.
            for field_name, location in field_location_map.items()
        ]
        return build_field_variable_catalog(entries)

    @staticmethod
    # 按字段原始位置拆分 catalog 映射，供混合场投影决策使用。
    def _catalog_maps_by_location(
        catalog: Dict[str, Dict[str, object]],
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """按字段原始位置拆分 catalog 映射，供混合场投影决策使用。"""
        point_map: Dict[str, List[str]] = {}
        # Initializes the `cell_map` as a collection for values from this processing stage.
        cell_map: Dict[str, List[str]] = {}
        for selection_key, entry in catalog.items():
            target = point_map if entry.get("location") == "point" else cell_map
            target[selection_key] = list(entry.get("component_keys") or [])
        # Returns `point_map, cell_map` to the caller for the next stage.
        return point_map, cell_map

    def _requires_cgns_boundary_conditions(self) -> bool:
        """仅在启用 cgns_based 时将边界条件解析错误视为读取失败。"""
        task_config = self.config.get("task", {})
        return task_config.get("local_feature_encoding") == "cgns_based"

    # Defines `_get_single_base` for  get single base logic.
    def _get_single_base(self, cgns_data, file_path: str):
        base_nodes = cgns_data.root.get_node_by_label("CGNSBase_t")
        if not base_nodes:
            raise ValueError(f"CGNS文件未找到任何CGNSBase_t节点: {file_path}")

        # Checks `len(base_nodes) != 1` before entering this conditional branch.
        if len(base_nodes) != 1:
            all_base_names = [b.name for b in base_nodes]
            raise ValueError(
                f"CGNS文件包含{len(base_nodes)}个Base，当前实现要求单Base: {file_path}. "
                # Completes the multi-line `ValueError` arguments or elements.
                f"可用Base: {all_base_names}"
            )

        return base_nodes[0]

    def _get_zone_nodes(self, cgns_data, file_path: str):
        # Stores the `base` for use by the remaining steps in this flow.
        base = self._get_single_base(cgns_data, file_path)
        zone_nodes = base.get_node_by_label("Zone_t")
        if not zone_nodes:
            raise ValueError(f"单Base下未找到任何Zone_t节点: {file_path}")
        # Returns `zone_nodes` to the caller for the next stage.
        return zone_nodes

    @staticmethod
    def _decode_cgns_text(value) -> Optional[str]:
        if value is None:
            # Returns `None` to the caller for the next stage.
            return None
        values = np.asarray(value).reshape(-1)
        if values.size == 0:
            return None
        # Selects the required item from `values` as the `raw_value`.
        raw_value = values[0]
        if isinstance(raw_value, bytes):
            return raw_value.decode(errors="replace").rstrip("\x00")
        return str(raw_value).rstrip("\x00")

    # 提取 Base 级 FamilyBC_t，为 FamilySpecified 边界条件提供类型。
    def _extract_family_bc_types(self, cgns_data, file_path: str) -> Dict[str, str]:
        """提取 Base 级 FamilyBC_t，为 FamilySpecified 边界条件提供类型。"""
        base = self._get_single_base(cgns_data, file_path)
        family_types: Dict[str, str] = {}
        # Family_t 与 FamilyBC_t 都是可选的直属节点；缺失时不应产生告警。
        family_nodes = [
            # Continues the surrounding statement or collection definition.
            node
            for node in (base.children or {}).values()
            if node.label == "Family_t"
        ]
        # Visits each item in `family_nodes` and applies the current processing step.
        for family in family_nodes:
            family_bc_nodes = [
                node
                for node in (family.children or {}).values()
                # Continues the surrounding statement or collection definition.
                if node.label == "FamilyBC_t"
            ]
            if not family_bc_nodes:
                continue
            # Stores the `family_bc_type` for use by the remaining steps in this flow.
            family_bc_type = self._decode_cgns_text(family_bc_nodes[0].data)
            if family_bc_type:
                family_types[family.name] = family_bc_type
        return family_types

    # Offset 1-based connectivity indices by point_offset.
    def _offset_connectivity(self, conn, point_offset: int):
        """Offset 1-based connectivity indices by point_offset."""
        if isinstance(conn, np.ndarray) and conn.dtype != object and conn.ndim == 2:
            return conn + point_offset

        shifted = []
        # Visits each item in `conn` and applies the current processing step.
        for cell in conn:
            cell_arr = np.asarray(cell, dtype=np.int64).reshape(-1)
            shifted.append(cell_arr + point_offset)

        return np.asarray(shifted, dtype=object)

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _expand_boundary_entity_ids(boundary_condition: Dict) -> np.ndarray:
        """将 CGNS PointList 或 PointRange 展开为原始实体编号。"""
        point_list = boundary_condition.get("point_list")
        if point_list is not None:
            # Returns `np.unique(np.asarray(point_list, dtype=np.int64).reshape(-1))` to the caller for the next stage.
            return np.unique(np.asarray(point_list, dtype=np.int64).reshape(-1))

        point_range = boundary_condition.get("point_range")
        if point_range is None:
            return np.empty((0,), dtype=np.int64)
        # Stores the `range_values` for use by the remaining steps in this flow.
        range_values = np.asarray(point_range, dtype=np.int64).reshape(-1)
        if range_values.size < 2:
            return np.empty((0,), dtype=np.int64)
        start_id = int(range_values.min())
        # Converts the `end_id` to `int` so later checks use a consistent type.
        end_id = int(range_values.max())
        return np.arange(start_id, end_id + 1, dtype=np.int64)

    @staticmethod
    def _normalize_structured_point_range(
        # Continues the surrounding statement or collection definition.
        point_range,
        index_dimension: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """将结构网格 PointRange 规范化为起止逻辑索引。"""
        values = np.asarray(point_range, dtype=np.int64)
        # Checks `values.shape == (index_dimension, 2)` before entering this conditional branch.
        if values.shape == (index_dimension, 2):
            lower, upper = values[:, 0], values[:, 1]
        elif values.shape == (2, index_dimension):
            lower, upper = values[0, :], values[1, :]
        # Checks `values.size == index_dimension * 2` before entering this conditional branch.
        elif values.size == index_dimension * 2:
            normalized = values.reshape(index_dimension, 2)
            lower, upper = normalized[:, 0], normalized[:, 1]
        else:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(
                f"{index_dimension} 维结构网格 PointRange 必须包含 "
                f"{index_dimension} 组起止索引"
            )
        # Validates `np.any(lower < 1) or np.any(upper < lower)` and raises `ValueError` when the input cannot be processed safely.
        if np.any(lower < 1) or np.any(upper < lower):
            raise ValueError("结构网格 PointRange 包含非法索引")
        return lower.astype(np.int64), upper.astype(np.int64)

    @staticmethod
    # 将结构网格 PointList 规范化为 [N, IndexDimension] 逻辑索引。
    def _normalize_structured_point_list(
        point_list,
        index_dimension: int,
    ) -> np.ndarray:
        """将结构网格 PointList 规范化为 [N, IndexDimension] 逻辑索引。"""
        # Stores the `values` for use by the remaining steps in this flow.
        values = np.asarray(point_list, dtype=np.int64)
        if values.ndim != 2:
            raise ValueError("结构网格 PointList 必须是二维逻辑索引矩阵")

        row_major = values.shape[0] == index_dimension
        # Stores the `column_major` for use by the remaining steps in this flow.
        column_major = values.shape[1] == index_dimension
        if row_major and column_major:
            raise ValueError(
                "结构网格 PointList 的行列方向无法判定；"
                # Completes the multi-line `ValueError` arguments or elements.
                "请使用非方阵 IndexDimension x N 或 N x IndexDimension，或改用 PointRange"
            )
        if row_major:
            logical_indices = values.transpose()
        # Checks `column_major` before entering this conditional branch.
        elif column_major:
            logical_indices = values
        else:
            raise ValueError(
                # Completes the multi-line `ValueError` arguments or elements.
                f"结构网格 PointList 必须包含 {index_dimension} 个逻辑索引分量"
            )
        if logical_indices.shape[0] == 0 or np.any(logical_indices < 1):
            raise ValueError("结构网格 PointList 包含非法索引")
        # Returns `np.asarray(logical_indices, dtype=np.int64)` to the caller for the next stage.
        return np.asarray(logical_indices, dtype=np.int64)

    @classmethod
    def _resolve_structured_vertex_indices(cls, zone, boundary_condition: Dict) -> np.ndarray:
        """将结构网格 Vertex PointRange/PointList 展开为零基节点索引。"""
        point_list = boundary_condition.get("point_list")
        # Reads `point_range` from the mapping into the `point_range`.
        point_range = boundary_condition.get("point_range")
        if point_list is None and point_range is None:
            return np.empty((0,), dtype=np.int64)
        vertex_dims = zone.get_structured_vertex_dimensions()
        # Checks `point_list is not None` before entering this conditional branch.
        if point_list is not None:
            logical_indices = cls._normalize_structured_point_list(
                point_list,
                index_dimension=len(vertex_dims),
            # Completes the multi-line `cls._normalize_structured_point_list` arguments or elements.
            )
            if np.any(logical_indices > vertex_dims):
                raise ValueError("结构网格 Vertex PointList 超出 Zone 顶点范围")
        else:
            # Stores the `(lower, upper)` for use by the remaining steps in this flow.
            lower, upper = cls._normalize_structured_point_range(
                point_range,
                index_dimension=len(vertex_dims),
            )
            # Validates `np.any(upper > vertex_dims)` and raises `ValueError` when the input cannot be processed safely.
            if np.any(upper > vertex_dims):
                raise ValueError("结构网格 Vertex PointRange 超出 Zone 顶点范围")
            logical_indices = np.asarray(
                [
                    # Completes the multi-line `collection` arguments or elements.
                    [i, j]
                    for j in range(int(lower[1]), int(upper[1]) + 1)
                    for i in range(int(lower[0]), int(upper[0]) + 1)
                ]
                # Completes the multi-line `len` arguments or elements.
                if len(vertex_dims) == 2
                else [
                    [i, j, k]
                    for k in range(int(lower[2]), int(upper[2]) + 1)
                    # Completes the multi-line `range` arguments or elements.
                    for j in range(int(lower[1]), int(upper[1]) + 1)
                    for i in range(int(lower[0]), int(upper[0]) + 1)
                ],
                dtype=np.int64,
            # Completes the multi-line `np.asarray` arguments or elements.
            )

        ni, nj = (int(value) for value in vertex_dims[:2])
        if len(vertex_dims) == 2:
            return np.asarray(
                # Completes the multi-line `np.asarray` arguments or elements.
                [
                    (int(index[0]) - 1) + ni * (int(index[1]) - 1)
                    for index in logical_indices
                ],
                # Completes the multi-line `np.asarray` arguments or elements.
                dtype=np.int64,
            )

        return np.asarray(
            [
                # Completes the multi-line `int` arguments or elements.
                (int(index[0]) - 1)
                + ni * ((int(index[1]) - 1) + nj * (int(index[2]) - 1))
                for index in logical_indices
            ],
            # Completes the multi-line `np.asarray` arguments or elements.
            dtype=np.int64,
        )

    @classmethod
    def _resolve_structured_surface_entity_node_indices(
        # Continues the surrounding statement or collection definition.
        cls,
        zone,
        boundary_condition: Dict,
    ) -> List[np.ndarray]:
        """将结构网格 I/J/K 面或边 PointRange/PointList 映射为实体节点。"""
        # Stores the `grid_location` for use by the remaining steps in this flow.
        grid_location = str(boundary_condition.get("grid_loc") or "").strip().lower()
        point_list = boundary_condition.get("point_list")
        point_range = boundary_condition.get("point_range")
        if point_list is None and point_range is None:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(f"{grid_location} 缺少 PointRange 或 PointList")

        vertex_dims = zone.get_structured_vertex_dimensions()
        if len(vertex_dims) == 2:
            axis_by_location = {"iedgecenter": 0, "jedgecenter": 1}
        # Continues the surrounding statement or collection definition.
        else:
            axis_by_location = {
                "ifacecenter": 0,
                "jfacecenter": 1,
                # Completes the multi-line `collection` arguments or elements.
                "kfacecenter": 2,
            }
        if grid_location not in axis_by_location:
            raise ValueError(f"不支持的结构网格实体位置: {grid_location}")

        # Stores the `face_dims` for use by the remaining steps in this flow.
        face_dims = vertex_dims.copy()
        for axis in range(len(vertex_dims)):
            if axis != axis_by_location[grid_location]:
                face_dims[axis] -= 1
        # Checks `point_list is not None` before entering this conditional branch.
        if point_list is not None:
            logical_indices = cls._normalize_structured_point_list(
                point_list,
                index_dimension=len(vertex_dims),
            # Completes the multi-line `cls._normalize_structured_point_list` arguments or elements.
            )
            if np.any(logical_indices > face_dims):
                raise ValueError(f"{grid_location} PointList 超出 Zone 实体范围")
        else:
            # Stores the `(lower, upper)` for use by the remaining steps in this flow.
            lower, upper = cls._normalize_structured_point_range(
                point_range,
                index_dimension=len(vertex_dims),
            )
            # Validates `np.any(upper > face_dims)` and raises `ValueError` when the input cannot be processed safely.
            if np.any(upper > face_dims):
                raise ValueError(f"{grid_location} PointRange 超出 Zone 实体范围")
            logical_indices = np.asarray(
                [
                    # Completes the multi-line `collection` arguments or elements.
                    [i, j]
                    for j in range(int(lower[1]), int(upper[1]) + 1)
                    for i in range(int(lower[0]), int(upper[0]) + 1)
                ]
                # Completes the multi-line `len` arguments or elements.
                if len(vertex_dims) == 2
                else [
                    [i, j, k]
                    for k in range(int(lower[2]), int(upper[2]) + 1)
                    # Completes the multi-line `range` arguments or elements.
                    for j in range(int(lower[1]), int(upper[1]) + 1)
                    for i in range(int(lower[0]), int(upper[0]) + 1)
                ],
                dtype=np.int64,
            # Completes the multi-line `np.asarray` arguments or elements.
            )

        ni, nj = (int(value) for value in vertex_dims[:2])
        if len(vertex_dims) == 2:
            def point_id(i: int, j: int) -> int:
                # Returns `i + ni * j` to the caller for the next stage.
                return i + ni * j

            node_groups = []
            for index in logical_indices:
                i, j = (int(value) - 1 for value in index)
                # Stores the `nodes` for use by the remaining steps in this flow.
                nodes = (
                    [point_id(i, j), point_id(i, j + 1)]
                    if grid_location == "iedgecenter"
                    else [point_id(i, j), point_id(i + 1, j)]
                # Continues the surrounding statement or collection definition.
                )
                node_groups.append(np.asarray(nodes, dtype=np.int64))
            return node_groups

        def point_id(i: int, j: int, k: int) -> int:
            # Returns `i + ni * (j + nj * k)` to the caller for the next stage.
            return i + ni * (j + nj * k)

        node_groups = []
        for index in logical_indices:
            i, j, k = (int(value) - 1 for value in index)
            # Checks `grid_location == "ifacecenter"` before entering this conditional branch.
            if grid_location == "ifacecenter":
                nodes = [
                    point_id(i, j, k), point_id(i, j + 1, k),
                    point_id(i, j + 1, k + 1), point_id(i, j, k + 1),
                # Completes the multi-line `collection` arguments or elements.
                ]
            elif grid_location == "jfacecenter":
                nodes = [
                    point_id(i, j, k), point_id(i + 1, j, k),
                    # Completes the multi-line `point_id` arguments or elements.
                    point_id(i + 1, j, k + 1), point_id(i, j, k + 1),
                ]
            else:
                nodes = [
                    # Completes the multi-line `point_id` arguments or elements.
                    point_id(i, j, k), point_id(i + 1, j, k),
                    point_id(i + 1, j + 1, k), point_id(i, j + 1, k),
                ]
            node_groups.append(np.asarray(nodes, dtype=np.int64))
        # Returns `node_groups` to the caller for the next stage.
        return node_groups

    def _get_standard_element_entries(self, zone):
        """返回标准 fixed/MIXED Elements_t 中的全局元素编号、节点和拓扑维度。"""
        sections = [
            child
            # Completes the multi-line `(getattr(zone, 'children', {}) or {}).values` arguments or elements.
            for child in (getattr(zone, "children", {}) or {}).values()
            if getattr(child, "label", None) == "Elements_t"
        ]
        parsed_sections = []
        # Visits each item in `sections` and applies the current processing step.
        for section in sections:
            try:
                range_node = section.children.get("ElementRange")
                range_values = np.asarray(range_node.data, dtype=np.int64).reshape(-1)
                # Checks `range_values.size < 2` before entering this conditional branch.
                if range_values.size < 2:
                    continue
                element_type = int(np.asarray(section.data).reshape(-1)[0])
                if element_type == cgnskeywords.ElementType["MIXED"]:
                    # Stores the `(rows, type_ids)` for use by the remaining steps in this flow.
                    rows, type_ids = zone._parse_mixed_section_connectivity(section)
                else:
                    rows, type_ids = zone._parse_fixed_section_connectivity(section)
                parsed_sections.append((int(range_values[0]), rows, type_ids))
            # Completes the multi-line `collection` arguments or elements.
            except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                continue

        entries = []
        for start_id, rows, type_ids in sorted(parsed_sections, key=lambda item: item[0]):
            # Visits each item in `enumerate(zip(rows, type_ids))` and applies the current processing step.
            for offset, (nodes, type_id) in enumerate(zip(rows, type_ids)):
                try:
                    dimension = zone._get_element_dimension(int(type_id))
                except (KeyError, TypeError, ValueError):
                    # Handles the data or control-flow case represented by this statement.
                    continue
                entries.append((start_id + offset, np.asarray(nodes, dtype=np.int64), dimension))
        return entries

    def _resolve_standard_boundary_entity_nodes(
        # Continues the surrounding statement or collection definition.
        self,
        zone,
        boundary_condition: Dict,
        entity_dimension: int,
    # Continues the surrounding statement or collection definition.
    ) -> List[np.ndarray]:
        requested_ids = set(self._expand_boundary_entity_ids(boundary_condition).tolist())
        if not requested_ids:
            return []
        # Builds the `nodes_by_id` using the comprehension's filter and mapping rules.
        nodes_by_id = {
            entity_id: nodes
            for entity_id, nodes, dimension in self._get_standard_element_entries(zone)
            if dimension == entity_dimension and entity_id in requested_ids
        # Continues the surrounding statement or collection definition.
        }
        return [
            np.asarray(nodes_by_id[entity_id], dtype=np.int64).reshape(-1) - 1
            for entity_id in sorted(requested_ids)
            # Continues the surrounding statement or collection definition.
            if entity_id in nodes_by_id
        ]

    def _resolve_cell_center_indices(self, zone, boundary_condition: Dict) -> np.ndarray:
        requested_ids = set(self._expand_boundary_entity_ids(boundary_condition).tolist())
        # Returns early when `not requested_ids` makes the remaining work inapplicable.
        if not requested_ids:
            return np.empty((0,), dtype=np.int64)

        point_list = boundary_condition.get("point_list")
        point_range = boundary_condition.get("point_range")
        # Checks `` before entering this conditional branch.
        if point_list is not None or (
            point_range is not None and np.asarray(point_range).size >= 4
        ):
            try:
                # Stores the `vertex_dims` for use by the remaining steps in this flow.
                vertex_dims = zone.get_structured_vertex_dimensions()
            except (AttributeError, ValueError):
                pass
            else:
                # Derives the `cell_dims` from the current values for downstream processing.
                cell_dims = np.asarray(vertex_dims, dtype=np.int64) - 1
                if point_list is not None:
                    logical_indices = self._normalize_structured_point_list(
                        point_list,
                        # Completes the multi-line `len` arguments or elements.
                        index_dimension=len(cell_dims),
                    )
                    if np.any(logical_indices > cell_dims):
                        raise ValueError("结构网格 CellCenter PointList 超出 Zone 单元范围")
                # Continues the surrounding statement or collection definition.
                else:
                    lower, upper = self._normalize_structured_point_range(
                        point_range,
                        index_dimension=len(cell_dims),
                    # Completes the multi-line `self._normalize_structured_point_range` arguments or elements.
                    )
                    if np.any(upper > cell_dims):
                        raise ValueError("结构网格 CellCenter PointRange 超出 Zone 单元范围")
                    logical_indices = np.asarray(
                        # Completes the multi-line `np.asarray` arguments or elements.
                        [
                            [i, j]
                            for j in range(int(lower[1]), int(upper[1]) + 1)
                            for i in range(int(lower[0]), int(upper[0]) + 1)
                        # Completes the multi-line `np.asarray` arguments or elements.
                        ]
                        if len(cell_dims) == 2
                        else [
                            [i, j, k]
                            # Completes the multi-line `range` arguments or elements.
                            for k in range(int(lower[2]), int(upper[2]) + 1)
                            for j in range(int(lower[1]), int(upper[1]) + 1)
                            for i in range(int(lower[0]), int(upper[0]) + 1)
                        ],
                        # Completes the multi-line `np.asarray` arguments or elements.
                        dtype=np.int64,
                    )

                ni, nj = (int(value) for value in cell_dims[:2])
                if len(cell_dims) == 2:
                    # Returns `` to the caller for the next stage.
                    return np.asarray(
                        [
                            (int(index[0]) - 1) + ni * (int(index[1]) - 1)
                            for index in logical_indices
                        # Completes the multi-line `np.asarray` arguments or elements.
                        ],
                        dtype=np.int64,
                    )

                return np.asarray(
                    # Completes the multi-line `np.asarray` arguments or elements.
                    [
                        (int(index[0]) - 1)
                        + ni * ((int(index[1]) - 1) + nj * (int(index[2]) - 1))
                        for index in logical_indices
                    # Completes the multi-line `np.asarray` arguments or elements.
                    ],
                    dtype=np.int64,
                )

        poly_cell_indices = []
        # Stores the `poly_cell_offset` for use by the remaining steps in this flow.
        poly_cell_offset = 0
        poly_sections = []
        for section in (getattr(zone, "children", {}) or {}).values():
            if getattr(section, "label", None) != "Elements_t":
                # Handles the data or control-flow case represented by this statement.
                continue
            try:
                element_type = int(np.asarray(section.data).reshape(-1)[0])
                if element_type != cgnskeywords.ElementType["NFACE_n"]:
                    # Handles the data or control-flow case represented by this statement.
                    continue
                element_range = np.asarray(
                    section.children["ElementRange"].data,
                    dtype=np.int64,
                # Completes the multi-line `np.asarray(section.children['ElementRange'].data, dtype=np.int64).reshape` arguments or elements.
                ).reshape(-1)
                poly_sections.append((int(element_range[0]), section))
            except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                continue

        # Visits each item in `sorted(poly_sections, key=lambda item: item[0])` and applies the current processing step.
        for start_id, section in sorted(poly_sections, key=lambda item: item[0]):
            slices = zone._parse_section_slices(section, section_kind="NFACE")
            for local_index in range(len(slices)):
                if start_id + local_index in requested_ids:
                    # Adds the current value to `poly_cell_indices` for later aggregation.
                    poly_cell_indices.append(poly_cell_offset + local_index)
            poly_cell_offset += len(slices)
        if poly_cell_indices:
            return np.asarray(poly_cell_indices, dtype=np.int64)

        # Stores the `entries` for use by the remaining steps in this flow.
        entries = self._get_standard_element_entries(zone)
        if not entries:
            return np.empty((0,), dtype=np.int64)
        mesh_dimension = max(dimension for _, _, dimension in entries)
        # Builds the `volume_ids` using the comprehension's filter and mapping rules.
        volume_ids = [
            entity_id
            for entity_id, _, dimension in entries
            if dimension == mesh_dimension
        # Continues the surrounding statement or collection definition.
        ]
        return np.asarray(
            [index for index, entity_id in enumerate(volume_ids) if entity_id in requested_ids],
            dtype=np.int64,
        # Completes the multi-line `np.asarray` arguments or elements.
        )

    def _resolve_face_center_node_indices(self, zone, boundary_condition: Dict) -> List[np.ndarray]:
        """将 FaceCenter 边界面编号解析为 Zone 内的零基节点编号分组。"""
        face_ids = self._expand_boundary_entity_ids(boundary_condition)
        if face_ids.size == 0:
            # Returns `[]` to the caller for the next stage.
            return []

        try:
            face_payload = zone.get_mesh_face2node()
            face_ranges = face_payload.get("face_ranges") or []
            # Stores the `offsets` for use by the remaining steps in this flow.
            offsets = np.asarray(face_payload.get("offset"), dtype=np.int64)
            connectivity = np.asarray(face_payload.get("connectivity"), dtype=np.int64)
            node_groups = []
            for face_id in face_ids:
                # Stores the `local_index` for use by the remaining steps in this flow.
                local_index = zone._face_id_to_local_idx(int(face_id), face_ranges)
                start, count = offsets[local_index]
                node_groups.append(
                    np.asarray(
                        # Completes the multi-line `int` arguments or elements.
                        connectivity[int(start): int(start + count)],
                        dtype=np.int64,
                    ).reshape(-1)
                    - 1
                # Completes the multi-line `node_groups.append` arguments or elements.
                )
            if node_groups:
                return node_groups
        except (AttributeError, IndexError, KeyError, NotImplementedError, TypeError, ValueError):
            # Handles the data or control-flow case represented by this statement.
            pass

        entries = self._get_standard_element_entries(zone)
        if not entries:
            return []

        # Stores the `mesh_dimension` for use by the remaining steps in this flow.
        mesh_dimension = max(dimension for _, _, dimension in entries)
        return self._resolve_standard_boundary_entity_nodes(
            zone=zone,
            boundary_condition=boundary_condition,
            # Completes the multi-line `self._resolve_standard_boundary_entity_nodes` arguments or elements.
            entity_dimension=mesh_dimension - 1,
        )

    def _copy_boundary_condition_for_merged_zone(
        self,
        # Continues the surrounding statement or collection definition.
        zone,
        boundary_condition: Dict,
        point_offset: int,
        cell_offset: int = 0,
    # Continues the surrounding statement or collection definition.
    ) -> Dict:
        """复制边界条件，并将节点索引转换到多 Zone 合并后的坐标系。"""
        copied = dict(boundary_condition)
        grid_location = str(copied.get("grid_loc") or "Vertex").strip().lower()
        if grid_location == "vertex":
            # Stores the `point_ids` for use by the remaining steps in this flow.
            point_ids = np.empty((0,), dtype=np.int64)
            point_list = copied.get("point_list")
            point_range = copied.get("point_range")
            if point_list is not None or (
                # Completes the multi-line `np.asarray` arguments or elements.
                point_range is not None and np.asarray(point_range).size >= 4
            ):
                try:
                    zone.get_structured_vertex_dimensions()
                # Completes the multi-line `collection` arguments or elements.
                except (AttributeError, ValueError):
                    point_ids = self._expand_boundary_entity_ids(copied)
                else:
                    point_ids = self._resolve_structured_vertex_indices(zone, copied) + 1
            # Continues the surrounding statement or collection definition.
            else:
                point_ids = self._expand_boundary_entity_ids(copied)
            if point_ids.size > 0:
                copied["boundary_point_indices"] = point_ids - 1 + point_offset
                # Stores the `copied['boundary_indices_zero_based']` for use by the remaining steps in this flow.
                copied["boundary_indices_zero_based"] = True
        elif grid_location in {
            "ifacecenter", "jfacecenter", "kfacecenter",
            "iedgecenter", "jedgecenter",
        # Completes the multi-line `collection` arguments or elements.
        }:
            node_groups = self._resolve_structured_surface_entity_node_indices(
                zone,
                copied,
            # Completes the multi-line `self._resolve_structured_surface_entity_node_indices` arguments or elements.
            )
            copied["source_grid_loc"] = copied.get("grid_loc")
            copied["grid_loc"] = (
                "EdgeCenter"
                # Completes the multi-line `collection` arguments or elements.
                if grid_location in {"iedgecenter", "jedgecenter"}
                else "FaceCenter"
            )
            node_key = (
                # Continues the surrounding statement or collection definition.
                "boundary_entity_nodes"
                if copied["grid_loc"] == "EdgeCenter"
                else "boundary_face_nodes"
            )
            # Builds the `copied[node_key]` using the comprehension's filter and mapping rules.
            copied[node_key] = [
                np.asarray(nodes, dtype=np.int64).reshape(-1) + point_offset
                for nodes in node_groups
            ]
            # Stores the `copied['boundary_point_indices']` for use by the remaining steps in this flow.
            copied["boundary_point_indices"] = np.unique(
                np.concatenate(copied[node_key])
            )
            copied["boundary_indices_zero_based"] = True
        # Checks `grid_location == "facecenter"` before entering this conditional branch.
        elif grid_location == "facecenter":
            node_groups = self._resolve_face_center_node_indices(zone, copied)
            if node_groups:
                copied["boundary_face_nodes"] = [
                    # Completes the multi-line `np.asarray(nodes, dtype=np.int64).reshape` arguments or elements.
                    np.asarray(nodes, dtype=np.int64).reshape(-1) + point_offset
                    for nodes in node_groups
                ]
                copied["boundary_point_indices"] = np.unique(
                    # Completes the multi-line `np.concatenate` arguments or elements.
                    np.concatenate(copied["boundary_face_nodes"])
                )
                copied["boundary_indices_zero_based"] = True
        elif grid_location == "edgecenter":
            # Stores the `entries` for use by the remaining steps in this flow.
            entries = self._get_standard_element_entries(zone)
            if entries:
                has_poly_volume = any(
                    getattr(section, "label", None) == "Elements_t"
                    # Completes the multi-line `int` arguments or elements.
                    and int(np.asarray(section.data).reshape(-1)[0])
                    == cgnskeywords.ElementType["NFACE_n"]
                    for section in (getattr(zone, "children", {}) or {}).values()
                )
                # Stores the `mesh_dimension` for use by the remaining steps in this flow.
                mesh_dimension = max(dimension for _, _, dimension in entries)
                edge_dimension = 1 if has_poly_volume or mesh_dimension >= 3 else mesh_dimension - 1
                node_groups = self._resolve_standard_boundary_entity_nodes(
                    zone=zone,
                    # Completes the multi-line `self._resolve_standard_boundary_entity_nodes` arguments or elements.
                    boundary_condition=copied,
                    entity_dimension=edge_dimension,
                )
                if node_groups:
                    # Builds the `copied['boundary_entity_nodes']` using the comprehension's filter and mapping rules.
                    copied["boundary_entity_nodes"] = [
                        np.asarray(nodes, dtype=np.int64).reshape(-1) + point_offset
                        for nodes in node_groups
                    ]
                    # Stores the `copied['boundary_point_indices']` for use by the remaining steps in this flow.
                    copied["boundary_point_indices"] = np.unique(
                        np.concatenate(copied["boundary_entity_nodes"])
                    )
                    copied["boundary_indices_zero_based"] = True
        # Checks `grid_location == "cellcenter"` before entering this conditional branch.
        elif grid_location == "cellcenter":
            cell_indices = self._resolve_cell_center_indices(zone, copied)
            if cell_indices.size > 0:
                copied["boundary_cell_indices"] = cell_indices + cell_offset
                # Stores the `copied['boundary_indices_zero_based']` for use by the remaining steps in this flow.
                copied["boundary_indices_zero_based"] = True
        return copied

    def _detect_zone_field_location(self, fields_dict: Dict[str, np.ndarray], points: np.ndarray, conn) -> str:
        if not fields_dict:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("Fields dictionary is empty")

        field_sizes = set()
        for values in fields_dict.values():
            arr = np.asarray(values)
            # Checks `arr.ndim >= 2` before entering this conditional branch.
            if arr.ndim >= 2:
                field_sizes.add(int(arr.shape[-1]))
            else:
                field_sizes.add(len(arr.reshape(-1)))
        # Validates `len(field_sizes) != 1` and raises `ValueError` when the input cannot be processed safely.
        if len(field_sizes) != 1:
            raise ValueError(f"同一Zone内字段长度不一致: {sorted(field_sizes)}")

        field_size = field_sizes.pop()
        point_size = len(points)
        # Stores the `cell_size` for use by the remaining steps in this flow.
        cell_size = len(conn)

        is_point = field_size == point_size
        is_cell = field_size == cell_size
        if is_point and is_cell:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(
                f"字段长度{field_size}与点数/单元数同时相等，无法判定point_or_cell"
            )
        if is_point:
            # Returns `'point'` to the caller for the next stage.
            return 'point'
        if is_cell:
            return 'cell'

        raise ValueError(
            # Completes the multi-line `ValueError` arguments or elements.
            f"字段长度{field_size}与点数{point_size}、单元数{cell_size}均不匹配"
        )

    def _resolve_point_or_cell_without_fields(self) -> str:
        configured = str(self.config.get('task', {}).get('point_or_cell', '')).strip().lower()
        # Returns early when `configured in {'point', 'cell'}` makes the remaining work inapplicable.
        if configured in {'point', 'cell'}:
            return configured
        if self.point_or_cell in {'point', 'cell'}:
            return self.point_or_cell
        # Returns `'point'` to the caller for the next stage.
        return 'point'

    @staticmethod
    def _decode_h5_attr_text(value) -> str:
        """将 HDF5 属性值解码为普通字符串。"""
        if isinstance(value, np.ndarray):
            # Returns early when `value.size == 0` makes the remaining work inapplicable.
            if value.size == 0:
                return ""
            value = value.reshape(-1)[0]
        if isinstance(value, (bytes, bytearray, np.bytes_)):
            # Returns `` to the caller for the next stage.
            return bytes(value).decode("utf-8", errors="ignore").rstrip("\x00 ").strip()
        if value is None:
            return ""
        return str(value).rstrip("\x00 ").strip()

    # 读取 HDF5 group 的 CGNS label。
    def _get_h5_group_label(self, group) -> str:
        """读取 HDF5 group 的 CGNS label。"""
        return self._decode_h5_attr_text(group.attrs.get("label"))

    def _iter_h5_children_by_label(self, group, label: str):
        """遍历指定 label 的直接子 group。"""
        for child in group.values():
            # Checks `` before entering this conditional branch.
            if isinstance(child, h5py.Group) and self._get_h5_group_label(child) == label:
                yield child

    def _get_h5_single_base_group(self, handle, file_path: str):
        """从 HDF5 文件中获取唯一的 Base group。"""
        base_groups = list(self._iter_h5_children_by_label(handle, "CGNSBase_t"))
        # Validates `not base_groups` and raises `ValueError` when the input cannot be processed safely.
        if not base_groups:
            raise ValueError(f"CGNS文件未找到任何CGNSBase_t节点: {file_path}")
        if len(base_groups) != 1:
            all_base_names = [group.name.split("/")[-1] for group in base_groups]
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(
                f"CGNS文件包含{len(base_groups)}个Base，当前实现要求单Base: {file_path}. "
                f"可用Base: {all_base_names}"
            )
        # Returns `base_groups[0]` to the caller for the next stage.
        return base_groups[0]

    def _get_h5_zone_groups(self, handle, file_path: str):
        """从 HDF5 文件中获取 Base 下的所有 Zone group。"""
        base_group = self._get_h5_single_base_group(handle, file_path)
        zone_groups = list(self._iter_h5_children_by_label(base_group, "Zone_t"))
        # Validates `not zone_groups` and raises `ValueError` when the input cannot be processed safely.
        if not zone_groups:
            raise ValueError(f"单Base下未找到任何Zone_t节点: {file_path}")
        return zone_groups

    @staticmethod
    # 读取 Zone/data 中的点数、单元数和边界点数。
    def _get_h5_zone_size_triplet(zone_group) -> np.ndarray:
        """读取 Zone/data 中的点数、单元数和边界点数。"""
        data = np.asarray(zone_group[" data"][()])
        if data.shape == (1, 3):
            normalized = data[0]
        # Checks `data.shape == (3, 1)` before entering this conditional branch.
        elif data.shape == (3, 1):
            normalized = data[:, 0]
        else:
            normalized = data.transpose().reshape(-1)
        # Validates `normalized.size < 3` and raises `ValueError` when the input cannot be processed safely.
        if normalized.size < 3:
            raise ValueError(f"Zone '{zone_group.name}' 的尺寸数据格式无效")
        return np.asarray(normalized[:3], dtype=np.int64)

    @staticmethod
    # 根据字段 dataset 形状提取最后一维长度。
    def _get_h5_field_extent(data_dataset) -> int:
        """根据字段 dataset 形状提取最后一维长度。"""
        shape = tuple(int(v) for v in data_dataset.shape)
        if not shape:
            return 1
        # Returns early when `len(shape) == 1` makes the remaining work inapplicable.
        if len(shape) == 1:
            return shape[0]
        return shape[-1]

    @staticmethod
    # 根据 ElementRange 读取 section 的元素数量。
    def _get_h5_element_range_count(section_group) -> Optional[int]:
        """根据 ElementRange 读取 section 的元素数量。"""
        range_group = section_group.get("ElementRange")
        if range_group is None or " data" not in range_group:
            return None
        # Stores the current data for use by the remaining steps in this flow.
        data = np.asarray(range_group[" data"][()]).transpose().reshape(-1)
        if data.size < 2:
            return None
        start_id = int(data[0])
        # Converts the `end_id` to `int` so later checks use a consistent type.
        end_id = int(data[1])
        if end_id < start_id:
            raise ValueError(f"Section '{section_group.name}' 的 ElementRange 无效: [{start_id}, {end_id}]")
        return end_id - start_id + 1

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _get_h5_element_dimension(elem_type: int) -> Optional[int]:
        """根据元素类型推断拓扑维度。"""
        if elem_type == cgnskeywords.ElementType["MIXED"]:
            return None
        # Returns early when `elem_type == cgnskeywords.ElementType["NFACE_n"]` makes the remaining work inapplicable.
        if elem_type == cgnskeywords.ElementType["NFACE_n"]:
            return 3
        if elem_type == cgnskeywords.ElementType["NGON_n"]:
            return 2
        # Returns early when `elem_type in cgnskeywords.ElementType0D` makes the remaining work inapplicable.
        if elem_type in cgnskeywords.ElementType0D:
            return 0
        if elem_type in cgnskeywords.ElementType1D:
            return 1
        # Returns early when `` makes the remaining work inapplicable.
        if elem_type in cgnskeywords.ElementType_tri or elem_type in cgnskeywords.ElementType_quad:
            return 2
        if (
            elem_type in cgnskeywords.ElementType_tetra
            # Continues the surrounding statement or collection definition.
            or elem_type in cgnskeywords.ElementType_pyra
            or elem_type in cgnskeywords.ElementType_penta
            or elem_type in cgnskeywords.ElementType_hexa
        ):
            # Returns `3` to the caller for the next stage.
            return 3
        return None

    def _count_h5_zone_cells(self, zone_group) -> int:
        """轻量统计 Zone 的体单元数量，尽量避免读取大数组。"""
        zone_sizes = self._get_h5_zone_size_triplet(zone_group)
        # Converts the `declared_cell_count` to `int` so later checks use a consistent type.
        declared_cell_count = int(zone_sizes[1])

        section_groups = list(self._iter_h5_children_by_label(zone_group, "Elements_t"))
        if not section_groups:
            return declared_cell_count

        # Stores the `nface_count` for use by the remaining steps in this flow.
        nface_count = 0
        for section_group in section_groups:
            elem_type = int(np.asarray(section_group[" data"][()]).transpose().reshape(-1)[0])
            if elem_type != cgnskeywords.ElementType["NFACE_n"]:
                # Handles the data or control-flow case represented by this statement.
                continue
            count = self._get_h5_element_range_count(section_group)
            if count is not None:
                nface_count += count
        # Returns early when `nface_count > 0` makes the remaining work inapplicable.
        if nface_count > 0:
            return nface_count

        sections_with_dim = []
        for section_group in section_groups:
            # Converts the `elem_type` to `int` so later checks use a consistent type.
            elem_type = int(np.asarray(section_group[" data"][()]).transpose().reshape(-1)[0])
            dim = self._get_h5_element_dimension(elem_type)
            count = self._get_h5_element_range_count(section_group)
            if dim is None or count is None:
                # Handles the data or control-flow case represented by this statement.
                continue
            sections_with_dim.append((dim, count))

        if not sections_with_dim:
            return declared_cell_count

        # Stores the `max_dim` for use by the remaining steps in this flow.
        max_dim = max(dim for dim, _ in sections_with_dim)
        return sum(count for dim, count in sections_with_dim if dim == max_dim)

    @staticmethod
    def _normalize_h5_element_start_offset(raw_offset, conn_size: int, section_name: str) -> np.ndarray:
        """灏?HDF5 ElementStartOffset 鏍囧噯鍖栦负 0-based offset銆?"""
        # Stores the `offset` for use by the remaining steps in this flow.
        offset = np.asarray(raw_offset, dtype=np.int64).reshape(-1)
        if offset.size < 2:
            raise ValueError(f"{section_name}: ElementStartOffset size must be >= 2")

        if offset[0] == 0:
            # Stores the `normalized` for use by the remaining steps in this flow.
            normalized = offset
        elif offset[0] == 1:
            normalized = offset - 1
        else:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(f"{section_name}: unsupported ElementStartOffset start value {offset[0]}")

        if np.any(np.diff(normalized) < 0):
            raise ValueError(f"{section_name}: ElementStartOffset is not monotonic")
        if normalized[0] != 0:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(f"{section_name}: normalized ElementStartOffset must start from 0")
        if normalized[-1] > conn_size:
            raise ValueError(
                f"{section_name}: ElementStartOffset tail {normalized[-1]} exceeds connectivity size {conn_size}"
            # Completes the multi-line `ValueError` arguments or elements.
            )
        return normalized

    def _infer_h5_mixed_section_dimension(self, section_group) -> Optional[int]:
        """浠?MIXED section 鐨勮繛鎺ユ暟鎹帹鏂渶楂樻嫇鎵戠淮搴︺€?"""
        conn_group = section_group.get("ElementConnectivity")
        # Returns early when `conn_group is None or " data" not in conn_group` makes the remaining work inapplicable.
        if conn_group is None or " data" not in conn_group:
            return None
        conn = np.asarray(conn_group[" data"][()], dtype=np.int64).reshape(-1)
        if conn.size == 0:
            # Returns `None` to the caller for the next stage.
            return None

        start_offset_group = section_group.get("ElementStartOffset")
        dimensions: List[int] = []
        if start_offset_group is not None and " data" in start_offset_group:
            # Stores the `offsets` for use by the remaining steps in this flow.
            offsets = self._normalize_h5_element_start_offset(
                start_offset_group[" data"][()],
                len(conn),
                f"MIXED:{section_group.name}",
            # Completes the multi-line `self._normalize_h5_element_start_offset` arguments or elements.
            )
            for idx in range(offsets.size - 1):
                start = int(offsets[idx])
                end = int(offsets[idx + 1])
                # Validates `end - start < 2` and raises `ValueError` when the input cannot be processed safely.
                if end - start < 2:
                    raise ValueError(f"MIXED:{section_group.name}: invalid element payload length {end - start}")
                dim = self._get_h5_element_dimension(int(conn[start]))
                if dim is not None:
                    # Adds the current value to `dimensions` for later aggregation.
                    dimensions.append(int(dim))
        else:
            start = 0
            while start < len(conn):
                # Converts the `elem_type` to `int` so later checks use a consistent type.
                elem_type = int(conn[start])
                npe = int(cgnskeywords.ElementTypeNPE_l[elem_type])
                if npe <= 0:
                    elem_name = cgnskeywords.ElementType_l[elem_type]
                    # Raises `ValueError` to report why this path cannot continue.
                    raise ValueError(f"Element type {elem_name} does not have a fixed positive node count")
                end = start + npe + 1
                if end > len(conn):
                    raise ValueError(
                        # Completes the multi-line `ValueError` arguments or elements.
                        f"MIXED:{section_group.name}: connectivity overflow at [{start}, {end})"
                    )
                dim = self._get_h5_element_dimension(elem_type)
                if dim is not None:
                    # Adds the current value to `dimensions` for later aggregation.
                    dimensions.append(int(dim))
                start = end

        if not dimensions:
            return None
        # Returns `max(dimensions)` to the caller for the next stage.
        return max(dimensions)

    def _infer_h5_zone_mesh_dimension(self, zone_group) -> Optional[int]:
        """鏍规嵁 Zone 鐨?Elements_t 鏁版嵁鎺ㄦ柇缃戞牸鎷撴墤缁村害銆?"""
        section_groups = list(self._iter_h5_children_by_label(zone_group, "Elements_t"))
        if not section_groups:
            # Returns `None` to the caller for the next stage.
            return None

        dimensions: List[int] = []
        for section_group in section_groups:
            elem_type = int(np.asarray(section_group[" data"][()]).transpose().reshape(-1)[0])
            # Checks `elem_type == cgnskeywords.ElementType["MIXED"]` before entering this conditional branch.
            if elem_type == cgnskeywords.ElementType["MIXED"]:
                dim = self._infer_h5_mixed_section_dimension(section_group)
            else:
                dim = self._get_h5_element_dimension(elem_type)
            # Checks `dim is not None` before entering this conditional branch.
            if dim is not None:
                dimensions.append(int(dim))

        if not dimensions:
            return None
        # Returns `max(dimensions)` to the caller for the next stage.
        return max(dimensions)

    def _infer_mesh_dimension_from_element_type_ids(
        self,
        element_type_ids: Optional[np.ndarray],
    # Continues the surrounding statement or collection definition.
    ) -> Optional[int]:
        """鏍规嵁 ElementTypeIds 鎺ㄦ柇缃戞牸鎷撴墤缁村害銆?"""
        if element_type_ids is None:
            return None
        normalized = np.asarray(element_type_ids, dtype=np.int32).reshape(-1)
        # Returns early when `normalized.size == 0` makes the remaining work inapplicable.
        if normalized.size == 0:
            return None

        dimensions: List[int] = []
        for elem_type in normalized:
            # Stores the `dim` for use by the remaining steps in this flow.
            dim = self._get_h5_element_dimension(int(elem_type))
            if dim is not None:
                dimensions.append(int(dim))

        if not dimensions:
            # Returns `None` to the caller for the next stage.
            return None
        return max(dimensions)

    def _resolve_h5_grid_location(self, solution_group) -> Optional[str]:
        """从 FlowSolution/GridLocation 解析 point_or_cell。"""
        grid_location_group = solution_group.get("GridLocation")
        # Returns early when `` makes the remaining work inapplicable.
        if grid_location_group is None or " data" not in grid_location_group:
            return None

        values = np.asarray(grid_location_group[" data"][()]).transpose().reshape(-1)
        text = bytes(np.asarray(values, dtype=np.int8).tolist()).decode(
            # Completes the multi-line `bytes(np.asarray(values, dtype=np.int8).tolist()).decode('utf-8', errors='ignore').rstrip('\x00 ').strip` arguments or elements.
            "utf-8",
            errors="ignore",
        ).rstrip("\x00 ").strip()
        normalized = text.lower()
        # Returns early when `normalized == "vertex"` makes the remaining work inapplicable.
        if normalized == "vertex":
            return "point"
        if normalized == "cellcenter":
            return "cell"
        # Returns `None` to the caller for the next stage.
        return None

    def _resolve_solution_node_grid_location(self, solution_node) -> Optional[str]:
        """从节点式 FlowSolution/GridLocation 解析 point_or_cell。"""
        if solution_node is None or not getattr(solution_node, "children", None):
            return None

        # Reads `GridLocation` from the mapping into the `grid_location_node`.
        grid_location_node = solution_node.children.get("GridLocation")
        if grid_location_node is None:
            return None

        values = np.asarray(getattr(grid_location_node, "data", None))
        # Returns early when `values.size == 0` makes the remaining work inapplicable.
        if values.size == 0:
            return None
        if isinstance(getattr(grid_location_node, "data", None), (bytes, np.bytes_)):
            text = bytes(getattr(grid_location_node, "data")).decode(
                # Completes the multi-line `bytes(getattr(grid_location_node, 'data')).decode('utf-8', errors='ignore').rstrip('\x00 ').strip` arguments or elements.
                "utf-8",
                errors="ignore",
            ).rstrip("\x00 ").strip()
        elif values.dtype.kind in {"S", "a"}:
            # Stores the `text` for use by the remaining steps in this flow.
            text = bytes(values.reshape(-1)[0]).decode(
                "utf-8",
                errors="ignore",
            ).rstrip("\x00 ").strip()
        # Continues the surrounding statement or collection definition.
        else:
            text = bytes(np.asarray(values, dtype=np.int8).reshape(-1).tolist()).decode(
                "utf-8",
                errors="ignore",
            # Completes the multi-line `bytes(np.asarray(values, dtype=np.int8).reshape(-1).tolist()).decode('utf-8', errors='ignore').rstrip('\x00 ').strip` arguments or elements.
            ).rstrip("\x00 ").strip()
        normalized = text.lower()
        if normalized == "vertex":
            return "point"
        # Returns early when `normalized == "cellcenter"` makes the remaining work inapplicable.
        if normalized == "cellcenter":
            return "cell"
        return None

    def _infer_h5_zone_field_location(
        # Continues the surrounding statement or collection definition.
        self,
        zone_group,
        solution_group,
        field_names: List[str],
    # Continues the surrounding statement or collection definition.
    ) -> str:
        """基于元数据推断单个 Zone 的字段位置。"""
        explicit_location = self._resolve_h5_grid_location(solution_group)
        if explicit_location in {"point", "cell"}:
            return explicit_location

        # Stores the `zone_sizes` for use by the remaining steps in this flow.
        zone_sizes = self._get_h5_zone_size_triplet(zone_group)
        point_size = int(zone_sizes[0])
        cell_size = int(self._count_h5_zone_cells(zone_group))

        field_sizes = set()
        # Visits each item in `field_names` and applies the current processing step.
        for field_name in field_names:
            field_group = solution_group[field_name]
            if " data" not in field_group:
                raise ValueError(f"FlowSolution 字段 '{field_name}' 缺少 data 数据集")
            # Adds the current value to `field_sizes` for later aggregation.
            field_sizes.add(self._get_h5_field_extent(field_group[" data"]))

        if len(field_sizes) != 1:
            raise ValueError(f"同一Zone内字段长度不一致: {sorted(field_sizes)}")

        field_size = field_sizes.pop()
        # Stores the `is_point` for use by the remaining steps in this flow.
        is_point = field_size == point_size
        is_cell = field_size == cell_size
        if is_point and is_cell:
            raise ValueError(f"字段长度{field_size}与点数/单元数同时相等，无法判定point_or_cell")
        # Returns early when `is_point` makes the remaining work inapplicable.
        if is_point:
            return "point"
        if is_cell:
            return "cell"
        # Raises `ValueError` to report why this path cannot continue.
        raise ValueError(
            f"字段长度{field_size}与点数{point_size}、单元数{cell_size}均不匹配"
        )

    def _resolve_variable_manifest_from_h5(
        # Continues the surrounding statement or collection definition.
        self,
        file_path: str,
        logical_variables: List[str],
    ) -> Dict[str, object]:
        """通过 HDF5 元数据快速解析变量清单，避免重建大网格连通性。"""
        # Stores the `expected_field_names` for use by the remaining steps in this flow.
        expected_field_names = None
        expected_field_location = None
        expected_mesh_dimension = None

        with h5py.File(file_path, "r") as handle:
            # Stores the `zone_groups` for use by the remaining steps in this flow.
            zone_groups = self._get_h5_zone_groups(handle, file_path)

            for zone_group in zone_groups:
                zone_mesh_dimension = self._infer_h5_zone_mesh_dimension(zone_group)
                if zone_mesh_dimension is not None and expected_mesh_dimension is None:
                    # Converts the `expected_mesh_dimension` to `int` so later checks use a consistent type.
                    expected_mesh_dimension = int(zone_mesh_dimension)
                elif (
                    zone_mesh_dimension is not None
                    and expected_mesh_dimension is not None
                    # Completes the multi-line `int` arguments or elements.
                    and int(zone_mesh_dimension) != int(expected_mesh_dimension)
                ):
                    raise ValueError(
                        f"Zone 缃戞牸缁村害涓嶄竴鑷? zone={zone_group.name.split('/')[-1]}, "
                        # Completes the multi-line `ValueError` arguments or elements.
                        f"expect={expected_mesh_dimension}, actual={zone_mesh_dimension}"
                    )

                solution_groups = list(self._iter_h5_children_by_label(zone_group, "FlowSolution_t"))
                if not solution_groups:
                    # Raises `ValueError` to report why this path cannot continue.
                    raise ValueError(
                        f"Zone '{zone_group.name.split('/')[-1]}' 缺少FlowSolution节点，无法解析变量清单: {file_path}"
                    )

                current_names = None
                # Stores the `current_location` for use by the remaining steps in this flow.
                current_location = None
                for solution_group in solution_groups:
                    names = [
                        name for name in solution_group.keys()
                        # Completes the multi-line `collection` arguments or elements.
                        if name not in {"GridLocation", "PointRange"}
                    ]
                    if not names:
                        raise ValueError(
                            # Completes the multi-line `zone_group.name.split` arguments or elements.
                            f"Zone '{zone_group.name.split('/')[-1]}' 中未找到任何可用场变量: {file_path}"
                        )

                    names_set = set(names)
                    if current_names is None:
                        # Stores the `current_names` for use by the remaining steps in this flow.
                        current_names = names_set
                    elif names_set != current_names:
                        missing = sorted(current_names - names_set)
                        extra = sorted(names_set - current_names)
                        # Raises `ValueError` to report why this path cannot continue.
                        raise ValueError(
                            f"Zone内不同时刻字段集合不一致: zone={zone_group.name.split('/')[-1]}, "
                            f"missing={missing}, extra={extra}"
                        )

                    # Stores the `solution_location` for use by the remaining steps in this flow.
                    solution_location = self._infer_h5_zone_field_location(
                        zone_group,
                        solution_group,
                        list(names_set),
                    # Completes the multi-line `self._infer_h5_zone_field_location` arguments or elements.
                    )
                    if current_location is None:
                        current_location = solution_location
                    elif solution_location != current_location:
                        # Raises `ValueError` to report why this path cannot continue.
                        raise ValueError(
                            f"Zone内不同时刻字段位置不一致: zone={zone_group.name.split('/')[-1]}, "
                            f"expect={current_location}, actual={solution_location}"
                        )

                # Checks `current_names is None` before entering this conditional branch.
                if current_names is None:
                    continue

                if expected_field_names is None:
                    expected_field_names = current_names
                # Checks `current_names != expected_field_names` before entering this conditional branch.
                elif current_names != expected_field_names:
                    missing = sorted(expected_field_names - current_names)
                    extra = sorted(current_names - expected_field_names)
                    raise ValueError(
                        # Completes the multi-line `zone_group.name.split` arguments or elements.
                        f"Zone字段集合不一致: zone={zone_group.name.split('/')[-1]}, "
                        f"missing={missing}, extra={extra}"
                    )

                if current_location is not None and expected_field_location is None:
                    # Stores the `expected_field_location` for use by the remaining steps in this flow.
                    expected_field_location = current_location
                elif current_location is not None and current_location != expected_field_location:
                    raise ValueError(
                        f"多Zone字段位置不一致: 期望{expected_field_location}, "
                        # Completes the multi-line `zone_group.name.split` arguments or elements.
                        f"zone={zone_group.name.split('/')[-1]}为{current_location}"
                    )

        if not expected_field_names:
            raise ValueError(f"单Base下未提取到任何变量字段: {file_path}")

        # Stores the `field_names` for use by the remaining steps in this flow.
        field_names = sorted(expected_field_names)
        resolved_location = expected_field_location or self._resolve_point_or_cell_without_fields()
        variable_catalog = self._build_cgns_variable_catalog(
            {field_name: resolved_location for field_name in field_names}
        # Completes the multi-line `self._build_cgns_variable_catalog` arguments or elements.
        )
        (
            field_component_map,
            effective_variables,
            # Completes the multi-line `collection` arguments or elements.
            effective_variable_sources,
            field_location_map,
            _,
        ) = select_catalog_variables(variable_catalog, logical_variables)
        # Returns `` to the caller for the next stage.
        return {
            'logical_variables': list(logical_variables),
            'effective_variables': list(effective_variables),
            'field_component_map': {k: list(v) for k, v in field_component_map.items()},
            # Completes the multi-line `collection` arguments or elements.
            'variable_catalog': variable_catalog,
            'effective_variable_sources': effective_variable_sources,
            'point_or_cell': resolved_location,
            'field_location_map': field_location_map,
            # Completes the multi-line `collection` arguments or elements.
            'projected_point_or_cell': resolved_location,
            'mixed_field_policy': 'majority_side',
            'projection_method': 'mean',
            'mesh_dimension': expected_mesh_dimension,
        # Completes the multi-line `collection` arguments or elements.
        }

    def _resolve_variable_manifest_from_h5_mixed(
        self,
        file_path: str,
        # Continues the surrounding statement or collection definition.
        logical_variables: List[str],
    ) -> Dict[str, object]:
        """通过 HDF5 元数据解析支持混合场的变量清单。"""
        expected_field_names = None
        expected_field_location_map: Dict[str, str] = {}
        # Stores the `expected_mesh_dimension` for use by the remaining steps in this flow.
        expected_mesh_dimension = None

        with h5py.File(file_path, "r") as handle:
            zone_groups = self._get_h5_zone_groups(handle, file_path)

            for zone_group in zone_groups:
                # Stores the `zone_mesh_dimension` for use by the remaining steps in this flow.
                zone_mesh_dimension = self._infer_h5_zone_mesh_dimension(zone_group)
                if zone_mesh_dimension is not None and expected_mesh_dimension is None:
                    expected_mesh_dimension = int(zone_mesh_dimension)
                elif (
                    # Continues the surrounding statement or collection definition.
                    zone_mesh_dimension is not None
                    and expected_mesh_dimension is not None
                    and int(zone_mesh_dimension) != int(expected_mesh_dimension)
                ):
                    # Raises `ValueError` to report why this path cannot continue.
                    raise ValueError(
                        f"Zone 网格维度不一致: zone={zone_group.name.split('/')[-1]}, "
                        f"expect={expected_mesh_dimension}, actual={zone_mesh_dimension}"
                    )

                # Stores the `solution_groups` for use by the remaining steps in this flow.
                solution_groups = list(self._iter_h5_children_by_label(zone_group, "FlowSolution_t"))
                if not solution_groups:
                    raise ValueError(
                        f"Zone '{zone_group.name.split('/')[-1]}' 缺少 FlowSolution 节点，无法解析变量清单: {file_path}"
                    # Completes the multi-line `ValueError` arguments or elements.
                    )

                current_names = set()
                current_location_map: Dict[str, str] = {}
                for solution_group in solution_groups:
                    # Builds the `names` using the comprehension's filter and mapping rules.
                    names = [
                        name for name in solution_group.keys()
                        if name not in {"GridLocation", "PointRange"}
                    ]
                    # Validates `not names` and raises `ValueError` when the input cannot be processed safely.
                    if not names:
                        raise ValueError(
                            f"Zone '{zone_group.name.split('/')[-1]}' 中未找到任何可用场变量: {file_path}"
                        )

                    # Stores the `names_set` for use by the remaining steps in this flow.
                    names_set = set(names)
                    solution_location = self._infer_h5_zone_field_location(
                        zone_group,
                        solution_group,
                        # Completes the multi-line `list` arguments or elements.
                        list(names_set),
                    )
                    for name in names_set:
                        existing = current_location_map.get(name)
                        # Validates `existing is not None and existing != solution_location` and raises `ValueError` when the input cannot be processed safely.
                        if existing is not None and existing != solution_location:
                            raise ValueError(
                                f"Zone 内字段位置不一致: zone={zone_group.name.split('/')[-1]}, "
                                f"field={name}, expect={existing}, actual={solution_location}"
                            # Completes the multi-line `ValueError` arguments or elements.
                            )
                        current_location_map[name] = solution_location
                    current_names.update(names_set)

                if not current_names:
                    # Handles the data or control-flow case represented by this statement.
                    continue

                if expected_field_names is None:
                    expected_field_names = set(current_names)
                elif current_names != expected_field_names:
                    # Stores the `missing` for use by the remaining steps in this flow.
                    missing = sorted(expected_field_names - current_names)
                    extra = sorted(current_names - expected_field_names)
                    raise ValueError(
                        f"Zone 字段集合不一致: zone={zone_group.name.split('/')[-1]}, "
                        # Completes the multi-line `ValueError` arguments or elements.
                        f"missing={missing}, extra={extra}"
                    )

                for name, location in current_location_map.items():
                    existing = expected_field_location_map.get(name)
                    # Validates `existing is not None and existing != location` and raises `ValueError` when the input cannot be processed safely.
                    if existing is not None and existing != location:
                        raise ValueError(
                            f"跨 Zone 字段位置不一致: zone={zone_group.name.split('/')[-1]}, "
                            f"field={name}, expect={existing}, actual={location}"
                        # Completes the multi-line `ValueError` arguments or elements.
                        )
                    expected_field_location_map[name] = location

        if not expected_field_names:
            raise ValueError(f"Base 下未提取到任何变量字段: {file_path}")

        # Stores the `variable_catalog` for use by the remaining steps in this flow.
        variable_catalog = self._build_cgns_variable_catalog(expected_field_location_map)
        (
            field_component_map,
            effective_variables,
            # Completes the multi-line `collection` arguments or elements.
            effective_variable_sources,
            field_location_map,
            resolved_selection,
        ) = select_catalog_variables(variable_catalog, logical_variables)
        # Stores the `(point_component_map, cell_component_map)` for use by the remaining steps in this flow.
        point_component_map, cell_component_map = self._catalog_maps_by_location(variable_catalog)
        selected_point_component_map = {
            key: point_component_map[key]
            for key in resolved_selection
            # Continues the surrounding statement or collection definition.
            if key in point_component_map
        }
        selected_cell_component_map = {
            key: cell_component_map[key]
            # Continues the surrounding statement or collection definition.
            for key in resolved_selection
            if key in cell_component_map
        }
        projected_point_or_cell = choose_projected_point_or_cell(
            # Completes the multi-line `choose_projected_point_or_cell` arguments or elements.
            requested_vars=resolved_selection,
            point_component_map=selected_point_component_map,
            cell_component_map=selected_cell_component_map,
            default_location="point",
        # Completes the multi-line `choose_projected_point_or_cell` arguments or elements.
        )
        return {
            'logical_variables': list(logical_variables),
            'effective_variables': list(effective_variables),
            # Completes the multi-line `list` arguments or elements.
            'field_component_map': {k: list(v) for k, v in field_component_map.items()},
            'variable_catalog': variable_catalog,
            'effective_variable_sources': effective_variable_sources,
            'point_or_cell': projected_point_or_cell,
            # Completes the multi-line `collection` arguments or elements.
            'field_location_map': field_location_map,
            'projected_point_or_cell': projected_point_or_cell,
            'mixed_field_policy': 'majority_side',
            'projection_method': 'mean',
            # Completes the multi-line `collection` arguments or elements.
            'mesh_dimension': expected_mesh_dimension,
        }

    def _merge_zone_data(
        self,
        # Continues the surrounding statement or collection definition.
        zone_nodes,
        file_path: str,
        allow_missing_fields: bool = False,
        max_time_steps: Optional[int] = None,
    # Continues the surrounding statement or collection definition.
    ):
        """Read all zones and merge them into a single mesh/field view."""
        merged_points_parts = []
        merged_conn_cells = []
        merged_fields_parts = {}
        # Initializes the `merged_boundary_conditions` as a collection for values from this processing stage.
        merged_boundary_conditions = {}
        merged_element_type_ids = []

        expected_field_names = None
        expected_field_location = None
        # Stores the `expected_mesh_dimension` for use by the remaining steps in this flow.
        expected_mesh_dimension = None
        point_offset = 0
        cell_offset = 0

        for zone in zone_nodes:
            # Stores the `points` for use by the remaining steps in this flow.
            points = zone.get_mesh_coords()
            conn_dict = zone.get_mesh_connectivity()
            current_element_type_ids = None
            if isinstance(conn_dict, dict) and "Elem2Node" in conn_dict:
                # Selects the required item from `conn_dict["Elem2Node"]` as the `conn`.
                conn = conn_dict["Elem2Node"]["connectivity"]
                current_element_type_ids = conn_dict.get("ElementTypeIds")
            else:
                conn = conn_dict

            # Stores the `current_mesh_dimension` for use by the remaining steps in this flow.
            current_mesh_dimension = self._infer_mesh_dimension_from_element_type_ids(
                current_element_type_ids,
            )
            if current_mesh_dimension is not None and expected_mesh_dimension is None:
                # Converts the `expected_mesh_dimension` to `int` so later checks use a consistent type.
                expected_mesh_dimension = int(current_mesh_dimension)
            elif (
                current_mesh_dimension is not None
                and expected_mesh_dimension is not None
                # Completes the multi-line `int` arguments or elements.
                and int(current_mesh_dimension) != int(expected_mesh_dimension)
            ):
                raise ValueError(
                    f"Zone 缃戞牸缁村害涓嶄竴鑷? zone={zone.name}, "
                    # Completes the multi-line `ValueError` arguments or elements.
                    f"expect={expected_mesh_dimension}, actual={current_mesh_dimension}"
                )

            all_fields = self._extract_fields(
                zone,
                # Completes the multi-line `self._extract_fields` arguments or elements.
                variables=None,
                allow_missing_fields=allow_missing_fields,
                max_time_steps=max_time_steps,
            )
            # Validates `not all_fields and not allow_missing_fields` and raises `ValueError` when the input cannot be processed safely.
            if not all_fields and not allow_missing_fields:
                raise ValueError(f"Zone '{zone.name}'未读取到任何Fields，无法合并")

            current_names = set(all_fields.keys())
            if expected_field_names is None:
                # Stores the `expected_field_names` for use by the remaining steps in this flow.
                expected_field_names = current_names
            elif current_names != expected_field_names:
                missing = sorted(expected_field_names - current_names)
                extra = sorted(current_names - expected_field_names)
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(
                    f"Zone字段集合不一致: zone={zone.name}, missing={missing}, extra={extra}"
                )

            current_location = self._detect_zone_field_location(all_fields, points, conn) if all_fields else None
            # Checks `current_location is not None and expected_field_location is None` before entering this conditional branch.
            if current_location is not None and expected_field_location is None:
                expected_field_location = current_location
            elif current_location is not None and current_location != expected_field_location:
                raise ValueError(
                    # Completes the multi-line `ValueError` arguments or elements.
                    f"多Zone字段位置不一致: 期望{expected_field_location}, zone={zone.name}为{current_location}"
                )

            shifted_conn = self._offset_connectivity(conn, point_offset)
            zone_point_count = len(points)

            # Adds the current value to `merged_points_parts` for later aggregation.
            merged_points_parts.append(points)
            for cell in shifted_conn:
                merged_conn_cells.append(np.asarray(cell, dtype=np.int64).reshape(-1))
            if current_element_type_ids is not None:
                # Adds the current value to `merged_element_type_ids` for later aggregation.
                merged_element_type_ids.extend(
                    np.asarray(current_element_type_ids, dtype=np.int32).reshape(-1).tolist()
                )

            for field_name, field_data in all_fields.items():
                # Adds the current value to `merged_fields_parts.setdefault(field_name, [])` for later aggregation.
                merged_fields_parts.setdefault(field_name, []).append(np.asarray(field_data))

            try:
                bc_info = zone.get_boundary_conditions(self._family_bc_types)
                if isinstance(bc_info, dict):
                    # Visits each item in `bc_info.items()` and applies the current processing step.
                    for bc_name, bc in bc_info.items():
                        prefixed_name = f"{zone.name}:{bc_name}"
                        merged_boundary_conditions[prefixed_name] = (
                            self._copy_boundary_condition_for_merged_zone(
                                # Completes the multi-line `self._copy_boundary_condition_for_merged_zone` arguments or elements.
                                zone=zone,
                                boundary_condition=bc,
                                point_offset=point_offset,
                                cell_offset=cell_offset,
                            # Completes the multi-line `self._copy_boundary_condition_for_merged_zone` arguments or elements.
                            )
                        )
            except Exception as e:
                if self._requires_cgns_boundary_conditions():
                    # Raises `ValueError` to report why this path cannot continue.
                    raise ValueError(
                        f"提取边界条件失败: zone={zone.name}, reason={e}"
                    ) from e
                _get_logger().warning(
                    # Completes the multi-line `_get_logger().warning` arguments or elements.
                    "提取边界条件失败: zone=%s reason=%s",
                    zone.name,
                    str(e),
                )

            # Keeps the operation and its recovery path within the same exception boundary.
            try:
                interface_info = zone.get_grid_connectivities()
                for interface_name, interface in interface_info.items():
                    prefixed_name = f"{zone.name}:interface:{interface_name}"
                    # Stores the `merged_boundary_conditions[prefixed_name]` for use by the remaining steps in this flow.
                    merged_boundary_conditions[prefixed_name] = (
                        self._copy_boundary_condition_for_merged_zone(
                            zone=zone,
                            boundary_condition=interface,
                            # Completes the multi-line `self._copy_boundary_condition_for_merged_zone` arguments or elements.
                            point_offset=point_offset,
                            cell_offset=cell_offset,
                        )
                    )
            # Continues the surrounding statement or collection definition.
            except Exception as e:
                if self._requires_cgns_boundary_conditions():
                    raise ValueError(
                        f"提取网格连接失败: zone={zone.name}, reason={e}"
                    # Completes the multi-line `ValueError` arguments or elements.
                    ) from e
                _get_logger().warning(
                    "提取网格连接失败: zone=%s reason=%s",
                    zone.name,
                    # Completes the multi-line `str` arguments or elements.
                    str(e),
                )

            point_offset += zone_point_count
            cell_offset += len(conn)

        # Stores the `merged_points` for use by the remaining steps in this flow.
        merged_points = np.vstack(merged_points_parts)
        merged_conn = np.asarray(merged_conn_cells, dtype=object)
        merged_fields = {}
        for field_name, parts in merged_fields_parts.items():
            # Stores the `first_part` for use by the remaining steps in this flow.
            first_part = np.asarray(parts[0])
            concat_axis = 0 if first_part.ndim <= 1 else first_part.ndim - 1
            merged_fields[field_name] = np.concatenate(parts, axis=concat_axis)

        try:
            # Stores the `merged_boundary_conditions['bc_flags']` for use by the remaining steps in this flow.
            merged_boundary_conditions['bc_flags'] = np.asarray(
                np.concatenate([zone.get_boundary_flags() for zone in zone_nodes], axis=0)
            ).reshape(-1)
            if len(merged_boundary_conditions['bc_flags']) != len(merged_points):
                # Stores the `merged_boundary_conditions['bc_flags']` for use by the remaining steps in this flow.
                merged_boundary_conditions['bc_flags'] = np.zeros(len(merged_points), dtype=int)
        except Exception:
            merged_boundary_conditions['bc_flags'] = np.zeros(len(merged_points), dtype=int)

        return {
            # Completes the multi-line `collection` arguments or elements.
            'points': merged_points,
            'conn': merged_conn,
            'element_type_ids': (
                np.asarray(merged_element_type_ids, dtype=np.int32)
                # Completes the multi-line `collection` arguments or elements.
                if merged_element_type_ids
                else None
            ),
            'all_fields': merged_fields,
            # Completes the multi-line `collection` arguments or elements.
            'boundary_conditions': merged_boundary_conditions,
            'point_or_cell': expected_field_location or self._resolve_point_or_cell_without_fields(),
            'mesh_dimension': expected_mesh_dimension,
            'source_file': file_path,
        # Completes the multi-line `collection` arguments or elements.
        }

    def _merge_zone_data_mixed(
        self,
        zone_nodes,
        # Continues the surrounding statement or collection definition.
        file_path: str,
        requested_vars: Optional[List[str]] = None,
        load_variables: Optional[List[str]] = None,
        allow_missing_fields: bool = False,
        # Continues the surrounding statement or collection definition.
        max_time_steps: Optional[int] = None,
    ):
        """支持混合场投影的多 Zone 合并。"""
        merged_points_parts = []
        merged_conn_cells = []
        # Initializes the `merged_point_fields_parts` as a collection for values from this processing stage.
        merged_point_fields_parts: Dict[str, List[np.ndarray]] = {}
        merged_cell_fields_parts: Dict[str, List[np.ndarray]] = {}
        merged_boundary_conditions = {}
        merged_element_type_ids = []

        # Stores the `expected_field_names` for use by the remaining steps in this flow.
        expected_field_names = None
        expected_field_location_map: Dict[str, str] = {}
        expected_mesh_dimension = None
        point_offset = 0
        # Stores the `cell_offset` for use by the remaining steps in this flow.
        cell_offset = 0

        for zone in zone_nodes:
            points = zone.get_mesh_coords()
            conn_dict = zone.get_mesh_connectivity()
            # Stores the `current_element_type_ids` for use by the remaining steps in this flow.
            current_element_type_ids = None
            if isinstance(conn_dict, dict) and "Elem2Node" in conn_dict:
                conn = conn_dict["Elem2Node"]["connectivity"]
                current_element_type_ids = conn_dict.get("ElementTypeIds")
            # Continues the surrounding statement or collection definition.
            else:
                conn = conn_dict

            current_mesh_dimension = self._infer_mesh_dimension_from_element_type_ids(
                current_element_type_ids,
            # Completes the multi-line `self._infer_mesh_dimension_from_element_type_ids` arguments or elements.
            )
            if current_mesh_dimension is not None and expected_mesh_dimension is None:
                expected_mesh_dimension = int(current_mesh_dimension)
            elif (
                # Continues the surrounding statement or collection definition.
                current_mesh_dimension is not None
                and expected_mesh_dimension is not None
                and int(current_mesh_dimension) != int(expected_mesh_dimension)
            ):
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(
                    f"Zone 网格维度不一致: zone={zone.name}, "
                    f"expect={expected_mesh_dimension}, actual={current_mesh_dimension}"
                )

            # Stores the `zone_payload` for use by the remaining steps in this flow.
            zone_payload = self._extract_zone_fields_with_locations(
                zone_node=zone,
                points=points,
                conn=conn,
                # Completes the multi-line `self._extract_zone_fields_with_locations` arguments or elements.
                variables=load_variables,
                allow_missing_fields=allow_missing_fields,
                max_time_steps=max_time_steps,
            )
            # Stores the `all_fields` for use by the remaining steps in this flow.
            all_fields = zone_payload.get("fields") or {}
            zone_field_location_map = zone_payload.get("field_location_map") or {}
            if not all_fields and not allow_missing_fields:
                raise ValueError(f"Zone '{zone.name}' 未读取到任何 Fields，无法合并")

            # Stores the `current_names` for use by the remaining steps in this flow.
            current_names = set(all_fields.keys())
            if expected_field_names is None:
                expected_field_names = set(current_names)
            elif current_names != expected_field_names:
                # Stores the `missing` for use by the remaining steps in this flow.
                missing = sorted(expected_field_names - current_names)
                extra = sorted(current_names - expected_field_names)
                raise ValueError(
                    f"Zone 字段集合不一致: zone={zone.name}, missing={missing}, extra={extra}"
                # Completes the multi-line `ValueError` arguments or elements.
                )

            for field_name, location in zone_field_location_map.items():
                existing = expected_field_location_map.get(field_name)
                if existing is not None and existing != location:
                    # Raises `ValueError` to report why this path cannot continue.
                    raise ValueError(
                        f"跨 Zone 字段位置不一致: zone={zone.name}, "
                        f"field={field_name}, expect={existing}, actual={location}"
                    )
                # Stores the `expected_field_location_map[field_name]` for use by the remaining steps in this flow.
                expected_field_location_map[field_name] = location

            shifted_conn = self._offset_connectivity(conn, point_offset)
            zone_point_count = len(points)

            merged_points_parts.append(points)
            # Visits each item in `shifted_conn` and applies the current processing step.
            for cell in shifted_conn:
                merged_conn_cells.append(np.asarray(cell, dtype=np.int64).reshape(-1))
            if current_element_type_ids is not None:
                merged_element_type_ids.extend(
                    # Completes the multi-line `np.asarray(current_element_type_ids, dtype=np.int32).reshape(-1).tolist` arguments or elements.
                    np.asarray(current_element_type_ids, dtype=np.int32).reshape(-1).tolist()
                )

            for field_name, field_data in all_fields.items():
                location = str(zone_field_location_map.get(field_name, "")).strip().lower()
                # Checks `location == "point"` before entering this conditional branch.
                if location == "point":
                    merged_point_fields_parts.setdefault(field_name, []).append(np.asarray(field_data))
                elif location == "cell":
                    merged_cell_fields_parts.setdefault(field_name, []).append(np.asarray(field_data))
                # Continues the surrounding statement or collection definition.
                else:
                    raise ValueError(f"Zone '{zone.name}' 字段 {field_name} 缺少合法位置标记")

            try:
                bc_info = zone.get_boundary_conditions(self._family_bc_types)
                # Checks `isinstance(bc_info, dict)` before entering this conditional branch.
                if isinstance(bc_info, dict):
                    for bc_name, bc in bc_info.items():
                        prefixed_name = f"{zone.name}:{bc_name}"
                        merged_boundary_conditions[prefixed_name] = (
                            # Completes the multi-line `self._copy_boundary_condition_for_merged_zone` arguments or elements.
                            self._copy_boundary_condition_for_merged_zone(
                                zone=zone,
                                boundary_condition=bc,
                                point_offset=point_offset,
                                # Completes the multi-line `self._copy_boundary_condition_for_merged_zone` arguments or elements.
                                cell_offset=cell_offset,
                            )
                        )
            except Exception as e:
                # Validates `self._requires_cgns_boundary_conditions()` and raises `ValueError` when the input cannot be processed safely.
                if self._requires_cgns_boundary_conditions():
                    raise ValueError(
                        f"提取边界条件失败: zone={zone.name}, reason={e}"
                    ) from e
                # Records the current execution state to make progress and failures traceable.
                _get_logger().warning(
                    "提取边界条件失败: zone=%s reason=%s",
                    zone.name,
                    str(e),
                # Completes the multi-line `_get_logger().warning` arguments or elements.
                )

            try:
                interface_info = zone.get_grid_connectivities()
                for interface_name, interface in interface_info.items():
                    # Stores the `prefixed_name` for use by the remaining steps in this flow.
                    prefixed_name = f"{zone.name}:interface:{interface_name}"
                    merged_boundary_conditions[prefixed_name] = (
                        self._copy_boundary_condition_for_merged_zone(
                            zone=zone,
                            # Completes the multi-line `self._copy_boundary_condition_for_merged_zone` arguments or elements.
                            boundary_condition=interface,
                            point_offset=point_offset,
                            cell_offset=cell_offset,
                        )
                    # Continues the surrounding statement or collection definition.
                    )
            except Exception as e:
                if self._requires_cgns_boundary_conditions():
                    raise ValueError(
                        # Completes the multi-line `ValueError` arguments or elements.
                        f"提取网格连接失败: zone={zone.name}, reason={e}"
                    ) from e
                _get_logger().warning(
                    "提取网格连接失败: zone=%s reason=%s",
                    # Completes the multi-line `_get_logger().warning` arguments or elements.
                    zone.name,
                    str(e),
                )

            point_offset += zone_point_count
            # Stores the `cell_offset` for use by the remaining steps in this flow.
            cell_offset += len(conn)

        merged_points = np.vstack(merged_points_parts)
        merged_conn = np.asarray(merged_conn_cells, dtype=object)

        def merge_field_parts(parts_map: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
            # Initializes the `merged_fields` as a collection for values from this processing stage.
            merged_fields = {}
            for field_name, parts in parts_map.items():
                first_part = np.asarray(parts[0])
                concat_axis = 0 if first_part.ndim <= 1 else first_part.ndim - 1
                # Stores the `merged_fields[field_name]` for use by the remaining steps in this flow.
                merged_fields[field_name] = np.concatenate(parts, axis=concat_axis)
            return merged_fields

        point_fields = merge_field_parts(merged_point_fields_parts)
        cell_fields = merge_field_parts(merged_cell_fields_parts)

        # Initializes the `raw_field_location_map` as a collection for values from this processing stage.
        raw_field_location_map = {
            **{field_name: "point" for field_name in point_fields},
            **{field_name: "cell" for field_name in cell_fields},
        }
        # Stores the `variable_catalog` for use by the remaining steps in this flow.
        variable_catalog = self._build_cgns_variable_catalog(raw_field_location_map)
        field_component_map = catalog_component_map(variable_catalog)
        all_effective_sources = catalog_sources_by_runtime_key(variable_catalog)
        point_component_map, cell_component_map = self._catalog_maps_by_location(variable_catalog)

        # Defines `alias_source_fields` for alias source fields logic.
        def alias_source_fields(
            source_fields: Dict[str, np.ndarray],
            location: str,
        ) -> Dict[str, np.ndarray]:
            # Initializes the `aliased_fields` as a collection for values from this processing stage.
            aliased_fields: Dict[str, np.ndarray] = {}
            for runtime_key, source in all_effective_sources.items():
                if source.get("location") != location:
                    continue
                # Converts the `raw_name` to `str` so later checks use a consistent type.
                raw_name = str(source.get("array_name") or "")
                if raw_name not in source_fields:
                    continue
                aliased_fields[runtime_key] = source_fields[raw_name]
            # Returns `aliased_fields` to the caller for the next stage.
            return aliased_fields

        point_fields = alias_source_fields(point_fields, "point")
        cell_fields = alias_source_fields(cell_fields, "cell")
        field_location_map = {
            # Completes the multi-line `str` arguments or elements.
            runtime_key: str(source.get("location"))
            for runtime_key, source in all_effective_sources.items()
        }

        selected_component_map = field_component_map
        # Stores the `selected_effective_variables` for use by the remaining steps in this flow.
        selected_effective_variables = list(all_effective_sources)
        resolved_selection = list(variable_catalog)
        if requested_vars:
            (
                # Completes the multi-line `collection` arguments or elements.
                selected_component_map,
                selected_effective_variables,
                _,
                _,
                # Completes the multi-line `collection` arguments or elements.
                resolved_selection,
            ) = select_catalog_variables(
                variable_catalog,
                requested_vars,
                # Completes the multi-line `select_catalog_variables` arguments or elements.
                strict_missing=False,
            )

        selected_point_component_map = {
            key: point_component_map[key]
            # Continues the surrounding statement or collection definition.
            for key in resolved_selection
            if key in point_component_map
        }
        selected_cell_component_map = {
            # Continues the surrounding statement or collection definition.
            key: cell_component_map[key]
            for key in resolved_selection
            if key in cell_component_map
        }
        # Stores the `projected_point_or_cell` for use by the remaining steps in this flow.
        projected_point_or_cell = choose_projected_point_or_cell(
            requested_vars=resolved_selection,
            point_component_map=selected_point_component_map,
            cell_component_map=selected_cell_component_map,
            # Completes the multi-line `choose_projected_point_or_cell` arguments or elements.
            default_location="point",
        )
        merged_source_fields = dict(point_fields)
        merged_source_fields.update(cell_fields)
        # Stores the `projection_source_fields` for use by the remaining steps in this flow.
        projection_source_fields = merged_source_fields
        if requested_vars and selected_effective_variables:
            selected_effective_set = set(selected_effective_variables)
            projection_source_fields = {
                # Continues the surrounding statement or collection definition.
                field_name: field_values
                for field_name, field_values in merged_source_fields.items()
                if field_name in selected_effective_set
            }
        # Stores the `projected_fields` for use by the remaining steps in this flow.
        projected_fields = project_fields_to_location(
            fields=projection_source_fields,
            field_location_map=field_location_map,
            target_location=projected_point_or_cell,
            # Completes the multi-line `project_fields_to_location` arguments or elements.
            connectivity=merged_conn,
            num_points=merged_points.shape[0],
            projection_method="mean",
        )

        # Keeps the operation and its recovery path within the same exception boundary.
        try:
            merged_boundary_conditions['bc_flags'] = np.asarray(
                np.concatenate([zone.get_boundary_flags() for zone in zone_nodes], axis=0)
            ).reshape(-1)
            # Checks `` before entering this conditional branch.
            if len(merged_boundary_conditions['bc_flags']) != len(merged_points):
                merged_boundary_conditions['bc_flags'] = np.zeros(len(merged_points), dtype=int)
        except Exception:
            merged_boundary_conditions['bc_flags'] = np.zeros(len(merged_points), dtype=int)

        # Returns `` to the caller for the next stage.
        return {
            'points': merged_points,
            'conn': merged_conn,
            'element_type_ids': (
                # Completes the multi-line `np.asarray` arguments or elements.
                np.asarray(merged_element_type_ids, dtype=np.int32)
                if merged_element_type_ids
                else None
            ),
            # Completes the multi-line `collection` arguments or elements.
            'all_fields': projected_fields,
            'field_component_map': field_component_map,
            'field_location_map': field_location_map,
            'variable_catalog': variable_catalog,
            # Completes the multi-line `collection` arguments or elements.
            'effective_variable_sources': all_effective_sources,
            'projected_point_or_cell': projected_point_or_cell,
            'boundary_conditions': merged_boundary_conditions,
            'point_or_cell': projected_point_or_cell,
            # Completes the multi-line `collection` arguments or elements.
            'mesh_dimension': expected_mesh_dimension,
            'source_file': file_path,
        }

    def _resolve_load_variables(
        # Continues the surrounding statement or collection definition.
        self,
        requested_vars: Optional[List[str]],
    ) -> Optional[List[str]]:
        """优先使用已解析的有效变量，避免将无关 FlowSolution 场转为数组。"""
        task_cfg = self.config.get("task", {})
        # Stores the `effective_sources` for use by the remaining steps in this flow.
        effective_sources = task_cfg.get("effective_variable_sources") or {}
        if isinstance(effective_sources, dict) and effective_sources:
            raw_names = []
            for source in effective_sources.values():
                # Checks `str(source.get("format") or "cgns").lower() != "cgns"` before entering this conditional branch.
                if str(source.get("format") or "cgns").lower() != "cgns":
                    continue
                raw_name = str(source.get("array_name") or "")
                if raw_name and raw_name not in raw_names:
                    # Adds the current value to `raw_names` for later aggregation.
                    raw_names.append(raw_name)
            if raw_names:
                return raw_names

        effective_variables = list(task_cfg.get("effective_variables") or [])
        # Returns early when `effective_variables` makes the remaining work inapplicable.
        if effective_variables:
            return effective_variables

        component_map = task_cfg.get("field_component_map") or {}
        if requested_vars and isinstance(component_map, dict) and component_map:
            # Keeps the operation and its recovery path within the same exception boundary.
            try:
                resolved = resolve_effective_variables(
                    requested_vars,
                    component_map,
                    # Completes the multi-line `resolve_effective_variables` arguments or elements.
                    strict_missing=False,
                )
            except ValueError:
                return None
            # Returns early when `resolved` makes the remaining work inapplicable.
            if resolved:
                return list(resolved)
        return None

    def _collect_field_names_from_zones(self, zone_nodes, file_path: str) -> List[str]:
        """Collect field names from FlowSolution nodes without loading heavy arrays."""
        # Stores the `expected_names` for use by the remaining steps in this flow.
        expected_names = None

        for zone in zone_nodes:
            solution_node = zone.get_solution_node()
            if solution_node is None:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(
                    f"Zone '{zone.name}' 缺少FlowSolution节点，无法解析变量: {file_path}"
                )

            current_names = {
                # Completes the multi-line `solution_node.children.keys` arguments or elements.
                name for name in solution_node.children.keys()
                if name not in {"GridLocation", "PointRange"}
            }

            if not current_names:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(
                    f"Zone '{zone.name}' 中未找到任何可用场变量: {file_path}"
                )

            if expected_names is None:
                # Stores the `expected_names` for use by the remaining steps in this flow.
                expected_names = current_names
                continue

            if current_names != expected_names:
                missing = sorted(expected_names - current_names)
                # Stores the `extra` for use by the remaining steps in this flow.
                extra = sorted(current_names - expected_names)
                raise ValueError(
                    f"Zone字段集合不一致: zone={zone.name}, missing={missing}, extra={extra}"
                )

        # Validates `not expected_names` and raises `ValueError` when the input cannot be processed safely.
        if not expected_names:
            raise ValueError(f"单Base下未提取到任何变量字段: {file_path}")

        return sorted(expected_names)

    def _decode_cgns_char_matrix(self, values) -> List[str]:
        """将 CGNS C1 字符数组解码为字符串列表。"""
        # Stores the `arr` for use by the remaining steps in this flow.
        arr = np.asarray(values, dtype=np.int8)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            # Stores the `arr` for use by the remaining steps in this flow.
            arr = arr.reshape(arr.shape[0], -1)

        names: List[str] = []
        for row in arr:
            text = bytes(np.asarray(row, dtype=np.int8).reshape(-1).tolist()).decode(
                # Completes the multi-line `bytes(np.asarray(row, dtype=np.int8).reshape(-1).tolist()).decode` arguments or elements.
                "utf-8",
                errors="ignore",
            )
            text = text.rstrip("\x00 ").strip()
            # Checks `text` before entering this conditional branch.
            if text:
                names.append(text)
        return names

    def _decode_cgns_char_matrix_compat(self, values, value_shape=None) -> List[str]:
        """兼容 CGNS C1 字符数组的多种读入形态。"""
        # Returns early when `values is None` makes the remaining work inapplicable.
        if values is None:
            return []

        def _normalize_char_matrix(arr: np.ndarray) -> np.ndarray:
            if arr.ndim == 0:
                # Stores the `arr` for use by the remaining steps in this flow.
                arr = arr.reshape(1, 1)
            elif arr.ndim == 1:
                arr = arr.reshape(1, -1)
            elif arr.ndim > 2:
                # Stores the `arr` for use by the remaining steps in this flow.
                arr = arr.reshape(arr.shape[0], -1)

            if arr.ndim == 2 and arr.shape[0] >= 16 and arr.shape[1] < arr.shape[0]:
                arr = arr.transpose()
            return arr

        # Defines `_decode_rows` for  decode rows logic.
        def _decode_rows(arr: np.ndarray) -> List[str]:
            arr = _normalize_char_matrix(arr)

            names: List[str] = []
            for row in arr:
                # Stores the `text` for use by the remaining steps in this flow.
                text = bytes(np.asarray(row, dtype=np.int8).reshape(-1).tolist()).decode(
                    "utf-8",
                    errors="ignore",
                )
                # Stores the `text` for use by the remaining steps in this flow.
                text = text.rstrip("\x00 ").strip()
                if text:
                    names.append(text)
            return names

        # Checks `isinstance(values, str)` before entering this conditional branch.
        if isinstance(values, str):
            text = values.rstrip("\x00 ").strip()
            return [text] if text else []

        if isinstance(values, (bytes, bytearray, np.bytes_)):
            # Stores the `raw` for use by the remaining steps in this flow.
            raw = bytes(values)
            if value_shape:
                try:
                    arr = np.frombuffer(raw, dtype=np.int8).reshape(tuple(value_shape))
                    # Returns `_decode_rows(arr)` to the caller for the next stage.
                    return _decode_rows(arr)
                except Exception:
                    pass
            text = raw.decode("utf-8", errors="ignore").rstrip("\x00 ").strip()
            # Returns `[text] if text else []` to the caller for the next stage.
            return [text] if text else []

        arr = np.asarray(values)
        if arr.dtype.kind in {"S", "a"}:
            names: List[str] = []
            # Visits each item in `arr.reshape(-1)` and applies the current processing step.
            for item in arr.reshape(-1):
                text = bytes(item).decode("utf-8", errors="ignore").rstrip("\x00 ").strip()
                if text:
                    names.append(text)
            # Returns `names` to the caller for the next stage.
            return names

        return _decode_rows(np.asarray(values, dtype=np.int8))

    def _extract_flow_solution_pointer_names(self, zone_node) -> List[str]:
        """优先从 ZoneIterativeData/FlowSolutionPointers 中解析时序解顺序。"""
        if not getattr(zone_node, "children", None):
            # Returns `[]` to the caller for the next stage.
            return []

        zone_iter_node = zone_node.children.get("ZoneIterativeData")
        if zone_iter_node is None or not getattr(zone_iter_node, "children", None):
            return []

        # Reads `FlowSolutionPointers` from the mapping into the `pointer_node`.
        pointer_node = zone_iter_node.children.get("FlowSolutionPointers")
        if pointer_node is None:
            return []

        pointer_values = getattr(pointer_node, "data", None)
        # Returns early when `pointer_values is None` makes the remaining work inapplicable.
        if pointer_values is None:
            return []

        return self._decode_cgns_char_matrix_compat(
            pointer_values,
            # Completes the multi-line `getattr` arguments or elements.
            value_shape=getattr(pointer_node, "shape", None),
        )

    def _get_ordered_solution_nodes(
        self,
        # Continues the surrounding statement or collection definition.
        zone_node,
        max_time_steps: Optional[int] = None,
    ) -> List:
        """按时序顺序返回 Zone 下的 FlowSolution 节点列表。"""
        if str(self.mode).strip().lower() != "temporal":
            # Stores the `solution_nodes` for use by the remaining steps in this flow.
            solution_nodes = zone_node.get_node_by_label("FlowSolution_t", sort=False) or []
            if solution_nodes:
                ordered_nodes = sorted(solution_nodes, key=lambda item: item.name)
                if self._extract_flow_solution_pointer_names(zone_node):
                    # 带时序指针的稳态文件只读取主解节点，避免误拼接为时序数据。
                    # Returns `[ordered_nodes[0]]` to the caller for the next stage.
                    return [ordered_nodes[0]]
                # 未声明时序指针的多个解节点可能按 point/cell 拆分字段，需要保留以便合并。
                return ordered_nodes
            solution_node = zone_node.get_solution_node()
            if solution_node is None:
                # Returns `[]` to the caller for the next stage.
                return []
            return [solution_node]

        solution_nodes = zone_node.get_node_by_label("FlowSolution_t", sort=False) or []
        if not solution_nodes:
            # Returns `[]` to the caller for the next stage.
            return []

        solution_map = {node.name: node for node in solution_nodes}
        pointer_names = self._extract_flow_solution_pointer_names(zone_node)

        ordered_nodes = []
        # Stores the `used_names` for use by the remaining steps in this flow.
        used_names = set()
        for name in pointer_names:
            node = solution_map.get(name)
            if node is None or name in used_names:
                # Handles the data or control-flow case represented by this statement.
                continue
            ordered_nodes.append(node)
            used_names.add(name)

        for node in sorted(solution_nodes, key=lambda item: item.name):
            # Checks `node.name in used_names` before entering this conditional branch.
            if node.name in used_names:
                continue
            ordered_nodes.append(node)

        if max_time_steps is not None and int(max_time_steps) > 0:
            # Returns `ordered_nodes[: int(max_time_steps)]` to the caller for the next stage.
            return ordered_nodes[: int(max_time_steps)]
        return ordered_nodes

    def _extract_solution_fields_from_node(
        self,
        # Continues the surrounding statement or collection definition.
        solution_node,
        variables: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """从单个 FlowSolution 节点提取物理量字段。"""
        if solution_node is None or not getattr(solution_node, "children", None):
            # Returns `{}` to the caller for the next stage.
            return {}

        fields = {}
        for name, child in solution_node.children.items():
            if name in {"GridLocation", "PointRange"}:
                # Handles the data or control-flow case represented by this statement.
                continue
            if variables is not None and name not in variables:
                continue
            fields[name] = np.asarray(child.data)
        # Returns `fields` to the caller for the next stage.
        return fields

    def _resolve_solution_node_field_location(
        self,
        solution_node,
        # Completes the multi-line `collection` arguments or elements.
        fields_dict: Dict[str, np.ndarray],
        points: np.ndarray,
        conn,
    ) -> str:
        # Stores the `explicit_location` for use by the remaining steps in this flow.
        explicit_location = self._resolve_solution_node_grid_location(solution_node)
        if explicit_location in {"point", "cell"}:
            return explicit_location
        return self._detect_zone_field_location(fields_dict, points, conn)

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _split_temporal_solution_name_suffix(solution_name: str):
        normalized_name = str(solution_name or "").strip()
        for suffix, location in [("_Point", "point"), ("_Cell", "cell")]:
            # Returns early when `normalized_name.endswith(suffix)` makes the remaining work inapplicable.
            if normalized_name.endswith(suffix):
                return normalized_name[: -len(suffix)], location
        return normalized_name, None

    def _build_mixed_temporal_payload(
        # Continues the surrounding statement or collection definition.
        self,
        zone_node,
        payloads: List[Dict[str, object]],
        max_time_steps: Optional[int] = None,
    # Completes the multi-line `collection` arguments or elements.
    ) -> Optional[Dict[str, object]]:
        if len(payloads) <= 1:
            return None

        payload_map = {str(item["solution_name"]): item for item in payloads}
        # Stores the `pointer_names` for use by the remaining steps in this flow.
        pointer_names = self._extract_flow_solution_pointer_names(zone_node)
        ordered_base_names: List[str] = []

        for pointer_name in pointer_names:
            base_name, _ = self._split_temporal_solution_name_suffix(pointer_name)
            # Checks `base_name and base_name not in ordered_base_names` before entering this conditional branch.
            if base_name and base_name not in ordered_base_names:
                ordered_base_names.append(base_name)

        for item in payloads:
            base_name, _ = self._split_temporal_solution_name_suffix(item["solution_name"])
            # Checks `base_name and base_name not in ordered_base_names` before entering this conditional branch.
            if base_name and base_name not in ordered_base_names:
                ordered_base_names.append(base_name)

        if max_time_steps is not None and int(max_time_steps) > 0:
            ordered_base_names = ordered_base_names[: int(max_time_steps)]

        # Returns early when `len(ordered_base_names) <= 1` makes the remaining work inapplicable.
        if len(ordered_base_names) <= 1:
            return None

        per_step_payloads: List[Dict[str, object]] = []
        for base_name in ordered_base_names:
            # Initializes the `candidate_names` as a collection for values from this processing stage.
            candidate_names = [
                base_name,
                f"{base_name}_Point",
                f"{base_name}_Cell",
            # Completes the multi-line `collection` arguments or elements.
            ]
            matched_items = [
                payload_map[name]
                for name in candidate_names
                # Continues the surrounding statement or collection definition.
                if name in payload_map
            ]
            if not matched_items:
                continue

            # Initializes the `step_fields` as a collection for values from this processing stage.
            step_fields: Dict[str, np.ndarray] = {}
            step_location_map: Dict[str, str] = {}
            for item in matched_items:
                location = str(item["location"])
                # Visits each item in `item["fields"].items()` and applies the current processing step.
                for field_name, values in item["fields"].items():
                    if field_name in step_fields:
                        raise ValueError(
                            f"Zone '{zone_node.name}' 的时序字段 {field_name} 在同一时间步出现重复来源"
                        # Completes the multi-line `ValueError` arguments or elements.
                        )
                    arr = np.asarray(values)
                    if arr.ndim == 0:
                        arr = arr.reshape(1)
                    # Continues the surrounding statement or collection definition.
                    else:
                        arr = arr.flatten()
                    step_fields[field_name] = arr
                    step_location_map[field_name] = location

            # Adds the current value to `per_step_payloads` for later aggregation.
            per_step_payloads.append(
                {
                    "fields": step_fields,
                    "field_location_map": step_location_map,
                # Completes the multi-line `collection` arguments or elements.
                }
            )

        if len(per_step_payloads) <= 1:
            return None

        # Stores the `reference_fields` for use by the remaining steps in this flow.
        reference_fields = set(per_step_payloads[0]["fields"].keys())
        reference_location_map = dict(per_step_payloads[0]["field_location_map"])
        for item in per_step_payloads[1:]:
            if set(item["fields"].keys()) != reference_fields:
                # Returns `None` to the caller for the next stage.
                return None
            if dict(item["field_location_map"]) != reference_location_map:
                return None

        temporal_field_chunks: Dict[str, List[np.ndarray]] = {}
        # Visits each item in `per_step_payloads` and applies the current processing step.
        for item in per_step_payloads:
            for field_name, values in item["fields"].items():
                temporal_field_chunks.setdefault(field_name, []).append(
                    np.asarray(values).reshape(1, -1)
                # Completes the multi-line `temporal_field_chunks.setdefault(field_name, []).append` arguments or elements.
                )

        temporal_fields = {
            name: np.concatenate(chunks, axis=0)
            for name, chunks in temporal_field_chunks.items()
        # Continues the surrounding statement or collection definition.
        }
        return {
            "fields": temporal_fields,
            "field_location_map": reference_location_map,
        # Completes the multi-line `collection` arguments or elements.
        }

    def _extract_zone_fields_with_locations(
        self,
        zone_node,
        # Continues the surrounding statement or collection definition.
        points: np.ndarray,
        conn,
        variables: Optional[List[str]] = None,
        allow_missing_fields: bool = False,
        # Continues the surrounding statement or collection definition.
        max_time_steps: Optional[int] = None,
    ) -> Dict[str, object]:
        """提取单个 Zone 的字段及其原始位置。"""
        search_by_label = getattr(zone_node, "_search_node_by_label", None)
        if allow_missing_fields and callable(search_by_label):
            # Returns early when `not search_by_label("FlowSolution_t")` makes the remaining work inapplicable.
            if not search_by_label("FlowSolution_t"):
                return {"fields": {}, "field_location_map": {}}
        elif allow_missing_fields and not getattr(zone_node, "children", None):
            return {"fields": {}, "field_location_map": {}}

        # Stores the `solution_nodes` for use by the remaining steps in this flow.
        solution_nodes = self._get_ordered_solution_nodes(
            zone_node,
            max_time_steps=None,
        )
        # Checks `` before entering this conditional branch.
        if not solution_nodes and str(self.mode).strip().lower() != "temporal":
            solution_node = zone_node.get_solution_node()
            if solution_node is not None:
                solution_nodes = [solution_node]
        # Returns early when `not solution_nodes` makes the remaining work inapplicable.
        if not solution_nodes:
            return {"fields": {}, "field_location_map": {}}

        payloads = []
        for solution_node in solution_nodes:
            # Stores the `fields` for use by the remaining steps in this flow.
            fields = self._extract_solution_fields_from_node(
                solution_node,
                variables=variables,
            )
            # Checks `not fields` before entering this conditional branch.
            if not fields:
                continue
            location = self._resolve_solution_node_field_location(
                solution_node=solution_node,
                # Completes the multi-line `self._resolve_solution_node_field_location` arguments or elements.
                fields_dict=fields,
                points=points,
                conn=conn,
            )
            # Adds the current value to `payloads` for later aggregation.
            payloads.append(
                {
                    "solution_node": solution_node,
                    "solution_name": solution_node.name,
                    # Completes the multi-line `collection` arguments or elements.
                    "fields": fields,
                    "location": location,
                }
            )

        # Returns early when `not payloads` makes the remaining work inapplicable.
        if not payloads:
            return {"fields": {}, "field_location_map": {}}

        treat_as_temporal = False
        if str(self.mode).strip().lower() == "temporal" and len(payloads) > 1:
            # Stores the `first_names` for use by the remaining steps in this flow.
            first_names = set(payloads[0]["fields"].keys())
            first_location = payloads[0]["location"]
            treat_as_temporal = all(
                set(item["fields"].keys()) == first_names and item["location"] == first_location
                # Completes the multi-line `all` arguments or elements.
                for item in payloads[1:]
            )

        if treat_as_temporal:
            temporal_field_chunks: Dict[str, List[np.ndarray]] = {}
            # Visits each item in `payloads[: int(max_time_steps)] if max_time_steps else payloads` and applies the current processing step.
            for item in payloads[: int(max_time_steps)] if max_time_steps else payloads:
                for name, data in item["fields"].items():
                    arr = np.asarray(data)
                    if arr.ndim == 0:
                        # Stores the `arr` for use by the remaining steps in this flow.
                        arr = arr.reshape(1)
                    temporal_field_chunks.setdefault(name, []).append(arr.reshape(1, -1))

            temporal_fields = {
                name: np.concatenate(chunks, axis=0)
                # Completes the multi-line `collection` arguments or elements.
                for name, chunks in temporal_field_chunks.items()
            }
            return {
                "fields": temporal_fields,
                # Completes the multi-line `collection` arguments or elements.
                "field_location_map": {
                    name: str(payloads[0]["location"])
                    for name in temporal_fields.keys()
                },
            # Completes the multi-line `collection` arguments or elements.
            }

        if str(self.mode).strip().lower() == "temporal":
            mixed_temporal_payload = self._build_mixed_temporal_payload(
                zone_node=zone_node,
                # Completes the multi-line `self._build_mixed_temporal_payload` arguments or elements.
                payloads=payloads,
                max_time_steps=max_time_steps,
            )
            if mixed_temporal_payload is not None:
                # Returns `mixed_temporal_payload` to the caller for the next stage.
                return mixed_temporal_payload

        merged_fields: Dict[str, np.ndarray] = {}
        field_location_map: Dict[str, str] = {}
        for item in payloads:
            # Converts the `location` to `str` so later checks use a consistent type.
            location = str(item["location"])
            for name, data in item["fields"].items():
                if name in merged_fields:
                    raise ValueError(
                        # Completes the multi-line `ValueError` arguments or elements.
                        f"Zone '{zone_node.name}' 中字段 {name} 同时出现在多个 FlowSolution 组，当前无法唯一判定来源"
                    )
                arr = np.asarray(data)
                if arr.ndim == 0:
                    # Stores the `arr` for use by the remaining steps in this flow.
                    arr = arr.reshape(1)
                else:
                    arr = arr.flatten()
                merged_fields[name] = arr
                # Stores the `field_location_map[name]` for use by the remaining steps in this flow.
                field_location_map[name] = location

        return {
            "fields": merged_fields,
            "field_location_map": field_location_map,
        # Completes the multi-line `collection` arguments or elements.
        }

    def _resolve_variable_manifest_legacy(
        self,
        file_path: str,
        # Continues the surrounding statement or collection definition.
        logical_variables: List[str],
        sample_order: Optional[int] = None,
    ) -> Dict[str, object]:
        """兼容保留的旧实现，已由下方统一入口替代。"""
        return self.resolve_variable_manifest(
            # Completes the multi-line `self.resolve_variable_manifest` arguments or elements.
            file_path=file_path,
            logical_variables=logical_variables,
            sample_order=sample_order,
        )

    # 读取全字段样本并解析训练变量，同时识别运动网格位移信息。
    def resolve_variable_manifest(
        self,
        file_path: str,
        logical_variables: List[str],
        # Continues the surrounding statement or collection definition.
        sample_order: Optional[int] = None,
    ) -> Dict[str, object]:
        """读取全字段样本并解析训练变量，同时识别运动网格位移信息。"""
        path_obj = Path(file_path)
        if self.mode == "temporal" and (
            # Completes the multi-line `path_obj.is_dir` arguments or elements.
            path_obj.is_dir() or is_temporal_manifest_path(file_path)
        ):
            sequence_files = list_temporal_sequence_files(
                self,
                # Completes the multi-line `list_temporal_sequence_files` arguments or elements.
                file_path,
                max_time_steps=None,
            )
            step_data = (
                # Completes the multi-line `read_single_sample` arguments or elements.
                read_single_sample(
                    self,
                    path,
                    sample_order,
                    # Completes the multi-line `read_single_sample` arguments or elements.
                    allow_missing_fields=True,
                    max_time_steps=None,
                )
                for path in sequence_files
            # Continues the surrounding statement or collection definition.
            )
            data = aggregate_field_temporal_sequence(
                step_data,
                file_path,
                # Completes the multi-line `len` arguments or elements.
                expected_time_steps=len(sequence_files),
            )
        elif path_obj.is_file():
            manifest = self._resolve_variable_manifest_from_h5_mixed(
                # Completes the multi-line `self._resolve_variable_manifest_from_h5_mixed` arguments or elements.
                file_path=file_path,
                logical_variables=logical_variables,
            )
            field_component_map = manifest.get('field_component_map') or {}
            # Reads `mesh_dimension` from the mapping into the `mesh_dimension`.
            mesh_dimension = manifest.get('mesh_dimension')
            coord_dim = int(mesh_dimension) if mesh_dimension is not None else 3
            displacement_logical_name = infer_displacement_logical_name(
                field_component_map,
                # Completes the multi-line `infer_displacement_logical_name` arguments or elements.
                coord_dim=coord_dim,
            )
            displacement_effective_variables = infer_displacement_effective_variables(
                field_component_map,
                # Completes the multi-line `infer_displacement_effective_variables` arguments or elements.
                displacement_logical_name,
                coord_dim=coord_dim,
            )
            manifest['geometry_regime'] = 'static_geometry'
            # Initializes the `manifest['geometry_motion']` as a collection for values from this processing stage.
            manifest['geometry_motion'] = {}
            manifest['displacement_logical_variable'] = displacement_logical_name
            manifest['displacement_effective_variables'] = list(displacement_effective_variables)
            return manifest
        # Continues the surrounding statement or collection definition.
        else:
            data = self.read(
                file_path,
                variables=[],
                # Completes the multi-line `self.read` arguments or elements.
                sample_order=sample_order,
                allow_missing_fields=True,
            )
        data_dict = data.get_data()
        # Stores the `field_component_map` for use by the remaining steps in this flow.
        field_component_map = data_dict.get('FieldComponentMap') or {}
        if not isinstance(field_component_map, dict) or not field_component_map:
            raise ValueError("样本中未解析出 FieldComponentMap，无法解析变量")

        variable_catalog = data_dict.get('VariableCatalog') or {}
        # Checks `isinstance(variable_catalog, dict) and variable_catalog` before entering this conditional branch.
        if isinstance(variable_catalog, dict) and variable_catalog:
            (
                field_component_map,
                effective_variables,
                # Completes the multi-line `collection` arguments or elements.
                effective_variable_sources,
                field_location_map,
                _,
            ) = select_catalog_variables(variable_catalog, logical_variables)
        # Continues the surrounding statement or collection definition.
        else:
            effective_variables = resolve_effective_variables(
                logical_variables,
                field_component_map,
                # Completes the multi-line `resolve_effective_variables` arguments or elements.
                strict_missing=True,
            )
            effective_variable_sources = {
                str(name): dict(source)
                # Completes the multi-line `collection` arguments or elements.
                for name, source in (data_dict.get('EffectiveVariableSources') or {}).items()
                if str(name) in effective_variables
            }
            field_location_map = {
                # Completes the multi-line `str` arguments or elements.
                str(k): str(v)
                for k, v in (data_dict.get('FieldLocationMap') or {}).items()
                if str(k) in effective_variables
            }
        # Validates `not effective_variables` and raises `ValueError` when the input cannot be processed safely.
        if not effective_variables:
            raise ValueError("样本中未解析出有效变量，无法解析变量")

        mesh_dimension = data_dict.get('MeshDimension')
        coord_dim = int(mesh_dimension) if mesh_dimension is not None else 3
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
            'geometry_regime': data.get_geometry_regime() or 'static_geometry',
            # Completes the multi-line `collection` arguments or elements.
            'geometry_motion': geometry_motion,
            'displacement_logical_variable': displacement_logical_name,
            'displacement_effective_variables': list(displacement_effective_variables),
        }

    # 读取CGNS文件，返回BaseData
    def read(
        self,
        file_path: str,
        variables: Optional[List[str]] = None,
        # Continues the surrounding statement or collection definition.
        sample_order: Optional[int] = None,
        allow_missing_fields: bool = False,
        max_time_steps: Optional[int] = None,
    ) -> BaseData:
        """读取CGNS文件，返回BaseData
        
        Args:
            file_path: CGNS文件路径
            variables: 所需的物理量名称列表，None表示读取所有物理量
            sample_order: 样本编号，None表示自动根据文件路径匹配
            
        Returns:
            解析后的BaseData
        """
        # 使用io_utils中的CGNSReader读取CGNS文件
        # Checks `` before entering this conditional branch.
        if self.mode == "temporal" and (
            Path(file_path).is_dir() or is_temporal_manifest_path(file_path)
        ):
            sequence_files = list_temporal_sequence_files(
                # Completes the multi-line `list_temporal_sequence_files` arguments or elements.
                self,
                file_path,
                max_time_steps=max_time_steps,
            )
            # Stores the `step_data` for use by the remaining steps in this flow.
            step_data = (
                read_single_sample(
                    self,
                    path,
                    # Completes the multi-line `read_single_sample` arguments or elements.
                    sample_order,
                    allow_missing_fields=allow_missing_fields,
                    max_time_steps=None,
                )
                # Continues the surrounding statement or collection definition.
                for path in sequence_files
            )
            return aggregate_field_temporal_sequence(
                step_data,
                # Completes the multi-line `aggregate_field_temporal_sequence` arguments or elements.
                file_path,
                expected_time_steps=len(sequence_files),
            )

        io_reader = IOCGNSReader(file_path)
        # Stores the `cgns_data` for use by the remaining steps in this flow.
        cgns_data = io_reader.read()
        zone_nodes = self._get_zone_nodes(cgns_data, file_path)
        self._family_bc_types = self._extract_family_bc_types(cgns_data, file_path)
        logical_variables = self.variables if variables is None else variables
        # Stores the `read_all_fields` for use by the remaining steps in this flow.
        read_all_fields = variables is not None and len(variables) == 0
        projection_requested_vars = None if read_all_fields else logical_variables
        load_variables = (
            None
            # Continues the surrounding statement or collection definition.
            if read_all_fields
            else self._resolve_load_variables(projection_requested_vars)
        )

        merged = self._merge_zone_data_mixed(
            # Completes the multi-line `self._merge_zone_data_mixed` arguments or elements.
            zone_nodes,
            file_path,
            requested_vars=projection_requested_vars,
            load_variables=load_variables,
            # Completes the multi-line `self._merge_zone_data_mixed` arguments or elements.
            allow_missing_fields=allow_missing_fields,
            max_time_steps=max_time_steps,
        )
        points = merged['points']
        # Selects the required item from `merged` as the `conn`.
        conn = merged['conn']
        element_type_ids = merged.get('element_type_ids')

        # 计算单元中心坐标
        cell_center_coords = self._get_cell_centers(points, conn, element_type_ids=element_type_ids)

        # 使用io_utils提供的方法提取全部物理量，再按逻辑变量解析为有效分量变量
        all_fields = merged['all_fields']
        # Reads `field_component_map` from the mapping into the `configured_component_map` with its declared fallback.
        configured_component_map = self.config.get('task', {}).get('field_component_map', {})
        configured_catalog = self.config.get('task', {}).get('variable_catalog', {})
        variable_catalog = merged.get('variable_catalog') or configured_catalog or {}
        if isinstance(variable_catalog, dict) and variable_catalog:
            # Checks `read_all_fields` before entering this conditional branch.
            if read_all_fields:
                field_component_map = catalog_component_map(variable_catalog)
                effective_variable_sources = catalog_sources_by_runtime_key(variable_catalog)
                effective_variables = list(effective_variable_sources)
                # Builds the `field_location_map` using the comprehension's filter and mapping rules.
                field_location_map = {
                    runtime_key: str(source.get('location'))
                    for runtime_key, source in effective_variable_sources.items()
                }
            # Continues the surrounding statement or collection definition.
            else:
                (
                    field_component_map,
                    effective_variables,
                    # Completes the multi-line `collection` arguments or elements.
                    effective_variable_sources,
                    field_location_map,
                    _,
                ) = select_catalog_variables(
                    # Completes the multi-line `select_catalog_variables` arguments or elements.
                    variable_catalog,
                    logical_variables,
                    strict_missing=not allow_missing_fields,
                )
        # Checks `all_fields` before entering this conditional branch.
        elif all_fields:
            field_component_map = {k: [k] for k in all_fields}
            effective_variables = (
                list(all_fields.keys())
                # Continues the surrounding statement or collection definition.
                if read_all_fields
                else resolve_effective_variables(logical_variables, field_component_map)
            )
            effective_variable_sources = {
                # Completes the multi-line `str` arguments or elements.
                str(name): dict(source)
                for name, source in (
                    self.config.get('task', {}).get('effective_variable_sources', {}) or {}
                ).items()
                # Completes the multi-line `str` arguments or elements.
                if str(name) in effective_variables
            }
            field_location_map = {
                str(k): str(v)
                # Completes the multi-line `collection` arguments or elements.
                for k, v in (merged.get('field_location_map') or {}).items()
                if str(k) in effective_variables
            }
        else:
            # Stores the `field_component_map` for use by the remaining steps in this flow.
            field_component_map = (
                {k: list(v) for k, v in configured_component_map.items()}
                if isinstance(configured_component_map, dict)
                else {}
            # Continues the surrounding statement or collection definition.
            )
            effective_variables = list(all_fields.keys()) if read_all_fields else list(
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
            field_location_map = {
                # Completes the multi-line `str` arguments or elements.
                str(k): str(v)
                for k, v in (
                    self.config.get('task', {}).get('field_location_map', {}) or {}
                ).items()
                # Completes the multi-line `str` arguments or elements.
                if str(k) in effective_variables
            }
        fields_dict = {k: all_fields[k] for k in effective_variables if k in all_fields}

        if not fields_dict and not allow_missing_fields:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(
                f"有效变量为空，logical_variables={logical_variables}, "
                f"可用字段={sorted(all_fields.keys())}"
            )

        # 多Zone合并后由每个Zone一致性判定得出point_or_cell
        # Stores the `point_or_cell` on the instance so other methods can reuse it.
        self.point_or_cell = merged['point_or_cell']

        # 冗余校验，确保最终字段维度仍然匹配
        if fields_dict:
            self.set_point_or_cell(fields_dict, points, cell_center_coords)

        boundary_conditions = merged['boundary_conditions']
        # Checks `not field_location_map` before entering this conditional branch.
        if not field_location_map:
            field_location_map = build_component_location_map(
                field_component_map,
                self.point_or_cell,
            # Completes the multi-line `build_component_location_map` arguments or elements.
            )
        projected_point_or_cell = merged.get('projected_point_or_cell') or self.point_or_cell
        
        # 构建BaseData所需的数据结构
        field_data = {
            'PointCoordinates': points,  # 节点坐标数组，形状为(N, 3)
            # Completes the multi-line `collection` arguments or elements.
            'CellCoordinates': cell_center_coords,  # 单元中心坐标数组
            'ElementConnectivity': conn,    # 单元连接数组
            'Fields': fields_dict,  # 物理量数据字典
            'LogicalVariables': list(logical_variables),
            # Completes the multi-line `list` arguments or elements.
            'EffectiveVariables': list(effective_variables),
            'FieldComponentMap': field_component_map,
            'VariableCatalog': variable_catalog,
            'EffectiveVariableSources': effective_variable_sources,
            # Completes the multi-line `collection` arguments or elements.
            'FieldLocationMap': field_location_map,
            'ProjectedPointOrCell': projected_point_or_cell,
            'MixedFieldPolicy': 'majority_side',
            'ProjectionMethod': 'mean',
            # Completes the multi-line `collection` arguments or elements.
            'BoundaryConditions': boundary_conditions,  # 边界条件字典
            'MaterialIDs': None,  # 材料ID数组（可选，目前为None）
            'SourceFile': merged['source_file'],
        }
        # Stores the `field_data['ElementTypeIds']` for use by the remaining steps in this flow.
        field_data['ElementTypeIds'] = element_type_ids
        field_data['MeshDimension'] = merged.get('mesh_dimension')
        
        # 获取样本信息
        sample_info = {'sample_order': sample_order}
        if self.sample_map:
            # Checks `sample_order is not None` before entering this conditional branch.
            if sample_order is not None:
                # 根据样本编号获取完整的样本信息
                full_sample_info = self.sample_map.get_sample_info(sample_order)
                if full_sample_info:
                    sample_info.update(full_sample_info)
            # Continues the surrounding statement or collection definition.
            else:
                for order in self.sample_map.get_all_samples():
                    # 使用get_file_path方法获取文件路径
                    sample_file_path = self.sample_map.get_file_path(order)
                    if not sample_file_path:
                        # Handles the data or control-flow case represented by this statement.
                        continue
                    if path_endswith(file_path, sample_file_path):
                        full_sample_info = self.sample_map.get_sample_info(order)
                        if full_sample_info:
                            # Adds the current value to `sample_info` for later aggregation.
                            sample_info.update(full_sample_info)
                            break
        
        # 创建并返回BaseData对象
        return BaseData(
            data=field_data, 
            # Completes the multi-line `BaseData` arguments or elements.
            task_type=self.task_type, 
            mode=self.mode,
            point_or_cell=self.point_or_cell,
            sample_map=sample_info
        # Completes the multi-line `BaseData` arguments or elements.
        )
    
    def _get_cell_centers(self, points, conn, element_type_ids: Optional[np.ndarray] = None):
        """根据网格数据计算单元体心"""
        
        points = np.asarray(points, dtype=float)
        normalized_type_ids = None
        # Checks `element_type_ids is not None` before entering this conditional branch.
        if element_type_ids is not None:
            normalized_type_ids = np.asarray(element_type_ids, dtype=np.int32).reshape(-1)

        cell_centers = []
        for cell_idx, cell in enumerate(conn):
            # Stores the `elem_type` for use by the remaining steps in this flow.
            elem_type = None
            if normalized_type_ids is not None and cell_idx < len(normalized_type_ids):
                elem_type = int(normalized_type_ids[cell_idx])
            cell_nodes = cgnsmesh.extract_corner_nodes(
                # Completes the multi-line `cgnsmesh.extract_corner_nodes` arguments or elements.
                cell,
                elem_type=elem_type,
                prefer_volume=True,
            )
            # Initializes the `pts` as a collection for values from this processing stage.
            pts = []
            for idx in cell_nodes:
                try:
                    pts.append(points[int(idx) - 1])
                # Continues the surrounding statement or collection definition.
                except Exception:
                    continue
            if pts:
                pts = np.asarray(pts, dtype=float)
                # Adds the current value to `cell_centers` for later aggregation.
                cell_centers.append(np.mean(pts, axis=0))
            else:
                cell_centers.append([np.nan, np.nan, np.nan])
        cell_centers = np.asarray(cell_centers, dtype=float)

        # Returns `cell_centers` to the caller for the next stage.
        return cell_centers
    
    def _extract_fields(
        self,
        zone_node,
        # Continues the surrounding statement or collection definition.
        variables: Optional[List[str]] = None,
        allow_missing_fields: bool = False,
        max_time_steps: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """从Zone节点提取物理量数据，利用io_utils提供的方法
        
        Args:
            zone_node: Zone节点
            variables: 所需的物理量名称列表
            
        Returns:
            物理量字典
        """
        # 使用get_solution_datadict方法获取所有物理量数据
        # Stores the `search_by_label` for use by the remaining steps in this flow.
        search_by_label = getattr(zone_node, "_search_node_by_label", None)
        if allow_missing_fields and callable(search_by_label):
            if not search_by_label("FlowSolution_t"):
                return {}
        # Returns early when `` makes the remaining work inapplicable.
        elif allow_missing_fields and not getattr(zone_node, "children", None):
            return {}

        solution_nodes = self._get_ordered_solution_nodes(
            zone_node,
            # Completes the multi-line `self._get_ordered_solution_nodes` arguments or elements.
            max_time_steps=max_time_steps,
        )
        if not solution_nodes:
            return {}

        # Checks `len(solution_nodes) > 1` before entering this conditional branch.
        if len(solution_nodes) > 1:
            temporal_field_chunks: Dict[str, List[np.ndarray]] = {}
            expected_names = None
            for solution_node in solution_nodes:
                # Stores the `step_fields` for use by the remaining steps in this flow.
                step_fields = self._extract_solution_fields_from_node(
                    solution_node,
                    variables=variables,
                )
                # Stores the `current_names` for use by the remaining steps in this flow.
                current_names = set(step_fields.keys())
                if expected_names is None:
                    expected_names = current_names
                elif current_names != expected_names:
                    # Stores the `missing` for use by the remaining steps in this flow.
                    missing = sorted(expected_names - current_names)
                    extra = sorted(current_names - expected_names)
                    raise ValueError(
                        f"Zone '{zone_node.name}' 的时序 FlowSolution 字段集合不一致, missing={missing}, extra={extra}"
                    # Completes the multi-line `ValueError` arguments or elements.
                    )

                for name, data in step_fields.items():
                    arr = np.asarray(data)
                    if arr.ndim == 0:
                        # Stores the `arr` for use by the remaining steps in this flow.
                        arr = arr.reshape(1)
                    temporal_field_chunks.setdefault(name, []).append(arr.reshape(1, -1))

            temporal_fields = {}
            for name, chunks in temporal_field_chunks.items():
                # Stores the `temporal_fields[name]` for use by the remaining steps in this flow.
                temporal_fields[name] = np.concatenate(chunks, axis=0)
            return temporal_fields

        fields_dict = self._extract_solution_fields_from_node(
            solution_nodes[0],
            # Completes the multi-line `self._extract_solution_fields_from_node` arguments or elements.
            variables=variables,
        )
        
        if fields_dict is None:
            return {}
        
        # 过滤掉不需要的字段并根据变量列表筛选
        # Initializes the `filtered_fields` as a collection for values from this processing stage.
        filtered_fields = {}
        for name, data in fields_dict.items():
            # 忽略GridLocation和PointRange字段
            if name in ["GridLocation", "PointRange"]:
                continue
            # 如果指定了变量列表，则只保留这些变量
            # Checks `variables is None or name in variables` before entering this conditional branch.
            if variables is None or name in variables:
                arr = np.asarray(data)
                if (
                    max_time_steps is not None
                    # Completes the multi-line `int` arguments or elements.
                    and int(max_time_steps) > 0
                    and arr.ndim >= 2
                ):
                    filtered_fields[name] = arr[: int(max_time_steps)]
                # Continues the surrounding statement or collection definition.
                else:
                    filtered_fields[name] = arr

        fields_dict = filtered_fields

        # 对数据进行flatten操作
        for name, data in fields_dict.items():
            # Checks `isinstance(data, np.ndarray)` before entering this conditional branch.
            if isinstance(data, np.ndarray):
                if max_time_steps is not None and data.ndim >= 2:
                    fields_dict[name] = data
                else:
                    # Stores the `fields_dict[name]` for use by the remaining steps in this flow.
                    fields_dict[name] = data.flatten()
        
        return fields_dict
    
    def set_point_or_cell(self, fields_dict: Dict, points: np.ndarray, 
                          cell_coords: Optional[np.ndarray]) -> None:
        """根据场数据维度自动检测point_or_cell参数
        
        检查Fields中的第一个场数据的维度：
        - 如果与PointCoordinates维度匹配，设置为'point'
        - 如果与CellCoordinates维度匹配，设置为'cell'
        - 否则抛出异常
        
        Args:
            fields_dict: 物理量字典
            points: 节点坐标数组，形状为(N, 3)
            cell_coords: 单元中心坐标数组，形状为(M, 3)或None
            
        Raises:
            ValueError: 如果fields_dict为空或场数据维度与坐标数据不匹配
        """
        # Validates `not fields_dict` and raises `ValueError` when the input cannot be processed safely.
        if not fields_dict:
            raise ValueError("Fields dictionary is empty")
        
        # 获取第一个场数据
        first_field_name = next(iter(fields_dict))
        first_field_data = fields_dict[first_field_name]
        
        # Stores the `first_field_arr` for use by the remaining steps in this flow.
        first_field_arr = np.asarray(first_field_data)
        if first_field_arr.ndim >= 2:
            field_size = int(first_field_arr.shape[-1])
        else:
            # Stores the `field_size` for use by the remaining steps in this flow.
            field_size = len(first_field_arr)
        point_size = len(points)
        cell_size = len(cell_coords) if cell_coords is not None else 0
        
        # 检查维度匹配
        if field_size == point_size:
            # Stores the `new_point_or_cell` for use by the remaining steps in this flow.
            new_point_or_cell = 'point'
        elif field_size == cell_size and cell_coords is not None:
            new_point_or_cell = 'cell'
        else:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(
                f"Field '{first_field_name}' dimension {field_size} doesn't match "
                f"PointCoordinates ({point_size}) or CellCoordinates ({cell_size})"
            )
        
        # 更新self.point_or_cell
        # Stores the `point_or_cell` on the instance so other methods can reuse it.
        self.point_or_cell = new_point_or_cell
        



