from __future__ import annotations

import logging
import os
# Imports `shutil` for reuse in this module.
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
# Imports `numpy` for reuse in this module.
import numpy as np

from tfai.io_utils.io_cgns import CGNSReader as IOCGNSReader
from tfai.utils.field_projection import normalize_field_location, project_field_values
from tfai.utils.logger import Logger


# 负责以 clean_display 方式将预测结果写回 CGNS 文件。
class CGNSPredictionWriter:
    """负责以 clean_display 方式将预测结果写回 CGNS 文件。"""

    SOLUTION_GROUP_NAME = "FlowSolution"
    CGNS_CHAR_WIDTH = 32
    TIME_ACCURATE_SIMULATION_TYPE = "TimeAccurate"

    # Defines `__init__` for   init   logic.
    def __init__(
        self,
        config: Dict,
        logger: Optional[logging.Logger] = None,
        # Continues the surrounding statement or collection definition.
        target_base_name: Optional[str] = None,
    ):
        self.config = config
        self.logger = logger
        # Stores the `target_base_name` on the instance so other methods can reuse it.
        self.target_base_name = target_base_name

    def _log(self, level: str, message: str) -> None:
        logger = self.logger
        if logger is None:
            # Stores the run logger for use by the remaining steps in this flow.
            logger = Logger.get_first_configured_logger(
                ["predict", "evaluate", "train"],
                fallback_name=__name__,
            )
        # Stores the `log_fn` for use by the remaining steps in this flow.
        log_fn = getattr(logger, level, logger.info)
        log_fn(message)

    @staticmethod
    def _decode_cgns_attr(raw) -> str:
        # Returns early when `raw is None` makes the remaining work inapplicable.
        if raw is None:
            return ""
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="ignore").rstrip("\x00")
        # Returns early when `hasattr(raw, "tobytes")` makes the remaining work inapplicable.
        if hasattr(raw, "tobytes"):
            return raw.tobytes().decode("utf-8", errors="ignore").rstrip("\x00")
        return str(raw).rstrip("\x00")

    @staticmethod
    # Defines `_encode_cgns_attr` for  encode cgns attr logic.
    def _encode_cgns_attr(value: str, size: int) -> np.bytes_:
        return np.bytes_(bytes(value, "utf-8").ljust(size, b"\x00"))

    def _set_group_attrs(
        self,
        # Continues the surrounding statement or collection definition.
        group: h5py.Group,
        name: str,
        label: str,
        type_code: str,
        # Continues the surrounding statement or collection definition.
        flags: Optional[np.ndarray] = None,
    ) -> None:
        group.attrs["name"] = self._encode_cgns_attr(name, 33)
        group.attrs["label"] = self._encode_cgns_attr(label, 33)
        # Stores the `attrs['type']` for use by the remaining steps in this flow.
        group.attrs["type"] = self._encode_cgns_attr(type_code, 3)
        if flags is not None:
            group.attrs["flags"] = np.asarray(flags, dtype=np.int32)

    @staticmethod
    # 构造单行 C1 字符串 payload，默认补 NUL 终止。
    def _build_c1_payload(text: str, min_width: int = 0, null_terminated: bool = False) -> np.ndarray:
        """构造单行 C1 字符串 payload，默认补 NUL 终止。"""
        encoded = text.encode("utf-8")
        payload_width = max(int(min_width), len(encoded) + (1 if null_terminated else 0))
        payload = np.full((payload_width,), ord(" "), dtype=np.int8)
        # Checks `encoded` before entering this conditional branch.
        if encoded:
            payload[: len(encoded)] = np.frombuffer(encoded, dtype=np.int8)
        if null_terminated and len(encoded) < payload_width:
            payload[len(encoded)] = 0
        # Returns `payload` to the caller for the next stage.
        return payload

    def _build_c1_row_payload(
        self,
        texts: List[str],
        # Continues the surrounding statement or collection definition.
        min_width: int = CGNS_CHAR_WIDTH,
        null_terminated: bool = False,
    ) -> np.ndarray:
        """构造多行 C1 字符串 payload，每行独立补 NUL 终止。"""
        normalized = [str(text).strip() for text in texts if str(text).strip()]
        # Checks `not normalized` before entering this conditional branch.
        if not normalized:
            normalized = [self.SOLUTION_GROUP_NAME]

        width = max(
            int(min_width),
            # Completes the multi-line `max` arguments or elements.
            max(len(text.encode("utf-8")) + (1 if null_terminated else 0) for text in normalized),
        )
        payload = np.full((len(normalized), width), ord(" "), dtype=np.int8)
        for row_idx, text in enumerate(normalized):
            # Stores the `row_payload` for use by the remaining steps in this flow.
            row_payload = self._build_c1_payload(
                text=text,
                min_width=width,
                null_terminated=null_terminated,
            # Completes the multi-line `self._build_c1_payload` arguments or elements.
            )
            payload[row_idx, :] = row_payload
        return payload

    def _ensure_child_group(
        # Continues the surrounding statement or collection definition.
        self,
        parent: h5py.Group,
        key: str,
        name: str,
        # Continues the surrounding statement or collection definition.
        label: str,
        type_code: str,
        flags: Optional[np.ndarray] = None,
    ) -> h5py.Group:
        # Checks `key in parent and isinstance(parent[key], h5py.Group)` before entering this conditional branch.
        if key in parent and isinstance(parent[key], h5py.Group):
            group = parent[key]
        else:
            group = parent.create_group(key)
        # Calls `self._set_group_attrs` to perform the operation required at this stage.
        self._set_group_attrs(group, name, label, type_code, flags)
        return group

    def _find_child_groups_by_label(self, parent: h5py.Group, target_label: str) -> List[h5py.Group]:
        matches: List[h5py.Group] = []
        # Visits each item in `parent.keys()` and applies the current processing step.
        for key in parent.keys():
            obj = parent[key]
            if not isinstance(obj, h5py.Group):
                continue
            # Checks `` before entering this conditional branch.
            if self._decode_cgns_attr(obj.attrs.get("label", b"")) == target_label:
                matches.append(obj)
        return matches

    def _find_child_group_by_label_and_name(
        # Continues the surrounding statement or collection definition.
        self,
        parent: h5py.Group,
        target_label: str,
        target_name: Optional[str] = None,
    # Continues the surrounding statement or collection definition.
    ) -> Optional[h5py.Group]:
        for group in self._find_child_groups_by_label(parent, target_label):
            if target_name is None:
                return group
            # Returns early when `` makes the remaining work inapplicable.
            if self._decode_cgns_attr(group.attrs.get("name", b"")) == target_name:
                return group
        return None

    def _delete_child_groups_by_label(self, parent: h5py.Group, target_label: str) -> None:
        # Initializes the `delete_keys` as a collection for values from this processing stage.
        delete_keys: List[str] = []
        for key in parent.keys():
            obj = parent[key]
            if not isinstance(obj, h5py.Group):
                # Handles the data or control-flow case represented by this statement.
                continue
            if self._decode_cgns_attr(obj.attrs.get("label", b"")) == target_label:
                delete_keys.append(key)
        for key in delete_keys:
            # Handles the data or control-flow case represented by this statement.
            del parent[key]

    def _write_cgns_char_dataset(self, group: h5py.Group, payload: np.ndarray) -> None:
        arr = np.asarray(payload, dtype=np.int8)
        if " data" in group:
            # Handles the data or control-flow case represented by this statement.
            del group[" data"]

        if arr.ndim <= 1:
            group.create_dataset(" data", data=arr)
            return

        # Stores the `expected_shape` for use by the remaining steps in this flow.
        expected_shape = tuple(int(v) for v in arr.shape[::-1])
        ds = group.create_dataset(" data", shape=expected_shape, dtype=arr.dtype)
        ds[...] = arr.transpose()

    def _ensure_simulation_type_group(
        # Continues the surrounding statement or collection definition.
        self,
        base_group: h5py.Group,
        simulation_type: str = TIME_ACCURATE_SIMULATION_TYPE,
    ) -> None:
        # Stores the `simulation_group` for use by the remaining steps in this flow.
        simulation_group = self._ensure_child_group(
            parent=base_group,
            key="SimulationType",
            name="SimulationType",
            # Completes the multi-line `self._ensure_child_group` arguments or elements.
            label="SimulationType_t",
            type_code="C1",
            flags=np.array([1], dtype=np.int32),
        )
        # Stores the `payload` for use by the remaining steps in this flow.
        payload = self._build_c1_payload(simulation_type)
        self._write_cgns_char_dataset(simulation_group, payload)

    def _remove_single_step_metadata(self, base_group: h5py.Group) -> None:
        self._delete_child_groups_by_label(base_group, "BaseIterativeData_t")
        # Calls `self._delete_child_groups_by_label` to perform the operation required at this stage.
        self._delete_child_groups_by_label(base_group, "SimulationType_t")

    def _has_child_group_with_label(self, parent: h5py.Group, target_label: str) -> bool:
        return bool(self._find_child_groups_by_label(parent, target_label))

    def _select_target_base_group(
        # Continues the surrounding statement or collection definition.
        self,
        base_groups: List[h5py.Group],
        sample_id: int,
    ) -> h5py.Group:
        # Stores the `preferred_base` for use by the remaining steps in this flow.
        preferred_base = self.target_base_name
        if preferred_base:
            for base_group in base_groups:
                base_name = self._decode_cgns_attr(base_group.attrs.get("name", b""))
                # Returns early when `base_name == preferred_base` makes the remaining work inapplicable.
                if base_name == preferred_base:
                    return base_group
            base_names = [self._decode_cgns_attr(group.attrs.get("name", b"")) for group in base_groups]
            raise ValueError(f"样本 {sample_id} 未找到目标 Base {preferred_base}，可用 Base: {base_names}")

        # Returns early when `len(base_groups) == 1` makes the remaining work inapplicable.
        if len(base_groups) == 1:
            return base_groups[0]

        base_names = [self._decode_cgns_attr(group.attrs.get("name", b"")) for group in base_groups]
        raise ValueError(f"样本 {sample_id} 检测到多个 Base，但未指定目标 Base，可用 Base: {base_names}")

    # Defines `_decode_cgns_char_rows` for  decode cgns char rows logic.
    def _decode_cgns_char_rows(self, values) -> List[str]:
        arr = np.asarray(values, dtype=np.int8)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        # Checks `arr.ndim == 1` before entering this conditional branch.
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)

        # Checks `` before entering this conditional branch.
        if arr.ndim == 2 and arr.shape[0] >= self.CGNS_CHAR_WIDTH and arr.shape[1] < arr.shape[0]:
            arr = arr.transpose()

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

    # Defines `_read_base_iterative_time_values` for  read base iterative time values logic.
    def _read_base_iterative_time_values(self, base_group: h5py.Group) -> List[float]:
        base_iter_group = self._find_child_group_by_label_and_name(
            parent=base_group,
            target_label="BaseIterativeData_t",
        # Completes the multi-line `self._find_child_group_by_label_and_name` arguments or elements.
        )
        if base_iter_group is None:
            return []

        time_values_group = self._find_child_group_by_label_and_name(
            # Completes the multi-line `self._find_child_group_by_label_and_name` arguments or elements.
            parent=base_iter_group,
            target_label="DataArray_t",
            target_name="TimeValues",
        )
        # Returns early when `time_values_group is None or " data" not in time_values_group` makes the remaining work inapplicable.
        if time_values_group is None or " data" not in time_values_group:
            return []

        raw = np.asarray(time_values_group[" data"][()]).reshape(-1)
        out: List[float] = []
        # Visits each item in `raw.tolist()` and applies the current processing step.
        for value in raw.tolist():
            try:
                out.append(float(value))
            except (TypeError, ValueError):
                # Handles the data or control-flow case represented by this statement.
                continue
        return out

    def _inspect_temporal_solution_names(self, zone_group: h5py.Group) -> List[str]:
        solution_groups = self._find_child_groups_by_label(zone_group, "FlowSolution_t")
        # Returns early when `not solution_groups` makes the remaining work inapplicable.
        if not solution_groups:
            return []

        solution_name_map = {
            self._decode_cgns_attr(group.attrs.get("name", b"")): group
            # Continues the surrounding statement or collection definition.
            for group in solution_groups
        }
        pointer_names: List[str] = []
        zone_iter_group = self._find_child_group_by_label_and_name(
            # Completes the multi-line `self._find_child_group_by_label_and_name` arguments or elements.
            parent=zone_group,
            target_label="ZoneIterativeData_t",
        )
        if zone_iter_group is not None:
            # Stores the `pointer_group` for use by the remaining steps in this flow.
            pointer_group = self._find_child_group_by_label_and_name(
                parent=zone_iter_group,
                target_label="DataArray_t",
                target_name="FlowSolutionPointers",
            # Completes the multi-line `self._find_child_group_by_label_and_name` arguments or elements.
            )
            if pointer_group is not None and " data" in pointer_group:
                pointer_names = self._decode_cgns_char_rows(pointer_group[" data"][()])

        ordered_names: List[str] = []
        # Stores the `used_names` for use by the remaining steps in this flow.
        used_names = set()
        for name in pointer_names:
            if name in solution_name_map and name not in used_names:
                ordered_names.append(name)
                # Adds the current value to `used_names` for later aggregation.
                used_names.add(name)

        for name in sorted(solution_name_map.keys()):
            if name not in used_names:
                ordered_names.append(name)

        # Returns `ordered_names` to the caller for the next stage.
        return ordered_names

    def _read_flow_solution_pointer_names(self, zone_group: h5py.Group) -> List[str]:
        zone_iter_group = self._find_child_group_by_label_and_name(
            parent=zone_group,
            # Completes the multi-line `self._find_child_group_by_label_and_name` arguments or elements.
            target_label="ZoneIterativeData_t",
        )
        if zone_iter_group is None:
            return []
        # Stores the `pointer_group` for use by the remaining steps in this flow.
        pointer_group = self._find_child_group_by_label_and_name(
            parent=zone_iter_group,
            target_label="DataArray_t",
            target_name="FlowSolutionPointers",
        # Completes the multi-line `self._find_child_group_by_label_and_name` arguments or elements.
        )
        if pointer_group is None or " data" not in pointer_group:
            return []
        return self._decode_cgns_char_rows(pointer_group[" data"][()])

    # Defines `inspect_temporal_layout` for inspect temporal layout logic.
    def inspect_temporal_layout(self, sample_file: str, sample_id: int) -> Optional[Dict[str, Any]]:
        sample_path = Path(sample_file)
        if sample_path.suffix.lower() != ".cgns" or not sample_path.is_file():
            return None

        # Scopes resource use so the context manager can release it on exit.
        with h5py.File(sample_path, "r") as f:
            base_groups = self._find_child_groups_by_label(f, "CGNSBase_t")
            if not base_groups:
                return None

            # Stores the `target_base_group` for use by the remaining steps in this flow.
            target_base_group = self._select_target_base_group(base_groups, sample_id)
            zone_groups = self._find_child_groups_by_label(target_base_group, "Zone_t")
            if not zone_groups:
                return None

            # Stores the `pointer_names` for use by the remaining steps in this flow.
            pointer_names = self._read_flow_solution_pointer_names(zone_groups[0])
            if len(pointer_names) <= 1:
                return None

            pointer_name_set = set(pointer_names)
            # Builds the `ordered_names` using the comprehension's filter and mapping rules.
            ordered_names = [
                name
                for name in self._inspect_temporal_solution_names(zone_groups[0])
                if name in pointer_name_set
            # Continues the surrounding statement or collection definition.
            ]
            if len(ordered_names) <= 1:
                return None

            time_values = self._read_base_iterative_time_values(target_base_group)
            # Checks `len(time_values) < len(ordered_names)` before entering this conditional branch.
            if len(time_values) < len(ordered_names):
                time_values.extend(float(index) for index in range(len(time_values), len(ordered_names)))

            resolved_path = str(sample_path.resolve())
            return {
                # Completes the multi-line `collection` arguments or elements.
                "source_path": resolved_path,
                "template_file": resolved_path,
                "layout_kind": "temporal_single_cgns_file",
                "ordered_files": [resolved_path],
                # Completes the multi-line `collection` arguments or elements.
                "step_relpaths": [],
                "time_values": time_values[: len(ordered_names)],
                "manifest_name": None,
                "solution_names": ordered_names,
                # Completes the multi-line `collection` arguments or elements.
                "output_name": sample_path.name,
            }

    def _get_cgns_zone_size_info(self, zone_group: h5py.Group) -> Dict[str, int]:
        if " data" not in zone_group:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("Zone åç»ç¼ºå° ' data' æ°æ®é")

        zone_data = np.asarray(zone_group[" data"][()])
        flattened = zone_data.reshape(-1)
        if flattened.size != 3:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("å½åä»æ¯æéç¨éç»æç½æ ¼ç Zone æ°æ®æ ¼å¼")

        return {
            "VertexSize": int(flattened[0]),
            "CellSize": int(flattened[1]),
            # Completes the multi-line `collection` arguments or elements.
            "VertexSizeBoundary": int(flattened[2]),
        }

    def _inspect_cgns_zone_specs(
        self,
        sample_file: str,
        # Continues the surrounding statement or collection definition.
        sample_id: int,
        point_or_cell: str,
    ) -> Tuple[str, List[Tuple[str, int]]]:
        with h5py.File(sample_file, "r") as f:
            # Validates `not base_groups` and raises `ValueError` when the input cannot be processed safely.
            base_groups = self._find_child_groups_by_label(f, "CGNSBase_t")
            if not base_groups:
                raise ValueError(f"样本 {sample_id} 的 CGNS 文件中未找到 Base 节点")

            target_base_group = self._select_target_base_group(base_groups, sample_id)
            zone_groups = self._find_child_groups_by_label(target_base_group, "Zone_t")
            # Stores the `base_name` for use by the remaining steps in this flow.
            if not zone_groups:
                base_name = self._decode_cgns_attr(target_base_group.attrs.get("name", b""))
                raise ValueError(f"样本 {sample_id} 在 Base {base_name} 下未找到 Zone 节点")

            zone_specs: List[Tuple[str, int]] = []
            for zone_group in zone_groups:
                # Stores the `zone_size_info` for use by the remaining steps in this flow.
                zone_name = self._decode_cgns_attr(zone_group.attrs.get("name", b""))
                zone_size_info = self._get_cgns_zone_size_info(zone_group)
                zone_size = zone_size_info["VertexSize"] if point_or_cell == "point" else zone_size_info["CellSize"]
                zone_specs.append((zone_name, int(zone_size)))

            return self._decode_cgns_attr(target_base_group.attrs.get("name", b"")), zone_specs

    @staticmethod
    # Initializes the `location_map` as a collection for values from this processing stage.
    def _normalize_location_map(raw_map: Dict[str, str]) -> Dict[str, str]:
        location_map: Dict[str, str] = {}
        for name, location in (raw_map or {}).items():
            if not isinstance(name, str):
                # Stores the `location_map[name]` for use by the remaining steps in this flow.
                continue
            location_map[name] = normalize_field_location(location)
        return location_map

    def _resolve_projected_point_or_cell(self) -> str:
        task_cfg = self.config.get("task", {}) if isinstance(self.config, dict) else {}
        # Completes the multi-line `task_cfg.get` arguments or elements.
        location = (
            task_cfg.get("projected_point_or_cell")
            or task_cfg.get("point_or_cell")
            or "point"
        # Returns `normalize_field_location(location)` to the caller for the next stage.
        )
        return normalize_field_location(location)

    def _select_target_base_node(self, base_nodes: List, sample_id: int):
        preferred_base = self.target_base_name
        if preferred_base:
            # Returns early when `getattr(base_node, "name", "") == preferred_base` makes the remaining work inapplicable.
            for base_node in base_nodes:
                if getattr(base_node, "name", "") == preferred_base:
                    return base_node
            base_names = [getattr(node, "name", "") for node in base_nodes]
            raise ValueError(f"样本 {sample_id} 未找到目标 Base {preferred_base}，可用 Base: {base_names}")

        # Returns `base_nodes[0]` to the caller for the next stage.
        if len(base_nodes) == 1:
            return base_nodes[0]

        base_names = [getattr(node, "name", "") for node in base_nodes]
        raise ValueError(f"样本 {sample_id} 检测到多个 Base，但未指定目标 Base，可用 Base: {base_names}")

    def _inspect_cgns_zone_runtime_specs(
        self,
        # Continues the surrounding statement or collection definition.
        sample_file: str,
        sample_id: int,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        io_reader = IOCGNSReader(sample_file)
        # Stores the `base_nodes` for use by the remaining steps in this flow.
        cgns_data = io_reader.read()
        base_nodes = cgns_data.root.get_node_by_label("CGNSBase_t")
        if not base_nodes:
            raise ValueError(f"样本 {sample_id} 的 CGNS 文件中未找到 Base 节点")

        target_base_node = self._select_target_base_node(base_nodes, sample_id)
        # Validates `not zone_nodes` and raises `ValueError` when the input cannot be processed safely.
        zone_nodes = target_base_node.get_node_by_label("Zone_t")
        if not zone_nodes:
            raise ValueError(f"样本 {sample_id} 在 Base {target_base_node.name} 下未找到 Zone 节点")

        zone_specs: List[Dict[str, Any]] = []
        for zone_node in zone_nodes:
            # Stores the `conn_dict` for use by the remaining steps in this flow.
            points = zone_node.get_mesh_coords()
            conn_dict = zone_node.get_mesh_connectivity()
            if isinstance(conn_dict, dict) and "Elem2Node" in conn_dict:
                connectivity = conn_dict["Elem2Node"]["connectivity"]
            # Stores the `connectivity` for use by the remaining steps in this flow.
            else:
                connectivity = conn_dict
            normalized_conn = np.asarray(
                [np.asarray(cell, dtype=np.int64).reshape(-1) for cell in connectivity],
                # Completes the multi-line `np.asarray` arguments or elements.
                dtype=object,
            )
            zone_specs.append(
                {
                    # Completes the multi-line `int` arguments or elements.
                    "zone_name": zone_node.name,
                    "point_size": int(len(points)),
                    "cell_size": int(len(normalized_conn)),
                    "connectivity": normalized_conn,
                # Completes the multi-line `zone_specs.append` arguments or elements.
                }
            )

        return target_base_node.name, zone_specs

    def _inspect_cgns_zone_connectivity_specs(
        self,
        sample_file: str,
        # Completes the multi-line `collection` arguments or elements.
        sample_id: int,
    ) -> Tuple[str, Dict[str, np.ndarray]]:
        """在缺少坐标节点时，仅复用 CGNS 拓扑连接信息。"""
        io_reader = IOCGNSReader(sample_file)
        cgns_data = io_reader.read()
        # Validates `not base_nodes` and raises `ValueError` when the input cannot be processed safely.
        base_nodes = cgns_data.root.get_node_by_label("CGNSBase_t")
        if not base_nodes:
            raise ValueError(f"样本 {sample_id} 的 CGNS 文件中未找到 Base 节点")

        target_base_node = self._select_target_base_node(base_nodes, sample_id)
        zone_nodes = target_base_node.get_node_by_label("Zone_t")
        # Raises `ValueError` to report why this path cannot continue.
        if not zone_nodes:
            raise ValueError(f"样本 {sample_id} 在 Base {target_base_node.name} 下未找到 Zone 节点")

        connectivity_map: Dict[str, np.ndarray] = {}
        for zone_node in zone_nodes:
            conn_dict = zone_node.get_mesh_connectivity()
            # Selects the required item from `conn_dict["Elem2Node"]` as the `connectivity`.
            if isinstance(conn_dict, dict) and "Elem2Node" in conn_dict:
                connectivity = conn_dict["Elem2Node"]["connectivity"]
            else:
                connectivity = conn_dict
            # Completes the multi-line `np.asarray(cell, dtype=np.int64).reshape` arguments or elements.
            connectivity_map[str(zone_node.name)] = np.asarray(
                [np.asarray(cell, dtype=np.int64).reshape(-1) for cell in connectivity],
                dtype=object,
            )

        return target_base_node.name, connectivity_map

    def _inspect_cgns_zone_projection_specs(
        # Continues the surrounding statement or collection definition.
        self,
        sample_file: str,
        sample_id: int,
        projected_point_or_cell: str,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        # Keeps the operation and its recovery path within the same exception boundary.
        """优先读取运行时拓扑；若模板缺少坐标节点，则退回到 HDF5 的 ZoneSize 元数据。"""
        try:
            return self._inspect_cgns_zone_runtime_specs(
                sample_file=sample_file,
                # Completes the multi-line `self._inspect_cgns_zone_runtime_specs` arguments or elements.
                sample_id=sample_id,
            )
        except Exception as exc:
            normalized_location = normalize_field_location(projected_point_or_cell)
            # Completes the multi-line `self._inspect_cgns_zone_specs` arguments or elements.
            target_base_name, raw_zone_specs = self._inspect_cgns_zone_specs(
                sample_file=sample_file,
                sample_id=sample_id,
                point_or_cell=normalized_location,
            # Initializes the `connectivity_map` as a collection for values from this processing stage.
            )
            connectivity_map: Dict[str, np.ndarray] = {}
            connectivity_error: Optional[Exception] = None
            try:
                # Completes the multi-line `self._inspect_cgns_zone_connectivity_specs` arguments or elements.
                _, connectivity_map = self._inspect_cgns_zone_connectivity_specs(
                    sample_file=sample_file,
                    sample_id=sample_id,
                )
            # Stores the `connectivity_error` for use by the remaining steps in this flow.
            except Exception as conn_exc:
                connectivity_error = conn_exc

            fallback_specs: List[Dict[str, Any]] = []
            with h5py.File(sample_file, "r") as f:
                base_groups = self._find_child_groups_by_label(f, "CGNSBase_t")
                # Stores the `zone_groups` for use by the remaining steps in this flow.
                target_base_group = self._select_target_base_group(base_groups, sample_id)
                zone_groups = self._find_child_groups_by_label(target_base_group, "Zone_t")
                zone_size_map = {}
                for zone_group in zone_groups:
                    # Stores the `zone_size_map[zone_name]` for use by the remaining steps in this flow.
                    zone_name = self._decode_cgns_attr(zone_group.attrs.get("name", b""))
                    zone_size_map[zone_name] = self._get_cgns_zone_size_info(zone_group)

            for zone_name, _ in raw_zone_specs:
                zone_size_info = zone_size_map[str(zone_name)]
                fallback_specs.append(
                    # Completes the multi-line `str` arguments or elements.
                    {
                        "zone_name": str(zone_name),
                        "point_size": int(zone_size_info["VertexSize"]),
                        "cell_size": int(zone_size_info["CellSize"]),
                        # Completes the multi-line `collection` arguments or elements.
                        "connectivity": connectivity_map.get(str(zone_name)),
                    }
                )

            if connectivity_map:
                self._log(
                    # Completes the multi-line `self._log` arguments or elements.
                    "warning",
                    f"样本 {sample_id} 未能读取运行时坐标，已退回 ZoneSize 并复用 Elements_t 拓扑: {exc}",
                )
            else:
                # Calls `self._log` to perform the operation required at this stage.
                suffix = f"; 连接关系恢复失败: {connectivity_error}" if connectivity_error else ""
                self._log(
                    "warning",
                    f"样本 {sample_id} 未能读取运行时拓扑，已退回到 ZoneSize 元数据: {exc}{suffix}",
                # Returns `target_base_name, fallback_specs` to the caller for the next stage.
                )
            return target_base_name, fallback_specs

    @staticmethod
    def _grid_location_string(point_or_cell: str) -> str:
        return "Vertex" if point_or_cell == "point" else "CellCenter"

    def _ensure_base_iterative_metadata(
        # Continues the surrounding statement or collection definition.
        self,
        base_group: h5py.Group,
        num_steps: int = 1,
        time_values: Optional[List[float]] = None,
    # Stores the `base_iter_group` for use by the remaining steps in this flow.
    ) -> None:
        base_iter_group = self._find_child_group_by_label_and_name(
            parent=base_group,
            target_label="BaseIterativeData_t",
        # Checks `base_iter_group is None` before entering this conditional branch.
        )
        if base_iter_group is None:
            base_iter_group = self._ensure_child_group(
                parent=base_group,
                # Completes the multi-line `self._ensure_child_group` arguments or elements.
                key="BaseIterativeData",
                name="BaseIterativeData",
                label="BaseIterativeData_t",
                type_code="I4",
                # Completes the multi-line `self._ensure_child_group` arguments or elements.
                flags=np.array([1], dtype=np.int32),
            )
        else:
            self._set_group_attrs(
                # Completes the multi-line `self._decode_cgns_attr` arguments or elements.
                base_iter_group,
                name=self._decode_cgns_attr(base_iter_group.attrs.get("name", b"")) or "BaseIterativeData",
                label="BaseIterativeData_t",
                type_code="I4",
                # Completes the multi-line `self._set_group_attrs` arguments or elements.
                flags=base_iter_group.attrs.get("flags", np.array([1], dtype=np.int32)),
            )

        steps = np.array([max(1, int(num_steps))], dtype=np.int32)
        if " data" in base_iter_group:
            del base_iter_group[" data"]
        base_iter_group.create_dataset(" data", data=steps)

        # Stores the `time_values_group` for use by the remaining steps in this flow.
        provided_time_values = list(time_values or [])
        time_values_group = self._ensure_child_group(
            parent=base_iter_group,
            key="TimeValues",
            # Completes the multi-line `self._ensure_child_group` arguments or elements.
            name="TimeValues",
            label="DataArray_t",
            type_code="R8",
            flags=np.array([1], dtype=np.int32),
        # Initializes the `normalized_time_values` as a collection for values from this processing stage.
        )
        normalized_time_values: List[float] = []
        for step_idx in range(int(steps[0])):
            if step_idx < len(provided_time_values):
                # Adds the current value to `normalized_time_values` for later aggregation.
                try:
                    normalized_time_values.append(float(provided_time_values[step_idx]))
                    continue
                except (TypeError, ValueError):
                    # Adds the current value to `normalized_time_values` for later aggregation.
                    pass
            normalized_time_values.append(float(step_idx))

        if " data" in time_values_group:
            del time_values_group[" data"]
        time_values_group.create_dataset(
            # Completes the multi-line `np.asarray` arguments or elements.
            " data",
            data=np.asarray(normalized_time_values, dtype=np.float64),
        )

        iteration_values = self._ensure_child_group(
            parent=base_iter_group,
            # Completes the multi-line `self._ensure_child_group` arguments or elements.
            key="IterationValues",
            name="IterationValues",
            label="DataArray_t",
            type_code="I4",
            # Completes the multi-line `self._ensure_child_group` arguments or elements.
            flags=np.array([1], dtype=np.int32),
        )
        if " data" in iteration_values:
            del iteration_values[" data"]
        iteration_values.create_dataset(" data", data=np.arange(steps[0], dtype=np.int32))

    # Stores the `grid_group` for use by the remaining steps in this flow.
    def _ensure_grid_location_group(self, solution_group: h5py.Group, grid_location: str) -> None:
        grid_group = self._ensure_child_group(
            parent=solution_group,
            key="GridLocation",
            # Completes the multi-line `self._ensure_child_group` arguments or elements.
            name="GridLocation",
            label="GridLocation_t",
            type_code="C1",
            flags=np.array([1], dtype=np.int32),
        # Stores the `raw` for use by the remaining steps in this flow.
        )
        raw = self._build_c1_payload(grid_location, null_terminated=False)
        self._write_cgns_char_dataset(grid_group, raw)

    def _ensure_zone_solution_group(
        self,
        # Continues the surrounding statement or collection definition.
        zone_group: h5py.Group,
        solution_name: str,
        point_or_cell: str,
    ) -> h5py.Group:
        # Handles the data or control-flow case represented by this statement.
        if solution_name in zone_group:
            del zone_group[solution_name]
        solution_group = self._ensure_child_group(
            parent=zone_group,
            # Completes the multi-line `self._ensure_child_group` arguments or elements.
            key=solution_name,
            name=solution_name,
            label="FlowSolution_t",
            type_code="MT",
            # Completes the multi-line `self._ensure_child_group` arguments or elements.
            flags=np.array([1], dtype=np.int32),
        )
        self._ensure_grid_location_group(solution_group, self._grid_location_string(point_or_cell))
        return solution_group

    def _clear_solution_data_arrays(self, solution_group: h5py.Group) -> None:
        # Visits each item in `solution_group.keys()` and applies the current processing step.
        delete_keys: List[str] = []
        for key in solution_group.keys():
            obj = solution_group[key]
            if not isinstance(obj, h5py.Group):
                # Checks `` before entering this conditional branch.
                continue
            if self._decode_cgns_attr(obj.attrs.get("label", b"")) == "DataArray_t":
                delete_keys.append(key)
        for key in delete_keys:
            del solution_group[key]

    # Continues the surrounding statement or collection definition.
    def _prune_solution_data_arrays(
        self,
        solution_group: h5py.Group,
        keep_names: List[str],
    # Builds the `keep_set` using the comprehension's filter and mapping rules.
    ) -> None:
        keep_set = {str(name).strip() for name in keep_names if str(name).strip()}
        delete_keys: List[str] = []
        for key in solution_group.keys():
            # Checks `not isinstance(obj, h5py.Group)` before entering this conditional branch.
            obj = solution_group[key]
            if not isinstance(obj, h5py.Group):
                continue
            if self._decode_cgns_attr(obj.attrs.get("label", b"")) != "DataArray_t":
                # Checks `key in keep_set` before entering this conditional branch.
                continue
            if key in keep_set:
                continue
            delete_keys.append(key)
        # Handles the data or control-flow case represented by this statement.
        for key in delete_keys:
            del solution_group[key]

    def _get_or_create_zone_field_group(
        self,
        solution_group: h5py.Group,
        # Continues the surrounding statement or collection definition.
        var_name: str,
        zone_data: np.ndarray,
    ) -> h5py.Group:
        if var_name in solution_group and isinstance(solution_group[var_name], h5py.Group):
            # Calls `self._set_group_attrs` to perform the operation required at this stage.
            data_group = solution_group[var_name]
            self._set_group_attrs(
                data_group,
                name=self._decode_cgns_attr(data_group.attrs.get("name", b"")) or var_name,
                # Completes the multi-line `self._infer_cgns_type_code` arguments or elements.
                label="DataArray_t",
                type_code=self._infer_cgns_type_code(zone_data),
                flags=data_group.attrs.get("flags", np.array([1], dtype=np.int32)),
            )
            return data_group

        # Completes the multi-line `self._ensure_child_group` arguments or elements.
        return self._ensure_child_group(
            parent=solution_group,
            key=var_name,
            name=var_name,
            # Completes the multi-line `self._infer_cgns_type_code` arguments or elements.
            label="DataArray_t",
            type_code=self._infer_cgns_type_code(zone_data),
            flags=np.array([1], dtype=np.int32),
        )

    def _list_solution_data_array_names(self, solution_group: h5py.Group) -> List[str]:
        # Visits each item in `solution_group.keys()` and applies the current processing step.
        names: List[str] = []
        for key in solution_group.keys():
            obj = solution_group[key]
            if not isinstance(obj, h5py.Group):
                # Checks `` before entering this conditional branch.
                continue
            if self._decode_cgns_attr(obj.attrs.get("label", b"")) != "DataArray_t":
                continue
            names.append(key)
        return names

    # Reads `GridLocation` from the mapping into the `grid_group`.
    def _read_solution_grid_location(self, solution_group: h5py.Group) -> str:
        grid_group = solution_group.get("GridLocation")
        if not isinstance(grid_group, h5py.Group):
            return ""
        # Returns early when `values is None` makes the remaining work inapplicable.
        values = grid_group.get(" data")
        if values is None:
            return ""
        arr = np.asarray(values[()], dtype=np.int8).reshape(-1)
        return bytes(arr.tolist()).decode("utf-8", errors="ignore").rstrip("\x00 ").strip()

    # Continues the surrounding statement or collection definition.
    def _select_single_step_solution_group(
        self,
        existing_groups: List[h5py.Group],
        point_or_cell: str,
        # Continues the surrounding statement or collection definition.
        target_var_names: List[str],
    ) -> h5py.Group:
        normalized_targets = {str(name).strip() for name in target_var_names if str(name).strip()}
        expected_location = self._grid_location_string(point_or_cell)

        best_group = existing_groups[0]
        # Visits each item in `existing_groups` and applies the current processing step.
        best_score: Optional[Tuple[int, int, int, int]] = None
        for group in existing_groups:
            group_key = group.name.split("/")[-1]
            group_name = self._decode_cgns_attr(group.attrs.get("name", b"")) or group_key
            # Stores the `overlap` for use by the remaining steps in this flow.
            data_array_names = set(self._list_solution_data_array_names(group))
            overlap = len(data_array_names & normalized_targets)
            location_match = int(self._read_solution_grid_location(group) == expected_location)
            preferred_name = int(group_key == self.SOLUTION_GROUP_NAME or group_name == self.SOLUTION_GROUP_NAME)
            # Checks `best_score is None or score > best_score` before entering this conditional branch.
            score = (overlap, location_match, preferred_name, len(data_array_names))
            if best_score is None or score > best_score:
                best_group = group
                best_score = score

        return best_group

    def _prepare_single_step_solution_group(
        # Continues the surrounding statement or collection definition.
        self,
        zone_group: h5py.Group,
        point_or_cell: str,
        target_var_names: List[str],
    # Stores the `existing_groups` for use by the remaining steps in this flow.
    ) -> h5py.Group:
        existing_groups = self._find_child_groups_by_label(zone_group, "FlowSolution_t")
        if not existing_groups:
            return self._ensure_zone_solution_group(
                # Completes the multi-line `self._ensure_zone_solution_group` arguments or elements.
                zone_group,
                self.SOLUTION_GROUP_NAME,
                point_or_cell,
            )

        primary_group = self._select_single_step_solution_group(
            # Completes the multi-line `self._select_single_step_solution_group` arguments or elements.
            existing_groups,
            point_or_cell,
            target_var_names,
        )
        # Stores the `primary_name` for use by the remaining steps in this flow.
        primary_key = primary_group.name.split("/")[-1]
        primary_name = self._decode_cgns_attr(primary_group.attrs.get("name", b"")) or primary_key
        self._set_group_attrs(
            primary_group,
            # Completes the multi-line `self._set_group_attrs` arguments or elements.
            name=primary_name,
            label="FlowSolution_t",
            type_code="MT",
            flags=primary_group.attrs.get("flags", np.array([1], dtype=np.int32)),
        )

        # Selects the required item from `zone_group` as the `obj`.
        for key in list(zone_group.keys()):
            obj = zone_group[key]
            if not isinstance(obj, h5py.Group):
                continue
            # Handles the data or control-flow case represented by this statement.
            if self._decode_cgns_attr(obj.attrs.get("label", b"")) != "FlowSolution_t":
                continue
            if key == primary_key:
                continue
            del zone_group[key]

        # Returns `primary_group` to the caller for the next stage.
        self._ensure_grid_location_group(primary_group, self._grid_location_string(point_or_cell))
        return primary_group

    @staticmethod
    def _build_location_solution_group_name(point_or_cell: str) -> str:
        return "FlowSolution_Point" if point_or_cell == "point" else "FlowSolution_Cell"

    def _ensure_named_solution_group(
        # Continues the surrounding statement or collection definition.
        self,
        zone_group: h5py.Group,
        solution_name: str,
        point_or_cell: str,
    # Checks `` before entering this conditional branch.
    ) -> h5py.Group:
        if solution_name in zone_group and isinstance(zone_group[solution_name], h5py.Group):
            solution_group = zone_group[solution_name]
            self._set_group_attrs(
                # Completes the multi-line `self._decode_cgns_attr` arguments or elements.
                solution_group,
                name=self._decode_cgns_attr(solution_group.attrs.get("name", b"")) or solution_name,
                label="FlowSolution_t",
                type_code="MT",
                # Completes the multi-line `self._set_group_attrs` arguments or elements.
                flags=solution_group.attrs.get("flags", np.array([1], dtype=np.int32)),
            )
            self._ensure_grid_location_group(solution_group, self._grid_location_string(point_or_cell))
            return solution_group

        solution_group = self._ensure_child_group(
            # Completes the multi-line `self._ensure_child_group` arguments or elements.
            parent=zone_group,
            key=solution_name,
            name=solution_name,
            label="FlowSolution_t",
            # Completes the multi-line `np.array` arguments or elements.
            type_code="MT",
            flags=np.array([1], dtype=np.int32),
        )
        self._ensure_grid_location_group(solution_group, self._grid_location_string(point_or_cell))
        return solution_group

    # Defines `_build_temporal_location_solution_name` for  build temporal location solution name logic.
    @staticmethod
    def _build_temporal_location_solution_name(
        primary_solution_name: str,
        point_or_cell: str,
        # Continues the surrounding statement or collection definition.
        primary_location: str,
    ) -> str:
        normalized_location = normalize_field_location(point_or_cell)
        normalized_primary = normalize_field_location(primary_location)
        # Returns `str(primary_solution_name)` to the caller for the next stage.
        if normalized_location == normalized_primary:
            return str(primary_solution_name)
        suffix = "Point" if normalized_location == "point" else "Cell"
        return f"{primary_solution_name}_{suffix}"

    def _prepare_single_step_solution_groups_by_location(
        # Continues the surrounding statement or collection definition.
        self,
        zone_group: h5py.Group,
        updates_by_location: Dict[str, Dict[str, np.ndarray]],
    ) -> Dict[str, h5py.Group]:
        # Stores the `existing_groups` for use by the remaining steps in this flow.
        selected_groups: Dict[str, h5py.Group] = {}
        existing_groups = self._find_child_groups_by_label(zone_group, "FlowSolution_t")
        available_groups = list(existing_groups)
        active_locations = [
            # Completes the multi-line `collection` arguments or elements.
            point_or_cell
            for point_or_cell, zone_updates in updates_by_location.items()
            if zone_updates
        ]
        use_legacy_default_name = len(active_locations) == 1

        # Checks `not zone_updates` before entering this conditional branch.
        for point_or_cell, zone_updates in updates_by_location.items():
            if not zone_updates:
                continue
            normalized_location = normalize_field_location(point_or_cell)
            # Stores the `solution_group` for use by the remaining steps in this flow.
            if available_groups:
                solution_group = self._select_single_step_solution_group(
                    available_groups,
                    normalized_location,
                    # Completes the multi-line `self._select_single_step_solution_group` arguments or elements.
                    list(zone_updates.keys()),
                )
                available_groups = [group for group in available_groups if group is not solution_group]
                solution_key = solution_group.name.split("/")[-1]
                # Calls `self._set_group_attrs` to perform the operation required at this stage.
                solution_name = self._decode_cgns_attr(solution_group.attrs.get("name", b"")) or solution_key
                self._set_group_attrs(
                    solution_group,
                    name=solution_name,
                    # Completes the multi-line `self._set_group_attrs` arguments or elements.
                    label="FlowSolution_t",
                    type_code="MT",
                    flags=solution_group.attrs.get("flags", np.array([1], dtype=np.int32)),
                )
                # Continues the surrounding statement or collection definition.
                self._ensure_grid_location_group(solution_group, self._grid_location_string(normalized_location))
            else:
                desired_key = (
                    self.SOLUTION_GROUP_NAME
                    # Completes the multi-line `self._build_location_solution_group_name` arguments or elements.
                    if use_legacy_default_name
                    else self._build_location_solution_group_name(normalized_location)
                )
                solution_group = self._ensure_zone_solution_group(
                    # Completes the multi-line `self._ensure_zone_solution_group` arguments or elements.
                    zone_group,
                    desired_key,
                    normalized_location,
                )
            selected_groups[normalized_location] = solution_group

        # Visits each item in `list(zone_group.keys())` and applies the current processing step.
        keep_keys = {group.name.split("/")[-1] for group in selected_groups.values()}
        for key in list(zone_group.keys()):
            obj = zone_group[key]
            if not isinstance(obj, h5py.Group):
                # Checks `` before entering this conditional branch.
                continue
            if self._decode_cgns_attr(obj.attrs.get("label", b"")) != "FlowSolution_t":
                continue
            if key in keep_keys:
                # Handles the data or control-flow case represented by this statement.
                continue
            del zone_group[key]

        return selected_groups

    def _prepare_temporal_solution_groups(
        self,
        zone_group: h5py.Group,
        # Continues the surrounding statement or collection definition.
        point_or_cell: str,
        solution_names: List[str],
    ) -> Dict[str, h5py.Group]:
        normalized_names = [str(name).strip() for name in solution_names if str(name).strip()]
        # Builds the `existing_by_name` using the comprehension's filter and mapping rules.
        existing_groups = self._find_child_groups_by_label(zone_group, "FlowSolution_t")
        existing_by_name = {
            self._decode_cgns_attr(group.attrs.get("name", b"")) or group.name.split("/")[-1]: group
            for group in existing_groups
        }

        # Visits each item in `normalized_names` and applies the current processing step.
        prepared: Dict[str, h5py.Group] = {}
        for solution_name in normalized_names:
            if solution_name in existing_by_name:
                solution_group = existing_by_name[solution_name]
                # Completes the multi-line `self._set_group_attrs` arguments or elements.
                self._set_group_attrs(
                    solution_group,
                    name=solution_name,
                    label="FlowSolution_t",
                    # Completes the multi-line `solution_group.attrs.get` arguments or elements.
                    type_code="MT",
                    flags=solution_group.attrs.get("flags", np.array([1], dtype=np.int32)),
                )
                self._ensure_grid_location_group(solution_group, self._grid_location_string(point_or_cell))
                # Handles the data or control-flow case represented by this statement.
                prepared[solution_name] = solution_group
                continue

            prepared[solution_name] = self._ensure_zone_solution_group(
                zone_group,
                solution_name,
                # Completes the multi-line `self._ensure_zone_solution_group` arguments or elements.
                point_or_cell,
            )

        keep_keys = {group.name.split("/")[-1] for group in prepared.values()}
        for key in list(zone_group.keys()):
            obj = zone_group[key]
            # Handles the data or control-flow case represented by this statement.
            if not isinstance(obj, h5py.Group):
                continue
            if self._decode_cgns_attr(obj.attrs.get("label", b"")) != "FlowSolution_t":
                continue
            # Handles the data or control-flow case represented by this statement.
            if key in keep_keys:
                continue
            del zone_group[key]

        return prepared

    def _update_flow_solution_pointers(
        self,
        # Continues the surrounding statement or collection definition.
        zone_group: h5py.Group,
        solution_names: List[str],
    ) -> None:
        zone_iter_group = self._ensure_child_group(
            # Completes the multi-line `self._ensure_child_group` arguments or elements.
            parent=zone_group,
            key="ZoneIterativeData",
            name="ZoneIterativeData",
            label="ZoneIterativeData_t",
            # Completes the multi-line `np.array` arguments or elements.
            type_code="MT",
            flags=np.array([1], dtype=np.int32),
        )
        data_group = self._ensure_child_group(
            # Completes the multi-line `self._ensure_child_group` arguments or elements.
            parent=zone_iter_group,
            key="FlowSolutionPointers",
            name="FlowSolutionPointers",
            label="DataArray_t",
            # Completes the multi-line `np.array` arguments or elements.
            type_code="C1",
            flags=np.array([1], dtype=np.int32),
        )
        normalized_names = [str(name).strip() for name in solution_names if str(name).strip()]
        # Initializes the `normalized_names` as a collection for values from this processing stage.
        if not normalized_names:
            normalized_names = [self.SOLUTION_GROUP_NAME]
        payload = self._build_c1_row_payload(
            normalized_names,
            # Completes the multi-line `self._build_c1_row_payload` arguments or elements.
            min_width=self.CGNS_CHAR_WIDTH,
            null_terminated=False,
        )
        self._write_cgns_char_dataset(data_group, payload)

    def _normalize_single_step_metadata(
        # Continues the surrounding statement or collection definition.
        self,
        base_group: h5py.Group,
        zone_group: h5py.Group,
        solution_name: str,
    ) -> None:
        # Stores the `has_base_iter` for use by the remaining steps in this flow.
        """尽量继承源文件的单步时序元数据，避免破坏商软导出的 CGNS 布局。"""
        has_base_iter = self._has_child_group_with_label(base_group, "BaseIterativeData_t")
        has_simulation_type = self._has_child_group_with_label(base_group, "SimulationType_t")
        has_zone_iter = self._has_child_group_with_label(zone_group, "ZoneIterativeData_t")

        if has_base_iter:
            # Stores the `single_step_time_values` for use by the remaining steps in this flow.
            original_time_values = self._read_base_iterative_time_values(base_group)
            single_step_time_values = original_time_values[:1] if original_time_values else [0.0]
            pointer_solution_names = self._read_flow_solution_pointer_names(zone_group)
            if solution_name in pointer_solution_names:
                # Checks `solution_index < len(original_time_values)` before entering this conditional branch.
                solution_index = pointer_solution_names.index(solution_name)
                if solution_index < len(original_time_values):
                    single_step_time_values = [original_time_values[solution_index]]
            self._ensure_base_iterative_metadata(
                # Completes the multi-line `self._ensure_base_iterative_metadata` arguments or elements.
                base_group=base_group,
                num_steps=1,
                time_values=single_step_time_values,
            )

        if has_simulation_type and not has_base_iter:
            # 某些文件只保留 SimulationType，不额外补建 BaseIterativeData。
            # Completes the multi-line `self._find_child_group_by_label_and_name` arguments or elements.
            simulation_group = self._find_child_group_by_label_and_name(
                parent=base_group,
                target_label="SimulationType_t",
            )
            # Calls `self._set_group_attrs` to perform the operation required at this stage.
            if simulation_group is not None:
                self._set_group_attrs(
                    simulation_group,
                    name=self._decode_cgns_attr(simulation_group.attrs.get("name", b"")) or "SimulationType",
                    # Completes the multi-line `self._set_group_attrs` arguments or elements.
                    label="SimulationType_t",
                    type_code="C1",
                    flags=simulation_group.attrs.get("flags", np.array([1], dtype=np.int32)),
                )

        if has_zone_iter:
            self._update_flow_solution_pointers(zone_group, [solution_name])

    # Stores the `dtype` for use by the remaining steps in this flow.
    def _infer_cgns_type_code(self, array: np.ndarray) -> str:
        dtype = np.asarray(array).dtype
        if dtype == np.float32:
            return "R4"
        # Returns `"R8"` to the caller for the next stage.
        if dtype == np.float64:
            return "R8"
        if dtype == np.int32:
            return "I4"
        # Returns `"I8"` to the caller for the next stage.
        if dtype == np.int64:
            return "I8"
        return "R8"

    def _write_zone_field(self, solution_group: h5py.Group, var_name: str, zone_data: np.ndarray) -> None:
        data_group = self._get_or_create_zone_field_group(solution_group, var_name, zone_data)
        # Checks `" data" in data_group` before entering this conditional branch.
        expected_shape = tuple(np.asarray(zone_data).shape[::-1])
        if " data" in data_group:
            del data_group[" data"]
        ds = data_group.create_dataset(" data", shape=expected_shape, dtype=np.asarray(zone_data).dtype)
        ds[...] = np.asarray(zone_data, dtype=ds.dtype).transpose()

    # Defines `_slice_zone_field` for  slice zone field logic.
    @staticmethod
    def _slice_zone_field(merged_field: np.ndarray, start: int, end: int) -> np.ndarray:
        if merged_field.ndim == 1:
            return np.asarray(merged_field[start:end])
        # Stores the `slicer[-1]` for use by the remaining steps in this flow.
        slicer = [slice(None)] * merged_field.ndim
        slicer[-1] = slice(start, end)
        return np.asarray(merged_field[tuple(slicer)])

    def _build_per_zone_var_data(
        self,
        # Completes the multi-line `collection` arguments or elements.
        sample_id: int,
        processed_fields: Dict[str, np.ndarray],
        zone_specs,
        variables: List[str],
    # Stores the `updated_count` for use by the remaining steps in this flow.
    ) -> Optional[Tuple[Dict[str, Dict[str, np.ndarray]], int]]:
        updated_count = 0
        per_zone_var_data: Dict[str, Dict[str, np.ndarray]] = {}
        for var_name in variables:
            # Handles the data or control-flow case represented by this statement.
            if var_name not in processed_fields:
                continue

            merged_field = np.asarray(processed_fields[var_name])
            spatial_size = int(merged_field.shape[0] if merged_field.ndim == 1 else merged_field.shape[-1])
            cursor = 0

            for zone_name, zone_size in zone_specs:
                # Checks `end_idx > spatial_size` before entering this conditional branch.
                end_idx = cursor + zone_size
                if end_idx > spatial_size:
                    self._log(
                        "warning",
                        # Completes the multi-line `self._log` arguments or elements.
                        f"样本 {sample_id} 的变量 {var_name} 长度不足: 需要 {end_idx}，实际只有 {spatial_size}",
                    )
                    return None

                zone_field = self._slice_zone_field(merged_field, cursor, end_idx)
                per_zone_var_data.setdefault(zone_name, {})[var_name] = zone_field
                # Stores the `updated_count` for use by the remaining steps in this flow.
                cursor = end_idx
                updated_count += 1

            if cursor != spatial_size:
                self._log(
                    "warning",
                    # Completes the multi-line `self._log` arguments or elements.
                    f"样本 {sample_id} 的变量 {var_name} 长度不匹配: 已消费 {cursor}，实际为 {spatial_size}",
                )
                return None

        if updated_count == 0:
            self._log("warning", f"样本 {sample_id} 没有可写回的变量")
            return None

        return per_zone_var_data, updated_count

    # Continues the surrounding statement or collection definition.
    def _build_per_zone_projected_var_data(
        self,
        sample_id: int,
        processed_fields: Dict[str, np.ndarray],
        # Continues the surrounding statement or collection definition.
        zone_specs: List[Dict[str, Any]],
        variables: List[str],
        projected_point_or_cell: str,
    ) -> Optional[Tuple[Dict[str, Dict[str, np.ndarray]], int]]:
        # Initializes the `per_zone_var_data` as a collection for values from this processing stage.
        updated_count = 0
        per_zone_var_data: Dict[str, Dict[str, np.ndarray]] = {}
        normalized_location = normalize_field_location(projected_point_or_cell)

        for var_name in variables:
            if var_name not in processed_fields:
                continue

            # Converts the `spatial_size` to `int` so later checks use a consistent type.
            merged_field = np.asarray(processed_fields[var_name])
            spatial_size = int(merged_field.shape[0] if merged_field.ndim == 1 else merged_field.shape[-1])
            cursor = 0

            for zone_spec in zone_specs:
                zone_name = str(zone_spec["zone_name"])
                # Derives the `end_idx` from the current values for downstream processing.
                zone_size = int(zone_spec["point_size"] if normalized_location == "point" else zone_spec["cell_size"])
                end_idx = cursor + zone_size
                if end_idx > spatial_size:
                    self._log(
                        # Completes the multi-line `self._log` arguments or elements.
                        "warning",
                        f"样本 {sample_id} 的变量 {var_name} 长度不足: 需要 {end_idx}，实际只有 {spatial_size}",
                    )
                    return None

                zone_field = self._slice_zone_field(merged_field, cursor, end_idx)
                # Stores the `cursor` for use by the remaining steps in this flow.
                per_zone_var_data.setdefault(zone_name, {})[var_name] = zone_field
                cursor = end_idx
                updated_count += 1

            if cursor != spatial_size:
                self._log(
                    # Completes the multi-line `self._log` arguments or elements.
                    "warning",
                    f"样本 {sample_id} 的变量 {var_name} 长度不匹配: 已消费 {cursor}，实际为 {spatial_size}",
                )
                return None

        if updated_count == 0:
            # Returns `None` to the caller for the next stage.
            self._log("warning", f"样本 {sample_id} 没有可写回的变量")
            return None

        return per_zone_var_data, updated_count

    def _project_per_zone_fields_back_to_original_locations(
        self,
        sample_id: int,
        # Completes the multi-line `collection` arguments or elements.
        per_zone_projected_fields: Dict[str, Dict[str, np.ndarray]],
        zone_specs: List[Dict[str, Any]],
        field_location_map: Dict[str, str],
        projected_point_or_cell: str,
    # Stores the task settings for use by the remaining steps in this flow.
    ) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        task_cfg = self.config.get("task", {}) if isinstance(self.config, dict) else {}
        projection_method = str(task_cfg.get("projection_method") or "mean")
        grouped: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {"point": {}, "cell": {}}
        # Builds the `zone_spec_map` using the comprehension's filter and mapping rules.
        normalized_projected_location = normalize_field_location(projected_point_or_cell)
        zone_spec_map = {str(item["zone_name"]): item for item in zone_specs}

        for zone_name, zone_fields in per_zone_projected_fields.items():
            zone_spec = zone_spec_map.get(str(zone_name))
            if zone_spec is None:
                # Selects the required item from `zone_spec` as the `connectivity`.
                raise ValueError(f"样本 {sample_id} 未找到 Zone {zone_name} 的运行时拓扑信息")
            connectivity = zone_spec["connectivity"]
            num_points = int(zone_spec["point_size"])

            for field_name, values in zone_fields.items():
                target_location = field_location_map.get(field_name, normalized_projected_location)
                # Validates `` and raises `ValueError` when the input cannot be processed safely.
                target_location = normalize_field_location(target_location)
                if (
                    target_location != normalized_projected_location
                    and connectivity is None
                # Raises `ValueError` to report why this path cannot continue.
                ):
                    raise ValueError(
                        f"样本 {sample_id} 的 Zone {zone_name} 缺少拓扑连接关系，"
                        f"无法将字段 {field_name} 从 {normalized_projected_location} 反投影到 {target_location}"
                    # Stores the `projected` for use by the remaining steps in this flow.
                    )
                projected = project_field_values(
                    values=np.asarray(values),
                    source_location=normalized_projected_location,
                    # Completes the multi-line `project_field_values` arguments or elements.
                    target_location=target_location,
                    connectivity=connectivity,
                    num_points=num_points,
                    projection_method=projection_method,
                # Stores the `setdefault(str(zone_name), {})[field_name]` for use by the remaining steps in this flow.
                )
                grouped[target_location].setdefault(str(zone_name), {})[field_name] = projected

        return grouped

    def _build_per_zone_point_coordinates(
        self,
        sample_id: int,
        # Completes the multi-line `collection` arguments or elements.
        point_coordinates: Optional[np.ndarray],
        zone_specs: List[Tuple[str, int]],
    ) -> Optional[Dict[str, np.ndarray]]:
        if point_coordinates is None:
            return None

        # Validates `merged_points.ndim != 2 or merged_points.shape[1] != 3` and raises `ValueError` when the input cannot be processed safely.
        merged_points = np.asarray(point_coordinates, dtype=np.float64)
        if merged_points.ndim != 2 or merged_points.shape[1] != 3:
            raise ValueError(f"样本 {sample_id} 的点坐标必须为 [N, 3]，当前为 {merged_points.shape}")

        per_zone_points: Dict[str, np.ndarray] = {}
        cursor = 0
        # Checks `isinstance(zone_spec, dict)` before entering this conditional branch.
        for zone_spec in zone_specs:
            if isinstance(zone_spec, dict):
                zone_name = str(zone_spec["zone_name"])
                zone_size = int(zone_spec["point_size"])
            # Stores the `(zone_name, zone_size)` for use by the remaining steps in this flow.
            else:
                zone_name, zone_size = zone_spec
                zone_name = str(zone_name)
                zone_size = int(zone_size)
            # Validates `end_idx > merged_points.shape[0]` and raises `ValueError` when the input cannot be processed safely.
            end_idx = cursor + zone_size
            if end_idx > merged_points.shape[0]:
                raise ValueError(
                    f"样本 {sample_id} 的点坐标长度不足以覆盖 Zone {zone_name}: "
                    # Completes the multi-line `ValueError` arguments or elements.
                    f"need={end_idx}, actual={merged_points.shape[0]}"
                )
            per_zone_points[zone_name] = np.asarray(merged_points[cursor:end_idx], dtype=np.float64)
            cursor = end_idx

        if cursor != merged_points.shape[0]:
            # Completes the multi-line `ValueError` arguments or elements.
            raise ValueError(
                f"样本 {sample_id} 的点坐标长度与 Zone 总点数不匹配: used={cursor}, actual={merged_points.shape[0]}"
            )
        return per_zone_points

    def _write_zone_coordinates(self, zone_group: h5py.Group, zone_points: np.ndarray) -> None:
        # Completes the multi-line `self._find_child_group_by_label_and_name` arguments or elements.
        grid_group = self._find_child_group_by_label_and_name(
            parent=zone_group,
            target_label="GridCoordinates_t",
        )
        # Raises `ValueError` to report why this path cannot continue.
        if grid_group is None:
            raise ValueError(f"Zone {zone_group.name} 缺少 GridCoordinates_t，无法更新运动网格坐标")

        normalized = np.asarray(zone_points, dtype=np.float64)
        if normalized.ndim != 2 or normalized.shape[1] != 3:
            raise ValueError(f"Zone {zone_group.name} 点坐标必须为 [N, 3]，当前为 {normalized.shape}")

        coordinate_names = ["CoordinateX", "CoordinateY", "CoordinateZ"]
        # Stores the `coord_group` for use by the remaining steps in this flow.
        for axis_idx, coordinate_name in enumerate(coordinate_names):
            coord_group = self._find_child_group_by_label_and_name(
                parent=grid_group,
                target_label="DataArray_t",
                # Completes the multi-line `self._find_child_group_by_label_and_name` arguments or elements.
                target_name=coordinate_name,
            )
            if coord_group is None:
                raise ValueError(f"Zone {zone_group.name} 缺少 {coordinate_name}，无法更新运动网格坐标")
            # Handles the data or control-flow case represented by this statement.
            if " data" in coord_group:
                del coord_group[" data"]
            coord_group.create_dataset(" data", data=np.asarray(normalized[:, axis_idx], dtype=np.float64))

    @staticmethod
    def _normalize_temporal_solution_names(
        # Continues the surrounding statement or collection definition.
        solution_names: Optional[List[str]],
        num_steps: int,
    ) -> List[str]:
        normalized: List[str] = []
        # Visits each item in `solution_names` and applies the current processing step.
        if solution_names:
            for name in solution_names:
                text = str(name).strip()
                if text:
                    # Repeats the current step while `len(normalized) < num_steps` remains true.
                    normalized.append(text)
        while len(normalized) < num_steps:
            normalized.append(f"FlowSolution_{len(normalized) + 1:04d}")
        return normalized[:num_steps]

    def _write_cgns_fields_inplace(
        # Continues the surrounding statement or collection definition.
        self,
        source_file: str,
        output_file: str,
        target_base_name: str,
        # Continues the surrounding statement or collection definition.
        zone_names: List[str],
        point_or_cell: str,
        per_zone_var_data: Dict[str, Dict[str, np.ndarray]],
        per_zone_point_coordinates: Optional[Dict[str, np.ndarray]] = None,
    # Calls `shutil.copy2` to perform the operation required at this stage.
    ) -> bool:
        shutil.copy2(source_file, output_file)

        with h5py.File(output_file, "r+") as f:
            base_group = self._find_child_group_by_label_and_name(
                parent=f,
                # Completes the multi-line `self._find_child_group_by_label_and_name` arguments or elements.
                target_label="CGNSBase_t",
                target_name=target_base_name,
            )
            if base_group is None:
                raise ValueError(f"未找到目标 Base: {target_base_name}")

            # Stores the `zone_group` for use by the remaining steps in this flow.
            for zone_name in zone_names:
                zone_group = self._find_child_group_by_label_and_name(
                    parent=base_group,
                    target_label="Zone_t",
                    # Completes the multi-line `self._find_child_group_by_label_and_name` arguments or elements.
                    target_name=zone_name,
                )
                if zone_group is None:
                    raise ValueError(f"未找到目标 Zone: {zone_name}")

                zone_updates = per_zone_var_data.get(zone_name, {})
                # Calls `self._write_zone_coordinates` to perform the operation required at this stage.
                if per_zone_point_coordinates and zone_name in per_zone_point_coordinates:
                    self._write_zone_coordinates(zone_group, per_zone_point_coordinates[zone_name])
                if not zone_updates:
                    continue

                solution_group = self._prepare_single_step_solution_group(
                    # Completes the multi-line `self._prepare_single_step_solution_group` arguments or elements.
                    zone_group,
                    point_or_cell,
                    list(zone_updates.keys()),
                )
                # Visits each item in `zone_updates.items()` and applies the current processing step.
                self._prune_solution_data_arrays(solution_group, list(zone_updates.keys()))
                for var_name, zone_data in zone_updates.items():
                    self._write_zone_field(solution_group, var_name, zone_data)
                solution_name = self._decode_cgns_attr(solution_group.attrs.get("name", b"")) or solution_group.name.split("/")[-1]
                self._normalize_single_step_metadata(base_group, zone_group, solution_name)

        return True

    # Continues the surrounding statement or collection definition.
    def _write_cgns_fields_inplace_mixed(
        self,
        source_file: str,
        output_file: str,
        # Continues the surrounding statement or collection definition.
        target_base_name: str,
        zone_names: List[str],
        per_zone_var_data_by_location: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        per_zone_point_coordinates: Optional[Dict[str, np.ndarray]] = None,
    # Calls `shutil.copy2` to perform the operation required at this stage.
    ) -> bool:
        shutil.copy2(source_file, output_file)

        with h5py.File(output_file, "r+") as f:
            base_group = self._find_child_group_by_label_and_name(
                parent=f,
                # Completes the multi-line `self._find_child_group_by_label_and_name` arguments or elements.
                target_label="CGNSBase_t",
                target_name=target_base_name,
            )
            if base_group is None:
                raise ValueError(f"未找到目标 Base: {target_base_name}")

            # Stores the `zone_group` for use by the remaining steps in this flow.
            for zone_name in zone_names:
                zone_group = self._find_child_group_by_label_and_name(
                    parent=base_group,
                    target_label="Zone_t",
                    # Completes the multi-line `self._find_child_group_by_label_and_name` arguments or elements.
                    target_name=zone_name,
                )
                if zone_group is None:
                    raise ValueError(f"未找到目标 Zone: {zone_name}")

                if per_zone_point_coordinates and zone_name in per_zone_point_coordinates:
                    self._write_zone_coordinates(zone_group, per_zone_point_coordinates[zone_name])

                # Completes the multi-line `per_zone_var_data_by_location.get('point', {}).get` arguments or elements.
                zone_updates_by_location = {
                    "point": per_zone_var_data_by_location.get("point", {}).get(zone_name, {}),
                    "cell": per_zone_var_data_by_location.get("cell", {}).get(zone_name, {}),
                }
                # Handles the data or control-flow case represented by this statement.
                if not zone_updates_by_location["point"] and not zone_updates_by_location["cell"]:
                    continue

                prepared_groups = self._prepare_single_step_solution_groups_by_location(
                    zone_group=zone_group,
                    updates_by_location=zone_updates_by_location,
                )

                # Visits each item in `["point", "cell"]` and applies the current processing step.
                primary_solution_name = None
                for point_or_cell in ["point", "cell"]:
                    zone_updates = zone_updates_by_location.get(point_or_cell, {})
                    if not zone_updates:
                        # Selects the required item from `prepared_groups` as the `solution_group`.
                        continue
                    solution_group = prepared_groups[point_or_cell]
                    self._prune_solution_data_arrays(solution_group, list(zone_updates.keys()))
                    for var_name, zone_data in zone_updates.items():
                        # Checks `primary_solution_name is None` before entering this conditional branch.
                        self._write_zone_field(solution_group, var_name, zone_data)
                    if primary_solution_name is None:
                        primary_solution_name = (
                            self._decode_cgns_attr(solution_group.attrs.get("name", b""))
                            # Continues the surrounding statement or collection definition.
                            or solution_group.name.split("/")[-1]
                        )

                if primary_solution_name:
                    self._normalize_single_step_metadata(base_group, zone_group, primary_solution_name)

        return True

    def _write_temporal_cgns_fields_inplace(
        self,
        # Continues the surrounding statement or collection definition.
        source_file: str,
        output_file: str,
        target_base_name: str,
        zone_names: List[str],
        # Completes the multi-line `collection` arguments or elements.
        point_or_cell: str,
        per_step_zone_var_data: List[Dict[str, Dict[str, np.ndarray]]],
        solution_names: List[str],
        time_values: Optional[List[float]] = None,
    # Calls `shutil.copy2` to perform the operation required at this stage.
    ) -> bool:
        shutil.copy2(source_file, output_file)

        with h5py.File(output_file, "r+") as f:
            base_group = self._find_child_group_by_label_and_name(
                parent=f,
                # Completes the multi-line `self._find_child_group_by_label_and_name` arguments or elements.
                target_label="CGNSBase_t",
                target_name=target_base_name,
            )
            if base_group is None:
                raise ValueError(f"未找到目标 Base: {target_base_name}")

            # Calls `self._ensure_base_iterative_metadata` to perform the operation required at this stage.
            self._ensure_simulation_type_group(base_group)
            self._ensure_base_iterative_metadata(
                base_group=base_group,
                num_steps=len(solution_names),
                # Completes the multi-line `self._ensure_base_iterative_metadata` arguments or elements.
                time_values=time_values,
            )

            for zone_name in zone_names:
                zone_group = self._find_child_group_by_label_and_name(
                    parent=base_group,
                    # Completes the multi-line `self._find_child_group_by_label_and_name` arguments or elements.
                    target_label="Zone_t",
                    target_name=zone_name,
                )
                if zone_group is None:
                    raise ValueError(f"未找到目标 Zone: {zone_name}")

                # Completes the multi-line `self._prepare_temporal_solution_groups` arguments or elements.
                prepared_groups = self._prepare_temporal_solution_groups(
                    zone_group,
                    point_or_cell,
                    solution_names,
                # Visits each item in `enumerate(solution_names)` and applies the current processing step.
                )
                for step_idx, solution_name in enumerate(solution_names):
                    zone_updates = per_step_zone_var_data[step_idx].get(zone_name, {})
                    if not zone_updates:
                        # Selects the required item from `prepared_groups` as the `solution_group`.
                        continue
                    solution_group = prepared_groups[solution_name]
                    self._prune_solution_data_arrays(solution_group, list(zone_updates.keys()))
                    for var_name, zone_data in zone_updates.items():
                        self._write_zone_field(solution_group, var_name, zone_data)

                self._update_flow_solution_pointers(zone_group, solution_names)

        return True

    # Continues the surrounding statement or collection definition.
    def _write_temporal_cgns_fields_inplace_mixed(
        self,
        source_file: str,
        output_file: str,
        # Continues the surrounding statement or collection definition.
        target_base_name: str,
        zone_names: List[str],
        per_step_zone_var_data_by_location: List[Dict[str, Dict[str, Dict[str, np.ndarray]]]],
        solution_names: List[str],
        # Continues the surrounding statement or collection definition.
        projected_point_or_cell: str,
        time_values: Optional[List[float]] = None,
    ) -> bool:
        shutil.copy2(source_file, output_file)

        normalized_projected_location = normalize_field_location(projected_point_or_cell)
        # Stores the `base_group` for use by the remaining steps in this flow.
        with h5py.File(output_file, "r+") as f:
            base_group = self._find_child_group_by_label_and_name(
                parent=f,
                target_label="CGNSBase_t",
                # Completes the multi-line `self._find_child_group_by_label_and_name` arguments or elements.
                target_name=target_base_name,
            )
            if base_group is None:
                raise ValueError(f"未找到目标 Base: {target_base_name}")

            self._ensure_simulation_type_group(base_group)
            # Completes the multi-line `self._ensure_base_iterative_metadata` arguments or elements.
            self._ensure_base_iterative_metadata(
                base_group=base_group,
                num_steps=len(solution_names),
                time_values=time_values,
            )

            # Stores the `zone_group` for use by the remaining steps in this flow.
            for zone_name in zone_names:
                zone_group = self._find_child_group_by_label_and_name(
                    parent=base_group,
                    target_label="Zone_t",
                    # Completes the multi-line `self._find_child_group_by_label_and_name` arguments or elements.
                    target_name=zone_name,
                )
                if zone_group is None:
                    raise ValueError(f"未找到目标 Zone: {zone_name}")

                keep_keys = set()
                # Initializes the `updates_by_location` as a collection for values from this processing stage.
                for step_idx, primary_solution_name in enumerate(solution_names):
                    updates_by_location = {
                        "point": per_step_zone_var_data_by_location[step_idx].get("point", {}).get(zone_name, {}),
                        "cell": per_step_zone_var_data_by_location[step_idx].get("cell", {}).get(zone_name, {}),
                    # Builds the `active_locations` using the comprehension's filter and mapping rules.
                    }
                    active_locations = [
                        location for location in ["point", "cell"]
                        if updates_by_location.get(location)
                    # Checks `not active_locations` before entering this conditional branch.
                    ]
                    if not active_locations:
                        continue

                    primary_location = normalized_projected_location
                    if primary_location not in active_locations:
                        primary_location = active_locations[0]

                    # Stores the `solution_name` for use by the remaining steps in this flow.
                    for point_or_cell in active_locations:
                        solution_name = self._build_temporal_location_solution_name(
                            primary_solution_name=str(primary_solution_name),
                            point_or_cell=point_or_cell,
                            # Completes the multi-line `self._build_temporal_location_solution_name` arguments or elements.
                            primary_location=primary_location,
                        )
                        solution_group = self._ensure_named_solution_group(
                            zone_group=zone_group,
                            # Completes the multi-line `self._ensure_named_solution_group` arguments or elements.
                            solution_name=solution_name,
                            point_or_cell=point_or_cell,
                        )
                        zone_updates = updates_by_location[point_or_cell]
                        # Visits each item in `zone_updates.items()` and applies the current processing step.
                        self._prune_solution_data_arrays(solution_group, list(zone_updates.keys()))
                        for var_name, zone_data in zone_updates.items():
                            self._write_zone_field(solution_group, var_name, zone_data)
                        keep_keys.add(solution_group.name.split("/")[-1])

                for key in list(zone_group.keys()):
                    # Checks `not isinstance(obj, h5py.Group)` before entering this conditional branch.
                    obj = zone_group[key]
                    if not isinstance(obj, h5py.Group):
                        continue
                    if self._decode_cgns_attr(obj.attrs.get("label", b"")) != "FlowSolution_t":
                        # Checks `key in keep_keys` before entering this conditional branch.
                        continue
                    if key in keep_keys:
                        continue
                    del zone_group[key]

                self._update_flow_solution_pointers(zone_group, solution_names)

        return True

    def _get_write_variables(self) -> List[str]:
        # Reads `effective_variables` from the mapping into the `effective` with its declared fallback.
        task_cfg = self.config.get("task", {}) if isinstance(self.config, dict) else {}
        effective = task_cfg.get("effective_variables", [])
        if effective:
            return list(effective)
        raise ValueError("写回结果时必须提供 task.effective_variables")

    # Stores the task settings for use by the remaining steps in this flow.
    def _get_cgns_effective_sources(self) -> Dict[str, Dict[str, object]]:
        task_cfg = self.config.get("task", {}) if isinstance(self.config, dict) else {}
        sources: Dict[str, Dict[str, object]] = {}
        for runtime_key, source in (task_cfg.get("effective_variable_sources", {}) or {}).items():
            # Handles the data or control-flow case represented by this statement.
            if not isinstance(runtime_key, str) or not isinstance(source, dict):
                continue
            if str(source.get("format") or "cgns").lower() != "cgns":
                continue
            # Stores the `location` for use by the remaining steps in this flow.
            raw_name = str(source.get("array_name") or "").strip()
            location = str(source.get("location") or "").strip().lower()
            if not raw_name or location not in {"point", "cell"}:
                continue
            # Stores the `normalized['array_name']` for use by the remaining steps in this flow.
            normalized = dict(source)
            normalized["array_name"] = raw_name
            normalized["location"] = location
            sources[runtime_key] = normalized
        return sources

    # 在按位置反投影后恢复 CGNS 原始字段名。
    @staticmethod
    def _rename_grouped_cgns_fields_by_sources(
        grouped_fields: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        effective_sources: Dict[str, Dict[str, object]],
    ) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        # Initializes the `renamed` as a collection for values from this processing stage.
        """在按位置反投影后恢复 CGNS 原始字段名。"""
        renamed: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {"point": {}, "cell": {}}
        for location, zone_payloads in grouped_fields.items():
            normalized_location = normalize_field_location(location)
            # Initializes the `zone_fields` as a collection for values from this processing stage.
            for zone_name, fields in zone_payloads.items():
                zone_fields: Dict[str, np.ndarray] = {}
                for runtime_key, values in fields.items():
                    source = effective_sources.get(runtime_key) or {}
                    # Converts the `source_location` to `str` so later checks use a consistent type.
                    raw_name = str(source.get("array_name") or runtime_key)
                    source_location = str(source.get("location") or normalized_location)
                    if source_location != normalized_location:
                        raise ValueError(
                            # Completes the multi-line `ValueError` arguments or elements.
                            "CGNS 字段来源位置与反投影结果不一致: "
                            f"field={runtime_key}, source={source_location}, actual={normalized_location}"
                        )
                    if raw_name in zone_fields:
                        # Completes the multi-line `ValueError` arguments or elements.
                        raise ValueError(
                            "同一 CGNS 位置存在重复原始字段名，无法无损写回: "
                            f"location={normalized_location}, field={raw_name}"
                        )
                    # Checks `zone_fields` before entering this conditional branch.
                    zone_fields[raw_name] = values
                if zone_fields:
                    renamed[normalized_location][str(zone_name)] = zone_fields
        return renamed

    def _write_legacy_single_location(
        # Continues the surrounding statement or collection definition.
        self,
        sample_id: int,
        sample_file: str,
        processed_fields: Dict[str, np.ndarray],
        # Continues the surrounding statement or collection definition.
        output_dir: str,
        output_basename: Optional[str] = None,
        point_coordinates: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """以 clean_display 模式将预测结果写回 CGNS。"""
        # Checks `point_or_cell not in {"point", "cell"}` before entering this conditional branch.
        point_or_cell = str(self.config.get("task", {}).get("point_or_cell", "")).strip().lower()
        if point_or_cell not in {"point", "cell"}:
            self._log("warning", f"样本 {sample_id} 缺少合法的 task.point_or_cell 配置")
            return None

        try:
            # Completes the multi-line `self._inspect_cgns_zone_specs` arguments or elements.
            target_base_name, zone_specs = self._inspect_cgns_zone_specs(
                sample_file=sample_file,
                sample_id=sample_id,
                point_or_cell=point_or_cell,
            # Continues the surrounding statement or collection definition.
            )
        except Exception as exc:
            self._log("error", str(exc))
            return None

        variables = self._get_write_variables()
        # Completes the multi-line `self._build_per_zone_var_data` arguments or elements.
        build_result = self._build_per_zone_var_data(
            sample_id=sample_id,
            processed_fields=processed_fields,
            zone_specs=zone_specs,
            # Completes the multi-line `self._build_per_zone_var_data` arguments or elements.
            variables=variables,
        )
        if build_result is None:
            return None
        # Keeps the operation and its recovery path within the same exception boundary.
        per_zone_var_data, updated_count = build_result
        try:
            per_zone_point_coordinates = self._build_per_zone_point_coordinates(
                sample_id=sample_id,
                # Completes the multi-line `self._build_per_zone_point_coordinates` arguments or elements.
                point_coordinates=point_coordinates,
                zone_specs=zone_specs,
            )
        except Exception as exc:
            # Returns `None` to the caller for the next stage.
            self._log("error", f"鏍锋湰 {sample_id} CGNS 运动网格坐标拆分失败: {exc}")
            return None

        basename = output_basename or f"predicted_{sample_id}"
        output_file = os.path.join(output_dir, f"{basename}.cgns")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # Calls `self._write_cgns_fields_inplace` to perform the operation required at this stage.
        try:
            self._write_cgns_fields_inplace(
                source_file=sample_file,
                output_file=output_file,
                # Completes the multi-line `collection` arguments or elements.
                target_base_name=target_base_name,
                zone_names=[name for name, _ in zone_specs],
                point_or_cell=point_or_cell,
                per_zone_var_data=per_zone_var_data,
                # Completes the multi-line `self._write_cgns_fields_inplace` arguments or elements.
                per_zone_point_coordinates=per_zone_point_coordinates,
            )
        except Exception as exc:
            self._log("error", f"样本 {sample_id} CGNS 写回失败: {exc}")
            return None

        # Returns `output_file` to the caller for the next stage.
        self._log("info", f"样本 {sample_id} 以 clean_display 模式写回 {updated_count} 个字段到 CGNS")
        return output_file

    def write(
        self,
        sample_id: int,
        # Completes the multi-line `collection` arguments or elements.
        sample_file: str,
        processed_fields: Dict[str, np.ndarray],
        output_dir: str,
        output_basename: Optional[str] = None,
        # Continues the surrounding statement or collection definition.
        point_coordinates: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """以统一投影后的位置接收字段，并按原始位置反投影写回 CGNS。"""
        try:
            projected_point_or_cell = self._resolve_projected_point_or_cell()
            # Completes the multi-line `self._inspect_cgns_zone_projection_specs` arguments or elements.
            target_base_name, zone_specs = self._inspect_cgns_zone_projection_specs(
                sample_file=sample_file,
                sample_id=sample_id,
                projected_point_or_cell=projected_point_or_cell,
            # Continues the surrounding statement or collection definition.
            )
        except Exception as exc:
            self._log("error", str(exc))
            return None

        variables = self._get_write_variables()
        # Stores the `field_location_map` for use by the remaining steps in this flow.
        task_cfg = self.config.get("task", {}) if isinstance(self.config, dict) else {}
        field_location_map = self._normalize_location_map(task_cfg.get("field_location_map", {}))
        effective_sources = self._get_cgns_effective_sources()
        for runtime_key, source in effective_sources.items():
            field_location_map[runtime_key] = str(source["location"])

        # Completes the multi-line `self._build_per_zone_projected_var_data` arguments or elements.
        build_result = self._build_per_zone_projected_var_data(
            sample_id=sample_id,
            processed_fields=processed_fields,
            zone_specs=zone_specs,
            # Completes the multi-line `self._build_per_zone_projected_var_data` arguments or elements.
            variables=variables,
            projected_point_or_cell=projected_point_or_cell,
        )
        if build_result is None:
            # Stores the `(per_zone_projected_var_data, updated_count)` for use by the remaining steps in this flow.
            return None
        per_zone_projected_var_data, updated_count = build_result

        per_zone_var_data_by_location = self._project_per_zone_fields_back_to_original_locations(
            sample_id=sample_id,
            per_zone_projected_fields=per_zone_projected_var_data,
            # Completes the multi-line `self._project_per_zone_fields_back_to_original_locations` arguments or elements.
            zone_specs=zone_specs,
            field_location_map=field_location_map,
            projected_point_or_cell=projected_point_or_cell,
        )
        # Stores the `per_zone_var_data_by_location` for use by the remaining steps in this flow.
        if effective_sources:
            per_zone_var_data_by_location = self._rename_grouped_cgns_fields_by_sources(
                per_zone_var_data_by_location,
                effective_sources,
            # Keeps the operation and its recovery path within the same exception boundary.
            )
        try:
            per_zone_point_coordinates = self._build_per_zone_point_coordinates(
                sample_id=sample_id,
                # Completes the multi-line `self._build_per_zone_point_coordinates` arguments or elements.
                point_coordinates=point_coordinates,
                zone_specs=zone_specs,
            )
        except Exception as exc:
            # Returns `None` to the caller for the next stage.
            self._log("error", f"样本 {sample_id} CGNS 运动网格坐标拆分失败: {exc}")
            return None

        basename = output_basename or f"predicted_{sample_id}"
        output_file = os.path.join(output_dir, f"{basename}.cgns")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # Calls `self._write_cgns_fields_inplace_mixed` to perform the operation required at this stage.
        try:
            self._write_cgns_fields_inplace_mixed(
                source_file=sample_file,
                output_file=output_file,
                # Completes the multi-line `str` arguments or elements.
                target_base_name=target_base_name,
                zone_names=[str(item["zone_name"]) for item in zone_specs],
                per_zone_var_data_by_location=per_zone_var_data_by_location,
                per_zone_point_coordinates=per_zone_point_coordinates,
            # Continues the surrounding statement or collection definition.
            )
        except Exception as exc:
            self._log("error", f"样本 {sample_id} CGNS 写回失败: {exc}")
            return None

        self._log("info", f"样本 {sample_id} 已写回 {updated_count} 个字段到 CGNS")
        return output_file

    # Continues the surrounding statement or collection definition.
    def write_temporal(
        self,
        sample_id: int,
        sample_file: str,
        # Continues the surrounding statement or collection definition.
        temporal_steps: List[Dict[str, np.ndarray]],
        output_dir: str,
        output_basename: Optional[str] = None,
        time_values: Optional[List[float]] = None,
        # Continues the surrounding statement or collection definition.
        solution_names: Optional[List[str]] = None,
    ) -> Optional[str]:
        """按统一投影后的字段切片写回时序 CGNS，并在写回前恢复原始点/单元位置。"""
        if not temporal_steps:
            self._log("warning", f"样本 {sample_id} 没有可写回的时序结果")
            return None

        # Stores the `projected_point_or_cell` for use by the remaining steps in this flow.
        try:
            projected_point_or_cell = self._resolve_projected_point_or_cell()
            target_base_name, zone_specs = self._inspect_cgns_zone_projection_specs(
                sample_file=sample_file,
                # Completes the multi-line `self._inspect_cgns_zone_projection_specs` arguments or elements.
                sample_id=sample_id,
                projected_point_or_cell=projected_point_or_cell,
            )
        except Exception as exc:
            # Returns `None` to the caller for the next stage.
            self._log("error", str(exc))
            return None

        variables = self._get_write_variables()
        task_cfg = self.config.get("task", {}) if isinstance(self.config, dict) else {}
        field_location_map = self._normalize_location_map(task_cfg.get("field_location_map", {}))
        # Visits each item in `effective_sources.items()` and applies the current processing step.
        effective_sources = self._get_cgns_effective_sources()
        for runtime_key, source in effective_sources.items():
            field_location_map[runtime_key] = str(source["location"])
        per_step_zone_var_data_by_location: List[Dict[str, Dict[str, Dict[str, np.ndarray]]]] = []
        total_updated_count = 0

        # Stores the `build_result` for use by the remaining steps in this flow.
        for step_fields in temporal_steps:
            build_result = self._build_per_zone_projected_var_data(
                sample_id=sample_id,
                processed_fields=step_fields,
                # Completes the multi-line `self._build_per_zone_projected_var_data` arguments or elements.
                zone_specs=zone_specs,
                variables=variables,
                projected_point_or_cell=projected_point_or_cell,
            )
            # Returns `None` to the caller for the next stage.
            if build_result is None:
                return None

            per_zone_projected_var_data, updated_count = build_result
            grouped_step_fields = self._project_per_zone_fields_back_to_original_locations(
                sample_id=sample_id,
                # Completes the multi-line `self._project_per_zone_fields_back_to_original_locations` arguments or elements.
                per_zone_projected_fields=per_zone_projected_var_data,
                zone_specs=zone_specs,
                field_location_map=field_location_map,
                projected_point_or_cell=projected_point_or_cell,
            # Checks `effective_sources` before entering this conditional branch.
            )
            if effective_sources:
                grouped_step_fields = self._rename_grouped_cgns_fields_by_sources(
                    grouped_step_fields,
                    # Completes the multi-line `self._rename_grouped_cgns_fields_by_sources` arguments or elements.
                    effective_sources,
                )
            per_step_zone_var_data_by_location.append(grouped_step_fields)
            total_updated_count += updated_count

        normalized_solution_names = self._normalize_temporal_solution_names(
            # Completes the multi-line `len` arguments or elements.
            solution_names,
            len(temporal_steps),
        )
        basename = output_basename or f"predicted_{sample_id}"
        # Calls `os.makedirs` to perform the operation required at this stage.
        output_file = os.path.join(output_dir, f"{basename}.cgns")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        try:
            self._write_temporal_cgns_fields_inplace_mixed(
                # Completes the multi-line `self._write_temporal_cgns_fields_inplace_mixed` arguments or elements.
                source_file=sample_file,
                output_file=output_file,
                target_base_name=target_base_name,
                zone_names=[str(item["zone_name"]) for item in zone_specs],
                # Completes the multi-line `self._write_temporal_cgns_fields_inplace_mixed` arguments or elements.
                per_step_zone_var_data_by_location=per_step_zone_var_data_by_location,
                solution_names=normalized_solution_names,
                projected_point_or_cell=projected_point_or_cell,
                time_values=time_values,
            # Continues the surrounding statement or collection definition.
            )
        except Exception as exc:
            self._log("error", f"样本 {sample_id} 时序 CGNS 写回失败: {exc}")
            return None

        self._log(
            # Completes the multi-line `len` arguments or elements.
            "info",
            f"样本 {sample_id} 已写回时序 CGNS，共 {len(temporal_steps)} 个时间步、{total_updated_count} 个字段切片",
        )
        return output_file

    def write_temporal_legacy(
        # Continues the surrounding statement or collection definition.
        self,
        sample_id: int,
        sample_file: str,
        temporal_steps: List[Dict[str, np.ndarray]],
        # Continues the surrounding statement or collection definition.
        output_dir: str,
        output_basename: Optional[str] = None,
        time_values: Optional[List[float]] = None,
        solution_names: Optional[List[str]] = None,
    ) -> Optional[str]:
        # Stores the `point_or_cell` for use by the remaining steps in this flow.
        """将时序预测结果按 clean_display 模式写回 CGNS。"""
        point_or_cell = str(self.config.get("task", {}).get("point_or_cell", "")).strip().lower()
        if point_or_cell not in {"point", "cell"}:
            self._log("warning", f"样本 {sample_id} 缺少合法的 task.point_or_cell 配置")
            # Checks `not temporal_steps` before entering this conditional branch.
            return None
        if not temporal_steps:
            self._log("warning", f"样本 {sample_id} 没有可写回的时序结果")
            return None

        try:
            # Completes the multi-line `self._inspect_cgns_zone_specs` arguments or elements.
            target_base_name, zone_specs = self._inspect_cgns_zone_specs(
                sample_file=sample_file,
                sample_id=sample_id,
                point_or_cell=point_or_cell,
            # Continues the surrounding statement or collection definition.
            )
        except Exception as exc:
            self._log("error", str(exc))
            return None

        variables = self._get_write_variables()
        # Stores the `total_updated_count` for use by the remaining steps in this flow.
        per_step_zone_var_data: List[Dict[str, Dict[str, np.ndarray]]] = []
        total_updated_count = 0
        for step_fields in temporal_steps:
            build_result = self._build_per_zone_var_data(
                # Completes the multi-line `self._build_per_zone_var_data` arguments or elements.
                sample_id=sample_id,
                processed_fields=step_fields,
                zone_specs=zone_specs,
                variables=variables,
            # Returns early when `build_result is None` makes the remaining work inapplicable.
            )
            if build_result is None:
                return None
            zone_var_data, updated_count = build_result
            # Stores the `total_updated_count` for use by the remaining steps in this flow.
            per_step_zone_var_data.append(zone_var_data)
            total_updated_count += updated_count

        normalized_solution_names = self._normalize_temporal_solution_names(
            solution_names,
            len(temporal_steps),
        # Stores the `basename` for use by the remaining steps in this flow.
        )
        basename = output_basename or f"predicted_{sample_id}"
        output_file = os.path.join(output_dir, f"{basename}.cgns")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # Calls `self._write_temporal_cgns_fields_inplace` to perform the operation required at this stage.
        try:
            self._write_temporal_cgns_fields_inplace(
                source_file=sample_file,
                output_file=output_file,
                # Completes the multi-line `collection` arguments or elements.
                target_base_name=target_base_name,
                zone_names=[name for name, _ in zone_specs],
                point_or_cell=point_or_cell,
                per_step_zone_var_data=per_step_zone_var_data,
                # Completes the multi-line `self._write_temporal_cgns_fields_inplace` arguments or elements.
                solution_names=normalized_solution_names,
                time_values=time_values,
            )
        except Exception as exc:
            # Returns `None` to the caller for the next stage.
            self._log("error", f"样本 {sample_id} 时序 CGNS 写回失败: {exc}")
            return None

        self._log(
            "info",
            f"样本 {sample_id} 已写回时序 CGNS，共 {len(temporal_steps)} 个时间步、{total_updated_count} 个字段切片",
        # Returns `output_file` to the caller for the next stage.
        )
        return output_file
