from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..io_utils import cgnskeywords, cgnsmesh


# Initializes the `BC_FEATURE_NAMES` as a collection for values from this processing stage.
BC_FEATURE_NAMES = [
    "bc_is_boundary",
    "bc_type_wall",
    "bc_type_inlet",
    # Completes the multi-line `collection` arguments or elements.
    "bc_type_outlet",
    "bc_type_symmetry",
    "bc_type_interface",
    "bc_type_other",
# Completes the multi-line `collection` arguments or elements.
]

_BC_TYPE_TO_INDEX = {
    "wall": 1,
    "inlet": 2,
    # Completes the multi-line `collection` arguments or elements.
    "outlet": 3,
    "symmetry": 4,
    "interface": 5,
    "other": 6,
# Completes the multi-line `collection` arguments or elements.
}


def _as_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    # Keeps the operation and its recovery path within the same exception boundary.
    try:
        return np.asarray(value)
    except Exception:
        return None


# Defines `normalize_field_bc_names` for normalize field bc names logic.
def normalize_field_bc_names(field_bc: Any) -> List[str]:
    if isinstance(field_bc, str):
        names = [field_bc]
    elif isinstance(field_bc, (list, tuple)):
        # Stores the `names` for use by the remaining steps in this flow.
        names = list(field_bc)
    else:
        raise ValueError("vtk_based 需要 field_bc 为字符串或字符串列表")

    normalized = []
    # Visits each item in `names` and applies the current processing step.
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("field_bc 中的字段名必须为非空字符串")
        normalized.append(name.strip())
    # Validates `not normalized` and raises `ValueError` when the input cannot be processed safely.
    if not normalized:
        raise ValueError("vtk_based 至少需要一个 field_bc 字段")
    return normalized


def _normalize_indices(values: Any, upper_bound: int) -> np.ndarray:
    # Stores the `array` for use by the remaining steps in this flow.
    array = _as_array(values)
    if array is None or array.size == 0 or upper_bound <= 0:
        return np.empty((0,), dtype=np.int64)

    indices = np.asarray(array, dtype=np.int64).reshape(-1)
    # Returns early when `indices.size == 0` makes the remaining work inapplicable.
    if indices.size == 0:
        return indices

    if int(indices.min()) >= 1 and int(indices.max()) <= upper_bound:
        indices = indices - 1

    # Returns `indices[(indices >= 0) & (indices < upper_bound)]` to the caller for the next stage.
    return indices[(indices >= 0) & (indices < upper_bound)]


def _iter_cell_nodes(connectivity: Any, num_points: int) -> Iterable[np.ndarray]:
    if connectivity is None:
        return []

    # Stores the `raw_connectivity` for use by the remaining steps in this flow.
    raw_connectivity = np.asarray(connectivity, dtype=object)
    if raw_connectivity.ndim == 1:
        if all(np.isscalar(value) for value in raw_connectivity):
            raw_cells = [raw_connectivity]
        # Continues the surrounding statement or collection definition.
        else:
            raw_cells = list(raw_connectivity)
    else:
        raw_cells = [row for row in raw_connectivity]

    # Initializes the `cells` as a collection for values from this processing stage.
    cells = []
    for cell in raw_cells:
        nodes = _normalize_indices(cell, num_points)
        if nodes.size > 0:
            # Adds the current value to `cells` for later aggregation.
            cells.append(np.unique(nodes))
    return cells


def _normalize_bc_kind(raw_value: Any) -> str:
    text = str(raw_value or "").strip().lower()
    # Returns early when `not text` makes the remaining work inapplicable.
    if not text:
        return "other"
    if "interface" in text or "overset" in text or "periodic" in text or "connect" in text:
        return "interface"
    # Returns early when `"symmetry" in text or text in {"sym", "bcsymmetryplane"}` makes the remaining work inapplicable.
    if "symmetry" in text or text in {"sym", "bcsymmetryplane"}:
        return "symmetry"
    if "inflow" in text or "inlet" in text or "bcinflow" in text:
        return "inlet"
    # Returns early when `"outflow" in text or "outlet" in text or "bcoutflow" in text` makes the remaining work inapplicable.
    if "outflow" in text or "outlet" in text or "bcoutflow" in text:
        return "outlet"
    if "wall" in text or "noslip" in text or "slip" in text:
        return "wall"
    # Returns `"other"` to the caller for the next stage.
    return "other"


def _normalize_cgns_flag_kind(flag_value: Any) -> str:
    try:
        flag = int(float(flag_value))
    # Completes the multi-line `collection` arguments or elements.
    except (TypeError, ValueError):
        return _normalize_bc_kind(flag_value)

    if flag == 0:
        return ""
    # Returns early when `flag == 1` makes the remaining work inapplicable.
    if flag == 1:
        return "symmetry"
    if flag == 2:
        return "outlet"
    # Returns early when `flag == 3` makes the remaining work inapplicable.
    if flag == 3:
        return "wall"
    if flag in {4, 5, 6}:
        return "inlet"
    # Returns early when `flag in {10, 11}` makes the remaining work inapplicable.
    if flag in {10, 11}:
        return "interface"
    return "other"


def _mark_features(features: np.ndarray, indices: np.ndarray, kind: str) -> None:
    # Returns early when `indices.size == 0` makes the remaining work inapplicable.
    if indices.size == 0:
        return
    normalized_kind = _normalize_bc_kind(kind)
    features[indices, 0] = 1.0
    # Stores the `get(normalized_kind, _BC_TYPE_TO_INDEX['other'])]` for use by the remaining steps in this flow.
    features[indices, _BC_TYPE_TO_INDEX.get(normalized_kind, _BC_TYPE_TO_INDEX["other"])] = 1.0


def _mark_from_cgns_flags(features: np.ndarray, flags: Any) -> bool:
    flag_array = _as_array(flags)
    if flag_array is None:
        # Returns `False` to the caller for the next stage.
        return False
    flat = flag_array.reshape(-1)
    if flat.shape[0] != features.shape[0]:
        return False

    # Stores the `marked` for use by the remaining steps in this flow.
    marked = False
    for raw_flag in np.unique(flat):
        kind = _normalize_cgns_flag_kind(raw_flag)
        if not kind:
            # Handles the data or control-flow case represented by this statement.
            continue
        indices = np.where(flat == raw_flag)[0].astype(np.int64)
        _mark_features(features, indices, kind)
        marked = marked or indices.size > 0
    # Returns `marked` to the caller for the next stage.
    return marked


def _aggregate_point_features_to_cells(
    point_features: np.ndarray,
    connectivity: Any,
    # Continues the surrounding statement or collection definition.
    num_cells: int,
    num_points: int,
) -> np.ndarray:
    cell_features = np.zeros((num_cells, point_features.shape[1]), dtype=np.float32)
    # Stores the `cells` for use by the remaining steps in this flow.
    cells = list(_iter_cell_nodes(connectivity, num_points))
    for cell_index, nodes in enumerate(cells[:num_cells]):
        if nodes.size == 0:
            continue
        # Stores the `cell_features[cell_index]` for use by the remaining steps in this flow.
        cell_features[cell_index] = point_features[nodes].max(axis=0)
    return cell_features


def _aggregate_cell_features_to_points(
    cell_features: np.ndarray,
    # Continues the surrounding statement or collection definition.
    connectivity: Any,
    num_points: int,
) -> np.ndarray:
    point_features = np.zeros((num_points, cell_features.shape[1]), dtype=np.float32)
    # Stores the `cells` for use by the remaining steps in this flow.
    cells = list(_iter_cell_nodes(connectivity, num_points))
    for cell_index, nodes in enumerate(cells[: cell_features.shape[0]]):
        if nodes.size == 0:
            continue
        # Stores the `point_features[nodes]` for use by the remaining steps in this flow.
        point_features[nodes] = np.maximum(point_features[nodes], cell_features[cell_index])
    return point_features


def _iter_ordered_cell_nodes(connectivity: Any, num_points: int) -> Iterable[np.ndarray]:
    if connectivity is None:
        # Returns `[]` to the caller for the next stage.
        return []

    raw_connectivity = np.asarray(connectivity, dtype=object)
    if raw_connectivity.ndim == 1:
        raw_cells = [raw_connectivity] if all(np.isscalar(value) for value in raw_connectivity) else list(raw_connectivity)
    # Continues the surrounding statement or collection definition.
    else:
        raw_cells = [row for row in raw_connectivity]

    cells = []
    for cell in raw_cells:
        # Stores the `nodes` for use by the remaining steps in this flow.
        nodes = _normalize_indices(cell, num_points)
        if nodes.size > 0:
            cells.append(nodes)
    return cells


# Defines `_get_volume_face_signatures` for  get volume face signatures logic.
def _get_volume_face_signatures(element_type: int, nodes: np.ndarray) -> set:
    corner_nodes = cgnsmesh.extract_corner_nodes(nodes, elem_type=element_type)
    if element_type in cgnskeywords.ElementType_tetra:
        templates = ((0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3))
    # Checks `element_type in cgnskeywords.ElementType_pyra` before entering this conditional branch.
    elif element_type in cgnskeywords.ElementType_pyra:
        templates = ((0, 1, 2, 3), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4))
    elif element_type in cgnskeywords.ElementType_penta:
        templates = ((0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5))
    # Checks `element_type in cgnskeywords.ElementType_hexa` before entering this conditional branch.
    elif element_type in cgnskeywords.ElementType_hexa:
        templates = (
            (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
            (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
        # Completes the multi-line `collection` arguments or elements.
        )
    else:
        return set()
    return {
        # Completes the multi-line `frozenset` arguments or elements.
        frozenset(int(corner_nodes[index]) for index in template)
        for template in templates
        if max(template) < len(corner_nodes)
    }


# Defines `_get_boundary_entity_signatures` for  get boundary entity signatures logic.
def _get_boundary_entity_signatures(
    entity_nodes: np.ndarray,
    grid_location: str,
) -> set:
    # Stores the `nodes` for use by the remaining steps in this flow.
    nodes = np.asarray(entity_nodes, dtype=np.int64).reshape(-1)
    if grid_location == "edgecenter":
        return {frozenset(int(node) for node in nodes[:2])} if nodes.size >= 2 else set()
    if nodes.size == 3:
        # Returns `{frozenset(int(node) for node in nodes[:3])}` to the caller for the next stage.
        return {frozenset(int(node) for node in nodes[:3])}
    if nodes.size == 4:
        return {frozenset(int(node) for node in nodes[:4])}
    if nodes.size >= 5:
        # Returns `` to the caller for the next stage.
        return {
            frozenset(int(node) for node in nodes[:3]),
            frozenset(int(node) for node in nodes[:4]),
        }
    # Returns `set()` to the caller for the next stage.
    return set()


def _resolve_topological_adjacent_cells(
    entity_nodes: np.ndarray,
    connectivity: Any,
    # Continues the surrounding statement or collection definition.
    num_points: int,
    element_type_ids: Any,
    grid_location: str,
) -> np.ndarray:
    """按体单元真实边/面模板解析相邻单元，缺少类型信息时兼容旧匹配。"""
    # Stores the `normalized_entity_nodes` for use by the remaining steps in this flow.
    normalized_entity_nodes = np.unique(np.asarray(entity_nodes, dtype=np.int64).reshape(-1))
    if normalized_entity_nodes.size == 0:
        return np.empty((0,), dtype=np.int64)

    type_ids = _as_array(element_type_ids)
    # Stores the `ordered_cells` for use by the remaining steps in this flow.
    ordered_cells = list(_iter_ordered_cell_nodes(connectivity, num_points))
    if type_ids is not None and type_ids.reshape(-1).size >= len(ordered_cells):
        entity_signatures = _get_boundary_entity_signatures(
            entity_nodes=normalized_entity_nodes,
            # Completes the multi-line `_get_boundary_entity_signatures` arguments or elements.
            grid_location=grid_location,
        )
        if entity_signatures:
            matched_cells = []
            # Stores the `has_resolvable_topology` for use by the remaining steps in this flow.
            has_resolvable_topology = False
            for cell_index, (cell_nodes, type_id) in enumerate(
                zip(ordered_cells, type_ids.reshape(-1))
            ):
                # Keeps the operation and its recovery path within the same exception boundary.
                try:
                    resolved_type_id = int(type_id)
                    if grid_location == "edgecenter" or resolved_type_id in cgnskeywords.ElementType2D:
                        cell_signatures = {
                            # Completes the multi-line `frozenset` arguments or elements.
                            frozenset(int(node) for node in edge)
                            for edge in cgnsmesh.get_edges(resolved_type_id, cell_nodes)
                        }
                    else:
                        # Stores the `cell_signatures` for use by the remaining steps in this flow.
                        cell_signatures = _get_volume_face_signatures(resolved_type_id, cell_nodes)
                except (TypeError, ValueError):
                    continue
                if cell_signatures:
                    # Stores the `has_resolvable_topology` for use by the remaining steps in this flow.
                    has_resolvable_topology = True
                if entity_signatures & cell_signatures:
                    matched_cells.append(cell_index)
            if has_resolvable_topology:
                # Returns `np.asarray(matched_cells, dtype=np.int64)` to the caller for the next stage.
                return np.asarray(matched_cells, dtype=np.int64)

    node_to_cells = [set() for _ in range(num_points)]
    for cell_index, cell_nodes in enumerate(_iter_cell_nodes(connectivity, num_points)):
        for node_index in cell_nodes:
            # Adds the current value to `node_to_cells[int(node_index)]` for later aggregation.
            node_to_cells[int(node_index)].add(cell_index)

    candidate_cells = None
    for node_index in normalized_entity_nodes:
        if node_index < 0 or node_index >= num_points:
            # Returns `np.empty((0,), dtype=np.int64)` to the caller for the next stage.
            return np.empty((0,), dtype=np.int64)
        node_cells = node_to_cells[int(node_index)]
        candidate_cells = set(node_cells) if candidate_cells is None else candidate_cells & node_cells
        if not candidate_cells:
            # Returns `np.empty((0,), dtype=np.int64)` to the caller for the next stage.
            return np.empty((0,), dtype=np.int64)

    resolved_cells = candidate_cells if candidate_cells is not None else set()
    return np.asarray(sorted(resolved_cells), dtype=np.int64)


def convert_features_to_location(
    # Continues the surrounding statement or collection definition.
    features: np.ndarray,
    source_location: str,
    target_location: str,
    data_dict: Dict[str, Any],
# Continues the surrounding statement or collection definition.
) -> np.ndarray:
    resolved_source = str(source_location or "").strip().lower()
    resolved_target = str(target_location or "").strip().lower()
    if resolved_source == resolved_target:
        # Returns `features.astype(np.float32, copy=False)` to the caller for the next stage.
        return features.astype(np.float32, copy=False)

    point_coordinates = _as_array(data_dict.get("PointCoordinates"))
    cell_coordinates = _as_array(data_dict.get("CellCoordinates"))
    num_points = int(point_coordinates.shape[0]) if point_coordinates is not None and point_coordinates.ndim >= 2 else 0
    # Stores the `num_cells` for use by the remaining steps in this flow.
    num_cells = int(cell_coordinates.shape[0]) if cell_coordinates is not None and cell_coordinates.ndim >= 2 else 0

    if resolved_source == "point" and resolved_target == "cell":
        return _aggregate_point_features_to_cells(
            point_features=features,
            # Completes the multi-line `data_dict.get` arguments or elements.
            connectivity=data_dict.get("ElementConnectivity"),
            num_cells=num_cells,
            num_points=num_points,
        )
    # Returns early when `resolved_source == "cell" and resolved_target == "point"` makes the remaining work inapplicable.
    if resolved_source == "cell" and resolved_target == "point":
        return _aggregate_cell_features_to_points(
            cell_features=features,
            connectivity=data_dict.get("ElementConnectivity"),
            # Completes the multi-line `_aggregate_cell_features_to_points` arguments or elements.
            num_points=num_points,
        )
    raise ValueError(f"不支持的 field_bc 位置转换: {source_location!r} -> {target_location!r}")


def _build_from_cgns_boundary_conditions(
    # Completes the multi-line `collection` arguments or elements.
    data_dict: Dict[str, Any],
    target_location: str,
    num_points: int,
    num_cells: int,
# Completes the multi-line `collection` arguments or elements.
) -> Tuple[Optional[np.ndarray], str]:
    boundary_conditions = data_dict.get("BoundaryConditions") or {}
    if not isinstance(boundary_conditions, dict) or not boundary_conditions:
        return None, "none"

    # Stores the `point_features` for use by the remaining steps in this flow.
    point_features = np.zeros((num_points, len(BC_FEATURE_NAMES)), dtype=np.float32)
    vertex_point_features = np.zeros_like(point_features)
    face_cell_features = np.zeros((num_cells, len(BC_FEATURE_NAMES)), dtype=np.float32)
    cell_center_features = np.zeros_like(face_cell_features)
    # Stores the `marked` for use by the remaining steps in this flow.
    marked = False

    explicit_conditions = [
        payload
        for name, payload in boundary_conditions.items()
        # Completes the multi-line `isinstance` arguments or elements.
        if name != "bc_flags" and isinstance(payload, dict)
    ]
    bc_flags = boundary_conditions.get("bc_flags")
    if not explicit_conditions and bc_flags is not None and _mark_from_cgns_flags(
        # Completes the multi-line `_mark_from_cgns_flags` arguments or elements.
        point_features,
        bc_flags,
    ):
        vertex_point_features[:] = point_features
        # Stores the `marked` for use by the remaining steps in this flow.
        marked = True

    for bc_name, bc_payload in boundary_conditions.items():
        if bc_name == "bc_flags" or not isinstance(bc_payload, dict):
            continue

        # Stores the `grid_location` for use by the remaining steps in this flow.
        grid_location = str(bc_payload.get("grid_loc") or "Vertex").strip().lower()
        kind = _normalize_bc_kind(bc_payload.get("type") or bc_name)
        indices = np.empty((0,), dtype=np.int64)
        indices_are_zero_based = bool(bc_payload.get("boundary_indices_zero_based"))

        # Checks `grid_location == "vertex"` before entering this conditional branch.
        if grid_location == "vertex":
            if indices_are_zero_based:
                indices = np.asarray(
                    bc_payload.get("boundary_point_indices", []),
                    # Completes the multi-line `np.asarray(bc_payload.get('boundary_point_indices', []), dtype=np.int64).reshape` arguments or elements.
                    dtype=np.int64,
                ).reshape(-1)
                indices = indices[(indices >= 0) & (indices < num_points)]
            else:
                # Reads `point_list` from the mapping into the `point_list`.
                point_list = bc_payload.get("point_list")
                if point_list is not None:
                    indices = _normalize_indices(point_list, num_points)
                else:
                    # Reads `point_range` from the mapping into the `point_range`.
                    point_range = bc_payload.get("point_range")
                    range_values = _normalize_indices(point_range, num_points)
                    if range_values.size >= 2:
                        start, end = int(range_values.min()), int(range_values.max())
                        # Stores the `indices` for use by the remaining steps in this flow.
                        indices = np.arange(start, end + 1, dtype=np.int64)
            if indices.size > 0:
                _mark_features(point_features, indices, kind)
                _mark_features(vertex_point_features, indices, kind)
                # Stores the `marked` for use by the remaining steps in this flow.
                marked = True
        elif grid_location in {"facecenter", "edgecenter"}:
            face_nodes = _as_array(bc_payload.get("boundary_point_indices"))
            if face_nodes is None:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(
                    f"无法解析 GridLocation={grid_location} 的边界实体节点: {bc_name}"
                )
            indices = np.unique(np.asarray(face_nodes, dtype=np.int64).reshape(-1))
            # Checks `not indices_are_zero_based` before entering this conditional branch.
            if not indices_are_zero_based:
                indices = _normalize_indices(indices, num_points)
            else:
                indices = indices[(indices >= 0) & (indices < num_points)]
            # Validates `indices.size == 0` and raises `ValueError` when the input cannot be processed safely.
            if indices.size == 0:
                raise ValueError(
                    f"GridLocation={grid_location} 的边界实体节点为空: {bc_name}"
                )

            # Calls `_mark_features` to perform the operation required at this stage.
            _mark_features(point_features, indices, kind)
            entity_node_groups = (
                bc_payload.get("boundary_face_nodes")
                if grid_location == "facecenter"
                # Completes the multi-line `bc_payload.get` arguments or elements.
                else bc_payload.get("boundary_entity_nodes")
            )
            if entity_node_groups is None:
                entity_node_groups = [indices]
            # Checks `isinstance(entity_node_groups, np.ndarray)` before entering this conditional branch.
            elif isinstance(entity_node_groups, np.ndarray):
                entity_node_groups = (
                    [entity_node_groups]
                    if entity_node_groups.ndim <= 1
                    # Completes the multi-line `list` arguments or elements.
                    else list(entity_node_groups)
                )
            for entity_node_group in entity_node_groups:
                adjacent_cells = _resolve_topological_adjacent_cells(
                    # Completes the multi-line `np.asarray` arguments or elements.
                    entity_nodes=np.asarray(entity_node_group, dtype=np.int64),
                    connectivity=data_dict.get("ElementConnectivity"),
                    num_points=num_points,
                    element_type_ids=data_dict.get("ElementTypeIds"),
                    # Completes the multi-line `_resolve_topological_adjacent_cells` arguments or elements.
                    grid_location=grid_location,
                )
                _mark_features(face_cell_features, adjacent_cells, kind)
            marked = True
        # Checks `grid_location == "cellcenter"` before entering this conditional branch.
        elif grid_location == "cellcenter":
            if indices_are_zero_based:
                cell_indices = np.asarray(
                    bc_payload.get("boundary_cell_indices", []),
                    # Completes the multi-line `np.asarray(bc_payload.get('boundary_cell_indices', []), dtype=np.int64).reshape` arguments or elements.
                    dtype=np.int64,
                ).reshape(-1)
                cell_indices = cell_indices[(cell_indices >= 0) & (cell_indices < num_cells)]
            else:
                # Stores the `cell_indices` for use by the remaining steps in this flow.
                cell_indices = _normalize_indices(
                    bc_payload.get("boundary_cell_indices"),
                    num_cells,
                )
            # Validates `cell_indices.size == 0` and raises `ValueError` when the input cannot be processed safely.
            if cell_indices.size == 0:
                raise ValueError(f"CellCenter 边界单元为空: {bc_name}")
            _mark_features(cell_center_features, cell_indices, kind)
            marked = True
        # Continues the surrounding statement or collection definition.
        else:
            raise ValueError(
                f"cgns_based 暂不支持 GridLocation={grid_location}: {bc_name}"
            )

    # Returns early when `not marked` makes the remaining work inapplicable.
    if not marked:
        return None, "none"

    if target_location == "point":
        return np.maximum(
            # Completes the multi-line `collection` arguments or elements.
            point_features,
            _aggregate_cell_features_to_points(
                cell_features=cell_center_features,
                connectivity=data_dict.get("ElementConnectivity"),
                # Completes the multi-line `_aggregate_cell_features_to_points` arguments or elements.
                num_points=num_points,
            ),
        ), "cgns_boundary_conditions"

    cell_features = _aggregate_point_features_to_cells(
        # Completes the multi-line `_aggregate_point_features_to_cells` arguments or elements.
        point_features=vertex_point_features,
        connectivity=data_dict.get("ElementConnectivity"),
        num_cells=num_cells,
        num_points=num_points,
    # Completes the multi-line `_aggregate_point_features_to_cells` arguments or elements.
    )
    return np.maximum(
        np.maximum(cell_features, face_cell_features),
        cell_center_features,
    # Completes the multi-line `collection` arguments or elements.
    ), "cgns_boundary_conditions"


def build_boundary_feature_matrix(
    data_dict: Dict[str, Any],
    target_location: str,
# Completes the multi-line `collection` arguments or elements.
) -> Tuple[np.ndarray, List[str], str]:
    """构建 cgns_based 固定通道的局部边界条件特征矩阵。"""
    resolved_location = str(target_location or "point").strip().lower()
    if resolved_location not in {"point", "cell"}:
        raise ValueError(f"局部边界条件特征仅支持 point/cell 位置，当前为 {target_location!r}")

    # Stores the `point_coordinates` for use by the remaining steps in this flow.
    point_coordinates = _as_array(data_dict.get("PointCoordinates"))
    cell_coordinates = _as_array(data_dict.get("CellCoordinates"))
    num_points = int(point_coordinates.shape[0]) if point_coordinates is not None and point_coordinates.ndim >= 2 else 0
    num_cells = int(cell_coordinates.shape[0]) if cell_coordinates is not None and cell_coordinates.ndim >= 2 else 0
    # Stores the `target_count` for use by the remaining steps in this flow.
    target_count = num_points if resolved_location == "point" else num_cells
    if target_count <= 0:
        return (
            np.zeros((0, len(BC_FEATURE_NAMES)), dtype=np.float32),
            # Completes the multi-line `list` arguments or elements.
            list(BC_FEATURE_NAMES),
            "none",
        )

    features, source = _build_from_cgns_boundary_conditions(
        # Completes the multi-line `_build_from_cgns_boundary_conditions` arguments or elements.
        data_dict=data_dict,
        target_location=resolved_location,
        num_points=num_points,
        num_cells=num_cells,
    # Completes the multi-line `_build_from_cgns_boundary_conditions` arguments or elements.
    )
    if features is not None and features.shape[0] == target_count:
        return features.astype(np.float32, copy=False), list(BC_FEATURE_NAMES), source

    return (
        # Completes the multi-line `np.zeros` arguments or elements.
        np.zeros((target_count, len(BC_FEATURE_NAMES)), dtype=np.float32),
        list(BC_FEATURE_NAMES),
        "none",
    )


# Defines `infer_field_location` for infer field location logic.
def infer_field_location(
    data_dict: Dict[str, Any],
    field_name: str,
    values: Any,
# Continues the surrounding statement or collection definition.
) -> str:
    array = _as_array(values)
    if array is None:
        raise ValueError(f"field_bc={field_name!r} 无法转为数组")

    # Stores the `location_map` for use by the remaining steps in this flow.
    location_map = data_dict.get("FieldLocationMap") or {}
    configured_location = str(location_map.get(field_name) or "").strip().lower()
    if configured_location in {"point", "cell"}:
        return configured_location

    # Stores the `point_coordinates` for use by the remaining steps in this flow.
    point_coordinates = _as_array(data_dict.get("PointCoordinates"))
    cell_coordinates = _as_array(data_dict.get("CellCoordinates"))
    num_points = int(point_coordinates.shape[0]) if point_coordinates is not None and point_coordinates.ndim >= 2 else 0
    num_cells = int(cell_coordinates.shape[0]) if cell_coordinates is not None and cell_coordinates.ndim >= 2 else 0

    # Validates `array.ndim == 0` and raises `ValueError` when the input cannot be processed safely.
    if array.ndim == 0:
        raise ValueError(f"field_bc={field_name!r} 不能是标量字段")
    leading = int(array.shape[0])
    if leading == num_points:
        # Returns `"point"` to the caller for the next stage.
        return "point"
    if leading == num_cells:
        return "cell"
    if array.ndim >= 2 and int(array.shape[1]) == num_points:
        # Returns `"point"` to the caller for the next stage.
        return "point"
    if array.ndim >= 2 and int(array.shape[1]) == num_cells:
        return "cell"
    raise ValueError(
        # Completes the multi-line `ValueError` arguments or elements.
        f"field_bc={field_name!r} 长度无法匹配点数或单元数: "
        f"field={leading}, points={num_points}, cells={num_cells}"
    )


def resolve_static_field_bc_values(
    # Completes the multi-line `collection` arguments or elements.
    data_dict: Dict[str, Any],
    field_name: str,
    values: Any,
    location: Optional[str] = None,
# Continues the surrounding statement or collection definition.
) -> np.ndarray:
    """提取静态局部 BC，兼容时序样本重复存储的边界字段。"""
    array = _as_array(values)
    if array is None:
        raise ValueError(f"field_bc={field_name!r} 无法转为数组")

    # Stores the `resolved_location` for use by the remaining steps in this flow.
    resolved_location = location or infer_field_location(data_dict, field_name, array)
    point_coordinates = _as_array(data_dict.get("PointCoordinates"))
    cell_coordinates = _as_array(data_dict.get("CellCoordinates"))
    target_count = (
        # Completes the multi-line `int` arguments or elements.
        int(point_coordinates.shape[0])
        if resolved_location == "point" and point_coordinates is not None and point_coordinates.ndim >= 2
        else int(cell_coordinates.shape[0])
        if resolved_location == "cell" and cell_coordinates is not None and cell_coordinates.ndim >= 2
        # Continues the surrounding statement or collection definition.
        else 0
    )
    if target_count <= 0:
        raise ValueError(
            # Completes the multi-line `ValueError` arguments or elements.
            f"field_bc={field_name!r} 无法确定 {resolved_location} 位置的空间数量"
        )
    if array.ndim == 0:
        raise ValueError(f"field_bc={field_name!r} 不能是标量字段")

    # Returns early when `int(array.shape[0]) == target_count` makes the remaining work inapplicable.
    if int(array.shape[0]) == target_count:
        return np.asarray(array)
    if array.ndim >= 2 and int(array.shape[1]) == target_count:
        first_step = np.asarray(array[0])
        # Validates `` and raises `ValueError` when the input cannot be processed safely.
        if not np.array_equal(array, np.broadcast_to(first_step, array.shape)):
            raise ValueError(
                f"field_bc={field_name!r} 在时间步之间发生变化；"
                "vtk_based 仅支持静态局部边界字段"
            # Completes the multi-line `ValueError` arguments or elements.
            )
        return first_step

    raise ValueError(
        f"field_bc={field_name!r} 的空间维度无法匹配 {resolved_location} 位置数量: "
        # Completes the multi-line `tuple` arguments or elements.
        f"shape={tuple(array.shape)}, count={target_count}"
    )


def infer_field_bc_mode(values: Any, field_name: str) -> str:
    array = _as_array(values)
    # Validates `array is None` and raises `ValueError` when the input cannot be processed safely.
    if array is None:
        raise ValueError(f"field_bc={field_name!r} 无法转为数组")
    if np.issubdtype(array.dtype, np.bool_) or np.issubdtype(array.dtype, np.integer):
        return "categorical"
    # Returns early when `np.issubdtype(array.dtype, np.floating)` makes the remaining work inapplicable.
    if np.issubdtype(array.dtype, np.floating):
        return "continuous"
    raise ValueError(
        f"field_bc={field_name!r} 仅支持 bool/int/float 字段，当前 dtype={array.dtype}"
    # Completes the multi-line `ValueError` arguments or elements.
    )


def _field_bc_channel_count(values: Any, field_name: str) -> int:
    array = _as_array(values)
    if array is None:
        # Raises `ValueError` to report why this path cannot continue.
        raise ValueError(f"field_bc={field_name!r} 无法转为数组")
    if array.ndim == 1:
        return 1
    if array.ndim == 2:
        # Returns `int(array.shape[1])` to the caller for the next stage.
        return int(array.shape[1])
    raise ValueError(f"field_bc={field_name!r} 仅支持一维或二维字段，当前维度={array.ndim}")


def _continuous_field_bc_matrix(values: Any, field_name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    # Checks `array.ndim == 1` before entering this conditional branch.
    if array.ndim == 1:
        matrix = array.reshape(-1, 1)
    elif array.ndim == 2:
        matrix = array
    # Continues the surrounding statement or collection definition.
    else:
        raise ValueError(f"field_bc={field_name!r} 连续字段仅支持一维或二维，当前维度={array.ndim}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"field_bc={field_name!r} 连续字段包含非有限值")
    # Returns `matrix.astype(np.float32, copy=False)` to the caller for the next stage.
    return matrix.astype(np.float32, copy=False)


def _categorical_field_bc_matrix(values: Any, field_schema: Dict[str, Any]) -> np.ndarray:
    field_name = str(field_schema["field_bc_name"])
    array = _as_array(values)
    # Validates `array is None` and raises `ValueError` when the input cannot be processed safely.
    if array is None:
        raise ValueError(f"field_bc={field_name!r} 无法转为数组")
    if array.ndim != 1:
        array = array.reshape(-1)

    # Keeps the operation and its recovery path within the same exception boundary.
    try:
        int_values = np.asarray(array).astype(np.int64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"field_bc={field_name!r} categorical 字段必须为整数或布尔值") from exc

    # Builds the `categories` using the comprehension's filter and mapping rules.
    categories = [int(category) for category in field_schema.get("categories_bc", [])]
    category_to_index = {category: index for index, category in enumerate(categories)}
    features = np.zeros((int_values.shape[0], len(categories)), dtype=np.float32)

    observed = set(int(value) for value in np.unique(int_values))
    # Stores the `unknown` for use by the remaining steps in this flow.
    unknown = sorted(value for value in observed if value != 0 and value not in category_to_index)
    if unknown:
        raise ValueError(
            f"field_bc={field_name!r} 出现训练类别表之外的类别: {unknown}; "
            # Completes the multi-line `ValueError` arguments or elements.
            f"已知类别={categories}"
        )

    for category, column in category_to_index.items():
        features[int_values == category, column] = 1.0
    # Returns `features` to the caller for the next stage.
    return features


def fit_field_bc_schema(
    data_items: Iterable[Dict[str, Any]],
    field_bc_names: List[str],
# Completes the multi-line `collection` arguments or elements.
) -> Dict[str, Any]:
    field_states: Dict[str, Dict[str, Any]] = {
        field_name: {
            "field_bc_name": field_name,
            # Completes the multi-line `collection` arguments or elements.
            "field_bc_mode": None,
            "field_bc_location": None,
            "categories_bc": set(),
            "channel_count_bc": None,
        # Completes the multi-line `collection` arguments or elements.
        }
        for field_name in field_bc_names
    }

    for data_dict in data_items:
        # Stores the `fields` for use by the remaining steps in this flow.
        fields = data_dict.get("Fields") or {}
        if not isinstance(fields, dict):
            raise ValueError("vtk_based 需要样本包含 Fields 字典")

        for field_name in field_bc_names:
            # Validates `field_name not in fields` and raises `ValueError` when the input cannot be processed safely.
            if field_name not in fields:
                raise ValueError(f"Fields 中找不到 field_bc={field_name!r}")
            raw_values = fields[field_name]
            location = infer_field_location(data_dict, field_name, raw_values)
            # Stores the `static_values` for use by the remaining steps in this flow.
            static_values = resolve_static_field_bc_values(
                data_dict=data_dict,
                field_name=field_name,
                values=raw_values,
                # Completes the multi-line `resolve_static_field_bc_values` arguments or elements.
                location=location,
            )
            mode = infer_field_bc_mode(static_values, field_name)
            state = field_states[field_name]
            # Checks `state["field_bc_mode"] is None` before entering this conditional branch.
            if state["field_bc_mode"] is None:
                state["field_bc_mode"] = mode
            elif state["field_bc_mode"] != mode:
                raise ValueError(
                    # Completes the multi-line `ValueError` arguments or elements.
                    f"field_bc={field_name!r} 在训练集中 dtype 模式不一致: "
                    f"{state['field_bc_mode']} vs {mode}"
                )

            if state["field_bc_location"] is None:
                # Stores the `state['field_bc_location']` for use by the remaining steps in this flow.
                state["field_bc_location"] = location
            elif state["field_bc_location"] != location:
                raise ValueError(
                    f"field_bc={field_name!r} 在训练集中位置不一致: "
                    # Completes the multi-line `ValueError` arguments or elements.
                    f"{state['field_bc_location']} vs {location}"
                )

            if mode == "categorical":
                values = np.asarray(static_values).reshape(-1).astype(np.int64, copy=False)
                # Adds the current value to `state['categories_bc']` for later aggregation.
                state["categories_bc"].update(
                    int(value) for value in np.unique(values) if int(value) != 0
                )
            else:
                # Stores the `channel_count` for use by the remaining steps in this flow.
                channel_count = _field_bc_channel_count(static_values, field_name)
                if state["channel_count_bc"] is None:
                    state["channel_count_bc"] = channel_count
                elif int(state["channel_count_bc"]) != int(channel_count):
                    # Raises `ValueError` to report why this path cannot continue.
                    raise ValueError(
                        f"field_bc={field_name!r} 连续字段通道数不一致: "
                        f"{state['channel_count_bc']} vs {channel_count}"
                    )

    # Builds the `field_locations` using the comprehension's filter and mapping rules.
    field_locations = {
        str(state["field_bc_location"])
        for state in field_states.values()
        if state["field_bc_location"] is not None
    # Continues the surrounding statement or collection definition.
    }
    if len(field_locations) > 1:
        raise ValueError(f"多个 field_bc 的 point/cell 位置不一致: {sorted(field_locations)}")

    fields_schema = []
    # Initializes the `feature_names` as a collection for values from this processing stage.
    feature_names: List[str] = []
    categorical_offsets: List[int] = []
    continuous_offsets: List[int] = []
    offset = 0
    # Visits each item in `field_bc_names` and applies the current processing step.
    for field_name in field_bc_names:
        state = field_states[field_name]
        mode = state["field_bc_mode"]
        if mode is None:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(f"field_bc={field_name!r} 未能从训练集拟合 schema")
        if mode == "categorical":
            categories = sorted(int(value) for value in state["categories_bc"])
            if not categories:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(f"field_bc={field_name!r} 训练集未出现非 0 类别")
            names = [f"{field_name}__bc_{category}" for category in categories]
            categorical_indices = list(range(offset, offset + len(names)))
            continuous_indices: List[int] = []
            # Initializes the `field_schema` as a collection for values from this processing stage.
            field_schema = {
                "field_bc_name": field_name,
                "field_bc_mode": "categorical",
                "field_bc_location": state["field_bc_location"],
                # Completes the multi-line `collection` arguments or elements.
                "categories_bc": categories,
                "feature_bc_names": names,
                "feature_bc_offsets": categorical_indices,
                "categorical_bc_offsets": categorical_indices,
                # Completes the multi-line `collection` arguments or elements.
                "continuous_bc_offsets": continuous_indices,
            }
            categorical_offsets.extend(categorical_indices)
        else:
            # Converts the `channel_count` to `int` so later checks use a consistent type.
            channel_count = int(state["channel_count_bc"] or 1)
            names = (
                [f"{field_name}__bc"]
                if channel_count == 1
                # Completes the multi-line `range` arguments or elements.
                else [f"{field_name}__bc_{index}" for index in range(channel_count)]
            )
            categorical_indices = []
            continuous_indices = list(range(offset, offset + len(names)))
            # Initializes the `field_schema` as a collection for values from this processing stage.
            field_schema = {
                "field_bc_name": field_name,
                "field_bc_mode": "continuous",
                "field_bc_location": state["field_bc_location"],
                # Completes the multi-line `collection` arguments or elements.
                "feature_bc_names": names,
                "feature_bc_offsets": continuous_indices,
                "categorical_bc_offsets": categorical_indices,
                "continuous_bc_offsets": continuous_indices,
            # Completes the multi-line `collection` arguments or elements.
            }
            continuous_offsets.extend(continuous_indices)

        fields_schema.append(field_schema)
        feature_names.extend(names)
        # Stores the `offset` for use by the remaining steps in this flow.
        offset += len(names)

    return {
        "local_feature_encoding": "vtk_based",
        "field_bc": list(field_bc_names),
        # Completes the multi-line `next` arguments or elements.
        "field_bc_location": next(iter(field_locations)) if field_locations else None,
        "fields_bc_schema": fields_schema,
        "feature_bc_names": feature_names,
        "feature_bc_offsets": list(range(offset)),
        # Completes the multi-line `collection` arguments or elements.
        "categorical_bc_offsets": categorical_offsets,
        "continuous_bc_offsets": continuous_offsets,
    }


def build_field_bc_feature_matrix(
    # Completes the multi-line `collection` arguments or elements.
    data_dict: Dict[str, Any],
    target_location: str,
    schema: Dict[str, Any],
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    # Stores the `fields` for use by the remaining steps in this flow.
    fields = data_dict.get("Fields") or {}
    if not isinstance(fields, dict):
        raise ValueError("vtk_based 需要样本包含 Fields 字典")

    matrices = []
    # Visits each item in `schema.get("fields_bc_schema", [])` and applies the current processing step.
    for field_schema in schema.get("fields_bc_schema", []):
        field_name = str(field_schema["field_bc_name"])
        if field_name not in fields:
            raise ValueError(f"Fields 中找不到 field_bc={field_name!r}")

        # Selects the required item from `fields` as the `raw_values`.
        raw_values = fields[field_name]
        location = infer_field_location(data_dict, field_name, raw_values)
        expected_location = str(field_schema.get("field_bc_location") or "").lower()
        if expected_location and location != expected_location:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(
                f"field_bc={field_name!r} 位置与训练 schema 不一致: "
                f"当前={location}, 训练={expected_location}"
            )

        # Stores the `static_values` for use by the remaining steps in this flow.
        static_values = resolve_static_field_bc_values(
            data_dict=data_dict,
            field_name=field_name,
            values=raw_values,
            # Completes the multi-line `resolve_static_field_bc_values` arguments or elements.
            location=location,
        )

        mode = str(field_schema.get("field_bc_mode") or "").lower()
        if mode == "categorical":
            # Stores the `source_features` for use by the remaining steps in this flow.
            source_features = _categorical_field_bc_matrix(static_values, field_schema)
        elif mode == "continuous":
            source_features = _continuous_field_bc_matrix(static_values, field_name)
            expected_channels = len(field_schema.get("feature_bc_names", []))
            # Validates `expected_channels != int(source_features.shape[1])` and raises `ValueError` when the input cannot be processed safely.
            if expected_channels != int(source_features.shape[1]):
                raise ValueError(
                    f"field_bc={field_name!r} 连续字段通道数与训练 schema 不一致: "
                    f"当前={source_features.shape[1]}, 训练={expected_channels}"
                # Completes the multi-line `ValueError` arguments or elements.
                )
        else:
            raise ValueError(f"field_bc={field_name!r} schema 模式无效: {mode!r}")

        matrices.append(
            # Completes the multi-line `convert_features_to_location` arguments or elements.
            convert_features_to_location(
                features=source_features,
                source_location=location,
                target_location=target_location,
                # Completes the multi-line `convert_features_to_location` arguments or elements.
                data_dict=data_dict,
            )
        )

    if matrices:
        # Stores the `features` for use by the remaining steps in this flow.
        features = np.concatenate(matrices, axis=1).astype(np.float32, copy=False)
    else:
        resolved_location = str(target_location or "point").strip().lower()
        point_coordinates = _as_array(data_dict.get("PointCoordinates"))
        # Stores the `cell_coordinates` for use by the remaining steps in this flow.
        cell_coordinates = _as_array(data_dict.get("CellCoordinates"))
        if resolved_location == "point" and point_coordinates is not None and point_coordinates.ndim >= 2:
            target_count = int(point_coordinates.shape[0])
        elif resolved_location == "cell" and cell_coordinates is not None and cell_coordinates.ndim >= 2:
            target_count = int(cell_coordinates.shape[0])
        else:
            target_count = 0
        features = np.zeros((target_count, 0), dtype=np.float32)

    return features, list(schema.get("feature_bc_names", [])), schema
