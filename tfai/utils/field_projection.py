from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tfai.io_utils.io_vtk import build_point_to_cells, cell_to_point, point_to_cell


def normalize_field_location(location: Optional[str]) -> str:
    text = str(location or "").strip().lower()
    if text not in {"point", "cell"}:
        raise ValueError(f"非法字段位置: {location}")
    return text


def build_component_location_map(
    component_map: Dict[str, List[str]],
    location: str,
) -> Dict[str, str]:
    normalized_location = normalize_field_location(location)
    location_map: Dict[str, str] = {}
    for raw_name, component_names in (component_map or {}).items():
        if not isinstance(raw_name, str):
            continue
        location_map[raw_name] = normalized_location
        for component_name in component_names or []:
            if isinstance(component_name, str):
                location_map[component_name] = normalized_location
    return location_map


def merge_field_location_maps(*location_maps: Dict[str, str]) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for location_map in location_maps:
        for name, location in (location_map or {}).items():
            normalized_location = normalize_field_location(location)
            existing = merged.get(name)
            if existing is not None and existing != normalized_location:
                raise ValueError(
                    f"字段位置冲突: field={name}, existing={existing}, incoming={normalized_location}"
                )
            merged[name] = normalized_location
    return merged


def resolve_single_requested_name(
    logical_var: str,
    component_map: Dict[str, List[str]],
) -> List[str]:
    components = component_map.get(logical_var)
    if components:
        return list(components)

    normalized = str(logical_var).casefold()
    matched_key = None
    for key in component_map.keys():
        if str(key).casefold() != normalized:
            continue
        if matched_key is not None and matched_key != key:
            raise ValueError(
                "逻辑变量名在大小写不敏感匹配下存在冲突: "
                f"conflict={matched_key} vs {key}"
            )
        matched_key = key

    if not matched_key:
        return []
    return list(component_map.get(matched_key, []))


def choose_projected_point_or_cell(
    requested_vars: Optional[List[str]],
    point_component_map: Dict[str, List[str]],
    cell_component_map: Dict[str, List[str]],
    default_location: str = "point",
) -> str:
    if requested_vars:
        requested = list(dict.fromkeys(requested_vars))
        point_hits: List[str] = []
        cell_hits: List[str] = []
        missing: List[str] = []
        point_score = 0
        cell_score = 0

        for logical_var in requested:
            point_resolved = resolve_single_requested_name(logical_var, point_component_map)
            cell_resolved = resolve_single_requested_name(logical_var, cell_component_map)

            if not point_resolved and not cell_resolved:
                missing.append(logical_var)
                continue

            if point_resolved:
                point_hits.append(logical_var)
                point_score += len(point_resolved)
            if cell_resolved:
                cell_hits.append(logical_var)
                cell_score += len(cell_resolved)

        if missing:
            raise ValueError(f"请求变量未在 point/cell 字段中找到: {missing}")

        if len(point_hits) == len(requested):
            return "point"
        if len(cell_hits) == len(requested):
            return "cell"
        if point_score >= cell_score:
            return "point"
        return "cell"

    point_components = sum(len(list(names or [])) for names in (point_component_map or {}).values())
    cell_components = sum(len(list(names or [])) for names in (cell_component_map or {}).values())
    if point_components == cell_components:
        return normalize_field_location(default_location)
    return "point" if point_components > cell_components else "cell"


def _move_entity_axis_to_front(values: np.ndarray, entity_size: int) -> Tuple[np.ndarray, int]:
    arr = np.asarray(values)
    if arr.ndim == 0:
        raise ValueError("字段值不能是标量字面量")

    if arr.shape[0] == entity_size and (arr.ndim == 1 or arr.shape[-1] != entity_size):
        return arr, 0
    if arr.shape[-1] == entity_size:
        return np.moveaxis(arr, -1, 0), -1
    if arr.shape[0] == entity_size:
        return arr, 0

    raise ValueError(
        f"无法自动识别字段空间轴: shape={arr.shape}, expected_entity_size={entity_size}"
    )


def _restore_entity_axis(values: np.ndarray, original_axis: int) -> np.ndarray:
    if original_axis == 0:
        return values
    return np.moveaxis(values, 0, original_axis)


def project_field_values(
    values: np.ndarray,
    source_location: str,
    target_location: str,
    connectivity: Sequence[Sequence[int]],
    num_points: int,
    projection_method: str = "mean",
) -> np.ndarray:
    normalized_source = normalize_field_location(source_location)
    normalized_target = normalize_field_location(target_location)
    if normalized_source == normalized_target:
        return np.asarray(values)

    if normalized_source == "point":
        entity_size = int(num_points)
    else:
        entity_size = int(len(connectivity))

    normalized_values, original_axis = _move_entity_axis_to_front(
        np.asarray(values),
        entity_size=entity_size,
    )
    if normalized_source == "point" and normalized_target == "cell":
        projected = point_to_cell(
            normalized_values,
            connectivity=connectivity,
            method=projection_method,
            one_based=True,
        )
    elif normalized_source == "cell" and normalized_target == "point":
        point_to_cells = build_point_to_cells(
            connectivity,
            num_points=int(num_points),
            one_based=True,
        )
        projected = cell_to_point(
            normalized_values,
            point_to_cells=point_to_cells,
            method=projection_method,
        )
    else:
        raise ValueError(
            f"不支持的字段位置投影: source={normalized_source}, target={normalized_target}"
        )

    return _restore_entity_axis(np.asarray(projected), original_axis)


def project_fields_to_location(
    fields: Dict[str, np.ndarray],
    field_location_map: Dict[str, str],
    target_location: str,
    connectivity: Sequence[Sequence[int]],
    num_points: int,
    projection_method: str = "mean",
) -> Dict[str, np.ndarray]:
    projected_fields: Dict[str, np.ndarray] = {}
    normalized_target = normalize_field_location(target_location)
    for name, values in (fields or {}).items():
        source_location = field_location_map.get(name, normalized_target)
        projected_fields[name] = project_field_values(
            values=np.asarray(values),
            source_location=source_location,
            target_location=normalized_target,
            connectivity=connectivity,
            num_points=num_points,
            projection_method=projection_method,
        )
    return projected_fields
