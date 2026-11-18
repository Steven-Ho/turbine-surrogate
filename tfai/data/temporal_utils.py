import inspect
import json
import re
# Imports `xml.etree.ElementTree` for reuse in this module.
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

# Imports `BaseData` from `base_data` for use in this module.
from .base_data import BaseData


DISPLACEMENT_LOGICAL_CANDIDATES = (
    "Displacement",
    "displacement",
    # Completes the multi-line `collection` arguments or elements.
    "DISPLACEMENT",
    "Disp",
    "disp",
    "DISP",
    # Completes the multi-line `collection` arguments or elements.
    "U",
    "u",
)


def _normalize_name(name: str) -> str:
    # Returns `str(name).strip().casefold()` to the caller for the next stage.
    return str(name).strip().casefold()


def _build_displacement_component_names(base_name: str, coord_dim: int) -> List[str]:
    suffixes = ["X", "Y", "Z"]
    if coord_dim <= 1:
        # Returns `[base_name]` to the caller for the next stage.
        return [base_name]
    return [f"{base_name}{suffix}" for suffix in suffixes[:coord_dim]]


def infer_displacement_logical_name(
    field_component_map: Dict[str, Sequence[str]],
    # Continues the surrounding statement or collection definition.
    coord_dim: int,
) -> Optional[str]:
    if not isinstance(field_component_map, dict) or not field_component_map:
        return None

    # Builds the `candidate_keys` using the comprehension's filter and mapping rules.
    candidate_keys = {_normalize_name(name) for name in DISPLACEMENT_LOGICAL_CANDIDATES}
    for logical_name, components in field_component_map.items():
        if _normalize_name(logical_name) not in candidate_keys:
            continue
        # Checks `len(list(components or [])) < max(int(coord_dim), 1)` before entering this conditional branch.
        if len(list(components or [])) < max(int(coord_dim), 1):
            continue
        return str(logical_name)
    return None


# Defines `infer_displacement_effective_variables` for infer displacement effective variables logic.
def infer_displacement_effective_variables(
    field_component_map: Dict[str, Sequence[str]],
    logical_name: Optional[str],
    coord_dim: int,
# Continues the surrounding statement or collection definition.
) -> List[str]:
    if not logical_name:
        return []
    components = list(field_component_map.get(logical_name) or [])
    # Returns early when `coord_dim > 0 and len(components) >= coord_dim` makes the remaining work inapplicable.
    if coord_dim > 0 and len(components) >= coord_dim:
        return components[:coord_dim]
    return components


def infer_scalar_displacement_effective_variables(
    # Completes the multi-line `collection` arguments or elements.
    fields: Dict[str, object],
    coord_dim: int,
) -> List[str]:
    """仅为运动网格元数据识别独立存储的位移标量字段。"""
    if not isinstance(fields, dict) or coord_dim <= 0:
        # Returns `[]` to the caller for the next stage.
        return []

    normalized_fields = {
        _normalize_name(field_name): str(field_name)
        for field_name in fields
    # Continues the surrounding statement or collection definition.
    }
    suffixes = ["X", "Y", "Z"][: min(int(coord_dim), 3)]
    for base_name in DISPLACEMENT_LOGICAL_CANDIDATES:
        candidates = [f"{base_name}{suffix}" for suffix in suffixes]
        # Builds the `resolved` using the comprehension's filter and mapping rules.
        resolved = [normalized_fields.get(_normalize_name(name)) for name in candidates]
        if all(resolved):
            return [str(name) for name in resolved]
    return []


# Defines `_ensure_motion_meta` for  ensure motion meta logic.
def _ensure_motion_meta(
    aggregated_data: Dict[str, object],
    *,
    has_temporal_point_motion: bool,
    # Continues the surrounding statement or collection definition.
    has_temporal_cell_motion: bool,
    displacement_logical_name: Optional[str] = None,
    displacement_effective_variables: Optional[List[str]] = None,
    displacement_source: Optional[str] = None,
# Continues the surrounding statement or collection definition.
) -> None:
    if not has_temporal_point_motion and not has_temporal_cell_motion:
        return

    motion_meta = dict(aggregated_data.get("GeometryMotion") or {})
    # Adds the current value to `motion_meta` for later aggregation.
    motion_meta.update(
        {
            "has_temporal_coordinates": True,
            "topology_invariant": True,
            # Completes the multi-line `collection` arguments or elements.
            "geometry_regime": "displaced_static_topology",
            "has_temporal_point_coordinates": bool(has_temporal_point_motion),
            "has_temporal_cell_coordinates": bool(has_temporal_cell_motion),
            "displacement_logical_name": displacement_logical_name,
            # Completes the multi-line `list` arguments or elements.
            "displacement_effective_variables": list(displacement_effective_variables or []),
            "displacement_source": displacement_source,
            "displacement_coordinate_convention": "reference_absolute",
        }
    # Completes the multi-line `motion_meta.update` arguments or elements.
    )
    aggregated_data["GeometryMotion"] = motion_meta
    aggregated_data["GeometryRegime"] = "displaced_static_topology"


def _append_sample_motion_meta(
    # Completes the multi-line `collection` arguments or elements.
    sample_info: Dict[str, object],
    *,
    temporal_steps: int,
    geometry_regime: Optional[str],
    # Continues the surrounding statement or collection definition.
    displacement_logical_name: Optional[str],
    displacement_effective_variables: Optional[List[str]],
    displacement_source: Optional[str],
) -> Dict[str, object]:
    # Stores the `updated` for use by the remaining steps in this flow.
    updated = dict(sample_info)
    updated["temporal_steps"] = int(temporal_steps)
    if geometry_regime:
        updated["geometry_regime"] = str(geometry_regime)
    # Checks `displacement_logical_name` before entering this conditional branch.
    if displacement_logical_name:
        updated["displacement_logical_name"] = str(displacement_logical_name)
    if displacement_effective_variables:
        updated["displacement_effective_variables"] = list(displacement_effective_variables)
    # Checks `displacement_source` before entering this conditional branch.
    if displacement_source:
        updated["displacement_source"] = str(displacement_source)
    return updated


def _resolve_displacement_validation_atol(
    # Continues the surrounding statement or collection definition.
    reference_points: np.ndarray,
    temporal_point_coordinates: np.ndarray,
    displacement_values: np.ndarray,
) -> float:
    """根据存储精度和坐标量级计算位移一致性校验的绝对误差阈值。"""
    # Builds the `float_dtypes` using the comprehension's filter and mapping rules.
    float_dtypes = [
        values.dtype
        for values in (
            np.asarray(reference_points),
            # Completes the multi-line `np.asarray` arguments or elements.
            np.asarray(temporal_point_coordinates),
            np.asarray(displacement_values),
        )
        if np.issubdtype(values.dtype, np.floating)
    # Continues the surrounding statement or collection definition.
    ]
    if not float_dtypes:
        return 1e-8

    machine_epsilon = max(float(np.finfo(dtype).eps) for dtype in float_dtypes)
    # Stores the `coordinate_scale` for use by the remaining steps in this flow.
    coordinate_scale = max(
        1.0,
        float(np.max(np.abs(reference_points))),
        float(np.max(np.abs(temporal_point_coordinates))),
    # Completes the multi-line `max` arguments or elements.
    )
    # VTK 的 float32 坐标相减会累积两次舍入误差，保留少量余量以避免误拒绝有效位移场。
    return max(1e-8, 8.0 * machine_epsilon * coordinate_scale)


def validate_point_displacement_fields(
    reference_points: np.ndarray,
    # Continues the surrounding statement or collection definition.
    temporal_point_coordinates: np.ndarray,
    fields: Dict[str, np.ndarray],
    displacement_effective_variables: Sequence[str],
) -> None:
    """校验文件位移字段是否表示相对参考构型的绝对位移。"""
    # Stores the `raw_reference` for use by the remaining steps in this flow.
    raw_reference = np.asarray(reference_points)
    raw_temporal_points = np.asarray(temporal_point_coordinates)
    reference = np.asarray(raw_reference, dtype=np.float64)
    temporal_points = np.asarray(raw_temporal_points, dtype=np.float64)
    # Validates `temporal_points.ndim != 3 or reference.ndim != 2` and raises `ValueError` when the input cannot be processed safely.
    if temporal_points.ndim != 3 or reference.ndim != 2:
        raise ValueError("无法校验运动网格位移字段：点坐标维度不符合 [T, N, C] / [N, C] 契约")
    if temporal_points.shape[1:] != reference.shape:
        raise ValueError(
            # Completes the multi-line `ValueError` arguments or elements.
            "无法校验运动网格位移字段：TemporalPointCoordinates 与 PointCoordinates 形状不一致"
        )

    expected = temporal_points - reference[np.newaxis, :, :]
    component_names = list(displacement_effective_variables or [])
    # Validates `not component_names` and raises `ValueError` when the input cannot be processed safely.
    if not component_names:
        raise ValueError("运动网格缺少位移分量字段，无法建立参考构型坐标契约")
    if len(component_names) > expected.shape[-1]:
        raise ValueError(
            # Completes the multi-line `ValueError` arguments or elements.
            "运动网格位移分量数量超过点坐标维度："
            f"components={len(component_names)}, coord_dim={expected.shape[-1]}"
        )

    for component_index, field_name in enumerate(component_names):
        # Validates `field_name not in fields` and raises `ValueError` when the input cannot be processed safely.
        if field_name not in fields:
            raise ValueError(f"运动网格缺少位移字段: {field_name}")
        raw_actual = np.asarray(fields[field_name])
        actual = np.asarray(raw_actual, dtype=np.float64)
        # Selects the required item from `expected` as the `expected_component`.
        expected_component = expected[:, :, component_index]
        if actual.shape != expected_component.shape:
            raise ValueError(
                "运动网格位移字段形状与时序点坐标不一致："
                # Completes the multi-line `ValueError` arguments or elements.
                f"field={field_name}, actual={actual.shape}, expected={expected_component.shape}"
            )
        atol = _resolve_displacement_validation_atol(
            raw_reference,
            # Completes the multi-line `_resolve_displacement_validation_atol` arguments or elements.
            raw_temporal_points,
            raw_actual,
        )
        if not np.allclose(actual, expected_component, rtol=1e-5, atol=atol, equal_nan=False):
            # Converts the `max_error` to `float` so later checks use a consistent type.
            max_error = float(np.max(np.abs(actual - expected_component)))
            raise ValueError(
                "运动网格位移字段必须表示相对参考构型的绝对位移，"
                f"field={field_name}, max_abs_error={max_error:.6g}"
            # Completes the multi-line `ValueError` arguments or elements.
            )


def derive_point_displacement_fields(
    reference_points: np.ndarray,
    temporal_point_coordinates: np.ndarray,
    # Completes the multi-line `collection` arguments or elements.
    field_component_map: Dict[str, List[str]],
    field_attributes: Dict[str, Dict[str, object]],
    fields: Dict[str, np.ndarray],
    *,
    # Continues the surrounding statement or collection definition.
    preferred_logical_name: str = "Displacement",
) -> Tuple[str, List[str], Dict[str, np.ndarray]]:
    reference = np.asarray(reference_points, dtype=np.float64)
    temporal_points = np.asarray(temporal_point_coordinates, dtype=np.float64)
    # Validates `temporal_points.ndim != 3` and raises `ValueError` when the input cannot be processed safely.
    if temporal_points.ndim != 3:
        raise ValueError(
            f"TemporalPointCoordinates 必须为三维数组 [T, N, C]，当前为 {temporal_points.shape}"
        )
    # Validates `reference.ndim != 2` and raises `ValueError` when the input cannot be processed safely.
    if reference.ndim != 2:
        raise ValueError(f"PointCoordinates 必须为二维数组 [N, C]，当前为 {reference.shape}")
    if temporal_points.shape[1:] != reference.shape:
        raise ValueError(
            # Completes the multi-line `ValueError` arguments or elements.
            "TemporalPointCoordinates 与 PointCoordinates 形状不一致，"
            f"reference={reference.shape}, temporal={temporal_points.shape}"
        )

    coord_dim = int(reference.shape[1])
    # Derives the `displacement` from the current values for downstream processing.
    displacement = temporal_points - reference[np.newaxis, :, :]
    logical_name = preferred_logical_name
    component_names = _build_displacement_component_names(logical_name, coord_dim)

    for comp_idx, component_name in enumerate(component_names):
        # Selects the required item from `displacement` as the `fields[component_name]`.
        fields[component_name] = displacement[:, :, comp_idx]
        field_attributes[component_name] = {
            "num_components": 1,
            "attribute": "scalar",
            # Completes the multi-line `collection` arguments or elements.
            "component_shape": (),
            "source_field": logical_name,
            "component_index": comp_idx,
        }

    # Stores the `field_component_map[logical_name]` for use by the remaining steps in this flow.
    field_component_map[logical_name] = list(component_names)
    return logical_name, component_names, fields


def get_reader_supported_extensions(reader) -> List[str]:
    """Infer supported file extensions from a reader instance."""
    if hasattr(reader, "supported_extensions"):
        # Stores the `exts` for use by the remaining steps in this flow.
        exts = getattr(reader, "supported_extensions")
        if isinstance(exts, (list, tuple, set)):
            normalized = [str(ext).lower() for ext in exts if str(ext).startswith(".")]
            if normalized:
                # Returns `normalized` to the caller for the next stage.
                return normalized

    name = type(reader).__name__.lower()
    if "vtk" in name:
        return [".vtu", ".vtk"]
    # Returns early when `"cgns" in name` makes the remaining work inapplicable.
    if "cgns" in name:
        return [".cgns"]
    if "csv" in name:
        return [".csv"]
    # Returns early when `"txt" in name` makes the remaining work inapplicable.
    if "txt" in name:
        return [".txt"]
    return []


def extract_time_index_from_name(file_name: str) -> Optional[int]:
    """Extract the sortable numeric time index from a file name."""
    # Stores the `stem` for use by the remaining steps in this flow.
    stem = Path(file_name).stem
    matches = re.findall(r"(\d+)", stem)
    if not matches:
        return None
    # Keeps the operation and its recovery path within the same exception boundary.
    try:
        return int(matches[-1])
    except ValueError:
        return None


# 判断输入路径是否是支持的时序清单文件。
def is_temporal_manifest_path(source_path: str) -> bool:
    """判断输入路径是否是支持的时序清单文件。"""
    lower_path = str(source_path).lower()
    return lower_path.endswith(".pvd") or lower_path.endswith(".series")


def _resolve_temporal_manifest_step_files(source_path: str) -> List[str]:
    """解析时序清单文件，返回按时间排序的 step 文件列表。"""
    # Stores the `manifest_path` for use by the remaining steps in this flow.
    manifest_path = Path(source_path)
    if not manifest_path.exists() or not manifest_path.is_file():
        raise ValueError(f"Temporal manifest file not found: {source_path}")

    lower_path = str(manifest_path).lower()
    # Returns early when `lower_path.endswith(".pvd")` makes the remaining work inapplicable.
    if lower_path.endswith(".pvd"):
        return _resolve_pvd_step_files(manifest_path)
    if lower_path.endswith(".series"):
        return _resolve_vtk_series_step_files(manifest_path)
    # Raises `ValueError` to report why this path cannot continue.
    raise ValueError(f"Unsupported temporal manifest format: {source_path}")


def _resolve_pvd_step_files(manifest_path: Path) -> List[str]:
    """解析 PVD 清单文件，按 timestep 顺序返回 step 文件。"""
    try:
        root = ET.parse(manifest_path).getroot()
    # Continues the surrounding statement or collection definition.
    except ET.ParseError as exc:
        raise ValueError(f"Failed to parse PVD temporal manifest: {manifest_path}") from exc

    entries = []
    for order, dataset in enumerate(root.iter()):
        # Checks `not str(dataset.tag).lower().endswith("dataset")` before entering this conditional branch.
        if not str(dataset.tag).lower().endswith("dataset"):
            continue

        file_name = dataset.attrib.get("file")
        if not file_name:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(f"PVD temporal manifest contains Dataset without file: {manifest_path}")

        timestep_text = dataset.attrib.get("timestep")
        if timestep_text is None:
            timestep = float(order)
        # Continues the surrounding statement or collection definition.
        else:
            try:
                timestep = float(timestep_text)
            except ValueError as exc:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(
                    f"PVD temporal manifest contains invalid timestep '{timestep_text}': {manifest_path}"
                ) from exc

        step_path = (manifest_path.parent / file_name).resolve()
        # Validates `not step_path.is_file()` and raises `ValueError` when the input cannot be processed safely.
        if not step_path.is_file():
            raise ValueError(
                f"PVD temporal manifest references missing step file: {step_path}"
            )
        # Adds the current value to `entries` for later aggregation.
        entries.append((timestep, order, str(step_path)))

    if not entries:
        raise ValueError(f"No Dataset entries found in PVD temporal manifest: {manifest_path}")

    entries.sort(key=lambda item: (item[0], item[1], item[2].lower()))
    # Returns `[item[2] for item in entries]` to the caller for the next stage.
    return [item[2] for item in entries]


def _resolve_vtk_series_step_files(manifest_path: Path) -> List[str]:
    """解析 .vtk.series 清单文件，按 time 顺序返回 step 文件。"""
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Continues the surrounding statement or collection definition.
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse .series manifest: {manifest_path}") from exc

    file_entries = payload.get("files")
    if not isinstance(file_entries, list) or not file_entries:
        # Raises `ValueError` to report why this path cannot continue.
        raise ValueError(f".series manifest does not contain a valid files list: {manifest_path}")

    entries = []
    for order, item in enumerate(file_entries):
        if not isinstance(item, dict):
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(f".series manifest contains invalid file entry: {manifest_path}")

        file_name = item.get("name")
        if not file_name:
            raise ValueError(f".series manifest contains file entry without name: {manifest_path}")

        # Reads `time` from the mapping into the `time_value` with its declared fallback.
        time_value = item.get("time", order)
        try:
            sortable_time = float(time_value)
        except (TypeError, ValueError) as exc:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(
                f".series manifest contains invalid time '{time_value}': {manifest_path}"
            ) from exc

        step_path = (manifest_path.parent / str(file_name)).resolve()
        # Validates `not step_path.is_file()` and raises `ValueError` when the input cannot be processed safely.
        if not step_path.is_file():
            raise ValueError(
                f".series manifest references missing step file: {step_path}"
            )
        # Adds the current value to `entries` for later aggregation.
        entries.append((sortable_time, order, str(step_path)))

    entries.sort(key=lambda item: (item[0], item[1], item[2].lower()))
    return [item[2] for item in entries]


def _collect_temporal_step_files_from_child_directories(
    # Continues the surrounding statement or collection definition.
    source: Path,
    exts: List[str],
) -> List[Path]:
    """Resolve `t0000/result.ext` style temporal step directories."""
    step_candidates: List[Tuple[int, str, Path]] = []
    # Initializes the `invalid_directories` as a collection for values from this processing stage.
    invalid_directories: List[str] = []

    for item in source.iterdir():
        if not item.is_dir():
            continue

        # Initializes the `supported_files` as a collection for values from this processing stage.
        supported_files = []
        for child in item.iterdir():
            if not child.is_file():
                continue
            # Checks `exts and child.suffix.lower() not in exts` before entering this conditional branch.
            if exts and child.suffix.lower() not in exts:
                continue
            supported_files.append(child)

        if not supported_files:
            # Handles the data or control-flow case represented by this statement.
            continue

        if len(supported_files) > 1:
            invalid_directories.append(item.name)
            continue

        # Stores the `step_index` for use by the remaining steps in this flow.
        step_index = extract_time_index_from_name(item.name)
        if step_index is None:
            step_index = extract_time_index_from_name(supported_files[0].name)
        if step_index is None:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(
                "Temporal step directories must contain sortable numeric indices in the "
                f"directory or file name: {item}"
            )

        # Adds the current value to `step_candidates` for later aggregation.
        step_candidates.append(
            (step_index, item.name.lower(), supported_files[0].resolve())
        )

    if invalid_directories:
        # Raises `ValueError` to report why this path cannot continue.
        raise ValueError(
            "Temporal step directories currently require exactly one supported result file "
            f"per step directory, but found multiple files in: {invalid_directories}"
        )

    # Calls `step_candidates.sort` to perform the operation required at this stage.
    step_candidates.sort(key=lambda entry: (entry[0], entry[1]))
    return [entry[2] for entry in step_candidates]


def list_temporal_sequence_files(
    reader,
    # Continues the surrounding statement or collection definition.
    source_path: str,
    max_time_steps: Optional[int] = None,
) -> List[str]:
    """列出时序 step 文件，支持目录序列和清单文件两种组织方式。"""
    source = Path(source_path)
    # Validates `not source.exists()` and raises `ValueError` when the input cannot be processed safely.
    if not source.exists():
        raise ValueError(f"Temporal source not found: {source_path}")

    if source.is_file():
        ordered_files = _resolve_temporal_manifest_step_files(source_path)
        # Checks `max_time_steps is not None` before entering this conditional branch.
        if max_time_steps is not None:
            ordered_files = ordered_files[: max(0, int(max_time_steps))]
        return ordered_files

    if not source.is_dir():
        # Raises `ValueError` to report why this path cannot continue.
        raise ValueError(f"Temporal source is neither a directory nor a manifest file: {source_path}")

    exts = get_reader_supported_extensions(reader)
    candidates = []
    for item in source.iterdir():
        # Checks `not item.is_file()` before entering this conditional branch.
        if not item.is_file():
            continue
        if exts and item.suffix.lower() not in exts:
            continue
        # Adds the current value to `candidates` for later aggregation.
        candidates.append(item)

    if not candidates:
        child_directory_candidates = _collect_temporal_step_files_from_child_directories(
            source,
            # Completes the multi-line `_collect_temporal_step_files_from_child_directories` arguments or elements.
            exts,
        )
        if not child_directory_candidates:
            supported = ", ".join(exts) if exts else "reader-specific files"
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(
                f"No temporal sequence files found in {source_path} (supported: {supported})"
            )
        ordered_files = [str(path) for path in child_directory_candidates]
        # Checks `max_time_steps is not None` before entering this conditional branch.
        if max_time_steps is not None:
            ordered_files = ordered_files[: max(0, int(max_time_steps))]
        return ordered_files

    parsed_steps = [extract_time_index_from_name(path.name) for path in candidates]
    # Builds the `missing_numeric` using the comprehension's filter and mapping rules.
    missing_numeric = [path.name for path, step in zip(candidates, parsed_steps) if step is None]
    if missing_numeric:
        raise ValueError(
            "Temporal VTK directory contains files without sortable numeric indices: "
            # Completes the multi-line `ValueError` arguments or elements.
            f"{missing_numeric}"
        )

    sorted_candidates = sorted(
        zip(candidates, parsed_steps),
        # Completes the multi-line `collection` arguments or elements.
        key=lambda item: (item[1], item[0].name.lower()),
    )
    ordered_files = [str(item[0]) for item in sorted_candidates]
    if max_time_steps is not None:
        # Selects the required item from `ordered_files` as the `ordered_files`.
        ordered_files = ordered_files[: max(0, int(max_time_steps))]
    return ordered_files


def truncate_temporal_base_data(
    data: BaseData,
    # Continues the surrounding statement or collection definition.
    max_time_steps: Optional[int],
) -> BaseData:
    """按前 K 步 history 截断 temporal 场样本。"""
    if max_time_steps is None or data.get_mode() != "temporal":
        return data

    # Converts the `limit` to `int` so later checks use a consistent type.
    limit = int(max_time_steps)
    if limit <= 0:
        return data

    raw_data = data.get_data()
    # Returns early when `not isinstance(raw_data, dict)` makes the remaining work inapplicable.
    if not isinstance(raw_data, dict):
        return data

    truncated_data = dict(raw_data)
    fields = raw_data.get("Fields")
    # Checks `isinstance(fields, dict) and fields` before entering this conditional branch.
    if isinstance(fields, dict) and fields:
        truncated_fields = {}
        for field_name, values in fields.items():
            if hasattr(values, "ndim") and int(values.ndim) >= 2:
                # Selects the required item from `values` as the `truncated_fields[field_name]`.
                truncated_fields[field_name] = values[:limit]
            else:
                truncated_fields[field_name] = values
        truncated_data["Fields"] = truncated_fields

    # Reads `TemporalPointCoordinates` from the mapping into the `temporal_point_coordinates`.
    temporal_point_coordinates = raw_data.get("TemporalPointCoordinates")
    if hasattr(temporal_point_coordinates, "ndim") and int(temporal_point_coordinates.ndim) >= 3:
        truncated_data["TemporalPointCoordinates"] = temporal_point_coordinates[:limit]

    temporal_cell_coordinates = raw_data.get("TemporalCellCoordinates")
    # Checks `` before entering this conditional branch.
    if hasattr(temporal_cell_coordinates, "ndim") and int(temporal_cell_coordinates.ndim) >= 3:
        truncated_data["TemporalCellCoordinates"] = temporal_cell_coordinates[:limit]

    geometry_motion = raw_data.get("GeometryMotion")
    if isinstance(geometry_motion, dict):
        # Stores the `truncated_data['GeometryMotion']` for use by the remaining steps in this flow.
        truncated_data["GeometryMotion"] = dict(geometry_motion)

    sample_info = dict(data.get_sample_map() or {})
    if "temporal_steps" in sample_info:
        try:
            # Stores the `sample_info['temporal_steps']` for use by the remaining steps in this flow.
            sample_info["temporal_steps"] = min(int(sample_info["temporal_steps"]), limit)
        except (TypeError, ValueError):
            sample_info["temporal_steps"] = limit

    return BaseData(
        # Completes the multi-line `BaseData` arguments or elements.
        data=truncated_data,
        task_type=data.get_task_type(),
        mode=data.get_mode(),
        point_or_cell=data.get_point_or_cell(),
        # Completes the multi-line `BaseData` arguments or elements.
        sample_map=sample_info,
    )


def read_single_sample(
    reader,
    # Continues the surrounding statement or collection definition.
    file_path: str,
    sample_id: Optional[int],
    allow_missing_fields: bool = False,
    max_time_steps: Optional[int] = None,
    # Continues the surrounding statement or collection definition.
    variables: Optional[List[str]] = None,
) -> BaseData:
    """Read a single file while remaining compatible with optional sample_order."""
    signature = inspect.signature(reader.read)
    kwargs = {}
    # Checks `"sample_order" in signature.parameters` before entering this conditional branch.
    if "sample_order" in signature.parameters:
        kwargs["sample_order"] = sample_id
    if allow_missing_fields and "allow_missing_fields" in signature.parameters:
        kwargs["allow_missing_fields"] = True
    # Checks `` before entering this conditional branch.
    if max_time_steps is not None and "max_time_steps" in signature.parameters:
        kwargs["max_time_steps"] = int(max_time_steps)
    if variables is not None and "variables" in signature.parameters:
        kwargs["variables"] = list(variables)

    # Stores the current data for use by the remaining steps in this flow.
    data = reader.read(file_path, **kwargs)
    return truncate_temporal_base_data(data, max_time_steps=max_time_steps)


def connectivity_to_tuples(connectivity) -> List[Tuple[int, ...]]:
    """Normalize element connectivity for temporal consistency checks."""
    tuples = []
    # Visits each item in `connectivity` and applies the current processing step.
    for cell in connectivity:
        cell_arr = np.asarray(cell).reshape(-1)
        tuples.append(tuple(int(v) for v in cell_arr.tolist()))
    return tuples


# 流式聚合时序场样本，静态字段仅保留首个时间步。
def aggregate_field_temporal_sequence(
    data_steps: Iterable[BaseData],
    source_dir: str,
    static_field_names: Optional[Sequence[str]] = None,
    # Continues the surrounding statement or collection definition.
    expected_time_steps: Optional[int] = None,
) -> BaseData:
    """流式聚合时序场样本，静态字段仅保留首个时间步。"""
    if expected_time_steps is None and hasattr(data_steps, "__len__"):
        expected_time_steps = len(data_steps)
    # Checks `expected_time_steps is not None` before entering this conditional branch.
    if expected_time_steps is not None:
        expected_time_steps = max(0, int(expected_time_steps))
    steps_iterator = iter(data_steps)
    try:
        # Stores the `first` for use by the remaining steps in this flow.
        first = next(steps_iterator)
    except StopIteration as exc:
        raise ValueError(f"No temporal samples were loaded from {source_dir}") from exc

    first_dict = first.get_data()
    # Stores the `point_or_cell` for use by the remaining steps in this flow.
    point_or_cell = first.point_or_cell
    static_names: Set[str] = {str(name) for name in (static_field_names or [])}

    template_points = np.asarray(first_dict["PointCoordinates"])
    template_cells = np.asarray(first_dict["CellCoordinates"])
    # Stores the `template_conn` for use by the remaining steps in this flow.
    template_conn = connectivity_to_tuples(first_dict["ElementConnectivity"])
    template_fields = first_dict.get("Fields", {})
    template_meta = first_dict.get("FieldAttributes", {})
    template_map = first_dict.get("FieldComponentMap", {})
    # Stores the `temporal_point_coordinates` for use by the remaining steps in this flow.
    temporal_point_coordinates: Optional[List[np.ndarray]] = None
    temporal_cell_coordinates: Optional[List[np.ndarray]] = None
    has_temporal_point_motion = False
    has_temporal_cell_motion = False
    # Stores the `loaded_steps` for use by the remaining steps in this flow.
    loaded_steps = 0

    def append_coordinates(
        current_points: np.ndarray,
        current_cells: np.ndarray,
        # Continues the surrounding statement or collection definition.
        step_idx: int,
        point_coordinates_match_reference: Optional[bool] = None,
        cell_coordinates_match_reference: Optional[bool] = None,
    ) -> None:
        # Handles the data or control-flow case represented by this statement.
        nonlocal temporal_point_coordinates
        nonlocal temporal_cell_coordinates
        nonlocal has_temporal_point_motion
        nonlocal has_temporal_cell_motion

        # Stores the `point_changed` for use by the remaining steps in this flow.
        point_changed = (
            not bool(point_coordinates_match_reference)
            if point_coordinates_match_reference is not None
            else not np.allclose(current_points, template_points)
        # Continues the surrounding statement or collection definition.
        )
        cell_changed = (
            not bool(cell_coordinates_match_reference)
            if cell_coordinates_match_reference is not None
            # Completes the multi-line `np.allclose` arguments or elements.
            else not np.allclose(current_cells, template_cells)
        )
        if point_changed:
            has_temporal_point_motion = True
            # Checks `temporal_point_coordinates is None` before entering this conditional branch.
            if temporal_point_coordinates is None:
                temporal_point_coordinates = [template_points] * step_idx
        if cell_changed:
            has_temporal_cell_motion = True
            # Checks `temporal_cell_coordinates is None` before entering this conditional branch.
            if temporal_cell_coordinates is None:
                temporal_cell_coordinates = [template_cells] * step_idx

        if temporal_point_coordinates is not None:
            temporal_point_coordinates.append(
                # Completes the multi-line `temporal_point_coordinates.append` arguments or elements.
                template_points if not point_changed else current_points
            )
        if temporal_cell_coordinates is not None:
            temporal_cell_coordinates.append(
                # Completes the multi-line `temporal_cell_coordinates.append` arguments or elements.
                template_cells if not cell_changed else current_cells
            )

    def append_step_coordinates(data_dict: Dict[str, object], step_idx: int) -> None:
        coordinates_prevalidated = bool(data_dict.get("TemporalCoordinatesPrevalidated", False))
        # Calls `append_coordinates` to perform the operation required at this stage.
        append_coordinates(
            np.asarray(data_dict["PointCoordinates"]),
            np.asarray(data_dict["CellCoordinates"]),
            step_idx,
            # Completes the multi-line `append_coordinates` arguments or elements.
            point_coordinates_match_reference=(
                bool(data_dict.get("TemporalPointCoordinatesMatchReference", False))
                if coordinates_prevalidated
                else None
            # Completes the multi-line `append_coordinates` arguments or elements.
            ),
            cell_coordinates_match_reference=(
                bool(data_dict.get("TemporalCellCoordinatesMatchReference", False))
                if coordinates_prevalidated
                # Completes the multi-line `append_coordinates` arguments or elements.
                else None
            ),
        )

    if not template_fields:
        # Visits each item in `enumerate([first], start=0)` and applies the current processing step.
        for step_idx, step_data in enumerate([first], start=0):
            data_dict = step_data.get_data()
            loaded_steps += 1
            append_step_coordinates(data_dict, step_idx)
        # Visits each item in `enumerate(steps_iterator, start=1)` and applies the current processing step.
        for step_idx, step_data in enumerate(steps_iterator, start=1):
            data_dict = step_data.get_data()
            loaded_steps += 1

            if step_data.point_or_cell != point_or_cell:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(
                    f"point_or_cell mismatch at temporal step {step_idx}: "
                    f"expected {point_or_cell}, got {step_data.point_or_cell}"
                )

            # Validates `` and raises `ValueError` when the input cannot be processed safely.
            if (
                not bool(data_dict.get("TemporalTopologyPrevalidated", False))
                and connectivity_to_tuples(data_dict["ElementConnectivity"]) != template_conn
            ):
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(f"ElementConnectivity changed across temporal steps in {source_dir}")

            append_step_coordinates(data_dict, step_idx)

        aggregated_data = dict(first_dict)
        if has_temporal_point_motion and temporal_point_coordinates is not None:
            # Stores the `aggregated_data['TemporalPointCoordinates']` for use by the remaining steps in this flow.
            aggregated_data["TemporalPointCoordinates"] = np.stack(temporal_point_coordinates, axis=0)
        if has_temporal_cell_motion and temporal_cell_coordinates is not None:
            aggregated_data["TemporalCellCoordinates"] = np.stack(temporal_cell_coordinates, axis=0)
        _ensure_motion_meta(
            # Completes the multi-line `_ensure_motion_meta` arguments or elements.
            aggregated_data,
            has_temporal_point_motion=has_temporal_point_motion,
            has_temporal_cell_motion=has_temporal_cell_motion,
        )

        # Stores the `sample_info` for use by the remaining steps in this flow.
        sample_info = dict(first.get_sample_map() or {})
        sample_info["file_path"] = source_dir
        sample_info = _append_sample_motion_meta(
            sample_info,
            # Completes the multi-line `_append_sample_motion_meta` arguments or elements.
            temporal_steps=loaded_steps,
            geometry_regime=aggregated_data.get("GeometryRegime"),
            displacement_logical_name=None,
            displacement_effective_variables=[],
            # Completes the multi-line `_append_sample_motion_meta` arguments or elements.
            displacement_source=None,
        )
        return BaseData(
            data=aggregated_data,
            # Completes the multi-line `BaseData` arguments or elements.
            task_type=first.task_type,
            mode="temporal",
            point_or_cell=point_or_cell,
            sample_map=sample_info,
        # Completes the multi-line `BaseData` arguments or elements.
        )

    field_chunks: Dict[str, List[np.ndarray]] = {
        name: [] for name in template_fields.keys() if name not in static_names
    }
    # Initializes the `field_buffers` as a collection for values from this processing stage.
    field_buffers: Dict[str, np.ndarray] = {}
    field_buffer_sizes: Dict[str, int] = {}
    static_fields: Dict[str, np.ndarray] = {
        name: np.asarray(values)
        # Completes the multi-line `collection` arguments or elements.
        for name, values in template_fields.items()
        if name in static_names
    }

    def append_dynamic_field(field_name: str, values: np.ndarray) -> None:
        # Stores the `normalized` for use by the remaining steps in this flow.
        normalized = values[np.newaxis, ...] if values.ndim == 1 else values
        if expected_time_steps is None:
            field_chunks[field_name].append(normalized)
            return

        # Stores the `current_size` for use by the remaining steps in this flow.
        current_size = field_buffer_sizes.get(field_name, 0)
        buffer = field_buffers.get(field_name)
        if buffer is None:
            capacity = max(
                # Completes the multi-line `int` arguments or elements.
                int(expected_time_steps) * int(normalized.shape[0]),
                int(normalized.shape[0]),
            )
            buffer = np.empty((capacity,) + normalized.shape[1:], dtype=normalized.dtype)
            # Stores the `field_buffers[field_name]` for use by the remaining steps in this flow.
            field_buffers[field_name] = buffer
        elif buffer.shape[1:] != normalized.shape[1:]:
            raise ValueError(
                f"时序字段 {field_name} 的非时间维度不一致: "
                # Completes the multi-line `ValueError` arguments or elements.
                f"expected={buffer.shape[1:]}, actual={normalized.shape[1:]}"
            )
        elif not np.can_cast(normalized.dtype, buffer.dtype, casting="safe"):
            promoted = np.empty(buffer.shape, dtype=np.result_type(buffer.dtype, normalized.dtype))
            # Selects the required item from `buffer` as the `promoted[:current_size]`.
            promoted[:current_size] = buffer[:current_size]
            buffer = promoted
            field_buffers[field_name] = buffer

        required_size = current_size + int(normalized.shape[0])
        # Checks `required_size > buffer.shape[0]` before entering this conditional branch.
        if required_size > buffer.shape[0]:
            expanded = np.empty(
                (max(required_size, buffer.shape[0] * 2),) + buffer.shape[1:],
                dtype=buffer.dtype,
            # Completes the multi-line `np.empty` arguments or elements.
            )
            expanded[:current_size] = buffer[:current_size]
            buffer = expanded
            field_buffers[field_name] = buffer
        # Stores the `buffer[current_size:required_size]` for use by the remaining steps in this flow.
        buffer[current_size:required_size] = normalized
        field_buffer_sizes[field_name] = required_size

    for step_idx, step_data in enumerate([first], start=0):
        data_dict = step_data.get_data()
        # Stores the `loaded_steps` for use by the remaining steps in this flow.
        loaded_steps += 1
        fields = data_dict.get("Fields", {})
        for field_name in field_chunks:
            append_dynamic_field(field_name, np.asarray(fields[field_name]))

        # Calls `append_step_coordinates` to perform the operation required at this stage.
        append_step_coordinates(data_dict, step_idx)

    for step_idx, step_data in enumerate(steps_iterator, start=1):
        data_dict = step_data.get_data()
        loaded_steps += 1

        # Validates `step_data.point_or_cell != point_or_cell` and raises `ValueError` when the input cannot be processed safely.
        if step_data.point_or_cell != point_or_cell:
            raise ValueError(
                f"point_or_cell mismatch at temporal step {step_idx}: "
                f"expected {point_or_cell}, got {step_data.point_or_cell}"
            # Completes the multi-line `ValueError` arguments or elements.
            )

        append_step_coordinates(data_dict, step_idx)

        if (
            not bool(data_dict.get("TemporalTopologyPrevalidated", False))
            # Completes the multi-line `connectivity_to_tuples` arguments or elements.
            and connectivity_to_tuples(data_dict["ElementConnectivity"]) != template_conn
        ):
            raise ValueError(f"ElementConnectivity changed across temporal steps in {source_dir}")

        fields = data_dict.get("Fields", {})
        # Validates `set(fields.keys()) != set(template_fields.keys())` and raises `ValueError` when the input cannot be processed safely.
        if set(fields.keys()) != set(template_fields.keys()):
            raise ValueError(f"Field set changed across temporal steps in {source_dir}")

        if data_dict.get("FieldAttributes", {}) != template_meta:
            raise ValueError(f"FieldAttributes changed across temporal steps in {source_dir}")

        # Validates `data_dict.get("FieldComponentMap", {}) != template_map` and raises `ValueError` when the input cannot be processed safely.
        if data_dict.get("FieldComponentMap", {}) != template_map:
            raise ValueError(f"FieldComponentMap changed across temporal steps in {source_dir}")

        for field_name in template_fields.keys():
            if field_name in static_names:
                # Validates `` and raises `ValueError` when the input cannot be processed safely.
                if not np.array_equal(np.asarray(fields[field_name]), static_fields[field_name]):
                    raise ValueError(
                        f"静态字段 {field_name!r} 在时间步之间发生变化: {source_dir}"
                    )
                # Handles the data or control-flow case represented by this statement.
                continue
            append_dynamic_field(field_name, np.asarray(fields[field_name]))

    aggregated_fields = dict(static_fields)
    for field_name, chunks in field_chunks.items():
        # Checks `expected_time_steps is not None` before entering this conditional branch.
        if expected_time_steps is not None:
            aggregated_fields[field_name] = field_buffers[field_name][
                : field_buffer_sizes[field_name]
            ]
            # Handles the data or control-flow case represented by this statement.
            continue
        try:
            aggregated_fields[field_name] = np.concatenate(chunks, axis=0)
        except ValueError as exc:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(
                f"Failed to concatenate temporal field {field_name} in {source_dir}"
            ) from exc

    aggregated_data = dict(first_dict)
    # Stores the `aggregated_data['Fields']` for use by the remaining steps in this flow.
    aggregated_data["Fields"] = aggregated_fields
    if has_temporal_point_motion and temporal_point_coordinates is not None:
        aggregated_data["TemporalPointCoordinates"] = np.stack(temporal_point_coordinates, axis=0)
    if has_temporal_cell_motion and temporal_cell_coordinates is not None:
        # Stores the `aggregated_data['TemporalCellCoordinates']` for use by the remaining steps in this flow.
        aggregated_data["TemporalCellCoordinates"] = np.stack(temporal_cell_coordinates, axis=0)

    displacement_logical_name = infer_displacement_logical_name(
        template_map,
        coord_dim=int(template_points.shape[1]) if template_points.ndim == 2 else 0,
    # Completes the multi-line `infer_displacement_logical_name` arguments or elements.
    )
    displacement_effective_variables = infer_displacement_effective_variables(
        template_map,
        displacement_logical_name,
        # Completes the multi-line `int` arguments or elements.
        coord_dim=int(template_points.shape[1]) if template_points.ndim == 2 else 0,
    )
    displacement_source = "file" if displacement_logical_name else None
    catalog_backed_fields = bool(first_dict.get("VariableCatalog"))

    # Checks `has_temporal_point_motion and displacement_logical_name is None` before entering this conditional branch.
    if has_temporal_point_motion and displacement_logical_name is None:
        displacement_effective_variables = infer_scalar_displacement_effective_variables(
            aggregated_fields,
            coord_dim=int(template_points.shape[1]) if template_points.ndim == 2 else 0,
        # Completes the multi-line `infer_scalar_displacement_effective_variables` arguments or elements.
        )
        if displacement_effective_variables:
            validate_point_displacement_fields(
                reference_points=template_points,
                # Completes the multi-line `validate_point_displacement_fields` arguments or elements.
                temporal_point_coordinates=aggregated_data["TemporalPointCoordinates"],
                fields=aggregated_fields,
                displacement_effective_variables=displacement_effective_variables,
            )
            # Stores the `displacement_source` for use by the remaining steps in this flow.
            displacement_source = "file"
        elif not catalog_backed_fields:
            displacement_logical_name, displacement_effective_variables, aggregated_fields = derive_point_displacement_fields(
                reference_points=template_points,
                # Completes the multi-line `derive_point_displacement_fields` arguments or elements.
                temporal_point_coordinates=aggregated_data["TemporalPointCoordinates"],
                field_component_map=template_map,
                field_attributes=template_meta,
                fields=aggregated_fields,
            # Completes the multi-line `derive_point_displacement_fields` arguments or elements.
            )
            aggregated_data["Fields"] = aggregated_fields
            aggregated_data["FieldAttributes"] = template_meta
            aggregated_data["FieldComponentMap"] = template_map
            # Stores the `displacement_source` for use by the remaining steps in this flow.
            displacement_source = "derived"
    elif has_temporal_point_motion:
        validate_point_displacement_fields(
            reference_points=template_points,
            # Completes the multi-line `validate_point_displacement_fields` arguments or elements.
            temporal_point_coordinates=aggregated_data["TemporalPointCoordinates"],
            fields=aggregated_fields,
            displacement_effective_variables=displacement_effective_variables,
        )

    # Calls `_ensure_motion_meta` to perform the operation required at this stage.
    _ensure_motion_meta(
        aggregated_data,
        has_temporal_point_motion=has_temporal_point_motion,
        has_temporal_cell_motion=has_temporal_cell_motion,
        # Completes the multi-line `_ensure_motion_meta` arguments or elements.
        displacement_logical_name=displacement_logical_name,
        displacement_effective_variables=displacement_effective_variables,
        displacement_source=displacement_source,
    )

    # Stores the `sample_info` for use by the remaining steps in this flow.
    sample_info = dict(first.get_sample_map() or {})
    sample_info["file_path"] = source_dir
    first_dynamic_field = next(iter(field_chunks), None)
    if first_dynamic_field is None:
        # Stores the `temporal_steps` for use by the remaining steps in this flow.
        temporal_steps = loaded_steps
    elif expected_time_steps is not None:
        temporal_steps = field_buffer_sizes[first_dynamic_field]
    else:
        # Stores the `temporal_steps` for use by the remaining steps in this flow.
        temporal_steps = sum(chunk.shape[0] for chunk in field_chunks[first_dynamic_field])
    sample_info = _append_sample_motion_meta(
        sample_info,
        temporal_steps=temporal_steps,
        # Completes the multi-line `aggregated_data.get` arguments or elements.
        geometry_regime=aggregated_data.get("GeometryRegime"),
        displacement_logical_name=displacement_logical_name,
        displacement_effective_variables=displacement_effective_variables,
        displacement_source=displacement_source,
    # Completes the multi-line `_append_sample_motion_meta` arguments or elements.
    )

    return BaseData(
        data=aggregated_data,
        task_type=first.task_type,
        # Completes the multi-line `BaseData` arguments or elements.
        mode="temporal",
        point_or_cell=point_or_cell,
        sample_map=sample_info,
    )
