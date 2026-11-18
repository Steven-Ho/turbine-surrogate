"""场任务变量目录与运行时通道元数据工具。"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple
from urllib.parse import quote


def _normalize_name(value: object) -> str:
    return str(value).strip().casefold()


def _normalize_location(value: object) -> str:
    location = str(value or "").strip().lower()
    if location not in {"point", "cell"}:
        raise ValueError(f"字段位置非法: {value}")
    return location


def _build_channel_id(
    *,
    format_name: str,
    location: str,
    array_name: str,
    occurrence: int,
    component_index: int,
) -> str:
    escaped_name = quote(str(array_name), safe="._-")
    occurrence_suffix = f"@{occurrence}" if occurrence > 0 else ""
    return (
        f"{format_name}:{location}:{escaped_name}"
        f"{occurrence_suffix}#{component_index}"
    )


def _deduplicate_component_keys(
    entries: List[MutableMapping[str, object]],
) -> None:
    """为冲突的显示分量键生成可读且唯一的运行时键。"""
    provisional: List[Tuple[int, int, str]] = []
    for entry_index, entry in enumerate(entries):
        array_name = str(entry["array_name"])
        component_count = int(entry["num_components"])
        for component_index in range(component_count):
            display_name = (
                array_name
                if component_count == 1
                else f"{array_name}_{component_index}"
            )
            provisional.append((entry_index, component_index, display_name))

    counts = Counter(_normalize_name(display_name) for _, _, display_name in provisional)
    occupied: Set[str] = set()
    for entry_index, component_index, display_name in provisional:
        entry = entries[entry_index]
        runtime_key = display_name
        if counts[_normalize_name(display_name)] > 1:
            runtime_key = f"{entry['location']}:{display_name}"

        candidate = runtime_key
        duplicate_index = 1
        while _normalize_name(candidate) in occupied:
            candidate = f"{runtime_key}@{duplicate_index}"
            duplicate_index += 1
        occupied.add(_normalize_name(candidate))

        component_keys = entry.setdefault("component_keys", [])
        component_displays = entry.setdefault("component_display_names", [])
        if not isinstance(component_keys, list) or not isinstance(component_displays, list):
            raise ValueError("变量目录分量键初始化失败")
        component_keys.append(candidate)
        component_displays.append(display_name)


def _deduplicate_selection_keys(entries: List[MutableMapping[str, object]]) -> None:
    raw_counts = Counter(_normalize_name(entry["array_name"]) for entry in entries)
    occupied: Set[str] = set()
    for entry in entries:
        raw_name = str(entry["array_name"])
        selection_key = raw_name
        if raw_counts[_normalize_name(raw_name)] > 1:
            selection_key = f"{entry['location']}:{raw_name}"

        candidate = selection_key
        duplicate_index = 1
        while _normalize_name(candidate) in occupied:
            candidate = f"{selection_key}@{duplicate_index}"
            duplicate_index += 1
        occupied.add(_normalize_name(candidate))
        entry["selection_key"] = candidate


def build_field_variable_catalog(
    entries: Iterable[Mapping[str, object]],
) -> Dict[str, Dict[str, object]]:
    """根据原始字段或数组描述创建完整变量目录。

    每个输入条目至少包含 ``format``、``location``、``array_name`` 和
    ``num_components``。CGNS 标量字段与 VTK 原始数组均可使用该结构。
    """
    prepared: List[MutableMapping[str, object]] = []
    occurrence_counters: Dict[Tuple[str, str, str], int] = {}
    for raw_entry in entries:
        format_name = str(raw_entry.get("format") or "").strip().lower()
        if format_name not in {"cgns", "vtk"}:
            raise ValueError(f"不支持的字段来源格式: {format_name}")
        location = _normalize_location(raw_entry.get("location"))
        array_name = str(raw_entry.get("array_name") or "").strip()
        if not array_name:
            raise ValueError("字段或数组名称不能为空")
        try:
            component_count = int(raw_entry.get("num_components") or 1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"字段 {array_name} 的分量数非法") from exc
        if component_count <= 0:
            raise ValueError(f"字段 {array_name} 的分量数必须大于零")

        occurrence_key = (format_name, location, _normalize_name(array_name))
        occurrence = occurrence_counters.get(occurrence_key, 0)
        occurrence_counters[occurrence_key] = occurrence + 1
        prepared.append(
            {
                "format": format_name,
                "location": location,
                "array_name": array_name,
                "num_components": component_count,
                "occurrence": occurrence,
                "component_keys": [],
                "component_display_names": [],
            }
        )

    _deduplicate_selection_keys(prepared)
    _deduplicate_component_keys(prepared)

    catalog: Dict[str, Dict[str, object]] = {}
    for entry in prepared:
        selection_key = str(entry["selection_key"])
        component_keys = list(entry["component_keys"])
        component_displays = list(entry["component_display_names"])
        sources = []
        for component_index, component_key in enumerate(component_keys):
            sources.append(
                {
                    "channel_id": _build_channel_id(
                        format_name=str(entry["format"]),
                        location=str(entry["location"]),
                        array_name=str(entry["array_name"]),
                        occurrence=int(entry["occurrence"]),
                        component_index=component_index,
                    ),
                    "format": str(entry["format"]),
                    "location": str(entry["location"]),
                    "array_name": str(entry["array_name"]),
                    "occurrence": int(entry["occurrence"]),
                    "component_index": component_index,
                    "num_components": int(entry["num_components"]),
                    "runtime_key": str(component_key),
                    "display_name": str(component_displays[component_index]),
                    "selection_key": selection_key,
                }
            )

        catalog[selection_key] = {
            "selection_key": selection_key,
            "display_name": str(entry["array_name"]),
            "format": str(entry["format"]),
            "location": str(entry["location"]),
            "array_name": str(entry["array_name"]),
            "occurrence": int(entry["occurrence"]),
            "num_components": int(entry["num_components"]),
            "component_keys": component_keys,
            "component_display_names": component_displays,
            "sources": sources,
        }
    return catalog


def _resolve_catalog_key(
    catalog: Mapping[str, Mapping[str, object]],
    requested_key: object,
) -> Optional[str]:
    requested = str(requested_key)
    if requested in catalog:
        return requested

    normalized = _normalize_name(requested)
    matches = [key for key in catalog if _normalize_name(key) == normalized]
    if len(matches) > 1:
        raise ValueError(
            "变量选择在大小写不敏感匹配下存在歧义: "
            f"requested={requested}, matches={matches}"
        )
    return matches[0] if matches else None


def select_catalog_variables(
    catalog: Mapping[str, Mapping[str, object]],
    requested_variables: Sequence[str],
    *,
    strict_missing: bool = True,
) -> Tuple[Dict[str, List[str]], List[str], Dict[str, Dict[str, object]], Dict[str, str], List[str]]:
    """从完整目录生成本次任务的映射、通道与来源元数据。"""
    selected_map: Dict[str, List[str]] = {}
    effective_variables: List[str] = []
    effective_sources: Dict[str, Dict[str, object]] = {}
    location_map: Dict[str, str] = {}
    resolved_selection: List[str] = []
    missing: List[str] = []

    for requested in dict.fromkeys(str(name) for name in (requested_variables or [])):
        catalog_key = _resolve_catalog_key(catalog, requested)
        if catalog_key is None:
            missing.append(requested)
            continue

        entry = catalog[catalog_key]
        component_keys = [str(name) for name in (entry.get("component_keys") or [])]
        sources = list(entry.get("sources") or [])
        if not component_keys or len(component_keys) != len(sources):
            raise ValueError(f"变量目录条目不完整: {catalog_key}")

        selected_map[catalog_key] = component_keys
        resolved_selection.append(catalog_key)
        for component_key, source in zip(component_keys, sources):
            if component_key in effective_sources:
                existing = effective_sources[component_key]
                if existing.get("channel_id") != source.get("channel_id"):
                    raise ValueError(f"运行时分量键冲突: {component_key}")
                continue
            effective_variables.append(component_key)
            effective_sources[component_key] = dict(source)
            location_map[component_key] = _normalize_location(source.get("location"))

    if missing and (strict_missing or not effective_variables):
        raise ValueError(
            "请求变量未在数据集目录中找到: "
            f"missing={missing}, available={list(catalog.keys())}"
        )
    return (
        selected_map,
        effective_variables,
        effective_sources,
        location_map,
        resolved_selection,
    )


def catalog_component_map(
    catalog: Mapping[str, Mapping[str, object]],
) -> Dict[str, List[str]]:
    """返回完整目录的兼容映射，仅用于变量发现或临时扫描。"""
    return {
        str(key): [str(name) for name in (entry.get("component_keys") or [])]
        for key, entry in (catalog or {}).items()
    }


def catalog_sources_by_runtime_key(
    catalog: Mapping[str, Mapping[str, object]],
) -> Dict[str, Dict[str, object]]:
    """返回目录中全部运行时分量键的来源描述。"""
    sources: Dict[str, Dict[str, object]] = {}
    for entry in (catalog or {}).values():
        for source in entry.get("sources") or []:
            runtime_key = str(source.get("runtime_key") or "")
            if runtime_key:
                sources[runtime_key] = dict(source)
    return sources
