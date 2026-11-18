from typing import Dict, List, Sequence


_VECTOR_SUFFIXES = ["X", "Y", "Z"]
_TENSOR6_SUFFIXES = ["XX", "YY", "ZZ", "XY", "YZ", "ZX"]
_TENSOR9_SUFFIXES = ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"]


def _normalize_name(name: str) -> str:
    return name.casefold()


def _build_case_insensitive_name_index(names: Sequence[str]) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for name in names:
        key = _normalize_name(name)
        if key in index and index[key] != name:
            raise ValueError(
                "Field names conflict by case-insensitive comparison. "
                f"conflict={index[key]} vs {name}"
            )
        index[key] = name
    return index


def _all_exist(name_index: Dict[str, str], required: Sequence[str], base: str) -> bool:
    base_key = _normalize_name(base)
    return all(_normalize_name(f"{base}{s}") in name_index for s in required)


def infer_component_map_from_field_names(field_names: Sequence[str]) -> Dict[str, List[str]]:
    """Infer logical-variable to component mapping from scalar component field names.

    Example:
    - VelocityX/VelocityY/VelocityZ -> Velocity
    - StressXX/.../StressZX -> Stress
    - Pressure -> Pressure
    """
    names = list(field_names)
    name_index = _build_case_insensitive_name_index(names)
    component_map: Dict[str, List[str]] = {}
    consumed = set()
    consumed_bases = set()

    # Tensor(9)
    for name in names:
        name_upper = name.upper()
        for suffix in _TENSOR9_SUFFIXES:
            if not name_upper.endswith(suffix):
                continue
            base = name[: -len(suffix)]
            base_key = _normalize_name(base)
            if base and _all_exist(name_index, _TENSOR9_SUFFIXES, base):
                component_map[base] = [name_index[_normalize_name(f"{base}{s}")] for s in _TENSOR9_SUFFIXES]
                consumed.update(component_map[base])
                consumed_bases.add(base_key)

    # Tensor(6)
    for name in names:
        name_upper = name.upper()
        for suffix in _TENSOR6_SUFFIXES:
            if not name_upper.endswith(suffix):
                continue
            base = name[: -len(suffix)]
            base_key = _normalize_name(base)
            if base and base_key not in consumed_bases and _all_exist(name_index, _TENSOR6_SUFFIXES, base):
                component_map[base] = [name_index[_normalize_name(f"{base}{s}")] for s in _TENSOR6_SUFFIXES]
                consumed.update(component_map[base])
                consumed_bases.add(base_key)

    # Vector(3)
    for name in names:
        name_upper = name.upper()
        for suffix in _VECTOR_SUFFIXES:
            if not name_upper.endswith(suffix):
                continue
            base = name[: -len(suffix)]
            base_key = _normalize_name(base)
            if base and base_key not in consumed_bases and _all_exist(name_index, _VECTOR_SUFFIXES, base):
                component_map[base] = [name_index[_normalize_name(f"{base}{s}")] for s in _VECTOR_SUFFIXES]
                consumed.update(component_map[base])
                consumed_bases.add(base_key)

    # Scalars (or unmatched fields) keep identity mapping
    for name in names:
        if name in consumed:
            continue
        if name in component_map:
            continue
        component_map[name] = [name]

    return component_map


def is_component_like_variable(name: str) -> bool:
    """Return True if a variable name looks like a component-style name."""
    if not name:
        return False

    upper_name = name.upper()

    # Tensor-style suffixes are specific enough; use case-insensitive matching.
    for suffix in _TENSOR9_SUFFIXES + _TENSOR6_SUFFIXES:
        if len(name) > len(suffix) and upper_name.endswith(suffix):
            return True

    # Vector-style single-letter suffixes are prone to false positives
    # (e.g., logical variable "velocity"). Keep this check strict: only
    # treat explicit uppercase suffix forms as component-like.
    for suffix in _VECTOR_SUFFIXES:
        if len(name) > len(suffix) and name.endswith(suffix):
            return True

    return False


def ensure_logical_variables_only(logical_variables: Sequence[str]) -> None:
    bad = [v for v in logical_variables if is_component_like_variable(v)]
    if bad:
        raise ValueError(
            "task.variables must contain logical variable names only (no component names). "
            f"Invalid entries: {bad}"
        )


def resolve_effective_variables(
    logical_variables: Sequence[str],
    component_map: Dict[str, List[str]],
    strict_missing: bool = True,
) -> List[str]:
    """Expand logical variable names to ordered component variable names."""
    normalized_index: Dict[str, str] = {}
    for key in component_map.keys():
        normalized_key = _normalize_name(key)
        if normalized_key in normalized_index and normalized_index[normalized_key] != key:
            raise ValueError(
                "Ambiguous logical variable names in component_map (case-insensitive conflict). "
                f"conflict={normalized_index[normalized_key]} vs {key}"
            )
        normalized_index[normalized_key] = key

    effective: List[str] = []
    missing: List[str] = []

    for logical in logical_variables:
        comps = component_map.get(logical)
        if not comps:
            mapped = normalized_index.get(_normalize_name(logical))
            comps = component_map.get(mapped) if mapped else None
        if not comps:
            missing.append(logical)
            continue
        effective.extend(comps)

    if missing and (strict_missing or not effective):
        available = sorted(component_map.keys())
        raise ValueError(
            "Requested logical variables not found in dataset. "
            f"missing={missing}, available={available}"
        )

    # keep order + deduplicate
    return list(dict.fromkeys(effective))


def build_variable_catalog(component_map: Dict[str, List[str]]) -> Dict[str, Dict[str, object]]:
    catalog: Dict[str, Dict[str, object]] = {}
    for logical, comps in component_map.items():
        n = len(comps)
        if n == 1:
            kind = "scalar"
        elif n == 3:
            kind = "vector"
        elif n in (6, 9):
            kind = "tensor"
        else:
            kind = "multi_component"
        catalog[logical] = {
            "kind": kind,
            "components": list(comps),
            "num_components": n,
        }
    return catalog