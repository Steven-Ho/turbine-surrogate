import copy
import hashlib
import json
# Imports `os` for reuse in this module.
import os
from pathlib import Path
from typing import Dict, List, Tuple
from xml.etree import ElementTree as ET

# Imports `SampleMap` from `tfai.readers.sample_map` for use in this module.
from tfai.readers.sample_map import SampleMap
from tfai.utils.torch_utils import atomic_torch_save, load_torch


PREPROCESS_MANIFEST_FILENAME = "preprocess_manifest.pt"
PREPROCESS_CACHE_SCHEMA_VERSION = 2

# Initializes the `_RUNTIME_GENERAL_KEYS` as a collection for values from this processing stage.
_RUNTIME_GENERAL_KEYS = {
    "project_name",
    "experiment",
    "train_processed_dir",
    # Completes the multi-line `collection` arguments or elements.
    "device",
    "cpu_affinity_ratio",
    "deterministic",
}


# 计算文件内容哈希，用于识别样本表修改。
def _file_sha256(path: Path) -> str:
    """计算文件内容哈希，用于识别样本表修改。"""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            # Adds the current value to `digest` for later aggregation.
            digest.update(chunk)
    return digest.hexdigest()


def _file_metadata(path: Path) -> Dict:
    """提取文件元数据；缺失或不可访问也编码进指纹以触发缓存失效。"""
    try:
        # Stores the `stat` for use by the remaining steps in this flow.
        stat = path.stat()
    except OSError as exc:
        return {
            "path": str(path),
            # Completes the multi-line `collection` arguments or elements.
            "status": "unavailable",
            "reason": type(exc).__name__,
        }
    return {
        # Completes the multi-line `str` arguments or elements.
        "path": str(path),
        "status": "file",
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    # Completes the multi-line `collection` arguments or elements.
    }


def _resolve_manifest_step_paths(source_path: Path) -> List[Path]:
    """从 PVD 或 vtk.series 中解析直接引用的时间步文件。"""
    suffix = source_path.suffix.lower()
    if suffix == ".pvd":
        # Stores the `root` for use by the remaining steps in this flow.
        root = ET.parse(source_path).getroot()
        result = []
        for node in root.iter():
            if str(node.tag).lower().endswith("dataset"):
                # Reads `file` from the mapping into the `file_name`.
                file_name = node.attrib.get("file")
                if file_name:
                    result.append((source_path.parent / file_name).resolve())
        return result

    # Checks `suffix == ".series"` before entering this conditional branch.
    if suffix == ".series":
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        entries = payload.get("files", []) if isinstance(payload, dict) else []
        return [
            # Completes the multi-line `(source_path.parent / str(item['name'])).resolve` arguments or elements.
            (source_path.parent / str(item["name"])).resolve()
            for item in entries
            if isinstance(item, dict) and item.get("name")
        ]
    # Returns `[]` to the caller for the next stage.
    return []


def _fingerprint_source_path(source_path: str) -> Dict:
    """构建单个源文件、时序清单或目录的数据元数据指纹。"""
    path = Path(source_path).resolve()
    if not path.exists():
        # Returns `{"path": str(path), "status": "missing"}` to the caller for the next stage.
        return {"path": str(path), "status": "missing"}

    if path.is_dir():
        entries = []
        try:
            # Visits each item in `` and applies the current processing step.
            for item in sorted(path.rglob("*"), key=lambda candidate: str(candidate).lower()):
                if item.is_file():
                    metadata = _file_metadata(item)
                    metadata["relative_path"] = str(item.relative_to(path))
                    # Calls `metadata.pop` to perform the operation required at this stage.
                    metadata.pop("path", None)
                    entries.append(metadata)
        except OSError as exc:
            return {
                # Completes the multi-line `str` arguments or elements.
                "path": str(path),
                "status": "unavailable_directory",
                "reason": type(exc).__name__,
            }
        # Returns `{"path": str(path), "status": "directory", "entries": entries}` to the caller for the next stage.
        return {"path": str(path), "status": "directory", "entries": entries}

    result = _file_metadata(path)
    if result.get("status") != "file" or path.suffix.lower() not in {".pvd", ".series"}:
        return result

    # Keeps the operation and its recovery path within the same exception boundary.
    try:
        step_paths = _resolve_manifest_step_paths(path)
    except (ET.ParseError, OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, KeyError) as exc:
        result["manifest_status"] = "unparseable"
        # Stores the `result['manifest_reason']` for use by the remaining steps in this flow.
        result["manifest_reason"] = type(exc).__name__
        return result

    result["manifest_steps"] = [
        _file_metadata(step_path)
        # Completes the multi-line `sorted` arguments or elements.
        for step_path in sorted(step_paths, key=lambda candidate: str(candidate).lower())
    ]
    return result


def build_training_data_fingerprint(config: Dict) -> Dict:
    """基于样本表和原始源文件构建缓存失效指纹。"""
    # Stores the `general_cfg` for use by the remaining steps in this flow.
    general_cfg = config.get("general", {}) if isinstance(config, dict) else {}
    samples_path = str(general_cfg.get("train_samples", "") or "").strip()
    if not samples_path:
        return {"status": "samples_path_missing"}

    # Normalizes the `samples_file` before subsequent file operations.
    samples_file = Path(samples_path).resolve()
    table_fingerprint = _file_metadata(samples_file)
    if table_fingerprint.get("status") == "file":
        try:
            # Stores the `table_fingerprint['sha256']` for use by the remaining steps in this flow.
            table_fingerprint["sha256"] = _file_sha256(samples_file)
        except OSError as exc:
            table_fingerprint["hash_status"] = "unavailable"
            table_fingerprint["hash_reason"] = type(exc).__name__

    # Initializes the `sources` as a collection for values from this processing stage.
    sources = []
    try:
        sample_map = SampleMap(str(samples_file))
        for sample_id in sorted(sample_map.get_all_samples()):
            # Stores the `raw_source` for use by the remaining steps in this flow.
            raw_source = sample_map.get_file_path(sample_id)
            if not raw_source:
                sources.append({"sample_id": int(sample_id), "status": "source_missing"})
                continue
            # Stores the `source_path` for use by the remaining steps in this flow.
            source_path = Path(raw_source)
            if not source_path.is_absolute():
                source_path = samples_file.parent / source_path
            sources.append(
                # Completes the multi-line `collection` arguments or elements.
                {
                    "sample_id": int(sample_id),
                    "source": _fingerprint_source_path(str(source_path)),
                }
            # Completes the multi-line `sources.append` arguments or elements.
            )
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        sources.append({"status": "sample_table_unreadable", "reason": type(exc).__name__})

    return {"sample_table": table_fingerprint, "sources": sources}


# 仅保留会影响预处理或训练集划分的 general 配置。
def _build_general_signature(config: Dict) -> Dict:
    """仅保留会影响预处理或训练集划分的 general 配置。"""
    general_cfg = config.get("general", {}) if isinstance(config, dict) else {}
    return {
        key: copy.deepcopy(value)
        # Completes the multi-line `collection` arguments or elements.
        for key, value in general_cfg.items()
        if key not in _RUNTIME_GENERAL_KEYS
    }


def _build_processor_signature(config: Dict) -> Dict:
    """移除处理器类型及当前连边策略下无效的图参数。"""
    # Stores the `processor_cfg` for use by the remaining steps in this flow.
    processor_cfg = copy.deepcopy(config.get("processor", {}))
    processor_cfg.pop("type", None)
    strategy = str(processor_cfg.get("graph_edge_strategy", "topology") or "topology").lower()
    if strategy == "topology":
        # Calls `processor_cfg.pop` to perform the operation required at this stage.
        processor_cfg.pop("graph_knn_k", None)
        processor_cfg.pop("graph_radius", None)
        processor_cfg.pop("graph_radius_max_neighbors", None)
    elif strategy == "knn":
        # Calls `processor_cfg.pop` to perform the operation required at this stage.
        processor_cfg.pop("graph_radius", None)
        processor_cfg.pop("graph_radius_max_neighbors", None)
    return processor_cfg


def build_preprocess_signature(config: Dict, processor_type: str) -> Dict:
    # Returns `` to the caller for the next stage.
    return {
        "schema_version": PREPROCESS_CACHE_SCHEMA_VERSION,
        "general": _build_general_signature(config),
        "data": build_training_data_fingerprint(config),
        # Completes the multi-line `collection` arguments or elements.
        "split": {
            "seed": copy.deepcopy(config.get("general", {}).get("seed")),
            "valid_ratio": copy.deepcopy(config.get("training", {}).get("valid_ratio", 0.1)),
        },
        # Completes the multi-line `copy.deepcopy` arguments or elements.
        "task": copy.deepcopy(config.get("task", {})),
        "processor": _build_processor_signature(config),
        "processor_type": processor_type,
    }


# Defines `has_required_preprocess_artifacts` for has required preprocess artifacts logic.
def has_required_preprocess_artifacts(processed_dir: str, processor_type: str = "") -> bool:
    processed_path = Path(processed_dir)
    norm_exists = (processed_path / "norm_params.pt").exists()
    has_samples = any(processed_path.rglob("sample_*.pt"))
    # Returns early when `not (norm_exists and has_samples)` makes the remaining work inapplicable.
    if not (norm_exists and has_samples):
        return False
    if processor_type == "MatrixProcessor" and not (processed_path / "modes.pt").exists():
        return False
    # Returns `True` to the caller for the next stage.
    return True


def requires_preprocess_manifest(processor_type: str) -> bool:
    normalized = str(processor_type or "").lower()
    return normalized not in {"", "defaultscalarprocessor"}


# Defines `get_preprocess_manifest_path` for get preprocess manifest path logic.
def get_preprocess_manifest_path(processed_dir: str) -> str:
    return str(Path(processed_dir) / PREPROCESS_MANIFEST_FILENAME)


def save_preprocess_manifest(
    processed_dir: str,
    # Continues the surrounding statement or collection definition.
    expected_sample_ids: List[int],
    completed_sample_ids: List[int],
    processor_type: str,
    stage: str = "train",
    # Continues the surrounding statement or collection definition.
    status: str = "in_progress",
) -> str:
    manifest_path = get_preprocess_manifest_path(processed_dir)
    payload = {
        # Completes the multi-line `collection` arguments or elements.
        "stage": stage,
        "status": status,
        "processor_type": processor_type,
        "expected_sample_ids": [int(sample_id) for sample_id in expected_sample_ids],
        # Completes the multi-line `int` arguments or elements.
        "completed_sample_ids": [int(sample_id) for sample_id in completed_sample_ids],
    }
    atomic_torch_save(payload, manifest_path)
    return manifest_path


# Defines `load_preprocess_manifest` for load preprocess manifest logic.
def load_preprocess_manifest(processed_dir: str) -> Dict:
    manifest_path = get_preprocess_manifest_path(processed_dir)
    if not os.path.exists(manifest_path):
        return {}
    # Keeps the operation and its recovery path within the same exception boundary.
    try:
        payload = load_torch(manifest_path, weights_only=False)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        # Returns `{}` to the caller for the next stage.
        return {}


def validate_preprocess_manifest(processed_dir: str, processor_type: str) -> Tuple[bool, str]:
    manifest = load_preprocess_manifest(processed_dir)
    if not manifest:
        # Returns `` to the caller for the next stage.
        return False, "Preprocess cache miss: preprocess_manifest.pt is missing or unreadable"

    expected_ids = [int(sample_id) for sample_id in manifest.get("expected_sample_ids", [])]
    completed_ids = [int(sample_id) for sample_id in manifest.get("completed_sample_ids", [])]
    if not expected_ids:
        # Returns `` to the caller for the next stage.
        return False, "Preprocess cache miss: preprocess manifest does not list expected samples"

    if manifest.get("processor_type") != processor_type:
        return False, "Preprocess cache miss: preprocess manifest processor type changed"

    if manifest.get("stage") != "train":
        # Returns `` to the caller for the next stage.
        return False, "Preprocess cache miss: preprocess manifest stage is not train"

    if sorted(completed_ids) != sorted(expected_ids):
        return False, "Preprocess cache miss: preprocess manifest is incomplete"

    processed_path = Path(processed_dir)
    # Builds the `missing_paths` using the comprehension's filter and mapping rules.
    missing_paths = [
        sample_id
        for sample_id in expected_ids
        if not (processed_path / f"sample_{sample_id}.pt").exists()
    # Continues the surrounding statement or collection definition.
    ]
    if missing_paths:
        return False, "Preprocess cache miss: preprocess manifest references missing sample files"

    return True, "preprocess_manifest.pt validated"


# Defines `matrix_modes_match_config` for matrix modes match config logic.
def matrix_modes_match_config(processed_dir: str, config: Dict, overrides: Dict) -> Tuple[bool, str]:
    modes_path = Path(processed_dir) / "modes.pt"
    if not modes_path.exists():
        return False, "Preprocess cache miss: missing modes.pt"

    # Keeps the operation and its recovery path within the same exception boundary.
    try:
        payload = load_torch(str(modes_path), weights_only=False)
    except Exception as exc:
        return False, f"Preprocess cache miss: modes.pt is unreadable ({exc})"

    # Returns early when `not isinstance(payload, dict)` makes the remaining work inapplicable.
    if not isinstance(payload, dict):
        return False, "Preprocess cache miss: modes.pt has an invalid format"

    var_names = list(payload.get("var_names") or [])
    coeff_sizes = payload.get("coeff_sizes") or {}
    # Stores the `coeff_slices` for use by the remaining steps in this flow.
    coeff_slices = payload.get("coeff_slices") or {}
    expected_vars = list(config.get("task", {}).get("effective_variables") or [])
    expected_out = (
        overrides.get("model", {}).get("out_channels")
        # Completes the multi-line `isinstance` arguments or elements.
        if isinstance(overrides, dict)
        else None
    )

    if expected_vars and var_names != expected_vars:
        # Returns `` to the caller for the next stage.
        return False, (
            "Preprocess cache miss: modes.pt variable list is stale "
            f"(cached={var_names}, expected={expected_vars})"
        )

    # Returns early when `not isinstance(coeff_sizes, dict) or not coeff_sizes` makes the remaining work inapplicable.
    if not isinstance(coeff_sizes, dict) or not coeff_sizes:
        return False, "Preprocess cache miss: modes.pt is missing coeff_sizes"
    if not isinstance(coeff_slices, dict) or not coeff_slices:
        return False, "Preprocess cache miss: modes.pt is missing coeff_slices"

    # Converts the `total_coeff_dim` to `int` so later checks use a consistent type.
    total_coeff_dim = int(sum(int(v) for v in coeff_sizes.values()))
    if expected_out is not None and int(expected_out) != total_coeff_dim:
        return False, (
            "Preprocess cache miss: modes.pt coefficient dimension is stale "
            # Completes the multi-line `collection` arguments or elements.
            f"(cached={total_coeff_dim}, expected={expected_out})"
        )

    return True, "modes.pt validated"


def load_preprocess_cache(cache_path: str) -> Dict:
    # Returns early when `not os.path.exists(cache_path)` makes the remaining work inapplicable.
    if not os.path.exists(cache_path):
        return {}
    try:
        payload = load_torch(cache_path, weights_only=False)
        # Returns `payload if isinstance(payload, dict) else {}` to the caller for the next stage.
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def is_cache_override_complete(overrides: Dict, processor_type: str) -> bool:
    # Stores the `model_overrides` for use by the remaining steps in this flow.
    model_overrides = overrides.get("model", {}) if isinstance(overrides, dict) else {}
    if not isinstance(model_overrides, dict):
        return False

    if "in_channels" not in model_overrides or "out_channels" not in model_overrides:
        # Returns `False` to the caller for the next stage.
        return False
    if processor_type == "ImageVoxelProcessor" and "img_size" not in model_overrides:
        return False
    if processor_type == "MatrixProcessor" and "output_shape" not in model_overrides:
        # Returns `False` to the caller for the next stage.
        return False
    return True


def equivalent_preprocess_signatures(left: Dict, right: Dict) -> bool:
    """Allow relocating identical sample tables while retaining source fingerprints."""
    def canonical(signature):
        value = copy.deepcopy(signature)
        value.get("general", {}).pop("train_samples", None)
        table = value.get("data", {}).get("sample_table", {})
        # A content hash is required before treating table location/mtime as incidental.
        if table.get("sha256") and table.get("status") == "file":
            table.pop("path", None)
            table.pop("mtime_ns", None)
        return value
    return canonical(left) == canonical(right)


def evaluate_preprocess_cache_hit(config: Dict, processed_dir: str, processor_type: str, cache_path: str):
    """Return whether preprocess artifacts can be reused safely."""
    payload = load_preprocess_cache(cache_path)
    # Returns early when `not payload` makes the remaining work inapplicable.
    if not payload:
        return False, "Preprocess cache miss: cache file is missing or unreadable", cache_path, {}

    if payload.get("schema_version") != PREPROCESS_CACHE_SCHEMA_VERSION:
        return False, "Preprocess cache miss: cache schema version changed", cache_path, {}

    # Stores the `current_signature` for use by the remaining steps in this flow.
    current_signature = build_preprocess_signature(config, processor_type)
    cached_signature = payload.get("signature", {}) if isinstance(payload, dict) else {}
    if not equivalent_preprocess_signatures(cached_signature, current_signature):
        if cached_signature.get("data") != current_signature.get("data"):
            # Returns `` to the caller for the next stage.
            return False, "Preprocess cache miss: raw data fingerprint changed", cache_path, {}
        if cached_signature.get("split") != current_signature.get("split"):
            return False, "Preprocess cache miss: training split configuration changed", cache_path, {}
        return False, "Preprocess cache miss: general/task/processor configuration changed", cache_path, {}

    # Returns early when `` makes the remaining work inapplicable.
    if not has_required_preprocess_artifacts(processed_dir, processor_type):
        return False, "Preprocess cache miss: missing norm_params.pt or sample_*.pt", cache_path, {}

    if requires_preprocess_manifest(processor_type):
        manifest_ok, manifest_status = validate_preprocess_manifest(
            # Completes the multi-line `validate_preprocess_manifest` arguments or elements.
            processed_dir=processed_dir,
            processor_type=processor_type,
        )
        if not manifest_ok:
            # Returns `False, manifest_status, cache_path, {}` to the caller for the next stage.
            return False, manifest_status, cache_path, {}

    cached_overrides = payload.get("overrides", {}) if isinstance(payload, dict) else {}
    if not is_cache_override_complete(cached_overrides, processor_type):
        return False, "Preprocess cache miss: cached model overrides are incomplete", cache_path, {}

    # Checks `processor_type == "MatrixProcessor"` before entering this conditional branch.
    if processor_type == "MatrixProcessor":
        modes_ok, modes_status = matrix_modes_match_config(
            processed_dir=processed_dir,
            config=config,
            # Completes the multi-line `matrix_modes_match_config` arguments or elements.
            overrides=cached_overrides,
        )
        if not modes_ok:
            return False, modes_status, cache_path, {}

    # Returns `True, "预处理缓存命中", cache_path, cached_overrides` to the caller for the next stage.
    return True, "预处理缓存命中", cache_path, cached_overrides


def apply_cached_overrides(config: Dict, overrides: Dict, processor_type: str) -> None:
    model_cfg = config.setdefault("model", {})
    task_cfg = config.setdefault("task", {})
    # Stores the `processor_cfg` for use by the remaining steps in this flow.
    processor_cfg = config.setdefault("processor", {})

    model_overrides = overrides.get("model", {}) if isinstance(overrides, dict) else {}
    for key in ["in_channels", "out_channels", "img_size", "output_shape"]:
        if key in model_overrides:
            # Stores the `model_cfg[key]` for use by the remaining steps in this flow.
            model_cfg[key] = copy.deepcopy(model_overrides[key])

    task_overrides = overrides.get("task", {}) if isinstance(overrides, dict) else {}
    for key in [
        "logical_variables",
        # Completes the multi-line `collection` arguments or elements.
        "effective_variables",
        "field_component_map",
        "variable_catalog",
        "effective_variable_sources",
        # Completes the multi-line `collection` arguments or elements.
        "point_or_cell",
        "field_location_map",
        "projected_point_or_cell",
        "mixed_field_policy",
        # Completes the multi-line `collection` arguments or elements.
        "projection_method",
        "mesh_dimension",
        "geometry_regime",
        "geometry_motion",
        # Completes the multi-line `collection` arguments or elements.
        "displacement_logical_variable",
        "displacement_effective_variables",
    ]:
        if key in task_overrides:
            # Stores the `task_cfg[key]` for use by the remaining steps in this flow.
            task_cfg[key] = copy.deepcopy(task_overrides[key])

    processor_cfg["type"] = processor_type


def restore_cached_processor_state(
    processor,
    # Continues the surrounding statement or collection definition.
    processed_dir: str,
    processor_type: str,
) -> bool:
    """在缓存命中后恢复归一化参数及处理器专有运行态。"""

    if processor is None:
        # Returns `False` to the caller for the next stage.
        return False
    if not str(processed_dir or "").strip():
        return False

    restored = False
    # Stores the `processed_path` for use by the remaining steps in this flow.
    processed_path = Path(processed_dir)
    norm_params_path = processed_path / "norm_params.pt"
    set_norm_params = getattr(processor, "set_normalization_params", None)
    if norm_params_path.exists() and callable(set_norm_params):
        # Stores the normalization statistics for use by the remaining steps in this flow.
        norm_params = load_torch(str(norm_params_path), weights_only=False)
        if not isinstance(norm_params, dict):
            raise ValueError(f"预处理缓存中的归一化参数格式无效: {norm_params_path}")
        set_norm_params(norm_params)
        # Stores the `restored` for use by the remaining steps in this flow.
        restored = True

    if processor_type != "MatrixProcessor":
        return restored

    coeff_slices = getattr(processor, "coeff_slices", {}) or {}
    # Stores the `pod_modes` for use by the remaining steps in this flow.
    pod_modes = getattr(processor, "pod_modes", {}) or {}
    mean_fields = getattr(processor, "mean_fields", {}) or {}
    coeff_sizes = getattr(processor, "coeff_sizes", {}) or {}
    if coeff_slices and pod_modes and mean_fields and coeff_sizes:
        # Returns `restored` to the caller for the next stage.
        return restored

    load_pod_modes = getattr(processor, "load_pod_modes", None)
    if not callable(load_pod_modes):
        raise ValueError("MatrixProcessor 缺少 load_pod_modes，无法从缓存恢复 POD 模态")

    # Derives the `modes_path` from the current values for downstream processing.
    modes_path = processed_path / "modes.pt"
    if not modes_path.exists():
        raise FileNotFoundError(f"缓存命中但缺少 POD 模态文件: {modes_path}")

    load_pod_modes(str(modes_path))
    # Returns `True` to the caller for the next stage.
    return True


def save_preprocess_cache(cache_path: str, config: Dict, processor_type: str) -> None:
    payload = {
        "schema_version": PREPROCESS_CACHE_SCHEMA_VERSION,
        # Completes the multi-line `build_preprocess_signature` arguments or elements.
        "signature": build_preprocess_signature(config, processor_type),
        "overrides": {
            "model": {
                key: copy.deepcopy(config.get("model", {}).get(key))
                # Completes the multi-line `collection` arguments or elements.
                for key in ["in_channels", "out_channels", "img_size", "output_shape"]
                if key in config.get("model", {})
            },
            "task": {
                # Completes the multi-line `copy.deepcopy` arguments or elements.
                key: copy.deepcopy(config.get("task", {}).get(key))
                for key in [
                    "logical_variables",
                    "effective_variables",
                    # Completes the multi-line `collection` arguments or elements.
                    "field_component_map",
                    "variable_catalog",
                    "effective_variable_sources",
                    "point_or_cell",
                    # Completes the multi-line `collection` arguments or elements.
                    "field_location_map",
                    "projected_point_or_cell",
                    "mixed_field_policy",
                    "projection_method",
                    # Completes the multi-line `collection` arguments or elements.
                    "mesh_dimension",
                    "geometry_regime",
                    "geometry_motion",
                    "displacement_logical_variable",
                    # Completes the multi-line `collection` arguments or elements.
                    "displacement_effective_variables",
                ]
                if key in config.get("task", {})
            },
            # Completes the multi-line `collection` arguments or elements.
            "processor_type": processor_type,
        },
    }
    atomic_torch_save(payload, cache_path)
