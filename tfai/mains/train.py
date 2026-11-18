"""训练接口。"""

import copy
import gc
import inspect
# Imports `json` for reuse in this module.
import json
import os
import shutil
import time
# Imports `Path` from `pathlib` for use in this module.
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.optim as optim

# Imports `DataPipeline` from `tfai.data.data_pipeline` for use in this module.
from tfai.data.data_pipeline import DataPipeline
from tfai.data.temporal_utils import is_temporal_manifest_path, list_temporal_sequence_files
from tfai.dataset import DataLoader, build_train_valid_datasets
from tfai.dataset.splitter import Splitter
# Imports `Evaluator` from `tfai.evaluation.evaluator` for use in this module.
from tfai.evaluation.evaluator import Evaluator
from tfai.evaluation.evaluation_runner import EvaluationRunner
from tfai.factories.model_factory import ModelFactory
from tfai.factories.processor_factory import ProcessorFactory
# Imports `ReaderFactory` from `tfai.factories.reader_factory` for use in this module.
from tfai.factories.reader_factory import ReaderFactory
from tfai.readers.sample_map import SampleMap
from tfai.training.trainer import (
    ResidualStageTrainer,
    # Continues the surrounding statement or collection definition.
    Trainer,
    get_processor_checkpoint_artifact_policy,
)
from tfai.training.preprocess_cache import (
    # Continues the surrounding statement or collection definition.
    apply_cached_overrides,
    build_preprocess_signature,
    evaluate_preprocess_cache_hit,
    restore_cached_processor_state,
    # Continues the surrounding statement or collection definition.
    save_preprocess_cache,
)
from tfai.training.preprocess_resume import (
    PreprocessResumeManager,
    # Continues the surrounding statement or collection definition.
    cleanup_resume_state_if_compatible,
    restore_matrix_resume_modes,
    restore_resume_processor_state,
)
# Imports `resolve_training_variables` from `tfai.training.variable_manifest` for use in this module.
from tfai.training.variable_manifest import resolve_training_variables
from tfai.utils.cpu_affinity import (
    apply_cpu_affinity_limit,
    get_current_cpu_affinity_ids,
    # Continues the surrounding statement or collection definition.
    restore_cpu_affinity,
)
from tfai.utils.logger import Logger
from tfai.utils.model_bundle import RESIDUAL_ADDITIVE_TWO_STAGE_BUNDLE
# Imports `validate_qiyuan_config` from `tfai.utils.qiyuan_constraints` for use in this module.
from tfai.utils.qiyuan_constraints import validate_qiyuan_config
from tfai.utils.path_utils import resolve_path_from_anchor
from tfai.utils.random_utils import set_global_seed
from tfai.utils.runtime_paths import resolve_runtime_paths, sanitize_config_for_checkpoint
# Imports `DEFAULT_SOURCE_DIRNAME, DEFAULT_SOURCE_META_FILENAME, resolve_sample_source_path` from `tfai.utils.sample_utils` for use in this module.
from tfai.utils.sample_utils import (
    DEFAULT_SOURCE_DIRNAME,
    DEFAULT_SOURCE_META_FILENAME,
    resolve_sample_source_path,
# Continues the surrounding statement or collection definition.
)
from tfai.utils.task_variable_config import normalize_scalar_task_variable_aliases
from tfai.utils.task_constraints import validate_matrix_pod_fixed_geometry_only
from tfai.utils.torch_utils import atomic_torch_save, load_torch

# Imports `PROJECT_ROOT, apply_train_experiment_dir, clear_train_default_output_paths` from `_shared` for use in this module.
from ._shared import (
    PROJECT_ROOT,
    apply_train_experiment_dir,
    clear_train_default_output_paths,
    # Continues the surrounding statement or collection definition.
    ensure_train_runtime_paths,
    infer_parameter_schema_from_samples,
    load_sample_rows,
    write_predict_samples_csv,
# Continues the surrounding statement or collection definition.
)
from .evaluate import _normalize_evaluate_result


def _resolve_reference_template_source(source_path: str) -> str:
    """解析可用于复制 reference 的模板文件路径。"""
    candidate = os.path.abspath(str(source_path))
    # Checks `os.path.isfile(candidate)` before entering this conditional branch.
    if os.path.isfile(candidate):
        if is_temporal_manifest_path(candidate):
            ordered_files = list_temporal_sequence_files(
                None,
                # Completes the multi-line `list_temporal_sequence_files` arguments or elements.
                candidate,
                max_time_steps=1,
            )
            if not ordered_files:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(f"Temporal manifest has no usable step files: {candidate}")
            return ordered_files[0]
        return candidate

    if os.path.isdir(candidate):
        # Keeps the operation and its recovery path within the same exception boundary.
        try:
            ordered_files = list_temporal_sequence_files(
                None,
                candidate,
                # Completes the multi-line `list_temporal_sequence_files` arguments or elements.
                max_time_steps=1,
            )
        except ValueError:
            ordered_files = []
        # Returns early when `ordered_files` makes the remaining work inapplicable.
        if ordered_files:
            return ordered_files[0]

        file_names = [
            name for name in os.listdir(candidate)
            # Completes the multi-line `os.path.isfile` arguments or elements.
            if os.path.isfile(os.path.join(candidate, name))
        ]
        if not file_names:
            raise ValueError(f"参考目录中没有可复制的文件: {candidate}")

        # Calls `file_names.sort` to perform the operation required at this stage.
        file_names.sort()
        return os.path.join(candidate, file_names[0])

    raise FileNotFoundError(f"参考模板路径不存在: {candidate}")


def _predict_preprocess_requires_reference(config: Dict, processor) -> bool:
    """判断固定几何预测预处理阶段是否需要共享 reference。"""
    # Reads `task` from the mapping into the task settings with its declared fallback.
    task_cfg = config.get("task", {})
    mode = str(task_cfg.get("mode", "")).strip().lower()
    temporal_strategy = str(task_cfg.get("temporal_strategy", "direct")).strip().lower()

    supports_fast_predict = getattr(processor, "supports_fast_predict_path", None)
    # Stores the `supports_reference_only` for use by the remaining steps in this flow.
    supports_reference_only = getattr(processor, "supports_reference_only_predict_path", None)
    supports_fixed_temporal = getattr(
        processor,
        "supports_fixed_geometry_temporal_predict_path",
        # Completes the multi-line `getattr` arguments or elements.
        None,
    )
    requires_fixed_temporal_reference = getattr(
        processor,
        # Completes the multi-line `getattr` arguments or elements.
        "requires_fixed_geometry_temporal_reference",
        None,
    )

    if mode == "steady":
        # Returns early when `callable(supports_reference_only) and supports_reference_only()` makes the remaining work inapplicable.
        if callable(supports_reference_only) and supports_reference_only():
            return True
        return False

    if mode == "temporal" and temporal_strategy == "direct":
        # Returns early when `callable(supports_fast_predict) and supports_fast_predict(False)` makes the remaining work inapplicable.
        if callable(supports_fast_predict) and supports_fast_predict(False):
            return False
        return True

    if mode == "temporal" and temporal_strategy == "autoregressive":
        # Checks `callable(supports_fixed_temporal) and supports_fixed_temporal()` before entering this conditional branch.
        if callable(supports_fixed_temporal) and supports_fixed_temporal():
            if callable(requires_fixed_temporal_reference):
                return bool(requires_fixed_temporal_reference())
            return False
        # Returns `False` to the caller for the next stage.
        return False

    return False


def _predict_writeback_requires_reference(processor) -> bool:
    """判断固定几何预测写回阶段是否需要共享 reference。"""
    requires_writeback_reference = getattr(
        # Completes the multi-line `getattr` arguments or elements.
        processor,
        "requires_fixed_geometry_predict_writeback_reference",
        None,
    )
    # Returns early when `callable(requires_writeback_reference)` makes the remaining work inapplicable.
    if callable(requires_writeback_reference):
        return bool(requires_writeback_reference())
    return True


def _set_nested_value(payload: Dict, path: Tuple[str, ...], value) -> None:
    # Stores the `current` for use by the remaining steps in this flow.
    current = payload
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


# 构建训练终止检查器。
def _build_terminate_training_checker(
    terminate_training_checker: Optional[Callable[[], bool]],
) -> Callable[[], bool]:
    """构建训练终止检查器。"""
    if terminate_training_checker is None:
        # Returns `lambda: False` to the caller for the next stage.
        return lambda: False

    if callable(terminate_training_checker):
        return lambda: bool(terminate_training_checker())

    raise TypeError("terminate_training_checker 仅支持 callable 或 None")


# 将后端入参映射回训练配置。
def _apply_train_params(config: Dict, train_params: Optional[Dict]) -> Dict:
    """将后端入参映射回训练配置。"""
    mapped_config = config
    payload = dict(train_params or {})

    mapping = {
        # Completes the multi-line `collection` arguments or elements.
        "device": ("general", "device"),
        "model_scale": ("model", "scale"),
        "norm_type": ("processor", "norm_type"),
        "base_resolution": ("processor", "base_resolution"),
        # Completes the multi-line `collection` arguments or elements.
        "graph_max_edges_per_node": ("processor", "graph_max_edges_per_node"),
        "batch_size": ("training", "batch_size"),
        "epochs": ("training", "epochs"),
        "max_steps": ("training", "max_steps"),
        # Completes the multi-line `collection` arguments or elements.
        "validate_every": ("training", "validate_every"),
        "validation_max_batches": ("training", "validation_max_batches"),
        "auto_evaluate_after_train": ("training", "auto_evaluate_after_train"),
        "temporal_rollout": ("training", "temporal_rollout"),
        # Completes the multi-line `collection` arguments or elements.
        "learning_rate": ("training", "learning_rate"),
        "valid_ratio": ("training", "valid_ratio"),
        "loss_function": ("training", "loss_function"),
        "patience": ("training", "early_stopping", "patience"),
        # Completes the multi-line `collection` arguments or elements.
        "graph_use_partition": ("training", "graph_use_partition"),
        "graph_partition_count": ("training", "graph_partition_count"),
    }
    nullable_keys = {
        # Completes the multi-line `collection` arguments or elements.
        "max_steps",
        "validate_every",
        "validation_max_batches",
        "auto_evaluate_after_train",
    # Completes the multi-line `collection` arguments or elements.
    }

    for key, path in mapping.items():
        if key in payload and (payload[key] is not None or key in nullable_keys):
            _set_nested_value(mapped_config, path, payload[key])

    # Returns `mapped_config` to the caller for the next stage.
    return mapped_config


def _is_auto_evaluate_after_train_enabled(training_cfg: Dict) -> bool:
    """解析训练结束自动评估开关，缺省时保持旧行为。"""
    value = training_cfg.get("auto_evaluate_after_train", True)
    if value is None:
        # Returns `True` to the caller for the next stage.
        return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            # Returns `True` to the caller for the next stage.
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return bool(value)


# 规范化训练样本列表并校验关键字段。
def _normalize_train_samples(
    samples_train: List[Dict],
    task_type: str,
) -> List[Dict]:
    """规范化训练样本列表并校验关键字段。"""
    # Validates `not samples_train` and raises `ValueError` when the input cannot be processed safely.
    if not samples_train:
        raise ValueError("samples_train 不能为空")

    normalized_items: List[Dict] = []
    seen_orders = set()
    # Initializes the `reserved_keys` as a collection for values from this processing stage.
    reserved_keys = {
        "order",
        "sample_id",
        "params",
        # Completes the multi-line `collection` arguments or elements.
        "source_file",
        "file_path",
        "resolved_path",
        "path_type",
        # Completes the multi-line `collection` arguments or elements.
        "processed_path",
        "subset",
        "exists",
    }
    # Stores the `require_source_file` for use by the remaining steps in this flow.
    require_source_file = str(task_type).lower() == "field"

    for index, item in enumerate(samples_train):
        if not isinstance(item, dict):
            raise TypeError("samples_train 中的每个样本都必须是字典")

        # Converts the `order` to `int` so later checks use a consistent type.
        order = int(item.get("order", item.get("sample_id", index)))
        if order in seen_orders:
            raise ValueError(f"samples_train 中存在重复的样本编号: {order}")
        seen_orders.add(order)

        # Stores the `source_file` for use by the remaining steps in this flow.
        source_file = resolve_sample_source_path(item)

        if require_source_file:
            if not source_file:
                raise ValueError(
                    # Completes the multi-line `ValueError` arguments or elements.
                    f"训练样本缺少 source_file（兼容 file_path/resolved_path）: order={order}"
                )
            if not os.path.exists(source_file):
                raise FileNotFoundError(f"训练原始样本文件不存在: {source_file}")

        # Stores the `params` for use by the remaining steps in this flow.
        params = dict(item.get("params") or {})
        for key, value in item.items():
            if key not in reserved_keys and key not in params:
                params[key] = value

        # Adds the current value to `normalized_items` for later aggregation.
        normalized_items.append(
            {
                "order": order,
                "source_file": source_file,
                # Completes the multi-line `collection` arguments or elements.
                "params": params,
            }
        )

    return normalized_items


# 将内存训练样本落为训练样本表。
def _materialize_train_samples_csv(config: Dict, samples_train: List[Dict]) -> str:
    """将内存训练样本落为训练样本表。"""
    general_cfg = config.setdefault("general", {})
    sample_csv_path = os.path.abspath(str(general_cfg["train_samples"]))
    sample_dir = os.path.dirname(sample_csv_path)
    # Calls `os.makedirs` to perform the operation required at this stage.
    os.makedirs(sample_dir, exist_ok=True)
    general_cfg["train_samples"] = sample_csv_path
    return write_predict_samples_csv(samples_train, sample_csv_path)


def _persist_prediction_variable_metadata(config: Dict, samples_train: List[Dict]) -> None:
    """将预测所需的输入输出变量元数据固化到训练配置。"""
    # Stores the task settings for use by the remaining steps in this flow.
    task_cfg = config.setdefault("task", {})
    parameter_schema = list(task_cfg.get("parameter_schema") or [])
    if not parameter_schema:
        parameter_schema = infer_parameter_schema_from_samples(samples_train)
    # Checks `parameter_schema` before entering this conditional branch.
    if parameter_schema:
        task_cfg["parameter_schema"] = parameter_schema

    if parameter_schema and not task_cfg.get("input_variables"):
        task_cfg["input_variables"] = [str(item["name"]) for item in parameter_schema]

    # Checks `not task_cfg.get("output_variables")` before entering this conditional branch.
    if not task_cfg.get("output_variables"):
        output_variables = (
            task_cfg.get("variables")
            or task_cfg.get("logical_variables")
            # Completes the multi-line `task_cfg.get` arguments or elements.
            or task_cfg.get("effective_variables")
            or []
        )
        task_cfg["output_variables"] = list(output_variables)


# 固定几何场景下复制参考几何文件。
def _copy_reference_file_if_needed(
    config: Dict,
    samples_csv: str,
    train_dir: str,
    # Continues the surrounding statement or collection definition.
    logger,
    processor=None,
    reader=None,
) -> None:
    """固定几何场景下复制参考几何文件。"""
    # Reads `task` from the mapping into the task settings with its declared fallback.
    task_cfg = config.get("task", {})
    if str(task_cfg.get("type", "field")).lower() != "field":
        return
    if bool(task_cfg.get("is_geometry_varying", False)):
        # Ends this function and returns control to its caller.
        return

    if processor is not None:
        preprocess_requires_reference = _predict_preprocess_requires_reference(
            config=config,
            # Completes the multi-line `_predict_preprocess_requires_reference` arguments or elements.
            processor=processor,
        )
        writeback_requires_reference = _predict_writeback_requires_reference(processor)
        if not preprocess_requires_reference and not writeback_requires_reference:
            # Records the current execution state to make progress and failures traceable.
            logger.info("当前固定几何预测链路不需要 reference，跳过复制参考文件")
            return

    sample_map = SampleMap(samples_csv)
    sample_ids = sample_map.get_all_samples()
    # Returns early when `not sample_ids` makes the remaining work inapplicable.
    if not sample_ids:
        return

    for raw_sample_id in sample_ids:
        sample_id = int(raw_sample_id)
        # Stores the `sample_file` for use by the remaining steps in this flow.
        sample_file = sample_map.get_file_path(sample_id) or ""
        if not sample_file:
            continue

        sample_path = resolve_path_from_anchor(samples_csv, sample_file)
        # Keeps the operation and its recovery path within the same exception boundary.
        try:
            template_source = _resolve_reference_template_source(sample_path)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("样本 %s 的参考几何模板不可用，继续尝试后续样本: %s", sample_id, str(exc))
            # Handles the data or control-flow case represented by this statement.
            continue

        read_fn = getattr(reader, "read", None) if reader is not None else None
        if callable(read_fn):
            read_signature = inspect.signature(read_fn)
            # Initializes the `read_kwargs` as a collection for values from this processing stage.
            read_kwargs = {"file_path": template_source}
            if "sample_order" in read_signature.parameters:
                read_kwargs["sample_order"] = sample_id
            if "variables" in read_signature.parameters:
                # Initializes the `read_kwargs['variables']` as a collection for values from this processing stage.
                read_kwargs["variables"] = []
            if "allow_missing_fields" in read_signature.parameters:
                read_kwargs["allow_missing_fields"] = True
            try:
                # Calls `read_fn` to perform the operation required at this stage.
                read_fn(**read_kwargs)
            except Exception as exc:
                logger.warning("样本 %s 的参考几何源文件不可读，继续尝试后续样本: %s", sample_id, str(exc))
                continue

        # Selects the required item from `os.path.splitext(template_source)` as the `ext`.
        ext = os.path.splitext(template_source)[1]
        reference_path = os.path.join(train_dir, f"reference{ext}")
        shutil.copy2(template_source, reference_path)
        logger.info("已复制参考几何文件: %s", reference_path)
        # Ends this function and returns control to its caller.
        return

    logger.warning("未找到可用的参考几何模板，跳过复制 reference 文件")


def _resolve_default_source_mode(config: Dict) -> str:
    """解析模型包运行时获取预测源的方式。"""
    task_cfg = config.get("task", {})
    # Returns early when `str(task_cfg.get("type", "field")).strip().lower() != "field"` makes the remaining work inapplicable.
    if str(task_cfg.get("type", "field")).strip().lower() != "field":
        return ""

    if bool(task_cfg.get("is_geometry_varying", False)):
        return "default_source"

    # Stores the `mode` for use by the remaining steps in this flow.
    mode = str(task_cfg.get("mode", "steady")).strip().lower()
    temporal_strategy = str(task_cfg.get("temporal_strategy", "direct")).strip().lower()
    input_length = int(task_cfg.get("input_length", 1) or 1)
    if mode == "temporal" and temporal_strategy == "autoregressive" and input_length > 1:
        # Returns `"default_source"` to the caller for the next stage.
        return "default_source"

    return "reference"


def _build_default_source_smoke_params(sample_info: Optional[Dict]) -> Dict:
    """从默认样本表行提取可复用的预测参数。"""
    if not isinstance(sample_info, dict):
        # Returns `{}` to the caller for the next stage.
        return {}
    return {
        str(key): value
        for key, value in sample_info.items()
        # Completes the multi-line `str(key).strip().lower` arguments or elements.
        if str(key).strip().lower() not in {"file_path", "source_file", "resolved_path"}
    }


def _write_default_source_meta(
    meta_path: str,
    # Continues the surrounding statement or collection definition.
    source_mode: str,
    sample_order: Optional[int] = None,
    sample_info: Optional[Dict] = None,
    entry_type: str = "",
    # Continues the surrounding statement or collection definition.
    entry_relpath: str = "",
    original_source_path: str = "",
    is_temporal: bool = False,
) -> None:
    """写入模型包预测元数据，并兼容旧版默认源入口字段。"""
    # Stores the `default_source` for use by the remaining steps in this flow.
    default_source = None
    if entry_relpath:
        default_source = {
            "entry_type": entry_type,
            # Completes the multi-line `entry_relpath.replace` arguments or elements.
            "entry_relpath": entry_relpath.replace("\\", "/"),
            "original_source_path": original_source_path,
            "sample_order": sample_order,
            "is_temporal": bool(is_temporal),
        # Completes the multi-line `collection` arguments or elements.
        }

    meta = {
        "schema_version": 2,
        "source_mode": source_mode,
        # Completes the multi-line `collection` arguments or elements.
        "default_source": default_source,
        "smoke_test": {
            "params": _build_default_source_smoke_params(sample_info),
            "expected_sample_count": 1,
        # Completes the multi-line `collection` arguments or elements.
        },
    }
    if default_source:
        # 保留顶层入口字段，确保已发布版本仍能解析默认源。
        meta.update(default_source)

    # Scopes resource use so the context manager can release it on exit.
    with open(meta_path, "w", encoding="utf-8") as meta_file:
        json.dump(meta, meta_file, ensure_ascii=False, indent=2)


def _package_default_source_if_needed(
    config: Dict,
    # Continues the surrounding statement or collection definition.
    samples_csv: str,
    train_dir: str,
    logger,
) -> None:
    """按预测链路需要打包默认源，并写入模型包预测元数据。"""
    # Reads `task` from the mapping into the task settings with its declared fallback.
    task_cfg = config.get("task", {})
    if str(task_cfg.get("type", "field")).lower() != "field":
        return

    sample_map = SampleMap(samples_csv)
    # Stores the sample identifiers for use by the remaining steps in this flow.
    sample_ids = sample_map.get_all_samples()
    package_dir = os.path.join(train_dir, DEFAULT_SOURCE_DIRNAME)
    meta_path = os.path.join(train_dir, DEFAULT_SOURCE_META_FILENAME)
    source_mode = _resolve_default_source_mode(config)

    # Checks `os.path.isdir(package_dir)` before entering this conditional branch.
    if os.path.isdir(package_dir):
        shutil.rmtree(package_dir, ignore_errors=True)
    elif os.path.exists(package_dir):
        os.remove(package_dir)
    # Checks `os.path.exists(meta_path)` before entering this conditional branch.
    if os.path.exists(meta_path):
        os.remove(meta_path)

    first_sample_order = int(sample_ids[0]) if sample_ids else None
    first_sample_info = (
        # Completes the multi-line `sample_map.get_sample_info` arguments or elements.
        sample_map.get_sample_info(first_sample_order)
        if first_sample_order is not None
        else None
    )
    # Checks `source_mode != "default_source"` before entering this conditional branch.
    if source_mode != "default_source":
        _write_default_source_meta(
            meta_path=meta_path,
            source_mode=source_mode,
            # Completes the multi-line `_write_default_source_meta` arguments or elements.
            sample_order=first_sample_order,
            sample_info=first_sample_info,
            is_temporal=str(task_cfg.get("mode", "")).strip().lower() == "temporal",
        )
        # Records the current execution state to make progress and failures traceable.
        logger.info("当前预测链路使用 %s，跳过打包 default_source", source_mode)
        return

    if not sample_ids:
        _write_default_source_meta(meta_path=meta_path, source_mode=source_mode)
        # Records the current execution state to make progress and failures traceable.
        logger.warning("训练样本为空，未能打包 default_source")
        return

    for raw_sample_id in sample_ids:
        sample_id = int(raw_sample_id)
        # Stores the `sample_file` for use by the remaining steps in this flow.
        sample_file = sample_map.get_file_path(sample_id) or ""
        if not sample_file:
            continue

        sample_path = resolve_path_from_anchor(samples_csv, sample_file)
        # Checks `not os.path.exists(sample_path)` before entering this conditional branch.
        if not os.path.exists(sample_path):
            logger.warning("样本 %s 的默认预测源不存在，继续尝试后续样本: %s", sample_id, sample_path)
            continue

        entry_type = "file"
        # Stores the `entry_relpath` for use by the remaining steps in this flow.
        entry_relpath = ""
        try:
            if os.path.isdir(sample_path):
                shutil.copytree(sample_path, package_dir)
                # Stores the `entry_type` for use by the remaining steps in this flow.
                entry_type = "directory"
                entry_relpath = DEFAULT_SOURCE_DIRNAME
            elif is_temporal_manifest_path(sample_path):
                shutil.copytree(os.path.dirname(sample_path), package_dir)
                # Stores the `entry_type` for use by the remaining steps in this flow.
                entry_type = "manifest"
                entry_relpath = os.path.join(DEFAULT_SOURCE_DIRNAME, os.path.basename(sample_path))
            else:
                os.makedirs(package_dir, exist_ok=True)
                # Stores the `target_path` for use by the remaining steps in this flow.
                target_path = os.path.join(package_dir, os.path.basename(sample_path))
                shutil.copy2(sample_path, target_path)
                entry_relpath = os.path.join(DEFAULT_SOURCE_DIRNAME, os.path.basename(sample_path))
        except Exception as exc:
            # Records the current execution state to make progress and failures traceable.
            logger.warning("样本 %s 的默认预测源打包失败，继续尝试后续样本: %s", sample_id, str(exc))
            continue

        _write_default_source_meta(
            meta_path=meta_path,
            # Completes the multi-line `_write_default_source_meta` arguments or elements.
            source_mode=source_mode,
            sample_order=sample_id,
            sample_info=sample_map.get_sample_info(sample_id),
            entry_type=entry_type,
            # Completes the multi-line `_write_default_source_meta` arguments or elements.
            entry_relpath=entry_relpath,
            original_source_path=os.path.abspath(sample_path),
            is_temporal=str(task_cfg.get("mode", "")).strip().lower() == "temporal",
        )
        # Records the current execution state to make progress and failures traceable.
        logger.info("已打包默认预测源: %s", os.path.join(train_dir, entry_relpath))
        return

    _write_default_source_meta(
        meta_path=meta_path,
        # Completes the multi-line `_write_default_source_meta` arguments or elements.
        source_mode=source_mode,
        sample_order=first_sample_order,
        sample_info=first_sample_info,
        is_temporal=str(task_cfg.get("mode", "")).strip().lower() == "temporal",
    # Completes the multi-line `_write_default_source_meta` arguments or elements.
    )
    logger.warning("未找到可用的默认预测源，跳过打包 default_source")


def _cleanup_incompatible_processed_artifacts(
    processed_dir: str,
    # Continues the surrounding statement or collection definition.
    processor,
    logger,
) -> None:
    """清理与当前处理器链路不兼容的历史附属产物。"""
    artifact_policy = get_processor_checkpoint_artifact_policy(processor)
    # Initializes the `artifact_targets` as a collection for values from this processing stage.
    artifact_targets = {
        "modes.pt": artifact_policy["modes"],
        "processor_state.pt": artifact_policy["processor_state"],
    }

    # Visits each item in `artifact_targets.items()` and applies the current processing step.
    for artifact_name, should_keep in artifact_targets.items():
        if should_keep:
            continue

        artifact_path = os.path.join(processed_dir, artifact_name)
        # Checks `not os.path.exists(artifact_path)` before entering this conditional branch.
        if not os.path.exists(artifact_path):
            continue

        try:
            os.remove(artifact_path)
            # Records the current execution state to make progress and failures traceable.
            logger.info("已清理与当前处理器不兼容的历史产物: %s", artifact_path)
        except OSError as exc:
            logger.warning(
                "清理历史产物失败，将继续使用当前链路重新生成: %s, 原因: %s",
                # Completes the multi-line `logger.warning` arguments or elements.
                artifact_path,
                str(exc),
            )


def _run_auto_evaluation(model, config: Dict, dataset) -> Dict:
    """训练结束后在指定数据集上执行轻量自动评估。"""
    # Stores the `evaluator` for use by the remaining steps in this flow.
    evaluator = Evaluator(model, config)
    metrics, _, _ = evaluator.evaluate(
        dataset,
        plot_scatter=False,
        # Completes the multi-line `evaluator.evaluate` arguments or elements.
        scatter_save_path=None,
        scatter_csv_path=None,
    )
    return {
        # Completes the multi-line `collection` arguments or elements.
        "metrics": metrics,
        "scatter_data": copy.deepcopy(
            evaluator.last_scatter_data
            or {
                # Completes the multi-line `collection` arguments or elements.
                "overall": {"predictions": [], "targets": []},
                "by_variable": {},
            }
        ),
    # Completes the multi-line `collection` arguments or elements.
    }


def _normalize_auto_evaluate_result(
    auto_evaluate_result: Dict,
    sample_rows: List[Dict],
# Continues the surrounding statement or collection definition.
) -> Dict:
    """将训练末尾自动评估结果整理为与 run_evaluate 一致的结构。"""
    normalized_samples: List[Dict] = []
    for item in sample_rows:
        normalized_samples.append(
            # Completes the multi-line `collection` arguments or elements.
            {
                "order": int(item["order"]),
                "params": dict(item.get("params") or {}),
                "source_file": item.get("source_file") or item.get("resolved_path") or item.get("file_path"),
                # Completes the multi-line `collection` arguments or elements.
                "evaluated_result_path": None,
            }
        )

    return {
        # Completes the multi-line `dict` arguments or elements.
        "metrics": dict(auto_evaluate_result.get("metrics") or {}),
        "scatter_data": copy.deepcopy(
            auto_evaluate_result.get("scatter_data")
            or {
                # Completes the multi-line `collection` arguments or elements.
                "overall": {"predictions": [], "targets": []},
                "by_variable": {},
            }
        ),
        # Completes the multi-line `copy.deepcopy` arguments or elements.
        "writeback_report": copy.deepcopy(
            auto_evaluate_result.get("writeback_report") or {}
        ),
        "samples_evaluated": normalized_samples,
    # Completes the multi-line `collection` arguments or elements.
    }


def _build_auto_evaluate_samples(
    sample_rows: List[Dict],
    task_type: str,
    # Continues the surrounding statement or collection definition.
    processed_dir: str,
    subset: str = "valid",
) -> List[Dict]:
    """为训练末尾自动评估构建 EvaluationRunner 所需的样本列表。"""
    normalized_samples: List[Dict] = []
    # Stores the `normalized_task_type` for use by the remaining steps in this flow.
    normalized_task_type = str(task_type).lower()
    for item in sample_rows:
        sample_order = int(item["order"])
        if normalized_task_type == "scalar":
            # Stores the `processed_path` for use by the remaining steps in this flow.
            processed_path = os.path.join(processed_dir, subset, f"sample_{sample_order}.pt")
        else:
            processed_path = os.path.join(processed_dir, f"sample_{sample_order}.pt")

        normalized_samples.append(
            # Completes the multi-line `collection` arguments or elements.
            {
                "order": sample_order,
                "params": dict(item.get("params") or {}),
                "source_file": item.get("source_file") or item.get("resolved_path") or item.get("file_path"),
                # Completes the multi-line `collection` arguments or elements.
                "processed_path": processed_path,
            }
        )
    return normalized_samples


# Defines `_extract_sample_ids_from_processed_paths` for  extract sample ids from processed paths logic.
def _extract_sample_ids_from_processed_paths(processed_paths: List[str]) -> List[int]:
    sample_ids: List[int] = []
    for processed_path in processed_paths:
        file_name = os.path.basename(str(processed_path))
        # Checks `` before entering this conditional branch.
        if not file_name.startswith("sample_") or not file_name.endswith(".pt"):
            continue
        raw_id = file_name[len("sample_") : -3]
        try:
            # Adds the current value to `sample_ids` for later aggregation.
            sample_ids.append(int(raw_id))
        except ValueError:
            continue
    return sample_ids


# Defines `_rows_to_sample_payload` for  rows to sample payload logic.
def _rows_to_sample_payload(rows: List[Dict]) -> List[Dict]:
    payload: List[Dict] = []
    for row in rows:
        payload.append(
            # Completes the multi-line `collection` arguments or elements.
            {
                "order": int(row["order"]),
                "source_file": row.get("source_file") or row.get("resolved_path") or row.get("file_path", ""),
                "params": dict(row.get("params") or {}),
            # Completes the multi-line `collection` arguments or elements.
            }
        )
    return payload


def _build_effective_train_valid_csvs(
    # Continues the surrounding statement or collection definition.
    train_csv: str,
    valid_csv: str,
    processed_paths: List[str],
    processed_dir: str,
    # Continues the surrounding statement or collection definition.
    task_type: str = "field",
    mode: Optional[str] = None,
    logger=None,
) -> Tuple[str, str]:
    """按预处理成功样本重写有效 train/valid 样本表。"""
    # Checks `logger is None` before entering this conditional branch.
    if logger is None:
        logger = Logger.get_first_configured_logger(["train"], fallback_name=__name__)
    successful_ids = set(_extract_sample_ids_from_processed_paths(processed_paths))
    if not successful_ids:
        # Raises `ValueError` to report why this path cannot continue.
        raise ValueError("预处理后没有任何有效样本，无法继续训练")

    original_train_rows = load_sample_rows(
        train_csv,
        task_type=task_type,
        # Completes the multi-line `load_sample_rows` arguments or elements.
        mode=mode,
    )
    original_valid_rows = load_sample_rows(
        valid_csv,
        # Completes the multi-line `load_sample_rows` arguments or elements.
        task_type=task_type,
        mode=mode,
    )
    original_train_count = len(original_train_rows)
    # Stores the `original_valid_count` for use by the remaining steps in this flow.
    original_valid_count = len(original_valid_rows)

    train_rows = [
        row for row in original_train_rows
        if int(row["order"]) in successful_ids
    # Continues the surrounding statement or collection definition.
    ]
    valid_rows = [
        row for row in original_valid_rows
        if int(row["order"]) in successful_ids
    # Continues the surrounding statement or collection definition.
    ]

    skipped_ids = sorted(
        {
            int(row["order"])
            # Completes the multi-line `sorted` arguments or elements.
            for row in original_train_rows + original_valid_rows
            if int(row["order"]) not in successful_ids
        }
    )
    # Checks `skipped_ids` before entering this conditional branch.
    if skipped_ids:
        logger.warning("以下样本因预处理失败或产物缺失被跳过: %s", skipped_ids)

    if not train_rows and valid_rows:
        logger.warning("预处理后训练集为空，使用验证集中的可用样本回填训练集")
        # Builds the `train_rows` using the comprehension's filter and mapping rules.
        train_rows = [dict(item) for item in valid_rows]
    if not valid_rows and train_rows:
        logger.warning("预处理后验证集为空，使用训练集中的可用样本回填验证集")
        valid_rows = [dict(item) for item in train_rows]

    # Validates `not train_rows and not valid_rows` and raises `ValueError` when the input cannot be processed safely.
    if not train_rows and not valid_rows:
        raise ValueError("预处理后的有效样本无法覆盖训练集和验证集，无法继续训练")

    effective_train_csv = os.path.join(processed_dir, "effective_train.csv")
    effective_valid_csv = os.path.join(processed_dir, "effective_valid.csv")
    # Calls `write_predict_samples_csv` to perform the operation required at this stage.
    write_predict_samples_csv(_rows_to_sample_payload(train_rows), effective_train_csv)
    write_predict_samples_csv(_rows_to_sample_payload(valid_rows), effective_valid_csv)

    logger.info(
        "预处理有效样本过滤完成: train=%s/%s valid=%s/%s",
        # Completes the multi-line `len` arguments or elements.
        len(train_rows),
        original_train_count,
        len(valid_rows),
        original_valid_count,
    # Completes the multi-line `logger.info` arguments or elements.
    )
    return effective_train_csv, effective_valid_csv


def _run_auto_evaluation_from_checkpoint(
    checkpoint_path: str,
    # Continues the surrounding statement or collection definition.
    experiment_dir: str,
    sample_rows: List[Dict],
    task_type: str,
    processed_dir: str,
    # Continues the surrounding statement or collection definition.
    split_name: str = "valid",
) -> Dict:
    """基于 checkpoint 执行完整自动评估，生成真实结果文件路径。"""
    normalized_split_name = str(split_name or "valid").strip().lower()
    evaluate_output_dir = os.path.join(experiment_dir, "evaluate", normalized_split_name)
    # Initializes the `evaluate_config` as a collection for values from this processing stage.
    evaluate_config = {
        "evaluate": {
            "experiment": experiment_dir,
            "output_dir": evaluate_output_dir,
            # Completes the multi-line `collection` arguments or elements.
            "split": normalized_split_name,
            "plot_scatter": False,
        }
    }
    # Stores the `samples_evaluate` for use by the remaining steps in this flow.
    samples_evaluate = _build_auto_evaluate_samples(
        sample_rows=sample_rows,
        task_type=task_type,
        processed_dir=processed_dir,
        # Completes the multi-line `_build_auto_evaluate_samples` arguments or elements.
        subset=normalized_split_name,
    )
    runner = EvaluationRunner(
        checkpoint_path=checkpoint_path,
        # Completes the multi-line `EvaluationRunner` arguments or elements.
        evaluate_config=evaluate_config,
        project_root=PROJECT_ROOT,
        logger=None,
    )
    # Stores the `(evaluate_info, _, samples_evaluated)` for use by the remaining steps in this flow.
    evaluate_info, _, samples_evaluated = runner.run(samples_evaluate=samples_evaluate)
    return _normalize_evaluate_result(
        evaluate_info=evaluate_info,
        samples_evaluated=samples_evaluated,
    # Completes the multi-line `_normalize_evaluate_result` arguments or elements.
    )


def _build_processed_sample_records(
    sample_rows: List[Dict],
    subset: str,
    # Continues the surrounding statement or collection definition.
    task_type: str,
    processed_dir: str,
) -> List[Dict]:
    """补齐训练返回样本的 subset 与 processed_path。"""
    records: List[Dict] = []
    # Stores the `normalized_task_type` for use by the remaining steps in this flow.
    normalized_task_type = str(task_type).lower()
    for item in sample_rows:
        sample_order = int(item["order"])
        if normalized_task_type == "scalar":
            # Stores the `processed_path` for use by the remaining steps in this flow.
            processed_path = os.path.join(processed_dir, subset, f"sample_{sample_order}.pt")
        else:
            processed_path = os.path.join(processed_dir, f"sample_{sample_order}.pt")

        record = dict(item)
        # Stores the `record['subset']` for use by the remaining steps in this flow.
        record["subset"] = subset
        record["processed_path"] = processed_path
        records.append(record)
    return records


# 收口训练结果。
def _build_train_info(
    experiment_dir: str,
    best_model_path: str,
    train_log_path: str,
    # Continues the surrounding statement or collection definition.
    history: Dict,
    auto_evaluate_metrics: Dict,
) -> Dict:
    """收口训练结果。"""
    val_history = list(history.get("val_loss", []))
    # Stores the `best_val_loss` for use by the remaining steps in this flow.
    best_val_loss = min(val_history) if val_history else None
    best_epoch = int(val_history.index(best_val_loss)) + 1 if val_history else None

    return {
        "experiment": experiment_dir,
        # Completes the multi-line `collection` arguments or elements.
        "best_model_path": best_model_path,
        "train_log_path": train_log_path,
        "loss_history": {
            "train_loss": list(history.get("train_loss", [])),
            # Completes the multi-line `collection` arguments or elements.
            "val_loss": val_history,
            "train_units": list(history.get("train_units", [])),
            "val_units": list(history.get("val_units", [])),
            "schedule_unit": str(history.get("schedule_unit", "step")),
            # Completes the multi-line `str` arguments or elements.
            "train_loss_scope": str(history.get("train_loss_scope", "single_step")),
            "val_loss_scope": str(history.get("val_loss_scope", "single_step")),
            "rollout_horizon": int(history.get("rollout_horizon", 1)),
        },
        # Completes the multi-line `collection` arguments or elements.
        "validation_summary": {
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
        },
        # Completes the multi-line `collection` arguments or elements.
        "auto_evaluate_metrics": auto_evaluate_metrics,
    }


def _release_train_runtime_resources(
    trainer: Optional[Trainer],
    # Continues the surrounding statement or collection definition.
    processor,
    reader,
    runtime_config: Optional[Dict],
    original_cpu_affinity_ids: Optional[List[int]] = None,
    # Continues the surrounding statement or collection definition.
    logger=None,
) -> None:
    """尽力释放单次训练链路持有的大对象引用与 CUDA 缓存。"""
    if processor is not None and hasattr(processor, "cached_topology"):
        try:
            # Stores the `cached_topology` for use by the remaining steps in this flow.
            processor.cached_topology = None
        except Exception:
            pass

    if reader is not None and hasattr(reader, "sample_map"):
        # Keeps the operation and its recovery path within the same exception boundary.
        try:
            reader.sample_map = None
        except Exception:
            pass

    # Checks `trainer is not None` before entering this conditional branch.
    if trainer is not None:
        for attr_name in ("model", "criterion", "optimizer", "config", "processor", "teacher_model"):
            if hasattr(trainer, attr_name):
                try:
                    # Calls `setattr` to perform the operation required at this stage.
                    setattr(trainer, attr_name, None)
                except Exception:
                    pass

    gc.collect()
    # Checks `torch.cuda.is_available()` before entering this conditional branch.
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            # Handles the data or control-flow case represented by this statement.
            pass

        ipc_collect = getattr(torch.cuda, "ipc_collect", None)
        if callable(ipc_collect):
            try:
                # Calls `ipc_collect` to perform the operation required at this stage.
                ipc_collect()
            except Exception:
                pass

    if original_cpu_affinity_ids is not None:
        # Keeps the operation and its recovery path within the same exception boundary.
        try:
            restored = restore_cpu_affinity(original_cpu_affinity_ids)
            if logger is not None and restored:
                logger.info("已恢复训练前的 CPU 亲和性: cpu_ids=%s", original_cpu_affinity_ids)
        # Continues the surrounding statement or collection definition.
        except Exception as exc:
            if logger is not None:
                logger.warning("恢复训练前 CPU 亲和性失败: %s", str(exc))

    Logger.close_logger(logger)


# 判断是否启用了多阶段训练。
def _is_multi_stage_training_enabled(config: Dict) -> bool:
    """判断是否启用了多阶段训练。"""
    training_cfg = config.get("training", {})
    multi_stage_cfg = training_cfg.get("multi_stage", {})
    return bool(multi_stage_cfg.get("enabled", False))


# 校验当前版本支持的两阶段残差训练配置。
def _validate_multi_stage_training_config(config: Dict) -> None:
    """校验当前版本支持的两阶段残差训练配置。"""
    if not _is_multi_stage_training_enabled(config):
        return

    task_cfg = config.get("task", {})
    # Reads `training` from the mapping into the `training_cfg` with its declared fallback.
    training_cfg = config.get("training", {})
    multi_stage_cfg = training_cfg.get("multi_stage", {})

    task_type = str(task_cfg.get("type", "field")).strip().lower()
    mode = str(multi_stage_cfg.get("mode", "")).strip().lower()
    # Converts the `stage_count` to `int` so later checks use a consistent type.
    stage_count = int(multi_stage_cfg.get("stage_count", 1))

    if task_type != "field":
        raise ValueError("两阶段残差训练当前仅支持 field 任务。")
    if mode != "residual_additive":
        # Raises `ValueError` to report why this path cannot continue.
        raise ValueError(
            "两阶段残差训练当前仅支持 training.multi_stage.mode=residual_additive。"
        )
    if stage_count != 2:
        # Raises `ValueError` to report why this path cannot continue.
        raise ValueError(
            "两阶段残差训练当前仅支持 training.multi_stage.stage_count=2。"
        )
    resume_cfg = training_cfg.get("resume", {})
    # Validates `bool(resume_cfg.get("enabled", False))` and raises `ValueError` when the input cannot be processed safely.
    if bool(resume_cfg.get("enabled", False)):
        raise ValueError("当前版本暂不支持两阶段残差训练的断点续训。")


def _build_stage_runtime_config(config: Dict, stage_dir: str) -> Dict:
    """为单个阶段构造独立输出配置。"""
    stage_config = copy.deepcopy(config)
    # Stores the `stage_model_cfg` for use by the remaining steps in this flow.
    stage_model_cfg = stage_config.setdefault("model", {})
    stage_training_cfg = stage_config.setdefault("training", {})
    stage_resume_cfg = stage_training_cfg.setdefault("resume", {})
    stage_model_cfg["checkpoint_path"] = os.path.join(stage_dir, "best_model.pt")
    # Stores the `stage_resume_cfg['path']` for use by the remaining steps in this flow.
    stage_resume_cfg["path"] = os.path.join(stage_dir, "last_checkpoint.pt")
    stage_resume_cfg["enabled"] = bool(stage_resume_cfg.get("enabled", False))
    return stage_config


def _build_trainer(
    # Continues the surrounding statement or collection definition.
    trainer_class,
    model,
    criterion,
    optimizer,
    # Continues the surrounding statement or collection definition.
    config: Dict,
    processor,
    logger,
    teacher_model=None,
    # Continues the surrounding statement or collection definition.
    residual_norm_params=None,
):
    """兼容不同 Trainer 签名构造训练器。"""
    trainer_kwargs = {
        "model": model,
        # Completes the multi-line `collection` arguments or elements.
        "criterion": criterion,
        "optimizer": optimizer,
        "config": config,
        "processor": processor,
    # Completes the multi-line `collection` arguments or elements.
    }
    if teacher_model is not None:
        trainer_kwargs["teacher_model"] = teacher_model
    if residual_norm_params is not None:
        # Stores the `trainer_kwargs['residual_norm_params']` for use by the remaining steps in this flow.
        trainer_kwargs["residual_norm_params"] = residual_norm_params

    try:
        return trainer_class(
            **trainer_kwargs,
            # Completes the multi-line `trainer_class` arguments or elements.
            logger=logger,
        )
    except TypeError as exc:
        message = str(exc)
        # Stores the `fallback_kwargs` for use by the remaining steps in this flow.
        fallback_kwargs = dict(trainer_kwargs)
        if "unexpected keyword argument 'residual_norm_params'" in message:
            fallback_kwargs.pop("residual_norm_params", None)
            try:
                # Returns `trainer_class( **fallback_kwargs, logger=logger, )` to the caller for the next stage.
                return trainer_class(
                    **fallback_kwargs,
                    logger=logger,
                )
            # Continues the surrounding statement or collection definition.
            except TypeError as inner_exc:
                if "unexpected keyword argument 'logger'" not in str(inner_exc):
                    raise
                return trainer_class(**fallback_kwargs)

        # Validates `"unexpected keyword argument 'logger'" not in message` and raises `exception` when the input cannot be processed safely.
        if "unexpected keyword argument 'logger'" not in message:
            raise
        return trainer_class(**trainer_kwargs)


def _run_stage_training(
    # Continues the surrounding statement or collection definition.
    runtime_config: Dict,
    resolved_model_type: str,
    task_type: str,
    processor,
    # Continues the surrounding statement or collection definition.
    train_dataset,
    valid_dataset,
    logger,
    progress_callback,
    # Continues the surrounding statement or collection definition.
    terminate_training_checker,
    trainer_class=Trainer,
    teacher_model=None,
    residual_norm_params=None,
    # Continues the surrounding statement or collection definition.
    stage_name: str = "stage",
) -> Dict:
    """执行单个阶段训练。"""
    training_cfg = runtime_config.setdefault("training", {})
    model = ModelFactory.create_model(
        # Completes the multi-line `ModelFactory.create_model` arguments or elements.
        model_type=resolved_model_type,
        task_type=task_type,
        config=runtime_config,
    )
    # Stores the `optimizer` for use by the remaining steps in this flow.
    optimizer = optim.Adam(model.parameters(), lr=float(training_cfg["learning_rate"]))
    criterion, resolved_loss_name = processor.build_criterion(training_cfg["loss_function"])
    logger.info("%s 损失函数已解析: %s", stage_name, resolved_loss_name)

    trainer = _build_trainer(
        # Completes the multi-line `_build_trainer` arguments or elements.
        trainer_class=trainer_class,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        # Completes the multi-line `_build_trainer` arguments or elements.
        config=runtime_config,
        processor=processor,
        logger=logger,
        teacher_model=teacher_model,
        # Completes the multi-line `_build_trainer` arguments or elements.
        residual_norm_params=residual_norm_params,
    )
    trainer.path = os.path.dirname(os.path.abspath(str(runtime_config["model"]["checkpoint_path"])))
    trainer.checkpoint_path = os.path.abspath(str(runtime_config["model"]["checkpoint_path"]))
    # Normalizes the `resume_path` before subsequent file operations.
    trainer.resume_path = os.path.abspath(str(training_cfg["resume"]["path"]))
    os.makedirs(trainer.path, exist_ok=True)

    if bool(training_cfg["resume"].get("enabled", False)) and os.path.exists(trainer.resume_path):
        trainer.load_checkpoint(trainer.resume_path)
        # Records the current execution state to make progress and failures traceable.
        logger.info("%s 已加载续训检查点: %s", stage_name, trainer.resume_path)

    training_terminated = False
    if terminate_training_checker():
        training_terminated = True
        # Stores the `training_terminated` for use by the remaining steps in this flow.
        trainer.training_terminated = True
        logger.info("检测到提前终止训练标记，跳过 %s 并直接返回当前历史", stage_name)
        history = dict(getattr(trainer, "history", {"train_loss": [], "val_loss": []}))
    else:
        # Records the current execution state to make progress and failures traceable.
        logger.info("开始执行 %s", stage_name)
        train_start = time.perf_counter()
        history = trainer.train(
            train_dataset,
            # Completes the multi-line `trainer.train` arguments or elements.
            valid_dataset,
            progress_callback=progress_callback,
            terminate_training_checker=terminate_training_checker,
        )
        # Calls `Logger.log_duration` to perform the operation required at this stage.
        Logger.log_duration(logger, stage_name, time.perf_counter() - train_start)
        training_terminated = bool(getattr(trainer, "training_terminated", False))
        if training_terminated:
            logger.info("%s 检测到终止信号，已提前结束", stage_name)

    # Normalizes the `best_model_path` before subsequent file operations.
    best_model_path = os.path.abspath(str(runtime_config["model"]["checkpoint_path"]))
    checkpoint = None
    if os.path.exists(best_model_path):
        checkpoint = load_torch(best_model_path, weights_only=False)
        # Calls `model.load_state_dict` to perform the operation required at this stage.
        model.load_state_dict(checkpoint["model_state_dict"])

    return {
        "model": model,
        "trainer": trainer,
        # Completes the multi-line `collection` arguments or elements.
        "history": history,
        "best_model_path": best_model_path,
        "checkpoint": checkpoint,
        "training_terminated": training_terminated,
    # Completes the multi-line `collection` arguments or elements.
    }


def _clone_norm_sample_value(value: Any) -> Any:
    """为残差统计构建可安全复用的 CPU 样本副本。"""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()

    # Stores the `cpu_fn` for use by the remaining steps in this flow.
    cpu_fn = getattr(value, "cpu", None)
    if callable(cpu_fn):
        try:
            return cpu_fn()
        # Continues the surrounding statement or collection definition.
        except Exception:
            pass

    return copy.deepcopy(value)


def _build_residual_norm_sample(inputs: Any, residual_targets: Any) -> Dict[str, Any]:
    """将单样本残差整理为处理器可复用的 pseudo sample。"""
    # Stores the `sample_inputs` for use by the remaining steps in this flow.
    sample_inputs = _clone_norm_sample_value(inputs)
    sample_targets = _clone_norm_sample_value(residual_targets)

    if (
        isinstance(sample_inputs, torch.Tensor)
        # Continues the surrounding statement or collection definition.
        and sample_inputs.ndim > 0
        and int(sample_inputs.shape[0]) == 1
    ):
        sample_inputs = sample_inputs.squeeze(0)
        # Checks `` before entering this conditional branch.
        if (
            isinstance(sample_targets, torch.Tensor)
            and sample_targets.ndim > 0
            and int(sample_targets.shape[0]) == 1
        # Continues the surrounding statement or collection definition.
        ):
            sample_targets = sample_targets.squeeze(0)

    return {
        "x": sample_inputs,
        # Completes the multi-line `collection` arguments or elements.
        "y": sample_targets,
        "meta": {},
    }


def _compute_stage2_residual_norm_params(
    # Continues the surrounding statement or collection definition.
    processor,
    train_dataset,
    teacher_model,
    runtime_config: Dict,
    # Continues the surrounding statement or collection definition.
    logger,
) -> Optional[Dict[str, Any]]:
    """基于第一阶段 teacher 输出统计第二阶段残差归一化参数。"""
    if processor is None or not hasattr(processor, "compute_norm_params"):
        logger.warning("当前处理器缺少 compute_norm_params，跳过第二阶段残差归一化统计。")
        # Returns `None` to the caller for the next stage.
        return None

    if train_dataset is None or len(train_dataset) == 0:
        logger.warning("训练数据集为空，跳过第二阶段残差归一化统计。")
        return None

    # Stores the `loader` for use by the remaining steps in this flow.
    loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    device = next(teacher_model.parameters()).device
    norm_type = str(runtime_config.get("processor", {}).get("norm_type", "standardization"))
    residual_samples: List[Dict[str, Any]] = []

    # Calls `teacher_model.eval` to perform the operation required at this stage.
    teacher_model.eval()
    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, (list, tuple)) or len(batch) != 2:
                # Records the current execution state to make progress and failures traceable.
                logger.warning("训练数据集样本格式不是 (inputs, targets)，跳过第二阶段残差归一化统计。")
                return None

            inputs, targets = batch
            if not isinstance(targets, torch.Tensor):
                # Records the current execution state to make progress and failures traceable.
                logger.warning("训练目标不是张量，跳过第二阶段残差归一化统计。")
                return None

            device_inputs = inputs.to(device) if hasattr(inputs, "to") else inputs
            device_targets = targets.to(device)
            # Stores the `teacher_outputs` for use by the remaining steps in this flow.
            teacher_outputs = teacher_model(device_inputs)
            residual_targets = device_targets - teacher_outputs.detach()
            residual_samples.append(
                _build_residual_norm_sample(
                    # Completes the multi-line `_build_residual_norm_sample` arguments or elements.
                    inputs=inputs,
                    residual_targets=residual_targets,
                )
            )

    # Checks `not residual_samples` before entering this conditional branch.
    if not residual_samples:
        logger.warning("未收集到任何残差样本，跳过第二阶段残差归一化统计。")
        return None

    processed_dir = str(runtime_config.get("general", {}).get("train_processed_dir", "") or "").strip()
    # Checks `` before entering this conditional branch.
    if restore_cached_processor_state(
        processor=processor,
        processed_dir=processed_dir,
        processor_type=type(processor).__name__,
    # Completes the multi-line `restore_cached_processor_state` arguments or elements.
    ):
        logger.info("第二阶段残差归一化前已恢复处理器运行态: %s", os.path.join(processed_dir, "modes.pt"))

    residual_norm_params = processor.compute_norm_params(
        train_samples=residual_samples,
        # Completes the multi-line `processor.compute_norm_params` arguments or elements.
        norm_type=norm_type,
        save_path=None,
    )
    if not isinstance(residual_norm_params, dict):
        # Records the current execution state to make progress and failures traceable.
        logger.warning("第二阶段残差归一化统计结果格式异常，跳过保存。")
        return None

    residual_norm_params["target_semantics"] = "stage2_residual_in_stage1_normalized_space"
    residual_norm_params["norm_scope"] = "stage2_residual"
    # Stores the `residual_norm_params['source_norm_type']` for use by the remaining steps in this flow.
    residual_norm_params["source_norm_type"] = norm_type
    return residual_norm_params


def _export_residual_additive_checkpoint(
    runtime_config: Dict,
    # Continues the surrounding statement or collection definition.
    stage1_checkpoint_path: str,
    stage2_checkpoint_path: str,
    final_checkpoint_path: str,
    stage2_residual_norm_params: Optional[Dict[str, Any]] = None,
    # Continues the surrounding statement or collection definition.
    extra_info: Optional[Dict] = None,
) -> Dict:
    """导出两阶段残差加和的最终组合 checkpoint。"""
    stage1_checkpoint = load_torch(stage1_checkpoint_path, map_location="cpu", weights_only=False)
    stage2_checkpoint = load_torch(stage2_checkpoint_path, map_location="cpu", weights_only=False)

    # Selects the required item from `runtime_config["task"]` as the `task_type`.
    task_type = runtime_config["task"]["type"]
    model_type = runtime_config["model"]["type"]
    resolved_model_type = ModelFactory.get_model(model_type, task_type, runtime_config)

    stage1_model = ModelFactory.create_model(
        # Completes the multi-line `ModelFactory.create_model` arguments or elements.
        model_type=resolved_model_type,
        task_type=task_type,
        config=runtime_config,
    )
    # Stores the `stage2_model` for use by the remaining steps in this flow.
    stage2_model = ModelFactory.create_model(
        model_type=resolved_model_type,
        task_type=task_type,
        config=runtime_config,
    # Completes the multi-line `ModelFactory.create_model` arguments or elements.
    )
    stage1_model.load_state_dict(stage1_checkpoint["model_state_dict"])
    stage2_model.load_state_dict(stage2_checkpoint["model_state_dict"])

    from tfai.models.residual_additive import ResidualAdditiveModel

    # Stores the `combined_model` for use by the remaining steps in this flow.
    combined_model = ResidualAdditiveModel(stage1_model, stage2_model)
    final_config = copy.deepcopy(runtime_config)
    final_config.setdefault("model", {})["checkpoint_path"] = final_checkpoint_path
    final_config.setdefault("training", {}).setdefault("resume", {})["path"] = os.path.join(
        # Completes the multi-line `os.path.dirname` arguments or elements.
        os.path.dirname(final_checkpoint_path),
        "last_checkpoint.pt",
    )

    payload = copy.deepcopy(stage2_checkpoint)
    # Adds the current value to `payload` for later aggregation.
    payload.update(
        {
            "model_bundle_type": RESIDUAL_ADDITIVE_TWO_STAGE_BUNDLE,
            "model_state_dict": combined_model.state_dict(),
            # Completes the multi-line `copy.deepcopy` arguments or elements.
            "stage1_model_state_dict": copy.deepcopy(stage1_checkpoint["model_state_dict"]),
            "stage2_model_state_dict": copy.deepcopy(stage2_checkpoint["model_state_dict"]),
            "config": sanitize_config_for_checkpoint(final_config),
            "stage2_residual_norm_params": copy.deepcopy(stage2_residual_norm_params),
            # Completes the multi-line `collection` arguments or elements.
            "multi_stage_info": {
                "enabled": True,
                "mode": "residual_additive",
                "stage_count": 2,
                # Completes the multi-line `collection` arguments or elements.
                "final_output": "stage1_plus_stage2",
                "stage2_target_space": "residual_in_stage1_normalized_space",
                "stage2_residual_norm_enabled": bool(stage2_residual_norm_params),
                "stage_1_best_checkpoint": os.path.abspath(stage1_checkpoint_path),
                # Completes the multi-line `os.path.abspath` arguments or elements.
                "stage_2_best_checkpoint": os.path.abspath(stage2_checkpoint_path),
            },
        }
    )
    # Checks `extra_info is not None` before entering this conditional branch.
    if extra_info is not None:
        payload["extra_info"] = copy.deepcopy(extra_info)

    atomic_torch_save(payload, final_checkpoint_path)
    return payload


# 写出两阶段训练摘要。
def _write_multi_stage_summary(
    train_dir: str,
    stage1_result: Dict,
    stage2_result: Dict,
    # Continues the surrounding statement or collection definition.
    final_checkpoint_path: str,
    stage2_residual_norm_path: Optional[str] = None,
) -> str:
    """写出两阶段训练摘要。"""
    summary_path = os.path.join(train_dir, "multi_stage_summary.json")
    # Initializes the `payload` as a collection for values from this processing stage.
    payload = {
        "enabled": True,
        "mode": "residual_additive",
        "stage_count": 2,
        # Completes the multi-line `collection` arguments or elements.
        "current_stage": 2,
        "completed_stages": [1, 2],
        "stage_1_best_checkpoint": stage1_result["best_model_path"],
        "stage_2_best_checkpoint": stage2_result["best_model_path"],
        # Completes the multi-line `os.path.abspath` arguments or elements.
        "final_checkpoint_path": os.path.abspath(final_checkpoint_path),
        "stage_2_residual_norm_path": (
            os.path.abspath(stage2_residual_norm_path)
            if stage2_residual_norm_path
            # Completes the multi-line `collection` arguments or elements.
            else None
        ),
        "stage_1_history": copy.deepcopy(stage1_result.get("history") or {}),
        "stage_2_history": copy.deepcopy(stage2_result.get("history") or {}),
    # Completes the multi-line `collection` arguments or elements.
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return summary_path


# 执行训练并返回训练信息与处理后样本。
def _run_train_impl(
    config: Dict,
    samples_train: List[Dict],
    experiment_dir: str,
    # Continues the surrounding statement or collection definition.
    model_type: str,
    train_params: Dict,
    progress_callback: Optional[Callable[[Dict], None]] = None,
    terminate_training_checker: Optional[Callable[[], bool]] = None,
    # Continues the surrounding statement or collection definition.
    extra_info: Optional[Dict] = None,
    cleanup_state: Optional[Dict] = None,
) -> Tuple[Dict, List[Dict]]:
    """执行训练并返回训练信息与处理后样本。"""
    if extra_info is not None and not isinstance(extra_info, dict):
        # Raises `TypeError` to report why this path cannot continue.
        raise TypeError("extra_info 必须为 dict 或 None")

    cleanup_state = cleanup_state if isinstance(cleanup_state, dict) else {}
    checkpoint_extra_info = copy.deepcopy(extra_info) if extra_info is not None else None
    runtime_config = copy.deepcopy(config)
    # Stores the `normalized_experiment_dir` for use by the remaining steps in this flow.
    normalized_experiment_dir = str(experiment_dir or "").strip()
    if not normalized_experiment_dir:
        raise ValueError("experiment_dir 不能为空")
    runtime_config = apply_train_experiment_dir(
        # Completes the multi-line `apply_train_experiment_dir` arguments or elements.
        runtime_config,
        experiment_dir=normalized_experiment_dir,
    )
    runtime_config = clear_train_default_output_paths(runtime_config)
    # Stores the `runtime_config` for use by the remaining steps in this flow.
    runtime_config = resolve_runtime_paths(runtime_config, project_root=PROJECT_ROOT)
    runtime_config = ensure_train_runtime_paths(runtime_config)
    runtime_config = _apply_train_params(runtime_config, train_params)
    runtime_config = normalize_scalar_task_variable_aliases(runtime_config)
    # Stores the `runtime_config` for use by the remaining steps in this flow.
    runtime_config = ensure_train_runtime_paths(runtime_config)
    cleanup_state["runtime_config"] = runtime_config
    terminate_training_checker = _build_terminate_training_checker(terminate_training_checker)

    general_cfg = runtime_config.setdefault("general", {})
    # Stores the task settings for use by the remaining steps in this flow.
    task_cfg = runtime_config.setdefault("task", {})
    training_cfg = runtime_config.setdefault("training", {})
    model_cfg = runtime_config.setdefault("model", {})
    processor_cfg = runtime_config.setdefault("processor", {})
    # Stores the `resume_cfg` for use by the remaining steps in this flow.
    resume_cfg = training_cfg.setdefault("resume", {})
    training_cfg.setdefault("max_steps", None)
    training_cfg.setdefault("validate_every", None)
    training_cfg.setdefault("validation_max_batches", None)
    # Calls `training_cfg.setdefault` to perform the operation required at this stage.
    training_cfg.setdefault("auto_evaluate_after_train", True)
    training_cfg.setdefault(
        "temporal_rollout",
        {
            # Completes the multi-line `collection` arguments or elements.
            "horizon": 1,
            "horizon_weighting": "uniform",
            "horizon_weight_gamma": 1.0,
            "training_noise": False,
            # Completes the multi-line `collection` arguments or elements.
            "training_noise_alpha": 0.01,
            "boundary_state_override": False,
        },
    )
    # Checks `isinstance(training_cfg.get("temporal_rollout"), dict)` before entering this conditional branch.
    if isinstance(training_cfg.get("temporal_rollout"), dict):
        training_cfg["temporal_rollout"].setdefault("training_noise_alpha", 0.01)

    normalized_model_type = str(model_type or "").strip()
    if not normalized_model_type:
        # Raises `ValueError` to report why this path cannot continue.
        raise ValueError("model_type 不能为空")
    model_cfg["type"] = normalized_model_type
    validate_matrix_pod_fixed_geometry_only(runtime_config, stage="training")

    seed = int(general_cfg.get("seed", 42))
    # Converts the `deterministic` to `bool` so later checks use a consistent type.
    deterministic = bool(general_cfg.get("deterministic", False))
    set_global_seed(seed, deterministic=deterministic)

    experiment_dir = os.path.abspath(str(general_cfg["experiment"]))
    train_dir = os.path.join(experiment_dir, "train")
    # Normalizes the `processed_dir` before subsequent file operations.
    processed_dir = os.path.abspath(str(general_cfg["train_processed_dir"]))
    train_log_path = os.path.join(train_dir, "train.log")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # Stores the run logger for use by the remaining steps in this flow.
    logger = Logger.setup_logger("tfai_package_train", train_log_path)
    cleanup_state["logger"] = logger
    logger.info("开始执行包接口训练流程")
    logger.info("实验目录: %s", experiment_dir)
    # Records the current execution state to make progress and failures traceable.
    logger.info("随机种子: seed=%s deterministic=%s", seed, deterministic)
    _validate_multi_stage_training_config(runtime_config)
    if _is_multi_stage_training_enabled(runtime_config):
        logger.info("已启用两阶段残差训练模式，将按 stage_1 -> stage_2 的顺序执行。")

    # Stores the `original_cpu_affinity_ids` for use by the remaining steps in this flow.
    original_cpu_affinity_ids = get_current_cpu_affinity_ids()
    cleanup_state["original_cpu_affinity_ids"] = list(original_cpu_affinity_ids)
    cpu_affinity_info = apply_cpu_affinity_limit(general_cfg.get("cpu_affinity_ratio", 0.5))
    if cpu_affinity_info is None:
        # Records the current execution state to make progress and failures traceable.
        logger.info("未启用 CPU 亲和性限制: general.cpu_affinity_ratio=None")
    else:
        logger.info(
            "CPU 亲和性限制已应用: ratio=%s available_cpu_count=%s limited_cpu_count=%s cpu_ids=%s applied=%s",
            # Completes the multi-line `logger.info` arguments or elements.
            cpu_affinity_info["ratio"],
            cpu_affinity_info["available_cpu_count"],
            cpu_affinity_info["limited_cpu_count"],
            cpu_affinity_info["cpu_ids"],
            # Completes the multi-line `logger.info` arguments or elements.
            cpu_affinity_info["applied"],
        )

    task_type = str(task_cfg.get("type", "field")).lower()
    normalized_samples = _normalize_train_samples(samples_train, task_type=task_type)
    # Calls `_persist_prediction_variable_metadata` to perform the operation required at this stage.
    _persist_prediction_variable_metadata(runtime_config, normalized_samples)
    samples_csv = _materialize_train_samples_csv(runtime_config, normalized_samples)
    logger.info("训练样本表已生成: %s", samples_csv)

    valid_ratio = float(training_cfg.get("valid_ratio", 0.1))
    # Records the current execution state to make progress and failures traceable.
    logger.info("开始划分训练集与验证集: valid_ratio=%s", valid_ratio)
    train_csv, valid_csv = Splitter.split_train_valid(
        samples_csv=samples_csv,
        output_dir=processed_dir,
        # Completes the multi-line `Splitter.split_train_valid` arguments or elements.
        valid_size=valid_ratio,
        seed=seed,
    )

    reader = ReaderFactory.get_reader(
        # Completes the multi-line `ReaderFactory.get_reader` arguments or elements.
        config=runtime_config,
        flag="train",
        sample_table_path_override=samples_csv,
    )
    # Stores the `cleanup_state['reader']` for use by the remaining steps in this flow.
    cleanup_state["reader"] = reader
    sample_map = SampleMap(samples_csv)
    resolve_training_variables(
        config=runtime_config,
        # Completes the multi-line `resolve_training_variables` arguments or elements.
        reader=reader,
        sample_map=sample_map,
        samples_csv=samples_csv,
        processed_dir=processed_dir,
        # Completes the multi-line `resolve_training_variables` arguments or elements.
        logger=logger,
    )

    resolved_model_type = ModelFactory.get_model(model_cfg["type"], task_type, runtime_config)
    processor = ProcessorFactory.get_processor(
        # Completes the multi-line `ProcessorFactory.get_processor` arguments or elements.
        task_type,
        resolved_model_type,
        runtime_config,
    )
    # Stores the `cleanup_state['processor']` for use by the remaining steps in this flow.
    cleanup_state["processor"] = processor
    processor_type = type(processor).__name__
    processor_cfg["type"] = processor_type

    _cleanup_incompatible_processed_artifacts(
        # Completes the multi-line `_cleanup_incompatible_processed_artifacts` arguments or elements.
        processed_dir=processed_dir,
        processor=processor,
        logger=logger,
    )

    # Calls `_copy_reference_file_if_needed` to perform the operation required at this stage.
    _copy_reference_file_if_needed(
        runtime_config,
        samples_csv,
        train_dir,
        # Completes the multi-line `_copy_reference_file_if_needed` arguments or elements.
        logger,
        processor=processor,
        reader=reader,
    )
    # Calls `_package_default_source_if_needed` to perform the operation required at this stage.
    _package_default_source_if_needed(
        runtime_config,
        samples_csv,
        train_dir,
        # Completes the multi-line `_package_default_source_if_needed` arguments or elements.
        logger,
    )

    cache_path = os.path.join(processed_dir, "preprocess_config_cache.pt")
    can_skip_preprocess, cache_status, cache_path, cached_overrides = evaluate_preprocess_cache_hit(
        # Completes the multi-line `evaluate_preprocess_cache_hit` arguments or elements.
        config=runtime_config,
        processed_dir=processed_dir,
        processor_type=processor_type,
        cache_path=cache_path,
    # Completes the multi-line `evaluate_preprocess_cache_hit` arguments or elements.
    )

    if training_cfg.get("require_preprocess_cache", False) and not can_skip_preprocess:
        raise RuntimeError(f"Required shared preprocessing cache unavailable: {cache_status}")

    if can_skip_preprocess:
        logger.info("%s，跳过预处理和模型参数推断: %s", cache_status, cache_path)
        apply_cached_overrides(runtime_config, cached_overrides, processor_type)
        # Checks `` before entering this conditional branch.
        if restore_cached_processor_state(processor, processed_dir, processor_type):
            logger.info("已从预处理缓存恢复处理器运行态: %s", processed_dir)
        cleanup_resume_state_if_compatible(
            processed_dir=processed_dir,
            # Completes the multi-line `build_preprocess_signature` arguments or elements.
            signature=build_preprocess_signature(runtime_config, processor_type),
            processor_type=processor_type,
            workflow="matrix" if processor_type == "MatrixProcessor" else "sample",
        )
        # Checks `task_type == "scalar"` before entering this conditional branch.
        if task_type == "scalar":
            scalar_train_dir = os.path.join(processed_dir, "train")
            scalar_valid_dir = os.path.join(processed_dir, "valid")
            train_processed_paths_for_norm = sorted(
                # Completes the multi-line `str` arguments or elements.
                str(path) for path in Path(scalar_train_dir).glob("sample_*.pt")
            )
            valid_processed_paths = sorted(
                str(path) for path in Path(scalar_valid_dir).glob("sample_*.pt")
            # Completes the multi-line `sorted` arguments or elements.
            )
            processed_paths = train_processed_paths_for_norm + valid_processed_paths
        else:
            train_processed_paths_for_norm = None
            # Stores the `cached_sample_map` for use by the remaining steps in this flow.
            cached_sample_map = SampleMap(samples_csv)
            processed_paths = [
                os.path.join(processed_dir, f"sample_{sample_id}.pt")
                for sample_id in cached_sample_map.get_all_samples()
                # Completes the multi-line `cached_sample_map.get_file_path` arguments or elements.
                if cached_sample_map.get_file_path(sample_id)
            ]
    else:
        logger.info(cache_status)
        # Stores the `workflow` for use by the remaining steps in this flow.
        workflow = "matrix" if processor_type == "MatrixProcessor" else "sample"
        resume_manager, resume_status = PreprocessResumeManager.prepare(
            processed_dir=processed_dir,
            signature=build_preprocess_signature(runtime_config, processor_type),
            # Completes the multi-line `PreprocessResumeManager.prepare` arguments or elements.
            processor_type=processor_type,
            workflow=workflow,
        )
        logger.info(resume_status)
        # Checks `processor_type == "MatrixProcessor"` before entering this conditional branch.
        if processor_type == "MatrixProcessor":
            if restore_matrix_resume_modes(resume_manager, processor):
                logger.info("已恢复中断前的 POD 暂存模态: %s", processed_dir)
        elif restore_resume_processor_state(resume_manager, processor):
            # Records the current execution state to make progress and failures traceable.
            logger.info("已恢复中断前的处理器运行态: %s", processed_dir)

        logger.info("开始样本预处理")
        if task_type == "scalar":
            scalar_train_dir = os.path.join(processed_dir, "train")
            # Stores the `scalar_valid_dir` for use by the remaining steps in this flow.
            scalar_valid_dir = os.path.join(processed_dir, "valid")
            train_processed_paths_for_norm = DataPipeline.call_process_with_reader(
                reader=reader,
                processor=processor,
                # Completes the multi-line `DataPipeline.call_process_with_reader` arguments or elements.
                sample_table_path=train_csv,
                processed_dir=scalar_train_dir,
                logger=logger,
                resume_manager=resume_manager,
            # Completes the multi-line `DataPipeline.call_process_with_reader` arguments or elements.
            )
            valid_processed_paths = DataPipeline.call_process_with_reader(
                reader=reader,
                processor=processor,
                # Completes the multi-line `DataPipeline.call_process_with_reader` arguments or elements.
                sample_table_path=valid_csv,
                processed_dir=scalar_valid_dir,
                logger=logger,
                resume_manager=resume_manager,
            # Completes the multi-line `DataPipeline.call_process_with_reader` arguments or elements.
            )
            processed_paths = train_processed_paths_for_norm + valid_processed_paths
        else:
            train_processed_paths_for_norm = None
            # Stores the `processed_paths` for use by the remaining steps in this flow.
            processed_paths = DataPipeline.call_process_with_reader(
                reader=reader,
                processor=processor,
                sample_table_path=samples_csv,
                # Completes the multi-line `DataPipeline.call_process_with_reader` arguments or elements.
                processed_dir=processed_dir,
                logger=logger,
                resume_manager=resume_manager,
            )

        # Records the current execution state to make progress and failures traceable.
        logger.info("预处理完成，样本数量: %s", len(processed_paths))
        norm_save_path = os.path.join(processed_dir, "norm_params.pt")
        DataPipeline.normalize_data(
            train_csv=train_csv,
            # Completes the multi-line `DataPipeline.normalize_data` arguments or elements.
            processed_paths=processed_paths,
            norm_save_path=norm_save_path,
            norm_type=processor_cfg["norm_type"],
            processor=processor,
            # Completes the multi-line `DataPipeline.normalize_data` arguments or elements.
            processor_type=processor_type,
            train_processed_paths=train_processed_paths_for_norm,
            resume_manager=resume_manager,
        )
        # Records the current execution state to make progress and failures traceable.
        logger.info("归一化参数已保存: %s", norm_save_path)
        if processor_type == "MatrixProcessor":
            resume_manager.commit_matrix_modes(processor)
            logger.info("POD 模态已提交: %s", os.path.join(processed_dir, "modes.pt"))

        # Checks `processed_paths` before entering this conditional branch.
        if processed_paths:
            model_probe_path = (
                train_processed_paths_for_norm[0]
                if train_processed_paths_for_norm
                # Continues the surrounding statement or collection definition.
                else processed_paths[0]
            )
            sample_data = load_torch(model_probe_path, weights_only=False)
            infer_model_overrides = getattr(processor, "infer_model_overrides", None)
            # Checks `callable(infer_model_overrides)` before entering this conditional branch.
            if callable(infer_model_overrides):
                inferred_model_cfg = infer_model_overrides(sample_data, config=None)
                if inferred_model_cfg:
                    runtime_config["model"].update(inferred_model_cfg)
                    # Records the current execution state to make progress and failures traceable.
                    logger.info("已根据处理后样本推断模型参数: %s", inferred_model_cfg)

        save_preprocess_cache(cache_path, runtime_config, processor_type)
        logger.info("预处理缓存已更新: %s", cache_path)
        if resume_manager is not None:
            # Calls `resume_manager.finalize_after_cache` to perform the operation required at this stage.
            resume_manager.finalize_after_cache()

    effective_train_csv = train_csv
    effective_valid_csv = valid_csv
    if task_type != "scalar" and processed_paths:
        # Stores the `(effective_train_csv, effective_valid_csv)` for use by the remaining steps in this flow.
        effective_train_csv, effective_valid_csv = _build_effective_train_valid_csvs(
            train_csv=train_csv,
            valid_csv=valid_csv,
            processed_paths=processed_paths,
            # Completes the multi-line `_build_effective_train_valid_csvs` arguments or elements.
            processed_dir=processed_dir,
            task_type=task_type,
            mode=task_cfg.get("mode"),
            logger=logger,
        # Completes the multi-line `_build_effective_train_valid_csvs` arguments or elements.
        )

    train_dataset, valid_dataset = build_train_valid_datasets(
        config=runtime_config,
        processor_type=processor_type,
        # Completes the multi-line `build_train_valid_datasets` arguments or elements.
        train_csv=effective_train_csv,
        valid_csv=effective_valid_csv,
        processed_dir=processed_dir,
        processor=processor,
    # Completes the multi-line `build_train_valid_datasets` arguments or elements.
    )
    logger.info("数据集已就绪: train=%s valid=%s", len(train_dataset), len(valid_dataset))

    multi_stage_enabled = _is_multi_stage_training_enabled(runtime_config)
    if multi_stage_enabled:
        # Records the current execution state to make progress and failures traceable.
        logger.info("开始执行两阶段残差训练编排")
        stage_root_dir = os.path.join(train_dir, "stages")
        stage1_dir = os.path.join(stage_root_dir, "stage_1")
        stage2_dir = os.path.join(stage_root_dir, "stage_2")
        # Stores the `stage1_config` for use by the remaining steps in this flow.
        stage1_config = _build_stage_runtime_config(runtime_config, stage1_dir)
        stage2_config = _build_stage_runtime_config(runtime_config, stage2_dir)

        stage1_result = _run_stage_training(
            runtime_config=stage1_config,
            # Completes the multi-line `_run_stage_training` arguments or elements.
            resolved_model_type=resolved_model_type,
            task_type=task_type,
            processor=processor,
            train_dataset=train_dataset,
            # Completes the multi-line `_run_stage_training` arguments or elements.
            valid_dataset=valid_dataset,
            logger=logger,
            # 双阶段只推送第二阶段进度，避免不含 stage 字段的旧回调在阶段切换时重置周期编号。
            progress_callback=None,
            terminate_training_checker=terminate_training_checker,
            # Completes the multi-line `_run_stage_training` arguments or elements.
            trainer_class=Trainer,
            teacher_model=None,
            stage_name="stage_1",
        )
        # Checks `stage1_result.get("training_terminated", False)` before entering this conditional branch.
        if stage1_result.get("training_terminated", False):
            logger.info(
                "第一阶段训练已被外部终止，跳过第二阶段残差训练、组合模型导出与训练结束自动评估"
            )
            # Stores the `stage1_history` for use by the remaining steps in this flow.
            stage1_history = dict(stage1_result.get("history") or {})
            history = {
                "train_loss": list(stage1_history.get("train_loss", [])),
                "val_loss": list(stage1_history.get("val_loss", [])),
                # Completes the multi-line `list` arguments or elements.
                "train_units": list(stage1_history.get("train_units", [])),
                "val_units": list(stage1_history.get("val_units", [])),
                "schedule_unit": str(stage1_history.get("schedule_unit", "step")),
                "stage_1": copy.deepcopy(stage1_history),
                # Completes the multi-line `collection` arguments or elements.
                "stage_2": {"train_loss": [], "val_loss": []},
            }
            cleanup_state["trainer"] = stage1_result["trainer"]

            train_rows = load_sample_rows(
                # Completes the multi-line `load_sample_rows` arguments or elements.
                effective_train_csv,
                task_type=task_type,
                mode=task_cfg.get("mode"),
            )
            # Stores the `valid_rows` for use by the remaining steps in this flow.
            valid_rows = load_sample_rows(
                effective_valid_csv,
                task_type=task_type,
                mode=task_cfg.get("mode"),
            # Completes the multi-line `load_sample_rows` arguments or elements.
            )
            samples_train_processed = _build_processed_sample_records(
                sample_rows=train_rows,
                subset="train",
                # Completes the multi-line `_build_processed_sample_records` arguments or elements.
                task_type=task_type,
                processed_dir=processed_dir,
            ) + _build_processed_sample_records(
                sample_rows=valid_rows,
                # Completes the multi-line `_build_processed_sample_records` arguments or elements.
                subset="valid",
                task_type=task_type,
                processed_dir=processed_dir,
            )
            # Stores the `train_info` for use by the remaining steps in this flow.
            train_info = _build_train_info(
                experiment_dir=experiment_dir,
                best_model_path=stage1_result["best_model_path"],
                train_log_path=train_log_path,
                # Completes the multi-line `_build_train_info` arguments or elements.
                history=history,
                auto_evaluate_metrics={},
            )
            return train_info, samples_train_processed

        # Validates `not os.path.exists(stage1_result["best_model_path"])` and raises `FileNotFoundError` when the input cannot be processed safely.
        if not os.path.exists(stage1_result["best_model_path"]):
            raise FileNotFoundError("第一阶段训练未生成 best_model.pt，无法继续第二阶段残差训练。")

        stage2_residual_norm_params = _compute_stage2_residual_norm_params(
            processor=processor,
            # Completes the multi-line `_compute_stage2_residual_norm_params` arguments or elements.
            train_dataset=train_dataset,
            teacher_model=stage1_result["model"],
            runtime_config=runtime_config,
            logger=logger,
        # Completes the multi-line `_compute_stage2_residual_norm_params` arguments or elements.
        )
        stage2_residual_norm_path = None
        if stage2_residual_norm_params:
            stage2_residual_norm_path = os.path.join(stage2_dir, "residual_norm_params.pt")
            # Calls `os.makedirs` to perform the operation required at this stage.
            os.makedirs(os.path.dirname(stage2_residual_norm_path), exist_ok=True)
            torch.save(stage2_residual_norm_params, stage2_residual_norm_path)
            logger.info("第二阶段残差归一化参数已保存: %s", stage2_residual_norm_path)
        else:
            # Records the current execution state to make progress and failures traceable.
            logger.warning("第二阶段残差归一化参数未生成，当前仍沿用现有残差训练语义。")

        stage2_result = _run_stage_training(
            runtime_config=stage2_config,
            resolved_model_type=resolved_model_type,
            # Completes the multi-line `_run_stage_training` arguments or elements.
            task_type=task_type,
            processor=processor,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            # Completes the multi-line `_run_stage_training` arguments or elements.
            logger=logger,
            progress_callback=progress_callback,
            terminate_training_checker=terminate_training_checker,
            trainer_class=ResidualStageTrainer,
            # Completes the multi-line `_run_stage_training` arguments or elements.
            teacher_model=stage1_result["model"],
            residual_norm_params=stage2_residual_norm_params,
            stage_name="stage_2",
        )
        # Checks `stage2_result.get("training_terminated", False)` before entering this conditional branch.
        if stage2_result.get("training_terminated", False):
            logger.info(
                "第二阶段训练已被外部终止，跳过组合模型导出与训练结束自动评估，保留第一阶段检查点"
            )
            # Stores the `stage1_history` for use by the remaining steps in this flow.
            stage1_history = dict(stage1_result.get("history") or {})
            stage2_history = dict(stage2_result.get("history") or {})
            history = {
                # 当前返回的 best_model_path 是第一阶段模型，主摘要必须与其对应。
                "train_loss": list(stage1_history.get("train_loss", [])),
                # Completes the multi-line `list` arguments or elements.
                "val_loss": list(stage1_history.get("val_loss", [])),
                "train_units": list(stage1_history.get("train_units", [])),
                "val_units": list(stage1_history.get("val_units", [])),
                "schedule_unit": str(stage1_history.get("schedule_unit", "step")),
                # Completes the multi-line `copy.deepcopy` arguments or elements.
                "stage_1": copy.deepcopy(stage1_history),
                "stage_2": copy.deepcopy(stage2_history),
            }
            cleanup_state["trainer"] = stage2_result["trainer"]

            # Stores the `train_rows` for use by the remaining steps in this flow.
            train_rows = load_sample_rows(
                effective_train_csv,
                task_type=task_type,
                mode=task_cfg.get("mode"),
            # Completes the multi-line `load_sample_rows` arguments or elements.
            )
            valid_rows = load_sample_rows(
                effective_valid_csv,
                task_type=task_type,
                # Completes the multi-line `task_cfg.get` arguments or elements.
                mode=task_cfg.get("mode"),
            )
            samples_train_processed = _build_processed_sample_records(
                sample_rows=train_rows,
                # Completes the multi-line `_build_processed_sample_records` arguments or elements.
                subset="train",
                task_type=task_type,
                processed_dir=processed_dir,
            ) + _build_processed_sample_records(
                # Completes the multi-line `_build_processed_sample_records` arguments or elements.
                sample_rows=valid_rows,
                subset="valid",
                task_type=task_type,
                processed_dir=processed_dir,
            # Completes the multi-line `_build_processed_sample_records` arguments or elements.
            )
            train_info = _build_train_info(
                experiment_dir=experiment_dir,
                best_model_path=stage1_result["best_model_path"],
                # Completes the multi-line `_build_train_info` arguments or elements.
                train_log_path=train_log_path,
                history=history,
                auto_evaluate_metrics={},
            )
            # Returns `train_info, samples_train_processed` to the caller for the next stage.
            return train_info, samples_train_processed

        if not os.path.exists(stage2_result["best_model_path"]):
            raise FileNotFoundError("第二阶段训练未生成 best_model.pt，无法导出组合模型。")

        best_model_path = os.path.abspath(str(model_cfg["checkpoint_path"]))
        # Stores the `checkpoint` for use by the remaining steps in this flow.
        checkpoint = _export_residual_additive_checkpoint(
            runtime_config=runtime_config,
            stage1_checkpoint_path=stage1_result["best_model_path"],
            stage2_checkpoint_path=stage2_result["best_model_path"],
            # Completes the multi-line `_export_residual_additive_checkpoint` arguments or elements.
            final_checkpoint_path=best_model_path,
            stage2_residual_norm_params=stage2_residual_norm_params,
            extra_info=checkpoint_extra_info,
        )
        # Stores the `summary_path` for use by the remaining steps in this flow.
        summary_path = _write_multi_stage_summary(
            train_dir=train_dir,
            stage1_result=stage1_result,
            stage2_result=stage2_result,
            # Completes the multi-line `_write_multi_stage_summary` arguments or elements.
            final_checkpoint_path=best_model_path,
            stage2_residual_norm_path=stage2_residual_norm_path,
        )
        logger.info("两阶段训练摘要已写出: %s", summary_path)

        # Normalizes the `stage2_last_checkpoint_path` before subsequent file operations.
        stage2_last_checkpoint_path = os.path.abspath(str(stage2_config["training"]["resume"]["path"]))
        top_level_last_checkpoint_path = os.path.abspath(str(resume_cfg["path"]))
        if os.path.exists(stage2_last_checkpoint_path):
            shutil.copy2(stage2_last_checkpoint_path, top_level_last_checkpoint_path)
            # Records the current execution state to make progress and failures traceable.
            logger.info("已同步第二阶段续训检查点到顶层目录: %s", top_level_last_checkpoint_path)

        trainer = stage2_result["trainer"]
        model = stage2_result["model"]
        history = {
            # Completes the multi-line `list` arguments or elements.
            "train_loss": list(stage2_result["history"].get("train_loss", [])),
            "val_loss": list(stage2_result["history"].get("val_loss", [])),
            "train_units": list(stage2_result["history"].get("train_units", [])),
            "val_units": list(stage2_result["history"].get("val_units", [])),
            # Completes the multi-line `str` arguments or elements.
            "schedule_unit": str(stage2_result["history"].get("schedule_unit", "step")),
            "stage_1": copy.deepcopy(stage1_result["history"]),
            "stage_2": copy.deepcopy(stage2_result["history"]),
        }
        # Converts the `training_terminated` to `bool` so later checks use a consistent type.
        training_terminated = bool(
            stage1_result.get("training_terminated", False)
            or stage2_result.get("training_terminated", False)
        )
        # Stores the `cleanup_state['trainer']` for use by the remaining steps in this flow.
        cleanup_state["trainer"] = trainer
    else:
        stage_result = _run_stage_training(
            runtime_config=runtime_config,
            # Completes the multi-line `_run_stage_training` arguments or elements.
            resolved_model_type=resolved_model_type,
            task_type=task_type,
            processor=processor,
            train_dataset=train_dataset,
            # Completes the multi-line `_run_stage_training` arguments or elements.
            valid_dataset=valid_dataset,
            logger=logger,
            progress_callback=progress_callback,
            terminate_training_checker=terminate_training_checker,
            # Completes the multi-line `_run_stage_training` arguments or elements.
            trainer_class=Trainer,
            teacher_model=None,
            stage_name="模型训练",
        )
        # Selects the required item from `stage_result` as the `trainer`.
        trainer = stage_result["trainer"]
        model = stage_result["model"]
        history = stage_result["history"]
        best_model_path = stage_result["best_model_path"]
        # Selects the required item from `stage_result` as the `checkpoint`.
        checkpoint = stage_result["checkpoint"]
        training_terminated = bool(stage_result.get("training_terminated", False))
        cleanup_state["trainer"] = trainer

    train_rows = load_sample_rows(
        # Completes the multi-line `load_sample_rows` arguments or elements.
        effective_train_csv,
        task_type=task_type,
        mode=task_cfg.get("mode"),
    )
    # Stores the `valid_rows` for use by the remaining steps in this flow.
    valid_rows = load_sample_rows(
        effective_valid_csv,
        task_type=task_type,
        mode=task_cfg.get("mode"),
    # Completes the multi-line `load_sample_rows` arguments or elements.
    )
    auto_metrics: Dict = {}
    auto_evaluate_enabled = _is_auto_evaluate_after_train_enabled(training_cfg)
    if not auto_evaluate_enabled:
        # Records the current execution state to make progress and failures traceable.
        logger.info("已关闭训练结束自动评估，跳过完整 train/valid 评估")
    elif training_terminated:
        logger.info("训练已被外部终止，跳过训练结束自动评估")
    elif not os.path.exists(best_model_path):
        # Records the current execution state to make progress and failures traceable.
        logger.warning("未找到 best_model.pt，跳过训练结束自动评估: %s", best_model_path)
    else:
        auto_evaluate_result_train = _run_auto_evaluation_from_checkpoint(
            checkpoint_path=best_model_path,
            # Completes the multi-line `_run_auto_evaluation_from_checkpoint` arguments or elements.
            experiment_dir=experiment_dir,
            sample_rows=train_rows,
            task_type=task_type,
            processed_dir=processed_dir,
            # Completes the multi-line `_run_auto_evaluation_from_checkpoint` arguments or elements.
            split_name="train",
        )
        auto_evaluate_result_valid = _run_auto_evaluation_from_checkpoint(
            checkpoint_path=best_model_path,
            # Completes the multi-line `_run_auto_evaluation_from_checkpoint` arguments or elements.
            experiment_dir=experiment_dir,
            sample_rows=valid_rows,
            task_type=task_type,
            processed_dir=processed_dir,
            # Completes the multi-line `_run_auto_evaluation_from_checkpoint` arguments or elements.
            split_name="valid",
        )
        auto_metrics = dict(auto_evaluate_result_valid.get("metrics") or {})
        trainer.update_checkpoint_evaluation(
            # Completes the multi-line `trainer.update_checkpoint_evaluation` arguments or elements.
            path=best_model_path,
            evaluate_result=auto_evaluate_result_train,
            field_name="evaluate_result_train",
            split="train",
            # Completes the multi-line `trainer.update_checkpoint_evaluation` arguments or elements.
            extra_info=checkpoint_extra_info,
        )
        trainer.update_checkpoint_evaluation(
            path=best_model_path,
            # Completes the multi-line `trainer.update_checkpoint_evaluation` arguments or elements.
            evaluate_result=auto_evaluate_result_valid,
            field_name="evaluate_result_valid",
            split="valid",
            extra_info=checkpoint_extra_info,
        # Completes the multi-line `trainer.update_checkpoint_evaluation` arguments or elements.
        )
        logger.info("已将训练集和验证集自动评估结果回写到 best_model.pt")

    samples_train_processed = _build_processed_sample_records(
        sample_rows=train_rows,
        # Completes the multi-line `_build_processed_sample_records` arguments or elements.
        subset="train",
        task_type=task_type,
        processed_dir=processed_dir,
    ) + _build_processed_sample_records(
        # Completes the multi-line `_build_processed_sample_records` arguments or elements.
        sample_rows=valid_rows,
        subset="valid",
        task_type=task_type,
        processed_dir=processed_dir,
    # Completes the multi-line `_build_processed_sample_records` arguments or elements.
    )

    train_info = _build_train_info(
        experiment_dir=experiment_dir,
        best_model_path=best_model_path,
        # Completes the multi-line `_build_train_info` arguments or elements.
        train_log_path=train_log_path,
        history=history,
        auto_evaluate_metrics=auto_metrics,
    )
    # Records the current execution state to make progress and failures traceable.
    logger.info("训练流程完成")
    return train_info, samples_train_processed


def run_train(
    config: Dict,
    # Continues the surrounding statement or collection definition.
    samples_train: List[Dict],
    experiment_dir: str,
    model_type: str,
    train_params: Dict,
    # Completes the multi-line `collection` arguments or elements.
    progress_callback: Optional[Callable[[Dict], None]] = None,
    terminate_training_checker: Optional[Callable[[], bool]] = None,
    extra_info: Optional[Dict] = None,
) -> Tuple[Dict, List[Dict]]:
    # Calls `validate_qiyuan_config` to perform the operation required at this stage.
    validate_qiyuan_config(config, model_type=model_type)
    cleanup_state: Dict = {}
    try:
        return _run_train_impl(
            # Completes the multi-line `_run_train_impl` arguments or elements.
            config=config,
            samples_train=samples_train,
            experiment_dir=experiment_dir,
            model_type=model_type,
            # Completes the multi-line `_run_train_impl` arguments or elements.
            train_params=train_params,
            progress_callback=progress_callback,
            terminate_training_checker=terminate_training_checker,
            extra_info=extra_info,
            # Completes the multi-line `_run_train_impl` arguments or elements.
            cleanup_state=cleanup_state,
        )
    finally:
        release_kwargs = {
            # Completes the multi-line `cleanup_state.get` arguments or elements.
            "trainer": cleanup_state.get("trainer"),
            "processor": cleanup_state.get("processor"),
            "reader": cleanup_state.get("reader"),
            "runtime_config": cleanup_state.get("runtime_config"),
            # Completes the multi-line `cleanup_state.get` arguments or elements.
            "original_cpu_affinity_ids": cleanup_state.get("original_cpu_affinity_ids"),
            "logger": cleanup_state.get("logger"),
        }
        release_signature = inspect.signature(_release_train_runtime_resources)
        # Builds the `supported_kwargs` using the comprehension's filter and mapping rules.
        supported_kwargs = {
            key: value
            for key, value in release_kwargs.items()
            if key in release_signature.parameters
        # Continues the surrounding statement or collection definition.
        }
        _release_train_runtime_resources(**supported_kwargs)
