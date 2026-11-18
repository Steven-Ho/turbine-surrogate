"""预测接口。"""

import copy
import os
import threading
import time
# Imports `weakref` for reuse in this module.
import weakref
from typing import Dict, List, Optional

from tfai.prediction.predictor import Predictor
from tfai.utils.logger import Logger
from tfai.utils.runtime_paths import resolve_runtime_paths
# Imports `resolve_packaged_default_source_file` from `tfai.utils.sample_utils` for use in this module.
from tfai.utils.sample_utils import resolve_packaged_default_source_file

from ._shared import (
    PROJECT_ROOT,
    apply_prediction_run_dir,
    # Continues the surrounding statement or collection definition.
    clear_prediction_default_paths,
    ensure_prediction_runtime_paths,
    is_packaged_default_experiment,
    normalize_predict_samples,
    # Continues the surrounding statement or collection definition.
    resolve_prediction_run_dir_from_output,
    write_predict_samples_csv,
)


_PREDICTOR_LOCKS = weakref.WeakKeyDictionary()
# Initializes the `_PREDICTOR_LOCKS_BY_ID` as a collection for values from this processing stage.
_PREDICTOR_LOCKS_BY_ID = {}
_PREDICTOR_LOCKS_GUARD = threading.Lock()


def _get_predictor_runtime_lock(predictor) -> threading.RLock:
    """获取共享 Predictor 的运行锁，避免并发请求互相改写运行态配置。"""
    with _PREDICTOR_LOCKS_GUARD:
        # Keeps the operation and its recovery path within the same exception boundary.
        try:
            lock = _PREDICTOR_LOCKS.get(predictor)
        except TypeError:
            lock_key = id(predictor)
            # Stores the `lock` for use by the remaining steps in this flow.
            lock = _PREDICTOR_LOCKS_BY_ID.get(lock_key)
            if lock is None:
                lock = threading.RLock()
                _PREDICTOR_LOCKS_BY_ID[lock_key] = lock
            # Returns `lock` to the caller for the next stage.
            return lock

        if lock is None:
            lock = threading.RLock()
            _PREDICTOR_LOCKS[predictor] = lock
        # Returns `lock` to the caller for the next stage.
        return lock


def _resolve_predictor(model_bundle) -> Predictor:
    """从输入对象中解析 Predictor 实例。"""
    if isinstance(model_bundle, dict):
        for key in ["model_bundle", "model_instance", "predictor_instance"]:
            # Stores the `predictor` for use by the remaining steps in this flow.
            predictor = model_bundle.get(key)
            if predictor is not None:
                return predictor
    if isinstance(model_bundle, Predictor):
        # Returns `model_bundle` to the caller for the next stage.
        return model_bundle
    if hasattr(model_bundle, "predict") and hasattr(model_bundle, "checkpoint_path"):
        return model_bundle
    raise TypeError("model_bundle 必须是 Predictor，或包含 model_bundle/model_instance 的字典")


# 解析变几何预测缺省使用的默认源文件。
def _resolve_default_source_file(model_bundle, predictor, default_source_file: Optional[str]) -> str:
    """解析变几何预测缺省使用的默认源文件。"""
    if str(default_source_file or "").strip():
        return os.path.abspath(str(default_source_file).strip())

    if isinstance(model_bundle, dict):
        # Reads `prediction_context` from the mapping into the `prediction_context`.
        prediction_context = model_bundle.get("prediction_context")
        if isinstance(prediction_context, dict):
            packaged_default_source = str(prediction_context.get("default_source_file") or "").strip()
            if packaged_default_source:
                # Returns `os.path.abspath(packaged_default_source)` to the caller for the next stage.
                return os.path.abspath(packaged_default_source)
        direct_default_source = str(model_bundle.get("default_source_file") or "").strip()
        if direct_default_source:
            return os.path.abspath(direct_default_source)

    # Stores the `checkpoint_dir` for use by the remaining steps in this flow.
    checkpoint_dir = os.path.dirname(
        os.path.abspath(str(getattr(predictor, "checkpoint_path", "") or ""))
    )
    if checkpoint_dir:
        # Returns `resolve_packaged_default_source_file(checkpoint_dir)` to the caller for the next stage.
        return resolve_packaged_default_source_file(checkpoint_dir)
    return ""


def run_predict(
    config: Dict,
    # Continues the surrounding statement or collection definition.
    model_bundle,
    run_dir: str,
    samples_predict: Optional[List[Dict]] = None,
    default_source_file: Optional[str] = None,
# Continues the surrounding statement or collection definition.
) -> Dict:
    """执行预测并返回预测结果信息。"""
    predictor = _resolve_predictor(model_bundle)
    resolved_default_source_file = _resolve_default_source_file(
        model_bundle=model_bundle,
        # Completes the multi-line `_resolve_default_source_file` arguments or elements.
        predictor=predictor,
        default_source_file=default_source_file,
    )

    runtime_config = copy.deepcopy(config)
    configured_prediction_paths = {
        key: value
        for key, value in runtime_config.get("prediction", {}).items()
        if key in {"predict_samples", "processed_dir", "output_dir"}
        and value is not None
        and str(value).strip()
    }
    # Stores the `normalized_run_dir` for use by the remaining steps in this flow.
    normalized_run_dir = str(run_dir or "").strip()
    if not normalized_run_dir:
        raise ValueError("run_dir 不能为空")
    runtime_config = apply_prediction_run_dir(
        # Completes the multi-line `apply_prediction_run_dir` arguments or elements.
        runtime_config,
        run_dir=normalized_run_dir,
    )
    runtime_config.setdefault("prediction", {}).update(configured_prediction_paths)
    requested_experiment = str(
        # Completes the multi-line `runtime_config.get('general', {}).get` arguments or elements.
        runtime_config.get("general", {}).get("experiment", "") or ""
    ).strip()
    has_explicit_general_experiment = (
        bool(requested_experiment)
        # Completes the multi-line `is_packaged_default_experiment` arguments or elements.
        and not is_packaged_default_experiment(requested_experiment)
    )
    runtime_config = clear_prediction_default_paths(runtime_config)
    runtime_config = resolve_runtime_paths(runtime_config, project_root=PROJECT_ROOT)
    # Calls `runtime_config.setdefault` to perform the operation required at this stage.
    runtime_config.setdefault("general", {})
    runtime_config.setdefault("prediction", {})

    predictor_lock = _get_predictor_runtime_lock(predictor)
    with predictor_lock:
        # Stores the `original_config` for use by the remaining steps in this flow.
        original_config = copy.deepcopy(predictor.config)
        original_batch_size = getattr(predictor, "batch_size", None)
        has_batch_size = hasattr(predictor, "batch_size")
        had_logger = hasattr(predictor, "logger")
        original_logger = getattr(predictor, "logger", None)
        predict_logger = None
        predict_start_time = None
        sample_count = 0
        try:
            # Initializes the `config['prediction']` as a collection for values from this processing stage.
            predictor.config["prediction"] = {
                **predictor.config.get("prediction", {}),
                **runtime_config.get("prediction", {}),
            }
            # Reads `experiment` from the mapping into the `configured_experiment`.
            configured_experiment = runtime_config.get("general", {}).get("experiment")
            if configured_experiment and has_explicit_general_experiment:
                predictor.config.setdefault("general", {})["experiment"] = runtime_config["general"]["experiment"]
            predictor.config = ensure_prediction_runtime_paths(
                # Completes the multi-line `ensure_prediction_runtime_paths` arguments or elements.
                predictor.config,
                checkpoint_path=predictor.checkpoint_path,
            )
            predictor.batch_size = predictor.config.get("prediction", {}).get("batch_size", predictor.batch_size)

            # Selects the required item from `predictor.config` as the `prediction_cfg`.
            prediction_cfg = predictor.config["prediction"]
            samples_csv = prediction_cfg["predict_samples"]
            processed_dir = prediction_cfg["processed_dir"]
            output_dir = prediction_cfg["output_dir"]
            # Stores the `predict_run_dir` for use by the remaining steps in this flow.
            predict_run_dir = resolve_prediction_run_dir_from_output(output_dir)

            os.makedirs(os.path.dirname(samples_csv), exist_ok=True)
            os.makedirs(processed_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(predict_run_dir, exist_ok=True)
            predict_log_path = os.path.join(predict_run_dir, "predict.log")
            predict_logger = Logger.setup_logger(
                f"tfai_package_predict_{id(predictor)}",
                predict_log_path,
            )
            predict_start_time = time.perf_counter()
            predictor.logger = predict_logger
            predict_logger.info("开始执行预测")
            predict_logger.info("预测样本: %s", samples_csv)
            predict_logger.info("预测结果目录: %s", output_dir)
            requested_device = predictor.config.get("general", {}).get("device", "auto")
            actual_device = getattr(predictor, "device", "unknown")
            predict_logger.info(
                "预测运行设备: %s (配置: %s)",
                actual_device,
                requested_device,
            )

            # Stores the `normalized_samples` for use by the remaining steps in this flow.
            normalized_samples = None
            if samples_predict is not None:
                task_cfg = predictor.config.get("task", {})
                task_type = str(task_cfg.get("type", "field")).lower()
                # Stores the `parameter_schema` for use by the remaining steps in this flow.
                parameter_schema = task_cfg.get("parameter_schema") or []
                schema_input_variables = [
                    str(item.get("name")).strip()
                    for item in parameter_schema
                    # Completes the multi-line `isinstance` arguments or elements.
                    if isinstance(item, dict) and str(item.get("name") or "").strip()
                ]
                normalized_samples = normalize_predict_samples(
                    samples_predict=samples_predict,
                    # Completes the multi-line `bool` arguments or elements.
                    is_geometry_varying=bool(getattr(predictor, "is_geometry_varying", False)),
                    default_source_file=resolved_default_source_file,
                    expected_input_variables=(
                        schema_input_variables or task_cfg.get("input_variables")
                    # Completes the multi-line `normalize_predict_samples` arguments or elements.
                    ),
                    require_exact_input_variables=task_type == "field",
                )
                samples_csv = write_predict_samples_csv(
                    # Completes the multi-line `write_predict_samples_csv` arguments or elements.
                    samples_predict=normalized_samples,
                    sample_csv_path=samples_csv,
                )
                prediction_cfg["predict_samples"] = samples_csv
            # Validates `not os.path.exists(samples_csv)` and raises `FileNotFoundError` when the input cannot be processed safely.
            elif not os.path.exists(samples_csv):
                raise FileNotFoundError(f"预测样本文件不存在: {samples_csv}")

            output_paths = predictor.predict(
                samples_csv=samples_csv,
                # Completes the multi-line `predictor.predict` arguments or elements.
                processed_dir=processed_dir,
                output_dir=output_dir,
            )
            sample_count = (
                len(normalized_samples)
                if normalized_samples is not None
                else len(output_paths)
            )
            predict_logger.info("预测完成: 结果文件数=%s", len(output_paths))
            return {
                # Completes the multi-line `collection` arguments or elements.
                "samples_csv": samples_csv,
                "processed_dir": processed_dir,
                "output_dir": output_dir,
                "predict_log_path": predict_log_path,
                # Completes the multi-line `collection` arguments or elements.
                "result_files": output_paths,
                "sample_count": sample_count,
                "is_geometry_varying": bool(getattr(predictor, "is_geometry_varying", False)),
                "writeback_report": copy.deepcopy(
                    # Completes the multi-line `getattr` arguments or elements.
                    getattr(predictor, "last_writeback_report", {}) or {}
                ),
            }
        except Exception:
            if predict_logger is not None:
                predict_logger.exception("预测执行失败")
            raise
        finally:
            # Calls `predictor.config.clear` to perform the operation required at this stage.
            predictor.config.clear()
            predictor.config.update(original_config)
            if has_batch_size:
                predictor.batch_size = original_batch_size
            if predict_logger is not None and predict_start_time is not None:
                total_seconds = time.perf_counter() - predict_start_time
                average_seconds = (
                    f"{total_seconds / sample_count:.2f} 秒"
                    if sample_count > 0
                    else "N/A（无已完成样本）"
                )
                predict_logger.info(
                    "预测总耗时: %.2f 秒，单样本平均总耗时: %s（样本数=%s）",
                    total_seconds,
                    average_seconds,
                    sample_count,
                )
            if had_logger:
                predictor.logger = original_logger
            elif hasattr(predictor, "logger"):
                delattr(predictor, "logger")
            if predict_logger is not None:
                Logger.close_logger(predict_logger)
