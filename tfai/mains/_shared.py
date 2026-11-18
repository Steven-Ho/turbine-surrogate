"""`tfai.mains` 内部共享工具。"""

import io
from pathlib import Path

import csv
import os
from numbers import Number
# Imports `Any, Dict, List` from `typing` for use in this module.
from typing import Any, Dict, List, Optional

import pandas as pd

from tfai.readers.sample_map import SampleMap
from tfai.utils.path_utils import resolve_path_from_anchor
# Imports `get_project_root` from `tfai.utils.runtime_paths` for use in this module.
from tfai.utils.runtime_paths import get_project_root
from tfai.utils.sample_utils import resolve_sample_source_path


PROJECT_ROOT = get_project_root()
DEFAULT_TEMPLATE_EXPERIMENT = "exp_HK612_demo"


# 判断路径值是否仍是模板占位或缺省值。
def should_reset_runtime_path(path_value) -> bool:
    """判断路径值是否仍是模板占位或缺省值。"""
    if path_value is None:
        return True

    raw = str(path_value).strip()
    # Returns early when `not raw` makes the remaining work inapplicable.
    if not raw:
        return True

    return "${" in raw or not os.path.isabs(raw)


def _is_missing_or_template_path(path_value) -> bool:
    """判断路径是否为空或仍包含未解析模板。"""
    # Returns early when `path_value is None` makes the remaining work inapplicable.
    if path_value is None:
        return True

    raw = str(path_value).strip()
    return not raw or "${" in raw


# 统一路径文本，便于识别包内模板默认路径。
def _normalize_path_text(path_value) -> str:
    """统一路径文本，便于识别包内模板默认路径。"""
    return os.path.normpath(str(path_value).strip()).replace("\\", "/").rstrip("/")


def _matches_path_text(path_value, candidates: List[str]) -> bool:
    normalized = _normalize_path_text(path_value)
    # Returns `` to the caller for the next stage.
    return any(normalized == _normalize_path_text(candidate) for candidate in candidates)


def is_packaged_default_experiment(path_value) -> bool:
    """判断 experiment 是否仍是随包模板默认目录。"""
    if _is_missing_or_template_path(path_value):
        return True
    # Returns `` to the caller for the next stage.
    return _matches_path_text(
        path_value,
        [
            DEFAULT_TEMPLATE_EXPERIMENT,
            # Completes the multi-line `os.path.join` arguments or elements.
            os.path.join(PROJECT_ROOT, DEFAULT_TEMPLATE_EXPERIMENT),
        ],
    )


def _prediction_default_candidates(path_suffix: str) -> List[str]:
    # Returns `` to the caller for the next stage.
    return [
        f"${{general.experiment}}/{path_suffix}",
        f"${{prediction.experiment}}/{path_suffix}",
        os.path.join(DEFAULT_TEMPLATE_EXPERIMENT, path_suffix),
        # Completes the multi-line `os.path.join` arguments or elements.
        os.path.join(PROJECT_ROOT, DEFAULT_TEMPLATE_EXPERIMENT, path_suffix),
    ]


def _is_packaged_prediction_default_path(path_value, path_suffix: str) -> bool:
    if _is_missing_or_template_path(path_value):
        # Returns `True` to the caller for the next stage.
        return True
    return _matches_path_text(path_value, _prediction_default_candidates(path_suffix))


def _is_packaged_evaluate_default_path(path_value, split_name: str) -> bool:
    if _is_missing_or_template_path(path_value):
        # Returns `True` to the caller for the next stage.
        return True

    split_name = str(split_name or "test")
    path_suffix = os.path.join("evaluate", split_name)
    candidates = [
        # Completes the multi-line `collection` arguments or elements.
        "${evaluate.experiment}/evaluate/${evaluate.split}",
        f"${{evaluate.experiment}}/evaluate/{split_name}",
        os.path.join(DEFAULT_TEMPLATE_EXPERIMENT, "evaluate", "${evaluate.split}"),
        os.path.join(DEFAULT_TEMPLATE_EXPERIMENT, path_suffix),
        # Completes the multi-line `os.path.join` arguments or elements.
        os.path.join(PROJECT_ROOT, DEFAULT_TEMPLATE_EXPERIMENT, path_suffix),
    ]
    return _matches_path_text(path_value, candidates)


def _is_packaged_train_samples_default_path(path_value) -> bool:
    # Returns early when `_is_missing_or_template_path(path_value)` makes the remaining work inapplicable.
    if _is_missing_or_template_path(path_value):
        return True

    path_suffix = os.path.join("data", "raw", "samples.csv")
    candidates = [
        # Completes the multi-line `collection` arguments or elements.
        "${general.experiment}/data/raw/samples.csv",
        os.path.join(DEFAULT_TEMPLATE_EXPERIMENT, path_suffix),
        os.path.join(PROJECT_ROOT, DEFAULT_TEMPLATE_EXPERIMENT, path_suffix),
    ]
    # Returns `_matches_path_text(path_value, candidates)` to the caller for the next stage.
    return _matches_path_text(path_value, candidates)


def apply_train_experiment_dir(config: Dict, experiment_dir: str) -> Dict:
    """将训练实验目录写入运行时配置，并重置派生路径。"""
    general_cfg = config.setdefault("general", {})
    training_cfg = config.setdefault("training", {})
    # Stores the model settings for use by the remaining steps in this flow.
    model_cfg = config.setdefault("model", {})
    resume_cfg = training_cfg.setdefault("resume", {})

    general_cfg["experiment"] = str(experiment_dir)
    general_cfg["train_samples"] = ""
    # Stores the `general_cfg['train_processed_dir']` for use by the remaining steps in this flow.
    general_cfg["train_processed_dir"] = ""
    model_cfg["checkpoint_path"] = ""
    resume_cfg["path"] = ""
    return config


# 将预测运行目录写入 experiment，并重置派生路径。
def apply_prediction_run_dir(config: Dict, run_dir: str) -> Dict:
    """将预测运行目录写入 experiment，并重置派生路径。"""
    general_cfg = config.setdefault("general", {})
    prediction_cfg = config.setdefault("prediction", {})
    general_cfg["experiment"] = str(run_dir)
    # Stores the `prediction_cfg['predict_samples']` for use by the remaining steps in this flow.
    prediction_cfg["predict_samples"] = ""
    prediction_cfg["processed_dir"] = ""
    prediction_cfg["output_dir"] = ""
    return config


# 将预测输出目录写入运行时配置。
def apply_predict_output_dir(config: Dict, output_dir: str) -> Dict:
    """将预测输出目录写入运行时配置。"""
    prediction_cfg = config.setdefault("prediction", {})
    prediction_cfg["output_dir"] = str(output_dir)
    return config


# 从预测输出目录反推出日志目录。
def resolve_prediction_run_dir_from_output(output_dir: str) -> str:
    """从预测输出目录反推出日志目录。"""
    normalized_output_dir = os.path.normpath(str(output_dir))
    if os.path.basename(normalized_output_dir).lower() == "results":
        return os.path.dirname(normalized_output_dir)
    # Returns `normalized_output_dir` to the caller for the next stage.
    return normalized_output_dir


def resolve_evaluate_run_dir_from_output(output_dir: str, split_name: str) -> Optional[str]:
    """从标准评估输出目录反推出实验目录。"""
    normalized_output_dir = os.path.normpath(str(output_dir))
    split_name = str(split_name or "test").strip().lower()
    # Returns early when `os.path.basename(normalized_output_dir).lower() != split_name` makes the remaining work inapplicable.
    if os.path.basename(normalized_output_dir).lower() != split_name:
        return None

    evaluate_dir = os.path.dirname(normalized_output_dir)
    if os.path.basename(evaluate_dir).lower() != "evaluate":
        # Returns `None` to the caller for the next stage.
        return None

    return os.path.dirname(evaluate_dir)


def apply_evaluate_run_dir(config: Dict, run_dir: str) -> Dict:
    """将评估运行目录写入 experiment，并重置派生输出路径。"""
    evaluate_cfg = config.setdefault("evaluate", {})
    # Converts the `evaluate_cfg['experiment']` to `str` so later checks use a consistent type.
    evaluate_cfg["experiment"] = str(run_dir)
    evaluate_cfg["output_dir"] = ""
    return config


def apply_evaluate_output_dir(config: Dict, output_dir: str) -> Dict:
    """将评估输出目录写入运行时配置。"""
    # Stores the `evaluate_cfg` for use by the remaining steps in this flow.
    evaluate_cfg = config.setdefault("evaluate", {})
    evaluate_cfg["output_dir"] = str(output_dir)
    return config


def ensure_train_runtime_paths(config: Dict) -> Dict:
    """补齐训练阶段约定的默认目录与关键文件路径。"""
    # Stores the `general_cfg` for use by the remaining steps in this flow.
    general_cfg = config.setdefault("general", {})
    training_cfg = config.setdefault("training", {})
    model_cfg = config.setdefault("model", {})
    resume_cfg = training_cfg.setdefault("resume", {})

    # Stores the `experiment_dir` for use by the remaining steps in this flow.
    experiment_dir = general_cfg.get("experiment") or os.path.join(PROJECT_ROOT, "exp_package_train")
    experiment_dir = os.path.abspath(str(experiment_dir))
    general_cfg["experiment"] = experiment_dir

    default_train_samples = os.path.join(experiment_dir, "data", "raw", "samples.csv")
    # Stores the `default_train_processed_dir` for use by the remaining steps in this flow.
    default_train_processed_dir = os.path.join(experiment_dir, "data", "processed")
    default_checkpoint_path = os.path.join(experiment_dir, "train", "best_model.pt")
    default_resume_path = os.path.join(experiment_dir, "train", "last_checkpoint.pt")

    if should_reset_runtime_path(general_cfg.get("train_samples")):
        # Stores the `general_cfg['train_samples']` for use by the remaining steps in this flow.
        general_cfg["train_samples"] = default_train_samples
    if should_reset_runtime_path(general_cfg.get("train_processed_dir")):
        general_cfg["train_processed_dir"] = default_train_processed_dir
    if should_reset_runtime_path(model_cfg.get("checkpoint_path")):
        # Stores the `model_cfg['checkpoint_path']` for use by the remaining steps in this flow.
        model_cfg["checkpoint_path"] = default_checkpoint_path
    if should_reset_runtime_path(resume_cfg.get("path")):
        resume_cfg["path"] = default_resume_path

    general_cfg["train_samples"] = os.path.abspath(str(general_cfg["train_samples"]))
    # Normalizes the `general_cfg['train_processed_dir']` before subsequent file operations.
    general_cfg["train_processed_dir"] = os.path.abspath(str(general_cfg["train_processed_dir"]))
    model_cfg["checkpoint_path"] = os.path.abspath(str(model_cfg["checkpoint_path"]))
    resume_cfg["path"] = os.path.abspath(str(resume_cfg["path"]))
    return config


# 清理训练配置中仍是模板默认值的输出路径。
def clear_train_default_output_paths(config: Dict) -> Dict:
    """清理训练配置中仍是模板默认值的输出路径。"""
    general_cfg = config.setdefault("general", {})
    training_cfg = config.setdefault("training", {})
    model_cfg = config.setdefault("model", {})
    # Stores the `resume_cfg` for use by the remaining steps in this flow.
    resume_cfg = training_cfg.setdefault("resume", {})

    if _is_packaged_train_samples_default_path(general_cfg.get("train_samples")):
        general_cfg["train_samples"] = ""
    if should_reset_runtime_path(general_cfg.get("train_processed_dir")):
        # Stores the `general_cfg['train_processed_dir']` for use by the remaining steps in this flow.
        general_cfg["train_processed_dir"] = ""
    if should_reset_runtime_path(model_cfg.get("checkpoint_path")):
        model_cfg["checkpoint_path"] = ""
    if should_reset_runtime_path(resume_cfg.get("path")):
        # Stores the `resume_cfg['path']` for use by the remaining steps in this flow.
        resume_cfg["path"] = ""
    return config


def ensure_prediction_runtime_paths(config: Dict, checkpoint_path: str) -> Dict:
    """补齐预测阶段默认目录约定。"""
    general_cfg = config.setdefault("general", {})
    # Stores the `prediction_cfg` for use by the remaining steps in this flow.
    prediction_cfg = config.setdefault("prediction", {})

    legacy_prediction_experiment = prediction_cfg.pop("experiment", None)
    experiment_dir = general_cfg.get("experiment") or legacy_prediction_experiment
    if not experiment_dir:
        # Stores the `experiment_dir` for use by the remaining steps in this flow.
        experiment_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path)))
    experiment_dir = os.path.abspath(str(experiment_dir))
    general_cfg["experiment"] = experiment_dir

    predict_dir = os.path.join(experiment_dir, "predict")
    # Stores the `default_processed_dir` for use by the remaining steps in this flow.
    default_processed_dir = os.path.join(predict_dir, "data", "processed")
    default_output_dir = os.path.join(predict_dir, "results")
    default_samples_csv = os.path.join(predict_dir, "data", "raw", "predict_samples.csv")

    processed_dir = prediction_cfg.get("processed_dir")
    # Reads `output_dir` from the mapping into the output directory.
    output_dir = prediction_cfg.get("output_dir")
    samples_csv = prediction_cfg.get("predict_samples")

    if should_reset_runtime_path(processed_dir):
        processed_dir = default_processed_dir
    # Checks `should_reset_runtime_path(output_dir)` before entering this conditional branch.
    if should_reset_runtime_path(output_dir):
        output_dir = default_output_dir
    if should_reset_runtime_path(samples_csv):
        samples_csv = default_samples_csv

    # Normalizes the `prediction_cfg['processed_dir']` before subsequent file operations.
    prediction_cfg["processed_dir"] = os.path.abspath(str(processed_dir))
    prediction_cfg["output_dir"] = os.path.abspath(str(output_dir))
    prediction_cfg["predict_samples"] = os.path.abspath(str(samples_csv))
    return config


# 清理预测配置中仍是包内默认值的路径覆盖项。
def clear_prediction_default_paths(config: Dict) -> Dict:
    """清理预测配置中仍是包内默认值的路径覆盖项。"""
    prediction_cfg = config.setdefault("prediction", {})
    default_suffixes = {
        "predict_samples": os.path.join("predict", "data", "raw", "predict_samples.csv"),
        # Completes the multi-line `os.path.join` arguments or elements.
        "processed_dir": os.path.join("predict", "data", "processed"),
        "output_dir": os.path.join("predict", "results"),
    }
    for key, path_suffix in default_suffixes.items():
        # Checks `` before entering this conditional branch.
        if _is_packaged_prediction_default_path(prediction_cfg.get(key), path_suffix):
            prediction_cfg[key] = ""
    return config


def clear_evaluate_default_paths(config: Dict) -> Dict:
    """清理评估配置中仍是包内默认值的输出路径。"""
    # Stores the `evaluate_cfg` for use by the remaining steps in this flow.
    evaluate_cfg = config.setdefault("evaluate", {})
    if _is_packaged_evaluate_default_path(
        evaluate_cfg.get("output_dir"),
        split_name=evaluate_cfg.get("split", "test"),
    # Completes the multi-line `_is_packaged_evaluate_default_path` arguments or elements.
    ):
        evaluate_cfg["output_dir"] = ""
    return config


def normalize_predict_samples(
    # Continues the surrounding statement or collection definition.
    samples_predict: List[Dict],
    is_geometry_varying: bool,
    default_source_file: str = "",
    expected_input_variables: Optional[List[str]] = None,
    # Continues the surrounding statement or collection definition.
    require_exact_input_variables: bool = False,
) -> List[Dict]:
    """规范化内存预测样本列表。"""
    if not samples_predict:
        raise ValueError("samples_predict 不能为空")

    # Initializes the `normalized_items` as a collection for values from this processing stage.
    normalized_items: List[Dict] = []
    seen_orders = set()
    normalized_default_source = ""
    if str(default_source_file or "").strip():
        # Normalizes the `normalized_default_source` before subsequent file operations.
        normalized_default_source = os.path.abspath(str(default_source_file).strip())
    expected_variables = list(dict.fromkeys(
        str(name).strip()
        for name in expected_input_variables or []
        # Completes the multi-line `str(name).strip` arguments or elements.
        if str(name).strip()
    ))
    reserved_keys = {
        "order",
        # Completes the multi-line `collection` arguments or elements.
        "sample_id",
        "params",
        "source_file",
        "file_path",
        # Completes the multi-line `collection` arguments or elements.
        "resolved_path",
        "path_type",
        "exists",
        "processed_path",
        # Completes the multi-line `collection` arguments or elements.
        "subset",
    }

    for index, item in enumerate(samples_predict):
        if not isinstance(item, dict):
            # Raises `TypeError` to report why this path cannot continue.
            raise TypeError("samples_predict 中的每个样本都必须是字典")

        order = int(item.get("order", item.get("sample_id", index)))
        if order in seen_orders:
            raise ValueError(f"samples_predict 中存在重复的样本编号: {order}")
        # Adds the current value to `seen_orders` for later aggregation.
        seen_orders.add(order)

        source_file = resolve_sample_source_path(item)
        if not source_file and normalized_default_source:
            source_file = normalized_default_source

        # Checks `is_geometry_varying` before entering this conditional branch.
        if is_geometry_varying:
            if not source_file:
                raise ValueError(
                    f"变几何预测样本缺少 source_file（兼容 file_path/resolved_path）: order={order}"
                # Completes the multi-line `ValueError` arguments or elements.
                )
            if not os.path.exists(source_file):
                raise FileNotFoundError(f"变几何预测原始样本文件不存在: {source_file}")

        params = dict(item.get("params") or {})
        # Visits each item in `item.items()` and applies the current processing step.
        for key, value in item.items():
            if key not in reserved_keys and key not in params:
                params[key] = value

        if require_exact_input_variables and expected_variables:
            # Builds the `missing_variables` using the comprehension's filter and mapping rules.
            missing_variables = [
                name for name in expected_variables if name not in params
            ]
            extra_variables = [
                # Completes the multi-line `str` arguments or elements.
                str(name) for name in params if str(name) not in expected_variables
            ]
            non_numeric_variables = [
                name
                # Continues the surrounding statement or collection definition.
                for name in expected_variables
                if name in params and not isinstance(params[name], Number)
            ]
            if missing_variables or extra_variables or non_numeric_variables:
                # Initializes the `details` as a collection for values from this processing stage.
                details = []
                if missing_variables:
                    details.append(f"缺少变量 {missing_variables}")
                if extra_variables:
                    # Adds the current value to `details` for later aggregation.
                    details.append(f"存在未声明变量 {extra_variables}")
                if non_numeric_variables:
                    details.append(f"非数值变量 {non_numeric_variables}")
                raise ValueError(
                    # Completes the multi-line `ValueError` arguments or elements.
                    f"预测样本 {order} 的输入参数 schema 与模型不一致："
                    f"期望 {expected_variables}；" + "；".join(details)
                )
            params = {name: params[name] for name in expected_variables}

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


# 将内存预测样本列表落为样本表。
def write_predict_samples_csv(
    samples_predict: List[Dict],
    sample_csv_path: str,
) -> str:
    """将内存预测样本列表落为样本表。"""
    # Calls `os.makedirs` to perform the operation required at this stage.
    os.makedirs(os.path.dirname(sample_csv_path), exist_ok=True)

    param_fields = sorted(
        {
            key
            # Completes the multi-line `sorted` arguments or elements.
            for item in samples_predict
            for key in dict(item.get("params") or {}).keys()
        }
    )
    # Derives the `fieldnames` from the current values for downstream processing.
    fieldnames = ["order"] + param_fields + ["file_path"]

    csv_file = io.StringIO(newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    # Visits each item in `samples_predict` and applies the current processing step.
    for item in samples_predict:
        row = {"order": int(item["order"]), "file_path": item.get("source_file", "")}
        row.update(dict(item.get("params") or {}))
        writer.writerow(row)

    content = csv_file.getvalue().encode("utf-8")
    # Keep content-identical tables stable so their preprocessing fingerprint
    # survives model changes, validation runs, and resumed training.
    if not os.path.isfile(sample_csv_path) or Path(sample_csv_path).read_bytes() != content:
        Path(sample_csv_path).write_bytes(content)
    return sample_csv_path


def load_sample_rows(
    sample_csv: str,
    task_type: Optional[str] = None,
    # Continues the surrounding statement or collection definition.
    mode: Optional[str] = None,
) -> List[Dict]:
    """从样本表构建统一样本列表。"""
    return _load_sample_rows_internal(
        sample_csv,
        # Completes the multi-line `_load_sample_rows_internal` arguments or elements.
        task_type=task_type,
        mode=mode,
    )


def _load_sample_rows_internal(
    # Continues the surrounding statement or collection definition.
    sample_csv: str,
    task_type: Optional[str] = None,
    mode: Optional[str] = None,
) -> List[Dict]:
    """按任务语义回退解析样本表。"""
    # Stores the `sample_map` for use by the remaining steps in this flow.
    sample_map = SampleMap(sample_csv)
    sample_ids = sample_map.get_all_samples()
    if sample_ids:
        rows: List[Dict] = []
        # Visits each item in `sample_ids` and applies the current processing step.
        for sample_id in sample_ids:
            info = dict(sample_map.get_sample_info(sample_id) or {})
            file_path = str(info.pop("file_path", "") or "")
            resolved_path = resolve_path_from_anchor(sample_csv, file_path) if file_path else ""
            # Adds the current value to `rows` for later aggregation.
            rows.append(
                {
                    "order": int(sample_id),
                    "file_path": file_path,
                    # Completes the multi-line `collection` arguments or elements.
                    "resolved_path": resolved_path,
                    "source_file": resolved_path,
                    "path_type": "directory" if (resolved_path and os.path.isdir(resolved_path)) else "file",
                    "exists": bool(resolved_path and os.path.exists(resolved_path)),
                    # Completes the multi-line `collection` arguments or elements.
                    "params": info,
                }
            )
        return rows

    # Stores the `normalized_task_type` for use by the remaining steps in this flow.
    normalized_task_type = str(task_type or "").strip().lower()
    normalized_mode = str(mode or "").strip().lower()
    df = pd.read_csv(sample_csv)

    if normalized_task_type == "scalar" and normalized_mode == "temporal":
        # Returns `` to the caller for the next stage.
        return [
            {
                "order": 0,
                "file_path": "",
                # Completes the multi-line `collection` arguments or elements.
                "resolved_path": "",
                "source_file": "",
                "path_type": "file",
                "exists": True,
                # Completes the multi-line `collection` arguments or elements.
                "params": {
                    "row_count": int(len(df)),
                },
            }
        # Completes the multi-line `collection` arguments or elements.
        ]

    rows = []
    for row_index, row in df.reset_index(drop=True).iterrows():
        params = {
            # Completes the multi-line `str` arguments or elements.
            str(key): value
            for key, value in row.to_dict().items()
        }
        rows.append(
            # Completes the multi-line `collection` arguments or elements.
            {
                "order": int(row_index),
                "file_path": "",
                "resolved_path": "",
                # Completes the multi-line `collection` arguments or elements.
                "source_file": "",
                "path_type": "file",
                "exists": True,
                "params": params,
            # Completes the multi-line `collection` arguments or elements.
            }
        )
    return rows


def infer_parameter_schema(samples_csv: Optional[str]) -> List[Dict]:
    """从样本表首条记录推断预测参数定义。"""
    # Returns early when `not samples_csv or not os.path.exists(samples_csv)` makes the remaining work inapplicable.
    if not samples_csv or not os.path.exists(samples_csv):
        return []

    sample_map = SampleMap(samples_csv)
    sample_ids = sample_map.get_all_samples()
    # Returns early when `not sample_ids` makes the remaining work inapplicable.
    if not sample_ids:
        return []

    first_info = dict(sample_map.get_sample_info(sample_ids[0]) or {})
    schema: List[Dict] = []
    # Visits each item in `first_info.items()` and applies the current processing step.
    for key, value in first_info.items():
        if key == "file_path":
            continue
        if isinstance(value, bool):
            # Stores the `value_type` for use by the remaining steps in this flow.
            value_type = "bool"
        elif isinstance(value, int):
            value_type = "int"
        elif isinstance(value, float):
            # Stores the `value_type` for use by the remaining steps in this flow.
            value_type = "float"
        else:
            value_type = "str"
        schema.append({"name": str(key), "type": value_type})
    # Returns `schema` to the caller for the next stage.
    return schema


def infer_parameter_schema_from_samples(samples: Optional[List[Dict]]) -> List[Dict]:
    """从内存样本参数推断预测参数定义，用于固化到模型配置。"""
    if not samples:
        return []

    # Initializes the `values_by_name` as a collection for values from this processing stage.
    values_by_name: Dict[str, object] = {}
    for item in samples:
        if not isinstance(item, dict):
            continue
        # Stores the `params` for use by the remaining steps in this flow.
        params = dict(item.get("params") or {})
        for key, value in params.items():
            name = str(key)
            if name not in values_by_name or values_by_name[name] is None:
                # Stores the `values_by_name[name]` for use by the remaining steps in this flow.
                values_by_name[name] = value

    schema: List[Dict] = []
    for name, value in values_by_name.items():
        if isinstance(value, bool):
            # Stores the `value_type` for use by the remaining steps in this flow.
            value_type = "bool"
        elif isinstance(value, int):
            value_type = "int"
        elif isinstance(value, float):
            # Stores the `value_type` for use by the remaining steps in this flow.
            value_type = "float"
        else:
            value_type = "str"
        schema.append({"name": name, "type": value_type})
    # Returns `schema` to the caller for the next stage.
    return schema
