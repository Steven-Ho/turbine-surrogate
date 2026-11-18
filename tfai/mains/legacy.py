"""旧配置式入口的兼容适配层。"""

import copy
import os
import warnings
# Imports `Dict, List, Optional` from `typing` for use in this module.
from typing import Dict, List, Optional

from tfai.utils.path_utils import resolve_path_from_base
from tfai.utils.runtime_paths import resolve_runtime_paths

from ._shared import PROJECT_ROOT
# Imports `read_train_samples` from `data_read` for use in this module.
from .data_read import read_train_samples
from .evaluate import run_evaluate
from .model_load import load_model_for_prediction
from .predict import run_predict
# Imports `run_train` from `train` for use in this module.
from .train import run_train


def build_processed_sample_records(
    sample_rows: List[Dict],
    subset: str,
    # Continues the surrounding statement or collection definition.
    processed_path_getter,
) -> List[Dict]:
    """兼容旧训练结果样本结构。"""
    records: List[Dict] = []
    for item in sample_rows:
        # Stores the `record` for use by the remaining steps in this flow.
        record = dict(item)
        sample_id = int(item["order"])
        record.update(
            {
                # Completes the multi-line `collection` arguments or elements.
                "sample_id": sample_id,
                "subset": subset,
                "processed_path": processed_path_getter(sample_id),
            }
        # Completes the multi-line `record.update` arguments or elements.
        )
        records.append(record)
    return records


def run_train_from_config(config: Dict):
    """兼容仅传入配置的旧训练调用。"""
    # Calls `warnings.warn` to perform the operation required at this stage.
    warnings.warn(
        "仅传入 config 的训练入口已弃用，请直接调用 tfai.mains.run_train。",
        DeprecationWarning,
        stacklevel=2,
    # Completes the multi-line `warnings.warn` arguments or elements.
    )
    runtime_config = resolve_runtime_paths(
        copy.deepcopy(config),
        project_root=PROJECT_ROOT,
    # Completes the multi-line `resolve_runtime_paths` arguments or elements.
    )
    general_cfg = runtime_config.setdefault("general", {})
    task_cfg = runtime_config.setdefault("task", {})
    samples_all, _ = read_train_samples(
        # Completes the multi-line `read_train_samples` arguments or elements.
        train_samples=general_cfg["train_samples"],
        is_geometry_varying=bool(task_cfg.get("is_geometry_varying", False)),
        mode=str(task_cfg.get("mode", "steady")),
        config=runtime_config,
    # Completes the multi-line `read_train_samples` arguments or elements.
    )
    return run_train(
        config=runtime_config,
        samples_train=samples_all,
        # Completes the multi-line `str` arguments or elements.
        experiment_dir=str(general_cfg["experiment"]),
        model_type=str(runtime_config.setdefault("model", {})["type"]),
        train_params={},
    )


# 兼容仅传入配置的旧预测调用。
def run_predict_from_config(
    predict_cfg: Dict,
    checkpoint_path: Optional[str] = None,
    samples_predict: Optional[List[Dict]] = None,
    # Continues the surrounding statement or collection definition.
    default_source_file: Optional[str] = None,
):
    """兼容仅传入配置的旧预测调用。"""
    warnings.warn(
        "仅传入 config 的预测入口已弃用，请组合 load_model_for_prediction 和 run_predict。",
        # Completes the multi-line `warnings.warn` arguments or elements.
        DeprecationWarning,
        stacklevel=2,
    )
    runtime_config = resolve_runtime_paths(
        # Completes the multi-line `copy.deepcopy` arguments or elements.
        copy.deepcopy(predict_cfg),
        project_root=PROJECT_ROOT,
    )
    general_cfg = runtime_config.setdefault("general", {})
    # Stores the `prediction_cfg` for use by the remaining steps in this flow.
    prediction_cfg = runtime_config.setdefault("prediction", {})
    has_explicit_general_experiment = bool(general_cfg.get("experiment"))
    legacy_experiment = prediction_cfg.pop("experiment", None)
    experiment_dir = general_cfg.get("experiment") or legacy_experiment or "exp_001"
    # Stores the `experiment_dir` for use by the remaining steps in this flow.
    experiment_dir = resolve_path_from_base(PROJECT_ROOT, str(experiment_dir))
    checkpoint_path = checkpoint_path or prediction_cfg.get("checkpoint_path")
    if not checkpoint_path:
        checkpoint_path = os.path.join(experiment_dir, "train", "best_model.pt")
    # Stores the checkpoint path for use by the remaining steps in this flow.
    checkpoint_path = resolve_path_from_base(PROJECT_ROOT, str(checkpoint_path))

    model_bundle = load_model_for_prediction(
        model_path=checkpoint_path,
        config=runtime_config,
    # Completes the multi-line `load_model_for_prediction` arguments or elements.
    )
    if not has_explicit_general_experiment:
        prediction_context = model_bundle.get("prediction_context") or {}
        experiment_dir = str(
            # Completes the multi-line `prediction_context.get` arguments or elements.
            prediction_context.get("experiment")
            or os.path.dirname(os.path.dirname(checkpoint_path))
        )
    return run_predict(
        # Completes the multi-line `run_predict` arguments or elements.
        config=runtime_config,
        model_bundle=model_bundle,
        run_dir=experiment_dir,
        samples_predict=samples_predict,
        # Completes the multi-line `run_predict` arguments or elements.
        default_source_file=default_source_file,
    )


def run_evaluate_from_config(
    evaluate_cfg: Dict,
    # Continues the surrounding statement or collection definition.
    checkpoint_path: Optional[str] = None,
    samples_evaluate: Optional[List[Dict]] = None,
):
    """兼容仅传入配置的旧评估调用。"""
    warnings.warn(
        # Completes the multi-line `warnings.warn` arguments or elements.
        "仅传入 config 的评估入口已弃用，请直接调用 tfai.mains.run_evaluate。",
        DeprecationWarning,
        stacklevel=2,
    )
    # Stores the `runtime_config` for use by the remaining steps in this flow.
    runtime_config = resolve_runtime_paths(
        copy.deepcopy(evaluate_cfg),
        project_root=PROJECT_ROOT,
    )
    # Stores the `evaluate_section` for use by the remaining steps in this flow.
    evaluate_section = runtime_config.setdefault("evaluate", {})
    experiment_dir = resolve_path_from_base(
        PROJECT_ROOT,
        str(evaluate_section.get("experiment") or "exp_001"),
    # Completes the multi-line `resolve_path_from_base` arguments or elements.
    )
    checkpoint_path = checkpoint_path or evaluate_section.get("checkpoint_path")
    if not checkpoint_path:
        checkpoint_path = os.path.join(experiment_dir, "train", "best_model.pt")
    # Stores the checkpoint path for use by the remaining steps in this flow.
    checkpoint_path = resolve_path_from_base(PROJECT_ROOT, str(checkpoint_path))
    return run_evaluate(
        config=runtime_config,
        checkpoint_path=checkpoint_path,
        # Completes the multi-line `run_evaluate` arguments or elements.
        run_dir=experiment_dir,
        samples_evaluate=samples_evaluate,
    )
