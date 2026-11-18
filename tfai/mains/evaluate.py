"""评估接口。"""

import copy
import gc
import os
# Imports `time` for reuse in this module.
import time
from typing import Dict, List, Optional

import torch

from tfai.dataset import build_dataset
# Imports `Evaluator` from `tfai.evaluation.evaluator` for use in this module.
from tfai.evaluation.evaluator import Evaluator
from tfai.evaluation.evaluation_runner import EvaluationRunner
from tfai.factories.processor_factory import ProcessorFactory
from tfai.utils.logger import Logger
# Imports `resolve_runtime_paths` from `tfai.utils.runtime_paths` for use in this module.
from tfai.utils.runtime_paths import resolve_runtime_paths
from tfai.utils.torch_utils import atomic_torch_save, load_torch

from ._shared import (
    PROJECT_ROOT,
    # Continues the surrounding statement or collection definition.
    apply_evaluate_run_dir,
    clear_evaluate_default_paths,
    load_sample_rows,
    resolve_evaluate_run_dir_from_output,
    # Continues the surrounding statement or collection definition.
    should_reset_runtime_path,
)


def resolve_evaluate_split(config: Dict) -> str:
    """校验并返回训练期划分名称。"""
    split_name = str(config.get("evaluate", {}).get("split", "test") or "test").strip().lower()
    # Validates `split_name not in {"train", "valid", "test"}` and raises `ValueError` when the input cannot be processed safely.
    if split_name not in {"train", "valid", "test"}:
        raise ValueError(f"不支持的 evaluate.split 值: {split_name}。仅支持 train、valid、test。")
    return split_name


def build_evaluate_dataset(config: Dict, split_name: str):
    """按训练期处理产物构建评估数据集。"""
    # Selects the required item from `config["task"]` as the `task_type`.
    task_type = config["task"]["type"]
    model_type = config["model"]["type"]
    processor = ProcessorFactory.get_processor(task_type, model_type, config)
    processor_type = type(processor).__name__
    # Stores the `setdefault('processor', {})['type']` for use by the remaining steps in this flow.
    config.setdefault("processor", {})["type"] = processor_type
    processed_dir = config["general"]["train_processed_dir"]
    split_csv = os.path.join(processed_dir, f"{split_name}.csv")
    if not os.path.exists(processed_dir):
        # Raises `FileNotFoundError` to report why this path cannot continue.
        raise FileNotFoundError(f"训练处理数据目录不存在: {processed_dir}")
    if not os.path.exists(split_csv):
        raise FileNotFoundError(f"评估数据划分文件不存在: {split_csv}")

    dataset_phase = split_name if split_name in {"valid", "test"} else "test"
    # Stores the `data_dir` for use by the remaining steps in this flow.
    data_dir = (
        os.path.join(processed_dir, split_name)
        if str(task_type).lower() == "scalar"
        else processed_dir
    # Continues the surrounding statement or collection definition.
    )
    dataset = build_dataset(
        config=config,
        processor_type=processor_type,
        # Completes the multi-line `build_dataset` arguments or elements.
        csv_path=split_csv,
        data_dir=data_dir,
        phase=dataset_phase,
        processor=processor,
    # Completes the multi-line `build_dataset` arguments or elements.
    )
    return dataset, processed_dir, split_csv, processor_type, dataset_phase


def run_evaluation_pipeline(
    model,
    # Continues the surrounding statement or collection definition.
    config: Dict,
    dataset,
    evaluate_dir: str,
    logger=None,
    # Continues the surrounding statement or collection definition.
    plot_scatter: bool = True,
    export_scatter_csv: bool = True,
    write_summary_log: bool = True,
) -> Dict:
    """对已构建模型和数据集执行低层评估。"""
    # Checks `write_summary_log or plot_scatter or export_scatter_csv` before entering this conditional branch.
    if write_summary_log or plot_scatter or export_scatter_csv:
        os.makedirs(evaluate_dir, exist_ok=True)
    evaluator = Evaluator(model, config)
    scatter_save_path = (
        # Completes the multi-line `os.path.join` arguments or elements.
        os.path.join(evaluate_dir, "evaluation_result.png")
        if plot_scatter
        else None
    )
    # Stores the `scatter_csv_path` for use by the remaining steps in this flow.
    scatter_csv_path = os.path.join(evaluate_dir, "scatter_all.csv") if export_scatter_csv else None
    evaluation_start = time.perf_counter()
    metrics, predictions, targets = evaluator.evaluate(
        dataset,
        # Completes the multi-line `evaluator.evaluate` arguments or elements.
        plot_scatter=plot_scatter,
        scatter_save_path=scatter_save_path,
        scatter_csv_path=scatter_csv_path,
    )
    # Calls `Logger.log_duration` to perform the operation required at this stage.
    Logger.log_duration(logger, "模型评估", time.perf_counter() - evaluation_start)
    evaluate_log_path = ""
    if write_summary_log:
        evaluate_log_path = evaluator.write_evaluation_log(
            # Completes the multi-line `evaluator.write_evaluation_log` arguments or elements.
            evaluate_dir=evaluate_dir,
            metrics=metrics,
            scatter_save_path=scatter_save_path,
            scatter_csv_path=scatter_csv_path,
        # Completes the multi-line `evaluator.write_evaluation_log` arguments or elements.
        )
    return {
        "metrics": metrics,
        "scatter_data": copy.deepcopy(
            # Completes the multi-line `copy.deepcopy` arguments or elements.
            evaluator.last_scatter_data
            or {"overall": {"predictions": [], "targets": []}, "by_variable": {}}
        ),
        "predictions": predictions,
        # Completes the multi-line `collection` arguments or elements.
        "targets": targets,
        "evaluate_log_path": evaluate_log_path,
        "scatter_save_path": scatter_save_path,
        "scatter_csv_path": scatter_csv_path,
    # Completes the multi-line `collection` arguments or elements.
    }


def _build_samples_evaluate_from_split(
    checkpoint_path: str,
    config: Dict,
    # Continues the surrounding statement or collection definition.
    split_name: str,
) -> List[Dict]:
    """基于已存在的划分文件构建评估样本列表。"""
    checkpoint = load_torch(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("config", {})
    # Validates `not checkpoint_config` and raises `ValueError` when the input cannot be processed safely.
    if not checkpoint_config:
        raise ValueError("检查点缺少 config，无法按 split 构建评估样本列表")

    runtime_seed_config = copy.deepcopy(checkpoint_config)
    configured_general = dict((config or {}).get("general", {}))
    if configured_general:
        checkpoint_general = runtime_seed_config.setdefault("general", {})
        checkpoint_general.update(configured_general)
    runtime_seed_config["evaluate"] = dict((config or {}).get("evaluate", {}))
    # Stores the `runtime_config` for use by the remaining steps in this flow.
    runtime_config = resolve_runtime_paths(runtime_seed_config, project_root=PROJECT_ROOT)
    processed_root = runtime_config.get("general", {}).get("train_processed_dir")
    if not processed_root:
        raise ValueError("检查点缺少 general.train_processed_dir，无法按 split 评估")

    # Stores the `split_csv` for use by the remaining steps in this flow.
    split_csv = os.path.join(processed_root, f"{split_name}.csv")
    if not os.path.exists(split_csv):
        raise FileNotFoundError(f"评估样本划分文件不存在: {split_csv}")

    task_type = str(runtime_config.get("task", {}).get("type", "field")).lower()
    # Stores the `sample_rows` for use by the remaining steps in this flow.
    sample_rows = load_sample_rows(
        split_csv,
        task_type=task_type,
        mode=runtime_config.get("task", {}).get("mode"),
    # Completes the multi-line `load_sample_rows` arguments or elements.
    )
    samples_evaluate: List[Dict] = []
    for item in sample_rows:
        sample_order = int(item["order"])
        # Checks `task_type == "scalar"` before entering this conditional branch.
        if task_type == "scalar":
            processed_path = os.path.join(processed_root, split_name, f"sample_{sample_order}.pt")
        else:
            processed_path = os.path.join(processed_root, f"sample_{sample_order}.pt")

        # Stores the `sample_item` for use by the remaining steps in this flow.
        sample_item = dict(item)
        sample_item["source_file"] = (
            sample_item.get("resolved_path") or sample_item.get("file_path")
        )
        # Checks `os.path.exists(processed_path)` before entering this conditional branch.
        if os.path.exists(processed_path):
            sample_item["processed_path"] = processed_path
        samples_evaluate.append(sample_item)
    return samples_evaluate


# 将内部评估结果收口为后端调用契约。
def _normalize_evaluate_result(
    evaluate_info: Dict,
    samples_evaluated: List[Dict],
) -> Dict:
    """将内部评估结果收口为后端调用契约。"""
    # Initializes the `normalized_samples` as a collection for values from this processing stage.
    normalized_samples: List[Dict] = []
    for item in samples_evaluated:
        normalized_item = {
            "order": int(item["order"]),
            # Completes the multi-line `dict` arguments or elements.
            "params": dict(item.get("params") or {}),
            "source_file": (
                item.get("source_file")
                or item.get("resolved_path")
                # Completes the multi-line `item.get` arguments or elements.
                or item.get("file_path")
            ),
            "evaluated_result_path": item.get("evaluated_result_path"),
        }
        # Checks `item.get("target_result_path")` before entering this conditional branch.
        if item.get("target_result_path"):
            normalized_item["target_result_path"] = item.get("target_result_path")
        normalized_samples.append(normalized_item)

    return {
        # Completes the multi-line `dict` arguments or elements.
        "metrics": dict(evaluate_info.get("metrics") or {}),
        "scatter_data": copy.deepcopy(
            evaluate_info.get("scatter_data")
            or {
                # Completes the multi-line `collection` arguments or elements.
                "overall": {"predictions": [], "targets": []},
                "by_variable": {},
            }
        ),
        # Completes the multi-line `copy.deepcopy` arguments or elements.
        "writeback_report": copy.deepcopy(evaluate_info.get("writeback_report") or {}),
        "samples_evaluated": normalized_samples,
    }


def _persist_checkpoint_evaluate_result(
    # Continues the surrounding statement or collection definition.
    checkpoint_path: str,
    evaluate_result: Dict,
    result_field: str,
    extra_info: Optional[Dict],
# Continues the surrounding statement or collection definition.
) -> None:
    """在评估完成后回写指定评估结果，并按需覆盖 checkpoint 中的 extra_info。"""
    checkpoint = load_torch(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint[result_field] = copy.deepcopy(evaluate_result)
    if extra_info is not None:
        # Validates `not isinstance(extra_info, dict)` and raises `TypeError` when the input cannot be processed safely.
        if not isinstance(extra_info, dict):
            raise TypeError("extra_info 必须为 dict 或 None")
        checkpoint["extra_info"] = copy.deepcopy(extra_info)
    atomic_torch_save(checkpoint, checkpoint_path)


# 尽力释放单次评估链路持有的大对象引用与 CUDA 缓存。
def _release_evaluate_runtime_resources(
    runner: Optional[EvaluationRunner],
) -> None:
    """尽力释放单次评估链路持有的大对象引用与 CUDA 缓存。"""
    if runner is not None:
        # Stores the data processor for use by the remaining steps in this flow.
        processor = getattr(runner, "processor", None)
        if processor is not None and hasattr(processor, "cached_topology"):
            try:
                processor.cached_topology = None
            # Continues the surrounding statement or collection definition.
            except Exception:
                pass

        for attr_name in (
            "model",
            # Completes the multi-line `collection` arguments or elements.
            "processor",
            "checkpoint",
            "runtime_config",
            "config",
            # Completes the multi-line `collection` arguments or elements.
            "norm_params",
        ):
            if hasattr(runner, attr_name):
                try:
                    # Calls `setattr` to perform the operation required at this stage.
                    setattr(runner, attr_name, None)
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


def run_evaluate(
    # Continues the surrounding statement or collection definition.
    config: Dict,
    checkpoint_path: str,
    run_dir: str,
    samples_evaluate: Optional[List[Dict]] = None,
    # Continues the surrounding statement or collection definition.
    extra_info: Optional[Dict] = None,
) -> Dict:
    """执行评估并返回面向后端的结果对象。

    未提供样本表或样本列表时，按 ``evaluate.split`` 读取训练期划分；
    提供 ``evaluate.evaluate_samples`` 或显式样本列表时，按独立 test 集评估。
    """
    checkpoint_path = os.path.abspath(str(checkpoint_path))
    if not os.path.exists(checkpoint_path):
        # Raises `FileNotFoundError` to report why this path cannot continue.
        raise FileNotFoundError(f"模型检查点不存在: {checkpoint_path}")

    evaluate_cfg = copy.deepcopy(config)
    normalized_run_dir = str(run_dir or "").strip()
    if not normalized_run_dir:
        # Raises `ValueError` to report why this path cannot continue.
        raise ValueError("run_dir 不能为空")
    evaluate_cfg = apply_evaluate_run_dir(
        evaluate_cfg,
        run_dir=normalized_run_dir,
    # Completes the multi-line `apply_evaluate_run_dir` arguments or elements.
    )
    evaluate_cfg = clear_evaluate_default_paths(evaluate_cfg)
    evaluate_cfg = resolve_runtime_paths(evaluate_cfg, project_root=PROJECT_ROOT)
    evaluate_section = evaluate_cfg.setdefault("evaluate", {})
    # Stores the `split_name` for use by the remaining steps in this flow.
    split_name = str(evaluate_section.get("split", "test") or "test").strip().lower()
    configured_samples_path = str(
        evaluate_section.get("evaluate_samples", "") or ""
    ).strip()
    if configured_samples_path:
        # A user-provided sample table is an independent test set.
        split_name = "test"
        evaluate_section["split"] = split_name
    configured_experiment = str(evaluate_section.get("experiment", "") or "").strip()

    output_dir = evaluate_section.get("output_dir")
    configured_run_dir = (
        # Completes the multi-line `resolve_evaluate_run_dir_from_output` arguments or elements.
        resolve_evaluate_run_dir_from_output(output_dir, split_name)
        if not should_reset_runtime_path(output_dir)
        else None
    )
    # Stores the `checkpoint_experiment` for use by the remaining steps in this flow.
    checkpoint_experiment = os.path.dirname(os.path.dirname(checkpoint_path))
    evaluate_experiment = configured_experiment or configured_run_dir or checkpoint_experiment
    evaluate_section["experiment"] = evaluate_experiment

    general_section = evaluate_cfg.setdefault("general", {})
    general_section["experiment"] = evaluate_experiment
    general_section["train_samples"] = os.path.join(
        evaluate_experiment, "data", "raw", "samples.csv"
    )
    general_section["train_processed_dir"] = os.path.join(
        evaluate_experiment, "data", "processed"
    )

    if should_reset_runtime_path(output_dir):
        # Stores the `evaluate_output_root` for use by the remaining steps in this flow.
        evaluate_output_root = os.path.join(evaluate_experiment, "evaluate")
        if samples_evaluate is None and not configured_samples_path:
            evaluate_section["output_dir"] = os.path.join(evaluate_output_root, split_name)
        else:
            # Stores the `evaluate_section['output_dir']` for use by the remaining steps in this flow.
            evaluate_section["output_dir"] = os.path.join(evaluate_output_root, "test")

    uses_configured_split = samples_evaluate is None and not configured_samples_path
    if configured_samples_path and samples_evaluate is None:
        if not os.path.isfile(configured_samples_path):
            raise FileNotFoundError(f"评估样本表不存在: {configured_samples_path}")
        checkpoint = load_torch(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_config = checkpoint.get("config", {})
        task_type = str(checkpoint_config.get("task", {}).get("type", "field")).lower()
        task_mode = checkpoint_config.get("task", {}).get("mode")
        del checkpoint, checkpoint_config
        samples_evaluate = load_sample_rows(
            configured_samples_path,
            task_type=task_type,
            mode=task_mode,
        )
        for item in samples_evaluate:
            item["source_file"] = item.get("resolved_path") or item.get("file_path") or ""

    if uses_configured_split:
        samples_evaluate = _build_samples_evaluate_from_split(
            # Completes the multi-line `_build_samples_evaluate_from_split` arguments or elements.
            checkpoint_path=checkpoint_path,
            config=evaluate_cfg,
            split_name=split_name,
        )

    # Stores the `result_field` for use by the remaining steps in this flow.
    result_field = (
        f"evaluate_result_{split_name}"
        if uses_configured_split
        else "evaluate_result_test"
    # Continues the surrounding statement or collection definition.
    )

    runner: Optional[EvaluationRunner] = None
    try:
        runner = EvaluationRunner(
            # Completes the multi-line `EvaluationRunner` arguments or elements.
            checkpoint_path=checkpoint_path,
            evaluate_config=evaluate_cfg,
            project_root=PROJECT_ROOT,
            logger=None,
        # Completes the multi-line `EvaluationRunner` arguments or elements.
        )
        evaluate_info, _, samples_evaluated = runner.run(samples_evaluate=samples_evaluate)
        normalized_evaluate_result = _normalize_evaluate_result(
            evaluate_info=evaluate_info,
            # Completes the multi-line `_normalize_evaluate_result` arguments or elements.
            samples_evaluated=samples_evaluated,
        )
        _persist_checkpoint_evaluate_result(
            checkpoint_path=checkpoint_path,
            # Completes the multi-line `_persist_checkpoint_evaluate_result` arguments or elements.
            evaluate_result=normalized_evaluate_result,
            result_field=result_field,
            extra_info=extra_info,
        )
        # Returns `normalized_evaluate_result` to the caller for the next stage.
        return normalized_evaluate_result
    finally:
        _release_evaluate_runtime_resources(runner)
