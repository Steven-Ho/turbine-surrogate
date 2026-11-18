import copy
import json
import logging
import os
# Imports `Path` from `pathlib` for use in this module.
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from .base_data import BaseData
# Imports `FieldSampleLoader, ScalarSampleLoader, ScalarTableSource` from `loaders` for use in this module.
from .loaders import FieldSampleLoader, ScalarSampleLoader, ScalarTableSource
from .sample_statistics import (
    extract_coord_ranges_from_sample,
    calculate_cumulative_coord_ranges,
# Continues the surrounding statement or collection definition.
)
from ..processors.field.field_processor import FieldProcessor
from ..readers.sample_map import SampleMap
from ..training.preprocess_cache import save_preprocess_manifest
# Imports `PreprocessResumeManager` from `training.preprocess_resume` for use in this module.
from ..training.preprocess_resume import PreprocessResumeManager
from ..utils.logger import Logger
from ..utils.path_utils import resolve_path_from_anchor
from ..utils.torch_utils import atomic_torch_save, load_torch


# 数据管线编排层。
class DataPipeline:
    """数据管线编排层。

    当前主线中，DataPipeline 主要负责：
    - 从样本表解析源文件路径
    - 调用 FieldSampleLoader 读取原始样本
    - 调用 Processor 处理样本并保存 `.pt`
    - 调用 Processor 完成训练/预测阶段归一化
    """

    @staticmethod
    def _validate_batch_processed_outputs(
        processor,
        # Continues the surrounding statement or collection definition.
        candidate_items,
        processed_list,
    ) -> None:
        if len(processed_list) != len(candidate_items):
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(
                f"{type(processor).__name__}.process_loaded_samples returned "
                f"{len(processed_list)} outputs for {len(candidate_items)} inputs"
            )

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _sync_processor_runtime_state(target_processor, source_processor) -> None:
        """将批量预处理探测阶段得到的状态同步回原处理器实例。"""
        target_processor.__dict__.clear()
        target_processor.__dict__.update(copy.deepcopy(source_processor.__dict__))

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _clone_processor_for_trial(processor):
        """为批量试跑创建继承运行态的新处理器实例。"""
        processor_cls = type(processor)
        trial_processor = processor_cls(copy.deepcopy(processor.config))
        # Adds the current value to `trial_processor.__dict__` for later aggregation.
        trial_processor.__dict__.update(copy.deepcopy(processor.__dict__))
        return trial_processor

    @staticmethod
    def _resolve_logger(logger=None, processor=None, reader=None) -> logging.Logger:
        """解析预处理链路使用的日志对象。"""
        # Returns early when `logger is not None` makes the remaining work inapplicable.
        if logger is not None:
            return logger

        processor_logger = getattr(processor, "logger", None)
        if processor_logger is not None:
            # Returns `processor_logger` to the caller for the next stage.
            return processor_logger

        reader_logger = getattr(reader, "logger", None)
        if reader_logger is not None:
            return reader_logger

        # Returns `` to the caller for the next stage.
        return Logger.get_first_configured_logger(
            ["train", "predict", "evaluate"],
            fallback_name=__name__,
        )

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def call_process_with_reader(
        reader,
        processor: FieldProcessor,
        # Continues the surrounding statement or collection definition.
        sample_table_path: str = None,
        stage: str = "train",
        processed_dir: str = None,
        logger=None,
        # Continues the surrounding statement or collection definition.
        resume_manager: Optional[PreprocessResumeManager] = None,
    ) -> List[str]:
        """兼容新旧签名，调用样本预处理入口。"""
        kwargs = dict(
            reader=reader,
            # Completes the multi-line `dict` arguments or elements.
            processor=processor,
            sample_table_path=sample_table_path,
            stage=stage,
            processed_dir=processed_dir,
            # Completes the multi-line `dict` arguments or elements.
            logger=logger,
            resume_manager=resume_manager,
        )
        while True:
            # Keeps the operation and its recovery path within the same exception boundary.
            try:
                return DataPipeline.process_with_reader(**kwargs)
            except TypeError as exc:
                error_message = str(exc)
                # Checks `"unexpected keyword argument 'resume_manager'" in error_message` before entering this conditional branch.
                if "unexpected keyword argument 'resume_manager'" in error_message:
                    kwargs.pop("resume_manager", None)
                    continue
                if "unexpected keyword argument 'logger'" in error_message:
                    # Calls `kwargs.pop` to perform the operation required at this stage.
                    kwargs.pop("logger", None)
                    continue
                raise

    @staticmethod
    # Defines `_run_batch_preprocess_trial` for  run batch preprocess trial logic.
    def _run_batch_preprocess_trial(
        processor,
        candidate_items,
        stage: str,
        # Continues the surrounding statement or collection definition.
        context: Dict,
    ):
        trial_processor = DataPipeline._clone_processor_for_trial(processor)
        processed_list = trial_processor.process_loaded_samples(
            # Completes the multi-line `trial_processor.process_loaded_samples` arguments or elements.
            [item[0] for item in candidate_items],
            stage=stage,
            context=context,
        )
        # Calls `DataPipeline._validate_batch_processed_outputs` to perform the operation required at this stage.
        DataPipeline._validate_batch_processed_outputs(
            processor=trial_processor,
            candidate_items=candidate_items,
            processed_list=processed_list,
        # Completes the multi-line `DataPipeline._validate_batch_processed_outputs` arguments or elements.
        )
        return trial_processor, processed_list

    @staticmethod
    def _extract_field_mesh_dimension(data) -> Optional[int]:
        """提取 field 样本携带的网格维度元信息。"""
        # Returns early when `not hasattr(data, "get_mesh_dimension")` makes the remaining work inapplicable.
        if not hasattr(data, "get_mesh_dimension"):
            return None
        mesh_dimension = data.get_mesh_dimension()
        if mesh_dimension is None:
            # Returns `None` to the caller for the next stage.
            return None
        return int(mesh_dimension)

    @staticmethod
    def _update_expected_field_mesh_dimension(
        # Continues the surrounding statement or collection definition.
        processor,
        reader,
        data,
        sample_id: int,
        # Continues the surrounding statement or collection definition.
        expected_mesh_dimension: Optional[int],
    ) -> Optional[int]:
        """校验训练样本网格维度一致，并回写到运行时配置。"""
        current_mesh_dimension = DataPipeline._extract_field_mesh_dimension(data)
        if current_mesh_dimension is None:
            # Returns `expected_mesh_dimension` to the caller for the next stage.
            return expected_mesh_dimension

        if expected_mesh_dimension is None:
            expected_mesh_dimension = int(current_mesh_dimension)
            runtime_config = getattr(processor, "config", None)
            # Checks `isinstance(runtime_config, dict)` before entering this conditional branch.
            if isinstance(runtime_config, dict):
                runtime_config.setdefault("task", {})["mesh_dimension"] = expected_mesh_dimension
            reader_config = getattr(reader, "config", None)
            if isinstance(reader_config, dict):
                # Stores the `setdefault('task', {})['mesh_dimension']` for use by the remaining steps in this flow.
                reader_config.setdefault("task", {})["mesh_dimension"] = expected_mesh_dimension
            return expected_mesh_dimension

        if int(current_mesh_dimension) != int(expected_mesh_dimension):
            raise ValueError(
                # Completes the multi-line `ValueError` arguments or elements.
                f"检测到 2D/3D 混合样本，当前训练实验不支持混训。"
                f" sample_id={sample_id}, expect={expected_mesh_dimension}D, actual={current_mesh_dimension}D"
            )
        return expected_mesh_dimension

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _process_batch_with_isolation(
        processor,
        collected_items,
        # Continues the surrounding statement or collection definition.
        stage: str,
        context: Dict,
        logger=None,
    ):
        # Keeps the operation and its recovery path within the same exception boundary.
        try:
            trial_processor, processed_list = DataPipeline._run_batch_preprocess_trial(
                processor=processor,
                candidate_items=collected_items,
                # Completes the multi-line `DataPipeline._run_batch_preprocess_trial` arguments or elements.
                stage=stage,
                context=context,
            )
            DataPipeline._sync_processor_runtime_state(processor, trial_processor)
            # Returns `collected_items, processed_list` to the caller for the next stage.
            return collected_items, processed_list
        except Exception as exc:
            if len(collected_items) <= 1:
                logger.warning("批量预处理失败，且没有可继续尝试的候选样本: %s", str(exc))
                # Returns `[], []` to the caller for the next stage.
                return [], []

            logger.warning(
                "%s 批量预处理失败，开始隔离异常样本: %s",
                type(processor).__name__,
                # Completes the multi-line `str` arguments or elements.
                str(exc),
            )

        valid_items = []
        for data, processed_path, sample_id in collected_items:
            # Derives the `candidate_items` from the current values for downstream processing.
            candidate_items = valid_items + [(data, processed_path, sample_id)]
            try:
                DataPipeline._run_batch_preprocess_trial(
                    processor=processor,
                    # Completes the multi-line `DataPipeline._run_batch_preprocess_trial` arguments or elements.
                    candidate_items=candidate_items,
                    stage=stage,
                    context=context,
                )
                # Stores the `valid_items` for use by the remaining steps in this flow.
                valid_items = candidate_items
            except Exception as exc:
                logger.warning("样本 %s 批量预处理失败，已跳过: %s", sample_id, str(exc))

        if not valid_items:
            # Records the current execution state to make progress and failures traceable.
            logger.warning(
                "%s 在批量预处理隔离后没有剩余有效样本",
                type(processor).__name__,
            )
            # Returns `[], []` to the caller for the next stage.
            return [], []

        final_processor, processed_list = DataPipeline._run_batch_preprocess_trial(
            processor=processor,
            candidate_items=valid_items,
            # Completes the multi-line `DataPipeline._run_batch_preprocess_trial` arguments or elements.
            stage=stage,
            context=context,
        )
        DataPipeline._sync_processor_runtime_state(processor, final_processor)
        # Returns `valid_items, processed_list` to the caller for the next stage.
        return valid_items, processed_list

    @staticmethod
    def process_with_reader(
        reader,
        # Continues the surrounding statement or collection definition.
        processor: FieldProcessor,
        sample_table_path: str = None,
        stage: str = "train",
        processed_dir: str = None,
        # Continues the surrounding statement or collection definition.
        logger=None,
        resume_manager: Optional[PreprocessResumeManager] = None,
    ) -> List[str]:
        """读取源样本，交给处理器处理，并保存为 `.pt` 中间产物。"""
        logger = DataPipeline._resolve_logger(
            # Completes the multi-line `DataPipeline._resolve_logger` arguments or elements.
            logger=logger,
            processor=processor,
            reader=reader,
        )
        # Validates `not sample_table_path` and raises `ValueError` when the input cannot be processed safely.
        if not sample_table_path:
            raise ValueError("sample_table_path is required")

        task_type = getattr(processor, "config", {}).get("task", {}).get(
            "type", getattr(reader, "task_type", None)
        # Completes the multi-line `getattr(processor, 'config', {}).get('task', {}).get` arguments or elements.
        )
        normalized_task_type = (task_type or "").lower()

        if normalized_task_type == "scalar":
            raise ValueError("启元发行包不支持 scalar 标量任务")

        # Validates `normalized_task_type != "field"` and raises `ValueError` when the input cannot be processed safely.
        if normalized_task_type != "field":
            raise ValueError(f"Unsupported task type for process_with_reader: {task_type}")

        return DataPipeline._process_field_with_reader(
            reader=reader,
            # Completes the multi-line `DataPipeline._process_field_with_reader` arguments or elements.
            processor=processor,
            sample_table_path=sample_table_path,
            stage=stage,
            processed_dir=processed_dir,
            # Completes the multi-line `DataPipeline._process_field_with_reader` arguments or elements.
            logger=logger,
            resume_manager=resume_manager,
        )

    @staticmethod
    # Field task processing path.
    def _process_field_with_reader(
        reader,
        processor: FieldProcessor,
        sample_table_path: str,
        # Continues the surrounding statement or collection definition.
        stage: str,
        processed_dir: str,
        logger=None,
        resume_manager: Optional[PreprocessResumeManager] = None,
    # Continues the surrounding statement or collection definition.
    ) -> List[str]:
        """Field task processing path."""
        mode = getattr(processor, "mode", getattr(reader, "mode", None))

        sample_map = SampleMap(sample_table_path)
        file_paths = []
        # Initializes the sample identifiers as a collection for values from this processing stage.
        sample_ids = []
        for sample_id in sample_map.get_all_samples():
            file_path = sample_map.get_file_path(sample_id)
            if not file_path:
                # Handles the data or control-flow case represented by this statement.
                continue
            absolute_file_path = resolve_path_from_anchor(sample_table_path, file_path)
            file_paths.append(absolute_file_path)
            sample_ids.append(sample_id)

        # Validates `not file_paths` and raises `ValueError` when the input cannot be processed safely.
        if not file_paths:
            raise ValueError("No source files were resolved from the sample table")

        if stage not in ["train", "evaluate", "predict"]:
            raise ValueError(f"stage must be 'train', 'evaluate' or 'predict', got: {stage}")

        # Validates `not processed_dir` and raises `ValueError` when the input cannot be processed safely.
        if not processed_dir:
            raise ValueError("processed_dir is required")

        output_dir = Path(processed_dir)
        os.makedirs(output_dir, exist_ok=True)
        # Builds the `processed_paths` using the comprehension's filter and mapping rules.
        processed_paths = [str(output_dir / f"sample_{sample_id}.pt") for sample_id in sample_ids]
        if resume_manager is not None and resume_manager.workflow == "matrix":
            return DataPipeline._process_matrix_with_resume(
                reader=reader,
                # Completes the multi-line `DataPipeline._process_matrix_with_resume` arguments or elements.
                processor=processor,
                file_paths=file_paths,
                sample_ids=sample_ids,
                processed_paths=processed_paths,
                # Completes the multi-line `DataPipeline._process_matrix_with_resume` arguments or elements.
                processed_dir=processed_dir,
                stage=stage,
                mode=mode,
                resume_manager=resume_manager,
                # Completes the multi-line `DataPipeline._process_matrix_with_resume` arguments or elements.
                logger=logger,
            )
        if resume_manager is not None and processor.requires_batch_preprocess():
            logger.info("%s 需要整批预处理，当前运行不启用样本级续处理", type(processor).__name__)
            # Stores the `resume_manager` for use by the remaining steps in this flow.
            resume_manager = None

        if resume_manager is not None:
            if resume_manager.status in {"normalizing", "committing"}:
                return resume_manager.get_active_paths(processed_paths)
            # Calls `resume_manager.register_expected_paths` to perform the operation required at this stage.
            resume_manager.register_expected_paths(processed_paths)
            reusable_paths = set(resume_manager.get_processed_paths(processed_paths))
        else:
            reusable_paths = set()

        # Builds the `completed_initial_ids` using the comprehension's filter and mapping rules.
        completed_initial_ids = [
            sample_id
            for sample_id, processed_path in zip(sample_ids, processed_paths)
            if processed_path in reusable_paths
        # Continues the surrounding statement or collection definition.
        ]
        save_preprocess_manifest(
            processed_dir=processed_dir,
            expected_sample_ids=sample_ids,
            # Completes the multi-line `save_preprocess_manifest` arguments or elements.
            completed_sample_ids=completed_initial_ids,
            processor_type=type(processor).__name__,
            stage=stage,
            status="in_progress",
        # Completes the multi-line `save_preprocess_manifest` arguments or elements.
        )

        logger.info("开始执行预处理: processor=%s stage=%s", type(processor).__name__, stage)
        task_cfg = getattr(processor, "config", {}).get("task", {})
        expected_mesh_dimension = task_cfg.get("mesh_dimension")
        # Keeps the operation and its recovery path within the same exception boundary.
        try:
            expected_mesh_dimension = (
                None if expected_mesh_dimension is None else int(expected_mesh_dimension)
            )
        # Completes the multi-line `collection` arguments or elements.
        except (TypeError, ValueError):
            expected_mesh_dimension = None
        is_predict_autoregressive = (
            stage == "predict"
            # Completes the multi-line `str(getattr(processor, 'temporal_strategy', 'direct')).lower` arguments or elements.
            and str(getattr(processor, "temporal_strategy", "direct")).lower() == "autoregressive"
        )
        is_predict_direct = (
            stage == "predict"
            # Completes the multi-line `str(mode or '').lower` arguments or elements.
            and str(mode or "").lower() == "temporal"
            and str(getattr(processor, "temporal_strategy", "direct")).lower() == "direct"
        )
        input_length = int(task_cfg.get("input_length", 1))
        # Stores the `sample_loader` for use by the remaining steps in this flow.
        sample_loader = FieldSampleLoader(
            reader=reader,
            mode=mode,
            allow_missing_fields=(
                # Completes the multi-line `FieldSampleLoader` arguments or elements.
                stage == "predict"
                and not (is_predict_autoregressive and input_length > 1)
            ),
            temporal_history_steps=(
                # Completes the multi-line `FieldSampleLoader` arguments or elements.
                input_length
                if is_predict_autoregressive
                else (1 if is_predict_direct else None)
            ),
        # Completes the multi-line `FieldSampleLoader` arguments or elements.
        )

        context = processor.build_processing_context(
            stage=stage,
            reader=reader,
            # Completes the multi-line `processor.build_processing_context` arguments or elements.
            file_paths=file_paths,
            sample_ids=sample_ids,
            processed_paths=processed_paths,
            processed_dir=processed_dir,
            # Completes the multi-line `processor.build_processing_context` arguments or elements.
            mode=mode,
            sample_loader=sample_loader.load,
            local_feature_sample_loader=lambda file_path, sample_id: sample_loader.load_local_feature_fields(
                file_path,
                # Completes the multi-line `sample_loader.load_local_feature_fields` arguments or elements.
                sample_id,
                max_time_steps=1,
            ),
            logger=logger,
            # Completes the multi-line `processor.build_processing_context` arguments or elements.
            extract_coord_ranges_from_sample=extract_coord_ranges_from_sample,
            cumulative_coord_range_loader=lambda fps, sids: calculate_cumulative_coord_ranges(
                sample_loader.load,
                fps,
                # Completes the multi-line `calculate_cumulative_coord_ranges` arguments or elements.
                sids,
                logger=logger,
            ),
        )

        # Checks `not processor.requires_batch_preprocess()` before entering this conditional branch.
        if not processor.requires_batch_preprocess():
            processed_paths_result = [
                processed_path for processed_path in processed_paths if processed_path in reusable_paths
            ]
            # Stores the `completed_sample_ids` for use by the remaining steps in this flow.
            completed_sample_ids = list(completed_initial_ids)
            for index, (file_path, sample_id, processed_path) in enumerate(
                zip(file_paths, sample_ids, processed_paths),
                start=1,
            # Completes the multi-line `enumerate` arguments or elements.
            ):
                if processed_path in reusable_paths:
                    logger.info("复用已完成预处理样本: %s/%s", index, len(file_paths))
                    continue
                # Records the current execution state to make progress and failures traceable.
                logger.info("读取样本: %s/%s", index, len(file_paths))
                try:
                    data = sample_loader.load(file_path, sample_id)
                except Exception as exc:
                    # Records the current execution state to make progress and failures traceable.
                    logger.warning(
                        "读取样本失败: index=%s sample_id=%s reason=%s",
                        index,
                        sample_id,
                        # Completes the multi-line `str` arguments or elements.
                        str(exc),
                    )
                    continue

                expected_mesh_dimension = DataPipeline._update_expected_field_mesh_dimension(
                    # Completes the multi-line `DataPipeline._update_expected_field_mesh_dimension` arguments or elements.
                    processor=processor,
                    reader=reader,
                    data=data,
                    sample_id=sample_id,
                    # Completes the multi-line `DataPipeline._update_expected_field_mesh_dimension` arguments or elements.
                    expected_mesh_dimension=expected_mesh_dimension,
                )

                try:
                    if stage == "predict":
                        # Stores the `processed_data` for use by the remaining steps in this flow.
                        processed_data = processor.process_predict_sample(data, context=context)
                    else:
                        processed_data = processor.process_single_sample(data, context=context)
                    processor.save_to_pt(processed_data, processed_path)
                    # Checks `resume_manager is not None` before entering this conditional branch.
                    if resume_manager is not None:
                        processor.save_processing_state(
                            stage=stage,
                            processed_dir=processed_dir,
                            # Completes the multi-line `collection` arguments or elements.
                            processed_paths=processed_paths_result + [processed_path],
                        )
                        resume_manager.mark_processed(processed_path)
                    processed_paths_result.append(processed_path)
                    # Adds the current value to `completed_sample_ids` for later aggregation.
                    completed_sample_ids.append(sample_id)
                    save_preprocess_manifest(
                        processed_dir=processed_dir,
                        expected_sample_ids=sample_ids,
                        # Completes the multi-line `save_preprocess_manifest` arguments or elements.
                        completed_sample_ids=completed_sample_ids,
                        processor_type=type(processor).__name__,
                        stage=stage,
                        status="in_progress",
                    # Completes the multi-line `save_preprocess_manifest` arguments or elements.
                    )
                except Exception as exc:
                    logger.warning("样本 %s 处理失败，已跳过: %s", sample_id, str(exc))
                finally:
                    # Handles the data or control-flow case represented by this statement.
                    del data

            if not processed_paths_result:
                logger.warning("预处理未收集到任何有效样本: processor=%s", type(processor).__name__)
                return []

            # Calls `processor.save_processing_state` to perform the operation required at this stage.
            processor.save_processing_state(
                stage=stage,
                processed_dir=processed_dir,
                processed_paths=processed_paths_result,
            # Completes the multi-line `processor.save_processing_state` arguments or elements.
            )
            save_preprocess_manifest(
                processed_dir=processed_dir,
                expected_sample_ids=sample_ids,
                # Completes the multi-line `save_preprocess_manifest` arguments or elements.
                completed_sample_ids=completed_sample_ids,
                processor_type=type(processor).__name__,
                stage=stage,
                status="completed",
            # Completes the multi-line `save_preprocess_manifest` arguments or elements.
            )
            return processed_paths_result

        collected_items = []
        for index, (file_path, sample_id, processed_path) in enumerate(
            # Completes the multi-line `zip` arguments or elements.
            zip(file_paths, sample_ids, processed_paths),
            start=1,
        ):
            logger.info("读取样本: %s/%s", index, len(file_paths))
            # Keeps the operation and its recovery path within the same exception boundary.
            try:
                data = sample_loader.load(file_path, sample_id)
            except Exception as exc:
                logger.warning(
                    # Completes the multi-line `logger.warning` arguments or elements.
                    "读取样本失败: index=%s sample_id=%s reason=%s",
                    index,
                    sample_id,
                    str(exc),
                # Completes the multi-line `logger.warning` arguments or elements.
                )
                continue
            expected_mesh_dimension = DataPipeline._update_expected_field_mesh_dimension(
                processor=processor,
                # Completes the multi-line `DataPipeline._update_expected_field_mesh_dimension` arguments or elements.
                reader=reader,
                data=data,
                sample_id=sample_id,
                expected_mesh_dimension=expected_mesh_dimension,
            # Completes the multi-line `DataPipeline._update_expected_field_mesh_dimension` arguments or elements.
            )
            collected_items.append((data, processed_path, sample_id))

        if not collected_items:
            logger.warning("预处理未收集到任何有效样本: processor=%s", type(processor).__name__)
            # Returns `[]` to the caller for the next stage.
            return []

        valid_items, processed_list = DataPipeline._process_batch_with_isolation(
            processor=processor,
            collected_items=collected_items,
            # Completes the multi-line `DataPipeline._process_batch_with_isolation` arguments or elements.
            stage=stage,
            context=context,
            logger=logger,
        )
        # Checks `not valid_items` before entering this conditional branch.
        if not valid_items:
            logger.warning("批量预处理后没有剩余有效样本: processor=%s", type(processor).__name__)
            return []

        processed_paths_result = []
        # Initializes the `completed_sample_ids` as a collection for values from this processing stage.
        completed_sample_ids = []
        for (data, processed_path, sample_id), processed_data in zip(valid_items, processed_list):
            processor.save_to_pt(processed_data, processed_path)
            processed_paths_result.append(processed_path)
            # Adds the current value to `completed_sample_ids` for later aggregation.
            completed_sample_ids.append(sample_id)
            del data

        processor.save_processing_state(
            stage=stage,
            # Completes the multi-line `processor.save_processing_state` arguments or elements.
            processed_dir=processed_dir,
            processed_paths=processed_paths_result,
        )
        save_preprocess_manifest(
            # Completes the multi-line `save_preprocess_manifest` arguments or elements.
            processed_dir=processed_dir,
            expected_sample_ids=sample_ids,
            completed_sample_ids=completed_sample_ids,
            processor_type=type(processor).__name__,
            # Completes the multi-line `save_preprocess_manifest` arguments or elements.
            stage=stage,
            status="completed",
        )
        return processed_paths_result

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _select_pod_fit_paths(processed_paths, processed_dir):
        """Fit POD on the actual training split, never validation targets."""
        train_ids = set(SampleMap(str(Path(processed_dir) / "train.csv")).get_all_samples())
        selected = [path for path in processed_paths
                    if int(Path(path).stem.split("_")[-1]) in train_ids]
        if not selected:
            raise ValueError("No training snapshots available for POD fitting")
        return selected

    @staticmethod
    def _process_matrix_with_resume(
        reader,
        processor,
        # Continues the surrounding statement or collection definition.
        file_paths: List[str],
        sample_ids: List[int],
        processed_paths: List[str],
        processed_dir: str,
        # Continues the surrounding statement or collection definition.
        stage: str,
        mode: str,
        resume_manager: PreprocessResumeManager,
        logger,
    # Continues the surrounding statement or collection definition.
    ) -> List[str]:
        """以快照收集、POD 拟合和系数编码三阶段处理 MatrixProcessor。"""
        if stage != "train":
            raise ValueError("MatrixProcessor 的可恢复预处理当前仅支持 train 阶段")

        task_cfg = getattr(processor, "config", {}).get("task", {})
        # Reads `mesh_dimension` from the mapping into the `expected_mesh_dimension`.
        expected_mesh_dimension = task_cfg.get("mesh_dimension")
        try:
            expected_mesh_dimension = (
                None if expected_mesh_dimension is None else int(expected_mesh_dimension)
            # Continues the surrounding statement or collection definition.
            )
        except (TypeError, ValueError):
            expected_mesh_dimension = None
        sample_loader = FieldSampleLoader(
            # Completes the multi-line `FieldSampleLoader` arguments or elements.
            reader=reader,
            mode=mode,
            allow_missing_fields=False,
            temporal_history_steps=None,
        # Completes the multi-line `FieldSampleLoader` arguments or elements.
        )
        path_to_item = {
            processed_path: (file_path, sample_id)
            for file_path, sample_id, processed_path in zip(file_paths, sample_ids, processed_paths)
        # Continues the surrounding statement or collection definition.
        }

        if resume_manager.status == "matrix_collect":
            resume_manager.register_expected_paths(processed_paths)
            reusable_paths = set(resume_manager.get_collected_paths(processed_paths))
            # Builds the `collected_paths` using the comprehension's filter and mapping rules.
            collected_paths = [path for path in processed_paths if path in reusable_paths]
            for index, (file_path, sample_id, processed_path) in enumerate(
                zip(file_paths, sample_ids, processed_paths),
                start=1,
            # Completes the multi-line `enumerate` arguments or elements.
            ):
                if processed_path in reusable_paths:
                    logger.info("复用已收集的 POD 快照: %s/%s", index, len(file_paths))
                    continue
                # Records the current execution state to make progress and failures traceable.
                logger.info("收集 POD 快照: %s/%s", index, len(file_paths))
                data = None
                try:
                    data = sample_loader.load(file_path, sample_id)
                    # Stores the `expected_mesh_dimension` for use by the remaining steps in this flow.
                    expected_mesh_dimension = DataPipeline._update_expected_field_mesh_dimension(
                        processor=processor,
                        reader=reader,
                        data=data,
                        # Completes the multi-line `DataPipeline._update_expected_field_mesh_dimension` arguments or elements.
                        sample_id=sample_id,
                        expected_mesh_dimension=expected_mesh_dimension,
                    )
                    snapshot = processor.extract_pod_snapshot(data)
                # Continues the surrounding statement or collection definition.
                except Exception as exc:
                    logger.warning("样本 %s 的 POD 快照收集失败，已跳过: %s", sample_id, str(exc))
                    continue
                finally:
                    # Checks `data is not None` before entering this conditional branch.
                    if data is not None:
                        del data

                snapshot_path = resume_manager.get_matrix_snapshot_path(processed_path)
                try:
                    # 暂存目录属于预处理运行态；写入失败不能按坏样本跳过，
                    # 否则会掩盖为后续“无样本可归一化”的连锁错误。
                    # Calls `os.makedirs` to perform the operation required at this stage.
                    os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
                    atomic_torch_save(snapshot, snapshot_path)
                    resume_manager.mark_collected(processed_path)
                except Exception as exc:
                    # Raises `RuntimeError` to report why this path cannot continue.
                    raise RuntimeError(
                        f"样本 {sample_id} 的 POD 快照持久化失败: {snapshot_path}"
                    ) from exc
                collected_paths.append(processed_path)

            # Checks `not collected_paths` before entering this conditional branch.
            if not collected_paths:
                logger.warning("MatrixProcessor 未收集到可用于 POD 的有效样本")
                return []
            resume_manager.begin_matrix_fit(collected_paths)

        # Stores the `active_paths` for use by the remaining steps in this flow.
        active_paths = resume_manager.get_active_paths(processed_paths)
        if resume_manager.status == "matrix_fit":
            fit_paths = DataPipeline._select_pod_fit_paths(active_paths, processed_dir)
            logger.info("仅从 %s 个训练快照拟合 POD 模态，验证样本仅编码", len(fit_paths))
            snapshot_paths = [resume_manager.get_matrix_snapshot_path(path) for path in fit_paths]
            Path(processed_dir, "pod_fit_samples.json").write_text(
                json.dumps({"sample_ids": [int(Path(path).stem.split("_")[-1]) for path in fit_paths]}, indent=2)
            )
            processor.fit_pod_from_snapshot_paths(snapshot_paths)
            # Calls `processor.save_pod_modes` to perform the operation required at this stage.
            processor.save_pod_modes(resume_manager.get_pending_modes_path())
            resume_manager.begin_matrix_encode()

        if resume_manager.status == "matrix_encode":
            load_modes = getattr(processor, "load_pod_modes", None)
            # Checks `` before entering this conditional branch.
            if not getattr(processor, "pod_modes", None) and callable(load_modes):
                load_modes(resume_manager.get_pending_modes_path())

            reusable_paths = set(resume_manager.get_processed_paths(active_paths))
            processed_paths_result = [path for path in active_paths if path in reusable_paths]
            # Builds the `completed_sample_ids` using the comprehension's filter and mapping rules.
            completed_sample_ids = [
                path_to_item[path][1]
                for path in active_paths
                if path in reusable_paths
            # Continues the surrounding statement or collection definition.
            ]
            save_preprocess_manifest(
                processed_dir=processed_dir,
                expected_sample_ids=[path_to_item[path][1] for path in active_paths],
                # Completes the multi-line `save_preprocess_manifest` arguments or elements.
                completed_sample_ids=completed_sample_ids,
                processor_type=type(processor).__name__,
                stage=stage,
                status="in_progress",
            # Completes the multi-line `save_preprocess_manifest` arguments or elements.
            )
            for index, processed_path in enumerate(active_paths, start=1):
                if processed_path in reusable_paths:
                    logger.info("复用已完成 POD 系数编码: %s/%s", index, len(active_paths))
                    # Handles the data or control-flow case represented by this statement.
                    continue
                file_path, sample_id = path_to_item[processed_path]
                logger.info("编码 POD 系数: %s/%s", index, len(active_paths))
                try:
                    # Stores the current data for use by the remaining steps in this flow.
                    data = sample_loader.load(file_path, sample_id)
                    expected_mesh_dimension = DataPipeline._update_expected_field_mesh_dimension(
                        processor=processor,
                        reader=reader,
                        # Completes the multi-line `DataPipeline._update_expected_field_mesh_dimension` arguments or elements.
                        data=data,
                        sample_id=sample_id,
                        expected_mesh_dimension=expected_mesh_dimension,
                    )
                    # Stores the `processed_data` for use by the remaining steps in this flow.
                    processed_data = processor.encode_sample_with_pod(data)
                    processor.save_to_pt(processed_data, processed_path)
                    resume_manager.mark_processed(processed_path)
                    processed_paths_result.append(processed_path)
                    # Adds the current value to `completed_sample_ids` for later aggregation.
                    completed_sample_ids.append(sample_id)
                    save_preprocess_manifest(
                        processed_dir=processed_dir,
                        expected_sample_ids=[path_to_item[path][1] for path in active_paths],
                        # Completes the multi-line `save_preprocess_manifest` arguments or elements.
                        completed_sample_ids=completed_sample_ids,
                        processor_type=type(processor).__name__,
                        stage=stage,
                        status="in_progress",
                    # Completes the multi-line `save_preprocess_manifest` arguments or elements.
                    )
                except Exception as exc:
                    logger.warning("样本 %s 的 POD 系数编码失败，已跳过: %s", sample_id, str(exc))
                finally:
                    # Checks `"data" in locals()` before entering this conditional branch.
                    if "data" in locals():
                        del data

            save_preprocess_manifest(
                processed_dir=processed_dir,
                # Completes the multi-line `save_preprocess_manifest` arguments or elements.
                expected_sample_ids=[path_to_item[path][1] for path in active_paths],
                completed_sample_ids=completed_sample_ids,
                processor_type=type(processor).__name__,
                stage=stage,
                # Completes the multi-line `save_preprocess_manifest` arguments or elements.
                status="completed",
            )
            return processed_paths_result

        if resume_manager.status in {"normalizing", "committing"}:
            # Returns `active_paths` to the caller for the next stage.
            return active_paths
        raise RuntimeError(f"MatrixProcessor 预处理恢复状态非法: {resume_manager.status}")

    @staticmethod
    def _process_scalar_with_reader(
        # Continues the surrounding statement or collection definition.
        reader,
        processor,
        sample_table_path: str,
        stage: str,
        # Continues the surrounding statement or collection definition.
        processed_dir: str,
        logger=None,
        resume_manager: Optional[PreprocessResumeManager] = None,
    ) -> List[str]:
        """Scalar task processing path where the table itself is the raw data."""
        # Validates `stage not in ["train", "evaluate", "predict"]` and raises `ValueError` when the input cannot be processed safely.
        if stage not in ["train", "evaluate", "predict"]:
            raise ValueError(f"stage must be 'train', 'evaluate' or 'predict', got: {stage}")

        if not processed_dir:
            raise ValueError("processed_dir is required")

        # Stores the output directory for use by the remaining steps in this flow.
        output_dir = Path(processed_dir)
        os.makedirs(output_dir, exist_ok=True)

        if str(getattr(processor, "mode", "")).lower() == "steady":
            loader = ScalarSampleLoader(reader)
            # Stores the `base_data` for use by the remaining steps in this flow.
            base_data = loader.load(sample_table_path, sample_id=None)
            scalar_data = base_data.get_scalar_data()
            if not hasattr(scalar_data, "iloc"):
                raise ValueError("steady scalar 预处理要求 reader 返回表格型数据")

            # Builds the `all_processed_paths` using the comprehension's filter and mapping rules.
            all_processed_paths = [
                str(output_dir / f"sample_{sample_id}.pt")
                for sample_id in range(len(scalar_data))
            ]
            # Checks `resume_manager is not None` before entering this conditional branch.
            if resume_manager is not None:
                if resume_manager.status in {"normalizing", "committing"}:
                    return resume_manager.get_active_paths(all_processed_paths)
                resume_manager.register_expected_paths(all_processed_paths)
                # Stores the `reusable_paths` for use by the remaining steps in this flow.
                reusable_paths = set(resume_manager.get_processed_paths(all_processed_paths))
            else:
                reusable_paths = set()
            processed_paths_result = [
                # Continues the surrounding statement or collection definition.
                processed_path
                for processed_path in all_processed_paths
                if processed_path in reusable_paths
            ]
            # Visits each item in `range(len(scalar_data))` and applies the current processing step.
            for sample_id in range(len(scalar_data)):
                processed_path = str(output_dir / f"sample_{sample_id}.pt")
                if processed_path in reusable_paths:
                    logger.info("复用已完成标量预处理样本: sample_id=%s", sample_id)
                    # Handles the data or control-flow case represented by this statement.
                    continue
                logger.info("读取标量样本行: sample_id=%s row_index=%s", sample_id, sample_id)
                try:
                    row_df = scalar_data.iloc[[sample_id]].reset_index(drop=True)
                    # Stores the `sample_map` for use by the remaining steps in this flow.
                    sample_map = dict(base_data.get_sample_map() or {})
                    sample_map["sample_order"] = sample_id
                    sample_map["row_index"] = sample_id
                    row_data = BaseData(
                        # Completes the multi-line `BaseData` arguments or elements.
                        data=row_df,
                        task_type="scalar",
                        mode="steady",
                        sample_map=sample_map,
                    # Completes the multi-line `BaseData` arguments or elements.
                    )
                    if stage == "predict":
                        process_predict_sample = getattr(processor, "process_predict_sample", None)
                        if callable(process_predict_sample):
                            # Stores the `processed_data` for use by the remaining steps in this flow.
                            processed_data = process_predict_sample(row_data)
                        else:
                            processed_data = processor.process(row_data)
                    else:
                        # Stores the `processed_data` for use by the remaining steps in this flow.
                        processed_data = processor.process(row_data)
                    processor.save_to_pt(processed_data, processed_path)
                    if resume_manager is not None:
                        resume_manager.mark_processed(processed_path)
                    # Adds the current value to `processed_paths_result` for later aggregation.
                    processed_paths_result.append(processed_path)
                except Exception as exc:
                    logger.warning("标量样本行 %s 处理失败，已跳过: %s", sample_id, str(exc))

            return processed_paths_result

        # Stores the `source` for use by the remaining steps in this flow.
        source = ScalarTableSource(sample_table_path)
        units = source.resolve_units()
        if not units:
            raise ValueError("No scalar processing units were resolved from the table path")

        # Stores the `loader` for use by the remaining steps in this flow.
        loader = ScalarSampleLoader(reader)
        all_processed_paths = [
            str(output_dir / f"sample_{sample_id}.pt")
            for sample_id, _ in units
        # Continues the surrounding statement or collection definition.
        ]
        if resume_manager is not None:
            if resume_manager.status in {"normalizing", "committing"}:
                return resume_manager.get_active_paths(all_processed_paths)
            # Calls `resume_manager.register_expected_paths` to perform the operation required at this stage.
            resume_manager.register_expected_paths(all_processed_paths)
            reusable_paths = set(resume_manager.get_processed_paths(all_processed_paths))
        else:
            reusable_paths = set()
        # Builds the `processed_paths_result` using the comprehension's filter and mapping rules.
        processed_paths_result = [
            processed_path
            for processed_path in all_processed_paths
            if processed_path in reusable_paths
        # Continues the surrounding statement or collection definition.
        ]

        for sample_id, table_path in units:
            processed_path = str(output_dir / f"sample_{sample_id}.pt")
            if processed_path in reusable_paths:
                # Records the current execution state to make progress and failures traceable.
                logger.info("复用已完成标量预处理样本: sample_id=%s", sample_id)
                continue
            logger.info("读取标量样本: sample_id=%s path=%s", sample_id, table_path)
            try:
                # Stores the current data for use by the remaining steps in this flow.
                data = loader.load(table_path, sample_id)
                if stage == "predict":
                    process_predict_sample = getattr(processor, "process_predict_sample", None)
                    if callable(process_predict_sample):
                        # Stores the `processed_data` for use by the remaining steps in this flow.
                        processed_data = process_predict_sample(data)
                    else:
                        processed_data = processor.process(data)
                else:
                    # Stores the `processed_data` for use by the remaining steps in this flow.
                    processed_data = processor.process(data)
                processor.save_to_pt(processed_data, processed_path)
                if resume_manager is not None:
                    resume_manager.mark_processed(processed_path)
                # Adds the current value to `processed_paths_result` for later aggregation.
                processed_paths_result.append(processed_path)
            except Exception as exc:
                logger.warning("标量样本 %s 处理失败，已跳过: %s", sample_id, str(exc))

        return processed_paths_result

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _apply_normalization(
        processed_paths: List[str],
        processor,
        # Continues the surrounding statement or collection definition.
        norm_params: Dict,
        norm_type: str,
        normalize_y: bool,
    ) -> None:
        """Apply processor-driven normalization to processed samples."""
        # Visits each item in `processed_paths` and applies the current processing step.
        for processed_path in processed_paths:
            data = load_torch(processed_path, weights_only=False)
            if normalize_y:
                normalized_data = processor.normalize_processed_sample(
                    # Completes the multi-line `processor.normalize_processed_sample` arguments or elements.
                    sample=data,
                    norm_params=norm_params,
                    norm_type=norm_type,
                    normalize_y=True,
                # Completes the multi-line `processor.normalize_processed_sample` arguments or elements.
                )
            else:
                normalized_data = processor.normalize_predict_sample(
                    sample=data,
                    # Completes the multi-line `processor.normalize_predict_sample` arguments or elements.
                    norm_params=norm_params,
                    norm_type=norm_type,
                )
            atomic_torch_save(normalized_data, processed_path)

    @staticmethod
    def _compute_mape_target_scales(train_processed_paths, processor, norm_params):
        """Compute exact per-variable abs-target p95 scales from raw train samples."""
        y_stats = norm_params.get("y") or {}
        matrix_specs = {
            name: item["slice"]
            for name, item in y_stats.items()
            if isinstance(item, dict) and item.get("slice")
        } if isinstance(y_stats, dict) else {}

        task_cfg = (getattr(processor, "config", {}) or {}).get("task", {}) or {}
        variable_names = (
            task_cfg.get("effective_variables")
            or task_cfg.get("variables")
            or getattr(processor, "var_names", None)
            or []
        )
        if isinstance(variable_names, dict):
            variable_names = list(variable_names)
        variable_names = [str(name) for name in variable_names]
        values_by_variable = {}

        for processed_path in train_processed_paths:
            sample = load_torch(processed_path, map_location="cpu", weights_only=False)
            target = sample.get("y")
            if not isinstance(target, torch.Tensor) or target.numel() == 0:
                continue
            target = target.detach().float()

            if matrix_specs:
                coeff_rows = target.reshape(1, -1) if target.ndim == 1 else target.reshape(-1, target.shape[-1])
                for coeff_row in coeff_rows:
                    reconstructed = processor._reconstruct_single_step(
                        coeff_row.detach().cpu().numpy()
                    )
                    for name in matrix_specs:
                        values = torch.as_tensor(reconstructed[name]).reshape(-1).abs()
                        if values.numel():
                            values_by_variable.setdefault(name, []).append(values)
                continue

            processor_type = str(norm_params.get("processor_type", ""))
            if processor_type == "ImageVoxelProcessor":
                if target.ndim == 5:
                    _, channel_axis = processor._resolve_temporal_layout(sample, target)
                else:
                    channel_axis = 0
                channel_count = int(target.shape[channel_axis])
                mask_channel = int(norm_params.get("x", {}).get("mask_channel", sample["x"].shape[0] - 1))
                mask = sample["x"][mask_channel] > 0.5
                for channel in range(channel_count):
                    values = processor._select_temporal_channel(target, channel, channel_axis)
                    expanded_mask = mask
                    while expanded_mask.ndim < values.ndim:
                        expanded_mask = expanded_mask.unsqueeze(0)
                    if expanded_mask.shape == values.shape or all(
                        left == right or left == 1 for left, right in zip(expanded_mask.shape, values.shape)
                    ):
                        values = values[expanded_mask.expand_as(values)]
                    else:
                        values = values.reshape(-1)
                    variable_name = variable_names[channel] if channel < len(variable_names) else f"field_{channel}"
                    if values.numel():
                        values_by_variable.setdefault(variable_name, []).append(values.abs().reshape(-1))
                continue

            channel_axis = -1 if target.ndim > 1 else None
            channel_count = int(target.shape[channel_axis]) if channel_axis is not None else 1
            for channel in range(channel_count):
                values = target[..., channel].reshape(-1) if channel_axis is not None else target.reshape(-1)
                variable_name = variable_names[channel] if channel < len(variable_names) else f"field_{channel}"
                if values.numel():
                    values_by_variable.setdefault(variable_name, []).append(values.abs())

        scales = {}
        for name, chunks in values_by_variable.items():
            finite_values = torch.cat(chunks)
            finite_values = finite_values[torch.isfinite(finite_values)]
            if finite_values.numel():
                # torch.quantile rejects tensors larger than 2**24 elements.
                # Full Rotor37 training exceeds that limit; NumPy computes the
                # same linear-interpolated percentile without subsampling.
                if finite_values.numel() > 2**24:
                    scales[name] = float(np.quantile(finite_values.numpy(), 0.95))
                else:
                    scales[name] = float(torch.quantile(finite_values, 0.95).item())
        return {"quantile": 0.95, "by_variable": scales}

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def normalize_data(
        train_csv: str,
        processed_paths: List[str],
        # Continues the surrounding statement or collection definition.
        norm_save_path: str,
        norm_type: str = "standardization",
        processor=None,
        processor_type: str = "ImageVoxelProcessor",
        # Continues the surrounding statement or collection definition.
        train_processed_paths: List[str] = None,
        resume_manager: Optional[PreprocessResumeManager] = None,
    ) -> None:
        """基于训练集样本统计归一化参数，并应用到所有中间产物。"""
        if processor is None:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("normalize_data requires a processor instance")

        if train_processed_paths is None:
            sample_map = SampleMap(train_csv)
            train_sample_ids = set(sample_map.get_all_samples())
            # Initializes the `train_processed_paths` as a collection for values from this processing stage.
            train_processed_paths = []
            for processed_path in processed_paths:
                file_name = Path(processed_path).name
                if not file_name.startswith("sample_") or not file_name.endswith(".pt"):
                    # Handles the data or control-flow case represented by this statement.
                    continue
                try:
                    sample_id = int(file_name[len("sample_") : -3])
                except ValueError:
                    # Handles the data or control-flow case represented by this statement.
                    continue
                if sample_id in train_sample_ids:
                    train_processed_paths.append(processed_path)

        if not train_processed_paths:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("No processed training samples were found for normalization")

        if resume_manager is not None:
            if resume_manager.status in {"sample_processing", "matrix_encode"}:
                resume_manager.begin_normalizing(processed_paths)
            # Validates `resume_manager.status not in {"normalizing", "committing"}` and raises `RuntimeError` when the input cannot be processed safely.
            elif resume_manager.status not in {"normalizing", "committing"}:
                raise RuntimeError(f"预处理恢复状态非法: {resume_manager.status}")
            pending_norm_path = resume_manager.get_pending_norm_params_path()
        else:
            # Stores the `pending_norm_path` for use by the remaining steps in this flow.
            pending_norm_path = norm_save_path

        if resume_manager is not None and os.path.exists(pending_norm_path):
            try:
                norm_params = load_torch(pending_norm_path, map_location="cpu", weights_only=False)
                # Validates `not isinstance(norm_params, dict)` and raises `ValueError` when the input cannot be processed safely.
                if not isinstance(norm_params, dict):
                    raise ValueError("归一化参数不是字典")
            except Exception as exc:
                if resume_manager.status == "committing":
                    # Raises `RuntimeError` to report why this path cannot continue.
                    raise RuntimeError("提交归一化结果时缺少有效的暂存归一化参数") from exc
                norm_params = None
        else:
            norm_params = None

        # Checks `norm_params is None` before entering this conditional branch.
        if norm_params is None:
            compute_from_paths = getattr(processor, "compute_norm_params_from_paths", None)
            if callable(compute_from_paths):
                norm_params = compute_from_paths(
                    # Completes the multi-line `compute_from_paths` arguments or elements.
                    train_processed_paths=train_processed_paths,
                    norm_type=norm_type,
                    save_path=pending_norm_path,
                )
            # Continues the surrounding statement or collection definition.
            else:
                train_samples = [
                    load_torch(processed_path, weights_only=False)
                    for processed_path in train_processed_paths
                # Continues the surrounding statement or collection definition.
                ]
                norm_params = processor.compute_norm_params(
                    train_samples=train_samples,
                    norm_type=norm_type,
                    # Completes the multi-line `processor.compute_norm_params` arguments or elements.
                    save_path=pending_norm_path,
                )
            atomic_torch_save(norm_params, pending_norm_path)

        if "mape_target_scale" not in norm_params:
            norm_params["mape_target_scale"] = DataPipeline._compute_mape_target_scales(
                train_processed_paths=train_processed_paths,
                processor=processor,
                norm_params=norm_params,
            )

        set_norm_params = getattr(processor, "set_normalization_params", None)
        # Checks `callable(set_norm_params)` before entering this conditional branch.
        if callable(set_norm_params):
            set_norm_params(norm_params)

        if resume_manager is None:
            DataPipeline._apply_normalization(
                # Completes the multi-line `DataPipeline._apply_normalization` arguments or elements.
                processed_paths=processed_paths,
                processor=processor,
                norm_params=norm_params,
                norm_type=norm_type,
                # Completes the multi-line `DataPipeline._apply_normalization` arguments or elements.
                normalize_y=True,
            )
            atomic_torch_save(norm_params, norm_save_path)
            return

        # Stores the `reusable_paths` for use by the remaining steps in this flow.
        reusable_paths = set(resume_manager.get_reusable_normalized_paths(processed_paths))
        for processed_path in processed_paths:
            if processed_path in reusable_paths:
                continue
            # Stores the current data for use by the remaining steps in this flow.
            data = load_torch(processed_path, weights_only=False)
            normalized_data = processor.normalize_processed_sample(
                sample=data,
                norm_params=norm_params,
                # Completes the multi-line `processor.normalize_processed_sample` arguments or elements.
                norm_type=norm_type,
                normalize_y=True,
            )
            atomic_torch_save(normalized_data, resume_manager.get_staging_path(processed_path))
            # Calls `resume_manager.mark_normalized` to perform the operation required at this stage.
            resume_manager.mark_normalized(processed_path)

        if resume_manager.status == "normalizing":
            resume_manager.begin_committing()
        for processed_path in processed_paths:
            # Checks `resume_manager.is_committed(processed_path)` before entering this conditional branch.
            if resume_manager.is_committed(processed_path):
                continue
            if resume_manager.reconcile_commit(processed_path):
                resume_manager.mark_committed(processed_path)
                # Handles the data or control-flow case represented by this statement.
                continue
            staged_path = resume_manager.get_staging_path(processed_path)
            normalized_data = load_torch(staged_path, weights_only=False)
            atomic_torch_save(normalized_data, processed_path)
            # Calls `resume_manager.mark_committed` to perform the operation required at this stage.
            resume_manager.mark_committed(processed_path)

        atomic_torch_save(norm_params, norm_save_path)

    @staticmethod
    def normalize_data_predict(
        # Continues the surrounding statement or collection definition.
        processed_paths: List[str],
        norm_params: Dict,
        norm_type: str = "standardization",
        processor=None,
        # Continues the surrounding statement or collection definition.
        processor_type: str = "ImageVoxelProcessor",
    ) -> None:
        """对预测阶段中间产物应用已有归一化参数。"""
        if processor is None:
            raise ValueError("normalize_data_predict requires a processor instance")

        # Stores the `set_norm_params` for use by the remaining steps in this flow.
        set_norm_params = getattr(processor, "set_normalization_params", None)
        if callable(set_norm_params):
            set_norm_params(norm_params)

        DataPipeline._apply_normalization(
            # Completes the multi-line `DataPipeline._apply_normalization` arguments or elements.
            processed_paths=processed_paths,
            processor=processor,
            norm_params=norm_params,
            norm_type=norm_type,
            # Completes the multi-line `DataPipeline._apply_normalization` arguments or elements.
            normalize_y=False,
        )

    @staticmethod
    def normalize_data_with_existing_params(
        # Continues the surrounding statement or collection definition.
        processed_paths: List[str],
        norm_params: Dict,
        norm_type: str = "standardization",
        processor=None,
        # Continues the surrounding statement or collection definition.
        normalize_y: bool = True,
    ) -> None:
        """使用既有归一化参数处理样本，可选择是否归一化目标值。"""
        if processor is None:
            raise ValueError("normalize_data_with_existing_params requires a processor instance")

        # Stores the `set_norm_params` for use by the remaining steps in this flow.
        set_norm_params = getattr(processor, "set_normalization_params", None)
        if callable(set_norm_params):
            set_norm_params(norm_params)

        DataPipeline._apply_normalization(
            # Completes the multi-line `DataPipeline._apply_normalization` arguments or elements.
            processed_paths=processed_paths,
            processor=processor,
            norm_params=norm_params,
            norm_type=norm_type,
            # Completes the multi-line `DataPipeline._apply_normalization` arguments or elements.
            normalize_y=normalize_y,
        )
