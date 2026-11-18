import copy
import logging
import csv
# Imports `os` for reuse in this module.
import os
import re
import time
from pathlib import Path
# Imports `Any, Dict, List` from `typing` for use in this module.
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

# Imports `DataLoader, build_dataset` from `tfai.dataset` for use in this module.
from tfai.dataset import DataLoader, build_dataset
from tfai.factories.processor_factory import ProcessorFactory
from tfai.factories.model_factory import ModelFactory
from tfai.factories.reader_factory import ReaderFactory
# Imports `DataPipeline` from `tfai.data.data_pipeline` for use in this module.
from tfai.data.data_pipeline import DataPipeline
from tfai.readers.sample_map import SampleMap
from tfai.utils.failure_report import build_failure_entry, summarize_writeback_failures
from tfai.utils.logger import Logger
# Imports `resolve_path_from_base` from `tfai.utils.path_utils` for use in this module.
from tfai.utils.path_utils import resolve_path_from_base
from tfai.utils.task_variable_config import (
    normalize_scalar_task_variable_aliases,
    resolve_scalar_target_variables,
# Continues the surrounding statement or collection definition.
)
from tfai.utils.task_constraints import validate_matrix_pod_fixed_geometry_only
from tfai.utils.torch_utils import load_torch, resolve_torch_device
from tfai.prediction.adapters import BasePredictAdapter
# Imports `build_model_from_checkpoint` from `tfai.utils.model_bundle` for use in this module.
from tfai.utils.model_bundle import build_model_from_checkpoint


class Predictor:
    """预测器。

    核心职责：
    - 加载检查点、模型和处理器
    - 构建预测阶段数据集
    - 执行推理并做反归一化
    - 调用预测适配器写回结果
    """

    def __init__(
        self,
        # Continues the surrounding statement or collection definition.
        checkpoint_path: str,
        predict_config: Dict = None,
        logger: Optional[logging.Logger] = None,
        defer_reader_initialization: bool = False,
    # Continues the surrounding statement or collection definition.
    ):
        """初始化预测器。

        Args:
            checkpoint_path: 模型检查点路径。
            predict_config: 预测阶段配置；覆盖 `prediction` 部分和显式指定的运行设备。
        """
        checkpoint = load_torch(checkpoint_path, map_location="cpu", weights_only=False)
        self.logger = logger
        self.checkpoint = checkpoint
        # Stores the checkpoint path on the instance so other methods can reuse it.
        self.checkpoint_path = checkpoint_path

        self.config = checkpoint.get("config", {})
        if predict_config and "prediction" in predict_config:
            self.config["prediction"] = predict_config["prediction"]
        predict_general_cfg = (predict_config or {}).get("general", {})
        if "device" in predict_general_cfg:
            self.config.setdefault("general", {})["device"] = predict_general_cfg["device"]

        # Validates `"config" not in checkpoint` and raises `ValueError` when the input cannot be processed safely.
        if "config" not in checkpoint:
            raise ValueError("检查点缺少 config，无法初始化 Predictor")
        if "model_state_dict" not in checkpoint:
            raise ValueError("检查点缺少 model_state_dict，无法加载模型")

        # Reads `variable_manifest` from the mapping into the `variable_manifest` with its declared fallback.
        variable_manifest = checkpoint.get("variable_manifest", {})
        task_cfg = self.config.get("task", {})
        if variable_manifest:
            task_cfg.setdefault(
                # Completes the multi-line `task_cfg.setdefault` arguments or elements.
                "logical_variables",
                variable_manifest.get("logical_variables", task_cfg.get("logical_variables", [])),
            )
            task_cfg.setdefault(
                # Completes the multi-line `task_cfg.setdefault` arguments or elements.
                "effective_variables",
                variable_manifest.get(
                    "effective_variables",
                    task_cfg.get("effective_variables", []),
                # Completes the multi-line `variable_manifest.get` arguments or elements.
                ),
            )
            task_cfg.setdefault(
                "field_component_map",
                # Completes the multi-line `variable_manifest.get` arguments or elements.
                variable_manifest.get(
                    "field_component_map",
                    task_cfg.get("field_component_map", {}),
                ),
            # Completes the multi-line `task_cfg.setdefault` arguments or elements.
            )
            task_cfg.setdefault(
                "variable_catalog",
                variable_manifest.get(
                    # Completes the multi-line `variable_manifest.get` arguments or elements.
                    "variable_catalog",
                    task_cfg.get("variable_catalog", {}),
                ),
            )
            # Calls `task_cfg.setdefault` to perform the operation required at this stage.
            task_cfg.setdefault(
                "effective_variable_sources",
                variable_manifest.get(
                    "effective_variable_sources",
                    # Completes the multi-line `task_cfg.get` arguments or elements.
                    task_cfg.get("effective_variable_sources", {}),
                ),
            )
            task_cfg.setdefault(
                # Completes the multi-line `task_cfg.setdefault` arguments or elements.
                "point_or_cell",
                variable_manifest.get(
                    "point_or_cell",
                    task_cfg.get("point_or_cell"),
                # Completes the multi-line `variable_manifest.get` arguments or elements.
                ),
            )
            task_cfg.setdefault(
                "field_location_map",
                # Completes the multi-line `variable_manifest.get` arguments or elements.
                variable_manifest.get(
                    "field_location_map",
                    task_cfg.get("field_location_map", {}),
                ),
            # Completes the multi-line `task_cfg.setdefault` arguments or elements.
            )
            task_cfg.setdefault(
                "projected_point_or_cell",
                variable_manifest.get(
                    # Completes the multi-line `variable_manifest.get` arguments or elements.
                    "projected_point_or_cell",
                    task_cfg.get("projected_point_or_cell"),
                ),
            )
            # Calls `task_cfg.setdefault` to perform the operation required at this stage.
            task_cfg.setdefault(
                "mixed_field_policy",
                variable_manifest.get(
                    "mixed_field_policy",
                    # Completes the multi-line `task_cfg.get` arguments or elements.
                    task_cfg.get("mixed_field_policy"),
                ),
            )
            task_cfg.setdefault(
                # Completes the multi-line `task_cfg.setdefault` arguments or elements.
                "projection_method",
                variable_manifest.get(
                    "projection_method",
                    task_cfg.get("projection_method"),
                # Completes the multi-line `variable_manifest.get` arguments or elements.
                ),
            )
            task_cfg.setdefault(
                "geometry_regime",
                # Completes the multi-line `variable_manifest.get` arguments or elements.
                variable_manifest.get(
                    "geometry_regime",
                    task_cfg.get("geometry_regime"),
                ),
            # Completes the multi-line `task_cfg.setdefault` arguments or elements.
            )
            task_cfg.setdefault(
                "geometry_motion",
                variable_manifest.get(
                    # Completes the multi-line `variable_manifest.get` arguments or elements.
                    "geometry_motion",
                    task_cfg.get("geometry_motion", {}),
                ),
            )
            # Calls `task_cfg.setdefault` to perform the operation required at this stage.
            task_cfg.setdefault(
                "geometry_representation",
                variable_manifest.get(
                    "geometry_representation",
                    # Completes the multi-line `task_cfg.get` arguments or elements.
                    task_cfg.get("geometry_representation"),
                ),
            )
            task_cfg.setdefault(
                # Completes the multi-line `task_cfg.setdefault` arguments or elements.
                "point_feature_fields",
                variable_manifest.get(
                    "point_feature_fields",
                    task_cfg.get("point_feature_fields", []),
                # Completes the multi-line `variable_manifest.get` arguments or elements.
                ),
            )
            task_cfg.setdefault(
                "point_target_fields",
                # Completes the multi-line `variable_manifest.get` arguments or elements.
                variable_manifest.get(
                    "point_target_fields",
                    task_cfg.get("point_target_fields", []),
                ),
            # Completes the multi-line `task_cfg.setdefault` arguments or elements.
            )
            task_cfg.setdefault(
                "displacement_logical_variable",
                variable_manifest.get(
                    # Completes the multi-line `variable_manifest.get` arguments or elements.
                    "displacement_logical_variable",
                    task_cfg.get("displacement_logical_variable"),
                ),
            )
            # Calls `task_cfg.setdefault` to perform the operation required at this stage.
            task_cfg.setdefault(
                "displacement_effective_variables",
                variable_manifest.get(
                    "displacement_effective_variables",
                    # Completes the multi-line `task_cfg.get` arguments or elements.
                    task_cfg.get("displacement_effective_variables", []),
                ),
            )

        normalize_scalar_task_variable_aliases(self.config)
        # Stores the `checkpoint_task_type` for use by the remaining steps in this flow.
        checkpoint_task_type = str(task_cfg.get("type", "")).lower()
        if checkpoint_task_type == "scalar" and not task_cfg.get("effective_variables"):
            task_cfg["effective_variables"] = resolve_scalar_target_variables(task_cfg)

        if checkpoint_task_type != "scalar" and (
            # Completes the multi-line `task_cfg.get` arguments or elements.
            "effective_variables" not in task_cfg or not task_cfg.get("effective_variables")
        ):
            raise ValueError("检查点缺少 task.effective_variables，无法保证变量语义一致")

        self.task_config = self.config["task"]
        # Stores the `is_geometry_varying` on the instance so other methods can reuse it.
        self.is_geometry_varying = self.task_config["is_geometry_varying"]
        self.model_config = self.config["model"]
        self.batch_size = self.config.get("prediction", {}).get("batch_size", 32)

        self.task_type = self.task_config["type"]
        # Stores the `mode` on the instance so other methods can reuse it.
        self.mode = self.task_config["mode"]
        self.temporal_strategy = self.task_config["temporal_strategy"]
        self.output_length = self.task_config["output_length"]
        self.rollout_steps = int(
            # Completes the multi-line `self.config.get('prediction', {}).get` arguments or elements.
            self.config.get("prediction", {}).get("rollout_steps", self.output_length)
        )
        self.strict_writeback = bool(
            self.config.get("prediction", {}).get("strict_writeback", False)
        # Completes the multi-line `bool` arguments or elements.
        )
        self.grid_type = self.task_config.get("grid_type")
        self.model_type = self.model_config["type"]
        self.last_writeback_report = summarize_writeback_failures(
            # Completes the multi-line `collection` arguments or elements.
            [],
            strict=bool(getattr(self, "strict_writeback", False)),
            expected_count=0,
            success_count=0,
        # Completes the multi-line `summarize_writeback_failures` arguments or elements.
        )

        validate_matrix_pod_fixed_geometry_only(self.config, stage="prediction")

        in_channels = self.model_config.get("in_channels", None)
        out_channels = self.model_config.get(
            # Completes the multi-line `self.model_config.get` arguments or elements.
            "out_channels", self.model_config.get("n_classes", None)
        )
        img_size = self.model_config.get("img_size", None)

        if in_channels is None or out_channels is None:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("检查点中的模型输入输出参数不完整，无法创建预测模型")

        models_require_img_size = {"UNet"}
        if self.model_type in models_require_img_size and img_size is None:
            raise ValueError("当前模型需要 img_size 参数，但检查点中未找到")

        # Stores the `model, _)` for use by the remaining steps in this flow.
        self.model, _ = build_model_from_checkpoint(checkpoint, self.config)

        self.device = resolve_torch_device(self.config)
        self.model.to(self.device)

        self.processor = ProcessorFactory.get_processor(
            # Completes the multi-line `ProcessorFactory.get_processor` arguments or elements.
            self.task_type, self.model_type, self.config
        )
        self.reader = None
        self.predict_adapter = None
        # Stores the `_deferred_reader_error` on the instance so other methods can reuse it.
        self._deferred_reader_error: Optional[Exception] = None
        if not defer_reader_initialization:
            try:
                self.reader = ReaderFactory.get_reader(self.config, flag="predict")
            # Continues the surrounding statement or collection definition.
            except Exception as exc:
                if self.task_type == "field" and not self.is_geometry_varying:
                    self._deferred_reader_error = exc
                else:
                    # Raises `the active exception` to report why this path cannot continue.
                    raise

        self.norm_params = checkpoint.get("norm_params", None)
        # if self.norm_params:
            # print("已从检查点加载归一化参数")
        # else:
            # print("检查点中未找到归一化参数")

        set_norm_params = getattr(self.processor, "set_normalization_params", None)
        if callable(set_norm_params):
            # Calls `set_norm_params` to perform the operation required at this stage.
            set_norm_params(self.norm_params)
        if hasattr(self.processor, "load_checkpoint_artifacts"):
            self.processor.load_checkpoint_artifacts(checkpoint)
        if hasattr(self.model, "attach_processor"):
            # Calls `self.model.attach_processor` to perform the operation required at this stage.
            self.model.attach_processor(self.processor)

        if self.reader is not None:
            self.predict_adapter = self._build_prediction_adapter()

    def _build_prediction_adapter(self) -> Any:
        # Validates `self.reader is None` and raises `ValueError` when the input cannot be processed safely.
        if self.reader is None:
            raise ValueError("reader 尚未初始化，无法构建预测适配器")
        if not hasattr(self.processor, "build_prediction_adapter"):
            raise AttributeError(
                # Completes the multi-line `type` arguments or elements.
                f"{type(self.processor).__name__} 缺少 build_prediction_adapter 接口"
            )
        return self.processor.build_prediction_adapter(
            reader=self.reader,
            # Completes the multi-line `self.processor.build_prediction_adapter` arguments or elements.
            config=self.config,
            is_geometry_varying=self.is_geometry_varying,
            resolve_reference_file=self._resolve_reference_file,
            logger=self.logger,
        # Completes the multi-line `self.processor.build_prediction_adapter` arguments or elements.
        )

    def _requires_fixed_geometry_predict_writeback_reference(self) -> bool:
        requires_reference = getattr(
            self.processor,
            # Completes the multi-line `getattr` arguments or elements.
            "requires_fixed_geometry_predict_writeback_reference",
            None,
        )
        if callable(requires_reference):
            # Returns `bool(requires_reference())` to the caller for the next stage.
            return bool(requires_reference())
        return True

    @staticmethod
    def _resolve_runtime_template_source(source_path: str) -> str:
        """将文件或目录形式的样本路径解析成可用于模板/扩展名推断的单文件路径。"""
        # Normalizes the `candidate` before subsequent file operations.
        candidate = Path(str(source_path)).resolve()
        if candidate.is_file():
            return str(candidate)

        if candidate.is_dir():
            # Stores the `child_files` for use by the remaining steps in this flow.
            child_files = sorted(
                path for path in candidate.iterdir()
                if path.is_file()
            )
            # Validates `not child_files` and raises `ValueError` when the input cannot be processed safely.
            if not child_files:
                raise ValueError(f"模板目录中没有可用文件: {candidate}")
            return str(child_files[0].resolve())

        raise ValueError(f"模板路径不存在: {candidate}")

    # 解析固定几何场景下用于补建 reader 或写回模板的源文件。
    def _resolve_fixed_geometry_runtime_source(
        self,
        sample_files: Optional[List[str]] = None,
    ) -> str:
        """解析固定几何场景下用于补建 reader 或写回模板的源文件。"""
        # Returns early when `self._requires_fixed_geometry_predict_writeback_reference()` makes the remaining work inapplicable.
        if self._requires_fixed_geometry_predict_writeback_reference():
            return self._resolve_reference_file()

        for sample_file in sample_files or []:
            sample_text = str(sample_file or "").strip()
            # Checks `not sample_text` before entering this conditional branch.
            if not sample_text:
                continue
            return self._resolve_runtime_template_source(sample_text)

        return self._resolve_reference_file()

    # 在固定几何预测场景下按需补建 reader。
    def _ensure_predict_runtime_objects(self, sample_files: Optional[List[str]] = None) -> None:
        """在固定几何预测场景下按需补建 reader。"""
        current_reader = getattr(self, "reader", None)
        if current_reader is not None:
            return

        # Checks `self.task_type == "field" and not self.is_geometry_varying` before entering this conditional branch.
        if self.task_type == "field" and not self.is_geometry_varying:
            runtime_source = self._resolve_fixed_geometry_runtime_source(sample_files)
        else:
            runtime_source = ""
            # Visits each item in `sample_files or []` and applies the current processing step.
            for sample_file in sample_files or []:
                sample_text = str(sample_file or "").strip()
                if sample_text:
                    runtime_source = self._resolve_runtime_template_source(sample_text)
                    # Handles the data or control-flow case represented by this statement.
                    break

        if not runtime_source:
            deferred_error = getattr(self, "_deferred_reader_error", None)
            if deferred_error is not None:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(
                    f"预测 reader 初始化失败: {str(deferred_error)}"
                ) from deferred_error
            raise ValueError("预测 reader 尚未初始化，且未提供可用于推断 reader 的样本源")

        # Stores the `extension` for use by the remaining steps in this flow.
        extension = ReaderFactory.infer_reader_extension_from_source_path(runtime_source)
        reader_config = copy.deepcopy(self.config)
        prediction_samples_csv = reader_config.get("prediction", {}).get("predict_samples")
        if prediction_samples_csv:
            # Stores the `setdefault('general', {})['train_samples']` for use by the remaining steps in this flow.
            reader_config.setdefault("general", {})["train_samples"] = prediction_samples_csv
        self.reader = ReaderFactory._create_reader_by_extension(extension, reader_config)
        self._deferred_reader_error = None

    def _create_filtered_samples_csv(
        # Continues the surrounding statement or collection definition.
        self, source_csv: str, output_csv: str, keep_ids: List[int]
    ) -> str:
        """基于 source_csv 按样本 ID 子集生成过滤后的样本表。"""
        keep_set = set(keep_ids)
        sample_map = SampleMap(source_csv)
        # Builds the `all_ids` using the comprehension's filter and mapping rules.
        all_ids = [sid for sid in sample_map.get_all_samples() if sid in keep_set]
        if not all_ids:
            raise ValueError("过滤后样本为空，无法生成有效样本表")

        first_info = sample_map.get_sample_info(all_ids[0]) or {}
        # Derives the `fieldnames` from the current values for downstream processing.
        fieldnames = ["order"] + [k for k in first_info.keys()]

        output_dir = os.path.dirname(output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Scopes resource use so the context manager can release it on exit.
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for sample_id in all_ids:
                # Stores the `info` for use by the remaining steps in this flow.
                info = sample_map.get_sample_info(sample_id) or {}
                row = {"order": sample_id}
                row.update(info)
                writer.writerow(row)

        # Returns `output_csv` to the caller for the next stage.
        return output_csv

    def _get_sample_files_from_csv(self, samples_csv: str) -> List[str]:
        """从样本表中提取样本文件绝对路径列表。"""
        sample_map = SampleMap(samples_csv)
        base_dir = os.path.dirname(os.path.abspath(samples_csv))
        # Initializes the `files` as a collection for values from this processing stage.
        files: List[str] = []

        for sample_id in sample_map.get_all_samples():
            file_path = sample_map.get_file_path(sample_id)
            if not file_path:
                # Handles the data or control-flow case represented by this statement.
                continue
            abs_path = resolve_path_from_base(base_dir, file_path)
            files.append(os.path.abspath(abs_path))

        return files

    # 创建临时样本表，将所有样本的 file_path 统一替换为 reference_file。
    def _create_reference_sample_csv(
        self, source_csv: str, output_csv: str, reference_file: str
    ) -> None:
        """创建临时样本表，将所有样本的 file_path 统一替换为 reference_file。"""
        sample_map = SampleMap(source_csv)
        # Stores the sample identifiers for use by the remaining steps in this flow.
        sample_ids = sample_map.get_all_samples()

        if not sample_ids:
            raise ValueError("样本表中没有可用样本，无法创建临时样本表")

        first_info = sample_map.get_sample_info(sample_ids[0]) or {}
        # Builds the `param_fields` using the comprehension's filter and mapping rules.
        param_fields = [k for k in first_info.keys() if k != "file_path"]
        fieldnames = ["order"] + param_fields + ["file_path"]

        output_dir = os.path.dirname(output_csv)
        if output_dir:
            # Calls `os.makedirs` to perform the operation required at this stage.
            os.makedirs(output_dir, exist_ok=True)

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            # Visits each item in `sample_ids` and applies the current processing step.
            for sample_id in sample_ids:
                info = sample_map.get_sample_info(sample_id) or {}
                row = {"order": sample_id, "file_path": reference_file}
                for key in param_fields:
                    # Stores the `row[key]` for use by the remaining steps in this flow.
                    row[key] = info.get(key, "")
                writer.writerow(row)

    def _resolve_reference_file(self) -> str:
        """解析几何不变任务的 reference 文件路径。"""
        ckpt_dir = os.path.dirname(os.path.abspath(self.checkpoint_path))
        # Checks `os.path.isdir(ckpt_dir)` before entering this conditional branch.
        if os.path.isdir(ckpt_dir):
            candidates = [
                os.path.join(ckpt_dir, f)
                for f in os.listdir(ckpt_dir)
                # Completes the multi-line `f.startswith` arguments or elements.
                if f.startswith("reference")
            ]
            if candidates:
                return os.path.abspath(candidates[0])

        # Raises `ValueError` to report why this path cannot continue.
        raise ValueError("几何不变任务未找到 reference 文件，请将其放在 checkpoint 同目录")

    def _predict_direct_batch(self, inputs) -> torch.Tensor:
        """稳态或时序 direct 分支：单次前向。"""
        return self.model(inputs)

    def _get_rollout_steps(self) -> int:
        """返回预测阶段实际使用的 rollout 步数。"""
        # Returns `` to the caller for the next stage.
        return int(getattr(self, "rollout_steps", getattr(self, "output_length", 1)))

    def _predict_autoregressive_batch(self, inputs) -> torch.Tensor:
        """时序 autoregressive 分支：滚动预测。"""
        if inputs.dim() < 3:
            raise ValueError(f"autoregressive 模式要求输入维度 >= 3，当前为 {inputs.dim()}")

        # Stores the `rollout_steps` for use by the remaining steps in this flow.
        rollout_steps = self._get_rollout_steps()
        batch_predictions = []
        current_input = inputs

        for step in range(rollout_steps):
            # Stores the output for use by the remaining steps in this flow.
            output = self.model(current_input)
            batch_predictions.append(output)

            if step < rollout_steps - 1:
                current_input = torch.cat([current_input[:, 1:], output.unsqueeze(1)], dim=1)

        # Returns `` to the caller for the next stage.
        return torch.cat(batch_predictions, dim=1) if len(batch_predictions) > 1 else batch_predictions[0]

    def _predict_autoregressive_with_processor(self, inputs) -> torch.Tensor:
        """优先委托 processor 执行 autoregressive rollout，缺省时回退旧实现。"""
        rollout_fn = getattr(self.processor, "predict_autoregressive_rollout", None)
        if callable(rollout_fn):
            # Returns `` to the caller for the next stage.
            return rollout_fn(
                model=self.model,
                inputs=inputs,
                rollout_steps=self._get_rollout_steps(),
            # Completes the multi-line `rollout_fn` arguments or elements.
            )
        return self._predict_autoregressive_batch(inputs)

    def _run_inference(self, dataset: Dataset) -> Union[np.ndarray, Dict[int, np.ndarray]]:
        """执行推理并完成反归一化。"""
        self.model.eval()
        # Stores the `dataloader` for use by the remaining steps in this flow.
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        batch_outputs = []

        with torch.no_grad():
            for inputs, _ in dataloader:
                # Stores the model inputs for use by the remaining steps in this flow.
                inputs = inputs.to(self.device) if hasattr(inputs, "to") else inputs

                if self.mode == "steady" or self.temporal_strategy == "direct":
                    outputs = self._predict_direct_batch(inputs)
                else:
                    # Stores the `outputs` for use by the remaining steps in this flow.
                    outputs = self._predict_autoregressive_with_processor(inputs)

                batch_outputs.append(
                    self.processor.collect_prediction_batch(inputs=inputs, outputs=outputs)
                )

        # Stores the `predictions` for use by the remaining steps in this flow.
        predictions = self.processor.merge_prediction_batches(batch_outputs)
        if isinstance(predictions, np.ndarray) and predictions.size == 0:
            return predictions
        if isinstance(predictions, dict) and not predictions:
            # Returns `predictions` to the caller for the next stage.
            return predictions

        # print("执行反归一化")
        return self.processor.denormalize_predictions(
            predictions=predictions,
            norm_params=self.norm_params,
        # Completes the multi-line `self.processor.denormalize_predictions` arguments or elements.
        )

    def _resolve_output_dir(self, output_dir: Optional[str] = None) -> str:
        """解析预测结果输出目录，允许调用方省略 output_dir。"""
        if output_dir:
            resolved_output_dir = os.path.abspath(output_dir)
        # Continues the surrounding statement or collection definition.
        else:
            prediction_cfg = self.config.setdefault("prediction", {})
            configured_output_dir = prediction_cfg.get("output_dir")
            if configured_output_dir:
                # Normalizes the `resolved_output_dir` before subsequent file operations.
                resolved_output_dir = os.path.abspath(str(configured_output_dir))
            else:
                experiment_dir = self.config.get("general", {}).get("experiment")
                if not experiment_dir:
                    # Raises `ValueError` to report why this path cannot continue.
                    raise ValueError("预测输出目录缺失，且配置中未找到 general.experiment")
                resolved_output_dir = os.path.abspath(
                    os.path.join(str(experiment_dir), "predict", "results")
                )
                # Stores the `prediction_cfg['output_dir']` for use by the remaining steps in this flow.
                prediction_cfg["output_dir"] = resolved_output_dir

        os.makedirs(resolved_output_dir, exist_ok=True)
        return resolved_output_dir

    def _postprocess_and_save(
        # Continues the surrounding statement or collection definition.
        self,
        predictions: Union[np.ndarray, Dict[int, np.ndarray]],
        samples_csv: str,
        output_dir: str,
    # Continues the surrounding statement or collection definition.
    ) -> List[str]:
        """后处理并写回预测结果。"""
        task_type = str(
            getattr(self, "task_type", None)
            or self.config.get("task", {}).get("type", "field")
        # Completes the multi-line `str(getattr(self, 'task_type', None) or self.config.get('task', {}).get('type', 'field')).strip().lower` arguments or elements.
        ).strip().lower()
        if task_type == "scalar":
            if getattr(self, "predict_adapter", None) is None:
                self.predict_adapter = self._build_prediction_adapter()
            # Returns `` to the caller for the next stage.
            return self.predict_adapter.process_outputs(
                predictions=predictions,
                samples_csv=samples_csv,
                output_dir=output_dir,
                # Completes the multi-line `collection` arguments or elements.
                source_sample_files=[],
            )

        source_sample_files = self._get_sample_files_from_csv(samples_csv)
        self._ensure_predict_runtime_objects(source_sample_files)
        # Checks `getattr(self, "predict_adapter", None) is None` before entering this conditional branch.
        if getattr(self, "predict_adapter", None) is None:
            self.predict_adapter = self._build_prediction_adapter()
        return self.predict_adapter.process_outputs(
            predictions=predictions,
            # Completes the multi-line `self.predict_adapter.process_outputs` arguments or elements.
            samples_csv=samples_csv,
            output_dir=output_dir,
            source_sample_files=source_sample_files,
        )

    # 记录预测分支选择信息。
    def _log_predict_branch(self) -> None:
        """记录预测分支选择信息。"""
        logger = getattr(self, "logger", None)
        if logger is None:
            return
        # Records the current execution state to make progress and failures traceable.
        logger.info(
            "预测分支: geometry=%s, processor=%s, mode=%s, strategy=%s",
            "varying" if self.is_geometry_varying else "fixed",
            type(self.processor).__name__,
            # Completes the multi-line `logger.info` arguments or elements.
            self.mode,
            self.temporal_strategy,
        )

    def _update_writeback_report(
        # Continues the surrounding statement or collection definition.
        self,
        expected_count: int,
        output_paths: List[str],
    ) -> Dict:
        """汇总预测写回阶段结果，并按需启用严格模式。"""
        # Initializes the `failures` as a collection for values from this processing stage.
        failures: List[Dict] = []
        adapter = getattr(self, "predict_adapter", None)
        adapter_failures = list(getattr(adapter, "writeback_failures", []) or [])
        failures.extend(adapter_failures)
        # Stores the `success_count` for use by the remaining steps in this flow.
        success_count = self._count_successful_writeback_units(output_paths)
        if expected_count > 0 and success_count < expected_count:
            failures.append(
                build_failure_entry(
                    # Completes the multi-line `build_failure_entry` arguments or elements.
                    stage="prediction_writeback",
                    component="Predictor",
                    message="预测结果写回数量少于预期",
                    details={
                        # Completes the multi-line `int` arguments or elements.
                        "expected_count": int(expected_count),
                        "success_count": int(success_count),
                    },
                )
            # Completes the multi-line `failures.append` arguments or elements.
            )

        report = summarize_writeback_failures(
            failures,
            strict=bool(getattr(self, "strict_writeback", False)),
            # Completes the multi-line `summarize_writeback_failures` arguments or elements.
            expected_count=expected_count,
            success_count=success_count,
        )
        self.last_writeback_report = report

        # Stores the run logger for use by the remaining steps in this flow.
        logger = getattr(self, "logger", None)
        if logger is not None:
            if report["status"] == "success":
                logger.info(
                    # Completes the multi-line `logger.info` arguments or elements.
                    "预测结果写回完成: success=%s expected=%s",
                    report["success_count"],
                    report["expected_count"],
                )
            # Checks `report["status"] == "idle"` before entering this conditional branch.
            elif report["status"] == "idle":
                logger.info("预测结果写回未执行")
            else:
                logger.warning(
                    # Completes the multi-line `logger.warning` arguments or elements.
                    "预测结果写回存在降级: status=%s success=%s expected=%s",
                    report["status"],
                    report["success_count"],
                    report["expected_count"],
                # Completes the multi-line `logger.warning` arguments or elements.
                )

        if bool(getattr(self, "strict_writeback", False)) and report["status"] in {"partial", "failed"}:
            raise RuntimeError(
                f"预测结果写回失败: status={report['status']} "
                # Completes the multi-line `RuntimeError` arguments or elements.
                f"success={report['success_count']}/{report['expected_count']}"
            )
        return report

    @staticmethod
    # Defines `_extract_writeback_sample_ids` for  extract writeback sample ids logic.
    def _extract_writeback_sample_ids(output_paths: List[str]) -> List[int]:
        sample_ids: List[int] = []
        for path in output_paths:
            normalized_path = str(path).replace("\\", "/")
            # Stores the `file_name` for use by the remaining steps in this flow.
            file_name = os.path.basename(path)

            flat_match = re.search(r"(?:predicted|evaluated|target)_(\d+)\.", file_name)
            if flat_match:
                sample_ids.append(int(flat_match.group(1)))
                # Handles the data or control-flow case represented by this statement.
                continue

            temporal_match = re.search(r"/sample_(\d+)(?:/|$)", normalized_path)
            if temporal_match:
                sample_ids.append(int(temporal_match.group(1)))
        # Returns `sample_ids` to the caller for the next stage.
        return sample_ids

    @classmethod
    def _count_successful_writeback_units(cls, output_paths: List[str]) -> int:
        sample_ids = cls._extract_writeback_sample_ids(output_paths)
        # Returns early when `sample_ids` makes the remaining work inapplicable.
        if sample_ids:
            return len(set(sample_ids))
        return len(output_paths)

    def _process_fixed_geometry_temporal_autoregressive(
        # Continues the surrounding statement or collection definition.
        self,
        samples_csv: str,
        processed_dir: str,
        resolved_sample_files: List[str],
    # Completes the multi-line `collection` arguments or elements.
    ) -> Optional[Tuple[Dataset, str]]:
        """优先走固定几何时序 autoregressive 专用快路径。"""
        if (
            self.is_geometry_varying
            or self.mode != "temporal"
            # Continues the surrounding statement or collection definition.
            or self.temporal_strategy != "autoregressive"
        ):
            return None

        self._ensure_predict_runtime_objects(resolved_sample_files)
        # Validates `` and raises `NotImplementedError` when the input cannot be processed safely.
        if not self.processor.supports_fixed_geometry_temporal_predict_path():
            raise NotImplementedError(
                "固定几何 temporal autoregressive 预测尚未实现。"
                "当前处理器未提供按样本构造 history 种子的专用快路径。"
            # Completes the multi-line `NotImplementedError` arguments or elements.
            )

        reference_file = None
        requires_reference = getattr(
            self.processor,
            # Completes the multi-line `getattr` arguments or elements.
            "requires_fixed_geometry_temporal_reference",
            None,
        )
        if callable(requires_reference) and requires_reference():
            # Stores the `reference_file` for use by the remaining steps in this flow.
            reference_file = self._resolve_reference_file()

        return self.processor.build_fixed_geometry_temporal_predict_dataset(
            reader=self.reader,
            samples_csv=samples_csv,
            # Completes the multi-line `self.processor.build_fixed_geometry_temporal_predict_dataset` arguments or elements.
            processed_dir=processed_dir,
            config=self.config,
            norm_params=self.norm_params,
            reference_file=reference_file,
        # Completes the multi-line `self.processor.build_fixed_geometry_temporal_predict_dataset` arguments or elements.
        )

    def _process_fixed_geometry_reference_only(
        self,
        samples_csv: str,
        # Continues the surrounding statement or collection definition.
        processed_dir: str,
        resolved_sample_files: List[str],
    ) -> Optional[Tuple[Dataset, str]]:
        """处理固定几何稳态 reference-only 预测快路径。"""
        if (
            # Continues the surrounding statement or collection definition.
            self.is_geometry_varying
            or self.mode != "steady"
            or not self.processor.supports_reference_only_predict_path()
        ):
            # Returns `None` to the caller for the next stage.
            return None

        self._ensure_predict_runtime_objects(resolved_sample_files)
        reference_file = self._resolve_reference_file()
        return self.processor.build_reference_only_predict_dataset(
            # Completes the multi-line `self.processor.build_reference_only_predict_dataset` arguments or elements.
            reader=self.reader,
            samples_csv=samples_csv,
            processed_dir=processed_dir,
            config=self.config,
            # Completes the multi-line `self.processor.build_reference_only_predict_dataset` arguments or elements.
            norm_params=self.norm_params,
            reference_file=reference_file,
        )

    def _process_fast_predict_path(
        # Continues the surrounding statement or collection definition.
        self,
        samples_csv: str,
        processed_dir: str,
        resolved_sample_files: List[str],
    # Completes the multi-line `collection` arguments or elements.
    ) -> Optional[Tuple[Dataset, str]]:
        """处理绕过原始场文件读取的预测快路径。"""
        if not self.processor.supports_fast_predict_path(self.is_geometry_varying):
            return None

        self._ensure_predict_runtime_objects(resolved_sample_files)
        # Returns `` to the caller for the next stage.
        return self.processor.build_fast_predict_dataset(
            samples_csv=samples_csv,
            processed_dir=processed_dir,
            config=self.config,
            # Completes the multi-line `self.processor.build_fast_predict_dataset` arguments or elements.
            norm_params=self.norm_params,
        )

    def _normalize_processed_predict_paths(self, processed_paths: List[str]) -> None:
        """按当前任务模式执行预测阶段归一化。"""
        norm_type = (
            # Completes the multi-line `self.norm_params.get` arguments or elements.
            self.norm_params.get(
                "norm_type",
                self.config.get("processor", {}).get("norm_type", "standardization"),
            )
            # Continues the surrounding statement or collection definition.
            if self.norm_params
            else "standardization"
        )

        if not self.norm_params:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("未从检查点找到归一化参数，请确认训练阶段已保存归一化参数")

        if self.mode == "temporal" and self.temporal_strategy == "autoregressive":
            DataPipeline.normalize_data_with_existing_params(
                processed_paths=processed_paths,
                # Completes the multi-line `DataPipeline.normalize_data_with_existing_params` arguments or elements.
                norm_params=self.norm_params,
                norm_type=norm_type,
                processor=self.processor,
                normalize_y=True,
            # Completes the multi-line `DataPipeline.normalize_data_with_existing_params` arguments or elements.
            )
            return

        DataPipeline.normalize_data_predict(
            processed_paths=processed_paths,
            # Completes the multi-line `DataPipeline.normalize_data_predict` arguments or elements.
            norm_params=self.norm_params,
            norm_type=norm_type,
            processor=self.processor,
            processor_type=type(self.processor).__name__,
        # Completes the multi-line `DataPipeline.normalize_data_predict` arguments or elements.
        )

    @staticmethod
    def _collect_success_ids(processed_paths: List[str]) -> List[int]:
        """从已处理样本路径中提取成功样本 ID。"""
        success_ids: List[int] = []
        # Visits each item in `processed_paths` and applies the current processing step.
        for path in processed_paths:
            name = os.path.basename(path)
            if name.startswith("sample_") and name.endswith(".pt"):
                success_ids.append(int(name[len("sample_") : -3]))
        # Returns `success_ids` to the caller for the next stage.
        return success_ids

    def _build_predict_dataset(
        self,
        effective_csv: str,
        # Continues the surrounding statement or collection definition.
        processed_dir: str,
    ) -> Tuple[Dataset, str]:
        """根据处理后的样本表构建预测数据集。"""
        dataset = build_dataset(
            config=self.config,
            # Completes the multi-line `type` arguments or elements.
            processor_type=type(self.processor).__name__,
            csv_path=effective_csv,
            data_dir=processed_dir,
            phase="predict",
            # Completes the multi-line `build_dataset` arguments or elements.
            processor=self.processor,
        )
        return dataset, effective_csv

    def _process_scalar_predict_pipeline(
        # Continues the surrounding statement or collection definition.
        self,
        samples_csv: str,
        processed_dir: str,
    ) -> Tuple[Dataset, str]:
        """处理标量任务预测链路。"""
        # Checks `self.reader is None` before entering this conditional branch.
        if self.reader is None:
            self.reader = ReaderFactory.get_reader(
                self.config,
                flag="predict",
                # Completes the multi-line `ReaderFactory.get_reader` arguments or elements.
                sample_table_path_override=samples_csv,
            )

        processed_paths = DataPipeline.call_process_with_reader(
            reader=self.reader,
            # Completes the multi-line `DataPipeline.call_process_with_reader` arguments or elements.
            processor=self.processor,
            sample_table_path=samples_csv,
            stage="predict",
            processed_dir=processed_dir,
            # Completes the multi-line `getattr` arguments or elements.
            logger=getattr(self, "logger", None),
        )
        self._normalize_processed_predict_paths(processed_paths)

        if not processed_paths:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("标量数据处理后无可用样本，无法继续预测")

        return self._build_predict_dataset(
            effective_csv=samples_csv,
            processed_dir=processed_dir,
        # Completes the multi-line `self._build_predict_dataset` arguments or elements.
        )

    def _process_general_reader_pipeline(
        self,
        samples_csv: str,
        # Continues the surrounding statement or collection definition.
        processed_dir: str,
        sample_map: SampleMap,
        resolved_sample_files: List[str],
    ) -> Tuple[Dataset, str]:
        """处理通用 reader + processor 预测链路。"""
        # Stores the `effective_input_csv` for use by the remaining steps in this flow.
        effective_input_csv = samples_csv
        self._ensure_predict_runtime_objects(resolved_sample_files)
        if self.is_geometry_varying:
            if len(resolved_sample_files) != len(sample_map.get_all_samples()):
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(
                    f"样本表中的文件路径数量({len(resolved_sample_files)}) "
                    f"与样本数量({len(sample_map.get_all_samples())}) 不匹配"
                )
        # Continues the surrounding statement or collection definition.
        else:
            self._ensure_predict_runtime_objects(resolved_sample_files)
            effective_input_csv = os.path.join(processed_dir, "temp_samples.csv")
            reference_file = self._resolve_reference_file()
            # Calls `self._create_reference_sample_csv` to perform the operation required at this stage.
            self._create_reference_sample_csv(
                samples_csv,
                effective_input_csv,
                reference_file,
            # Completes the multi-line `self._create_reference_sample_csv` arguments or elements.
            )

        processed_paths = DataPipeline.call_process_with_reader(
            reader=self.reader,
            processor=self.processor,
            # Completes the multi-line `DataPipeline.call_process_with_reader` arguments or elements.
            sample_table_path=effective_input_csv,
            stage="predict",
            processed_dir=processed_dir,
            logger=getattr(self, "logger", None),
        # Completes the multi-line `DataPipeline.call_process_with_reader` arguments or elements.
        )
        self._normalize_processed_predict_paths(processed_paths)

        if not processed_paths:
            raise ValueError("数据处理后无可用样本，无法继续预测")

        # Stores the `success_ids` for use by the remaining steps in this flow.
        success_ids = self._collect_success_ids(processed_paths)
        effective_csv = effective_input_csv
        source_ids = SampleMap(effective_input_csv).get_all_samples()
        if len(success_ids) < len(source_ids):
            # Stores the `filtered_csv` for use by the remaining steps in this flow.
            filtered_csv = os.path.join(processed_dir, "effective_samples.csv")
            effective_csv = self._create_filtered_samples_csv(
                effective_input_csv,
                filtered_csv,
                # Completes the multi-line `self._create_filtered_samples_csv` arguments or elements.
                success_ids,
            )

        return self._build_predict_dataset(
            effective_csv=effective_csv,
            # Completes the multi-line `self._build_predict_dataset` arguments or elements.
            processed_dir=processed_dir,
        )

    def process_data(self, samples_csv: str, processed_dir: str) -> Tuple[Dataset, str]:
        """处理预测输入并返回数据集与实际生效的样本表。"""
        if not samples_csv:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("process_data 要求显式传入 samples_csv")
        if not processed_dir:
            raise ValueError("process_data 要求显式传入 processed_dir")
        validate_matrix_pod_fixed_geometry_only(self.config, stage="prediction")

        # Normalizes the `samples_csv` before subsequent file operations.
        samples_csv = os.path.abspath(samples_csv)
        processed_dir = os.path.abspath(processed_dir)
        os.makedirs(processed_dir, exist_ok=True)

        task_type = str(
            # Completes the multi-line `getattr` arguments or elements.
            getattr(self, "task_type", None)
            or self.config.get("task", {}).get("type", "field")
        ).strip().lower()
        if task_type == "scalar":
            # Returns `` to the caller for the next stage.
            return self._process_scalar_predict_pipeline(
                samples_csv=samples_csv,
                processed_dir=processed_dir,
            )

        # Stores the `sample_map` for use by the remaining steps in this flow.
        sample_map = SampleMap(samples_csv)
        resolved_sample_files = self._get_sample_files_from_csv(samples_csv)
        self._log_predict_branch()

        fixed_temporal_dataset = self._process_fixed_geometry_temporal_autoregressive(
            # Completes the multi-line `self._process_fixed_geometry_temporal_autoregressive` arguments or elements.
            samples_csv=samples_csv,
            processed_dir=processed_dir,
            resolved_sample_files=resolved_sample_files,
        )
        # Returns early when `fixed_temporal_dataset is not None` makes the remaining work inapplicable.
        if fixed_temporal_dataset is not None:
            return fixed_temporal_dataset

        reference_only_dataset = self._process_fixed_geometry_reference_only(
            samples_csv=samples_csv,
            # Completes the multi-line `self._process_fixed_geometry_reference_only` arguments or elements.
            processed_dir=processed_dir,
            resolved_sample_files=resolved_sample_files,
        )
        if reference_only_dataset is not None:
            # Returns `reference_only_dataset` to the caller for the next stage.
            return reference_only_dataset

        fast_predict_dataset = self._process_fast_predict_path(
            samples_csv=samples_csv,
            processed_dir=processed_dir,
            # Completes the multi-line `self._process_fast_predict_path` arguments or elements.
            resolved_sample_files=resolved_sample_files,
        )
        if fast_predict_dataset is not None:
            return fast_predict_dataset

        # Returns `` to the caller for the next stage.
        return self._process_general_reader_pipeline(
            samples_csv=samples_csv,
            processed_dir=processed_dir,
            sample_map=sample_map,
            # Completes the multi-line `self._process_general_reader_pipeline` arguments or elements.
            resolved_sample_files=resolved_sample_files,
        )

    def predict_with_timing(
        self,
        # Continues the surrounding statement or collection definition.
        samples_csv: str,
        processed_dir: str,
        output_dir: Optional[str] = None,
    ) -> Tuple[List[str], Dict[str, float]]:
        """执行完整预测流程，并在每个阶段结束时记录耗时。"""
        # Validates `not samples_csv` and raises `ValueError` when the input cannot be processed safely.
        if not samples_csv:
            raise ValueError("predict 要求显式传入 samples_csv")
        if not processed_dir:
            raise ValueError("predict 要求显式传入 processed_dir")

        # Normalizes the `samples_csv` before subsequent file operations.
        samples_csv = os.path.abspath(samples_csv)
        processed_dir = os.path.abspath(processed_dir)
        output_dir = self._resolve_output_dir(output_dir)

        process_start = time.perf_counter()
        # Stores the `(dataset, effective_csv)` for use by the remaining steps in this flow.
        dataset, effective_csv = self.process_data(
            samples_csv=samples_csv,
            processed_dir=processed_dir,
        )
        # Derives the `process_elapsed` from the current values for downstream processing.
        process_elapsed = time.perf_counter() - process_start
        Logger.log_duration(getattr(self, "logger", None), "数据处理", process_elapsed)

        inference_start = time.perf_counter()
        predictions = self._run_inference(dataset)
        # Derives the `inference_elapsed` from the current values for downstream processing.
        inference_elapsed = time.perf_counter() - inference_start
        Logger.log_duration(getattr(self, "logger", None), "模型预测", inference_elapsed)

        write_start = time.perf_counter()
        output_paths = self._postprocess_and_save(
            # Completes the multi-line `self._postprocess_and_save` arguments or elements.
            predictions=predictions,
            samples_csv=effective_csv,
            output_dir=output_dir,
        )
        # Derives the `write_elapsed` from the current values for downstream processing.
        write_elapsed = time.perf_counter() - write_start
        Logger.log_duration(getattr(self, "logger", None), "文件写回", write_elapsed)

        expected_count = 0
        task_type = str(
            # Completes the multi-line `getattr` arguments or elements.
            getattr(self, "task_type", None)
            or self.config.get("task", {}).get("type", "field")
        ).strip().lower()
        if task_type == "scalar":
            # Stores the `expected_count` for use by the remaining steps in this flow.
            expected_count = len(output_paths)
        elif os.path.isfile(effective_csv):
            expected_count = len(SampleMap(effective_csv).get_all_samples())
        if expected_count == 0 and hasattr(dataset, "__len__") and not isinstance(
            # Completes the multi-line `collection` arguments or elements.
            dataset, (str, bytes, bytearray)
        ):
            try:
                expected_count = int(len(dataset))
            # Continues the surrounding statement or collection definition.
            except Exception:
                expected_count = 0
        self._update_writeback_report(
            expected_count=expected_count,
            # Completes the multi-line `self._update_writeback_report` arguments or elements.
            output_paths=output_paths,
        )

        return output_paths, {
            "data_processing_seconds": process_elapsed,
            # Completes the multi-line `collection` arguments or elements.
            "model_predict_seconds": inference_elapsed,
            "file_write_seconds": write_elapsed,
        }

    def predict(
        # Continues the surrounding statement or collection definition.
        self,
        samples_csv: str,
        processed_dir: str,
        output_dir: Optional[str] = None,
    # Continues the surrounding statement or collection definition.
    ) -> List[str]:
        """执行完整预测流程。"""
        output_paths, _ = self.predict_with_timing(
            samples_csv=samples_csv,
            processed_dir=processed_dir,
            # Completes the multi-line `self.predict_with_timing` arguments or elements.
            output_dir=output_dir,
        )
        return output_paths
