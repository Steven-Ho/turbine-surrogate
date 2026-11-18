import csv
import os
import re
# Imports `Any, Dict, List` from `typing` for use in this module.
from typing import Any, Dict, List, Optional, Tuple

import torch

from tfai.data.data_pipeline import DataPipeline
from tfai.dataset.full_sample_dataset import FullSampleDataset
# Imports `Evaluator` from `tfai.evaluation.evaluator` for use in this module.
from tfai.evaluation.evaluator import Evaluator
from tfai.factories.model_factory import ModelFactory
from tfai.factories.processor_factory import ProcessorFactory
from tfai.factories.reader_factory import ReaderFactory
# Imports `BasePredictAdapter` from `tfai.prediction.adapters` for use in this module.
from tfai.prediction.adapters import BasePredictAdapter
from tfai.utils.failure_report import build_failure_entry, summarize_writeback_failures
from tfai.utils.model_bundle import build_model_from_checkpoint
from tfai.utils.runtime_paths import resolve_runtime_paths
# Imports `resolve_sample_source_path` from `tfai.utils.sample_utils` for use in this module.
from tfai.utils.sample_utils import resolve_sample_source_path
from tfai.utils.task_variable_config import normalize_scalar_task_variable_aliases
from tfai.utils.torch_utils import load_torch


_DUMMY_TARGET_SOURCES = {"dummy", "seed", "input_only", "predict"}
# Initializes the `_DUMMY_PREDICTION_VIEWS` as a collection for values from this processing stage.
_DUMMY_PREDICTION_VIEWS = {
    "steady_input_only",
    "direct_input_only",
    "autoregressive_seed_only",
# Completes the multi-line `collection` arguments or elements.
}


def should_use_full_autoregressive_rollout(config: Dict) -> bool:
    """判断评估是否应使用全长度自回归自由滚动。"""
    task_cfg = config.get("task", {}) or {}
    mode = str(task_cfg.get("mode", "")).strip().lower()
    # Stores the `temporal_strategy` for use by the remaining steps in this flow.
    temporal_strategy = str(task_cfg.get("temporal_strategy", "")).strip().lower()
    if mode != "temporal" or temporal_strategy != "autoregressive":
        return False

    output_length = int(task_cfg.get("output_length", 1) or 1)
    # Validates `output_length != 1` and raises `NotImplementedError` when the input cannot be processed safely.
    if output_length != 1:
        raise NotImplementedError(
            "全长度 autoregressive 评估当前仅支持 task.output_length=1；"
            "多步输出模型需要实现 block rollout 后才能评估。"
        # Completes the multi-line `NotImplementedError` arguments or elements.
        )
    return True


class EvaluationRunner:
    """基于 checkpoint 驱动的评估运行器。"""

    def __init__(
        # Continues the surrounding statement or collection definition.
        self,
        checkpoint_path: str,
        evaluate_config: Dict,
        project_root: str,
        # Continues the surrounding statement or collection definition.
        logger=None,
    ) -> None:
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.project_root = project_root
        # Stores the run logger on the instance so other methods can reuse it.
        self.logger = logger
        self.checkpoint = load_torch(
            self.checkpoint_path,
            map_location="cpu",
            # Completes the multi-line `load_torch` arguments or elements.
            weights_only=False,
        )

        checkpoint_config = self.checkpoint.get("config", {})
        if not checkpoint_config:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("checkpoint 缺少 config，无法执行评估")
        if "model_state_dict" not in self.checkpoint:
            raise ValueError("checkpoint 缺少 model_state_dict，无法执行评估")

        self.config = checkpoint_config
        configured_general = dict((evaluate_config or {}).get("general", {}))
        if configured_general:
            checkpoint_general = self.config.setdefault("general", {})
            checkpoint_general.update(configured_general)
        # Stores the `config['evaluate']` for use by the remaining steps in this flow.
        self.config["evaluate"] = dict((evaluate_config or {}).get("evaluate", {}))
        self._restore_variable_manifest()
        normalize_scalar_task_variable_aliases(self.config)
        self.runtime_config = resolve_runtime_paths(
            # Completes the multi-line `resolve_runtime_paths` arguments or elements.
            self.config,
            project_root=self.project_root,
        )

        self.norm_params = self.checkpoint.get("norm_params")
        # Stores the `task_config` on the instance so other methods can reuse it.
        self.task_config = self.runtime_config["task"]
        self.task_type = self.task_config["type"]
        self.model_type = self.runtime_config["model"]["type"]
        self.grid_type = self.task_config.get("grid_type")
        # Stores the `is_geometry_varying` on the instance so other methods can reuse it.
        self.is_geometry_varying = bool(self.task_config["is_geometry_varying"])
        self.strict_writeback = bool(
            self.runtime_config.get("evaluate", {}).get("strict_writeback", False)
        )
        # Stores the `last_writeback_report` on the instance so other methods can reuse it.
        self.last_writeback_report = {
            "strict": self.strict_writeback,
            "status": "idle",
            "evaluated": summarize_writeback_failures(
                # Completes the multi-line `collection` arguments or elements.
                [],
                strict=self.strict_writeback,
                expected_count=0,
                success_count=0,
            # Completes the multi-line `summarize_writeback_failures` arguments or elements.
            ),
            "target": summarize_writeback_failures(
                [],
                strict=self.strict_writeback,
                # Completes the multi-line `summarize_writeback_failures` arguments or elements.
                expected_count=0,
                success_count=0,
            ),
        }

        # Stores the `model` on the instance so other methods can reuse it.
        self.model = self._build_model()
        self.processor = ProcessorFactory.get_processor(
            self.task_type,
            self.model_type,
            # Completes the multi-line `ProcessorFactory.get_processor` arguments or elements.
            self.runtime_config,
        )
        set_norm_params = getattr(self.processor, "set_normalization_params", None)
        if callable(set_norm_params):
            # Calls `set_norm_params` to perform the operation required at this stage.
            set_norm_params(self.norm_params)
        if hasattr(self.processor, "load_checkpoint_artifacts"):
            self.processor.load_checkpoint_artifacts(self.checkpoint)
        if hasattr(self.model, "attach_processor"):
            # Calls `self.model.attach_processor` to perform the operation required at this stage.
            self.model.attach_processor(self.processor)

    def _restore_variable_manifest(self) -> None:
        variable_manifest = self.checkpoint.get("variable_manifest", {})
        if not variable_manifest:
            # Ends this function and returns control to its caller.
            return

        task_cfg = self.config.get("task", {})
        task_cfg.setdefault(
            "logical_variables",
            # Completes the multi-line `variable_manifest.get` arguments or elements.
            variable_manifest.get("logical_variables", task_cfg.get("logical_variables", [])),
        )
        task_cfg.setdefault(
            "effective_variables",
            # Completes the multi-line `variable_manifest.get` arguments or elements.
            variable_manifest.get(
                "effective_variables",
                task_cfg.get("effective_variables", []),
            ),
        # Completes the multi-line `task_cfg.setdefault` arguments or elements.
        )
        task_cfg.setdefault(
            "field_component_map",
            variable_manifest.get(
                # Completes the multi-line `variable_manifest.get` arguments or elements.
                "field_component_map",
                task_cfg.get("field_component_map", {}),
            ),
        )
        # Calls `task_cfg.setdefault` to perform the operation required at this stage.
        task_cfg.setdefault(
            "variable_catalog",
            variable_manifest.get(
                "variable_catalog",
                # Completes the multi-line `task_cfg.get` arguments or elements.
                task_cfg.get("variable_catalog", {}),
            ),
        )
        task_cfg.setdefault(
            # Completes the multi-line `task_cfg.setdefault` arguments or elements.
            "effective_variable_sources",
            variable_manifest.get(
                "effective_variable_sources",
                task_cfg.get("effective_variable_sources", {}),
            # Completes the multi-line `variable_manifest.get` arguments or elements.
            ),
        )
        task_cfg.setdefault(
            "point_or_cell",
            # Completes the multi-line `variable_manifest.get` arguments or elements.
            variable_manifest.get(
                "point_or_cell",
                task_cfg.get("point_or_cell"),
            ),
        # Completes the multi-line `task_cfg.setdefault` arguments or elements.
        )
        task_cfg.setdefault(
            "field_location_map",
            variable_manifest.get(
                # Completes the multi-line `variable_manifest.get` arguments or elements.
                "field_location_map",
                task_cfg.get("field_location_map", {}),
            ),
        )
        # Calls `task_cfg.setdefault` to perform the operation required at this stage.
        task_cfg.setdefault(
            "projected_point_or_cell",
            variable_manifest.get(
                "projected_point_or_cell",
                # Completes the multi-line `task_cfg.get` arguments or elements.
                task_cfg.get("projected_point_or_cell"),
            ),
        )
        task_cfg.setdefault(
            # Completes the multi-line `task_cfg.setdefault` arguments or elements.
            "mixed_field_policy",
            variable_manifest.get(
                "mixed_field_policy",
                task_cfg.get("mixed_field_policy"),
            # Completes the multi-line `variable_manifest.get` arguments or elements.
            ),
        )
        task_cfg.setdefault(
            "projection_method",
            # Completes the multi-line `variable_manifest.get` arguments or elements.
            variable_manifest.get(
                "projection_method",
                task_cfg.get("projection_method"),
            ),
        # Completes the multi-line `task_cfg.setdefault` arguments or elements.
        )
        task_cfg.setdefault(
            "geometry_representation",
            variable_manifest.get(
                # Completes the multi-line `variable_manifest.get` arguments or elements.
                "geometry_representation",
                task_cfg.get("geometry_representation"),
            ),
        )
        # Calls `task_cfg.setdefault` to perform the operation required at this stage.
        task_cfg.setdefault(
            "geometry_motion",
            variable_manifest.get(
                "geometry_motion",
                # Completes the multi-line `task_cfg.get` arguments or elements.
                task_cfg.get("geometry_motion", {}),
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

    def _build_model(self):
        model, _ = build_model_from_checkpoint(self.checkpoint, self.runtime_config)
        return model

    # Defines `_resolve_reference_file` for  resolve reference file logic.
    def _resolve_reference_file(self) -> str:
        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        if os.path.isdir(checkpoint_dir):
            for file_name in os.listdir(checkpoint_dir):
                # Returns early when `file_name.startswith("reference")` makes the remaining work inapplicable.
                if file_name.startswith("reference"):
                    return os.path.abspath(os.path.join(checkpoint_dir, file_name))
        raise ValueError("固定几何评估未找到参考几何文件 reference.*")

    def _normalize_samples_evaluate(
        # Continues the surrounding statement or collection definition.
        self,
        samples_evaluate: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        # Stores the `seen_orders` for use by the remaining steps in this flow.
        seen_orders = set()

        for index, item in enumerate(samples_evaluate, start=1):
            normalized_item = dict(item)
            order = normalized_item.get("order", normalized_item.get("sample_id", index))
            # Converts the `order` to `int` so later checks use a consistent type.
            order = int(order)
            if order in seen_orders:
                raise ValueError(f"评估样本 order 重复: {order}")
            seen_orders.add(order)
            # Stores the `normalized_item['order']` for use by the remaining steps in this flow.
            normalized_item["order"] = order

            processed_path = normalized_item.get("processed_path")
            if processed_path:
                normalized_item["processed_path"] = os.path.abspath(str(processed_path))

            # Stores the `source_file` for use by the remaining steps in this flow.
            source_file = resolve_sample_source_path(normalized_item)
            if source_file:
                normalized_item["source_file"] = source_file

            normalized.append(normalized_item)

        # Returns `normalized` to the caller for the next stage.
        return normalized

    @staticmethod
    def _build_processed_sample_record(
        item: Dict[str, Any],
        # Continues the surrounding statement or collection definition.
        split_name: str,
    ) -> Dict[str, Any]:
        sample_order = int(item["order"])
        source_file = item.get("source_file")
        # Reads `processed_path` from the mapping into the `processed_path`.
        processed_path = item.get("processed_path")
        params = item.get("params")
        if not isinstance(params, dict):
            params = {}

        # Stores the `record` for use by the remaining steps in this flow.
        record = dict(item)
        record.pop("sample_id", None)
        record.update(
            {
                # Completes the multi-line `collection` arguments or elements.
                "order": sample_order,
                "split": split_name,
                "source_file": source_file or None,
                "processed_path": processed_path or None,
                # Completes the multi-line `item.get` arguments or elements.
                "path_type": item.get("path_type"),
                "params": dict(params),
                "source_exists": bool(source_file and os.path.exists(source_file)),
                "processed_exists": bool(processed_path and os.path.exists(processed_path)),
                # Completes the multi-line `collection` arguments or elements.
                "evaluation_status": "processed",
            }
        )
        return record

    # Continues the surrounding statement or collection definition.
    @classmethod
    def _build_evaluated_sample_record(
        cls,
        item: Dict[str, Any],
        # Continues the surrounding statement or collection definition.
        split_name: str,
        evaluated_result_path: Optional[str],
        target_result_path: Optional[str],
    ) -> Dict[str, Any]:
        # Stores the `record` for use by the remaining steps in this flow.
        record = cls._build_processed_sample_record(item=item, split_name=split_name)
        record.update(
            {
                "evaluated_result_path": evaluated_result_path,
                # Completes the multi-line `collection` arguments or elements.
                "target_result_path": target_result_path,
                "evaluated_result_exists": bool(
                    evaluated_result_path and os.path.exists(evaluated_result_path)
                ),
                # Completes the multi-line `bool` arguments or elements.
                "target_result_exists": bool(
                    target_result_path and os.path.exists(target_result_path)
                ),
                "evaluation_status": "evaluated",
            # Completes the multi-line `collection` arguments or elements.
            }
        )
        return record

    @staticmethod
    # Defines `_collect_param_fields` for  collect param fields logic.
    def _collect_param_fields(sample_items: List[Dict[str, Any]]) -> List[str]:
        reserved = {
            "order",
            "sample_id",
            # Completes the multi-line `collection` arguments or elements.
            "params",
            "processed_path",
            "resolved_path",
            "source_file",
            # Completes the multi-line `collection` arguments or elements.
            "exists",
            "path_type",
            "subset",
            "split",
            # Completes the multi-line `collection` arguments or elements.
            "source_exists",
            "processed_exists",
            "evaluation_status",
            "evaluated_result_path",
            # Completes the multi-line `collection` arguments or elements.
            "evaluated_result_exists",
            "target_result_path",
            "target_result_exists",
        }
        # Initializes the `fields` as a collection for values from this processing stage.
        fields: List[str] = []
        for item in sample_items:
            params = item.get("params")
            if isinstance(params, dict):
                # Visits each item in `params.keys()` and applies the current processing step.
                for key in params.keys():
                    if key not in fields:
                        fields.append(str(key))
            for key in item.keys():
                # Checks `key in reserved or key == "file_path"` before entering this conditional branch.
                if key in reserved or key == "file_path":
                    continue
                if key not in fields:
                    fields.append(str(key))
        # Returns `fields` to the caller for the next stage.
        return fields

    def _write_samples_csv(
        self,
        sample_items: List[Dict[str, Any]],
        # Continues the surrounding statement or collection definition.
        csv_path: str,
    ) -> str:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        param_fields = self._collect_param_fields(sample_items)
        # Derives the `fieldnames` from the current values for downstream processing.
        fieldnames = ["order"] + param_fields + ["file_path"]

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            # Visits each item in `sample_items` and applies the current processing step.
            for item in sample_items:
                row = {
                    "order": int(item["order"]),
                    "file_path": resolve_sample_source_path(item),
                # Completes the multi-line `collection` arguments or elements.
                }

                params = item.get("params")
                if isinstance(params, dict):
                    for key in param_fields:
                        # Checks `key in params` before entering this conditional branch.
                        if key in params:
                            row[key] = params[key]

                for key in param_fields:
                    if key in row:
                        # Handles the data or control-flow case represented by this statement.
                        continue
                    if key in item:
                        row[key] = item[key]

                writer.writerow(row)

        # Returns `csv_path` to the caller for the next stage.
        return csv_path

    @staticmethod
    def _assert_evaluate_processed_sample_has_source_target(processed_path: str) -> None:
        payload = load_torch(processed_path, map_location="cpu", weights_only=False)
        # Validates `not isinstance(payload, dict)` and raises `ValueError` when the input cannot be processed safely.
        if not isinstance(payload, dict):
            raise ValueError(f"评估预处理样本格式非法: {processed_path}")
        if "y" not in payload:
            raise ValueError(f"评估预处理样本缺少真实目标 y: {processed_path}")

        # Stores the `meta` for use by the remaining steps in this flow.
        meta = payload.get("meta", {}) or {}
        if not isinstance(meta, dict):
            meta = {}

        target_source = str(meta.get("target_source") or "").strip().lower()
        # Stores the `prediction_view` for use by the remaining steps in this flow.
        prediction_view = str(meta.get("prediction_view") or "").strip().lower()
        if target_source in _DUMMY_TARGET_SOURCES or prediction_view in _DUMMY_PREDICTION_VIEWS:
            raise ValueError(
                "评估预处理生成了预测专用目标，无法计算真实指标: "
                # Completes the multi-line `ValueError` arguments or elements.
                f"path={processed_path}, target_source={target_source or '未标记'}, "
                f"prediction_view={prediction_view or '未标记'}"
            )

        target = payload["y"]
        # Checks `isinstance(target, torch.Tensor)` before entering this conditional branch.
        if isinstance(target, torch.Tensor):
            if target.numel() == 0:
                raise ValueError(f"评估预处理样本 y 为空: {processed_path}")
            return

        # Keeps the operation and its recovery path within the same exception boundary.
        try:
            if len(target) == 0:
                raise ValueError(f"评估预处理样本 y 为空: {processed_path}")
        except TypeError:
            # Ends this function and returns control to its caller.
            return

    @classmethod
    def _assert_evaluate_processed_samples_have_source_targets(
        cls,
        # Continues the surrounding statement or collection definition.
        processed_paths: List[str],
    ) -> None:
        for processed_path in processed_paths:
            cls._assert_evaluate_processed_sample_has_source_target(processed_path)

    # Defines `_prepare_processed_samples` for  prepare processed samples logic.
    def _prepare_processed_samples(
        self,
        sample_items: List[Dict[str, Any]],
        processed_dir: str,
    # Completes the multi-line `collection` arguments or elements.
    ) -> Tuple[List[Dict[str, Any]], str]:
        normalized = self._normalize_samples_evaluate(sample_items)
        raw_items = [
            item
            # Continues the surrounding statement or collection definition.
            for item in normalized
            if not item.get("processed_path") or not os.path.exists(item["processed_path"])
        ]

        if raw_items:
            # Builds the `missing_source` using the comprehension's filter and mapping rules.
            missing_source = [item["order"] for item in raw_items if not item.get("source_file")]
            if missing_source:
                raise ValueError(
                    "以下评估样本缺少 source_file"
                    # Completes the multi-line `ValueError` arguments or elements.
                    f"（兼容 file_path/resolved_path），无法执行原始样本预处理: {missing_source}"
                )

            raw_csv = os.path.join(processed_dir, "evaluate_samples_raw.csv")
            self._write_samples_csv(raw_items, raw_csv)
            # Stores the data reader for use by the remaining steps in this flow.
            reader = ReaderFactory.get_reader(
                self.runtime_config,
                flag="evaluate",
                sample_table_path_override=raw_csv,
            # Completes the multi-line `ReaderFactory.get_reader` arguments or elements.
            )
            raw_processed_paths = DataPipeline.call_process_with_reader(
                reader=reader,
                processor=self.processor,
                # Completes the multi-line `DataPipeline.call_process_with_reader` arguments or elements.
                sample_table_path=raw_csv,
                stage="evaluate",
                processed_dir=processed_dir,
                logger=self.logger,
            # Completes the multi-line `DataPipeline.call_process_with_reader` arguments or elements.
            )
            if not raw_processed_paths:
                raise ValueError("评估预处理未生成任何 processed 样本")

            self._assert_evaluate_processed_samples_have_source_targets(raw_processed_paths)

            # Validates `not self.norm_params` and raises `ValueError` when the input cannot be processed safely.
            if not self.norm_params:
                raise ValueError("checkpoint 缺少 norm_params，无法执行评估预处理")

            norm_type = self.norm_params.get(
                "norm_type",
                # Completes the multi-line `self.runtime_config.get('processor', {}).get` arguments or elements.
                self.runtime_config.get("processor", {}).get("norm_type", "standardization"),
            )
            DataPipeline.normalize_data_with_existing_params(
                processed_paths=raw_processed_paths,
                # Completes the multi-line `DataPipeline.normalize_data_with_existing_params` arguments or elements.
                norm_params=self.norm_params,
                norm_type=norm_type,
                processor=self.processor,
                normalize_y=True,
            # Completes the multi-line `DataPipeline.normalize_data_with_existing_params` arguments or elements.
            )

            raw_processed_map: Dict[int, str] = {}
            for path in raw_processed_paths:
                file_name = os.path.basename(path)
                # Stores the `match` for use by the remaining steps in this flow.
                match = re.match(r"sample_(\d+)\.pt$", file_name)
                if match:
                    raw_processed_map[int(match.group(1))] = os.path.abspath(path)

            for item in normalized:
                # Checks `` before entering this conditional branch.
                if item.get("processed_path") and os.path.exists(item["processed_path"]):
                    continue
                item["processed_path"] = raw_processed_map.get(int(item["order"]))
                if not item["processed_path"]:
                    # Raises `ValueError` to report why this path cannot continue.
                    raise ValueError(f"样本 {item['order']} 预处理后未找到 processed_path")
        else:
            for item in normalized:
                item["processed_path"] = os.path.abspath(str(item["processed_path"]))

        # Stores the `all_csv` for use by the remaining steps in this flow.
        all_csv = os.path.join(processed_dir, "evaluate_samples_all.csv")
        self._write_samples_csv(normalized, all_csv)
        return normalized, all_csv

    def _build_dataset_from_processed_items(self, processed_items: List[Dict[str, Any]]):
        # Builds the `sample_entries` using the comprehension's filter and mapping rules.
        sample_entries = [
            {
                "sample_id": int(item["order"]),
                "processed_path": item["processed_path"],
                # Completes the multi-line `item.get` arguments or elements.
                "source_file": item.get("source_file"),
            }
            for item in processed_items
        ]
        # Stores the dataset for use by the remaining steps in this flow.
        dataset = FullSampleDataset.from_sample_entries(
            config=self.runtime_config,
            sample_entries=sample_entries,
        )

        # Returns `dataset` to the caller for the next stage.
        return dataset

    def _build_prediction_adapter(
        self,
        sample_table_path: str,
    # Completes the multi-line `collection` arguments or elements.
    ) -> Tuple[Optional[BasePredictAdapter], Optional[Dict[str, Any]]]:
        if not hasattr(self.processor, "build_prediction_adapter"):
            return None, build_failure_entry(
                stage="evaluation_writeback",
                # Completes the multi-line `type` arguments or elements.
                component=type(self.processor).__name__,
                message="当前处理器不支持评估结果写回",
            )

        try:
            # Stores the data reader for use by the remaining steps in this flow.
            reader = ReaderFactory.get_reader(
                self.runtime_config,
                flag="evaluate",
                sample_table_path_override=sample_table_path,
            # Completes the multi-line `ReaderFactory.get_reader` arguments or elements.
            )
        except Exception as exc:
            return None, build_failure_entry(
                stage="evaluation_writeback",
                # Completes the multi-line `collection` arguments or elements.
                component="ReaderFactory",
                message="构建评估写回 reader 失败",
                error=exc,
            )

        # Keeps the operation and its recovery path within the same exception boundary.
        try:
            adapter = self.processor.build_prediction_adapter(
                reader=reader,
                config=self.runtime_config,
                # Completes the multi-line `self.processor.build_prediction_adapter` arguments or elements.
                is_geometry_varying=self.is_geometry_varying,
                resolve_reference_file=self._resolve_reference_file,
                logger=self.logger,
            )
        # Continues the surrounding statement or collection definition.
        except Exception as exc:
            return None, build_failure_entry(
                stage="evaluation_writeback",
                component=type(self.processor).__name__,
                # Completes the multi-line `collection` arguments or elements.
                message="构建评估写回适配器失败",
                error=exc,
            )
        return adapter, None

    # 合并评估结果与目标结果的写回摘要。
    def _merge_writeback_reports(
        self,
        evaluated_report: Dict[str, Any],
        target_report: Dict[str, Any],
    # Completes the multi-line `collection` arguments or elements.
    ) -> Dict[str, Any]:
        """合并评估结果与目标结果的写回摘要。"""
        statuses = {
            evaluated_report.get("status", "idle"),
            target_report.get("status", "idle"),
        # Completes the multi-line `collection` arguments or elements.
        }
        if "failed" in statuses:
            status = "failed"
        elif "partial" in statuses:
            # Stores the `status` for use by the remaining steps in this flow.
            status = "partial"
        elif statuses == {"idle"}:
            status = "idle"
        else:
            # Stores the `status` for use by the remaining steps in this flow.
            status = "success"

        return {
            "strict": self.strict_writeback,
            "status": status,
            # Completes the multi-line `collection` arguments or elements.
            "evaluated": evaluated_report,
            "target": target_report,
        }

    def _handle_writeback_report(
        # Continues the surrounding statement or collection definition.
        self,
        label: str,
        report: Dict[str, Any],
    ) -> None:
        """记录写回摘要，并在严格模式下直接失败。"""
        # Stores the run logger for use by the remaining steps in this flow.
        logger = getattr(self, "logger", None)
        if logger is not None:
            if report["status"] == "success":
                logger.info(
                    # Completes the multi-line `logger.info` arguments or elements.
                    "%s写回完成: success=%s expected=%s",
                    label,
                    report["success_count"],
                    report["expected_count"],
                # Completes the multi-line `logger.info` arguments or elements.
                )
            elif report["status"] == "idle":
                logger.info("%s写回未执行", label)
            else:
                # Records the current execution state to make progress and failures traceable.
                logger.warning(
                    "%s写回存在降级: status=%s success=%s expected=%s",
                    label,
                    report["status"],
                    # Completes the multi-line `logger.warning` arguments or elements.
                    report["success_count"],
                    report["expected_count"],
                )

        if self.strict_writeback and report["status"] in {"partial", "failed"}:
            # Raises `RuntimeError` to report why this path cannot continue.
            raise RuntimeError(
                f"{label}写回失败: status={report['status']} "
                f"success={report['success_count']}/{report['expected_count']}"
            )

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _rename_output_files(
        output_paths: List[str],
        output_prefix: str,
    # Continues the surrounding statement or collection definition.
    ) -> List[str]:
        renamed_paths: List[str] = []
        for path in output_paths:
            directory = os.path.dirname(path)
            # Stores the `file_name` for use by the remaining steps in this flow.
            file_name = os.path.basename(path)
            if file_name.startswith("predicted_"):
                target_name = output_prefix + "_" + file_name[len("predicted_") :]
                target_path = os.path.join(directory, target_name)
                # Calls `os.replace` to perform the operation required at this stage.
                os.replace(path, target_path)
                renamed_paths.append(target_path)
            else:
                renamed_paths.append(path)
        # Returns `renamed_paths` to the caller for the next stage.
        return renamed_paths

    @staticmethod
    def _extract_output_path_mapping(output_paths: List[str]) -> Dict[int, str]:
        mapping: Dict[int, str] = {}
        # Visits each item in `output_paths` and applies the current processing step.
        for path in output_paths:
            normalized_path = str(path).replace("\\", "/")
            file_name = os.path.basename(path)

            flat_match = re.search(r"(?:predicted|evaluated|target)_(\d+)\.", file_name)
            # Checks `flat_match` before entering this conditional branch.
            if flat_match:
                mapping[int(flat_match.group(1))] = path
                continue

            temporal_match = re.search(r"/sample_(\d+)(?:/|$)", normalized_path)
            # Checks `temporal_match` before entering this conditional branch.
            if temporal_match:
                sample_id = int(temporal_match.group(1))
                mapping.setdefault(sample_id, path)
        return mapping

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _build_sample_prediction_map(
        predictions: Any,
        processed_items: List[Dict[str, Any]],
    # Completes the multi-line `collection` arguments or elements.
    ) -> Optional[Dict[int, Any]]:
        if isinstance(predictions, dict):
            return {
                int(sample_id): payload
                # Completes the multi-line `collection` arguments or elements.
                for sample_id, payload in predictions.items()
            }

        if isinstance(predictions, tuple):
            predictions = list(predictions)

        # Checks `isinstance(predictions, list)` before entering this conditional branch.
        if isinstance(predictions, list):
            if len(predictions) != len(processed_items):
                return None
            return {
                # Completes the multi-line `int` arguments or elements.
                int(item["order"]): predictions[index]
                for index, item in enumerate(processed_items)
            }

        try:
            # Stores the `total` for use by the remaining steps in this flow.
            total = len(predictions)
        except TypeError:
            return None

        if total != len(processed_items):
            # Returns `None` to the caller for the next stage.
            return None

        return {
            int(item["order"]): predictions[index]
            for index, item in enumerate(processed_items)
        # Continues the surrounding statement or collection definition.
        }

    def _write_result_files(
        self,
        predictions,
        # Continues the surrounding statement or collection definition.
        sample_table_path: str,
        processed_items: List[Dict[str, Any]],
        output_dir: str,
        output_prefix: str,
    # Completes the multi-line `collection` arguments or elements.
    ) -> Tuple[Dict[int, str], Dict[str, Any]]:
        label_map = {
            "evaluated": "评估结果",
            "target": "目标结果",
        # Completes the multi-line `collection` arguments or elements.
        }
        label = label_map.get(output_prefix, f"{output_prefix}结果")
        expected_count = len(processed_items)

        if self.task_type != "field":
            # Stores the `report` for use by the remaining steps in this flow.
            report = summarize_writeback_failures(
                [],
                strict=self.strict_writeback,
                expected_count=0,
                # Completes the multi-line `summarize_writeback_failures` arguments or elements.
                success_count=0,
            )
            self._handle_writeback_report(label, report)
            return {}, report

        # Initializes the `failures` as a collection for values from this processing stage.
        failures: List[Dict[str, Any]] = []
        adapter_result = self._build_prediction_adapter(sample_table_path)
        if isinstance(adapter_result, tuple):
            adapter, adapter_failure = adapter_result
        # Continues the surrounding statement or collection definition.
        else:
            # 兼容旧测试或外部 monkeypatch 直接返回 adapter 的写法。
            adapter, adapter_failure = adapter_result, None
        if adapter_failure is not None:
            failures.append(adapter_failure)
        # Checks `adapter is None` before entering this conditional branch.
        if adapter is None:
            report = summarize_writeback_failures(
                failures,
                strict=self.strict_writeback,
                # Completes the multi-line `summarize_writeback_failures` arguments or elements.
                expected_count=expected_count,
                success_count=0,
            )
            self._handle_writeback_report(label, report)
            # Returns `{}, report` to the caller for the next stage.
            return {}, report

        source_sample_files = [item.get("source_file", "") for item in processed_items]
        if self.is_geometry_varying and any(not path for path in source_sample_files):
            missing_orders = [
                # Completes the multi-line `int` arguments or elements.
                int(item["order"])
                for item in processed_items
                if not item.get("source_file")
            ]
            # Adds the current value to `failures` for later aggregation.
            failures.append(
                build_failure_entry(
                    stage="evaluation_writeback",
                    component=type(adapter).__name__,
                    # Completes the multi-line `build_failure_entry` arguments or elements.
                    message="变几何评估写回缺少源样本文件",
                    details={"missing_orders": missing_orders},
                )
            )
            # Stores the `report` for use by the remaining steps in this flow.
            report = summarize_writeback_failures(
                failures,
                strict=self.strict_writeback,
                expected_count=expected_count,
                # Completes the multi-line `summarize_writeback_failures` arguments or elements.
                success_count=0,
            )
            self._handle_writeback_report(label, report)
            return {}, report

        # Keeps the operation and its recovery path within the same exception boundary.
        try:
            output_paths = adapter.process_outputs(
                predictions=predictions,
                samples_csv=sample_table_path,
                # Completes the multi-line `adapter.process_outputs` arguments or elements.
                output_dir=output_dir,
                source_sample_files=source_sample_files,
            )
        except Exception as exc:
            # Adds the current value to `failures` for later aggregation.
            failures.append(
                build_failure_entry(
                    stage="evaluation_writeback",
                    component=type(adapter).__name__,
                    # Completes the multi-line `build_failure_entry` arguments or elements.
                    message="评估写回执行失败",
                    error=exc,
                )
            )
            # Stores the `report` for use by the remaining steps in this flow.
            report = summarize_writeback_failures(
                failures,
                strict=self.strict_writeback,
                expected_count=expected_count,
                # Completes the multi-line `summarize_writeback_failures` arguments or elements.
                success_count=0,
            )
            self._handle_writeback_report(label, report)
            return {}, report

        # Checks `not output_paths` before entering this conditional branch.
        if not output_paths:
            failures.append(
                build_failure_entry(
                    stage="evaluation_writeback",
                    # Completes the multi-line `type` arguments or elements.
                    component=type(adapter).__name__,
                    message="写回适配器未生成任何结果文件",
                )
            )
            # Stores the `report` for use by the remaining steps in this flow.
            report = summarize_writeback_failures(
                failures,
                strict=self.strict_writeback,
                expected_count=expected_count,
                # Completes the multi-line `summarize_writeback_failures` arguments or elements.
                success_count=0,
            )
            self._handle_writeback_report(label, report)
            return {}, report

        # Keeps the operation and its recovery path within the same exception boundary.
        try:
            output_paths = self._rename_output_files(
                output_paths=output_paths,
                output_prefix=output_prefix,
            # Completes the multi-line `self._rename_output_files` arguments or elements.
            )
            output_path_mapping = self._extract_output_path_mapping(output_paths)
        except Exception as exc:
            failures.append(
                # Completes the multi-line `build_failure_entry` arguments or elements.
                build_failure_entry(
                    stage="evaluation_writeback",
                    component=type(adapter).__name__,
                    message="整理评估写回结果文件失败",
                    # Completes the multi-line `build_failure_entry` arguments or elements.
                    error=exc,
                )
            )
            report = summarize_writeback_failures(
                # Completes the multi-line `summarize_writeback_failures` arguments or elements.
                failures,
                strict=self.strict_writeback,
                expected_count=expected_count,
                success_count=0,
            # Completes the multi-line `summarize_writeback_failures` arguments or elements.
            )
            self._handle_writeback_report(label, report)
            return {}, report

        if len(output_path_mapping) < expected_count:
            # Builds the `expected_orders` using the comprehension's filter and mapping rules.
            expected_orders = {int(item["order"]) for item in processed_items}
            mapped_orders = {int(order) for order in output_path_mapping.keys()}
            missing_orders = sorted(expected_orders - mapped_orders)
            failures.append(
                # Completes the multi-line `build_failure_entry` arguments or elements.
                build_failure_entry(
                    stage="evaluation_writeback",
                    component=type(adapter).__name__,
                    message="部分评估样本未生成可追踪的结果文件",
                    # Completes the multi-line `collection` arguments or elements.
                    details={
                        "missing_orders": missing_orders,
                        "output_file_count": len(output_paths),
                        "mapped_file_count": len(output_path_mapping),
                    # Completes the multi-line `collection` arguments or elements.
                    },
                )
            )

        report = summarize_writeback_failures(
            # Completes the multi-line `summarize_writeback_failures` arguments or elements.
            failures,
            strict=self.strict_writeback,
            expected_count=expected_count,
            success_count=len(output_path_mapping),
        # Completes the multi-line `summarize_writeback_failures` arguments or elements.
        )
        self._handle_writeback_report(label, report)
        return output_path_mapping, report

    def run(self, samples_evaluate: List[Dict[str, Any]]) -> Tuple[Dict, List[Dict], List[Dict]]:
        # Reads `evaluate` from the mapping into the `evaluate_cfg` with its declared fallback.
        evaluate_cfg = self.runtime_config.get("evaluate", {})
        split_name = str(evaluate_cfg.get("split", "custom") or "custom")
        evaluate_dir = evaluate_cfg["output_dir"]
        processed_dir = os.path.join(evaluate_dir, "test_processed")
        # Stores the `results_dir` for use by the remaining steps in this flow.
        results_dir = os.path.join(evaluate_dir, "results")
        evaluated_results_dir = os.path.join(results_dir, "evaluated")
        target_results_dir = os.path.join(results_dir, "target")
        os.makedirs(evaluate_dir, exist_ok=True)
        # Calls `os.makedirs` to perform the operation required at this stage.
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(evaluated_results_dir, exist_ok=True)
        os.makedirs(target_results_dir, exist_ok=True)

        # Stores the `(processed_items, sample_table_path)` for use by the remaining steps in this flow.
        processed_items, sample_table_path = self._prepare_processed_samples(
            sample_items=samples_evaluate,
            processed_dir=processed_dir,
        )
        # Builds the `samples_evaluate_processed` using the comprehension's filter and mapping rules.
        samples_evaluate_processed = [
            self._build_processed_sample_record(item=item, split_name=split_name)
            for item in processed_items
        ]
        # Stores the dataset for use by the remaining steps in this flow.
        dataset = self._build_dataset_from_processed_items(processed_items)

        if self.logger is not None:
            self.logger.info("开始初始化评估器")
            self.logger.info("评估样本数: %s", len(processed_items))
            # Records the current execution state to make progress and failures traceable.
            self.logger.info("评估处理目录: %s", processed_dir)

        evaluator_config = dict(self.runtime_config)
        evaluator_config["__eval_norm_params__"] = self.norm_params
        evaluator = Evaluator(self.model, evaluator_config)
        # Stores the `scatter_save_path` for use by the remaining steps in this flow.
        scatter_save_path = os.path.join(evaluate_dir, "evaluation_result.png")
        scatter_csv_path = os.path.join(evaluate_dir, "scatter_all.csv")
        is_full_rollout_eval = should_use_full_autoregressive_rollout(
            self.runtime_config
        # Completes the multi-line `should_use_full_autoregressive_rollout` arguments or elements.
        )
        evaluate_fn = (
            evaluator.evaluate_autoregressive_full_rollout
            if is_full_rollout_eval else evaluator.evaluate
        # Continues the surrounding statement or collection definition.
        )
        metrics, predictions, targets = evaluate_fn(
            dataset,
            plot_scatter=bool(evaluate_cfg.get("plot_scatter", True)),
            # Completes the multi-line `evaluate_fn` arguments or elements.
            scatter_save_path=scatter_save_path,
            scatter_csv_path=scatter_csv_path,
        )
        evaluate_log_path = evaluator.write_evaluation_log(
            # Completes the multi-line `evaluator.write_evaluation_log` arguments or elements.
            evaluate_dir=evaluate_dir,
            metrics=metrics,
            scatter_save_path=scatter_save_path,
            scatter_csv_path=scatter_csv_path,
        # Completes the multi-line `evaluator.write_evaluation_log` arguments or elements.
        )
        if self.logger is not None:
            evaluator.log_evaluation_summary(
                logger=self.logger,
                # Completes the multi-line `evaluator.log_evaluation_summary` arguments or elements.
                metrics=metrics,
                scatter_save_path=scatter_save_path,
                scatter_csv_path=scatter_csv_path,
                evaluate_log_path=evaluate_log_path,
            # Completes the multi-line `evaluator.log_evaluation_summary` arguments or elements.
            )

        merged_payload = evaluator.last_merged_payload or {}
        scatter_data = evaluator.last_scatter_data or evaluator.build_scatter_data(merged_payload)
        predictions_for_writeback = (
            # Completes the multi-line `merged_payload.get` arguments or elements.
            merged_payload.get("sample_predictions")
            or self._build_sample_prediction_map(predictions, processed_items)
            or predictions
        )
        # Stores the `targets_for_writeback` for use by the remaining steps in this flow.
        targets_for_writeback = (
            merged_payload.get("sample_targets")
            or self._build_sample_prediction_map(targets, processed_items)
            or targets
        # Continues the surrounding statement or collection definition.
        )

        evaluated_result_map, evaluated_writeback_report = self._write_result_files(
            predictions=predictions_for_writeback,
            sample_table_path=sample_table_path,
            # Completes the multi-line `self._write_result_files` arguments or elements.
            processed_items=samples_evaluate_processed,
            output_dir=evaluated_results_dir,
            output_prefix="evaluated",
        )
        # Stores the `(target_result_map, target_writeback_report)` for use by the remaining steps in this flow.
        target_result_map, target_writeback_report = self._write_result_files(
            predictions=targets_for_writeback,
            sample_table_path=sample_table_path,
            processed_items=samples_evaluate_processed,
            # Completes the multi-line `self._write_result_files` arguments or elements.
            output_dir=target_results_dir,
            output_prefix="target",
        )
        combined_writeback_report = self._merge_writeback_reports(
            # Completes the multi-line `self._merge_writeback_reports` arguments or elements.
            evaluated_writeback_report,
            target_writeback_report,
        )
        self.last_writeback_report = combined_writeback_report

        # Initializes the `samples_evaluated` as a collection for values from this processing stage.
        samples_evaluated: List[Dict[str, Any]] = []
        for item in samples_evaluate_processed:
            sample_order = int(item["order"])
            result_path = evaluated_result_map.get(sample_order)
            # Stores the `target_path` for use by the remaining steps in this flow.
            target_path = target_result_map.get(sample_order)
            samples_evaluated.append(
                self._build_evaluated_sample_record(
                    item=item,
                    # Completes the multi-line `self._build_evaluated_sample_record` arguments or elements.
                    split_name=split_name,
                    evaluated_result_path=result_path,
                    target_result_path=target_path,
                )
            # Completes the multi-line `samples_evaluated.append` arguments or elements.
            )

        evaluate_info = {
            "split": split_name,
            "evaluate_log_path": evaluate_log_path,
            # Completes the multi-line `collection` arguments or elements.
            "metrics": metrics,
            "scatter_data": scatter_data,
            "processed_dir": processed_dir,
            "results_dir": results_dir,
            # Completes the multi-line `collection` arguments or elements.
            "evaluated_results_dir": evaluated_results_dir,
            "target_results_dir": target_results_dir,
            "sample_count": len(samples_evaluate),
            "processed_sample_count": len(samples_evaluate_processed),
            # Completes the multi-line `len` arguments or elements.
            "evaluated_sample_count": len(samples_evaluated),
            "evaluated_result_count": len(evaluated_result_map),
            "target_result_count": len(target_result_map),
            "writeback_report": combined_writeback_report,
        # Completes the multi-line `collection` arguments or elements.
        }
        return evaluate_info, samples_evaluate_processed, samples_evaluated
