from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
# Imports `json` for reuse in this module.
import json
import logging
import os
from pathlib import Path
# Imports `Any, Callable, Dict` from `typing` for use in this module.
from typing import Any, Callable, Dict, List, Optional
import xml.etree.ElementTree as ET

import numpy as np

from tfai.data.temporal_utils import extract_time_index_from_name, is_temporal_manifest_path, list_temporal_sequence_files
# Imports `SampleMap` from `tfai.readers.sample_map` for use in this module.
from tfai.readers.sample_map import SampleMap
from tfai.utils.path_utils import resolve_path_from_base
from tfai.utils.failure_report import build_failure_entry
from tfai.utils.logger import Logger

# Imports `CGNSPredictionWriter, CSVPredictionWriter, VTKPredictionWriter` from `writers` for use in this module.
from ..writers import (
    CGNSPredictionWriter,
    CSVPredictionWriter,
    VTKPredictionWriter,
# Continues the surrounding statement or collection definition.
)


class BasePredictAdapter(ABC):
    """预测后处理适配器基类。

    这里仅保留流程编排职责：
    1. 遍历样本并调用子类后处理
    2. 解析待写回的源文件
    3. 按文件类型分发到对应 writer
    """

    TARGET_CGNS_BASE_NAME = None
    VTK_FILE_EXTENSIONS = {
        # Completes the multi-line `collection` arguments or elements.
        ".vtk",
        ".vtu",
        ".vts",
        ".vtr",
        # Completes the multi-line `collection` arguments or elements.
        ".vti",
        ".vtp",
        ".pvtu",
        ".pvts",
        # Completes the multi-line `collection` arguments or elements.
        ".pvtr",
        ".pvti",
        ".pvtp",
        ".vtm",
        # Completes the multi-line `collection` arguments or elements.
        ".vtmb",
    }

    def __init__(
        self,
        # Continues the surrounding statement or collection definition.
        processor: Any,
        reader: Any,
        config: Dict,
        is_geometry_varying: bool,
        # Completes the multi-line `collection` arguments or elements.
        resolve_reference_file: Callable[[], str],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.processor = processor
        # Stores the data reader on the instance so other methods can reuse it.
        self.reader = reader
        self.config = config
        self.is_geometry_varying = is_geometry_varying
        self.resolve_reference_file = resolve_reference_file
        # Stores the run logger on the instance so other methods can reuse it.
        self.logger = logger
        self._reference_file_cache: Optional[str] = None

        self.cgns_writer = CGNSPredictionWriter(
            config=config,
            # Completes the multi-line `CGNSPredictionWriter` arguments or elements.
            logger=logger,
            target_base_name=self.TARGET_CGNS_BASE_NAME,
        )
        self.vtk_writer = VTKPredictionWriter(config=config, logger=logger)
        # Stores the `csv_writer` on the instance so other methods can reuse it.
        self.csv_writer = CSVPredictionWriter(config=config, logger=logger)
        self._sample_writeback_contexts: Dict[int, Dict[str, Any]] = {}
        self.writeback_failures: List[Dict[str, Any]] = []

    def _log(self, level: str, message: str, exc_info: bool = False) -> None:
        # Stores the run logger for use by the remaining steps in this flow.
        logger = self.logger
        if logger is None:
            logger = Logger.get_first_configured_logger(
                ["predict", "evaluate", "train"],
                # Completes the multi-line `Logger.get_first_configured_logger` arguments or elements.
                fallback_name=__name__,
            )
        log_fn = getattr(logger, level, logger.info)
        log_fn(message, exc_info=exc_info)

    # 读取训练产物声明的运动网格契约。
    def _configured_motion_contract(self) -> Dict[str, Any]:
        """读取训练产物声明的运动网格契约。"""
        task_cfg = self.config.get("task", {}) if isinstance(self.config, dict) else {}
        geometry_regime = str(task_cfg.get("geometry_regime") or "").strip().lower()
        return {
            # Completes the multi-line `collection` arguments or elements.
            "geometry_regime": geometry_regime,
            "displacement_effective_variables": list(
                task_cfg.get("displacement_effective_variables", []) or []
            ),
            # Completes the multi-line `task_cfg.get` arguments or elements.
            "displacement_logical_variable": task_cfg.get("displacement_logical_variable"),
            "displacement_source": task_cfg.get("displacement_source"),
        }

    def _default_writeback_context(self, sample_file: str) -> Dict[str, Any]:
        # Stores the `resolved` for use by the remaining steps in this flow.
        resolved = str(sample_file or "").strip()
        motion_contract = self._configured_motion_contract()
        return {
            "source_path": resolved,
            # Completes the multi-line `collection` arguments or elements.
            "template_file": resolved,
            "layout_kind": "single_file",
            "ordered_files": [resolved] if resolved else [],
            "step_relpaths": [],
            # Completes the multi-line `collection` arguments or elements.
            "time_values": [],
            "manifest_name": None,
            "geometry_regime": (
                "displaced_static_topology"
                # Completes the multi-line `collection` arguments or elements.
                if motion_contract["geometry_regime"] == "displaced_static_topology"
                else "static_geometry"
            ),
            "reference_point_coordinates": None,
            # Completes the multi-line `collection` arguments or elements.
            "temporal_point_coordinates": None,
            "displacement_effective_variables": motion_contract[
                "displacement_effective_variables"
            ],
            # Completes the multi-line `collection` arguments or elements.
            "displacement_logical_variable": motion_contract[
                "displacement_logical_variable"
            ],
            "displacement_source": motion_contract["displacement_source"],
        # Completes the multi-line `collection` arguments or elements.
        }

    def _register_writeback_failure(
        self,
        sample_id: int,
        # Continues the surrounding statement or collection definition.
        message: str,
        *,
        details: Optional[Dict[str, Any]] = None,
        error: Optional[BaseException] = None,
    # Continues the surrounding statement or collection definition.
    ) -> None:
        self.writeback_failures.append(
            build_failure_entry(
                stage="prediction_writeback",
                # Completes the multi-line `type` arguments or elements.
                component=type(self).__name__,
                message=f"样本 {sample_id}: {message}",
                error=error,
                details=details,
            # Completes the multi-line `build_failure_entry` arguments or elements.
            )
        )

    def _augment_motion_writeback_context(
        self,
        # Continues the surrounding statement or collection definition.
        sample_id: int,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        source_path = str(context.get("source_path") or context.get("template_file") or "").strip()
        # Returns early when `not source_path or self.reader is None` makes the remaining work inapplicable.
        if not source_path or self.reader is None:
            return context

        configured_contract = self._configured_motion_contract()
        try:
            # Stores the `read_signature` for use by the remaining steps in this flow.
            read_signature = inspect.signature(self.reader.read)
            read_kwargs: Dict[str, Any] = {}
            if "sample_order" in read_signature.parameters:
                read_kwargs["sample_order"] = sample_id
            # Checks `"allow_missing_fields" in read_signature.parameters` before entering this conditional branch.
            if "allow_missing_fields" in read_signature.parameters:
                read_kwargs["allow_missing_fields"] = True
            if "max_time_steps" in read_signature.parameters:
                # 写回仅需参考构型，不能为此读取未来真实网格坐标。
                read_kwargs["max_time_steps"] = 1
            # Stores the current data for use by the remaining steps in this flow.
            data = self.reader.read(source_path, **read_kwargs)
        except Exception as exc:
            self._log("warning", f"样本 {sample_id} 解析运动网格写回上下文失败: {str(exc)}")
            return context

        # Stores the `data_dict` for use by the remaining steps in this flow.
        data_dict = data.get_data() if hasattr(data, "get_data") else {}
        motion_meta = data_dict.get("GeometryMotion") or {}
        source_geometry_regime = (
            getattr(data, "get_geometry_regime", lambda: None)()
            # Completes the multi-line `motion_meta.get` arguments or elements.
            or motion_meta.get("geometry_regime")
            or context.get("geometry_regime")
            or "static_geometry"
        )
        # Stores the `geometry_regime` for use by the remaining steps in this flow.
        geometry_regime = (
            "displaced_static_topology"
            if configured_contract["geometry_regime"] == "displaced_static_topology"
            or str(source_geometry_regime).strip().lower() == "displaced_static_topology"
            # Completes the multi-line `str` arguments or elements.
            else str(source_geometry_regime)
        )
        if geometry_regime != "displaced_static_topology":
            context["geometry_regime"] = str(geometry_regime)
            # Returns `context` to the caller for the next stage.
            return context

        reference_points = data_dict.get("PointCoordinates")
        displacement_effective_variables = list(
            motion_meta.get("displacement_effective_variables")
            # Completes the multi-line `list` arguments or elements.
            or configured_contract["displacement_effective_variables"]
            or context.get("displacement_effective_variables")
            or []
        )
        # Stores the `displacement_logical_variable` for use by the remaining steps in this flow.
        displacement_logical_variable = (
            motion_meta.get("displacement_logical_name")
            or configured_contract["displacement_logical_variable"]
            or context.get("displacement_logical_variable")
        # Continues the surrounding statement or collection definition.
        )
        displacement_source = (
            motion_meta.get("displacement_source")
            or configured_contract["displacement_source"]
            # Completes the multi-line `context.get` arguments or elements.
            or context.get("displacement_source")
        )
        context.update(
            {
                # Completes the multi-line `collection` arguments or elements.
                "geometry_regime": "displaced_static_topology",
                "reference_point_coordinates": None if reference_points is None else np.asarray(reference_points),
                # 时序未来坐标属于标签信息，预测写回只能使用参考构型和模型位移输出。
                "temporal_point_coordinates": None,
                "displacement_effective_variables": displacement_effective_variables,
                # Completes the multi-line `collection` arguments or elements.
                "displacement_logical_variable": displacement_logical_variable,
                "displacement_source": displacement_source,
            }
        )
        # 拓扑不变的运动网格每一步都可复用首帧模板，不依赖未来源文件。
        # Stores the `template_file` for use by the remaining steps in this flow.
        template_file = str(context.get("template_file") or "").strip()
        if template_file:
            context["ordered_files"] = [template_file]
        return context

    # Defines `_build_motion_writeback_point_coordinates` for  build motion writeback point coordinates logic.
    def _build_motion_writeback_point_coordinates(
        self,
        sample_id: int,
        step_fields: Dict[str, np.ndarray],
        # Continues the surrounding statement or collection definition.
        step_index: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        del step_index
        context = self._sample_writeback_contexts.get(int(sample_id)) or {}
        # Returns early when `` makes the remaining work inapplicable.
        if str(context.get("geometry_regime") or "").strip().lower() != "displaced_static_topology":
            return None

        reference_points = context.get("reference_point_coordinates")
        if reference_points is None:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("缺少参考点坐标，无法根据预测位移更新网格")

        displacement_names = list(context.get("displacement_effective_variables") or [])
        if not displacement_names:
            raise ValueError("缺少位移分量字段定义，无法根据预测位移更新网格")

        # Builds the `missing_names` using the comprehension's filter and mapping rules.
        missing_names = [name for name in displacement_names if name not in step_fields]
        if missing_names:
            raise ValueError(f"预测结果缺少位移字段: {missing_names}")

        reference = np.asarray(reference_points, dtype=np.float64)
        # Initializes the `displacement_components` as a collection for values from this processing stage.
        displacement_components: List[np.ndarray] = []
        for field_name in displacement_names:
            values = np.asarray(step_fields[field_name], dtype=np.float64).reshape(-1)
            if values.shape[0] != reference.shape[0]:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(
                    f"位移字段 {field_name} 与参考点数不匹配: actual={values.shape[0]}, expected={reference.shape[0]}"
                )
            displacement_components.append(values)

        # Stores the `displacement` for use by the remaining steps in this flow.
        displacement = np.stack(displacement_components, axis=-1)
        updated_points = np.asarray(reference, dtype=np.float64).copy()
        coord_dim = min(updated_points.shape[1], displacement.shape[1])
        updated_points[:, :coord_dim] = reference[:, :coord_dim] + displacement[:, :coord_dim]
        # Returns `updated_points` to the caller for the next stage.
        return updated_points

    @staticmethod
    def _safe_relpath(path: str, base_dir: str) -> str:
        try:
            # Stores the `rel` for use by the remaining steps in this flow.
            rel = os.path.relpath(path, base_dir)
        except ValueError:
            rel = os.path.basename(path)
        return str(rel)

    # Defines `_parse_temporal_manifest_context` for  parse temporal manifest context logic.
    def _parse_temporal_manifest_context(self, source_path: str) -> Dict[str, Any]:
        manifest_path = Path(source_path).resolve()
        lower_path = str(manifest_path).lower()

        if lower_path.endswith(".pvd"):
            # Stores the `root` for use by the remaining steps in this flow.
            root = ET.parse(manifest_path).getroot()
            entries = []
            for order, dataset in enumerate(root.iter()):
                if not str(dataset.tag).lower().endswith("dataset"):
                    # Handles the data or control-flow case represented by this statement.
                    continue
                file_name = dataset.attrib.get("file")
                if not file_name:
                    raise ValueError(f"PVD manifest contains Dataset without file: {manifest_path}")
                # Reads `timestep` from the mapping into the `timestep_text`.
                timestep_text = dataset.attrib.get("timestep")
                timestep = float(order) if timestep_text is None else float(timestep_text)
                step_path = (manifest_path.parent / file_name).resolve()
                entries.append((timestep, order, str(step_path), str(file_name)))
            # Validates `not entries` and raises `ValueError` when the input cannot be processed safely.
            if not entries:
                raise ValueError(f"No Dataset entries found in PVD manifest: {manifest_path}")
            entries.sort(key=lambda item: (item[0], item[1], item[2].lower()))
            return {
                # Completes the multi-line `str` arguments or elements.
                "source_path": str(manifest_path),
                "template_file": entries[0][2],
                "layout_kind": "temporal_manifest_pvd",
                "ordered_files": [item[2] for item in entries],
                # Completes the multi-line `str(item[3]).replace('/', os.sep).replace` arguments or elements.
                "step_relpaths": [str(item[3]).replace("/", os.sep).replace("\\", os.sep) for item in entries],
                "time_values": [float(item[0]) for item in entries],
                "manifest_name": manifest_path.name,
            }

        # Checks `lower_path.endswith(".series")` before entering this conditional branch.
        if lower_path.endswith(".series"):
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            file_entries = payload.get("files")
            if not isinstance(file_entries, list) or not file_entries:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(f".series manifest does not contain a valid files list: {manifest_path}")
            entries = []
            for order, item in enumerate(file_entries):
                if not isinstance(item, dict):
                    # Raises `ValueError` to report why this path cannot continue.
                    raise ValueError(f".series manifest contains invalid file entry: {manifest_path}")
                file_name = item.get("name")
                if not file_name:
                    raise ValueError(f".series manifest contains file entry without name: {manifest_path}")
                # Converts the `time_value` to `float` so later checks use a consistent type.
                time_value = float(item.get("time", order))
                step_path = (manifest_path.parent / str(file_name)).resolve()
                entries.append((time_value, order, str(step_path), str(file_name)))
            entries.sort(key=lambda item: (item[0], item[1], item[2].lower()))
            # Returns `` to the caller for the next stage.
            return {
                "source_path": str(manifest_path),
                "template_file": entries[0][2],
                "layout_kind": "temporal_manifest_series",
                # Completes the multi-line `collection` arguments or elements.
                "ordered_files": [item[2] for item in entries],
                "step_relpaths": [str(item[3]).replace("/", os.sep).replace("\\", os.sep) for item in entries],
                "time_values": [float(item[0]) for item in entries],
                "manifest_name": manifest_path.name,
            # Completes the multi-line `collection` arguments or elements.
            }

        raise ValueError(f"Unsupported temporal manifest format: {manifest_path}")

    def _parse_temporal_directory_context(self, source_path: str) -> Dict[str, Any]:
        ordered_files = list_temporal_sequence_files(self.reader, source_path)
        # Validates `not ordered_files` and raises `ValueError` when the input cannot be processed safely.
        if not ordered_files:
            raise ValueError(f"No temporal step files found in {source_path}")

        base_dir = str(Path(source_path).resolve())
        step_relpaths = [self._safe_relpath(path, base_dir) for path in ordered_files]
        # Stores the `has_nested_subdirs` for use by the remaining steps in this flow.
        has_nested_subdirs = any(
            os.sep in rel_path or "/" in rel_path or "\\" in rel_path
            for rel_path in step_relpaths
        )
        # Initializes the `time_values` as a collection for values from this processing stage.
        time_values = []
        for index, rel_path in enumerate(step_relpaths):
            step_index = extract_time_index_from_name(Path(rel_path).name)
            time_values.append(float(step_index if step_index is not None else index))

        # Returns `` to the caller for the next stage.
        return {
            "source_path": base_dir,
            "template_file": ordered_files[0],
            "layout_kind": "temporal_step_directories" if has_nested_subdirs else "temporal_flat_files",
            # Completes the multi-line `collection` arguments or elements.
            "ordered_files": ordered_files,
            "step_relpaths": step_relpaths,
            "time_values": time_values,
            "manifest_name": None,
        # Completes the multi-line `collection` arguments or elements.
        }

    def _resolve_writeback_template_context(
        self,
        sample_file: str,
        # Continues the surrounding statement or collection definition.
        sample_id: int,
    ) -> Dict[str, Any]:
        resolved = str(sample_file or "").strip()
        if not resolved:
            # Returns `self._default_writeback_context(sample_file)` to the caller for the next stage.
            return self._default_writeback_context(sample_file)

        try:
            if is_temporal_manifest_path(resolved):
                return self._parse_temporal_manifest_context(resolved)
            # Returns early when `os.path.isdir(resolved)` makes the remaining work inapplicable.
            if os.path.isdir(resolved):
                return self._parse_temporal_directory_context(resolved)
            if os.path.isfile(resolved) and os.path.splitext(resolved)[1].lower() == ".cgns":
                temporal_context = self.cgns_writer.inspect_temporal_layout(
                    # Completes the multi-line `self.cgns_writer.inspect_temporal_layout` arguments or elements.
                    resolved,
                    sample_id,
                )
                if temporal_context:
                    # Returns `temporal_context` to the caller for the next stage.
                    return temporal_context
        except Exception as exc:
            self._log(
                "warning",
                # Completes the multi-line `str` arguments or elements.
                f"样本 {sample_id} 解析时序组织失败，回退为单文件模板: {str(exc)}",
            )

        return self._default_writeback_context(resolved)

    def process_outputs(
        # Continues the surrounding statement or collection definition.
        self,
        predictions: Any,
        samples_csv: str,
        output_dir: str,
        # Continues the surrounding statement or collection definition.
        source_sample_files: List[str],
    ) -> List[str]:
        """定义完整的预测后处理流程。"""
        self.writeback_failures = []
        if isinstance(predictions, dict):
            # Returns `` to the caller for the next stage.
            return self._process_dict_outputs(
                predictions=predictions,
                samples_csv=samples_csv,
                output_dir=output_dir,
                # Completes the multi-line `self._process_dict_outputs` arguments or elements.
                source_sample_files=source_sample_files,
            )

        sample_ids = SampleMap(samples_csv).get_all_samples()
        if len(predictions) == 0 or len(sample_ids) != len(predictions):
            # Returns `[]` to the caller for the next stage.
            return []

        return self.write_back_to_original(
            predictions=predictions,
            sample_ids=sample_ids,
            # Completes the multi-line `self.write_back_to_original` arguments or elements.
            sample_files=source_sample_files,
            output_dir=output_dir,
        )

    def _process_dict_outputs(
        # Continues the surrounding statement or collection definition.
        self,
        predictions: Dict[int, np.ndarray],
        samples_csv: str,
        output_dir: str,
        # Continues the surrounding statement or collection definition.
        source_sample_files: List[str],
    ) -> List[str]:
        """处理按 sample_id 组织的预测结果。"""
        os.makedirs(output_dir, exist_ok=True)
        output_paths: List[str] = []

        # Stores the `sample_map` for use by the remaining steps in this flow.
        sample_map = SampleMap(samples_csv)
        csv_sample_ids = sample_map.get_all_samples()
        base_dir = os.path.dirname(os.path.abspath(samples_csv))
        fallback_source_map = {
            # Completes the multi-line `int` arguments or elements.
            int(sample_id): str(sample_file)
            for sample_id, sample_file in zip(csv_sample_ids, source_sample_files)
            if sample_file
        }

        # Visits each item in `predictions.items()` and applies the current processing step.
        for sample_id, prediction in predictions.items():
            file_path = sample_map.get_file_path(int(sample_id))
            resolved_source = ""
            if file_path:
                # Stores the `resolved_source` for use by the remaining steps in this flow.
                resolved_source = resolve_path_from_base(base_dir, file_path)
            elif int(sample_id) in fallback_source_map:
                resolved_source = fallback_source_map[int(sample_id)]

            sample_file = self._get_sample_file(
                # Completes the multi-line `self._get_sample_file` arguments or elements.
                index=0,
                sample_id=int(sample_id),
                sample_files=[resolved_source] if resolved_source else [],
            )
            # Checks `not sample_file` before entering this conditional branch.
            if not sample_file:
                continue

            try:
                processed_fields = self.postprocess_prediction(
                    # Completes the multi-line `self.postprocess_prediction` arguments or elements.
                    prediction,
                    int(sample_id),
                    sample_file,
                )
                # Checks `processed_fields is None` before entering this conditional branch.
                if processed_fields is None:
                    self._log("warning", f"样本 {sample_id} 后处理失败，已跳过")
                    continue

                output_paths.extend(
                    # Completes the multi-line `self._write_processed_fields` arguments or elements.
                    self._write_processed_fields(
                        sample_id=int(sample_id),
                        sample_file=sample_file,
                        processed_fields=processed_fields,
                        # Completes the multi-line `self._write_processed_fields` arguments or elements.
                        output_dir=output_dir,
                    )
                )
            except Exception as exc:
                # Calls `self._log` to perform the operation required at this stage.
                self._log(
                    "error",
                    f"样本 {sample_id} 写回失败: {str(exc)}",
                    exc_info=True,
                # Completes the multi-line `self._log` arguments or elements.
                )

        return output_paths

    @abstractmethod
    def postprocess_prediction(
        # Continues the surrounding statement or collection definition.
        self,
        prediction: np.ndarray,
        sample_id: int,
        sample_file: str,
    # Completes the multi-line `collection` arguments or elements.
    ) -> Optional[Dict[str, np.ndarray]]:
        """对子类的单样本预测结果做后处理。"""

    def write_back_to_original(
        self,
        predictions: np.ndarray,
        # Continues the surrounding statement or collection definition.
        sample_ids: List[int],
        sample_files: List[str],
        output_dir: str,
    ) -> List[str]:
        """将预测结果写回原始格式文件。"""
        # Calls `os.makedirs` to perform the operation required at this stage.
        os.makedirs(output_dir, exist_ok=True)
        output_paths: List[str] = []

        for index, (sample_id, prediction) in enumerate(zip(sample_ids, predictions)):
            sample_file = self._get_sample_file(index, sample_id, sample_files)
            # Checks `not sample_file` before entering this conditional branch.
            if not sample_file:
                continue

            try:
                processed_fields = self.postprocess_prediction(
                    # Completes the multi-line `self.postprocess_prediction` arguments or elements.
                    prediction,
                    sample_id,
                    sample_file,
                )
                # Checks `processed_fields is None` before entering this conditional branch.
                if processed_fields is None:
                    self._log("warning", f"样本 {sample_id} 后处理失败，已跳过")
                    continue

                output_paths.extend(
                    # Completes the multi-line `self._write_processed_fields` arguments or elements.
                    self._write_processed_fields(
                        sample_id=sample_id,
                        sample_file=sample_file,
                        processed_fields=processed_fields,
                        # Completes the multi-line `self._write_processed_fields` arguments or elements.
                        output_dir=output_dir,
                    )
                )
            except Exception as exc:
                # Calls `self._log` to perform the operation required at this stage.
                self._log(
                    "error",
                    f"样本 {sample_id} 写回失败: {str(exc)}",
                    exc_info=True,
                # Completes the multi-line `self._log` arguments or elements.
                )

        return output_paths

    def _get_sample_file(
        self,
        # Continues the surrounding statement or collection definition.
        index: int,
        sample_id: int,
        sample_files: List[str],
    ) -> Optional[str]:
        """解析当前样本对应的源文件。"""
        # Checks `self.is_geometry_varying` before entering this conditional branch.
        if self.is_geometry_varying:
            if not sample_files or index >= len(sample_files):
                self._log("warning", f"样本 {sample_id} 缺少源文件路径，已跳过")
                return None
            # Stores the `context` for use by the remaining steps in this flow.
            context = self._resolve_writeback_template_context(sample_files[index], sample_id)
            context = self._augment_motion_writeback_context(sample_id, context)
            self._sample_writeback_contexts[int(sample_id)] = context
            return context.get("template_file")

        # Stores the `requires_reference` for use by the remaining steps in this flow.
        requires_reference = True
        requires_reference_fn = getattr(
            self.processor,
            "requires_fixed_geometry_predict_writeback_reference",
            # Completes the multi-line `getattr` arguments or elements.
            None,
        )
        if callable(requires_reference_fn):
            requires_reference = bool(requires_reference_fn())

        # Checks `` before entering this conditional branch.
        if sample_files and index < len(sample_files) and (
            not requires_reference or os.path.isfile(str(sample_files[index] or ""))
        ):
            sample_file = str(sample_files[index] or "").strip()
            if sample_file:
                context = self._resolve_writeback_template_context(sample_file, sample_id)
                # Stores the `context` for use by the remaining steps in this flow.
                context = self._augment_motion_writeback_context(sample_id, context)
                self._sample_writeback_contexts[int(sample_id)] = context
                return context.get("template_file")

        try:
            # Checks `self._reference_file_cache is None` before entering this conditional branch.
            if self._reference_file_cache is None:
                self._reference_file_cache = self.resolve_reference_file()
            context = self._resolve_writeback_template_context(
                self._reference_file_cache,
                # Completes the multi-line `self._resolve_writeback_template_context` arguments or elements.
                sample_id,
            )
            context = self._augment_motion_writeback_context(sample_id, context)
            self._sample_writeback_contexts[int(sample_id)] = context
            # Returns `context.get("template_file")` to the caller for the next stage.
            return context.get("template_file")
        except Exception as exc:
            self._log("error", f"样本 {sample_id} 解析参考文件失败: {str(exc)}")
            return None
    # Resolves the template source used to preserve the sample's original file layout.
    def _resolve_writeback_template_source(
        self,
        sample_file: str,
        sample_id: int,
    # Continues the surrounding statement or collection definition.
    ) -> Optional[str]:
        """Return the source file used as the write-back template for a sample."""
        context = self._resolve_writeback_template_context(sample_file, sample_id)
        return context.get("template_file")

    @staticmethod
    # Defines `_build_temporal_step_output_basename` for  build temporal step output basename logic.
    def _build_temporal_step_output_basename(sample_id: int, step_idx: int) -> str:
        return os.path.join(f"sample_{sample_id}", f"step_{step_idx:04d}")

    def _resolve_temporal_writeback_source_index(
        self,
        # Continues the surrounding statement or collection definition.
        *,
        step_index: int,
        available_steps: int,
    ) -> Optional[int]:
        """解析当前输出步应复用哪一帧已知几何或模板。"""
        # Returns early when `available_steps <= 0` makes the remaining work inapplicable.
        if available_steps <= 0:
            return None

        task_cfg = self.config.get("task", {}) if isinstance(self.config, dict) else {}
        mode = str(task_cfg.get("mode", "")).strip().lower()
        # Stores the `temporal_strategy` for use by the remaining steps in this flow.
        temporal_strategy = str(task_cfg.get("temporal_strategy", "direct")).strip().lower()

        if mode == "temporal" and temporal_strategy == "autoregressive":
            input_length = max(1, int(task_cfg.get("input_length", 1) or 1))
            source_index = input_length - 1 + int(step_index)
        # Continues the surrounding statement or collection definition.
        else:
            source_index = max(int(step_index) - 1, 0)

        return min(source_index, int(available_steps) - 1)

    def _resolve_temporal_step_template_file(
        # Continues the surrounding statement or collection definition.
        self,
        *,
        sample_id: int,
        sample_file: str,
        # Continues the surrounding statement or collection definition.
        step_index: int,
    ) -> str:
        """为时序输出选择当前步的模板文件。"""
        context = self._sample_writeback_contexts.get(int(sample_id)) or {}
        ordered_files = list(context.get("ordered_files") or [])
        # Stores the `source_index` for use by the remaining steps in this flow.
        source_index = self._resolve_temporal_writeback_source_index(
            step_index=step_index,
            available_steps=len(ordered_files),
        )
        # Returns early when `source_index is None` makes the remaining work inapplicable.
        if source_index is None:
            return sample_file
        return ordered_files[source_index]

    def _should_inherit_temporal_layout(
        # Continues the surrounding statement or collection definition.
        self,
        sample_id: int,
        sample_file: str,
    ) -> bool:
        # Stores the `context` for use by the remaining steps in this flow.
        context = self._sample_writeback_contexts.get(int(sample_id)) or {}
        layout_kind = str(context.get("layout_kind", "")).strip().lower()
        if not layout_kind.startswith("temporal_") or layout_kind == "temporal_single_cgns_file":
            return False
        # Stores the `template_file` for use by the remaining steps in this flow.
        template_file = str(context.get("template_file") or sample_file or "").strip()
        file_ext = os.path.splitext(template_file)[1].lower()
        return file_ext in self.VTK_FILE_EXTENSIONS or file_ext == ".cgns"

    @staticmethod
    # Defines `_normalize_time_values` for  normalize time values logic.
    def _normalize_time_values(time_values: List[Any], step_count: int) -> List[float]:
        normalized: List[float] = []
        for step_idx in range(step_count):
            if step_idx < len(time_values):
                # Keeps the operation and its recovery path within the same exception boundary.
                try:
                    normalized.append(float(time_values[step_idx]))
                    continue
                except (TypeError, ValueError):
                    # Handles the data or control-flow case represented by this statement.
                    pass
            normalized.append(float(step_idx))
        return normalized

    def _write_vtk_pvd_manifest(
        # Continues the surrounding statement or collection definition.
        self,
        manifest_path: str,
        relative_step_paths: List[str],
        time_values: List[float],
    # Continues the surrounding statement or collection definition.
    ) -> None:
        manifest_dir = os.path.dirname(manifest_path)
        os.makedirs(manifest_dir, exist_ok=True)
        dataset_lines = [
            # Completes the multi-line `rel_path.replace` arguments or elements.
            f'    <DataSet timestep="{time_value}" file="{rel_path.replace(os.sep, "/")}"/>'
            for rel_path, time_value in zip(relative_step_paths, time_values)
        ]
        payload = [
            # Completes the multi-line `collection` arguments or elements.
            "<?xml version=\"1.0\"?>",
            "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">",
            "  <Collection>",
            *dataset_lines,
            # Completes the multi-line `collection` arguments or elements.
            "  </Collection>",
            "</VTKFile>",
            "",
        ]
        # Calls `Path(manifest_path).write_text` to perform the operation required at this stage.
        Path(manifest_path).write_text("\n".join(payload), encoding="utf-8")

    def _write_vtk_series_manifest(
        self,
        manifest_path: str,
        # Continues the surrounding statement or collection definition.
        relative_step_paths: List[str],
        time_values: List[float],
    ) -> None:
        manifest_dir = os.path.dirname(manifest_path)
        # Calls `os.makedirs` to perform the operation required at this stage.
        os.makedirs(manifest_dir, exist_ok=True)
        payload = {
            "file-series-version": "1.0",
            "files": [
                # Completes the multi-line `collection` arguments or elements.
                {"name": rel_path.replace(os.sep, "/"), "time": time_value}
                for rel_path, time_value in zip(relative_step_paths, time_values)
            ],
        }
        # Calls `Path(manifest_path).write_text` to perform the operation required at this stage.
        Path(manifest_path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Defines `_write_temporal_vtk_outputs_with_inherited_layout` for  write temporal vtk outputs with inherited layout logic.
    def _write_temporal_vtk_outputs_with_inherited_layout(
        self,
        sample_id: int,
        sample_file: str,
        # Completes the multi-line `collection` arguments or elements.
        temporal_steps: List[Dict[str, np.ndarray]],
        output_dir: str,
    ) -> List[str]:
        context = self._sample_writeback_contexts.get(int(sample_id)) or {}
        # Stores the `ordered_files` for use by the remaining steps in this flow.
        ordered_files = list(context.get("ordered_files") or [])
        step_relpaths = list(context.get("step_relpaths") or [])
        sample_root = os.path.join(output_dir, f"sample_{sample_id}")

        output_paths: List[str] = []
        # Initializes the `relative_output_paths` as a collection for values from this processing stage.
        relative_output_paths: List[str] = []
        for step_idx, step_fields in enumerate(temporal_steps, start=1):
            template_file = self._resolve_temporal_step_template_file(
                sample_id=sample_id,
                # Completes the multi-line `self._resolve_temporal_step_template_file` arguments or elements.
                sample_file=sample_file,
                step_index=step_idx,
            )

            if step_idx - 1 < len(step_relpaths):
                # Converts the `rel_path` to `str` so later checks use a consistent type.
                rel_path = str(step_relpaths[step_idx - 1])
                basename = os.path.join(
                    f"sample_{sample_id}",
                    os.path.splitext(rel_path)[0],
                # Completes the multi-line `os.path.join` arguments or elements.
                )
            else:
                basename = self._build_temporal_step_output_basename(sample_id, step_idx)

            try:
                # Stores the `write_payload` for use by the remaining steps in this flow.
                write_payload = self._prepare_write_payload(
                    sample_id=sample_id,
                    processed_fields=step_fields,
                    step_index=step_idx,
                # Completes the multi-line `self._prepare_write_payload` arguments or elements.
                )
            except Exception as exc:
                self._register_writeback_failure(
                    sample_id,
                    # Completes the multi-line `self._register_writeback_failure` arguments or elements.
                    f"第 {step_idx} 个时间步根据预测位移更新网格坐标失败",
                    error=exc,
                    details={"step_idx": step_idx},
                )
                # Calls `self._log` to perform the operation required at this stage.
                self._log("error", f"样本 {sample_id} 第 {step_idx} 个时间步坐标更新失败: {str(exc)}")
                continue
            output_file = self._call_write_to_file(
                sample_id=sample_id,
                # Completes the multi-line `self._call_write_to_file` arguments or elements.
                sample_file=template_file,
                processed_fields=write_payload["processed_fields"],
                output_dir=output_dir,
                output_basename=basename,
                # Completes the multi-line `self._call_write_to_file` arguments or elements.
                point_coordinates=write_payload["point_coordinates"],
            )
            if output_file:
                output_paths.append(output_file)
                # Adds the current value to `relative_output_paths` for later aggregation.
                relative_output_paths.append(self._safe_relpath(output_file, sample_root))

        layout_kind = str(context.get("layout_kind", "")).strip().lower()
        manifest_name = str(context.get("manifest_name") or "").strip()
        time_values = self._normalize_time_values(list(context.get("time_values") or []), len(relative_output_paths))
        # Checks `manifest_name and len(relative_output_paths) > 0` before entering this conditional branch.
        if manifest_name and len(relative_output_paths) > 0:
            manifest_path = os.path.join(sample_root, manifest_name)
            if layout_kind == "temporal_manifest_pvd":
                self._write_vtk_pvd_manifest(manifest_path, relative_output_paths, time_values)
            # Checks `layout_kind == "temporal_manifest_series"` before entering this conditional branch.
            elif layout_kind == "temporal_manifest_series":
                self._write_vtk_series_manifest(manifest_path, relative_output_paths, time_values)

        return output_paths

    def _write_temporal_cgns_single_file_with_inherited_layout(
        # Continues the surrounding statement or collection definition.
        self,
        sample_id: int,
        sample_file: str,
        temporal_steps: List[Dict[str, np.ndarray]],
        # Continues the surrounding statement or collection definition.
        output_dir: str,
    ) -> List[str]:
        context = self._sample_writeback_contexts.get(int(sample_id)) or {}
        source_path = str(context.get("source_path") or sample_file or "").strip()
        # Converts the `source_name` to `str` so later checks use a consistent type.
        source_name = str(context.get("output_name") or os.path.basename(source_path) or os.path.basename(sample_file))
        basename = os.path.join(
            f"sample_{sample_id}",
            os.path.splitext(source_name)[0],
        # Completes the multi-line `os.path.join` arguments or elements.
        )
        output_file = self.cgns_writer.write_temporal(
            sample_id=sample_id,
            sample_file=sample_file,
            # Completes the multi-line `self.cgns_writer.write_temporal` arguments or elements.
            temporal_steps=temporal_steps,
            output_dir=output_dir,
            output_basename=basename,
            time_values=list(context.get("time_values") or []),
            # Completes the multi-line `list` arguments or elements.
            solution_names=list(context.get("solution_names") or []),
        )
        return [output_file] if output_file else []

    def _split_temporal_processed_fields(
        # Continues the surrounding statement or collection definition.
        self,
        processed_fields: Dict[str, np.ndarray],
    ) -> List[Dict[str, np.ndarray]]:
        """按时间步拆分后处理字段，便于逐步写回时序结果。"""
        task_mode = str(self.config.get("task", {}).get("mode", "")).strip().lower()
        # Returns early when `task_mode != "temporal" or not processed_fields` makes the remaining work inapplicable.
        if task_mode != "temporal" or not processed_fields:
            return []

        normalized_fields = {
            name: np.asarray(values)
            # Completes the multi-line `collection` arguments or elements.
            for name, values in processed_fields.items()
        }
        temporal_field_names = [
            name for name, values in normalized_fields.items()
            # Continues the surrounding statement or collection definition.
            if values.ndim >= 2
        ]

        if not temporal_field_names:
            return [normalized_fields]

        # Converts the `step_count` to `int` so later checks use a consistent type.
        step_count = int(normalized_fields[temporal_field_names[0]].shape[0])
        for name in temporal_field_names[1:]:
            if int(normalized_fields[name].shape[0]) != step_count:
                raise ValueError(
                    # Completes the multi-line `ValueError` arguments or elements.
                    f"时序字段 {name} 的时间步数量与其他字段不一致"
                )

        temporal_steps: List[Dict[str, np.ndarray]] = []
        for step_idx in range(step_count):
            # Initializes the `step_fields` as a collection for values from this processing stage.
            step_fields: Dict[str, np.ndarray] = {}
            for name, values in normalized_fields.items():
                step_fields[name] = values[step_idx] if values.ndim >= 2 else values
            temporal_steps.append(step_fields)
        # Returns `temporal_steps` to the caller for the next stage.
        return temporal_steps

    def _prepare_write_payload(
        self,
        sample_id: int,
        # Completes the multi-line `collection` arguments or elements.
        processed_fields: Dict[str, np.ndarray],
        step_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        point_coordinates = self._build_motion_writeback_point_coordinates(
            # Completes the multi-line `self._build_motion_writeback_point_coordinates` arguments or elements.
            sample_id=sample_id,
            step_fields=processed_fields,
            step_index=step_index,
        )
        # Returns `` to the caller for the next stage.
        return {
            "processed_fields": processed_fields,
            "point_coordinates": point_coordinates,
        }

    # Defines `_call_write_to_file` for  call write to file logic.
    def _call_write_to_file(
        self,
        *,
        sample_id: int,
        # Continues the surrounding statement or collection definition.
        sample_file: str,
        processed_fields: Dict[str, np.ndarray],
        output_dir: str,
        output_basename: Optional[str] = None,
        # Continues the surrounding statement or collection definition.
        point_coordinates: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        write_fn = self._write_to_file
        write_signature = inspect.signature(write_fn)
        # Initializes the `kwargs` as a collection for values from this processing stage.
        kwargs: Dict[str, Any] = {
            "sample_id": sample_id,
            "sample_file": sample_file,
            "processed_fields": processed_fields,
            # Completes the multi-line `collection` arguments or elements.
            "output_dir": output_dir,
            "output_basename": output_basename,
        }
        if "point_coordinates" in write_signature.parameters:
            # Stores the `kwargs['point_coordinates']` for use by the remaining steps in this flow.
            kwargs["point_coordinates"] = point_coordinates
        return write_fn(**kwargs)

    def _write_processed_fields(
        self,
        # Continues the surrounding statement or collection definition.
        sample_id: int,
        sample_file: str,
        processed_fields: Dict[str, np.ndarray],
        output_dir: str,
    # Continues the surrounding statement or collection definition.
    ) -> List[str]:
        """Write processed temporal fields and return the paths created for this sample."""
        temporal_steps = self._split_temporal_processed_fields(processed_fields)
        if not temporal_steps:
            try:
                # Stores the `write_payload` for use by the remaining steps in this flow.
                write_payload = self._prepare_write_payload(
                    sample_id=sample_id,
                    processed_fields=processed_fields,
                    step_index=1,
                # Completes the multi-line `self._prepare_write_payload` arguments or elements.
                )
            except Exception as exc:
                self._register_writeback_failure(
                    sample_id,
                    # Completes the multi-line `self._register_writeback_failure` arguments or elements.
                    "根据预测位移更新网格坐标失败",
                    error=exc,
                )
                self._log("error", f"样本 {sample_id} 写回前坐标更新失败: {str(exc)}")
                # Returns `[]` to the caller for the next stage.
                return []
            template_file = self._resolve_temporal_step_template_file(
                sample_id=sample_id,
                sample_file=sample_file,
                # Completes the multi-line `self._resolve_temporal_step_template_file` arguments or elements.
                step_index=1,
            )
            output_file = self._call_write_to_file(
                sample_id=sample_id,
                # Completes the multi-line `self._call_write_to_file` arguments or elements.
                sample_file=template_file,
                processed_fields=write_payload["processed_fields"],
                output_dir=output_dir,
                point_coordinates=write_payload["point_coordinates"],
            # Completes the multi-line `self._call_write_to_file` arguments or elements.
            )
            return [output_file] if output_file else []

        context = self._sample_writeback_contexts.get(int(sample_id)) or {}
        layout_kind = str(context.get("layout_kind", "")).strip().lower()
        # Stores the `template_file` for use by the remaining steps in this flow.
        template_file = str(context.get("template_file") or sample_file or "").strip()
        file_ext = os.path.splitext(template_file)[1].lower()

        if layout_kind == "temporal_single_cgns_file" and file_ext == ".cgns":
            if str(context.get("geometry_regime") or "").strip().lower() == "displaced_static_topology":
                # Calls `self._register_writeback_failure` to perform the operation required at this stage.
                self._register_writeback_failure(
                    sample_id,
                    "当前不支持在单个时序 CGNS 文件中写回逐步更新的运动网格坐标",
                    details={"layout_kind": layout_kind},
                # Completes the multi-line `self._register_writeback_failure` arguments or elements.
                )
                self._log("error", f"样本 {sample_id} 暂不支持 temporal_single_cgns_file 的运动网格坐标写回")
                return []
            return self._write_temporal_cgns_single_file_with_inherited_layout(
                # Completes the multi-line `self._write_temporal_cgns_single_file_with_inherited_layout` arguments or elements.
                sample_id=sample_id,
                sample_file=sample_file,
                temporal_steps=temporal_steps,
                output_dir=output_dir,
            # Completes the multi-line `self._write_temporal_cgns_single_file_with_inherited_layout` arguments or elements.
            )

        if self._should_inherit_temporal_layout(sample_id=sample_id, sample_file=sample_file):
            return self._write_temporal_vtk_outputs_with_inherited_layout(
                sample_id=sample_id,
                # Completes the multi-line `self._write_temporal_vtk_outputs_with_inherited_layout` arguments or elements.
                sample_file=sample_file,
                temporal_steps=temporal_steps,
                output_dir=output_dir,
            )

        # Initializes the `output_paths` as a collection for values from this processing stage.
        output_paths: List[str] = []
        for step_idx, step_fields in enumerate(temporal_steps, start=1):
            try:
                write_payload = self._prepare_write_payload(
                    # Completes the multi-line `self._prepare_write_payload` arguments or elements.
                    sample_id=sample_id,
                    processed_fields=step_fields,
                    step_index=step_idx,
                )
            # Continues the surrounding statement or collection definition.
            except Exception as exc:
                self._register_writeback_failure(
                    sample_id,
                    f"第 {step_idx} 个时间步根据预测位移更新网格坐标失败",
                    # Completes the multi-line `self._register_writeback_failure` arguments or elements.
                    error=exc,
                    details={"step_idx": step_idx},
                )
                self._log("error", f"样本 {sample_id} 第 {step_idx} 个时间步坐标更新失败: {str(exc)}")
                # Handles the data or control-flow case represented by this statement.
                continue
            template_file = self._resolve_temporal_step_template_file(
                sample_id=sample_id,
                sample_file=sample_file,
                # Completes the multi-line `self._resolve_temporal_step_template_file` arguments or elements.
                step_index=step_idx,
            )
            output_file = self._call_write_to_file(
                sample_id=sample_id,
                # Completes the multi-line `self._call_write_to_file` arguments or elements.
                sample_file=template_file,
                processed_fields=write_payload["processed_fields"],
                output_dir=output_dir,
                output_basename=self._build_temporal_step_output_basename(sample_id, step_idx),
                # Completes the multi-line `self._call_write_to_file` arguments or elements.
                point_coordinates=write_payload["point_coordinates"],
            )
            if output_file:
                output_paths.append(output_file)
        # Returns `output_paths` to the caller for the next stage.
        return output_paths

    def _write_to_file(
        self,
        sample_id: int,
        # Continues the surrounding statement or collection definition.
        sample_file: str,
        processed_fields: Dict[str, np.ndarray],
        output_dir: str,
        output_basename: Optional[str] = None,
        # Continues the surrounding statement or collection definition.
        point_coordinates: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """按文件扩展名分发到对应 writer。"""
        file_ext = os.path.splitext(sample_file)[1].lower()
        handler = self._get_write_handler(file_ext)
        # Checks `handler is None` before entering this conditional branch.
        if handler is None:
            self._log("warning", f"样本 {sample_id} 的文件格式暂不支持: {file_ext}")
            return None

        return handler(
            # Completes the multi-line `handler` arguments or elements.
            sample_id=sample_id,
            sample_file=sample_file,
            processed_fields=processed_fields,
            output_dir=output_dir,
            # Completes the multi-line `handler` arguments or elements.
            output_basename=output_basename,
            point_coordinates=point_coordinates,
        )

    def _get_write_location(
        # Continues the surrounding statement or collection definition.
        self,
        sample_id: int,
        write_fields: Dict[str, np.ndarray],
        num_points: int,
        # Continues the surrounding statement or collection definition.
        num_cells: int,
    ) -> str:
        """复用 VTK writer 的写回位置校验逻辑。"""
        return self.vtk_writer._get_write_location(
            sample_id=sample_id,
            # Completes the multi-line `self.vtk_writer._get_write_location` arguments or elements.
            write_fields=write_fields,
            num_points=num_points,
            num_cells=num_cells,
        )

    # 根据扩展名返回对应的写回处理函数。
    def _get_write_handler(
        self,
        file_ext: str,
    ) -> Optional[Callable[..., Optional[str]]]:
        """根据扩展名返回对应的写回处理函数。"""
        # Initializes the `write_handlers` as a collection for values from this processing stage.
        write_handlers = {
            ".cgns": self._write_to_cgns,
            ".csv": self._write_to_csv,
            ".ply": self._write_to_point_cloud,
            # Completes the multi-line `collection` arguments or elements.
            ".pcd": self._write_to_point_cloud,
            ".xyz": self._write_to_point_cloud,
            ".pts": self._write_to_point_cloud,
        }
        # Adds the current value to `write_handlers` for later aggregation.
        write_handlers.update(
            {ext: self._write_to_vtk for ext in self.VTK_FILE_EXTENSIONS}
        )
        return write_handlers.get(file_ext)

    # 将预测结果写回 CGNS。
    def _write_to_cgns(
        self,
        sample_id: int,
        sample_file: str,
        # Completes the multi-line `collection` arguments or elements.
        processed_fields: Dict[str, np.ndarray],
        output_dir: str,
        output_basename: Optional[str] = None,
        point_coordinates: Optional[np.ndarray] = None,
    # Continues the surrounding statement or collection definition.
    ) -> Optional[str]:
        """将预测结果写回 CGNS。"""
        return self.cgns_writer.write(
            sample_id=sample_id,
            sample_file=sample_file,
            # Completes the multi-line `self.cgns_writer.write` arguments or elements.
            processed_fields=processed_fields,
            output_dir=output_dir,
            output_basename=output_basename,
            point_coordinates=point_coordinates,
        # Completes the multi-line `self.cgns_writer.write` arguments or elements.
        )

    def _write_to_vtk(
        self,
        sample_id: int,
        # Continues the surrounding statement or collection definition.
        sample_file: str,
        processed_fields: Dict[str, np.ndarray],
        output_dir: str,
        output_basename: Optional[str] = None,
        # Continues the surrounding statement or collection definition.
        point_coordinates: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """按配置的 point_or_cell 将预测结果写回 VTK。"""
        return self.vtk_writer.write(
            sample_id=sample_id,
            # Completes the multi-line `self.vtk_writer.write` arguments or elements.
            sample_file=sample_file,
            processed_fields=processed_fields,
            output_dir=output_dir,
            output_basename=output_basename,
            # Completes the multi-line `self.vtk_writer.write` arguments or elements.
            point_coordinates=point_coordinates,
        )

    def _write_to_csv(
        self,
        # Continues the surrounding statement or collection definition.
        sample_id: int,
        sample_file: str,
        processed_fields: Dict[str, np.ndarray],
        output_dir: str,
        # Continues the surrounding statement or collection definition.
        output_basename: Optional[str] = None,
        point_coordinates: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """将预测结果写回 CSV。"""
        del point_coordinates
        # Returns `` to the caller for the next stage.
        return self.csv_writer.write(
            sample_id=sample_id,
            sample_file=sample_file,
            processed_fields=processed_fields,
            # Completes the multi-line `self.csv_writer.write` arguments or elements.
            output_dir=output_dir,
            output_basename=output_basename,
        )

    def _write_to_point_cloud(
        # Continues the surrounding statement or collection definition.
        self,
        sample_id: int,
        sample_file: str,
        processed_fields: Dict[str, np.ndarray],
        # Continues the surrounding statement or collection definition.
        output_dir: str,
        output_basename: Optional[str] = None,
        point_coordinates: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """将预测结果写回点云文件。"""
        # Returns `` to the caller for the next stage.
        return self.point_cloud_writer.write(
            sample_id=sample_id,
            sample_file=sample_file,
            processed_fields=processed_fields,
            # Completes the multi-line `self.point_cloud_writer.write` arguments or elements.
            output_dir=output_dir,
            output_basename=output_basename,
            point_coordinates=point_coordinates,
        )
