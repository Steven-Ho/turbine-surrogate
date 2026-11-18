from pathlib import Path
from typing import Dict, Optional, Type

from tfai.factories.reader_factory import ReaderFactory
# Imports `SampleMap` from `tfai.readers.sample_map` for use in this module.
from tfai.readers.sample_map import SampleMap
from tfai.processors.field.field_processor import FieldProcessor
from tfai.processors.field.graph_processor import GraphProcessor
from tfai.processors.field.image_voxel_processor import ImageVoxelProcessor
# Imports `MatrixProcessor` from `tfai.processors.field.matrix_processor` for use in this module.
from tfai.processors.field.matrix_processor import MatrixProcessor
from tfai.utils.qiyuan_constraints import validate_qiyuan_config
from tfai.utils.path_utils import resolve_path_from_anchor


class ProcessorFactory:
    """启元发行包的稳态场处理器工厂。"""

    # Initializes the `_MATRIX_MODEL_TYPES` as a collection for values from this processing stage.
    _MATRIX_MODEL_TYPES = {"POD"}
    _IMAGE_MODEL_TYPES = {"UNet"}
    _GRAPH_MODEL_TYPES = {"GCN", "Graph_UNet", "Graph_Transformer", "Transolver_3", "GeoTransolver"}

    _PROCESSOR_NAME_MAP = {
        # Completes the multi-line `collection` arguments or elements.
        "MatrixProcessor": MatrixProcessor,
        "ImageVoxelProcessor": ImageVoxelProcessor,
        "GraphProcessor": GraphProcessor,
    }

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _collect_field_sample_table_paths(config: Dict) -> list[str]:
        candidate_paths = []
        for section_name, key in (
            # Completes the multi-line `collection` arguments or elements.
            ("general", "train_samples"),
            ("prediction", "predict_samples"),
            ("evaluate", "samples"),
        ):
            # Stores the `value` for use by the remaining steps in this flow.
            value = (config.get(section_name, {}) or {}).get(key)
            if str(value or "").strip():
                candidate_paths.append(str(value).strip())
        return candidate_paths

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _infer_geometry_representation_from_samples(config: Dict) -> str:
        for sample_table_path in ProcessorFactory._collect_field_sample_table_paths(config):
            if not Path(sample_table_path).exists():
                # Handles the data or control-flow case represented by this statement.
                continue
            try:
                sample_map = SampleMap(sample_table_path)
                sample_ids = sample_map.get_all_samples()
                # Checks `not sample_ids` before entering this conditional branch.
                if not sample_ids:
                    continue
                source_path = sample_map.get_file_path(sample_ids[0])
                if not source_path:
                    # Handles the data or control-flow case represented by this statement.
                    continue
                extension = ReaderFactory.infer_reader_extension_from_source_path(
                    resolve_path_from_anchor(sample_table_path, source_path)
                )
                # Returns early when `` makes the remaining work inapplicable.
                if extension == ".cgns" or extension in {".vtk", ".vtu", ".vts", ".vtr", ".vti"}:
                    return "mesh"
            except Exception:
                continue
        # Returns `""` to the caller for the next stage.
        return ""

    @staticmethod
    def _create_processor_by_name(processor_name: str, config: Dict) -> FieldProcessor:
        processor_cls = ProcessorFactory._PROCESSOR_NAME_MAP.get(str(processor_name).strip())
        # Checks `processor_cls is None` before entering this conditional branch.
        if processor_cls is None:
            available = sorted(ProcessorFactory._PROCESSOR_NAME_MAP)
            raise ValueError(f"未知的 processor.type: {processor_name}。可选值: {available}")
        return processor_cls(config)

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _resolve_processor_class(task_type: str, model_type: str) -> Type[FieldProcessor]:
        if str(task_type or "").strip().lower() != "field":
            raise ValueError("启元发行包仅支持 field 场任务")
        # Stores the `normalized_model_type` for use by the remaining steps in this flow.
        normalized_model_type = str(model_type or "").strip()
        if normalized_model_type not in {
            *ProcessorFactory._MATRIX_MODEL_TYPES,
            *ProcessorFactory._IMAGE_MODEL_TYPES,
            # Completes the multi-line `collection` arguments or elements.
            *ProcessorFactory._GRAPH_MODEL_TYPES,
        }:
            raise ValueError(f"不支持的稳态场模型: {normalized_model_type}")
        if normalized_model_type in ProcessorFactory._MATRIX_MODEL_TYPES:
            # Returns `MatrixProcessor` to the caller for the next stage.
            return MatrixProcessor
        if normalized_model_type in ProcessorFactory._IMAGE_MODEL_TYPES:
            return ImageVoxelProcessor
        return GraphProcessor

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _validate_processor_compatibility(
        *, task_type: str, model_type: str, processor_cls: Type[FieldProcessor], config: Dict
    ) -> None:
        # Calls `validate_qiyuan_config` to perform the operation required at this stage.
        validate_qiyuan_config(config, model_type=model_type)
        geometry_representation = str(
            (config.get("task", {}) or {}).get("geometry_representation", "") or ""
        ).strip().lower()
        # Checks `not geometry_representation` before entering this conditional branch.
        if not geometry_representation:
            geometry_representation = ProcessorFactory._infer_geometry_representation_from_samples(config)
        if geometry_representation == "point_cloud":
            raise NotImplementedError("启元发行包不支持 point_cloud 几何输入")

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def get_processor(
        task_type: str,
        model_type: str,
        # Continues the surrounding statement or collection definition.
        config_or_grid_type,
        config: Optional[Dict] = None,
    ) -> FieldProcessor:
        """创建稳态场处理器。"""
        if isinstance(config_or_grid_type, dict) and config is None:
            # Stores the runtime configuration for use by the remaining steps in this flow.
            config = config_or_grid_type
        else:
            config = config or {}
        if not isinstance(config, dict):
            # Raises `TypeError` to report why this path cannot continue.
            raise TypeError("config 必须是字典")

        validate_qiyuan_config(config, model_type=model_type)
        processor_type = str((config.get("processor", {}) or {}).get("type", "") or "").strip()
        processor_cls = (
            # Completes the multi-line `ProcessorFactory._PROCESSOR_NAME_MAP.get` arguments or elements.
            ProcessorFactory._PROCESSOR_NAME_MAP.get(processor_type)
            if processor_type
            else ProcessorFactory._resolve_processor_class(task_type, model_type)
        )
        # Returns early when `processor_cls is None` makes the remaining work inapplicable.
        if processor_cls is None:
            return ProcessorFactory._create_processor_by_name(processor_type, config)
        ProcessorFactory._validate_processor_compatibility(
            task_type=task_type,
            # Completes the multi-line `ProcessorFactory._validate_processor_compatibility` arguments or elements.
            model_type=model_type,
            processor_cls=processor_cls,
            config=config,
        )
        # Returns `processor_cls(config)` to the caller for the next stage.
        return processor_cls(config)
