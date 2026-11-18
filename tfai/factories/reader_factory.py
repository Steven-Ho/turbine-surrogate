import copy
from pathlib import Path
from typing import Dict, Optional

# Imports `is_temporal_manifest_path, list_temporal_sequence_files` from `tfai.data.temporal_utils` for use in this module.
from tfai.data.temporal_utils import (
    is_temporal_manifest_path,
    list_temporal_sequence_files,
)
# Imports `CGNSReader` from `tfai.readers.CGNSReader` for use in this module.
from tfai.readers.CGNSReader import CGNSReader
from tfai.readers.VTKReader import VTKReader
from tfai.readers.sample_map import SampleMap
from tfai.utils.path_utils import resolve_path_from_anchor


# Defines `ReaderFactory` to group related data and behavior.
class ReaderFactory:
    @staticmethod
    def infer_reader_extension_from_source_path(source_path: str) -> str:
        """根据原始样本源路径推断底层 reader 应使用的文件后缀。"""
        resolved_source = Path(str(source_path)).resolve()
        # Validates `not resolved_source.exists()` and raises `FileNotFoundError` when the input cannot be processed safely.
        if not resolved_source.exists():
            raise FileNotFoundError(f"样本源不存在: {resolved_source}")

        if resolved_source.is_file():
            if is_temporal_manifest_path(str(resolved_source)):
                # Stores the `temporal_files` for use by the remaining steps in this flow.
                temporal_files = list_temporal_sequence_files(
                    reader=None,
                    source_path=str(resolved_source),
                )
                # Validates `not temporal_files` and raises `ValueError` when the input cannot be processed safely.
                if not temporal_files:
                    raise ValueError(f"时序清单中未解析到任何 step 文件: {resolved_source}")
                return Path(temporal_files[0]).suffix.lower()
            return resolved_source.suffix.lower()

        # Validates `not resolved_source.is_dir()` and raises `ValueError` when the input cannot be processed safely.
        if not resolved_source.is_dir():
            raise ValueError(f"样本源既不是文件也不是目录: {resolved_source}")

        manifest_candidates = sorted(
            item
            # Completes the multi-line `resolved_source.iterdir` arguments or elements.
            for item in resolved_source.iterdir()
            if item.is_file() and is_temporal_manifest_path(str(item))
        )
        if manifest_candidates:
            # Stores the `temporal_files` for use by the remaining steps in this flow.
            temporal_files = list_temporal_sequence_files(
                reader=None,
                source_path=str(manifest_candidates[0]),
            )
            # Validates `not temporal_files` and raises `ValueError` when the input cannot be processed safely.
            if not temporal_files:
                raise ValueError(f"时序清单中未解析到任何 step 文件: {manifest_candidates[0]}")
            return Path(temporal_files[0]).suffix.lower()

        supported_suffixes = (
            # Completes the multi-line `set` arguments or elements.
            set(VTKReader.SUPPORTED_EXTENSIONS)
            | set(CGNSReader.SUPPORTED_EXTENSIONS)
        )

        direct_files = sorted(
            # Completes the multi-line `sorted` arguments or elements.
            item
            for item in resolved_source.iterdir()
            if item.is_file() and item.suffix.lower() in supported_suffixes
        )
        # Returns early when `direct_files` makes the remaining work inapplicable.
        if direct_files:
            return direct_files[0].suffix.lower()

        for child_dir in sorted(item for item in resolved_source.iterdir() if item.is_dir()):
            nested_files = sorted(
                # Completes the multi-line `sorted` arguments or elements.
                child
                for child in child_dir.iterdir()
                if child.is_file() and child.suffix.lower() in supported_suffixes
            )
            # Returns early when `nested_files` makes the remaining work inapplicable.
            if nested_files:
                return nested_files[0].suffix.lower()

        raise ValueError(f"无法从样本源推断文件后缀: {resolved_source}")

    @staticmethod
    # 从 field 样本源推断底层读取器应使用的文件后缀。
    def _infer_field_file_extension(
        sample_table_path: str,
        file_path: str,
    ) -> str:
        """从 field 样本源推断底层读取器应使用的文件后缀。"""
        # Normalizes the `resolved_source` before subsequent file operations.
        resolved_source = Path(resolve_path_from_anchor(sample_table_path, file_path)).resolve()
        return ReaderFactory.infer_reader_extension_from_source_path(str(resolved_source))

    @staticmethod
    def get_reader(
        # Continues the surrounding statement or collection definition.
        config: Dict,
        flag: str,
        sample_table_path_override: Optional[str] = None,
    ):
        """根据配置自动创建对应的读取器实例。"""
        # Stores the `effective_config` for use by the remaining steps in this flow.
        effective_config = copy.deepcopy(config)
        task_type = effective_config["task"]["type"]
        sample_table_path = sample_table_path_override

        if not sample_table_path:
            # Checks `flag == "train"` before entering this conditional branch.
            if flag == "train":
                sample_table_path = effective_config.get("general", {}).get("train_samples")
            elif flag == "predict":
                sample_table_path = effective_config.get("prediction", {}).get("predict_samples")
            # Checks `flag == "evaluate"` before entering this conditional branch.
            elif flag == "evaluate":
                sample_table_path = effective_config.get("evaluate", {}).get("samples")

        if not sample_table_path:
            raise ValueError(f"未找到样本表路径配置，flag={flag}")

        # Reader 初始化时会从 general.train_samples 构建 sample_map。
        # 对评估/预测阶段的临时样本表，需要在这里同步覆盖，避免继续绑定旧训练样本表。
        # Calls `effective_config.setdefault` to perform the operation required at this stage.
        effective_config.setdefault("general", {})
        effective_config["general"]["train_samples"] = str(sample_table_path)

        if str(task_type).strip().lower() != "field":
            raise ValueError("启元发行包仅支持 field 场任务")

        # Keeps the operation and its recovery path within the same exception boundary.
        try:
            sample_map = SampleMap(sample_table_path)
            sample_ids = sample_map.get_all_samples()

            first_file_path = ""
            # Checks `sample_ids` before entering this conditional branch.
            if sample_ids:
                first_file_path = sample_map.get_file_path(sample_ids[0]) or ""

            if not first_file_path:
                raise ValueError("样本表中未找到 file_path")

            # Stores the `file_extension` for use by the remaining steps in this flow.
            file_extension = ReaderFactory._infer_field_file_extension(
                sample_table_path=sample_table_path,
                file_path=first_file_path,
            )
            # Returns `` to the caller for the next stage.
            return ReaderFactory._create_reader_by_extension(file_extension, effective_config)
        except Exception as exc:
            raise ValueError(f"读取样本表失败: {str(exc)}") from exc

    @staticmethod
    # 根据文件后缀名创建对应的读取器实例。
    def _create_reader_by_extension(extension: str, config: Dict):
        """根据文件后缀名创建对应的读取器实例。"""
        if not extension.startswith("."):
            extension = "." + extension

        extension = extension.lower()
        # Returns early when `extension == ".cgns"` makes the remaining work inapplicable.
        if extension == ".cgns":
            return CGNSReader(config)
        if extension in VTKReader.SUPPORTED_EXTENSIONS:
            return VTKReader(config)
        # Raises `ValueError` to report why this path cannot continue.
        raise ValueError(f"不支持的文件后缀名: {extension}")
