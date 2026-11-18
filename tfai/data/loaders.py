import inspect
import os
from typing import List, Optional, Tuple

# Imports `BaseData` from `base_data` for use in this module.
from .base_data import BaseData
from .temporal_utils import (
    aggregate_field_temporal_sequence,
    is_temporal_manifest_path,
    # Continues the surrounding statement or collection definition.
    list_temporal_sequence_files,
    read_single_sample,
)
from ..utils.boundary_feature_encoding import normalize_field_bc_names


# Field sample loading strategy for single-file and temporal-directory inputs.
class FieldSampleLoader:
    """Field sample loading strategy for single-file and temporal-directory inputs."""

    def __init__(
        self,
        reader,
        # Continues the surrounding statement or collection definition.
        mode: Optional[str],
        allow_missing_fields: bool = False,
        temporal_history_steps: Optional[int] = None,
    ):
        # Stores the data reader on the instance so other methods can reuse it.
        self.reader = reader
        self.mode = (mode or "").lower()
        self.allow_missing_fields = bool(allow_missing_fields)
        self.temporal_history_steps = temporal_history_steps
        # Reads `task` from the mapping into the task settings with its declared fallback.
        task_cfg = getattr(reader, "config", {}).get("task", {})
        if str(task_cfg.get("local_feature_encoding", "")).strip().lower() == "vtk_based":
            self.static_field_names = normalize_field_bc_names(task_cfg.get("field_bc"))
        else:
            # Stores the `static_field_names` on the instance so other methods can reuse it.
            self.static_field_names = []

    def load(
        self,
        file_path: str,
        # Continues the surrounding statement or collection definition.
        sample_id: Optional[int],
        variables: Optional[List[str]] = None,
        max_time_steps: Optional[int] = None,
    ) -> BaseData:
        """Load one field sample, aggregating supported temporal field sources when needed."""
        # Stores the `resolved_max_time_steps` for use by the remaining steps in this flow.
        resolved_max_time_steps = (
            self.temporal_history_steps if max_time_steps is None else max_time_steps
        )
        reader_name = type(self.reader).__name__.lower()
        # Converts the `supports_temporal_sequence` to `bool` so later checks use a consistent type.
        supports_temporal_sequence = bool(getattr(self.reader, "supports_temporal_sequence", False))
        supports_temporal_directory = supports_temporal_sequence or "vtk" in reader_name or "cgns" in reader_name
        supports_temporal_manifest = supports_temporal_directory and is_temporal_manifest_path(file_path)
        if self.mode == "temporal" and supports_temporal_directory and (
            # Completes the multi-line `os.path.isdir` arguments or elements.
            os.path.isdir(file_path) or supports_temporal_manifest
        ):
            read_temporal_sequence = getattr(self.reader, "read_temporal_sequence", None)
            if callable(read_temporal_sequence):
                # Returns `` to the caller for the next stage.
                return read_temporal_sequence(
                    file_path=file_path,
                    variables=variables,
                    sample_order=sample_id,
                    # Completes the multi-line `read_temporal_sequence` arguments or elements.
                    allow_missing_fields=self.allow_missing_fields,
                    max_time_steps=resolved_max_time_steps,
                )

            sequence_files = list_temporal_sequence_files(
                # Completes the multi-line `list_temporal_sequence_files` arguments or elements.
                self.reader,
                file_path,
                max_time_steps=resolved_max_time_steps,
            )
            # Stores the `step_data` for use by the remaining steps in this flow.
            step_data = (
                read_single_sample(
                    self.reader,
                    path,
                    # Completes the multi-line `read_single_sample` arguments or elements.
                    sample_id,
                    allow_missing_fields=self.allow_missing_fields,
                    max_time_steps=None,
                    variables=variables,
                # Completes the multi-line `read_single_sample` arguments or elements.
                )
                for path in sequence_files
            )
            return aggregate_field_temporal_sequence(
                # Completes the multi-line `aggregate_field_temporal_sequence` arguments or elements.
                step_data,
                file_path,
                static_field_names=self.static_field_names,
                expected_time_steps=len(sequence_files),
            # Completes the multi-line `aggregate_field_temporal_sequence` arguments or elements.
            )

        if not os.path.isdir(file_path):
            return read_single_sample(
                self.reader,
                # Completes the multi-line `read_single_sample` arguments or elements.
                file_path,
                sample_id,
                allow_missing_fields=self.allow_missing_fields,
                max_time_steps=resolved_max_time_steps,
                # Completes the multi-line `read_single_sample` arguments or elements.
                variables=variables,
            )

        raise ValueError(
            "Automatic temporal loading currently only supports "
            # Completes the multi-line `ValueError` arguments or elements.
            f"field-temporal readers with temporal-sequence capability, including VTK/CGNS directory inputs. mode={self.mode}, "
            f"reader={type(self.reader).__name__}, path={file_path}"
        )

    def load_local_feature_fields(
        # Continues the surrounding statement or collection definition.
        self,
        file_path: str,
        sample_id: Optional[int],
        max_time_steps: Optional[int] = 1,
    # Continues the surrounding statement or collection definition.
    ) -> BaseData:
        """仅读取 vtk_based schema 拟合所需的首个时间步局部边界字段。"""
        if not self.static_field_names:
            return self.load(file_path, sample_id, max_time_steps=max_time_steps)
        return self.load(
            # Completes the multi-line `self.load` arguments or elements.
            file_path,
            sample_id,
            variables=self.static_field_names,
            max_time_steps=max_time_steps,
        # Completes the multi-line `self.load` arguments or elements.
        )


class ScalarSampleLoader:
    """Scalar table loading strategy."""

    def __init__(self, reader):
        self.reader = reader

    # Load one scalar source table into BaseData.
    def load(self, table_path: str, sample_id: Optional[int] = None) -> BaseData:
        """Load one scalar source table into BaseData."""
        signature = inspect.signature(self.reader.read)
        if "sample_order" in signature.parameters:
            return self.reader.read(table_path, sample_order=sample_id)
        # Returns `self.reader.read(table_path)` to the caller for the next stage.
        return self.reader.read(table_path)


class ScalarTableSource:
    """Resolve scalar table inputs into pipeline processing units."""

    def __init__(self, table_path: str):
        self.table_path = table_path

    # Return the source table path as the scalar preprocessing input unit.
    def resolve_units(self) -> List[Tuple[int, str]]:
        """Return the source table path as the scalar preprocessing input unit.

        具体样本粒度由后续预处理阶段决定：
        - steady: 按行拆分，一个样本对应表中的一行
        - temporal: 整张表或整段序列作为一个时序样本
        """
        return [(0, self.table_path)]
