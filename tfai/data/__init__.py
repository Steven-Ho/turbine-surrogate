"""Public exports for the data package.

This module keeps imports lightweight to avoid circular dependencies between
processors and the data pipeline.
"""

from .base_data import BaseData
from .loaders import FieldSampleLoader, ScalarSampleLoader, ScalarTableSource
from .sample_statistics import (
    # Continues the surrounding statement or collection definition.
    calculate_cumulative_coord_ranges,
    extract_coord_ranges_from_sample,
)
from .temporal_utils import (
    # Continues the surrounding statement or collection definition.
    aggregate_field_temporal_sequence,
    connectivity_to_tuples,
    extract_time_index_from_name,
    get_reader_supported_extensions,
    # Continues the surrounding statement or collection definition.
    list_temporal_sequence_files,
    read_single_sample,
)

__all__ = [
    # Completes the multi-line `collection` arguments or elements.
    "BaseData",
    "DataPipeline",
    "FieldSampleLoader",
    "ScalarSampleLoader",
    # Completes the multi-line `collection` arguments or elements.
    "ScalarTableSource",
    "extract_coord_ranges_from_sample",
    "calculate_cumulative_coord_ranges",
    "get_reader_supported_extensions",
    # Completes the multi-line `collection` arguments or elements.
    "extract_time_index_from_name",
    "list_temporal_sequence_files",
    "read_single_sample",
    "connectivity_to_tuples",
    # Completes the multi-line `collection` arguments or elements.
    "aggregate_field_temporal_sequence",
]


def __getattr__(name):
    if name == "DataPipeline":
        # Imports `DataPipeline` from `data_pipeline` for use in this module.
        from .data_pipeline import DataPipeline

        return DataPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
