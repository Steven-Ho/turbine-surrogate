from .base_adapter import BasePredictAdapter
from .image_voxel_adapter import ImageVoxelPredictAdapter
from .matrix_adapter import MatrixPredictAdapter
# Imports `GraphPredictAdapter` from `graph_adapter` for use in this module.
from .graph_adapter import GraphPredictAdapter

__all__ = [
    "BasePredictAdapter",
    "ImageVoxelPredictAdapter",
    # Completes the multi-line `collection` arguments or elements.
    "MatrixPredictAdapter",
    "GraphPredictAdapter",
]
