from __future__ import annotations

from contextlib import contextmanager
from threading import Lock
# Imports `Dict, List, Optional` from `typing` for use in this module.
from typing import Dict, List, Optional, Sequence, Tuple
import os

import numpy as np
import vtk
# Keeps the operation and its recovery path within the same exception boundary.
try:
    from vtkmodules.util import numpy_support
except Exception:  # pragma: no cover - fallback for alternative vtk packaging
    from vtk.util import numpy_support  # type: ignore[import-not-found]

# Checks `"bool" not in np.__dict__` before entering this conditional branch.
# Checks `"bool" not in np.__dict__` before entering this conditional branch.
# Checks `"bool" not in np.__dict__` before entering this conditional branch.
if "bool" not in np.__dict__:
    np.bool = np.bool_  # type: ignore[attr-defined]


_VTK_OUTPUT_LOCK = Lock()


@contextmanager
# 在读取阶段临时关闭 VTK 底层 stderr 输出。
def _silence_vtk_runtime_output():
    """在读取阶段临时关闭 VTK 底层 stderr 输出。"""
    output_window_cls = getattr(vtk, "vtkOutputWindow", None)
    file_output_window_cls = getattr(vtk, "vtkFileOutputWindow", None)
    string_output_window_cls = getattr(vtk, "vtkStringOutputWindow", None)
    # Stores the `object_cls` for use by the remaining steps in this flow.
    object_cls = getattr(vtk, "vtkObject", None)
    logger_cls = getattr(vtk, "vtkLogger", None)

    previous_output_window = None
    previous_warning_state = None
    # Stores the `previous_display_mode` for use by the remaining steps in this flow.
    previous_display_mode = None
    previous_logger_cutoff = None
    stderr_backup_fd = None
    devnull_fd = None

    # Scopes resource use so the context manager can release it on exit.
    with _VTK_OUTPUT_LOCK:
        if output_window_cls is not None and hasattr(output_window_cls, "GetInstance"):
            previous_output_window = output_window_cls.GetInstance()
        if object_cls is not None and hasattr(object_cls, "GetGlobalWarningDisplay"):
            # Converts the `previous_warning_state` to `bool` so later checks use a consistent type.
            previous_warning_state = bool(object_cls.GetGlobalWarningDisplay())
        if previous_output_window is not None and hasattr(previous_output_window, "GetDisplayMode"):
            previous_display_mode = previous_output_window.GetDisplayMode()
        if logger_cls is not None and hasattr(logger_cls, "GetCurrentVerbosityCutoff"):
            # Stores the `previous_logger_cutoff` for use by the remaining steps in this flow.
            previous_logger_cutoff = logger_cls.GetCurrentVerbosityCutoff()

        muted_output_window = None
        if file_output_window_cls is not None:
            muted_output_window = file_output_window_cls()
            # Calls `muted_output_window.SetFileName` to perform the operation required at this stage.
            muted_output_window.SetFileName(os.devnull)
            if hasattr(muted_output_window, "FlushOn"):
                muted_output_window.FlushOn()
        elif string_output_window_cls is not None:
            # Stores the `muted_output_window` for use by the remaining steps in this flow.
            muted_output_window = string_output_window_cls()

        if muted_output_window is not None:
            if hasattr(muted_output_window, "SetDisplayModeToNever"):
                muted_output_window.SetDisplayModeToNever()
            # Checks `hasattr(muted_output_window, "PromptUserOff")` before entering this conditional branch.
            if hasattr(muted_output_window, "PromptUserOff"):
                muted_output_window.PromptUserOff()
            if output_window_cls is not None and hasattr(output_window_cls, "SetInstance"):
                output_window_cls.SetInstance(muted_output_window)

        # Checks `` before entering this conditional branch.
        if object_cls is not None and hasattr(object_cls, "GlobalWarningDisplayOff"):
            object_cls.GlobalWarningDisplayOff()
        if logger_cls is not None and hasattr(logger_cls, "SetStderrVerbosity"):
            logger_cls.SetStderrVerbosity(logger_cls.VERBOSITY_OFF)
        # Keeps the operation and its recovery path within the same exception boundary.
        try:
            stderr_backup_fd = os.dup(2)
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull_fd, 2)
        # Continues the surrounding statement or collection definition.
        except OSError:
            stderr_backup_fd = None
            devnull_fd = None

        try:
            # Handles the data or control-flow case represented by this statement.
            yield
        finally:
            if stderr_backup_fd is not None:
                try:
                    # Calls `os.dup2` to perform the operation required at this stage.
                    os.dup2(stderr_backup_fd, 2)
                finally:
                    os.close(stderr_backup_fd)
            if devnull_fd is not None:
                # Calls `os.close` to perform the operation required at this stage.
                os.close(devnull_fd)
            if logger_cls is not None and hasattr(logger_cls, "SetStderrVerbosity") and previous_logger_cutoff is not None:
                logger_cls.SetStderrVerbosity(previous_logger_cutoff)
            if object_cls is not None:
                # Checks `` before entering this conditional branch.
                if previous_warning_state and hasattr(object_cls, "GlobalWarningDisplayOn"):
                    object_cls.GlobalWarningDisplayOn()
                elif not previous_warning_state and hasattr(object_cls, "GlobalWarningDisplayOff"):
                    object_cls.GlobalWarningDisplayOff()

            # Checks `` before entering this conditional branch.
            if (
                output_window_cls is not None
                and previous_output_window is not None
                and hasattr(output_window_cls, "SetInstance")
            # Continues the surrounding statement or collection definition.
            ):
                if previous_display_mode is not None and hasattr(previous_output_window, "SetDisplayMode"):
                    previous_output_window.SetDisplayMode(previous_display_mode)
                output_window_cls.SetInstance(previous_output_window)


# Build cell->points adjacency from connectivity.
def build_cell_to_points(connectivity: Sequence[Sequence[int]], one_based: bool = True) -> List[List[int]]:
    """Build cell->points adjacency from connectivity."""
    offset = 1 if one_based else 0
    result: List[List[int]] = []
    for cell in connectivity:
        # Builds the `pts` using the comprehension's filter and mapping rules.
        pts = [int(p) - offset for p in cell]
        result.append(pts)
    return result


def build_point_to_cells(connectivity: Sequence[Sequence[int]], num_points: int, one_based: bool = True) -> List[List[int]]:
    """Build point->cells adjacency from connectivity."""
    # Stores the `offset` for use by the remaining steps in this flow.
    offset = 1 if one_based else 0
    point_to_cells: List[List[int]] = [[] for _ in range(num_points)]
    for ci, cell in enumerate(connectivity):
        for p in cell:
            # Derives the `pi` from the current values for downstream processing.
            pi = int(p) - offset
            if 0 <= pi < num_points:
                point_to_cells[pi].append(ci)
    return point_to_cells


# Flatten entity-wise values to shape (N, C).
def _flatten_entity_values(values: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...], int]:
    """Flatten entity-wise values to shape (N, C).

    Returns:
        flat_values: (N, C)
        component_shape: original component shape after entity axis
        original_ndim: original ndim of input values
    """
    arr = np.asarray(values)
    original_ndim = arr.ndim

    if arr.ndim == 1:
        # Returns `arr[:, None], (), original_ndim` to the caller for the next stage.
        return arr[:, None], (), original_ndim

    n_entities = arr.shape[0]
    component_shape = tuple(arr.shape[1:])
    return arr.reshape(n_entities, -1), component_shape, original_ndim


# Restore flattened entity-wise values to original component shape.
def _restore_entity_values(flat_values: np.ndarray, component_shape: Tuple[int, ...], original_ndim: int) -> np.ndarray:
    """Restore flattened entity-wise values to original component shape."""
    if original_ndim == 1:
        return flat_values.squeeze(-1)
    if not component_shape:
        # Returns `flat_values` to the caller for the next stage.
        return flat_values
    return flat_values.reshape((flat_values.shape[0],) + component_shape)


def _classify_field_attribute(num_components: int) -> str:
    if num_components == 1:
        # Returns `"scalar"` to the caller for the next stage.
        return "scalar"
    if num_components == 3:
        return "vector"
    if num_components in (6, 9):
        # Returns `"tensor"` to the caller for the next stage.
        return "tensor"
    return "multi_component"


def point_to_cell(
    point_values: np.ndarray,
    # Continues the surrounding statement or collection definition.
    connectivity: Sequence[Sequence[int]],
    method: str = "mean",
    weights: Optional[Sequence[np.ndarray]] = None,
    one_based: bool = True,
# Continues the surrounding statement or collection definition.
) -> np.ndarray:
    """Convert point fields to cell fields by aggregating points in each cell."""
    values, component_shape, original_ndim = _flatten_entity_values(np.asarray(point_values))

    cells = build_cell_to_points(connectivity, one_based=one_based)
    out = np.zeros((len(cells), values.shape[1]), dtype=values.dtype)

    # Visits each item in `enumerate(cells)` and applies the current processing step.
    for ci, pts in enumerate(cells):
        if not pts:
            continue
        cell_vals = values[pts]

        # Checks `` before entering this conditional branch.
        if method == "weighted" and weights is not None and ci < len(weights):
            w = np.asarray(weights[ci], dtype=np.float64)
            if w.ndim != 1 or w.shape[0] != cell_vals.shape[0]:
                raise ValueError("point_to_cell weighted mode expects per-cell weights matching point count")
            # Converts the `wsum` to `float` so later checks use a consistent type.
            wsum = float(np.sum(w))
            if wsum <= 0:
                out[ci] = np.mean(cell_vals, axis=0)
            else:
                # Derives the `out[ci]` from the current values for downstream processing.
                out[ci] = np.sum(cell_vals * w[:, None], axis=0) / wsum
        else:
            out[ci] = np.mean(cell_vals, axis=0)

    return _restore_entity_values(out, component_shape, original_ndim)


# Convert cell fields to point fields via neighboring cells aggregation.
def cell_to_point(
    cell_values: np.ndarray,
    point_to_cells: Sequence[Sequence[int]],
    method: str = "volume_weighted",
    # Continues the surrounding statement or collection definition.
    cell_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert cell fields to point fields via neighboring cells aggregation."""
    values, component_shape, original_ndim = _flatten_entity_values(np.asarray(cell_values))

    out = np.zeros((len(point_to_cells), values.shape[1]), dtype=values.dtype)

    # Visits each item in `enumerate(point_to_cells)` and applies the current processing step.
    for pi, cells in enumerate(point_to_cells):
        if not cells:
            continue

        neigh_vals = values[cells]
        # Checks `method == "volume_weighted" and cell_weights is not None` before entering this conditional branch.
        if method == "volume_weighted" and cell_weights is not None:
            w = np.asarray(cell_weights, dtype=np.float64)[cells]
            wsum = float(np.sum(w))
            if wsum <= 0:
                # Stores the `out[pi]` for use by the remaining steps in this flow.
                out[pi] = np.mean(neigh_vals, axis=0)
            else:
                out[pi] = np.sum(neigh_vals * w[:, None], axis=0) / wsum
        else:
            # Stores the `out[pi]` for use by the remaining steps in this flow.
            out[pi] = np.mean(neigh_vals, axis=0)

    return _restore_entity_values(out, component_shape, original_ndim)


class VTKIOReader:
    """Low-level VTK IO reader utilities."""

    supported_extensions = (
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
    )

    _reader_class_names = {
        ".vtk": "vtkDataSetReader",
        # Completes the multi-line `collection` arguments or elements.
        ".vtu": "vtkXMLUnstructuredGridReader",
        ".vts": "vtkXMLStructuredGridReader",
        ".vtr": "vtkXMLRectilinearGridReader",
        ".vti": "vtkXMLImageDataReader",
        # Completes the multi-line `collection` arguments or elements.
        ".vtp": "vtkXMLPolyDataReader",
        ".pvtu": "vtkXMLPUnstructuredGridReader",
        ".pvts": "vtkXMLPStructuredGridReader",
        ".pvtr": "vtkXMLPRectilinearGridReader",
        # Completes the multi-line `collection` arguments or elements.
        ".pvti": "vtkXMLPImageDataReader",
        ".pvtp": "vtkXMLPPolyDataReader",
        ".vtm": "vtkXMLMultiBlockDataReader",
        ".vtmb": "vtkXMLMultiBlockDataReader",
    # Completes the multi-line `collection` arguments or elements.
    }

    @staticmethod
    def create_reader(file_path: str):
        ext = os.path.splitext(file_path)[1].lower()
        # Validates `ext not in VTKIOReader.supported_extensions` and raises `ValueError` when the input cannot be processed safely.
        if ext not in VTKIOReader.supported_extensions:
            raise ValueError(f"Unsupported VTK extension: {ext}")

        class_name = VTKIOReader._reader_class_names.get(ext)
        reader_cls = getattr(vtk, class_name, None)
        # Validates `reader_cls is None` and raises `ValueError` when the input cannot be processed safely.
        if reader_cls is None:
            raise ValueError(f"VTK reader '{class_name}' is not available for extension: {ext}")
        return reader_cls()

    @staticmethod
    # Defines `read_dataset` for read dataset logic.
    def read_dataset(file_path: str):
        reader = VTKIOReader.create_reader(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        reader.SetFileName(file_path)
        # legacy .vtk 默认可能只暴露每类属性的第一个数组，这里显式开启全部读取。
        # Checks `` before entering this conditional branch.
        if ext == ".vtk" and reader.__class__.__name__ == "vtkDataSetReader":
            for method_name in [
                "ReadAllScalarsOn",
                "ReadAllVectorsOn",
                # Completes the multi-line `collection` arguments or elements.
                "ReadAllTensorsOn",
                "ReadAllFieldsOn",
                "ReadAllColorScalarsOn",
                "ReadAllNormalsOn",
                # Completes the multi-line `collection` arguments or elements.
                "ReadAllTCoordsOn",
            ]:
                method = getattr(reader, method_name, None)
                if callable(method):
                    # Calls `method` to perform the operation required at this stage.
                    method()
        with _silence_vtk_runtime_output():
            reader.Update()
        ds = reader.GetOutput()
        # Validates `ds is None` and raises `ValueError` when the input cannot be processed safely.
        if ds is None:
            raise ValueError(f"Failed to read VTK dataset: {file_path}")
        return ds

    @staticmethod
    # Defines `read_dataset` for read dataset logic.
    def read_dataset(file_path: str):
        ext = os.path.splitext(file_path)[1].lower()
        with _silence_vtk_runtime_output():
            reader = VTKIOReader.create_reader(file_path)
            # Calls `reader.SetFileName` to perform the operation required at this stage.
            reader.SetFileName(file_path)
            # legacy .vtk 默认可能只暴露每类属性的第一个数组，这里显式开启全部读取。
            if ext == ".vtk" and reader.__class__.__name__ == "vtkDataSetReader":
                for method_name in [
                    "ReadAllScalarsOn",
                    # Completes the multi-line `collection` arguments or elements.
                    "ReadAllVectorsOn",
                    "ReadAllTensorsOn",
                    "ReadAllFieldsOn",
                    "ReadAllColorScalarsOn",
                    # Completes the multi-line `collection` arguments or elements.
                    "ReadAllNormalsOn",
                    "ReadAllTCoordsOn",
                ]:
                    method = getattr(reader, method_name, None)
                    # Checks `callable(method)` before entering this conditional branch.
                    if callable(method):
                        method()
            reader.Update()
            ds = reader.GetOutput()
            # Validates `ds is None` and raises `ValueError` when the input cannot be processed safely.
            if ds is None:
                raise ValueError(f"Failed to read VTK dataset: {file_path}")

            dataset_copy = ds.NewInstance()
            dataset_copy.ShallowCopy(ds)
            # Handles the data or control-flow case represented by this statement.
            del ds
            del reader
            return dataset_copy

    @staticmethod
    # Defines `iter_leaf_datasets` for iter leaf datasets logic.
    def iter_leaf_datasets(dataset) -> List[object]:
        if dataset is None:
            return []

        if isinstance(dataset, vtk.vtkDataSet):
            # Returns `[dataset]` to the caller for the next stage.
            return [dataset]

        if hasattr(dataset, "NewIterator"):
            iterator = dataset.NewIterator()
            try:
                # Calls `iterator.InitTraversal` to perform the operation required at this stage.
                iterator.InitTraversal()
                leaves: List[object] = []
                while not iterator.IsDoneWithTraversal():
                    current = iterator.GetCurrentDataObject()
                    # Adds the current value to `leaves` for later aggregation.
                    leaves.extend(VTKIOReader.iter_leaf_datasets(current))
                    iterator.GoToNextItem()
                return leaves
            finally:
                # Checks `hasattr(iterator, "UnRegister")` before entering this conditional branch.
                if hasattr(iterator, "UnRegister"):
                    iterator.UnRegister(None)

        raise ValueError(f"Unsupported VTK data object type: {dataset.GetClassName()}")

    @staticmethod
    # Defines `canonicalize_dataset` for canonicalize dataset logic.
    def canonicalize_dataset(dataset):
        leaves = VTKIOReader.iter_leaf_datasets(dataset)
        if not leaves:
            raise ValueError("VTK dataset does not contain any readable leaf datasets")

        # Checks `` before entering this conditional branch.
        if len(leaves) == 1 and isinstance(leaves[0], vtk.vtkUnstructuredGrid):
            canonical = vtk.vtkUnstructuredGrid()
            canonical.ShallowCopy(leaves[0])
            return canonical

        # Stores the `append_filter` for use by the remaining steps in this flow.
        append_filter = vtk.vtkAppendFilter()
        for leaf in leaves:
            append_filter.AddInputData(leaf)
        append_filter.Update()

        # Stores the output for use by the remaining steps in this flow.
        output = append_filter.GetOutput()
        if output is None or output.GetNumberOfPoints() == 0:
            raise ValueError("Failed to canonicalize VTK dataset into an unstructured grid")

        canonical = vtk.vtkUnstructuredGrid()
        # Calls `canonical.ShallowCopy` to perform the operation required at this stage.
        canonical.ShallowCopy(output)
        return canonical

    @staticmethod
    def extract_dataset_meta(dataset, canonical_dataset=None) -> Dict[str, object]:
        # Stores the `leaf_datasets` for use by the remaining steps in this flow.
        leaf_datasets = VTKIOReader.iter_leaf_datasets(dataset)
        if canonical_dataset is None:
            canonical_dataset = dataset if isinstance(dataset, vtk.vtkDataSet) else None

        meta: Dict[str, object] = {
            # Completes the multi-line `dataset.GetClassName` arguments or elements.
            "source_dataset_type": dataset.GetClassName() if dataset is not None else None,
            "canonical_dataset_type": canonical_dataset.GetClassName() if canonical_dataset is not None else None,
            "leaf_dataset_count": len(leaf_datasets),
            "leaf_dataset_types": [leaf.GetClassName() for leaf in leaf_datasets],
        # Completes the multi-line `collection` arguments or elements.
        }

        if canonical_dataset is not None:
            meta["num_points"] = int(canonical_dataset.GetNumberOfPoints())
            meta["num_cells"] = int(canonical_dataset.GetNumberOfCells())
            # Stores the `meta['cell_types']` for use by the remaining steps in this flow.
            meta["cell_types"] = np.unique(VTKIOReader.extract_cell_types(canonical_dataset)).tolist()

        return meta

    @staticmethod
    def read_canonical_dataset(file_path: str, geometry_policy: str = "3d_coords"):
        # Stores the `(dataset, _)` for use by the remaining steps in this flow.
        dataset, _ = VTKIOReader.read_canonical_dataset_with_meta(
            file_path=file_path,
            geometry_policy=geometry_policy,
        )
        # Returns `dataset` to the caller for the next stage.
        return dataset

    @staticmethod
    def read_canonical_dataset_with_meta(file_path: str, geometry_policy: str = "3d_coords"):
        raw_dataset = VTKIOReader.read_dataset(file_path)
        # Stores the `canonical_dataset` for use by the remaining steps in this flow.
        canonical_dataset = VTKIOReader.canonicalize_dataset(raw_dataset)
        VTKIOReader.validate_3d_dataset(canonical_dataset, geometry_policy=geometry_policy)
        meta = VTKIOReader.extract_dataset_meta(raw_dataset, canonical_dataset=canonical_dataset)
        return canonical_dataset, meta

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def read_canonical_dataset_without_meta(file_path: str, geometry_policy: str = "3d_coords"):
        """读取规范化数据集但跳过只供首帧使用的元数据抽取。"""
        raw_dataset = VTKIOReader.read_dataset(file_path)
        canonical_dataset = VTKIOReader.canonicalize_dataset(raw_dataset)
        # Calls `VTKIOReader.validate_3d_dataset` to perform the operation required at this stage.
        VTKIOReader.validate_3d_dataset(canonical_dataset, geometry_policy=geometry_policy)
        return canonical_dataset

    @staticmethod
    def validate_3d_dataset(dataset, geometry_policy: str = "3d_coords") -> None:
        # Stores the `points` for use by the remaining steps in this flow.
        points = dataset.GetPoints()
        if points is None:
            raise ValueError("VTK dataset has no points")

        first_pt = points.GetPoint(0) if points.GetNumberOfPoints() > 0 else None
        # Validates `first_pt is None or len(first_pt) != 3` and raises `ValueError` when the input cannot be processed safely.
        if first_pt is None or len(first_pt) != 3:
            raise ValueError("Only 3D point coordinates are supported")

        if geometry_policy == "3d_coords":
            return

        # Validates `geometry_policy != "3d_volume_only"` and raises `ValueError` when the input cannot be processed safely.
        if geometry_policy != "3d_volume_only":
            raise ValueError(f"Unsupported VTK geometry policy: {geometry_policy}")

        # Structured grids have explicit dimensions; require all dims > 1 for 3D volume.
        if hasattr(dataset, "GetDimensions"):
            try:
                # Stores the `dims` for use by the remaining steps in this flow.
                dims = dataset.GetDimensions()
                if dims is not None and len(dims) == 3 and (dims[0] <= 1 or dims[1] <= 1 or dims[2] <= 1):
                    raise ValueError(f"Only 3D volumetric datasets are supported, got dimensions={dims}")
            except TypeError:
                # Some datasets expose GetDimensions with out-arg style.
                # Handles the data or control-flow case represented by this statement.
                pass

        # Cell-dimension based validation for unstructured-like datasets.
        if dataset.GetNumberOfCells() > 0:
            max_dim = 0
            for i in range(dataset.GetNumberOfCells()):
                # Converts the `cdim` to `int` so later checks use a consistent type.
                cdim = int(dataset.GetCell(i).GetCellDimension())
                if cdim > max_dim:
                    max_dim = cdim
            if max_dim < 3:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError("Only 3D volumetric cells are supported (cell dimension < 3)")

    @staticmethod
    def extract_points(dataset) -> np.ndarray:
        points = dataset.GetPoints()
        # Returns early when `points is None` makes the remaining work inapplicable.
        if points is None:
            return np.empty((0, 3), dtype=np.float64)
        return numpy_support.vtk_to_numpy(points.GetData())

    @staticmethod
    # Defines `extract_connectivity` for extract connectivity logic.
    def extract_connectivity(dataset, one_based: bool = True) -> List[Tuple[int, ...]]:
        offset = 1 if one_based else 0
        connectivity: List[Tuple[int, ...]] = []

        for i in range(dataset.GetNumberOfCells()):
            # Stores the `cell` for use by the remaining steps in this flow.
            cell = dataset.GetCell(i)
            ids = [cell.GetPointId(j) + offset for j in range(cell.GetNumberOfPoints())]
            connectivity.append(tuple(ids))

        return connectivity

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def extract_connectivity_signature(dataset) -> Dict[str, np.ndarray]:
        """以紧凑数组提取单元连接签名，避免逐单元 Python 转换。"""
        cell_array = dataset.GetCells() if dataset is not None else None
        get_offsets = getattr(cell_array, "GetOffsetsArray", None)
        # Stores the `get_connectivity` for use by the remaining steps in this flow.
        get_connectivity = getattr(cell_array, "GetConnectivityArray", None)
        if callable(get_offsets) and callable(get_connectivity):
            offsets_array = get_offsets()
            connectivity_array = get_connectivity()
            # Returns early when `offsets_array is not None and connectivity_array is not None` makes the remaining work inapplicable.
            if offsets_array is not None and connectivity_array is not None:
                return {
                    "num_points": np.asarray([int(dataset.GetNumberOfPoints())], dtype=np.int64),
                    "num_cells": np.asarray([int(dataset.GetNumberOfCells())], dtype=np.int64),
                    # Completes the multi-line `np.asarray(numpy_support.vtk_to_numpy(offsets_array), dtype=np.int64).copy` arguments or elements.
                    "offsets": np.asarray(
                        numpy_support.vtk_to_numpy(offsets_array),
                        dtype=np.int64,
                    ).copy(),
                    # Completes the multi-line `np.asarray(numpy_support.vtk_to_numpy(connectivity_array), dtype=np.int64).copy` arguments or elements.
                    "connectivity": np.asarray(
                        numpy_support.vtk_to_numpy(connectivity_array),
                        dtype=np.int64,
                    ).copy(),
                    # Completes the multi-line `VTKIOReader._extract_cell_types_fast` arguments or elements.
                    "cell_types": VTKIOReader._extract_cell_types_fast(dataset),
                }

        # 兼容旧 VTK 版本和不暴露 vtkCellArray 数组的特殊数据集。
        rows = VTKIOReader.extract_connectivity(dataset, one_based=False)
        offsets = [0]
        # Initializes the `values` as a collection for values from this processing stage.
        values: List[int] = []
        for row in rows:
            values.extend(int(value) for value in row)
            offsets.append(len(values))
        # Returns `` to the caller for the next stage.
        return {
            "num_points": np.asarray([int(dataset.GetNumberOfPoints())], dtype=np.int64),
            "num_cells": np.asarray([len(rows)], dtype=np.int64),
            "offsets": np.asarray(offsets, dtype=np.int64),
            # Completes the multi-line `np.asarray` arguments or elements.
            "connectivity": np.asarray(values, dtype=np.int64),
            "cell_types": VTKIOReader._extract_cell_types_fast(dataset),
        }

    @staticmethod
    # 优先从 VTK 原始类型数组读取单元类型，旧版本回退到逐单元读取。
    def _extract_cell_types_fast(dataset) -> np.ndarray:
        """优先从 VTK 原始类型数组读取单元类型，旧版本回退到逐单元读取。"""
        get_types = getattr(dataset, "GetCellTypesArray", None)
        if callable(get_types):
            types_array = get_types()
            # Returns early when `types_array is not None` makes the remaining work inapplicable.
            if types_array is not None:
                return np.asarray(numpy_support.vtk_to_numpy(types_array), dtype=np.int32).copy()
        return VTKIOReader.extract_cell_types(dataset)

    @staticmethod
    # 比较连接签名，保持单元顺序和节点顺序的严格一致性。
    def connectivity_signatures_equal(
        expected: Dict[str, np.ndarray],
        actual: Dict[str, np.ndarray],
    ) -> bool:
        """比较连接签名，保持单元顺序和节点顺序的严格一致性。"""
        # Returns early when `not isinstance(expected, dict) or not isinstance(actual, dict)` makes the remaining work inapplicable.
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            return False
        return all(
            np.array_equal(expected.get(key), actual.get(key))
            # Completes the multi-line `collection` arguments or elements.
            for key in ("num_points", "num_cells", "offsets", "connectivity", "cell_types")
        )

    @staticmethod
    def extract_cell_center_interpolation_weights(dataset) -> Optional[List[np.ndarray]]:
        """提取 VTK 参数中心对应的节点插值权重，供同拓扑运动网格复用。"""
        # Keeps the operation and its recovery path within the same exception boundary.
        try:
            weights_by_cell: List[np.ndarray] = []
            for cell_index in range(int(dataset.GetNumberOfCells())):
                cell = dataset.GetCell(cell_index)
                # Returns early when `cell is None` makes the remaining work inapplicable.
                if cell is None:
                    return None
                point_count = int(cell.GetNumberOfPoints())
                if point_count <= 0:
                    # Returns `None` to the caller for the next stage.
                    return None

                pcoords = [0.0, 0.0, 0.0]
                sub_id = vtk.reference(int(cell.GetParametricCenter(pcoords)))
                weights = [0.0] * point_count
                # Initializes the `center` as a collection for values from this processing stage.
                center = [0.0, 0.0, 0.0]
                cell.EvaluateLocation(sub_id, pcoords, center, weights)
                values = np.asarray(weights, dtype=np.float64)
                if (
                    # Completes the multi-line `collection` arguments or elements.
                    values.shape != (point_count,)
                    or not np.all(np.isfinite(values))
                    or not np.isclose(float(values.sum()), 1.0, rtol=1e-6, atol=1e-8)
                ):
                    # Returns `None` to the caller for the next stage.
                    return None
                weights_by_cell.append(values)
            return weights_by_cell
        except Exception:
            # 少数特殊单元不支持参数中心插值，后续时间步回退到 VTK filter。
            # Returns `None` to the caller for the next stage.
            return None

    @staticmethod
    def compute_cell_centers_from_connectivity(
        points: np.ndarray,
        # Completes the multi-line `collection` arguments or elements.
        connectivity: List[Tuple[int, ...]],
        one_based: bool = True,
        cell_center_weights: Optional[Sequence[np.ndarray]] = None,
    ) -> np.ndarray:
        """由复用的连接关系和首帧参数中心权重计算单元中心。"""
        # Stores the `point_values` for use by the remaining steps in this flow.
        point_values = np.asarray(points, dtype=np.float64)
        if point_values.ndim != 2:
            raise ValueError(f"VTK 点坐标必须为二维数组，当前为 {point_values.shape}")
        if len(connectivity) == 0:
            # Returns `np.empty((0, point_values.shape[1]), dtype=point_values.dtype)` to the caller for the next stage.
            return np.empty((0, point_values.shape[1]), dtype=point_values.dtype)
        if cell_center_weights is not None and len(cell_center_weights) != len(connectivity):
            raise ValueError(
                "VTK 单元中心插值权重数量与连接关系数量不一致："
                # Completes the multi-line `len` arguments or elements.
                f"weights={len(cell_center_weights)}, cells={len(connectivity)}"
            )

        offset = 1 if one_based else 0
        centers = np.empty((len(connectivity), point_values.shape[1]), dtype=point_values.dtype)
        # Visits each item in `enumerate(connectivity)` and applies the current processing step.
        for cell_index, cell in enumerate(connectivity):
            node_ids = np.asarray(cell, dtype=np.int64).reshape(-1) - offset
            if node_ids.size == 0 or np.any(node_ids < 0) or np.any(node_ids >= point_values.shape[0]):
                raise ValueError(f"VTK 单元连接包含无效节点索引: cell_index={cell_index}")
            # Selects the required item from `point_values` as the `node_points`.
            node_points = point_values[node_ids]
            if cell_center_weights is None:
                centers[cell_index] = np.mean(node_points, axis=0)
                continue

            # Stores the `weights` for use by the remaining steps in this flow.
            weights = np.asarray(cell_center_weights[cell_index], dtype=np.float64).reshape(-1)
            if weights.shape[0] != node_ids.shape[0]:
                raise ValueError(
                    "VTK 单元中心插值权重与单元节点数不一致："
                    # Completes the multi-line `ValueError` arguments or elements.
                    f"cell_index={cell_index}, weights={weights.shape[0]}, nodes={node_ids.shape[0]}"
                )
            centers[cell_index] = np.sum(node_points * weights[:, np.newaxis], axis=0)
        return centers

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def extract_cell_types(dataset) -> np.ndarray:
        types = np.zeros((dataset.GetNumberOfCells(),), dtype=np.int32)
        for i in range(dataset.GetNumberOfCells()):
            # Converts the `types[i]` to `int` so later checks use a consistent type.
            types[i] = int(dataset.GetCellType(i))
        return types

    @staticmethod
    def extract_cells_with_types(dataset, one_based: bool = True) -> List[Tuple[int, Tuple[int, ...]]]:
        # Stores the `offset` for use by the remaining steps in this flow.
        offset = 1 if one_based else 0
        typed_cells: List[Tuple[int, Tuple[int, ...]]] = []
        for i in range(dataset.GetNumberOfCells()):
            cell = dataset.GetCell(i)
            # Stores the `ids` for use by the remaining steps in this flow.
            ids = tuple(cell.GetPointId(j) + offset for j in range(cell.GetNumberOfPoints()))
            typed_cells.append((int(dataset.GetCellType(i)), ids))
        return typed_cells

    @staticmethod
    # Defines `extract_cell_centers` for extract cell centers logic.
    def extract_cell_centers(dataset) -> np.ndarray:
        centers_filter = vtk.vtkCellCenters()
        centers_filter.SetInputData(dataset)
        centers_filter.Update()
        # Stores the `centers` for use by the remaining steps in this flow.
        centers = centers_filter.GetOutput()
        if centers is None or centers.GetPoints() is None:
            return np.empty((0, 3), dtype=np.float64)
        return numpy_support.vtk_to_numpy(centers.GetPoints().GetData())

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _normalize_extracted_array(np_arr: np.ndarray, expected_size: int) -> Optional[np.ndarray]:
        """Normalize vtk_to_numpy array to entity-first shape.

        Accepts:
        - (N,) scalar
        - (N, C) vector/tensor(flat)
        - (N, a, b, ...) tensor-like
        """
        arr = np.asarray(np_arr)

        if arr.ndim == 1:
            # Returns `arr if arr.shape[0] == expected_size else None` to the caller for the next stage.
            return arr if arr.shape[0] == expected_size else None

        if arr.shape[0] == expected_size:
            return arr

        # Fallback: flatten and try to reshape by expected_size
        if arr.size % expected_size != 0:
            # Returns `None` to the caller for the next stage.
            return None
        return arr.reshape(expected_size, -1)

    @staticmethod
    def _extract_field_dict_with_meta(
        # Continues the surrounding statement or collection definition.
        data_attr,
        expected_size: int,
        variables: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, object]]]:
        # Initializes the `out` as a collection for values from this processing stage.
        out: Dict[str, np.ndarray] = {}
        meta: Dict[str, Dict[str, object]] = {}
        if data_attr is None:
            return out, meta

        # Stores the `allowed` for use by the remaining steps in this flow.
        allowed = set(variables) if variables else None
        n_arrays = data_attr.GetNumberOfArrays()

        for i in range(n_arrays):
            arr = data_attr.GetArray(i)
            # Checks `arr is None` before entering this conditional branch.
            if arr is None:
                continue
            name = arr.GetName() or f"array_{i}"
            if allowed is not None and name not in allowed:
                # Handles the data or control-flow case represented by this statement.
                continue

            np_arr = numpy_support.vtk_to_numpy(arr)
            normalized = VTKIOReader._normalize_extracted_array(np_arr, expected_size)
            if normalized is None:
                # Handles the data or control-flow case represented by this statement.
                continue
            out[name] = normalized

            n_comp = int(arr.GetNumberOfComponents())
            comp_shape = tuple(normalized.shape[1:]) if normalized.ndim > 1 else ()
            # Initializes the `meta[name]` as a collection for values from this processing stage.
            meta[name] = {
                "num_components": n_comp,
                "attribute": _classify_field_attribute(n_comp),
                "component_shape": comp_shape,
            # Completes the multi-line `collection` arguments or elements.
            }

        return out, meta

    @staticmethod
    def _extract_named_field_dict_with_meta(
        # Continues the surrounding statement or collection definition.
        data_attr,
        expected_size: int,
        field_names: Sequence[str],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, object]]]:
        """按字段名直接提取数组，避免对时序后续帧遍历全部元数据。"""
        # Initializes the `out` as a collection for values from this processing stage.
        out: Dict[str, np.ndarray] = {}
        meta: Dict[str, Dict[str, object]] = {}
        if data_attr is None:
            return out, meta

        # Visits each item in `dict.fromkeys(str(field_name) for field_name in field_names)` and applies the current processing step.
        for name in dict.fromkeys(str(field_name) for field_name in field_names):
            arr = data_attr.GetArray(name)
            if arr is None:
                continue
            # Stores the `np_arr` for use by the remaining steps in this flow.
            np_arr = numpy_support.vtk_to_numpy(arr)
            normalized = VTKIOReader._normalize_extracted_array(np_arr, expected_size)
            if normalized is None:
                continue
            # Stores the `resolved_name` for use by the remaining steps in this flow.
            resolved_name = arr.GetName() or name
            out[resolved_name] = normalized

            component_count = int(arr.GetNumberOfComponents())
            component_shape = tuple(normalized.shape[1:]) if normalized.ndim > 1 else ()
            # Initializes the `meta[resolved_name]` as a collection for values from this processing stage.
            meta[resolved_name] = {
                "num_components": component_count,
                "attribute": _classify_field_attribute(component_count),
                "component_shape": component_shape,
            # Completes the multi-line `collection` arguments or elements.
            }
        return out, meta

    @staticmethod
    def _extract_field_dict(data_attr, expected_size: int, variables: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        # Stores the `(out, _)` for use by the remaining steps in this flow.
        out, _ = VTKIOReader._extract_field_dict_with_meta(data_attr, expected_size, variables)
        return out

    @staticmethod
    def extract_point_fields(dataset, variables: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        # Stores the `pd` for use by the remaining steps in this flow.
        pd = dataset.GetPointData()
        return VTKIOReader._extract_field_dict(pd, dataset.GetNumberOfPoints(), variables)

    @staticmethod
    def extract_cell_fields(dataset, variables: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        # Stores the `cd` for use by the remaining steps in this flow.
        cd = dataset.GetCellData()
        return VTKIOReader._extract_field_dict(cd, dataset.GetNumberOfCells(), variables)

    @staticmethod
    def extract_point_fields_with_meta(
        # Continues the surrounding statement or collection definition.
        dataset,
        variables: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, object]]]:
        pd = dataset.GetPointData()
        # Returns `` to the caller for the next stage.
        return VTKIOReader._extract_field_dict_with_meta(pd, dataset.GetNumberOfPoints(), variables)

    @staticmethod
    def extract_named_point_fields_with_meta(
        dataset,
        # Continues the surrounding statement or collection definition.
        field_names: Sequence[str],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, object]]]:
        """按名称直接提取 PointData 字段。"""
        return VTKIOReader._extract_named_field_dict_with_meta(
            dataset.GetPointData(),
            # Completes the multi-line `dataset.GetNumberOfPoints` arguments or elements.
            dataset.GetNumberOfPoints(),
            field_names,
        )

    @staticmethod
    # Defines `extract_cell_fields_with_meta` for extract cell fields with meta logic.
    def extract_cell_fields_with_meta(
        dataset,
        variables: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, object]]]:
        # Stores the `cd` for use by the remaining steps in this flow.
        cd = dataset.GetCellData()
        return VTKIOReader._extract_field_dict_with_meta(cd, dataset.GetNumberOfCells(), variables)

    @staticmethod
    def extract_named_cell_fields_with_meta(
        # Continues the surrounding statement or collection definition.
        dataset,
        field_names: Sequence[str],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, object]]]:
        """按名称直接提取 CellData 字段。"""
        return VTKIOReader._extract_named_field_dict_with_meta(
            # Completes the multi-line `dataset.GetCellData` arguments or elements.
            dataset.GetCellData(),
            dataset.GetNumberOfCells(),
            field_names,
        )

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def detect_field_location(point_fields: Dict[str, np.ndarray], cell_fields: Dict[str, np.ndarray]) -> str:
        has_point = len(point_fields) > 0
        has_cell = len(cell_fields) > 0
        # Returns early when `has_point and has_cell` makes the remaining work inapplicable.
        if has_point and has_cell:
            return "mixed"
        if has_point:
            return "point"
        # Returns early when `has_cell` makes the remaining work inapplicable.
        if has_cell:
            return "cell"
        return "empty"


class VTKIOWriter:
    """Low-level VTK IO writer utilities."""

    # Initializes the `supported_output_extensions` as a collection for values from this processing stage.
    supported_output_extensions = (".vtk", ".vtu", ".vts", ".vtr", ".vti", ".vtp")

    _family_to_extensions = {
        "unstructured": {".vtu", ".vtk"},
        "structured": {".vts", ".vtk"},
        # Completes the multi-line `collection` arguments or elements.
        "rectilinear": {".vtr", ".vtk"},
        "image": {".vti", ".vtk"},
        "polydata": {".vtp", ".vtk"},
    }

    # Initializes the `_unsupported_unstructured_cell_types` as a collection for values from this processing stage.
    _unsupported_unstructured_cell_types = {
        int(vtk.VTK_POLYHEDRON),
    }

    @staticmethod
    # Defines `infer_dataset_family` for infer dataset family logic.
    def infer_dataset_family(dataset) -> str:
        if isinstance(dataset, vtk.vtkUnstructuredGrid):
            return "unstructured"
        if isinstance(dataset, vtk.vtkStructuredGrid):
            # Returns `"structured"` to the caller for the next stage.
            return "structured"
        if isinstance(dataset, vtk.vtkRectilinearGrid):
            return "rectilinear"
        if isinstance(dataset, vtk.vtkImageData):
            # Returns `"image"` to the caller for the next stage.
            return "image"
        if isinstance(dataset, vtk.vtkPolyData):
            return "polydata"
        return "unsupported"

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def can_write_dataset_as_extension(dataset, ext: str) -> bool:
        normalized_ext = str(ext or "").strip().lower()
        if normalized_ext not in VTKIOWriter.supported_output_extensions:
            # Returns `False` to the caller for the next stage.
            return False

        if normalized_ext == ".vtk":
            return isinstance(dataset, vtk.vtkDataSet)

        family = VTKIOWriter.infer_dataset_family(dataset)
        # Returns `` to the caller for the next stage.
        return normalized_ext in VTKIOWriter._family_to_extensions.get(family, set())

    @staticmethod
    def resolve_output_extension_for_dataset(dataset, source_ext: str) -> Tuple[str, bool]:
        normalized_ext = str(source_ext or "").strip().lower()
        # Returns early when `` makes the remaining work inapplicable.
        if VTKIOWriter.can_write_dataset_as_extension(dataset, normalized_ext):
            return normalized_ext, False
        return ".vtu", True

    @staticmethod
    # Defines `validate_supported_cell_types` for validate supported cell types logic.
    def validate_supported_cell_types(typed_cells: Sequence[Tuple[int, Sequence[int]]]) -> None:
        unsupported = sorted(
            {
                int(cell_type)
                # Completes the multi-line `collection` arguments or elements.
                for cell_type, _ in typed_cells
                if int(cell_type) in VTKIOWriter._unsupported_unstructured_cell_types
            }
        )
        # Validates `unsupported` and raises `ValueError` when the input cannot be processed safely.
        if unsupported:
            raise ValueError(
                f"Unsupported VTK cell types for unstructured fallback export: {unsupported}"
            )

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _guess_vtk_cell_type_from_size(num_points: int) -> int:
        if num_points == 1:
            return int(vtk.VTK_VERTEX)
        # Returns early when `num_points == 2` makes the remaining work inapplicable.
        if num_points == 2:
            return int(vtk.VTK_LINE)
        if num_points == 3:
            return int(vtk.VTK_TRIANGLE)
        # Returns early when `num_points == 4` makes the remaining work inapplicable.
        if num_points == 4:
            return int(vtk.VTK_TETRA)
        if num_points == 5:
            return int(vtk.VTK_PYRAMID)
        # Returns early when `num_points == 6` makes the remaining work inapplicable.
        if num_points == 6:
            return int(vtk.VTK_WEDGE)
        if num_points == 8:
            return int(vtk.VTK_HEXAHEDRON)
        # Returns `int(vtk.VTK_CONVEX_POINT_SET)` to the caller for the next stage.
        return int(vtk.VTK_CONVEX_POINT_SET)

    @staticmethod
    def _build_unstructured_grid(
        points: np.ndarray,
        # Continues the surrounding statement or collection definition.
        connectivity: Sequence[Sequence[int]],
        one_based: bool = True,
    ):
        typed_cells = [
            # Completes the multi-line `collection` arguments or elements.
            (VTKIOWriter._guess_vtk_cell_type_from_size(len(cell)), tuple(cell))
            for cell in connectivity
        ]
        return VTKIOWriter.build_unstructured_grid_with_cell_types(
            # Completes the multi-line `VTKIOWriter.build_unstructured_grid_with_cell_types` arguments or elements.
            points=points,
            typed_cells=typed_cells,
            one_based=one_based,
        )

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def build_unstructured_grid_with_cell_types(
        points: np.ndarray,
        typed_cells: Sequence[Tuple[int, Sequence[int]]],
        # Continues the surrounding statement or collection definition.
        one_based: bool = True,
    ):
        grid = vtk.vtkUnstructuredGrid()

        vtk_points = vtk.vtkPoints()
        # Visits each item in `np.asarray(points)` and applies the current processing step.
        for pt in np.asarray(points):
            vtk_points.InsertNextPoint(float(pt[0]), float(pt[1]), float(pt[2]))
        grid.SetPoints(vtk_points)

        offset = 1 if one_based else 0
        # Calls `VTKIOWriter.validate_supported_cell_types` to perform the operation required at this stage.
        VTKIOWriter.validate_supported_cell_types(typed_cells)

        for cell_type, cell in typed_cells:
            ids = [int(i) - offset for i in cell]
            id_list = vtk.vtkIdList()
            # Visits each item in `ids` and applies the current processing step.
            for pid in ids:
                id_list.InsertNextId(pid)

            grid.InsertNextCell(int(cell_type), id_list)

        return grid

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _attach_fields(dataset, fields: Dict[str, np.ndarray], location: str) -> None:
        if location not in {"point", "cell"}:
            raise ValueError(f"location must be 'point' or 'cell', got {location}")

        # Stores the `attr` for use by the remaining steps in this flow.
        attr = dataset.GetPointData() if location == "point" else dataset.GetCellData()
        expected = dataset.GetNumberOfPoints() if location == "point" else dataset.GetNumberOfCells()

        for name, values in fields.items():
            arr = np.asarray(values)
            # Validates `arr.ndim == 0` and raises `ValueError` when the input cannot be processed safely.
            if arr.ndim == 0:
                raise ValueError(f"Field '{name}' cannot be scalar literal")

            if arr.shape[0] != expected:
                if arr.size % expected != 0:
                    # Raises `ValueError` to report why this path cannot continue.
                    raise ValueError(
                        f"Field '{name}' size mismatch for {location} data: {arr.shape[0]} vs {expected}"
                    )
                arr = arr.reshape(expected, -1)

            # Ensure numpy_to_vtk receives scalar(1D) or multi-component(2D) arrays.
            # Checks `arr.ndim > 2` before entering this conditional branch.
            if arr.ndim > 2:
                arr = arr.reshape(expected, -1)

            if arr.shape[0] != expected:
                raise ValueError(
                    # Completes the multi-line `ValueError` arguments or elements.
                    f"Field '{name}' size mismatch for {location} data: {arr.shape[0]} vs {expected}"
                )

            attr.RemoveArray(name)
            vtk_arr = numpy_support.numpy_to_vtk(arr, deep=True)
            # Calls `vtk_arr.SetName` to perform the operation required at this stage.
            vtk_arr.SetName(name)
            attr.AddArray(vtk_arr)

    @staticmethod
    def _clear_data_arrays(data_attr) -> None:
        # Returns early when `data_attr is None` makes the remaining work inapplicable.
        if data_attr is None:
            return
        while data_attr.GetNumberOfArrays() > 0:
            arr = data_attr.GetArray(0)
            # Stores the name for use by the remaining steps in this flow.
            name = arr.GetName() if arr is not None else None
            if name:
                data_attr.RemoveArray(name)
            else:
                # Calls `data_attr.RemoveArray` to perform the operation required at this stage.
                data_attr.RemoveArray(0)

    @staticmethod
    def clear_dataset_field_arrays(dataset) -> None:
        VTKIOWriter._clear_data_arrays(dataset.GetPointData())
        # Calls `VTKIOWriter._clear_data_arrays` to perform the operation required at this stage.
        VTKIOWriter._clear_data_arrays(dataset.GetCellData())

    @staticmethod
    def update_dataset_points(dataset, points: np.ndarray) -> None:
        normalized = np.asarray(points, dtype=np.float64)
        # Validates `normalized.ndim != 2 or normalized.shape[1] != 3` and raises `ValueError` when the input cannot be processed safely.
        if normalized.ndim != 2 or normalized.shape[1] != 3:
            raise ValueError(f"VTK 点坐标必须为 [N, 3]，当前为 {normalized.shape}")
        if int(dataset.GetNumberOfPoints()) != int(normalized.shape[0]):
            raise ValueError(
                # Completes the multi-line `ValueError` arguments or elements.
                "VTK 点坐标数量与模板网格不匹配: "
                f"actual={normalized.shape[0]}, expected={dataset.GetNumberOfPoints()}"
            )
        vtk_points = vtk.vtkPoints()
        # Calls `vtk_points.SetData` to perform the operation required at this stage.
        vtk_points.SetData(numpy_support.numpy_to_vtk(normalized, deep=True))
        dataset.SetPoints(vtk_points)

    @staticmethod
    def _write_dataset(dataset, output_file: str) -> None:
        # Stores the `ext` for use by the remaining steps in this flow.
        ext = os.path.splitext(output_file)[1].lower()
        if not VTKIOWriter.can_write_dataset_as_extension(dataset, ext):
            family = VTKIOWriter.infer_dataset_family(dataset)
            raise ValueError(
                # Completes the multi-line `ValueError` arguments or elements.
                f"VTK dataset type mismatch for output extension: ext={ext}, dataset_family={family}, "
                f"dataset_type={dataset.GetClassName()}"
            )
        if ext == ".vtu":
            # Stores the `writer` for use by the remaining steps in this flow.
            writer = vtk.vtkXMLUnstructuredGridWriter()
        elif ext == ".vtk":
            writer = vtk.vtkDataSetWriter()
        elif ext == ".vts":
            # Stores the `writer` for use by the remaining steps in this flow.
            writer = vtk.vtkXMLStructuredGridWriter()
        elif ext == ".vtr":
            writer = vtk.vtkXMLRectilinearGridWriter()
        elif ext == ".vti":
            # Stores the `writer` for use by the remaining steps in this flow.
            writer = vtk.vtkXMLImageDataWriter()
        elif ext == ".vtp":
            writer = vtk.vtkXMLPolyDataWriter()
        else:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(f"Unsupported VTK output extension: {ext}")

        writer.SetFileName(output_file)
        writer.SetInputData(dataset)
        result = writer.Write()
        # Validates `result == 0` and raises `ValueError` when the input cannot be processed safely.
        if result == 0:
            raise ValueError(f"Failed to write VTK dataset: {output_file}")

    @staticmethod
    def write_unstructured_grid(
        # Continues the surrounding statement or collection definition.
        points: np.ndarray,
        connectivity: Sequence[Sequence[int]],
        fields: Dict[str, np.ndarray],
        location: str,
        # Continues the surrounding statement or collection definition.
        output_file: str,
        one_based: bool = True,
    ) -> None:
        grid = VTKIOWriter._build_unstructured_grid(points, connectivity, one_based=one_based)
        # Calls `VTKIOWriter._attach_fields` to perform the operation required at this stage.
        VTKIOWriter._attach_fields(grid, fields, location)
        VTKIOWriter._write_dataset(grid, output_file)

    @staticmethod
    def write_typed_unstructured_grid(
        # Continues the surrounding statement or collection definition.
        points: np.ndarray,
        typed_cells: Sequence[Tuple[int, Sequence[int]]],
        fields: Dict[str, np.ndarray],
        location: str,
        # Continues the surrounding statement or collection definition.
        output_file: str,
        one_based: bool = True,
    ) -> None:
        grid = VTKIOWriter.build_unstructured_grid_with_cell_types(
            # Completes the multi-line `VTKIOWriter.build_unstructured_grid_with_cell_types` arguments or elements.
            points=points,
            typed_cells=typed_cells,
            one_based=one_based,
        )
        # Calls `VTKIOWriter._attach_fields` to perform the operation required at this stage.
        VTKIOWriter._attach_fields(grid, fields, location)
        VTKIOWriter._write_dataset(grid, output_file)

    @staticmethod
    def update_fields_and_write(source_file: str, output_file: str, fields: Dict[str, np.ndarray], location: str) -> None:
        # Stores the dataset for use by the remaining steps in this flow.
        dataset = VTKIOReader.read_dataset(source_file)
        VTKIOWriter.clear_dataset_field_arrays(dataset)
        VTKIOWriter._attach_fields(dataset, fields, location)
        VTKIOWriter._write_dataset(dataset, output_file)

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def update_dataset_and_write(
        dataset,
        output_file: str,
        # Completes the multi-line `collection` arguments or elements.
        fields: Dict[str, np.ndarray],
        location: str,
        point_coordinates: Optional[np.ndarray] = None,
    ) -> None:
        # Checks `point_coordinates is not None` before entering this conditional branch.
        if point_coordinates is not None:
            VTKIOWriter.update_dataset_points(dataset, point_coordinates)
        VTKIOWriter.clear_dataset_field_arrays(dataset)
        VTKIOWriter._attach_fields(dataset, fields, location)
        # Calls `VTKIOWriter._write_dataset` to perform the operation required at this stage.
        VTKIOWriter._write_dataset(dataset, output_file)
