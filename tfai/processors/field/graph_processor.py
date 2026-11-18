import copy
import logging
import os
# Imports `Any, Dict, List` from `typing` for use in this module.
from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np
from scipy.spatial import cKDTree
# Imports `checkpoint` from `torch.utils.checkpoint` for use in this module.
from torch.utils.checkpoint import checkpoint
from .field_processor import FieldProcessor
from ...data.base_data import BaseData
from ...utils.logger import Logger
# Imports `atomic_torch_save, load_torch` from `utils.torch_utils` for use in this module.
from ...utils.torch_utils import atomic_torch_save, load_torch


def _get_logger() -> logging.Logger:
    return Logger.get_first_configured_logger(
        ["train", "predict", "evaluate"],
        # Completes the multi-line `Logger.get_first_configured_logger` arguments or elements.
        fallback_name=__name__,
    )


_GRAPH_NON_PARAM_SAMPLE_KEYS = {
    'file',
    # Completes the multi-line `collection` arguments or elements.
    'file_path',
    'params',
    'source_file',
    'resolved_path',
    # Completes the multi-line `collection` arguments or elements.
    'path_type',
    'exists',
    'processed_path',
    'subset',
    # Completes the multi-line `collection` arguments or elements.
    'split',
    'sample_order',
    'order',
    'sample_id',
    # Completes the multi-line `collection` arguments or elements.
    'source_exists',
    'processed_exists',
    'evaluation_status',
    'evaluated_result_path',
    # Completes the multi-line `collection` arguments or elements.
    'evaluated_result_exists',
    'target_result_path',
    'target_result_exists',
    'temporal_steps',
    # Completes the multi-line `collection` arguments or elements.
    'total_time_steps',
    'time_axis',
    'geometry_regime',
    'displacement_logical_name',
    # Completes the multi-line `collection` arguments or elements.
    'displacement_logical_variable',
    'displacement_effective_variables',
    'displacement_source',
    'displacement_coordinate_convention',
# Completes the multi-line `collection` arguments or elements.
}

_ROLLOUT_CHECKPOINT_MODEL_NAMES = {"GCN", "GraphTransformer", "GraphUNet"}


class GraphProcessor(FieldProcessor):
    """图结构格式场处理器
    
    将场数据处理为图结构格式，适用于图神经网络等模型。
    
    关键特性：
    - 自动识别字段位置（节点 vs 单元）
    - 几何固定时自动缓存拓扑，几何变化时每次重建
    - 输出为 PyG Data 对象，兼容当前 dataset 数据栈
    """
    
    def __init__(self, config: Dict):
        """初始化图结构处理器
        
        Args:
            config: 处理器配置参数
        """
        # Calls `super().__init__` to perform the operation required at this stage.
        super().__init__(config)
        self.output_format = 'graph'
        processor_cfg = config.get('processor', {})
        self.graph_edge_strategy = str(processor_cfg.get('graph_edge_strategy', 'topology') or 'topology').lower()
        # Validates `self.graph_edge_strategy not in {'topology', 'knn', 'radius'}` and raises `ValueError` when the input cannot be processed safely.
        if self.graph_edge_strategy not in {'topology', 'knn', 'radius'}:
            raise ValueError(
                f"processor.graph_edge_strategy 仅支持 'topology' 或 'knn'，当前为 {self.graph_edge_strategy!r}"
            )
        # Stores the `graph_knn_k` on the instance so other methods can reuse it.
        self.graph_knn_k = int(processor_cfg.get('graph_knn_k', 0) or 0)
        self.graph_radius = float(processor_cfg.get('graph_radius', 0.0) or 0.0)
        legacy_radius_neighbor_cap = processor_cfg.get('graph_radius_max_neighbors', None)
        self.graph_radius_max_neighbors = (
            # Completes the multi-line `int` arguments or elements.
            int(legacy_radius_neighbor_cap or 0)
            if legacy_radius_neighbor_cap is not None
            else 0
        )
        # 可选：每个节点的最大边数（incident edges），默认不启用。
        # Stores the `graph_max_edges_per_node` on the instance so other methods can reuse it.
        self.graph_max_edges_per_node = int(processor_cfg.get('graph_max_edges_per_node', 0) or 0)
        
        # 几何固定时的缓存
        self.graph_radius_candidate_limit = self.graph_max_edges_per_node
        if legacy_radius_neighbor_cap is not None:
            logger = _get_logger()
            # Checks `self.graph_max_edges_per_node > 0` before entering this conditional branch.
            if self.graph_max_edges_per_node > 0:
                if self.graph_radius_max_neighbors > 0 and self.graph_radius_max_neighbors != self.graph_max_edges_per_node:
                    logger.warning(
                        "processor.graph_radius_max_neighbors 已废弃，当前优先使用 graph_max_edges_per_node=%s，忽略 graph_radius_max_neighbors=%s",
                        # Completes the multi-line `logger.warning` arguments or elements.
                        self.graph_max_edges_per_node,
                        self.graph_radius_max_neighbors,
                    )
                else:
                    # Records the current execution state to make progress and failures traceable.
                    logger.warning(
                        "processor.graph_radius_max_neighbors 已废弃，请改用 graph_max_edges_per_node"
                    )
            elif self.graph_radius_max_neighbors > 0:
                # Stores the `graph_radius_candidate_limit` on the instance so other methods can reuse it.
                self.graph_radius_candidate_limit = self.graph_radius_max_neighbors
                self.graph_max_edges_per_node = self.graph_radius_max_neighbors
                logger.warning(
                    "processor.graph_radius_max_neighbors 已废弃，已自动映射到 graph_max_edges_per_node=%s，请尽快更新配置",
                    # Completes the multi-line `logger.warning` arguments or elements.
                    self.graph_max_edges_per_node,
                )
        self.cached_topology = None  # {edge_index, field_location, num_nodes}
        self.first_process_done = False  # 标记首次处理是否完成
        # Stores the `_last_local_feature_meta` on the instance so other methods can reuse it.
        self._last_local_feature_meta: Dict[str, Any] = {}

    def infer_model_overrides(self, sample: Dict, config: Dict = None) -> Dict:
        if not isinstance(sample, dict):
            raise ValueError("GraphProcessor.infer_model_overrides expects a dict sample payload")
        # Validates `'x' not in sample or 'y' not in sample` and raises `ValueError` when the input cannot be processed safely.
        if 'x' not in sample or 'y' not in sample:
            raise ValueError("GraphProcessor.infer_model_overrides requires processed sample keys: x, y")

        x = sample['x']
        y = sample['y']
        # Validates `not hasattr(x, 'x')` and raises `ValueError` when the input cannot be processed safely.
        if not hasattr(x, 'x'):
            raise ValueError("GraphProcessor.infer_model_overrides expects sample['x'] to be a graph object with node features in .x")
        if not hasattr(x.x, 'shape') or len(x.x.shape) != 2:
            raise ValueError("GraphProcessor.infer_model_overrides expects sample['x'].x to be a 2D node feature tensor")
        # Validates `not hasattr(y, 'shape')` and raises `ValueError` when the input cannot be processed safely.
        if not hasattr(y, 'shape'):
            raise ValueError("GraphProcessor.infer_model_overrides expects sample['y'] to be a tensor-like object with shape")

        task_cfg = (config or self.config).get("task", {})
        mode = str(task_cfg.get("mode", self.mode)).lower()
        # Stores the `temporal_strategy` for use by the remaining steps in this flow.
        temporal_strategy = str(task_cfg.get("temporal_strategy", "direct")).lower()

        if mode == "temporal" and temporal_strategy == "direct":
            if y.ndim != 3:
                raise ValueError(
                    # Completes the multi-line `ValueError` arguments or elements.
                    f"GraphProcessor temporal direct 期望时序 y 形状为 [T, N, C]，当前维度为 {y.ndim}"
                )
            out_channels = int(y.shape[0]) * int(y.shape[-1])
            return {
                # Completes the multi-line `int` arguments or elements.
                'in_channels': int(x.x.shape[1]),
                'out_channels': out_channels,
            }

        if mode == "temporal" and temporal_strategy == "autoregressive":
            # Converts the `input_length` to `int` so later checks use a consistent type.
            input_length = int(task_cfg.get("input_length", 1))
            output_length = int(task_cfg.get("output_length", 1))
            if output_length != 1:
                raise NotImplementedError(
                    # Completes the multi-line `NotImplementedError` arguments or elements.
                    "GraphProcessor 当前仅支持 output_length=1 的 autoregressive 训练视图。"
                )
            if y.ndim != 3:
                raise ValueError(
                    # Completes the multi-line `ValueError` arguments or elements.
                    f"GraphProcessor autoregressive 期望时序 y 形状为 [T, N, C]，当前维度为 {y.ndim}"
                )
            out_channels = int(y.shape[-1])
            return {
                # Completes the multi-line `int` arguments or elements.
                'in_channels': int(x.x.shape[1]) + input_length * out_channels,
                'out_channels': out_channels,
            }

        out_channels = int(y.shape[-1]) if y.ndim >= 2 else int(y.numel())
        # Returns `` to the caller for the next stage.
        return {
            'in_channels': int(x.x.shape[1]),
            'out_channels': out_channels,
        }

    # Defines `_get_output_field_channels` for  get output field channels logic.
    def _get_output_field_channels(self) -> int:
        field_names = (
            self.config.get("task", {}).get("effective_variables")
            or self.config.get("task", {}).get("variables", [])
        # Continues the surrounding statement or collection definition.
        )
        if not field_names:
            raise ValueError("GraphProcessor 无法确定输出场通道数，请检查 task.variables 配置")
        return int(len(field_names))

    # Defines `_reshape_temporal_direct_target` for  reshape temporal direct target logic.
    def _reshape_temporal_direct_target(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim != 3:
            return y
        return y.permute(1, 0, 2).reshape(y.shape[1], -1)

    # Defines `_restore_temporal_direct_output` for  restore temporal direct output logic.
    def _restore_temporal_direct_output(self, values: Any) -> Any:
        field_channels = self._get_output_field_channels()

        if isinstance(values, torch.Tensor):
            if values.ndim != 2:
                # Returns `values` to the caller for the next stage.
                return values
            folded_channels = int(values.shape[-1])
            if folded_channels % field_channels != 0:
                return values
            # Derives the `time_steps` from the current values for downstream processing.
            time_steps = folded_channels // field_channels
            return values.reshape(values.shape[0], time_steps, field_channels).permute(1, 0, 2)

        array = np.asarray(values)
        if array.ndim != 2:
            # Returns `array` to the caller for the next stage.
            return array
        folded_channels = int(array.shape[-1])
        if folded_channels % field_channels != 0:
            return array
        # Derives the `time_steps` from the current values for downstream processing.
        time_steps = folded_channels // field_channels
        return np.transpose(
            array.reshape(array.shape[0], time_steps, field_channels),
            (1, 0, 2),
        # Completes the multi-line `np.transpose` arguments or elements.
        )

    def build_prediction_adapter(
        self,
        reader,
        # Continues the surrounding statement or collection definition.
        config: Dict,
        is_geometry_varying: bool,
        resolve_reference_file,
        logger=None,
    # Continues the surrounding statement or collection definition.
    ):
        from ...prediction.adapters import GraphPredictAdapter

        return GraphPredictAdapter(
            processor=self,
            # Completes the multi-line `GraphPredictAdapter` arguments or elements.
            reader=reader,
            config=config,
            is_geometry_varying=is_geometry_varying,
            resolve_reference_file=resolve_reference_file,
            # Completes the multi-line `GraphPredictAdapter` arguments or elements.
            logger=logger,
        )

    def build_direct_view(
        self,
        # Completes the multi-line `collection` arguments or elements.
        sample: Dict[str, Any],
    ) -> Tuple[Any, Any]:
        graph = sample["x"]
        y = sample["y"]
        # Returns early when `` makes the remaining work inapplicable.
        if (
            str(self.config.get("task", {}).get("mode", self.mode)).lower() == "temporal"
            and isinstance(y, torch.Tensor)
            and y.ndim == 3
        # Continues the surrounding statement or collection definition.
        ):
            return graph, self._reshape_temporal_direct_target(y)
        return graph, y

    @staticmethod
    # Defines `_is_displaced_static_topology` for  is displaced static topology logic.
    def _is_displaced_static_topology(meta: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(meta, dict):
            return False
        return (
            # Completes the multi-line `str(meta.get('geometry_regime') or '').strip().lower` arguments or elements.
            str(meta.get("geometry_regime") or "").strip().lower()
            == "displaced_static_topology"
        )

    def _has_explicit_motion_graph_contract(
        # Continues the surrounding statement or collection definition.
        self,
        meta: Optional[Dict[str, Any]],
    ) -> bool:
        """仅在目标变量明确包含位移分量时启用运动网格图状态。"""
        if not self._is_displaced_static_topology(meta):
            # Returns `False` to the caller for the next stage.
            return False

        task_cfg = self.config.get("task", {})
        effective_variables = list(
            task_cfg.get("effective_variables")
            # Completes the multi-line `task_cfg.get` arguments or elements.
            or task_cfg.get("variables")
            or []
        )
        displacement_variables = list(
            # Completes the multi-line `(meta or {}).get` arguments or elements.
            (meta or {}).get("displacement_effective_variables")
            or task_cfg.get("displacement_effective_variables")
            or []
        )
        # Returns `` to the caller for the next stage.
        return bool(displacement_variables) and all(
            field_name in effective_variables
            for field_name in displacement_variables
        )

    # 解析目标通道中位移分量到坐标轴的映射。
    def _resolve_motion_displacement_layout(
        self,
        meta: Optional[Dict[str, Any]],
        out_channels: Optional[int] = None,
    # Completes the multi-line `collection` arguments or elements.
    ) -> List[Tuple[int, int]]:
        """解析目标通道中位移分量到坐标轴的映射。"""
        task_cfg = self.config.get("task", {})
        effective_variables = list(
            task_cfg.get("effective_variables")
            # Completes the multi-line `task_cfg.get` arguments or elements.
            or task_cfg.get("variables")
            or []
        )
        displacement_variables = list(
            # Completes the multi-line `(meta or {}).get` arguments or elements.
            (meta or {}).get("displacement_effective_variables")
            or task_cfg.get("displacement_effective_variables")
            or []
        )
        # Validates `not effective_variables` and raises `ValueError` when the input cannot be processed safely.
        if not effective_variables:
            raise ValueError("GraphProcessor 无法解析运动网格位移通道：task.effective_variables 为空")
        if not displacement_variables:
            raise ValueError("GraphProcessor 运动网格缺少 displacement_effective_variables")

        # Initializes the `target_indexes` as a collection for values from this processing stage.
        target_indexes: List[int] = []
        for field_name in displacement_variables:
            try:
                target_indexes.append(effective_variables.index(field_name))
            # Continues the surrounding statement or collection definition.
            except ValueError as exc:
                raise ValueError(
                    "GraphProcessor 运动网格位移字段未包含在目标变量中："
                    f"field={field_name}, effective_variables={effective_variables}"
                # Completes the multi-line `ValueError` arguments or elements.
                ) from exc

        if out_channels is not None and any(index >= int(out_channels) for index in target_indexes):
            raise ValueError(
                "GraphProcessor 模型输出通道不足以承载运动网格位移字段："
                # Completes the multi-line `ValueError` arguments or elements.
                f"out_channels={out_channels}, displacement_indexes={target_indexes}"
            )

        layout: List[Tuple[int, int]] = []
        used_axes = set()
        # Visits each item in `enumerate( zip(displacement_variables, target_indexes) )` and applies the current processing step.
        for component_index, (field_name, target_index) in enumerate(
            zip(displacement_variables, target_indexes)
        ):
            normalized_name = str(field_name).strip().casefold()
            # Checks `normalized_name.endswith("x")` before entering this conditional branch.
            if normalized_name.endswith("x"):
                coordinate_index = 0
            elif normalized_name.endswith("y"):
                coordinate_index = 1
            # Checks `normalized_name.endswith("z")` before entering this conditional branch.
            elif normalized_name.endswith("z"):
                coordinate_index = 2
            else:
                coordinate_index = component_index
            # Validates `coordinate_index >= 3` and raises `ValueError` when the input cannot be processed safely.
            if coordinate_index >= 3:
                raise ValueError(
                    "GraphProcessor 仅支持最多三个空间位移分量，"
                    f"field={field_name}, coordinate_index={coordinate_index}"
                # Completes the multi-line `ValueError` arguments or elements.
                )
            if coordinate_index in used_axes:
                raise ValueError(
                    "GraphProcessor 运动网格位移字段存在重复坐标轴映射："
                    # Completes the multi-line `ValueError` arguments or elements.
                    f"field={field_name}, coordinate_index={coordinate_index}"
                )
            used_axes.add(coordinate_index)
            layout.append((target_index, coordinate_index))
        # Returns `layout` to the caller for the next stage.
        return layout

    def _build_motion_coordinate_transform(
        self,
        meta: Optional[Dict[str, Any]],
        # Completes the multi-line `collection` arguments or elements.
        norm_params: Optional[Dict[str, Any]] = None,
        norm_type: Optional[str] = None,
        targets_normalized: bool = False,
    ) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
        """构建目标位移到当前图坐标的线性变换。"""
        # Stores the `layout` for use by the remaining steps in this flow.
        layout = self._resolve_motion_displacement_layout(meta)
        component_count = len(layout)
        if not norm_params:
            return (
                # Completes the multi-line `collection` arguments or elements.
                layout,
                np.ones((component_count,), dtype=np.float32),
                np.zeros((component_count,), dtype=np.float32),
            )

        # Stores the `geometry` for use by the remaining steps in this flow.
        geometry = norm_params.get("geometry") or {}
        coord_min = np.asarray(geometry.get("coord_min") or [], dtype=np.float64)
        coord_max = np.asarray(geometry.get("coord_max") or [], dtype=np.float64)
        if coord_min.shape != (3,) or coord_max.shape != (3,):
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("GraphProcessor 运动网格归一化缺少三维 geometry 坐标范围")
        coord_span = coord_max - coord_min
        safe_coord_span = np.where(np.abs(coord_span) < 1e-12, 1.0, coord_span)

        y_stats = norm_params.get("y") or {}
        # Stores the `resolved_norm_type` for use by the remaining steps in this flow.
        resolved_norm_type = str(norm_type or norm_params.get("norm_type") or "").lower()

        def get_target_stats(name: str) -> np.ndarray:
            values = y_stats.get(name)
            return np.asarray([] if values is None else values, dtype=np.float64)

        # Initializes the `scales` as a collection for values from this processing stage.
        scales: List[float] = []
        offsets: List[float] = []
        for target_index, coordinate_index in layout:
            target_scale = 1.0
            # Stores the `target_offset` for use by the remaining steps in this flow.
            target_offset = 0.0
            if targets_normalized:
                if resolved_norm_type == "standardization":
                    std_values = get_target_stats("std")
                    # Stores the `mean_values` for use by the remaining steps in this flow.
                    mean_values = get_target_stats("mean")
                    if target_index >= len(std_values) or target_index >= len(mean_values):
                        raise ValueError("GraphProcessor 运动网格归一化缺少位移通道统计量")
                    target_scale = float(std_values[target_index])
                    # Checks `abs(target_scale) < 1e-12` before entering this conditional branch.
                    if abs(target_scale) < 1e-12:
                        target_scale = 1.0
                    target_offset = float(mean_values[target_index])
                elif resolved_norm_type == "min_max":
                    # Stores the `min_values` for use by the remaining steps in this flow.
                    min_values = get_target_stats("min")
                    max_values = get_target_stats("max")
                    if target_index >= len(min_values) or target_index >= len(max_values):
                        raise ValueError("GraphProcessor 运动网格归一化缺少位移通道统计量")
                    # Converts the `target_offset` to `float` so later checks use a consistent type.
                    target_offset = float(min_values[target_index])
                    target_scale = float(max_values[target_index] - min_values[target_index])
                    if abs(target_scale) < 1e-12:
                        target_scale = 1.0
                # Continues the surrounding statement or collection definition.
                else:
                    raise ValueError(f"GraphProcessor 不支持的运动网格目标归一化方式: {resolved_norm_type}")

            coordinate_scale = 2.0 / float(safe_coord_span[coordinate_index])
            scales.append(float(coordinate_scale * target_scale))
            # Adds the current value to `offsets` for later aggregation.
            offsets.append(float(coordinate_scale * target_offset))

        return (
            layout,
            np.asarray(scales, dtype=np.float32),
            # Completes the multi-line `np.asarray` arguments or elements.
            np.asarray(offsets, dtype=np.float32),
        )

    def _attach_motion_graph_state(
        self,
        # Continues the surrounding statement or collection definition.
        graph: Any,
        meta: Optional[Dict[str, Any]],
        norm_params: Optional[Dict[str, Any]] = None,
        norm_type: Optional[str] = None,
        # Continues the surrounding statement or collection definition.
        targets_normalized: bool = False,
    ) -> bool:
        """为运动网格图保留参考坐标和位移到坐标的转换参数。"""
        if not self._has_explicit_motion_graph_contract(meta):
            return False
        # Validates `self.graph_edge_strategy != "topology"` and raises `NotImplementedError` when the input cannot be processed safely.
        if self.graph_edge_strategy != "topology":
            raise NotImplementedError(
                "GraphProcessor 运动网格当前仅支持 processor.graph_edge_strategy=topology"
            )
        # Stores the `field_location` for use by the remaining steps in this flow.
        field_location = str(getattr(graph, "field_location", "point") or "point").lower()
        if field_location != "point":
            raise ValueError("GraphProcessor 运动网格当前仅支持点场图")
        if not hasattr(graph, "x") or graph.x.ndim != 2 or graph.x.shape[1] < 3:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("GraphProcessor 运动网格图缺少前三个坐标特征")

        _, scales, offsets = self._build_motion_coordinate_transform(
            meta=meta,
            norm_params=norm_params,
            # Completes the multi-line `self._build_motion_coordinate_transform` arguments or elements.
            norm_type=norm_type,
            targets_normalized=targets_normalized,
        )
        node_count = int(graph.x.shape[0])
        # Stores the `dtype` for use by the remaining steps in this flow.
        dtype = graph.x.dtype
        device = graph.x.device
        graph.motion_reference_pos = graph.x[:, :3].detach().clone()
        graph.motion_coordinate_scale = torch.as_tensor(
            # Completes the multi-line `torch.as_tensor(scales, dtype=dtype, device=device).reshape(1, -1).expand(node_count, -1).clone` arguments or elements.
            scales,
            dtype=dtype,
            device=device,
        ).reshape(1, -1).expand(node_count, -1).clone()
        # Stores the `motion_coordinate_offset` for use by the remaining steps in this flow.
        graph.motion_coordinate_offset = torch.as_tensor(
            offsets,
            dtype=dtype,
            device=device,
        # Completes the multi-line `torch.as_tensor(offsets, dtype=dtype, device=device).reshape(1, -1).expand(node_count, -1).clone` arguments or elements.
        ).reshape(1, -1).expand(node_count, -1).clone()
        return True

    def _ensure_motion_graph_state(
        self,
        # Continues the surrounding statement or collection definition.
        graph: Any,
        meta: Optional[Dict[str, Any]],
    ) -> bool:
        """兼容旧处理产物，为缺失状态的运动网格补齐默认物理坐标转换。"""
        has_state = all(
            # Completes the multi-line `hasattr` arguments or elements.
            hasattr(graph, attr)
            for attr in (
                "motion_reference_pos",
                "motion_coordinate_scale",
                # Completes the multi-line `collection` arguments or elements.
                "motion_coordinate_offset",
            )
        )
        if has_state:
            # Returns `True` to the caller for the next stage.
            return True
        return self._attach_motion_graph_state(graph=graph, meta=meta)

    def _update_motion_graph_coordinates(
        self,
        # Continues the surrounding statement or collection definition.
        graph: Any,
        displacement_values: torch.Tensor,
        meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """根据绝对位移更新当前图坐标和边属性。"""
        # Returns early when `not self._ensure_motion_graph_state(graph=graph, meta=meta)` makes the remaining work inapplicable.
        if not self._ensure_motion_graph_state(graph=graph, meta=meta):
            return False
        if displacement_values.ndim != 2:
            raise ValueError(
                # Completes the multi-line `ValueError` arguments or elements.
                "GraphProcessor 运动网格位移张量应为 [N, C]，"
                f"当前为 {tuple(displacement_values.shape)}"
            )

        reference_pos = graph.motion_reference_pos
        # Validates `int(reference_pos.shape[0]) != int(displacement_values.shape[0])` and raises `ValueError` when the input cannot be processed safely.
        if int(reference_pos.shape[0]) != int(displacement_values.shape[0]):
            raise ValueError("GraphProcessor 运动网格位移节点数与参考坐标不一致")
        layout = self._resolve_motion_displacement_layout(
            meta=meta,
            # Completes the multi-line `int` arguments or elements.
            out_channels=int(displacement_values.shape[-1]),
        )
        scales = graph.motion_coordinate_scale
        offsets = graph.motion_coordinate_offset
        # Validates `` and raises `ValueError` when the input cannot be processed safely.
        if int(scales.shape[1]) != len(layout) or int(offsets.shape[1]) != len(layout):
            raise ValueError("GraphProcessor 运动网格坐标转换状态与位移分量数量不一致")

        coords = reference_pos.clone()
        for component_index, (target_index, coordinate_index) in enumerate(layout):
            # Derives the `coords[:, coordinate_index]` from the current values for downstream processing.
            coords[:, coordinate_index] = (
                reference_pos[:, coordinate_index]
                + displacement_values[:, target_index] * scales[:, component_index]
                + offsets[:, component_index]
            # Continues the surrounding statement or collection definition.
            )
        graph.x[:, :3] = coords
        graph.edge_attr = self._build_edge_attr(
            edge_index=graph.edge_index,
            # Completes the multi-line `self._build_edge_attr` arguments or elements.
            node_coords=coords,
        )
        return True

    def supports_partition_training(self) -> bool:
        """图处理器支持 partition 子图训练。"""
        # Returns `True` to the caller for the next stage.
        return True

    def supports_fast_predict_path(self, is_geometry_varying: bool) -> bool:
        return self.mode == 'steady'

    def supports_reference_only_predict_path(self) -> bool:
        """固定几何稳态预测时，GraphProcessor 支持仅依赖 reference 的快路径。"""
        # Reads `task` from the mapping into the task settings with its declared fallback.
        task_cfg = self.config.get("task", {})
        return (
            str(task_cfg.get("mode", self.mode)).lower() == "steady"
            and not bool(task_cfg.get("is_geometry_varying", self.is_geometry_varying))
        # Continues the surrounding statement or collection definition.
        )

    def supports_fixed_geometry_temporal_predict_path(self) -> bool:
        """固定几何下，GraphProcessor 支持时序 autoregressive 预测快路径。"""
        task_cfg = self.config.get("task", {})
        return (
            # Completes the multi-line `str(task_cfg.get('mode', self.mode)).lower` arguments or elements.
            str(task_cfg.get("mode", self.mode)).lower() == "temporal"
            and str(task_cfg.get("temporal_strategy", "direct")).lower() == "autoregressive"
            and not bool(task_cfg.get("is_geometry_varying", self.is_geometry_varying))
        )

    # Defines `requires_fixed_geometry_temporal_reference` for requires fixed geometry temporal reference logic.
    def requires_fixed_geometry_temporal_reference(self) -> bool:
        input_length = int(self.config.get("task", {}).get("input_length", 1))
        return input_length <= 1

    def requires_fixed_geometry_predict_writeback_reference(self) -> bool:
        # Reads `task` from the mapping into the task settings with its declared fallback.
        task_cfg = self.config.get("task", {})
        if (
            str(task_cfg.get("mode", self.mode)).lower() == "temporal"
            and str(task_cfg.get("temporal_strategy", "direct")).lower() == "autoregressive"
            # Completes the multi-line `int` arguments or elements.
            and int(task_cfg.get("input_length", 1)) > 1
        ):
            return False
        return True

    # Defines `build_fast_predict_dataset` for build fast predict dataset logic.
    def build_fast_predict_dataset(
        self,
        samples_csv: str,
        processed_dir: str,
        # Completes the multi-line `collection` arguments or elements.
        config: Dict[str, Any],
        norm_params: Optional[Dict[str, Any]],
    ):
        from ...data.data_pipeline import DataPipeline
        # Imports `build_dataset` from `dataset` for use in this module.
        from ...dataset import build_dataset
        from ...factories.reader_factory import ReaderFactory
        from ...readers.sample_map import SampleMap
        from ...utils.path_utils import resolve_path_from_base

        # Stores the `sample_map` for use by the remaining steps in this flow.
        sample_map = SampleMap(samples_csv)
        sample_ids = sample_map.get_all_samples()
        if not sample_ids:
            raise ValueError("样本表为空，无法进行 Graph 预测")

        # Calls `os.makedirs` to perform the operation required at this stage.
        os.makedirs(processed_dir, exist_ok=True)
        base_dir = os.path.dirname(os.path.abspath(samples_csv))
        reader = ReaderFactory.get_reader(config, flag="predict")

        processed_paths: List[str] = []
        # Initializes the `success_ids` as a collection for values from this processing stage.
        success_ids: List[int] = []

        for sample_id in sample_ids:
            sample_file = sample_map.get_file_path(sample_id)
            if not sample_file:
                # Handles the data or control-flow case represented by this statement.
                continue

            abs_path = resolve_path_from_base(base_dir, sample_file)
            data = reader.read(
                abs_path,
                # Completes the multi-line `reader.read` arguments or elements.
                sample_order=sample_id,
                allow_missing_fields=True,
            )
            processed_data = self.process_input_only(data)
            # Stores the `processed_path` for use by the remaining steps in this flow.
            processed_path = os.path.join(processed_dir, f"sample_{sample_id}.pt")
            self.save_to_pt(processed_data, processed_path)
            processed_paths.append(processed_path)
            success_ids.append(sample_id)

        # Validates `not processed_paths` and raises `ValueError` when the input cannot be processed safely.
        if not processed_paths:
            raise ValueError("Graph 预测输入构建失败，没有可用样本")

        if norm_params:
            norm_type = norm_params.get(
                # Completes the multi-line `norm_params.get` arguments or elements.
                "norm_type",
                config.get("processor", {}).get("norm_type", "standardization"),
            )
            DataPipeline.normalize_data_predict(
                # Completes the multi-line `DataPipeline.normalize_data_predict` arguments or elements.
                processed_paths=processed_paths,
                norm_params=norm_params,
                norm_type=norm_type,
                processor=self,
                # Completes the multi-line `type` arguments or elements.
                processor_type=type(self).__name__,
            )
        else:
            raise ValueError("未从检查点找到归一化参数")

        # Stores the `effective_csv` for use by the remaining steps in this flow.
        effective_csv = samples_csv
        if len(success_ids) < len(sample_ids):
            filtered_csv = os.path.join(processed_dir, "effective_samples.csv")
            effective_csv = self._create_filtered_samples_csv(
                # Completes the multi-line `self._create_filtered_samples_csv` arguments or elements.
                samples_csv,
                filtered_csv,
                success_ids,
            )

        # Stores the dataset for use by the remaining steps in this flow.
        dataset = build_dataset(
            config=config,
            processor_type=type(self).__name__,
            csv_path=effective_csv,
            # Completes the multi-line `build_dataset` arguments or elements.
            data_dir=processed_dir,
            phase="predict",
            processor=self,
        )
        # Returns `dataset, effective_csv` to the caller for the next stage.
        return dataset, effective_csv

    def build_fixed_geometry_temporal_predict_dataset(
        self,
        reader,
        # Continues the surrounding statement or collection definition.
        samples_csv: str,
        processed_dir: str,
        config: Dict[str, Any],
        norm_params: Optional[Dict[str, Any]],
        # Continues the surrounding statement or collection definition.
        reference_file: Optional[str],
    ):
        """构建固定几何时序 autoregressive 预测数据集。"""
        from ...data.data_pipeline import DataPipeline
        from ...data.loaders import FieldSampleLoader
        # Imports `build_dataset` from `dataset` for use in this module.
        from ...dataset import build_dataset
        from ...readers.sample_map import SampleMap
        from ...utils.path_utils import resolve_path_from_base

        if not self.supports_fixed_geometry_temporal_predict_path():
            # Raises `NotImplementedError` to report why this path cannot continue.
            raise NotImplementedError(
                "GraphProcessor 固定几何时序预测快路径仅支持 temporal autoregressive 模式。"
            )

        sample_map = SampleMap(samples_csv)
        # Stores the sample identifiers for use by the remaining steps in this flow.
        sample_ids = sample_map.get_all_samples()
        if not sample_ids:
            raise ValueError("样本表为空，无法构造 Graph 固定几何时序预测数据集")

        os.makedirs(processed_dir, exist_ok=True)
        # Stores the `base_dir` for use by the remaining steps in this flow.
        base_dir = os.path.dirname(os.path.abspath(samples_csv))
        input_length = int(self.config.get("task", {}).get("input_length", 1))
        sample_loader = FieldSampleLoader(
            reader=reader,
            # Completes the multi-line `FieldSampleLoader` arguments or elements.
            mode=self.mode,
            allow_missing_fields=input_length <= 1,
            temporal_history_steps=input_length if input_length > 1 else None,
        )

        # Stores the `reference_data` for use by the remaining steps in this flow.
        reference_data = None

        processed_paths: List[str] = []
        success_ids: List[int] = []

        for sample_id in sample_ids:
            # Stores the `sample_file` for use by the remaining steps in this flow.
            sample_file = sample_map.get_file_path(sample_id)
            if input_length <= 1:
                if reference_data is None:
                    if not reference_file:
                        # Raises `ValueError` to report why this path cannot continue.
                        raise ValueError("Graph 固定几何 K=1 时序预测需要 reference 文件")
                    try:
                        reference_data = reader.read(
                            reference_file,
                            # Completes the multi-line `reader.read` arguments or elements.
                            sample_order=sample_id,
                            allow_missing_fields=True,
                        )
                    except TypeError:
                        # Stores the `reference_data` for use by the remaining steps in this flow.
                        reference_data = reader.read(reference_file, sample_order=sample_id)
                sample_info = dict(sample_map.get_sample_info(sample_id) or {})
                sample_info["sample_order"] = int(sample_id)
                sample_info.setdefault("file_path", sample_file or "")
                # Stores the current data for use by the remaining steps in this flow.
                data = BaseData(
                    data=reference_data.get_data(),
                    task_type=reference_data.get_task_type(),
                    mode="temporal",
                    # Completes the multi-line `reference_data.get_point_or_cell` arguments or elements.
                    point_or_cell=reference_data.get_point_or_cell(),
                    sample_map=sample_info,
                )
                processed_data = self._build_zero_autoregressive_predict_seed_sample(data)
            # Continues the surrounding statement or collection definition.
            else:
                if not sample_file:
                    raise ValueError("K>1 的 Graph autoregressive 预测必须提供样本原始文件")
                abs_path = resolve_path_from_base(base_dir, sample_file)
                # Stores the current data for use by the remaining steps in this flow.
                data = sample_loader.load(abs_path, sample_id)
                processed_data = self._build_autoregressive_predict_seed_sample(data)
            processed_path = os.path.join(processed_dir, f"sample_{sample_id}.pt")
            self.save_to_pt(processed_data, processed_path)
            # Adds the current value to `processed_paths` for later aggregation.
            processed_paths.append(processed_path)
            success_ids.append(sample_id)

        if not processed_paths:
            raise ValueError("Graph 固定几何时序预测输入构建失败，没有可用样本")

        # Checks `norm_params` before entering this conditional branch.
        if norm_params:
            norm_type = norm_params.get(
                "norm_type",
                config.get("processor", {}).get("norm_type", "standardization"),
            # Completes the multi-line `norm_params.get` arguments or elements.
            )
            DataPipeline.normalize_data_with_existing_params(
                processed_paths=processed_paths,
                norm_params=norm_params,
                # Completes the multi-line `DataPipeline.normalize_data_with_existing_params` arguments or elements.
                norm_type=norm_type,
                processor=self,
                normalize_y=True,
            )
        # Continues the surrounding statement or collection definition.
        else:
            raise ValueError("未从检查点找到归一化参数")

        effective_csv = samples_csv
        if len(success_ids) < len(sample_ids):
            # Stores the `filtered_csv` for use by the remaining steps in this flow.
            filtered_csv = os.path.join(processed_dir, "effective_samples.csv")
            effective_csv = self._create_filtered_samples_csv(
                samples_csv,
                filtered_csv,
                # Completes the multi-line `self._create_filtered_samples_csv` arguments or elements.
                success_ids,
            )

        dataset = build_dataset(
            config=config,
            # Completes the multi-line `type` arguments or elements.
            processor_type=type(self).__name__,
            csv_path=effective_csv,
            data_dir=processed_dir,
            phase="predict",
            # Completes the multi-line `build_dataset` arguments or elements.
            processor=self,
        )
        return dataset, effective_csv

    def build_reference_only_predict_dataset(
        # Continues the surrounding statement or collection definition.
        self,
        reader,
        samples_csv: str,
        processed_dir: str,
        # Completes the multi-line `collection` arguments or elements.
        config: Dict[str, Any],
        norm_params: Optional[Dict[str, Any]],
        reference_file: str,
    ):
        """构建仅依赖 reference 和样本参数的 Graph 稳态预测数据集。"""
        # Imports `DataPipeline` from `data.data_pipeline` for use in this module.
        from ...data.data_pipeline import DataPipeline
        from ...dataset import build_dataset
        from ...readers.sample_map import SampleMap

        if not self.supports_reference_only_predict_path():
            # Raises `NotImplementedError` to report why this path cannot continue.
            raise NotImplementedError(
                "GraphProcessor reference-only 预测快路径仅支持 fixed-geometry steady 模式。"
            )

        sample_map = SampleMap(samples_csv)
        # Stores the sample identifiers for use by the remaining steps in this flow.
        sample_ids = sample_map.get_all_samples()
        if not sample_ids:
            raise ValueError("样本表为空，无法构建 Graph reference-only 预测数据集")

        os.makedirs(processed_dir, exist_ok=True)
        # Keeps the operation and its recovery path within the same exception boundary.
        try:
            reference_data = reader.read(
                reference_file,
                sample_order=sample_ids[0],
                # Completes the multi-line `reader.read` arguments or elements.
                allow_missing_fields=True,
            )
        except TypeError:
            reference_data = reader.read(
                # Completes the multi-line `reader.read` arguments or elements.
                reference_file,
                sample_order=sample_ids[0],
            )

        processed_paths: List[str] = []
        # Initializes the `success_ids` as a collection for values from this processing stage.
        success_ids: List[int] = []

        for sample_id in sample_ids:
            sample_info = dict(sample_map.get_sample_info(sample_id) or {})
            sample_info["sample_order"] = int(sample_id)
            # Calls `sample_info.setdefault` to perform the operation required at this stage.
            sample_info.setdefault("file_path", "")
            sample_data = BaseData(
                data=reference_data.get_data(),
                task_type=reference_data.get_task_type(),
                # Completes the multi-line `reference_data.get_mode` arguments or elements.
                mode=reference_data.get_mode(),
                point_or_cell=reference_data.get_point_or_cell(),
                sample_map=sample_info,
            )
            # Stores the `processed_data` for use by the remaining steps in this flow.
            processed_data = self.process_input_only(sample_data)
            processed_path = os.path.join(processed_dir, f"sample_{sample_id}.pt")
            self.save_to_pt(processed_data, processed_path)
            processed_paths.append(processed_path)
            # Adds the current value to `success_ids` for later aggregation.
            success_ids.append(sample_id)

        if not processed_paths:
            raise ValueError("Graph reference-only 预测输入构建失败，没有可用样本")

        if norm_params:
            # Reads `norm_type` from the mapping into the `norm_type` with its declared fallback.
            norm_type = norm_params.get(
                "norm_type",
                config.get("processor", {}).get("norm_type", "standardization"),
            )
            # Calls `DataPipeline.normalize_data_predict` to perform the operation required at this stage.
            DataPipeline.normalize_data_predict(
                processed_paths=processed_paths,
                norm_params=norm_params,
                norm_type=norm_type,
                # Completes the multi-line `DataPipeline.normalize_data_predict` arguments or elements.
                processor=self,
                processor_type=type(self).__name__,
            )
        else:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("未从检查点找到归一化参数")

        effective_csv = samples_csv
        if len(success_ids) < len(sample_ids):
            filtered_csv = os.path.join(processed_dir, "effective_samples.csv")
            # Stores the `effective_csv` for use by the remaining steps in this flow.
            effective_csv = self._create_filtered_samples_csv(
                samples_csv,
                filtered_csv,
                success_ids,
            # Completes the multi-line `self._create_filtered_samples_csv` arguments or elements.
            )

        dataset = build_dataset(
            config=config,
            processor_type=type(self).__name__,
            # Completes the multi-line `build_dataset` arguments or elements.
            csv_path=effective_csv,
            data_dir=processed_dir,
            phase="predict",
            processor=self,
        # Completes the multi-line `build_dataset` arguments or elements.
        )
        return dataset, effective_csv

    def _refresh_topology_edge_attr(self, data: BaseData, topology: Dict[str, Any]) -> Dict[str, Any]:
        """复用结构边索引，并按当前参考构型重建度量边属性。"""
        refreshed = dict(topology)
        # Stores the `data_dict` for use by the remaining steps in this flow.
        data_dict = data.get_data()
        field_location = str(refreshed.get("field_location") or data.point_or_cell).lower()
        if field_location == "point":
            coords = data_dict["PointCoordinates"]
        # Continues the surrounding statement or collection definition.
        else:
            coords = data_dict["CellCoordinates"]
        refreshed["edge_attr"] = self._build_edge_attr(
            edge_index=refreshed["edge_index"],
            # Completes the multi-line `self._build_edge_attr` arguments or elements.
            node_coords=coords,
        )
        return refreshed

    def _cache_graph_structure(self, topology: Dict[str, Any]) -> None:
        """缓存仅与连通关系有关的图结构，不缓存时变度量边属性。"""
        # Stores the `cached_topology` on the instance so other methods can reuse it.
        self.cached_topology = {
            key: value
            for key, value in topology.items()
            if key != "edge_attr"
        # Continues the surrounding statement or collection definition.
        }
        self.first_process_done = True

    def process_input_only(self, data: BaseData) -> Dict:
        if self.is_geometry_varying or not self.first_process_done or self.cached_topology is None:
            # Stores the `topology` for use by the remaining steps in this flow.
            topology = self._build_graph_topology(data)
            if not self.is_geometry_varying:
                self._cache_graph_structure(topology)
        else:
            # Stores the `topology` for use by the remaining steps in this flow.
            topology = self._refresh_topology_edge_attr(data, self.cached_topology)

        x_tensor = self._extract_node_features(data, topology)
        y_tensor = self._build_dummy_target(topology['num_nodes'])
        meta = self._build_processed_sample_meta(data=data, y=y_tensor, time_axis=None)
        # Stores the `meta` for use by the remaining steps in this flow.
        meta = self._with_local_feature_meta(meta, **self._last_local_feature_meta)
        if str(data.get_mode()).lower() == "temporal":
            meta = self._build_temporal_direct_predict_meta(meta)
        else:
            # Stores the `meta` for use by the remaining steps in this flow.
            meta = self._build_steady_input_only_meta(meta)
        return {
            'x': x_tensor,
            'edge_index': topology['edge_index'],
            # Completes the multi-line `collection` arguments or elements.
            'edge_attr': topology['edge_attr'],
            'y': y_tensor,
            'num_nodes': topology['num_nodes'],
            'field_location': topology['field_location'],
            # Completes the multi-line `self._extract_sample_id` arguments or elements.
            'sample_id': self._extract_sample_id(data),
            'meta': meta,
        }

    def _build_dummy_target(self, num_nodes: int) -> torch.Tensor:
        # Stores the `effective_variables` for use by the remaining steps in this flow.
        effective_variables = (
            self.config.get('task', {}).get('effective_variables')
            or self.config.get('task', {}).get('variables', [])
        )
        # Stores the `out_channels` for use by the remaining steps in this flow.
        out_channels = len(effective_variables)
        if out_channels <= 0:
            raise ValueError("Graph 预测阶段无法确定输出通道数")
        return torch.zeros((int(num_nodes), out_channels), dtype=torch.float32)

    # Defines `process_predict_sample` for process predict sample logic.
    def process_predict_sample(
        self,
        data: BaseData,
        context: Optional[Dict[str, Any]] = None,
    # Completes the multi-line `collection` arguments or elements.
    ) -> Dict[str, Any]:
        if self.temporal_strategy == "autoregressive":
            input_length = int(self.config.get("task", {}).get("input_length", 1))
            if input_length <= 1:
                # Returns `self._build_zero_autoregressive_predict_seed_sample(data)` to the caller for the next stage.
                return self._build_zero_autoregressive_predict_seed_sample(data)
            return self._build_autoregressive_predict_seed_sample(data)
        return self.process_input_only(data)

    def _create_filtered_samples_csv(
        # Continues the surrounding statement or collection definition.
        self,
        source_csv: str,
        output_csv: str,
        keep_ids: List[int],
    # Continues the surrounding statement or collection definition.
    ) -> str:
        import csv
        from ...readers.sample_map import SampleMap

        keep_set = set(keep_ids)
        # Stores the `sample_map` for use by the remaining steps in this flow.
        sample_map = SampleMap(source_csv)
        all_ids = [sid for sid in sample_map.get_all_samples() if sid in keep_set]
        if not all_ids:
            raise ValueError("过滤后样本为空，无法生成有效样本表")

        # Stores the `first_info` for use by the remaining steps in this flow.
        first_info = sample_map.get_sample_info(all_ids[0]) or {}
        fieldnames = ['order'] + [k for k in first_info.keys()]

        output_dir = os.path.dirname(output_csv)
        if output_dir:
            # Calls `os.makedirs` to perform the operation required at this stage.
            os.makedirs(output_dir, exist_ok=True)

        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            # Visits each item in `all_ids` and applies the current processing step.
            for sample_id in all_ids:
                info = sample_map.get_sample_info(sample_id) or {}
                row = {'order': sample_id}
                row.update(info)
                # Calls `writer.writerow` to perform the operation required at this stage.
                writer.writerow(row)

        return output_csv

    def save_processing_state(
        self,
        # Continues the surrounding statement or collection definition.
        stage: str,
        processed_dir: str,
        processed_paths: List[str],
    ) -> None:
        # Returns early when `stage != 'train' or not processed_paths` makes the remaining work inapplicable.
        if stage != 'train' or not processed_paths:
            return
        if self.is_geometry_varying and not self.local_feature_schema:
            return
        # Stores the `processor_state_path` for use by the remaining steps in this flow.
        processor_state_path = os.path.join(os.path.dirname(processed_paths[0]), 'processor_state.pt')
        processor_state = self.export_state()
        atomic_torch_save(processor_state, processor_state_path)
        _get_logger().info("处理器状态已保存: %s", processor_state_path)

    # Defines `denormalize_predictions` for denormalize predictions logic.
    def denormalize_predictions(self, predictions, norm_params: Dict = None):
        if norm_params is None or 'y' not in norm_params:
            return predictions

        if isinstance(predictions, dict):
            # Returns `` to the caller for the next stage.
            return {
                key: self.denormalize_predictions(value, norm_params)
                for key, value in predictions.items()
            }

        # Selects the required item from `norm_params` as the `method`.
        method = norm_params['norm_type']
        y_params = norm_params['y']
        denorm = np.array(predictions, copy=True)
        if denorm.ndim >= 2:
            # Selects the required item from `denorm.shape` as the `m`.
            m = denorm.shape[-1]
            for i in range(m):
                if method == 'standardization' and i < len(y_params.get('mean', [])):
                    mean = y_params['mean'][i]
                    # Selects the required item from `y_params['std']` as the `std`.
                    std = y_params['std'][i]
                    denorm[..., i] = denorm[..., i] * (std + 1e-12) + mean
                elif method == 'min_max' and i < len(y_params.get('min', [])):
                    min_val = y_params['min'][i]
                    # Selects the required item from `y_params['max']` as the `max_val`.
                    max_val = y_params['max'][i]
                    denorm[..., i] = denorm[..., i] * (max_val - min_val + 1e-12) + min_val
        return denorm

    def normalize_output_tensor(
        # Continues the surrounding statement or collection definition.
        self,
        outputs: Any,
        norm_params: Optional[Dict[str, Any]],
        inputs: Any = None,
    # Continues the surrounding statement or collection definition.
    ) -> Any:
        if norm_params is None or 'y' not in norm_params or not isinstance(outputs, torch.Tensor):
            return outputs

        normalized = outputs.float().clone()
        # Stores the `original_shape` for use by the remaining steps in this flow.
        original_shape = normalized.shape
        if normalized.ndim == 1:
            normalized = normalized.unsqueeze(0)
        elif normalized.ndim >= 3:
            # Stores the `normalized` for use by the remaining steps in this flow.
            normalized = normalized.reshape(-1, original_shape[-1])

        normalized = self._apply_channel_norm(
            normalized,
            norm_params['y'],
            # Completes the multi-line `self._apply_channel_norm` arguments or elements.
            norm_params['norm_type'],
        )

        if len(original_shape) == 1:
            return normalized.squeeze(0)
        # Returns early when `len(original_shape) >= 3` makes the remaining work inapplicable.
        if len(original_shape) >= 3:
            return normalized.reshape(original_shape)
        return normalized

    def denormalize_output_tensor(
        # Continues the surrounding statement or collection definition.
        self,
        outputs: Any,
        norm_params: Optional[Dict[str, Any]],
        inputs: Any = None,
    # Continues the surrounding statement or collection definition.
    ) -> Any:
        if norm_params is None or 'y' not in norm_params or not isinstance(outputs, torch.Tensor):
            return outputs

        denormalized = outputs.float().clone()
        # Stores the `original_shape` for use by the remaining steps in this flow.
        original_shape = denormalized.shape
        if denormalized.ndim == 1:
            denormalized = denormalized.unsqueeze(0)
        elif denormalized.ndim >= 3:
            # Stores the `denormalized` for use by the remaining steps in this flow.
            denormalized = denormalized.reshape(-1, original_shape[-1])

        denormalized = self._apply_channel_denorm(
            denormalized,
            norm_params['y'],
            # Completes the multi-line `self._apply_channel_denorm` arguments or elements.
            norm_params['norm_type'],
        )

        if len(original_shape) == 1:
            return denormalized.squeeze(0)
        # Returns early when `len(original_shape) >= 3` makes the remaining work inapplicable.
        if len(original_shape) >= 3:
            return denormalized.reshape(original_shape)
        return denormalized

    def _reshape_targets_for_weighting(self, values: torch.Tensor) -> torch.Tensor:
        # Checks `values.ndim == 2` before entering this conditional branch.
        if values.ndim == 2:
            task_cfg = self.config.get("task", {})
            if (
                str(task_cfg.get("mode", self.mode)).lower() == "temporal"
                # Completes the multi-line `str(task_cfg.get('temporal_strategy', 'direct')).lower` arguments or elements.
                and str(task_cfg.get("temporal_strategy", "direct")).lower() == "direct"
            ):
                field_channels = self._get_output_field_channels()
                folded_channels = int(values.shape[-1])
                # Validates `folded_channels % field_channels != 0` and raises `ValueError` when the input cannot be processed safely.
                if folded_channels % field_channels != 0:
                    raise ValueError(
                        f"GraphProcessor 无法还原 temporal direct 通道布局："
                        f" folded_channels={folded_channels}, field_channels={field_channels}"
                    # Completes the multi-line `ValueError` arguments or elements.
                    )
                time_steps = folded_channels // field_channels
                return values.reshape(values.shape[0], time_steps, field_channels)
            return values.unsqueeze(1)

        # Returns early when `values.ndim == 3` makes the remaining work inapplicable.
        if values.ndim == 3:
            return values.permute(1, 0, 2)

        raise ValueError(
            f"GraphProcessor 关键区域加权期望目标张量维度为 2 或 3，当前为 {values.ndim}"
        # Completes the multi-line `ValueError` arguments or elements.
        )

    @staticmethod
    def _compute_graph_gradient_proxy(
        node_values: torch.Tensor,
        # Continues the surrounding statement or collection definition.
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        squeeze_back = False
        # Checks `node_values.ndim == 1` before entering this conditional branch.
        if node_values.ndim == 1:
            node_values = node_values.unsqueeze(1)
            squeeze_back = True
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.shape[1] == 0:
            # Stores the `empty` for use by the remaining steps in this flow.
            empty = torch.zeros((num_nodes, node_values.shape[1]), dtype=node_values.dtype, device=node_values.device)
            return empty.squeeze(1) if squeeze_back else empty

        src = edge_index[0].long()
        dst = edge_index[1].long()
        # Stores the `diffs` for use by the remaining steps in this flow.
        diffs = torch.abs(node_values[src] - node_values[dst])

        gradient = torch.zeros((num_nodes, node_values.shape[1]), dtype=node_values.dtype, device=node_values.device)
        degree = torch.zeros((num_nodes, 1), dtype=node_values.dtype, device=node_values.device)
        ones = torch.ones((diffs.shape[0], 1), dtype=node_values.dtype, device=node_values.device)

        # Calls `gradient.index_add_` to perform the operation required at this stage.
        gradient.index_add_(0, src, diffs)
        gradient.index_add_(0, dst, diffs)
        degree.index_add_(0, src, ones)
        degree.index_add_(0, dst, ones)
        # Derives the `gradient` from the current values for downstream processing.
        gradient = gradient / torch.clamp(degree, min=1.0)
        return gradient.squeeze(1) if squeeze_back else gradient

    @staticmethod
    def _smooth_graph_weight_map(
        # Continues the surrounding statement or collection definition.
        node_weight: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        squeeze_back = False
        # Checks `node_weight.ndim == 1` before entering this conditional branch.
        if node_weight.ndim == 1:
            node_weight = node_weight.unsqueeze(1)
            squeeze_back = True
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.shape[1] == 0:
            # Returns `node_weight.squeeze(1) if squeeze_back else node_weight` to the caller for the next stage.
            return node_weight.squeeze(1) if squeeze_back else node_weight

        src = edge_index[0].long()
        dst = edge_index[1].long()
        neighbor_sum = torch.zeros_like(node_weight)
        # Stores the `degree` for use by the remaining steps in this flow.
        degree = torch.zeros((node_weight.shape[0], 1), dtype=node_weight.dtype, device=node_weight.device)
        ones = torch.ones((edge_index.shape[1], 1), dtype=node_weight.dtype, device=node_weight.device)

        neighbor_sum.index_add_(0, src, node_weight[dst])
        neighbor_sum.index_add_(0, dst, node_weight[src])
        # Calls `degree.index_add_` to perform the operation required at this stage.
        degree.index_add_(0, src, ones)
        degree.index_add_(0, dst, ones)

        neighbor_mean = neighbor_sum / torch.clamp(degree, min=1.0)
        smoothed = 0.5 * node_weight + 0.5 * neighbor_mean
        # Returns `smoothed.squeeze(1) if squeeze_back else smoothed` to the caller for the next stage.
        return smoothed.squeeze(1) if squeeze_back else smoothed

    @staticmethod
    def _resolve_graph_node_slices(
        inputs: Any,
        # Continues the surrounding statement or collection definition.
        total_nodes: int,
        batch: torch.Tensor,
    ) -> List[Tuple[int, int, int]]:
        ptr = getattr(inputs, "ptr", None)
        # Checks `isinstance(ptr, torch.Tensor)` before entering this conditional branch.
        if isinstance(ptr, torch.Tensor):
            ptr = ptr.to(device=batch.device, dtype=torch.long)
            if ptr.ndim == 1 and ptr.numel() >= 2 and int(ptr[-1].item()) == int(total_nodes):
                slices: List[Tuple[int, int, int]] = []
                # Visits each item in `range(int(ptr.numel()) - 1)` and applies the current processing step.
                for graph_id in range(int(ptr.numel()) - 1):
                    start = int(ptr[graph_id].item())
                    end = int(ptr[graph_id + 1].item())
                    if end > start:
                        # Adds the current value to `slices` for later aggregation.
                        slices.append((graph_id, start, end))
                if slices:
                    return slices

        unique_graph_ids = torch.unique(batch, sorted=True)
        # Initializes the `slices` as a collection for values from this processing stage.
        slices = []
        for graph_id in unique_graph_ids.tolist():
            node_indices = torch.nonzero(batch == int(graph_id), as_tuple=False).reshape(-1)
            if node_indices.numel() == 0:
                # Handles the data or control-flow case represented by this statement.
                continue
            slices.append(
                (
                    int(graph_id),
                    # Completes the multi-line `int` arguments or elements.
                    int(node_indices[0].item()),
                    int(node_indices[-1].item()) + 1,
                )
            )
        # Returns `slices` to the caller for the next stage.
        return slices

    def build_loss_weight(
        self,
        inputs: Any,
        # Continues the surrounding statement or collection definition.
        targets: Any,
        outputs: Any = None,
    ) -> Optional[torch.Tensor]:
        if not isinstance(targets, torch.Tensor):
            # Raises `TypeError` to report why this path cannot continue.
            raise TypeError("GraphProcessor 关键区域加权需要张量形式的 targets")
        if not hasattr(inputs, "edge_index"):
            raise TypeError("GraphProcessor 关键区域加权需要图结构输入")

        reshaped_targets = self._reshape_targets_for_weighting(targets.float())
        # Stores the `(total_nodes, time_steps, variable_count)` for use by the remaining steps in this flow.
        total_nodes, time_steps, variable_count = reshaped_targets.shape

        batch = getattr(inputs, "batch", None)
        if batch is None:
            batch = torch.zeros((total_nodes,), dtype=torch.long, device=reshaped_targets.device)
        # Continues the surrounding statement or collection definition.
        else:
            batch = batch.to(device=reshaped_targets.device, dtype=torch.long)

        edge_index = inputs.edge_index.to(device=reshaped_targets.device, dtype=torch.long)
        edge_graph_ids = batch[edge_index[0]] if edge_index.numel() > 0 else torch.zeros((0,), dtype=torch.long, device=reshaped_targets.device)
        # Stores the `edge_intra_graph_mask` for use by the remaining steps in this flow.
        edge_intra_graph_mask = (
            edge_graph_ids == batch[edge_index[1]]
            if edge_index.numel() > 0
            else torch.zeros((0,), dtype=torch.bool, device=reshaped_targets.device)
        # Continues the surrounding statement or collection definition.
        )
        shared_weight = torch.ones(
            (total_nodes, time_steps),
            dtype=reshaped_targets.dtype,
            # Completes the multi-line `torch.ones` arguments or elements.
            device=reshaped_targets.device,
        )

        graph_slices = self._resolve_graph_node_slices(inputs, total_nodes, batch)
        for graph_id, start, end in graph_slices:
            # Converts the `local_node_count` to `int` so later checks use a consistent type.
            local_node_count = int(end - start)
            if local_node_count <= 0:
                continue

            local_edge_mask = edge_intra_graph_mask & (edge_graph_ids == int(graph_id))
            # Selects the required item from `edge_index` as the `local_edge_index`.
            local_edge_index = edge_index[:, local_edge_mask]
            if local_edge_index.numel() > 0:
                local_edge_index = local_edge_index - int(start)

            local_targets = reshaped_targets[start:end]
            # Stores the `local_feature_values` for use by the remaining steps in this flow.
            local_feature_values = local_targets.reshape(local_node_count, time_steps * variable_count)
            value_score = self._build_quantile_score(
                local_feature_values.transpose(0, 1)
            ).reshape(time_steps, variable_count, local_node_count)

            # Stores the `gradient_proxy` for use by the remaining steps in this flow.
            gradient_proxy = self._compute_graph_gradient_proxy(
                node_values=local_feature_values,
                edge_index=local_edge_index,
                num_nodes=local_node_count,
            # Completes the multi-line `self._compute_graph_gradient_proxy` arguments or elements.
            )
            gradient_score = self._build_quantile_score(
                gradient_proxy.transpose(0, 1)
            ).reshape(time_steps, variable_count, local_node_count)

            # Stores the `local_value_map` for use by the remaining steps in this flow.
            local_value_map = value_score.amax(dim=1).transpose(0, 1)
            local_gradient_map = gradient_score.amax(dim=1).transpose(0, 1)
            local_shared_weight = 1.0 + 2.0 * local_value_map + 2.0 * local_gradient_map
            local_shared_weight = self._smooth_graph_weight_map(
                # Completes the multi-line `self._smooth_graph_weight_map` arguments or elements.
                local_shared_weight,
                local_edge_index,
            )
            local_shared_weight = self._normalize_shared_weight_map(
                # Completes the multi-line `local_shared_weight.transpose` arguments or elements.
                local_shared_weight.transpose(0, 1)
            ).transpose(0, 1)
            shared_weight[start:end] = local_shared_weight

        if reshaped_targets.shape[1] == 1:
            # Returns `shared_weight[:, :1]` to the caller for the next stage.
            return shared_weight[:, :1]

        return shared_weight.unsqueeze(-1).expand(-1, -1, variable_count).reshape(total_nodes, -1)

    def prepare_eval_payload(
        self,
        # Continues the surrounding statement or collection definition.
        inputs: Any,
        outputs: Any,
        targets: Any,
    ) -> Tuple[Any, Any]:
        # Returns early when `` makes the remaining work inapplicable.
        if (
            self.mode == 'temporal'
            and self.config.get('task', {}).get('temporal_strategy', 'direct') == 'direct'
        ):
            # Returns `` to the caller for the next stage.
            return (
                self._restore_temporal_direct_output(outputs),
                self._restore_temporal_direct_output(targets),
            )
        # Returns `` to the caller for the next stage.
        return super().prepare_eval_payload(inputs=inputs, outputs=outputs, targets=targets)

    def collect_prediction_batch(
        self,
        inputs: Any,
        # Continues the surrounding statement or collection definition.
        outputs: Any,
    ) -> Dict[int, np.ndarray]:
        return self._split_graph_batch_outputs(inputs, outputs)

    def _split_graph_batch_outputs(
        # Continues the surrounding statement or collection definition.
        self,
        inputs: Any,
        outputs: Any,
    ) -> Dict[int, np.ndarray]:
        # Stores the `outputs_np` for use by the remaining steps in this flow.
        outputs_np = self._to_numpy(outputs)
        if (
            self.mode == 'temporal'
            and self.config.get('task', {}).get('temporal_strategy', 'direct') == 'direct'
        # Continues the surrounding statement or collection definition.
        ):
            outputs_np = np.asarray(self._restore_temporal_direct_output(outputs_np))
        batch_assignment = getattr(inputs, "batch", None)
        sample_ids = getattr(inputs, "sample_id", None)

        # Checks `batch_assignment is None` before entering this conditional branch.
        if batch_assignment is None:
            sample_id = 0
            if sample_ids is not None and len(sample_ids) > 0:
                sample_id = int(sample_ids[0].item())
            # Stores the `single_output` for use by the remaining steps in this flow.
            single_output = np.asarray(outputs_np)
            if (
                single_output.ndim == 3
                and hasattr(inputs, "num_nodes")
                # Completes the multi-line `int` arguments or elements.
                and int(single_output.shape[0]) == int(inputs.num_nodes)
            ):
                single_output = np.transpose(single_output, (1, 0, 2))
            return {sample_id: single_output}

        # Stores the `batch_assignment` for use by the remaining steps in this flow.
        batch_assignment = batch_assignment.detach().cpu()
        num_graphs = getattr(inputs, "num_graphs", int(batch_assignment.max().item()) + 1)
        results: Dict[int, np.ndarray] = {}

        for graph_idx in range(num_graphs):
            # Stores the `mask` for use by the remaining steps in this flow.
            mask = batch_assignment == graph_idx
            sample_id = graph_idx
            if sample_ids is not None and len(sample_ids) > graph_idx:
                sample_id = int(sample_ids[graph_idx].item())
            # Stores the `graph_output` for use by the remaining steps in this flow.
            graph_output = np.asarray(outputs_np)
            if graph_output.ndim == 3 and graph_output.shape[0] == batch_assignment.shape[0]:
                results[sample_id] = np.transpose(graph_output[mask.numpy()], (1, 0, 2))
            elif graph_output.ndim == 3 and graph_output.shape[1] == batch_assignment.shape[0]:
                # Selects the required item from `graph_output` as the `results[sample_id]`.
                results[sample_id] = graph_output[:, mask.numpy(), :]
            else:
                results[sample_id] = graph_output[mask.numpy()]

        return results

    # Defines `collect_eval_batch` for collect eval batch logic.
    def collect_eval_batch(
        self,
        inputs: Any,
        outputs: Any,
        # Continues the surrounding statement or collection definition.
        targets: Any,
    ) -> Dict[str, Any]:
        payload = super().collect_eval_batch(inputs, outputs, targets)
        payload["sample_predictions"] = self._split_graph_batch_outputs(inputs, outputs)
        # Stores the `payload['sample_targets']` for use by the remaining steps in this flow.
        payload["sample_targets"] = self._split_graph_batch_outputs(inputs, targets)
        return payload

    def merge_prediction_batches(
        self,
        # Completes the multi-line `collection` arguments or elements.
        batch_outputs: List[Dict[int, np.ndarray]],
    ) -> Dict[int, np.ndarray]:
        merged: Dict[int, np.ndarray] = {}
        for batch_result in batch_outputs:
            # Adds the current value to `merged` for later aggregation.
            merged.update(batch_result)
        return merged

    def merge_eval_batches(
        self,
        # Completes the multi-line `collection` arguments or elements.
        batch_payloads: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        raw_predictions = [np.asarray(payload['predictions']) for payload in batch_payloads]
        # Builds the `raw_targets` using the comprehension's filter and mapping rules.
        raw_targets = [np.asarray(payload['targets']) for payload in batch_payloads]
        norm_params = self._resolve_eval_norm_params(config)
        denormalized_predictions = [
            np.asarray(self.denormalize_predictions(pred_arr, norm_params))
            # Continues the surrounding statement or collection definition.
            for pred_arr in raw_predictions
        ]
        denormalized_targets = [
            np.asarray(self.denormalize_predictions(tgt_arr, norm_params))
            # Continues the surrounding statement or collection definition.
            for tgt_arr in raw_targets
        ]

        var_names = (
            config.get('task', {}).get('effective_variables')
            # Completes the multi-line `config.get('task', {}).get` arguments or elements.
            or config.get('task', {}).get('variables', [])
        )
        scatter_by_var: Dict[str, Dict[str, np.ndarray]] = {}
        predictions_by_var: Dict[str, List[np.ndarray]] = {}
        # Initializes the `targets_by_var` as a collection for values from this processing stage.
        targets_by_var: Dict[str, List[np.ndarray]] = {}
        fallback_pred_parts: List[np.ndarray] = []
        fallback_tgt_parts: List[np.ndarray] = []

        for payload, pred_arr, tgt_arr in zip(
            # Completes the multi-line `zip` arguments or elements.
            batch_payloads,
            denormalized_predictions,
            denormalized_targets,
        ):
            # Checks `pred_arr.shape != tgt_arr.shape` before entering this conditional branch.
            if pred_arr.shape != tgt_arr.shape:
                fallback_pred_parts.append(pred_arr.reshape(-1))
                fallback_tgt_parts.append(tgt_arr.reshape(-1))
                continue

            # Reads `metric_mask` from the mapping into the `metric_mask`.
            metric_mask = payload.get("metric_mask")
            if metric_mask is not None:
                metric_mask = np.asarray(metric_mask, dtype=bool)
                if metric_mask.shape != pred_arr.shape:
                    # Raises `ValueError` to report why this path cannot continue.
                    raise ValueError("评估掩码与图预测形状不一致")

            if pred_arr.ndim >= 2:
                num_vars = pred_arr.shape[-1]
                names = var_names if len(var_names) == num_vars else [f'field_{i}' for i in range(num_vars)]
                # Visits each item in `range(num_vars)` and applies the current processing step.
                for i in range(num_vars):
                    name = names[i]
                    pred_values = pred_arr[..., i]
                    target_values = tgt_arr[..., i]
                    # Checks `metric_mask is not None` before entering this conditional branch.
                    if metric_mask is not None:
                        variable_mask = metric_mask[..., i]
                        pred_values = pred_values[variable_mask]
                        target_values = target_values[variable_mask]
                    # Adds the current value to `predictions_by_var.setdefault(name, [])` for later aggregation.
                    predictions_by_var.setdefault(name, []).append(pred_values.reshape(-1))
                    targets_by_var.setdefault(name, []).append(target_values.reshape(-1))
            else:
                if metric_mask is not None:
                    # Adds the current value to `fallback_pred_parts` for later aggregation.
                    fallback_pred_parts.append(pred_arr[metric_mask].reshape(-1))
                    fallback_tgt_parts.append(tgt_arr[metric_mask].reshape(-1))
                else:
                    fallback_pred_parts.append(pred_arr.reshape(-1))
                    # Adds the current value to `fallback_tgt_parts` for later aggregation.
                    fallback_tgt_parts.append(tgt_arr.reshape(-1))

        overall_pred_parts: List[np.ndarray] = []
        overall_tgt_parts: List[np.ndarray] = []
        for var_name, pred_parts in predictions_by_var.items():
            # Stores the `merged_pred` for use by the remaining steps in this flow.
            merged_pred = np.concatenate(pred_parts, axis=0)
            merged_tgt = np.concatenate(targets_by_var[var_name], axis=0)
            scatter_by_var[var_name] = {
                'predictions': merged_pred,
                # Completes the multi-line `collection` arguments or elements.
                'targets': merged_tgt,
            }
            overall_pred_parts.append(merged_pred)
            overall_tgt_parts.append(merged_tgt)

        # Checks `fallback_pred_parts` before entering this conditional branch.
        if fallback_pred_parts:
            overall_pred_parts.append(np.concatenate(fallback_pred_parts, axis=0))
            overall_tgt_parts.append(np.concatenate(fallback_tgt_parts, axis=0))

        overall_predictions = (
            # Completes the multi-line `np.concatenate` arguments or elements.
            np.concatenate(overall_pred_parts, axis=0)
            if overall_pred_parts
            else np.array([])
        )
        # Stores the `overall_targets` for use by the remaining steps in this flow.
        overall_targets = (
            np.concatenate(overall_tgt_parts, axis=0)
            if overall_tgt_parts
            else np.array([])
        # Continues the surrounding statement or collection definition.
        )

        sample_predictions = self._denormalize_sample_output_map(
            self._merge_sample_output_maps(
                batch_payloads=batch_payloads,
                # Completes the multi-line `self._merge_sample_output_maps` arguments or elements.
                payload_key='sample_predictions',
                fallback_key='predictions',
            ),
            config,
        # Completes the multi-line `self._denormalize_sample_output_map` arguments or elements.
        )
        sample_targets = self._denormalize_sample_output_map(
            self._merge_sample_output_maps(
                batch_payloads=batch_payloads,
                # Completes the multi-line `self._merge_sample_output_maps` arguments or elements.
                payload_key='sample_targets',
                fallback_key='targets',
            ),
            config,
        # Completes the multi-line `self._denormalize_sample_output_map` arguments or elements.
        )

        return {
            'raw_predictions': raw_predictions,
            'raw_targets': raw_targets,
            # Completes the multi-line `collection` arguments or elements.
            'denormalized_predictions': denormalized_predictions,
            'denormalized_targets': denormalized_targets,
            'overall_predictions': overall_predictions,
            'overall_targets': overall_targets,
            # Completes the multi-line `collection` arguments or elements.
            'scatter_by_var': scatter_by_var,
            'sample_predictions': sample_predictions,
            'sample_targets': sample_targets,
        }

    # 收集参考构型及全部训练时刻的物理坐标范围。
    def _collect_geometry_values_for_normalization(
        self,
        graph: Any,
        y: Optional[torch.Tensor],
        # Completes the multi-line `collection` arguments or elements.
        meta: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        """收集参考构型及全部训练时刻的物理坐标范围。"""
        if not hasattr(graph, "x"):
            raise ValueError("GraphProcessor 样本缺少 graph.x")
        # Stores the `reference` for use by the remaining steps in this flow.
        reference = graph.x[:, :3].float()
        if not self._has_explicit_motion_graph_contract(meta) or y is None or y.ndim != 3:
            return reference

        layout = self._resolve_motion_displacement_layout(
            # Completes the multi-line `self._resolve_motion_displacement_layout` arguments or elements.
            meta=meta,
            out_channels=int(y.shape[-1]),
        )
        if int(y.shape[1]) != int(reference.shape[0]):
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("GraphProcessor 运动网格目标节点数与参考坐标不一致")

        temporal_coords = reference.unsqueeze(0).expand(int(y.shape[0]), -1, -1).clone()
        for target_index, coordinate_index in layout:
            temporal_coords[:, :, coordinate_index] = (
                # Completes the multi-line `reference[:, coordinate_index].unsqueeze` arguments or elements.
                reference[:, coordinate_index].unsqueeze(0)
                + y[:, :, target_index].float()
            )
        return torch.cat([reference, temporal_coords.reshape(-1, 3)], dim=0)

    # Defines `compute_norm_params` for compute norm params logic.
    def compute_norm_params(
        self,
        train_samples: List[Dict],
        norm_type: str,
        # Continues the surrounding statement or collection definition.
        save_path: str = None,
    ) -> Dict:
        if not train_samples:
            raise ValueError("GraphProcessor 训练样本为空，无法计算归一化参数")

        # Initializes the `coord_values` as a collection for values from this processing stage.
        coord_values = []
        param_values = []
        y_values = []

        for sample in train_samples:
            # Selects the required item from `sample` as the `graph`.
            graph = sample['x']
            y = sample.get('y')
            if not hasattr(graph, 'x'):
                raise ValueError("GraphProcessor 样本缺少 graph.x")

            # Stores the `node_features` for use by the remaining steps in this flow.
            node_features = graph.x.float()
            if node_features.ndim != 2 or node_features.shape[1] < 3:
                raise ValueError(f"GraphProcessor 的节点特征应为(num_nodes, 3+c)，当前为{tuple(node_features.shape)}")

            coord_values.append(
                # Completes the multi-line `self._collect_geometry_values_for_normalization` arguments or elements.
                self._collect_geometry_values_for_normalization(
                    graph=graph,
                    y=y if isinstance(y, torch.Tensor) else None,
                    meta=sample.get("meta", {}),
                # Completes the multi-line `self._collect_geometry_values_for_normalization` arguments or elements.
                )
            )
            categorical_indices = {
                int(index)
                # Completes the multi-line `sample.get('meta', {}).get` arguments or elements.
                for index in sample.get("meta", {}).get("categorical_bc_indices", [])
            }
            feature_indices = [
                index
                # Completes the multi-line `range` arguments or elements.
                for index in range(3, int(node_features.shape[1]))
                if index not in categorical_indices
            ]
            if feature_indices:
                # Adds the current value to `param_values` for later aggregation.
                param_values.append(node_features[:, feature_indices])

            if y is not None:
                y = y.float()
                if y.ndim not in (2, 3):
                    # Raises `ValueError` to report why this path cannot continue.
                    raise ValueError(f"GraphProcessor 的 y 维度应为2或3，当前为{y.ndim}")
                y_values.append(y.reshape(-1, y.shape[-1]))

        norm_params = {
            'processor_type': 'GraphProcessor',
            # Completes the multi-line `collection` arguments or elements.
            'norm_type': norm_type,
            'geometry': self._compute_geometry_params(torch.cat(coord_values, dim=0)),
            'x': {'param_offset': 3},
            'y': {},
            # Completes the multi-line `collection` arguments or elements.
            'meta': {
                'coord_channels': [0, 1, 2],
                'edge_attr_source': 'recomputed_from_normalized_coords',
            },
        # Completes the multi-line `collection` arguments or elements.
        }

        if param_values:
            norm_params['x'].update(self._compute_channel_stats(torch.cat(param_values, dim=0)))
            first_graph = train_samples[0]['x']
            # Builds the `first_categorical_indices` using the comprehension's filter and mapping rules.
            first_categorical_indices = {
                int(index)
                for index in train_samples[0].get("meta", {}).get("categorical_bc_indices", [])
            }
            # Builds the `norm_params['x']['feature_indices']` using the comprehension's filter and mapping rules.
            norm_params['x']['feature_indices'] = [
                index
                for index in range(3, int(first_graph.x.shape[1]))
                if index not in first_categorical_indices
            # Continues the surrounding statement or collection definition.
            ]
            norm_params['x']['categorical_bc_indices'] = sorted(first_categorical_indices)
            norm_params['x']['continuous_bc_indices'] = [
                int(index)
                # Completes the multi-line `train_samples[0].get('meta', {}).get` arguments or elements.
                for index in train_samples[0].get("meta", {}).get("continuous_bc_indices", [])
            ]

        if y_values:
            merged_y = torch.cat(y_values, dim=0)
            # Stores the `norm_params['y']` for use by the remaining steps in this flow.
            norm_params['y'] = self._compute_channel_stats(merged_y)
            if self.uses_autoregressive_increment():
                delta_values = []
                for sample in train_samples:
                    # Reads `y` from the mapping into the `y`.
                    y = sample.get("y")
                    if isinstance(y, torch.Tensor) and y.ndim == 3 and y.shape[0] >= 2:
                        delta_values.append((y[1:] - y[:-1]).reshape(-1, y.shape[-1]))
                if not delta_values:
                    # Raises `ValueError` to report why this path cannot continue.
                    raise ValueError("增量自回归需要至少一条含两个时间步的训练序列")
                norm_params["autoregressive_delta"] = self._finalize_delta_stats(
                    self._compute_channel_stats(torch.cat(delta_values, dim=0))
                )

        # Returns `norm_params` to the caller for the next stage.
        return norm_params

    def compute_norm_params_from_paths(
        self,
        train_processed_paths: List[str],
        # Continues the surrounding statement or collection definition.
        norm_type: str,
        save_path: str = None,
    ) -> Dict:
        if not train_processed_paths:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("GraphProcessor training sample paths are empty")

        geometry_stats = None
        param_stats = None
        y_stats = None

        # Visits each item in `train_processed_paths` and applies the current processing step.
        for processed_path in train_processed_paths:
            sample = load_torch(processed_path, weights_only=False)
            graph = sample["x"]
            y = sample.get("y")
            # Validates `not hasattr(graph, "x")` and raises `ValueError` when the input cannot be processed safely.
            if not hasattr(graph, "x"):
                raise ValueError("GraphProcessor 鏍锋湰缂哄皯 graph.x")

            node_features = graph.x.float()
            if node_features.ndim != 2 or node_features.shape[1] < 3:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(
                    f"GraphProcessor 鐨勮妭鐐圭壒寰佸簲涓?num_nodes, 3+c)锛屽綋鍓嶄负{tuple(node_features.shape)}"
                )

            geometry_stats = self._update_running_geometry_stats(
                # Completes the multi-line `self._update_running_geometry_stats` arguments or elements.
                geometry_stats,
                self._collect_geometry_values_for_normalization(
                    graph=graph,
                    y=y if isinstance(y, torch.Tensor) else None,
                    # Completes the multi-line `sample.get` arguments or elements.
                    meta=sample.get("meta", {}),
                ),
            )
            categorical_indices = {
                # Completes the multi-line `int` arguments or elements.
                int(index)
                for index in sample.get("meta", {}).get("categorical_bc_indices", [])
            }
            feature_indices = [
                # Continues the surrounding statement or collection definition.
                index
                for index in range(3, int(node_features.shape[1]))
                if index not in categorical_indices
            ]
            # Checks `feature_indices` before entering this conditional branch.
            if feature_indices:
                param_stats = self._update_running_channel_stats(
                    param_stats,
                    node_features[:, feature_indices],
                # Completes the multi-line `self._update_running_channel_stats` arguments or elements.
                )

            if y is not None:
                y = y.float()
                if y.ndim not in (2, 3):
                    # Raises `ValueError` to report why this path cannot continue.
                    raise ValueError(f"GraphProcessor 鐨?y 缁村害搴斾负2鎴?锛屽綋鍓嶄负{y.ndim}")
                y_stats = self._update_running_channel_stats(
                    y_stats,
                    y.reshape(-1, y.shape[-1]),
                # Completes the multi-line `self._update_running_channel_stats` arguments or elements.
                )

        norm_params = {
            "processor_type": "GraphProcessor",
            "norm_type": norm_type,
            # Completes the multi-line `self._finalize_running_geometry_stats` arguments or elements.
            "geometry": self._finalize_running_geometry_stats(geometry_stats),
            "x": {"param_offset": 3},
            "y": {},
            "meta": {
                # Completes the multi-line `collection` arguments or elements.
                "coord_channels": [0, 1, 2],
                "edge_attr_source": "recomputed_from_normalized_coords",
            },
        }

        # Stores the `param_summary` for use by the remaining steps in this flow.
        param_summary = self._finalize_running_channel_stats(param_stats)
        if param_summary:
            norm_params["x"].update(param_summary)
            first_sample = load_torch(train_processed_paths[0], weights_only=False)
            # Selects the required item from `first_sample` as the `first_graph`.
            first_graph = first_sample["x"]
            first_categorical_indices = {
                int(index)
                for index in first_sample.get("meta", {}).get("categorical_bc_indices", [])
            # Continues the surrounding statement or collection definition.
            }
            norm_params["x"]["feature_indices"] = [
                index
                for index in range(3, int(first_graph.x.shape[1]))
                # Continues the surrounding statement or collection definition.
                if index not in first_categorical_indices
            ]
            norm_params["x"]["categorical_bc_indices"] = sorted(first_categorical_indices)
            norm_params["x"]["continuous_bc_indices"] = [
                # Completes the multi-line `int` arguments or elements.
                int(index)
                for index in first_sample.get("meta", {}).get("continuous_bc_indices", [])
            ]

        y_summary = self._finalize_running_channel_stats(y_stats)
        # Checks `y_summary` before entering this conditional branch.
        if y_summary:
            norm_params["y"] = y_summary

        if self.uses_autoregressive_increment():
            delta_stats = None
            # Visits each item in `train_processed_paths` and applies the current processing step.
            for processed_path in train_processed_paths:
                y = load_torch(processed_path, weights_only=False).get("y")
                if not isinstance(y, torch.Tensor) or y.ndim != 3 or y.shape[0] < 2:
                    continue
                # Stores the `delta_stats` for use by the remaining steps in this flow.
                delta_stats = self._update_running_channel_stats(
                    delta_stats,
                    (y[1:] - y[:-1]).reshape(-1, y.shape[-1]),
                )
            # Stores the `delta_summary` for use by the remaining steps in this flow.
            delta_summary = self._finalize_running_channel_stats(delta_stats)
            if not delta_summary:
                raise ValueError("增量自回归需要至少一条含两个时间步的训练序列")
            norm_params["autoregressive_delta"] = self._finalize_delta_stats(delta_summary)

        # Returns `norm_params` to the caller for the next stage.
        return norm_params

    def normalize_processed_sample(
        self,
        sample: Dict,
        # Continues the surrounding statement or collection definition.
        norm_params: Dict,
        norm_type: str,
        normalize_y: bool = True,
    ) -> Dict:
        # Stores the `normalized` for use by the remaining steps in this flow.
        normalized = dict(sample)
        normalized["meta"] = dict(sample.get("meta", {}))
        graph = sample['x'].clone()
        x = graph.x.float().clone()

        # Stores the `coords` for use by the remaining steps in this flow.
        coords = self._normalize_coordinates(x[:, :3], norm_params['geometry'])
        x = torch.cat([coords, x[:, 3:]], dim=1) if x.shape[1] > 3 else coords
        feature_indices = [
            int(index)
            # Completes the multi-line `norm_params.get('x', {}).get` arguments or elements.
            for index in norm_params.get('x', {}).get('feature_indices', [])
            if int(index) < int(x.shape[1])
        ]
        if feature_indices and all(k in norm_params.get('x', {}) for k in ['mean', 'std', 'min', 'max']):
            # Stores the `x[:, feature_indices]` for use by the remaining steps in this flow.
            x[:, feature_indices] = self._apply_channel_norm(
                x[:, feature_indices],
                norm_params['x'],
                norm_type,
            # Completes the multi-line `self._apply_channel_norm` arguments or elements.
            )

        graph.x = x
        graph.edge_attr = self._build_edge_attr(
            edge_index=graph.edge_index,
            # Completes the multi-line `self._build_edge_attr` arguments or elements.
            node_coords=coords,
        )
        if self._has_explicit_motion_graph_contract(normalized["meta"]):
            self._attach_motion_graph_state(
                # Completes the multi-line `self._attach_motion_graph_state` arguments or elements.
                graph=graph,
                meta=normalized["meta"],
                norm_params=norm_params,
                norm_type=norm_type,
                # Completes the multi-line `bool` arguments or elements.
                targets_normalized=bool(normalize_y and norm_params.get("y")),
            )
        normalized['x'] = graph

        if normalize_y and 'y' in sample and norm_params.get('y'):
            # Stores the `y` for use by the remaining steps in this flow.
            y = sample['y'].float().clone()
            y_stats = norm_params['y']
            if y.ndim == 2:
                normalized['y'] = self._apply_channel_norm(y, y_stats, norm_type)
            # Checks `y.ndim == 3` before entering this conditional branch.
            elif y.ndim == 3:
                original_shape = y.shape
                normalized['y'] = self._apply_channel_norm(
                    y.reshape(-1, original_shape[-1]),
                    # Completes the multi-line `self._apply_channel_norm(y.reshape(-1, original_shape[-1]), y_stats, norm_type).reshape` arguments or elements.
                    y_stats,
                    norm_type,
                ).reshape(original_shape)
            else:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(f"GraphProcessor 的 y 维度应为2或3，当前为{y.ndim}")

        if normalize_y and 'y_seed' in sample and norm_params.get('y'):
            y_seed = sample['y_seed'].float().clone()
            y_stats = norm_params['y']
            # Checks `y_seed.ndim == 2` before entering this conditional branch.
            if y_seed.ndim == 2:
                normalized['y_seed'] = self._apply_channel_norm(y_seed, y_stats, norm_type)
            elif y_seed.ndim == 3:
                original_shape = y_seed.shape
                # Stores the `normalized['y_seed']` for use by the remaining steps in this flow.
                normalized['y_seed'] = self._apply_channel_norm(
                    y_seed.reshape(-1, original_shape[-1]),
                    y_stats,
                    norm_type,
                # Completes the multi-line `self._apply_channel_norm(y_seed.reshape(-1, original_shape[-1]), y_stats, norm_type).reshape` arguments or elements.
                ).reshape(original_shape)
            else:
                raise ValueError(f"GraphProcessor 的 y_seed 维度应为2或3，当前为{y_seed.ndim}")

        return normalized
    
    # 处理场数据，返回图结构格式
    def process(self, data: BaseData) -> Dict:
        """处理场数据，返回图结构格式
        
        Args:
            data: 场数据（BaseData对象）
            
        Returns:
            处理后的数据字典，包含：
                - x: 节点特征 (num_nodes, feature_dim) = 坐标(3) + 工况参数(params_dim)
                - edge_index: 图边索引 (2, num_edges)
                - y: 物理场数据 (num_nodes, output_dim) 或 (t, num_nodes, output_dim)
                - num_nodes: 节点数量
                - field_location: 'point' | 'cell'
        """
        # 1. 判断是否需要构建拓扑
        if self.is_geometry_varying or not self.first_process_done:
            # 几何变化或首次处理：构建拓扑
            topology = self._build_graph_topology(data)
            if not self.is_geometry_varying:
                # 几何固定：缓存拓扑，标记首次处理完成
                # Calls `self._cache_graph_structure` to perform the operation required at this stage.
                self._cache_graph_structure(topology)
        else:
            # 几何固定且已处理过：使用缓存
            topology = self._refresh_topology_edge_attr(data, self.cached_topology)
        
        # 2. 提取节点特征（坐标 + 工况参数）
        x_tensor = self._extract_node_features(data, topology)
        
        # 3. 提取目标字段（物理量）
        # Stores the `y_tensor` for use by the remaining steps in this flow.
        y_tensor = self._extract_target_field(data)
        meta = self._build_processed_sample_meta(
            data=data,
            y=y_tensor,
            # Completes the multi-line `self._build_processed_sample_meta` arguments or elements.
            time_axis=0 if y_tensor.ndim >= 3 else None,
        )
        meta = self._with_local_feature_meta(meta, **self._last_local_feature_meta)
        # 连续 VTK 局部特征（例如几何法向量）没有离散边界类别，
        # 不应强制构造 boundary_mask/boundary_entity_codes。离散边界
        # 特征仍沿用原有路径，用于训练噪声和边界状态覆盖。
        has_categorical_local_features = bool(meta.get("categorical_bc_indices"))
        # Stores the boundary mask for use by the remaining steps in this flow.
        boundary_mask = (
            self.build_boundary_mask(x_tensor, meta)
            if self.local_feature_encoding and has_categorical_local_features
            else None
        # Continues the surrounding statement or collection definition.
        )
        boundary_entity_codes = (
            self.build_boundary_entity_codes(x_tensor, meta)
            if self.local_feature_encoding and has_categorical_local_features
            # Continues the surrounding statement or collection definition.
            else None
        )
        
        # 4. 返回图数据
        return {
            'x': x_tensor,
            # Completes the multi-line `collection` arguments or elements.
            'edge_index': topology['edge_index'],
            'edge_attr': topology['edge_attr'],
            'y': y_tensor,
            'num_nodes': topology['num_nodes'],
            # Completes the multi-line `collection` arguments or elements.
            'field_location': topology['field_location'],
            'sample_id': self._extract_sample_id(data),
            'meta': meta,
            'boundary_mask': boundary_mask,
            # Completes the multi-line `collection` arguments or elements.
            'boundary_entity_codes': boundary_entity_codes,
        }

    def build_autoregressive_view(
        self,
        # Completes the multi-line `collection` arguments or elements.
        sample: Dict[str, Any],
        start_t: int,
        input_length: int,
        output_length: int,
    # Completes the multi-line `collection` arguments or elements.
    ) -> Tuple[Any, Any]:
        graph = sample["x"]
        y = sample["y"]
        if not hasattr(graph, "x") or not hasattr(graph, "edge_index"):
            # Raises `TypeError` to report why this path cannot continue.
            raise TypeError("GraphProcessor autoregressive 视图要求 sample['x'] 为图对象")
        if not isinstance(y, torch.Tensor) or y.ndim != 3:
            raise ValueError(
                f"GraphProcessor autoregressive 期望 y 形状为 [T, N, C]，当前为 {getattr(y, 'shape', None)}"
            # Completes the multi-line `ValueError` arguments or elements.
            )

        end_input_t = start_t + input_length
        end_target_t = end_input_t + output_length
        history = y[start_t:end_input_t]
        # Selects the required item from `y` as the `future`.
        future = y[end_input_t:end_target_t]

        if history.shape[0] != input_length:
            raise ValueError(
                f"GraphProcessor history 长度不足，期望 {input_length}，实际 {history.shape[0]}"
            # Completes the multi-line `ValueError` arguments or elements.
            )
        if future.shape[0] != output_length:
            raise ValueError(
                f"GraphProcessor future 长度不足，期望 {output_length}，实际 {future.shape[0]}"
            # Completes the multi-line `ValueError` arguments or elements.
            )
        if history.shape[1] != graph.x.shape[0]:
            raise ValueError(
                "GraphProcessor history 节点数与图节点数不一致，无法构造 autoregressive 视图"
            # Completes the multi-line `ValueError` arguments or elements.
            )

        history_flat = history.permute(1, 0, 2).reshape(graph.x.shape[0], -1)
        graph_view = graph.clone()
        self._update_motion_graph_coordinates(
            # Completes the multi-line `self._update_motion_graph_coordinates` arguments or elements.
            graph=graph_view,
            displacement_values=history[-1],
            meta=sample.get("meta", {}),
        )
        # Stores the `x` for use by the remaining steps in this flow.
        graph_view.x = torch.cat([graph_view.x, history_flat], dim=1)
        if self.local_feature_encoding:
            graph_view = self.attach_boundary_mask(
                graph_view,
                # Completes the multi-line `sample.get` arguments or elements.
                meta=sample.get("meta", {}),
                boundary_mask=sample.get("boundary_mask"),
            )
        if (
            # Completes the multi-line `self.config.get('training', {}).get` arguments or elements.
            self.config.get("training", {}).get("temporal_rollout", {}).get(
                "boundary_state_override", False
            )
        ):
            # Reads `boundary_entity_codes` from the mapping into the `boundary_entity_codes`.
            boundary_entity_codes = sample.get("boundary_entity_codes")
            if boundary_entity_codes is None:
                boundary_entity_codes = self.build_boundary_entity_codes(
                    graph.x, sample.get("meta", {})
                # Completes the multi-line `self.build_boundary_entity_codes` arguments or elements.
                )
            graph_view = self.attach_boundary_state_override(
                graph_view,
                history_last_state=history[-1],
                # Completes the multi-line `self.attach_boundary_state_override` arguments or elements.
                boundary_entity_codes=boundary_entity_codes,
            )
        graph_view.sample_id = getattr(graph, "sample_id", torch.tensor([int(sample.get("meta", {}).get("sample_id", 0))], dtype=torch.long))
        graph_view.num_full_graph_nodes = int(graph.x.shape[0])
        # Stores the `target` for use by the remaining steps in this flow.
        target = future.permute(1, 0, 2)
        if output_length == 1:
            target = target[:, 0, :]
        return graph_view, target

    # Defines `restore_autoregressive_model_output` for restore autoregressive model output logic.
    def restore_autoregressive_model_output(self, inputs: Any, raw_output: Any) -> Any:
        if not self.uses_autoregressive_increment():
            return raw_output
        if not hasattr(inputs, "x") or not isinstance(raw_output, torch.Tensor):
            # Raises `TypeError` to report why this path cannot continue.
            raise TypeError("GraphProcessor 增量自回归单步训练要求图输入和张量输出")
        output_step = raw_output[:, 0, :] if raw_output.ndim == 3 and raw_output.shape[1] == 1 else raw_output
        if output_step.ndim != 2:
            raise ValueError("GraphProcessor 增量自回归仅支持单步模型输出")
        # Converts the `input_length` to `int` so later checks use a consistent type.
        input_length = int(self.config.get("task", {}).get("input_length", 1))
        history_channels = input_length * int(output_step.shape[-1])
        history = inputs.x[:, -history_channels:].reshape(inputs.x.shape[0], input_length, -1)
        state_step = self.restore_autoregressive_state(history[:, -1, :], output_step)
        # Returns `state_step.unsqueeze(1) if raw_output.ndim == 3 else state_step` to the caller for the next stage.
        return state_step.unsqueeze(1) if raw_output.ndim == 3 else state_step

    def _build_autoregressive_predict_seed_sample(self, data: BaseData) -> Dict:
        """构建仅包含初始 history 的 Graph autoregressive 预测样本。"""
        processed = self.process(data)
        y = processed["y"]
        # Validates `not isinstance(y, torch.Tensor) or y.ndim != 3` and raises `ValueError` when the input cannot be processed safely.
        if not isinstance(y, torch.Tensor) or y.ndim != 3:
            raise ValueError(
                f"GraphProcessor autoregressive 预测样本期望完整 y 形状为 [T, N, C]，当前为 {getattr(y, 'shape', None)}"
            )

        # Converts the `input_length` to `int` so later checks use a consistent type.
        input_length = int(self.config.get("task", {}).get("input_length", 1))
        history = y[:input_length].clone()
        if int(history.shape[0]) != input_length:
            raise ValueError(
                # Completes the multi-line `ValueError` arguments or elements.
                f"GraphProcessor 预测 history 长度不足："
                f" got={history.shape[0]}, expected={input_length}"
            )

        dummy_target = self._build_dummy_target(int(history.shape[1]))
        # Stores the `meta` for use by the remaining steps in this flow.
        meta = self._build_temporal_rollout_seed_meta(
            processed.get("meta", {}),
            seed_length=input_length,
            total_time_steps=int(y.shape[0]),
        # Completes the multi-line `self._build_temporal_rollout_seed_meta` arguments or elements.
        )
        output = dict(processed)
        output["y"] = dummy_target
        output["y_seed"] = history
        # Stores the `output['meta']` for use by the remaining steps in this flow.
        output["meta"] = meta
        return output

    def _build_zero_autoregressive_predict_seed_sample(self, data: BaseData) -> Dict:
        """构建使用零初始场的 Graph autoregressive 预测样本。"""
        processed = self.process_input_only(data)
        # Selects the required item from `processed` as the `dummy_target`.
        dummy_target = processed["y"]
        meta = self._build_temporal_rollout_seed_meta(
            processed.get("meta", {}),
            seed_length=1,
            # Completes the multi-line `self._build_temporal_rollout_seed_meta` arguments or elements.
            init_history_strategy="zero",
            total_time_steps=1,
        )
        output = dict(processed)
        # Stores the `output['y']` for use by the remaining steps in this flow.
        output["y"] = dummy_target
        output["y_seed"] = dummy_target.unsqueeze(0).clone()
        output["meta"] = meta
        return output

    # Defines `build_autoregressive_predict_seed` for build autoregressive predict seed logic.
    def build_autoregressive_predict_seed(
        self,
        sample: Dict[str, Any],
        input_length: int,
    # Continues the surrounding statement or collection definition.
    ) -> Any:
        y_seed = sample.get("y_seed")
        if y_seed is None:
            return super().build_autoregressive_predict_seed(
                # Completes the multi-line `super().build_autoregressive_predict_seed` arguments or elements.
                sample=sample,
                input_length=input_length,
            )

        graph = sample["x"]
        # Stores the `history` for use by the remaining steps in this flow.
        history = y_seed.float()
        if history.ndim != 3:
            raise ValueError(
                f"GraphProcessor 预测 seed 期望 y_seed 形状为 [K, N, C]，当前为 {tuple(history.shape)}"
            # Completes the multi-line `ValueError` arguments or elements.
            )
        if int(history.shape[0]) != int(input_length):
            raise ValueError(
                f"GraphProcessor 预测 seed 的历史长度与 input_length 不一致："
                # Completes the multi-line `ValueError` arguments or elements.
                f" got={history.shape[0]}, expected={input_length}"
            )
        if int(history.shape[1]) != int(graph.x.shape[0]):
            raise ValueError("GraphProcessor 预测 seed 的节点数与图节点数不一致")

        # Stores the `history_flat` for use by the remaining steps in this flow.
        history_flat = history.permute(1, 0, 2).reshape(graph.x.shape[0], -1)
        graph_view = graph.clone()
        self._update_motion_graph_coordinates(
            graph=graph_view,
            # Completes the multi-line `self._update_motion_graph_coordinates` arguments or elements.
            displacement_values=history[-1],
            meta=sample.get("meta", {}),
        )
        graph_view.x = torch.cat([graph_view.x, history_flat], dim=1)
        # Checks `self.local_feature_encoding` before entering this conditional branch.
        if self.local_feature_encoding:
            graph_view = self.attach_boundary_mask(
                graph_view,
                meta=sample.get("meta", {}),
                # Completes the multi-line `sample.get` arguments or elements.
                boundary_mask=sample.get("boundary_mask"),
            )
        if (
            self.config.get("training", {}).get("temporal_rollout", {}).get(
                # Completes the multi-line `self.config.get('training', {}).get('temporal_rollout', {}).get` arguments or elements.
                "boundary_state_override", False
            )
        ):
            boundary_entity_codes = sample.get("boundary_entity_codes")
            # Checks `boundary_entity_codes is None` before entering this conditional branch.
            if boundary_entity_codes is None:
                boundary_entity_codes = self.build_boundary_entity_codes(
                    graph.x, sample.get("meta", {})
                )
            # Stores the `graph_view` for use by the remaining steps in this flow.
            graph_view = self.attach_boundary_state_override(
                graph_view,
                history_last_state=history[-1],
                boundary_entity_codes=boundary_entity_codes,
            # Completes the multi-line `self.attach_boundary_state_override` arguments or elements.
            )
        graph_view.sample_id = getattr(
            graph,
            "sample_id",
            # Completes the multi-line `torch.tensor` arguments or elements.
            torch.tensor([int(sample.get("meta", {}).get("sample_id", 0))], dtype=torch.long),
        )
        graph_view.num_full_graph_nodes = int(graph.x.shape[0])
        return graph_view
    
    # ========== 核心方法 ==========
    
    # Defines `predict_autoregressive_rollout` for predict autoregressive rollout logic.
    def predict_autoregressive_rollout(
        self,
        model,
        inputs: Any,
        # Continues the surrounding statement or collection definition.
        rollout_steps: int,
    ) -> Any:
        if not hasattr(inputs, "x") or not hasattr(inputs, "edge_index"):
            raise TypeError("GraphProcessor autoregressive rollout 仅支持图对象输入")

        # Converts the `steps` to `int` so later checks use a consistent type.
        steps = int(rollout_steps)
        if steps <= 0:
            raise ValueError(f"rollout_steps 必须为正整数，当前为 {rollout_steps}")

        input_length = int(self.config.get("task", {}).get("input_length", 1))
        # Validates `input_length <= 0` and raises `ValueError` when the input cannot be processed safely.
        if input_length <= 0:
            raise ValueError(f"input_length 必须为正整数，当前为 {input_length}")

        if (
            not hasattr(inputs, "boundary_state_override_mask")
            # Completes the multi-line `self._should_checkpoint_fixed_graph_rollout` arguments or elements.
            and self._should_checkpoint_fixed_graph_rollout(model=model, inputs=inputs)
        ):
            checkpoint_segments = self._resolve_rollout_checkpoint_segments(
                rollout_steps=steps
            # Completes the multi-line `self._resolve_rollout_checkpoint_segments` arguments or elements.
            )
            if checkpoint_segments > 0:
                if getattr(inputs, "motion_reference_pos", None) is not None:
                    return self._predict_checkpointed_motion_graph_rollout(
                        # Completes the multi-line `self._predict_checkpointed_motion_graph_rollout` arguments or elements.
                        model=model,
                        inputs=inputs,
                        rollout_steps=steps,
                        input_length=input_length,
                        # Completes the multi-line `self._predict_checkpointed_motion_graph_rollout` arguments or elements.
                        checkpoint_segments=checkpoint_segments,
                    )
                return self._predict_checkpointed_fixed_graph_rollout(
                    model=model,
                    # Completes the multi-line `self._predict_checkpointed_fixed_graph_rollout` arguments or elements.
                    inputs=inputs,
                    rollout_steps=steps,
                    input_length=input_length,
                    checkpoint_segments=checkpoint_segments,
                # Completes the multi-line `self._predict_checkpointed_fixed_graph_rollout` arguments or elements.
                )

        current_graph = inputs.clone()
        batch_predictions = []

        for _ in range(steps):
            # Stores the output for use by the remaining steps in this flow.
            output = model(current_graph)
            if output.dim() == 3 and output.shape[1] == 1:
                output_step = output[:, 0, :]
            elif output.dim() == 2:
                # Stores the `output_step` for use by the remaining steps in this flow.
                output_step = output
            else:
                raise ValueError(
                    "GraphProcessor 当前仅支持单步 autoregressive 输出，"
                    # Completes the multi-line `tuple` arguments or elements.
                    f"模型输出形状为 {tuple(output.shape)}"
                )

            out_channels = int(output_step.shape[-1])
            history_channels = input_length * out_channels
            # Derives the `static_dim` from the current values for downstream processing.
            static_dim = int(current_graph.x.shape[1]) - history_channels
            if static_dim < 0:
                raise ValueError(
                    "GraphProcessor 无法从图节点特征推断静态上下文维度，"
                    # Completes the multi-line `ValueError` arguments or elements.
                    f"input_dim={current_graph.x.shape[1]}, history_channels={history_channels}"
                )

            static_context = current_graph.x[:, :static_dim].clone()
            history = current_graph.x[:, static_dim:]
            # Stores the `history` for use by the remaining steps in this flow.
            history = history.reshape(history.shape[0], input_length, out_channels)
            state_step = self.restore_autoregressive_state(history[:, -1, :], output_step)
            state_step = self.apply_boundary_state_override(current_graph, state_step)
            next_history = torch.cat([history[:, 1:, :], state_step.unsqueeze(1)], dim=1)

            # Stores the `current_graph` for use by the remaining steps in this flow.
            current_graph = current_graph.clone()
            motion_updated = self._update_motion_graph_coordinates(
                graph=current_graph,
                displacement_values=state_step,
            # Completes the multi-line `self._update_motion_graph_coordinates` arguments or elements.
            )
            if motion_updated:
                static_context[:, :3] = current_graph.x[:, :3]
            current_graph.x = torch.cat(
                # Completes the multi-line `collection` arguments or elements.
                [static_context, next_history.reshape(next_history.shape[0], -1)],
                dim=1,
            )
            batch_predictions.append(state_step)

        # Returns `torch.stack(batch_predictions, dim=1)` to the caller for the next stage.
        return torch.stack(batch_predictions, dim=1)

    def _should_checkpoint_fixed_graph_rollout(
        self,
        model,
        # Continues the surrounding statement or collection definition.
        inputs: Any,
    ) -> bool:
        """判断是否处于可使用时间段重算的拓扑固定图模型训练期。"""
        return (
            type(model).__name__ in _ROLLOUT_CHECKPOINT_MODEL_NAMES
            # Completes the multi-line `bool` arguments or elements.
            and bool(getattr(model, "training", False))
            and torch.is_grad_enabled()
        )

    def _resolve_rollout_checkpoint_segments(self, rollout_steps: int) -> int:
        """解析时间维检查点段数，0 表示不启用。"""
        # Reads `checkpoint_segments` from the mapping into the `raw_value` with its declared fallback.
        raw_value = (
            self.config.get("training", {})
            .get("temporal_rollout", {})
            .get("checkpoint_segments", 0)
        # Continues the surrounding statement or collection definition.
        )
        if isinstance(raw_value, bool):
            raise ValueError("training.temporal_rollout.checkpoint_segments 必须为非负整数")
        if isinstance(raw_value, float) and not raw_value.is_integer():
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("training.temporal_rollout.checkpoint_segments 必须为非负整数")
        try:
            checkpoint_segments = int(raw_value)
        except (TypeError, ValueError) as exc:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(
                "training.temporal_rollout.checkpoint_segments 必须为非负整数"
            ) from exc
        if checkpoint_segments < 0:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("training.temporal_rollout.checkpoint_segments 必须为非负整数")
        if checkpoint_segments > rollout_steps:
            raise ValueError(
                "training.temporal_rollout.checkpoint_segments 不能大于本次 rollout 步数"
            # Completes the multi-line `ValueError` arguments or elements.
            )
        return checkpoint_segments

    def _run_fixed_graph_rollout_segment(
        self,
        # Continues the surrounding statement or collection definition.
        model,
        graph_template: Any,
        current_x: torch.Tensor,
        input_length: int,
        # Continues the surrounding statement or collection definition.
        segment_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """重放固定网格的一段自由滚动，输入和返回值均为可检查点的张量。"""
        predictions = []
        for _ in range(segment_steps):
            # Stores the `current_graph` for use by the remaining steps in this flow.
            current_graph = copy.copy(graph_template)
            current_graph.x = current_x
            output = model(current_graph)
            if output.dim() == 3 and output.shape[1] == 1:
                # Selects the required item from `output` as the `output_step`.
                output_step = output[:, 0, :]
            elif output.dim() == 2:
                output_step = output
            else:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(
                    "GraphProcessor 当前仅支持单步 autoregressive 输出，"
                    f"模型输出形状为 {tuple(output.shape)}"
                )

            # Converts the `out_channels` to `int` so later checks use a consistent type.
            out_channels = int(output_step.shape[-1])
            history_channels = input_length * out_channels
            static_dim = int(current_x.shape[1]) - history_channels
            if static_dim < 0:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(
                    "GraphProcessor 无法从图节点特征推断静态上下文维度，"
                    f"input_dim={current_x.shape[1]}, history_channels={history_channels}"
                )

            # Selects the required item from `current_x` as the `static_context`.
            static_context = current_x[:, :static_dim]
            history = current_x[:, static_dim:].reshape(
                current_x.shape[0],
                input_length,
                # Completes the multi-line `current_x[:, static_dim:].reshape` arguments or elements.
                out_channels,
            )
            state_step = self.restore_autoregressive_state(history[:, -1, :], output_step)
            next_history = torch.cat([history[:, 1:, :], state_step.unsqueeze(1)], dim=1)
            # Stores the `current_x` for use by the remaining steps in this flow.
            current_x = torch.cat(
                [static_context, next_history.reshape(next_history.shape[0], -1)],
                dim=1,
            )
            # Adds the current value to `predictions` for later aggregation.
            predictions.append(state_step)

        return current_x, torch.stack(predictions, dim=1)

    def _predict_checkpointed_fixed_graph_rollout(
        self,
        # Continues the surrounding statement or collection definition.
        model,
        inputs: Any,
        rollout_steps: int,
        input_length: int,
        # Continues the surrounding statement or collection definition.
        checkpoint_segments: int,
    ) -> torch.Tensor:
        """以时间段 checkpoint 保留跨步梯度，同时避免保存完整 H 步计算图。"""
        segment_size = (rollout_steps + checkpoint_segments - 1) // checkpoint_segments
        graph_template = copy.copy(inputs)
        # Stores the `current_x` for use by the remaining steps in this flow.
        current_x = inputs.x
        predictions = []
        completed_steps = 0

        while completed_steps < rollout_steps:
            # Stores the `current_segment_steps` for use by the remaining steps in this flow.
            current_segment_steps = min(segment_size, rollout_steps - completed_steps)

            def run_segment(
                segment_input: torch.Tensor,
                steps_in_segment: int = current_segment_steps,
            # Completes the multi-line `collection` arguments or elements.
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                return self._run_fixed_graph_rollout_segment(
                    model=model,
                    graph_template=graph_template,
                    # Completes the multi-line `self._run_fixed_graph_rollout_segment` arguments or elements.
                    current_x=segment_input,
                    input_length=input_length,
                    segment_steps=steps_in_segment,
                )

            # Stores the `(current_x, segment_predictions)` for use by the remaining steps in this flow.
            current_x, segment_predictions = checkpoint(
                run_segment,
                current_x,
                use_reentrant=False,
            # Completes the multi-line `checkpoint` arguments or elements.
            )
            predictions.append(segment_predictions)
            completed_steps += current_segment_steps

        return torch.cat(predictions, dim=1)

    # 重放拓扑固定运动网格的一段自由滚动，并传递坐标派生边特征。
    def _run_motion_graph_rollout_segment(
        self,
        model,
        graph_template: Any,
        # Continues the surrounding statement or collection definition.
        current_x: torch.Tensor,
        current_edge_attr: torch.Tensor,
        input_length: int,
        segment_steps: int,
    # Completes the multi-line `collection` arguments or elements.
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """重放拓扑固定运动网格的一段自由滚动，并传递坐标派生边特征。"""
        predictions = []
        for _ in range(segment_steps):
            current_graph = copy.copy(graph_template)
            # Stores the `x` for use by the remaining steps in this flow.
            current_graph.x = current_x
            current_graph.edge_attr = current_edge_attr
            output = model(current_graph)
            if output.dim() == 3 and output.shape[1] == 1:
                # Selects the required item from `output` as the `output_step`.
                output_step = output[:, 0, :]
            elif output.dim() == 2:
                output_step = output
            else:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(
                    "GraphProcessor 当前仅支持单步 autoregressive 输出，"
                    f"模型输出形状为 {tuple(output.shape)}"
                )

            # Converts the `out_channels` to `int` so later checks use a consistent type.
            out_channels = int(output_step.shape[-1])
            history_channels = input_length * out_channels
            static_dim = int(current_x.shape[1]) - history_channels
            if static_dim < 3:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(
                    "GraphProcessor 运动网格图缺少三个坐标静态特征，"
                    f"static_dim={static_dim}"
                )

            # Stores the `static_context` for use by the remaining steps in this flow.
            static_context = current_x[:, :static_dim].clone()
            history = current_x[:, static_dim:].reshape(
                current_x.shape[0],
                input_length,
                # Completes the multi-line `current_x[:, static_dim:].reshape` arguments or elements.
                out_channels,
            )
            state_step = self.restore_autoregressive_state(history[:, -1, :], output_step)
            next_history = torch.cat([history[:, 1:, :], state_step.unsqueeze(1)], dim=1)

            # Stores the `next_graph` for use by the remaining steps in this flow.
            next_graph = copy.copy(graph_template)
            next_graph.x = current_x.clone()
            next_graph.edge_attr = current_edge_attr
            if not self._update_motion_graph_coordinates(
                # Completes the multi-line `self._update_motion_graph_coordinates` arguments or elements.
                graph=next_graph,
                displacement_values=state_step,
            ):
                raise RuntimeError("GraphProcessor 运动网格检查点缺少坐标更新状态")
            # Selects the required item from `next_graph.x` as the `static_context[:, :3]`.
            static_context[:, :3] = next_graph.x[:, :3]
            current_x = torch.cat(
                [static_context, next_history.reshape(next_history.shape[0], -1)],
                dim=1,
            # Completes the multi-line `torch.cat` arguments or elements.
            )
            current_edge_attr = next_graph.edge_attr
            predictions.append(state_step)

        return current_x, current_edge_attr, torch.stack(predictions, dim=1)

    # 以时间段 checkpoint 重放拓扑固定运动网格的坐标和边特征更新。
    def _predict_checkpointed_motion_graph_rollout(
        self,
        model,
        inputs: Any,
        # Continues the surrounding statement or collection definition.
        rollout_steps: int,
        input_length: int,
        checkpoint_segments: int,
    ) -> torch.Tensor:
        """以时间段 checkpoint 重放拓扑固定运动网格的坐标和边特征更新。"""
        # Stores the `initial_edge_attr` for use by the remaining steps in this flow.
        initial_edge_attr = getattr(inputs, "edge_attr", None)
        if not isinstance(initial_edge_attr, torch.Tensor):
            raise ValueError("GraphProcessor 运动网格检查点要求 edge_attr 为张量")

        segment_size = (rollout_steps + checkpoint_segments - 1) // checkpoint_segments
        # Stores the `graph_template` for use by the remaining steps in this flow.
        graph_template = copy.copy(inputs)
        current_x = inputs.x
        current_edge_attr = initial_edge_attr
        predictions = []
        # Stores the `completed_steps` for use by the remaining steps in this flow.
        completed_steps = 0

        while completed_steps < rollout_steps:
            current_segment_steps = min(segment_size, rollout_steps - completed_steps)

            def run_segment(
                # Continues the surrounding statement or collection definition.
                segment_x: torch.Tensor,
                segment_edge_attr: torch.Tensor,
                steps_in_segment: int = current_segment_steps,
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                # Returns `` to the caller for the next stage.
                return self._run_motion_graph_rollout_segment(
                    model=model,
                    graph_template=graph_template,
                    current_x=segment_x,
                    # Completes the multi-line `self._run_motion_graph_rollout_segment` arguments or elements.
                    current_edge_attr=segment_edge_attr,
                    input_length=input_length,
                    segment_steps=steps_in_segment,
                )

            # Stores the `(current_x, current_edge_attr, segment_predictions)` for use by the remaining steps in this flow.
            current_x, current_edge_attr, segment_predictions = checkpoint(
                run_segment,
                current_x,
                current_edge_attr,
                # Completes the multi-line `checkpoint` arguments or elements.
                use_reentrant=False,
            )
            predictions.append(segment_predictions)
            completed_steps += current_segment_steps

        # Returns `torch.cat(predictions, dim=1)` to the caller for the next stage.
        return torch.cat(predictions, dim=1)

    def _build_graph_topology(self, data: BaseData) -> Dict:
        """构建图拓扑（从单个样本）
        
        BaseData保证了point_or_cell与Fields维度的一致性：
        - point_or_cell='point' → Fields维度为(num_points, ...)
        - point_or_cell='cell' → Fields维度为(num_cells, ...)
        
        Args:
            data: BaseData 对象
            
        Returns:
            拓扑字典: {
                'edge_index': torch.Tensor (2, num_edges),
                'field_location': 'point' | 'cell',
                'num_nodes': int
            }
        """
        data_dict = data.get_data()
        points = data_dict['PointCoordinates']  # (num_points, 3)
        # Selects the required item from `data_dict` as the `cells`.
        cells = data_dict['ElementConnectivity']  # (num_cells, nodes_per_cell)
        field_location = data.point_or_cell  # 'point' 或 'cell'
        geometry_representation = str(data_dict.get('GeometryRepresentation', '') or '').strip().lower()
        has_topology = cells is not None and len(cells) > 0

        # Checks `field_location == 'point'` before entering this conditional branch.
        if field_location == 'point':
            # 字段在节点上：网格点作节点，单元连接作边
            if has_topology:
                edge_index = self._build_edge_index_from_cells(cells, len(points))
            else:
                # Stores the graph edge indices for use by the remaining steps in this flow.
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            node_coords = np.asarray(points, dtype=np.float32)
            num_nodes = len(points)
        else:
            # 字段在单元上：单元中心作节点，相邻单元作边
            # Validates `not has_topology` and raises `ValueError` when the input cannot be processed safely.
            if not has_topology:
                raise ValueError("GraphProcessor 在 cell 场景下需要有效的单元拓扑。")
            edge_index = self._build_edge_index_from_adjacency(cells)
            node_coords = np.asarray(data_dict['CellCoordinates'], dtype=np.float32)
            # Stores the `num_nodes` for use by the remaining steps in this flow.
            num_nodes = len(cells)

        effective_edge_strategy = self.graph_edge_strategy
        effective_knn_k = self.graph_knn_k
        if (
            # Continues the surrounding statement or collection definition.
            effective_edge_strategy == 'topology'
            and not has_topology
            and geometry_representation == 'point_cloud'
        ):
            # Stores the `effective_edge_strategy` for use by the remaining steps in this flow.
            effective_edge_strategy = 'knn'
            effective_knn_k = effective_knn_k if effective_knn_k > 0 else 8
            _get_logger().info(
                "检测到点云样本缺少显式拓扑，GraphProcessor 自动回退到 knn 建图: k=%s",
                # Completes the multi-line `_get_logger().info` arguments or elements.
                effective_knn_k,
            )

        if effective_edge_strategy == 'knn':
            edge_index = self._build_edge_index_from_knn(
                # Completes the multi-line `self._build_edge_index_from_knn` arguments or elements.
                node_coords=node_coords,
                k=effective_knn_k,
            )
        elif effective_edge_strategy == 'radius':
            # Stores the graph edge indices for use by the remaining steps in this flow.
            edge_index = self._build_edge_index_from_radius(
                node_coords=node_coords,
                radius=self.graph_radius,
                max_neighbors=self.graph_radius_candidate_limit,
            # Completes the multi-line `self._build_edge_index_from_radius` arguments or elements.
            )

        # 先计算一维距离用于裁剪，再对保留边构建完整4维边特征，减少无效计算
        edge_distance = self._build_edge_distance(edge_index=edge_index, node_coords=node_coords)
        edge_index, keep_idx = self._limit_edges_per_node(
            edge_index=edge_index,
            # Completes the multi-line `self._limit_edges_per_node` arguments or elements.
            edge_distance=edge_distance,
            num_nodes=num_nodes,
            edge_strategy=effective_edge_strategy,
        )
        # Stores the `kept_distance` for use by the remaining steps in this flow.
        kept_distance = edge_distance[keep_idx] if keep_idx.numel() > 0 else torch.zeros((0,), dtype=torch.float32)
        edge_index, kept_distance = self._make_bidirectional(
            edge_index=edge_index,
            edge_distance=kept_distance,
        # Completes the multi-line `self._make_bidirectional` arguments or elements.
        )
        edge_index, kept_distance = self._append_self_loops(
            edge_index=edge_index,
            edge_distance=kept_distance,
            # Completes the multi-line `self._append_self_loops` arguments or elements.
            num_nodes=num_nodes,
        )
        edge_attr = self._build_edge_attr(
            edge_index=edge_index,
            # Completes the multi-line `self._build_edge_attr` arguments or elements.
            node_coords=node_coords,
            precomputed_distance=kept_distance,
        )
        return {
            # Completes the multi-line `collection` arguments or elements.
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'field_location': field_location,
            'num_nodes': num_nodes
        # Completes the multi-line `collection` arguments or elements.
        }
    
    def _extract_node_features(self, data: BaseData, topology: Dict) -> torch.Tensor:
        """提取节点特征：坐标 + 工况参数
        
        x = 节点坐标(3) + 工况参数(c)，所有节点共享工况参数
        形状：(num_nodes, 3+c)
        
        工况参数从 sample_map 中提取，排除 sample_order 和 file_path
        示例 sample_map: {'sample_order': 1, 'velocity': 5, 'heatPower': 100, 'file_path': '1\\result.cgns'}
        
        Args:
            data: BaseData 对象
            topology: 拓扑信息（包含字段位置）
            
        Returns:
            torch.Tensor (num_nodes, feature_dim)
        """
        data_dict = data.get_data()
        field_location = topology['field_location']
        # Stores the `_last_local_feature_meta` on the instance so other methods can reuse it.
        self._last_local_feature_meta = {}
        
        # 1. 获取节点坐标
        # 第一阶段图结构仍锚定主模板几何（通常为 t=0），不随时序坐标快照动态更新。
        if field_location == 'point':
            coords = data_dict['PointCoordinates']
        else:
            # Selects the required item from `data_dict` as the `coords`.
            coords = data_dict['CellCoordinates']
        
        coords_tensor = torch.tensor(coords, dtype=torch.float32)  # (num_nodes, 3)
        
        # 2. 获取工况参数（全局特征）
        # 从 sample_map 中排除 sample_order 和 file_path，其余都是工况参数
        sample_map = data.get_sample_map()
        if sample_map:
            # Builds the `working_condition` using the comprehension's filter and mapping rules.
            working_condition = {
                k: v
                for k, v in sample_map.items()
                if k not in _GRAPH_NON_PARAM_SAMPLE_KEYS
            # Continues the surrounding statement or collection definition.
            }
        else:
            working_condition = {}
        
        if working_condition:
            # 工况参数排序后拼接
            # Initializes the `param_values` as a collection for values from this processing stage.
            param_values = []
            for param_name in sorted(working_condition.keys()):
                param_value = working_condition[param_name]
                try:
                    # Adds the current value to `param_values` for later aggregation.
                    param_values.append(float(param_value))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"GraphProcessor 无法将工况参数转为数值: "
                        # Completes the multi-line `ValueError` arguments or elements.
                        f"param={param_name}, value={param_value!r}"
                    ) from exc
            params_array = np.array(param_values, dtype=np.float32)
            params_tensor = torch.tensor(params_array, dtype=torch.float32)
        # Continues the surrounding statement or collection definition.
        else:
            params_tensor = torch.empty((0,), dtype=torch.float32)
        
        # 3. 复制工况参数到每个节点（广播）
        num_nodes = coords_tensor.shape[0]
        if params_tensor.numel() > 0:
            # Stores the `params_broadcast` for use by the remaining steps in this flow.
            params_broadcast = params_tensor.unsqueeze(0).expand(num_nodes, -1)  # (num_nodes, c)
            x_tensor = torch.cat([coords_tensor, params_broadcast], dim=1)  # (num_nodes, 3+c)
        else:
            x_tensor = coords_tensor  # (num_nodes, 3)

        # Checks `self.local_feature_encoding` before entering this conditional branch.
        if self.local_feature_encoding:
            local_features, feature_names, feature_schema = self._build_local_feature_matrix(
                data_dict=data_dict,
                target_location=field_location,
            # Completes the multi-line `self._build_local_feature_matrix` arguments or elements.
            )
            if local_features.shape[0] != num_nodes:
                raise ValueError(
                    "GraphProcessor 局部边界条件特征数量与节点数不一致："
                    # Completes the multi-line `ValueError` arguments or elements.
                    f"features={local_features.shape[0]}, nodes={num_nodes}"
                )
            local_tensor = torch.tensor(local_features, dtype=torch.float32)
            feature_start = int(x_tensor.shape[1])
            # Stores the `x_tensor` for use by the remaining steps in this flow.
            x_tensor = torch.cat([x_tensor, local_tensor], dim=1)
            self._last_local_feature_meta = self._build_local_feature_meta_payload(
                schema=feature_schema,
                feature_start=feature_start,
                # Completes the multi-line `self._build_local_feature_meta_payload` arguments or elements.
                feature_location=field_location,
            )
        
        return x_tensor
    
    def _extract_target_field(self, data: BaseData) -> torch.Tensor:
        """提取目标字段（物理量的叠加）
        
        y = 所有物理量拼接
        稳态：(num_nodes, m)
        时序：(t, num_nodes, m)
        
        Returns:
            torch.Tensor
        """
        # Stores the `data_dict` for use by the remaining steps in this flow.
        data_dict = data.get_data()
        fields = data_dict.get('Fields', {})
        
        if not fields:
            raise ValueError("没有字段数据可提取")
        
        # 按字段名排序，逐个提取
        # Stores the `configured_targets` for use by the remaining steps in this flow.
        configured_targets = list(
            self.config.get('task', {}).get('effective_variables')
            or []
        )
        # Checks `configured_targets` before entering this conditional branch.
        if configured_targets:
            missing_fields = [name for name in configured_targets if name not in fields]
            if missing_fields:
                raise ValueError(f"GraphProcessor 鐩爣瀛楁缂哄け: {missing_fields}")
            # Stores the `target_field_names` for use by the remaining steps in this flow.
            target_field_names = configured_targets
        else:
            target_field_names = [
                name
                # Completes the multi-line `sorted` arguments or elements.
                for name in sorted(fields.keys())
                if not (self.local_feature_encoding == "vtk_based" and name in self.field_bc)
            ]
            if not target_field_names:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError("GraphProcessor 未找到可作为预测目标的物理字段")

        field_list = []
        for field_name in target_field_names:
            field_array = fields[field_name]
            
            # Checks `isinstance(field_array, torch.Tensor)` before entering this conditional branch.
            if isinstance(field_array, torch.Tensor):
                field_tensor = field_array
            else:
                field_tensor = torch.tensor(field_array, dtype=torch.float32)
            
            # Adds the current value to `field_list` for later aggregation.
            field_list.append(field_tensor)
        
        # 拼接所有物理量
        # 稳态：(N,) → (N, 1)，拼接后 (N, m)
        # 时序：(t, N) → (t, N, 1)，拼接后 (t, N, m)
        if field_list[0].ndim == 1:
            # 稳态：展开为 (N, 1)
            field_list = [f.unsqueeze(1) if f.ndim == 1 else f for f in field_list]
            y_tensor = torch.cat(field_list, dim=1)  # (N, m)
        # Checks `field_list[0].ndim == 2` before entering this conditional branch.
        elif field_list[0].ndim == 2:
            # 时序：展开为 (t, N, 1)
            field_list = [f.unsqueeze(2) if f.ndim == 2 else f for f in field_list]
            y_tensor = torch.cat(field_list, dim=2)  # (t, N, m)
        else:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(f"不支持的字段维度：{field_list[0].ndim}")
        
        return y_tensor.float()

    @staticmethod
    def _extract_sample_id(data: BaseData) -> int:
        """Extract sample id from BaseData metadata when available."""
        # Stores the `sample_map` for use by the remaining steps in this flow.
        sample_map = data.get_sample_map() or {}
        sample_id = sample_map.get('sample_order', 0)
        try:
            return int(sample_id)
        # Completes the multi-line `collection` arguments or elements.
        except (TypeError, ValueError):
            return 0
    
    
    # ========== 图构建方法 ==========
    
    @staticmethod
    def _normalize_point_cell_indices(cells: np.ndarray, num_nodes: int) -> np.ndarray:
        """将单元节点索引标准化为 0-based，兼容 CGNS 的 1-based 约定。"""
        # Stores the `normalized` for use by the remaining steps in this flow.
        normalized = np.empty(len(cells), dtype=object)

        for idx, cell in enumerate(cells):
            indices = np.asarray(cell, dtype=np.int64).reshape(-1)
            if indices.size == 0:
                # Stores the `normalized[idx]` for use by the remaining steps in this flow.
                normalized[idx] = indices
                continue

            if np.all(indices >= 1) and np.all(indices <= num_nodes):
                normalized[idx] = indices - 1
            # Continues the surrounding statement or collection definition.
            else:
                normalized[idx] = indices

        return normalized

    def _build_edge_index_from_cells(self, cells: np.ndarray, num_nodes: int) -> torch.Tensor:
        """从单元构建边：单元内所有相连节点
        
        使用 NumPy 向量化操作替代嵌套循环，提升 10-50x 性能
        
        Args:
            cells: 单元连接数组 (num_cells, nodes_per_cell)
            num_nodes: 节点总数
            
        Returns:
            torch.Tensor (2, num_edges)
        """
        # Stores the `cells` for use by the remaining steps in this flow.
        cells = self._normalize_point_cell_indices(cells, num_nodes)
        pair_chunks = []

        for cell in cells:
            indices = np.asarray(cell, dtype=np.int64).reshape(-1)
            # Checks `indices.size < 2` before entering this conditional branch.
            if indices.size < 2:
                continue

            # 使用上三角索引直接生成组合，避免构造n*n中间矩阵
            i_idx, j_idx = np.triu_indices(indices.size, k=1)
            if i_idx.size == 0:
                # Handles the data or control-flow case represented by this statement.
                continue

            a = indices[i_idx]
            b = indices[j_idx]
            # 规范为(min, max)便于跨单元去重
            lo = np.minimum(a, b)
            # Stores the `hi` for use by the remaining steps in this flow.
            hi = np.maximum(a, b)
            pair_chunks.append(np.stack([lo, hi], axis=1))

        if pair_chunks:
            pairs = np.concatenate(pair_chunks, axis=0)
            # Stores the `pairs` for use by the remaining steps in this flow.
            pairs = np.unique(pairs, axis=0)
            pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
            edge_index = torch.as_tensor(pairs.T, dtype=torch.long)
        else:
            # Stores the graph edge indices for use by the remaining steps in this flow.
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return edge_index
    
    def _build_edge_index_from_adjacency(self, cells: np.ndarray) -> torch.Tensor:
        """从单元构建边：共享至少一个顶点的相邻单元
        
        使用 NumPy 向量化操作替代嵌套循环，提升 5-20x 性能
        
        Args:
            cells: 单元连接数组 (num_cells, nodes_per_cell)
            
        Returns:
            torch.Tensor (2, num_edges)
        """
        num_cells = len(cells)
        
        # 预计算：node → cells 映射（反向索引）
        # 这使得相邻检测从 O(C²) 降低到 O(C×K×Ā)
        # Initializes the `node_to_cells` as a collection for values from this processing stage.
        node_to_cells = {}
        for cell_idx, cell in enumerate(cells):
            for node in cell:
                node = int(node)
                # Checks `node not in node_to_cells` before entering this conditional branch.
                if node not in node_to_cells:
                    node_to_cells[node] = []
                node_to_cells[node].append(cell_idx)
        
        # 用映射加速相邻检测
        edges = set()
        
        # Visits each item in `range(num_cells)` and applies the current processing step.
        for cell_i in range(num_cells):
            cell_i_nodes = set(cells[cell_i])
            
            # 只检查通过共享节点可达的单元
            candidate_cells = set()
            for node in cell_i_nodes:
                # Adds the current value to `candidate_cells` for later aggregation.
                candidate_cells.update(node_to_cells.get(int(node), []))
            
            # 只检查 cell_i 之后的单元（避免重复）
            for cell_j in candidate_cells:
                if cell_i < cell_j:
                    edges.add((cell_i, cell_j))
        
        # Checks `edges` before entering this conditional branch.
        if edges:
            edges_list = sorted(list(edges))
            edge_index = torch.tensor(edges_list, dtype=torch.long).T
        else:
            # Stores the graph edge indices for use by the remaining steps in this flow.
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return edge_index

    @staticmethod
    def _build_edge_index_from_knn(node_coords: np.ndarray, k: int) -> torch.Tensor:
        """根据节点坐标构建无向 kNN 候选边。"""
        # Stores the `coords` for use by the remaining steps in this flow.
        coords = np.asarray(node_coords, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"node_coords形状应为(N, 3)，当前为{tuple(coords.shape)}")

        num_nodes = int(coords.shape[0])
        # Returns early when `num_nodes <= 1` makes the remaining work inapplicable.
        if num_nodes <= 1:
            return torch.zeros((2, 0), dtype=torch.long)

        if int(k) <= 0:
            raise ValueError(f"processor.graph_knn_k 必须 >= 1，当前为 {k}")

        # Stores the `effective_k` for use by the remaining steps in this flow.
        effective_k = min(int(k), num_nodes - 1)
        tree = cKDTree(coords)
        _, neighbor_indices = tree.query(coords, k=effective_k + 1)

        neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
        # Checks `neighbor_indices.ndim == 1` before entering this conditional branch.
        if neighbor_indices.ndim == 1:
            neighbor_indices = neighbor_indices.reshape(num_nodes, 1)
        neighbor_indices = neighbor_indices[:, 1:]
        if neighbor_indices.size == 0:
            # Returns `torch.zeros((2, 0), dtype=torch.long)` to the caller for the next stage.
            return torch.zeros((2, 0), dtype=torch.long)

        src = np.repeat(np.arange(num_nodes, dtype=np.int64), neighbor_indices.shape[1])
        dst = neighbor_indices.reshape(-1)
        valid_mask = src != dst
        # Returns early when `not np.any(valid_mask)` makes the remaining work inapplicable.
        if not np.any(valid_mask):
            return torch.zeros((2, 0), dtype=torch.long)

        src = src[valid_mask]
        dst = dst[valid_mask]
        # Stores the `pairs` for use by the remaining steps in this flow.
        pairs = np.stack([np.minimum(src, dst), np.maximum(src, dst)], axis=1)
        pairs = np.unique(pairs, axis=0)
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
        return torch.as_tensor(pairs.T, dtype=torch.long)

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _build_edge_index_from_radius(
        node_coords: np.ndarray,
        radius: float,
        # Continues the surrounding statement or collection definition.
        max_neighbors: int = 0,
    ) -> torch.Tensor:
        """根据半径阈值构建无向 radius 图候选边。"""
        coords = np.asarray(node_coords, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != 3:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(f"node_coords褰㈢姸搴斾负(N, 3)锛屽綋鍓嶄负{tuple(coords.shape)}")

        num_nodes = int(coords.shape[0])
        if num_nodes <= 1:
            return torch.zeros((2, 0), dtype=torch.long)

        # Validates `float(radius) <= 0` and raises `ValueError` when the input cannot be processed safely.
        if float(radius) <= 0:
            raise ValueError(f"processor.graph_radius 蹇呴』 > 0锛屽綋鍓嶄负 {radius}")

        if int(max_neighbors) < 0:
            raise ValueError(
                # Completes the multi-line `ValueError` arguments or elements.
                f"processor.graph_radius_max_neighbors 蹇呴』 >= 0锛屽綋鍓嶄负 {max_neighbors}"
            )

        pair_chunks = []
        neighbor_cap = int(max_neighbors)
        # Stores the `tree` for use by the remaining steps in this flow.
        tree = cKDTree(coords)

        if neighbor_cap > 0:
            query_k = min(neighbor_cap + 1, num_nodes)
            distances, neighbor_indices = tree.query(
                # Completes the multi-line `tree.query` arguments or elements.
                coords,
                k=query_k,
                distance_upper_bound=float(radius),
            )
            # Stores the `distances` for use by the remaining steps in this flow.
            distances = np.asarray(distances)
            neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
            if distances.ndim == 1:
                distances = distances.reshape(num_nodes, 1)
                # Stores the `neighbor_indices` for use by the remaining steps in this flow.
                neighbor_indices = neighbor_indices.reshape(num_nodes, 1)

            for src in range(num_nodes):
                distance_row = distances[src]
                index_row = neighbor_indices[src]
                # Stores the valid-item mask for use by the remaining steps in this flow.
                valid_mask = np.isfinite(distance_row)
                valid_indices = index_row[valid_mask]
                filtered = [int(dst) for dst in valid_indices.tolist() if int(dst) != src and int(dst) < num_nodes]
                if not filtered:
                    # Handles the data or control-flow case represented by this statement.
                    continue

                src_array = np.full((len(filtered),), src, dtype=np.int64)
                dst_array = np.asarray(filtered, dtype=np.int64)
                pair_chunks.append(
                    # Completes the multi-line `np.stack` arguments or elements.
                    np.stack([np.minimum(src_array, dst_array), np.maximum(src_array, dst_array)], axis=1)
                )
        else:
            neighbor_groups = tree.query_ball_point(coords, r=float(radius))
            # Visits each item in `enumerate(neighbor_groups)` and applies the current processing step.
            for src, neighbors in enumerate(neighbor_groups):
                filtered = [int(dst) for dst in neighbors if int(dst) != src]
                if not filtered:
                    continue

                # Stores the `src_array` for use by the remaining steps in this flow.
                src_array = np.full((len(filtered),), src, dtype=np.int64)
                dst_array = np.asarray(filtered, dtype=np.int64)
                pair_chunks.append(
                    np.stack([np.minimum(src_array, dst_array), np.maximum(src_array, dst_array)], axis=1)
                # Completes the multi-line `pair_chunks.append` arguments or elements.
                )

        if not pair_chunks:
            return torch.zeros((2, 0), dtype=torch.long)

        pairs = np.concatenate(pair_chunks, axis=0)
        # Stores the `pairs` for use by the remaining steps in this flow.
        pairs = np.unique(pairs, axis=0)
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
        return torch.as_tensor(pairs.T, dtype=torch.long)

    @staticmethod
    # 根据边两端坐标构建欧氏距离（一维）。
    def _build_edge_distance(edge_index: torch.Tensor, node_coords: np.ndarray) -> torch.Tensor:
        """根据边两端坐标构建欧氏距离（一维）。"""
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index形状应为(2, E)，当前为{tuple(edge_index.shape)}")

        coords = torch.as_tensor(node_coords, dtype=torch.float32)
        # Validates `coords.ndim != 2 or coords.shape[1] != 3` and raises `ValueError` when the input cannot be processed safely.
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"node_coords形状应为(N, 3)，当前为{tuple(coords.shape)}")

        if edge_index.shape[1] == 0:
            return torch.zeros((0,), dtype=torch.float32)

        # Stores the `src` for use by the remaining steps in this flow.
        src = edge_index[0].to(dtype=torch.long)
        dst = edge_index[1].to(dtype=torch.long)
        num_nodes = coords.shape[0]

        if src.min().item() < 0 or dst.min().item() < 0 or src.max().item() >= num_nodes or dst.max().item() >= num_nodes:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(
                f"edge_index存在越界索引，节点数={num_nodes}, src范围=[{src.min().item()}, {src.max().item()}], "
                f"dst范围=[{dst.min().item()}, {dst.max().item()}]"
            )

        # Derives the `deltas` from the current values for downstream processing.
        deltas = coords[dst] - coords[src]
        return torch.norm(deltas, dim=1)

    @staticmethod
    def _build_edge_attr(
        # Continues the surrounding statement or collection definition.
        edge_index: torch.Tensor,
        node_coords: np.ndarray,
        precomputed_distance: torch.Tensor = None,
    ) -> torch.Tensor:
        """根据边两端坐标构建边特征。

        edge_attr = [|dx|, |dy|, |dz|, euclidean_distance]
        形状: (num_edges, 4)
        """
        # Validates `edge_index.ndim != 2 or edge_index.shape[0] != 2` and raises `ValueError` when the input cannot be processed safely.
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index形状应为(2, E)，当前为{tuple(edge_index.shape)}")

        if isinstance(node_coords, torch.Tensor):
            coords = node_coords.to(dtype=torch.float32)
        # Continues the surrounding statement or collection definition.
        else:
            coords = torch.as_tensor(node_coords, dtype=torch.float32)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"node_coords形状应为(N, 3)，当前为{tuple(coords.shape)}")

        # Returns early when `edge_index.shape[1] == 0` makes the remaining work inapplicable.
        if edge_index.shape[1] == 0:
            return torch.zeros((0, 4), dtype=torch.float32)

        src = edge_index[0].to(dtype=torch.long, device=coords.device)
        dst = edge_index[1].to(dtype=torch.long, device=coords.device)
        # Selects the required item from `coords.shape` as the `num_nodes`.
        num_nodes = coords.shape[0]

        if src.min().item() < 0 or dst.min().item() < 0 or src.max().item() >= num_nodes or dst.max().item() >= num_nodes:
            raise ValueError(
                f"edge_index存在越界索引，节点数={num_nodes}, src范围=[{src.min().item()}, {src.max().item()}], "
                # Completes the multi-line `dst.min().item` arguments or elements.
                f"dst范围=[{dst.min().item()}, {dst.max().item()}]"
            )

        deltas = coords[dst] - coords[src]
        abs_deltas = torch.abs(deltas)
        # Checks `precomputed_distance is None` before entering this conditional branch.
        if precomputed_distance is None:
            distances = torch.norm(deltas, dim=1, keepdim=True)
        else:
            distances = precomputed_distance.reshape(-1, 1).to(dtype=torch.float32, device=deltas.device)
        # Returns `torch.cat([abs_deltas, distances], dim=1)` to the caller for the next stage.
        return torch.cat([abs_deltas, distances], dim=1)

    @staticmethod
    def _make_bidirectional(edge_index: torch.Tensor, edge_distance: torch.Tensor):
        """Expand each non-loop edge to two directions: u->v and v->u."""
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            # Returns `edge_index, edge_distance` to the caller for the next stage.
            return edge_index, edge_distance
        if edge_distance.ndim != 1 or edge_distance.shape[0] != edge_index.shape[1]:
            return edge_index, edge_distance
        if edge_index.shape[1] == 0:
            # Returns `edge_index, edge_distance` to the caller for the next stage.
            return edge_index, edge_distance

        non_loop_mask = edge_index[0] != edge_index[1]
        edge_index_wo_loops = edge_index[:, non_loop_mask]
        edge_distance_wo_loops = edge_distance[non_loop_mask]
        # Returns early when `edge_index_wo_loops.shape[1] == 0` makes the remaining work inapplicable.
        if edge_index_wo_loops.shape[1] == 0:
            return edge_index_wo_loops, edge_distance_wo_loops

        reverse_edge_index = torch.stack([edge_index_wo_loops[1], edge_index_wo_loops[0]], dim=0)
        bidirectional_edge_index = torch.cat([edge_index_wo_loops, reverse_edge_index], dim=1)
        # Stores the `bidirectional_edge_distance` for use by the remaining steps in this flow.
        bidirectional_edge_distance = torch.cat([edge_distance_wo_loops, edge_distance_wo_loops], dim=0)
        return bidirectional_edge_index, bidirectional_edge_distance

    @staticmethod
    def _append_self_loops(edge_index: torch.Tensor, edge_distance: torch.Tensor, num_nodes: int):
        """在图中追加每个节点的自环，且自环距离为0。"""
        # Returns early when `num_nodes <= 0` makes the remaining work inapplicable.
        if num_nodes <= 0:
            return edge_index, edge_distance
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            return edge_index, edge_distance
        # Returns early when `edge_distance.ndim != 1` makes the remaining work inapplicable.
        if edge_distance.ndim != 1:
            return edge_index, edge_distance
        if edge_distance.shape[0] != edge_index.shape[1]:
            return edge_index, edge_distance

        # Stores the `non_loop_mask` for use by the remaining steps in this flow.
        non_loop_mask = edge_index[0] != edge_index[1]
        if non_loop_mask.shape[0] != edge_index.shape[1]:
            return edge_index, edge_distance

        edge_index_wo_loops = edge_index[:, non_loop_mask]
        # Selects the required item from `edge_distance` as the `edge_distance_wo_loops`.
        edge_distance_wo_loops = edge_distance[non_loop_mask]

        loop_nodes = torch.arange(num_nodes, dtype=edge_index.dtype, device=edge_index.device)
        loop_edges = torch.stack([loop_nodes, loop_nodes], dim=0)
        loop_distance = torch.zeros((num_nodes,), dtype=edge_distance.dtype, device=edge_distance.device)

        # Stores the `edge_index_out` for use by the remaining steps in this flow.
        edge_index_out = torch.cat([edge_index_wo_loops, loop_edges], dim=1)
        edge_distance_out = torch.cat([edge_distance_wo_loops, loop_distance], dim=0)
        return edge_index_out, edge_distance_out

    def _limit_edges_per_node(
        # Continues the surrounding statement or collection definition.
        self,
        edge_index: torch.Tensor,
        edge_distance: torch.Tensor,
        num_nodes: int,
        # Continues the surrounding statement or collection definition.
        edge_strategy: Optional[str] = None,
    ):
        """按欧氏距离做每节点top-k裁剪（并集语义）。

        对每个节点，保留incident edges中欧氏距离最短的k条，
        最终结果取所有节点保留集合的并集。
        """
        effective_edge_strategy = str(edge_strategy or self.graph_edge_strategy).lower()
        if effective_edge_strategy == 'knn':
            # Stores the `keep_idx` for use by the remaining steps in this flow.
            keep_idx = torch.arange(edge_index.shape[1], dtype=torch.long, device=edge_index.device)
            return edge_index, keep_idx
        if self.graph_max_edges_per_node <= 0:
            keep_idx = torch.arange(edge_index.shape[1], dtype=torch.long, device=edge_index.device)
            # Returns `edge_index, keep_idx` to the caller for the next stage.
            return edge_index, keep_idx
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            keep_idx = torch.arange(edge_index.shape[1], dtype=torch.long, device=edge_index.device)
            return edge_index, keep_idx
        # Checks `edge_distance.ndim != 1` before entering this conditional branch.
        if edge_distance.ndim != 1:
            keep_idx = torch.arange(edge_index.shape[1], dtype=torch.long, device=edge_index.device)
            return edge_index, keep_idx
        if edge_distance.shape[0] != edge_index.shape[1]:
            # Stores the `keep_idx` for use by the remaining steps in this flow.
            keep_idx = torch.arange(edge_index.shape[1], dtype=torch.long, device=edge_index.device)
            return edge_index, keep_idx
        if num_nodes <= 0:
            keep_idx = torch.arange(edge_index.shape[1], dtype=torch.long, device=edge_index.device)
            # Returns `edge_index, keep_idx` to the caller for the next stage.
            return edge_index, keep_idx

        num_edges = edge_index.shape[1]
        if num_edges == 0:
            keep_idx = torch.zeros((0,), dtype=torch.long, device=edge_index.device)
            # Returns `edge_index, keep_idx` to the caller for the next stage.
            return edge_index, keep_idx

        max_degree = int(self.graph_max_edges_per_node)
        if max_degree <= 0:
            keep_idx = torch.arange(num_edges, dtype=torch.long, device=edge_index.device)
            # Returns `edge_index, keep_idx` to the caller for the next stage.
            return edge_index, keep_idx

        src = edge_index[0].detach().cpu().long()
        dst = edge_index[1].detach().cpu().long()
        distances = edge_distance.detach().cpu().float().numpy()

        # Builds the `incident` using the comprehension's filter and mapping rules.
        incident = [[] for _ in range(num_nodes)]
        for idx in range(num_edges):
            u = int(src[idx])
            v = int(dst[idx])
            # Checks `u < 0 or u >= num_nodes or v < 0 or v >= num_nodes` before entering this conditional branch.
            if u < 0 or u >= num_nodes or v < 0 or v >= num_nodes:
                continue
            incident[u].append(idx)
            incident[v].append(idx)

        # Stores the `keep_set` for use by the remaining steps in this flow.
        keep_set = set()
        for node in range(num_nodes):
            idxs = incident[node]
            if not idxs:
                # Handles the data or control-flow case represented by this statement.
                continue
            if len(idxs) <= max_degree:
                keep_set.update(idxs)
                continue

            # Stores the `idx_arr` for use by the remaining steps in this flow.
            idx_arr = np.asarray(idxs, dtype=np.int64)
            dist_arr = distances[idx_arr]
            # 线性选择前k，再做小范围稳定排序
            part = np.argpartition(dist_arr, kth=max_degree - 1)[:max_degree]
            chosen = idx_arr[part]
            # Selects the required item from `chosen` as the `chosen`.
            chosen = chosen[np.lexsort((chosen, distances[chosen]))]
            keep_set.update(chosen.tolist())

        if len(keep_set) == num_edges:
            keep_idx = torch.arange(num_edges, dtype=torch.long, device=edge_index.device)
            # Returns `edge_index, keep_idx` to the caller for the next stage.
            return edge_index, keep_idx

        if not keep_set:
            empty_index = torch.zeros((2, 0), dtype=edge_index.dtype, device=edge_index.device)
            keep_idx = torch.zeros((0,), dtype=torch.long, device=edge_index.device)
            # Returns `empty_index, keep_idx` to the caller for the next stage.
            return empty_index, keep_idx

        keep_idx = torch.tensor(sorted(keep_set), dtype=torch.long, device=edge_index.device)
        return edge_index[:, keep_idx], keep_idx
    
    # ========== 状态管理 ==========
    
    def export_state(self) -> Dict:
        """导出处理器状态"""
        # Returns `` to the caller for the next stage.
        return {
            'processor_type': 'GraphProcessor',
            'cached_topology': self.cached_topology,
            'first_process_done': self.first_process_done,
            # Completes the multi-line `collection` arguments or elements.
            'is_geometry_varying': self.is_geometry_varying,
            'local_feature_encoding': self.local_feature_encoding,
            'field_bc': list(self.field_bc),
            'local_feature_schema': self.local_feature_schema,
            # Completes the multi-line `collection` arguments or elements.
            'boundary_state_override_schema': self.boundary_state_override_schema,
        }
    
    def load_state(self, state: Dict) -> None:
        """加载处理器状态"""
        if not state:
            # Ends this function and returns control to its caller.
            return
        self.cached_topology = state.get('cached_topology')
        self.first_process_done = state.get('first_process_done', False)
        self.local_feature_schema = state.get('local_feature_schema')
        # Stores the `boundary_state_override_schema` on the instance so other methods can reuse it.
        self.boundary_state_override_schema = state.get('boundary_state_override_schema')
    
    # ========== 保存方法 ==========
    
    def save_to_pt(self, processed_data: Dict, save_path: str) -> None:
        """保存图数据为pt文件（x为PyG Data对象，y为Tensor）
        
        Args:
            processed_data: 处理后的数据字典
            save_path: 保存路径
        """
        from torch_geometric.data import Data as PyGData
        
        # 构造PyG Data对象作为x
        # 注意：x 包含了节点特征（坐标+工况参数），PyGData 的 x 应该是节点特征
        graph_data = PyGData(
            # Completes the multi-line `PyGData` arguments or elements.
            x=processed_data['x'],
            edge_index=processed_data['edge_index'],
            edge_attr=processed_data.get(
                'edge_attr',
                # Completes the multi-line `torch.zeros` arguments or elements.
                torch.zeros(
                    (processed_data['edge_index'].shape[1], 4),
                    dtype=torch.float32,
                ),
            # Completes the multi-line `processed_data.get` arguments or elements.
            ),
            field_location=processed_data.get('field_location', 'point')
        )
        graph_data.global_entity_id = torch.arange(
            # Completes the multi-line `torch.arange` arguments or elements.
            graph_data.x.shape[0],
            dtype=torch.long,
        )
        graph_data.sample_id = torch.tensor(
            # Completes the multi-line `collection` arguments or elements.
            [int(processed_data.get('sample_id', 0))],
            dtype=torch.long,
        )
        graph_data.num_full_graph_nodes = int(graph_data.x.shape[0])
        # Calls `self._attach_motion_graph_state` to perform the operation required at this stage.
        self._attach_motion_graph_state(
            graph=graph_data,
            meta=processed_data.get("meta", {}),
        )
        
        # 统一的 {x, y} 格式（x是PyG Data，y是物理场Tensor）
        # Initializes the output as a collection for values from this processing stage.
        output = {
            'x': graph_data,  # PyG Data对象，包含节点特征和拓扑信息
            'y': processed_data['y']  # 物理场数据
        }
        # Reads `meta` from the mapping into the `output['meta']` with its declared fallback.
        output['meta'] = processed_data.get('meta', {})
        for key in ('boundary_mask', 'boundary_entity_codes'):
            if processed_data.get(key) is not None:
                output[key] = processed_data[key]
        # Checks `'y_seed' in processed_data` before entering this conditional branch.
        if 'y_seed' in processed_data:
            output['y_seed'] = processed_data['y_seed']
        atomic_torch_save(output, save_path)
