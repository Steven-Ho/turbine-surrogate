import logging
import os
from contextlib import contextmanager
# Imports `Any, Dict, List` from `typing` for use in this module.
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.checkpoint import checkpoint
from .field_processor import FieldProcessor
# Imports `BaseData` from `data.base_data` for use in this module.
from ...data.base_data import BaseData
from ...training.losses import mask_huber, mask_mae, mask_mse
from ...utils.logger import Logger
from ...utils.torch_utils import atomic_torch_save, load_torch
# Imports `numpy` for reuse in this module.
import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy.spatial import Delaunay
# Imports `cgnsmesh` from `io_utils` for use in this module.
from ...io_utils import cgnsmesh


def _get_logger() -> logging.Logger:
    return Logger.get_first_configured_logger(
        ["train", "predict", "evaluate"],
        # Completes the multi-line `Logger.get_first_configured_logger` arguments or elements.
        fallback_name=__name__,
    )


_IMAGE_VOXEL_NON_PARAM_SAMPLE_KEYS = {
    'file',
    # Completes the multi-line `collection` arguments or elements.
    'file_path',
    'sample_order',
    'order',
    'sample_id',
    # Completes the multi-line `collection` arguments or elements.
    'temporal_steps',
    'total_time_steps',
    'time_axis',
}


# 图像体素化格式场处理器
class ImageVoxelProcessor(FieldProcessor):
    """图像体素化格式场处理器
    
    将3D场数据处理为体素化图像格式，适用于3D UNet等模型
    """
    
    def __init__(self, config: Dict):
        """初始化图像体素化处理器
        
        Args:
            config: 处理器配置参数
        """
        super().__init__(config)
        self.output_format = 'image_voxel'
        # Stores the `base_resolution` on the instance so other methods can reuse it.
        self.base_resolution = config['processor']['base_resolution']
        # 缓存掩码和相关信息
        self.cached_mask = None
        self.cached_voxel_centers = None
        self.cached_coord_ranges = None
        # Stores the `cached_x_grid` on the instance so other methods can reuse it.
        self.cached_x_grid = None
        self.cached_y_grid = None
        self.cached_z_grid = None
        self.cached_voxel_size = None
        # Stores the `_last_local_feature_meta` on the instance so other methods can reuse it.
        self._last_local_feature_meta: Dict[str, Any] = {}

    def export_state(self) -> Dict:
        """导出处理器状态，用于训练后保存与预测恢复。"""
        return {
            'processor_type': 'ImageVoxelProcessor',
            # Completes the multi-line `collection` arguments or elements.
            'cached_coord_ranges': self.cached_coord_ranges,
            'cached_voxel_centers': self.cached_voxel_centers,
            'cached_mask': self.cached_mask,
            'cached_x_grid': self.cached_x_grid,
            # Completes the multi-line `collection` arguments or elements.
            'cached_y_grid': self.cached_y_grid,
            'cached_z_grid': self.cached_z_grid,
            'cached_voxel_size': self.cached_voxel_size,
            'local_feature_encoding': self.local_feature_encoding,
            # Completes the multi-line `list` arguments or elements.
            'field_bc': list(self.field_bc),
            'local_feature_schema': self.local_feature_schema,
        }

    def infer_model_overrides(self, sample: Dict, config: Dict = None) -> Dict:
        # Validates `not isinstance(sample, dict)` and raises `ValueError` when the input cannot be processed safely.
        if not isinstance(sample, dict):
            raise ValueError("ImageVoxelProcessor.infer_model_overrides expects a dict sample payload")
        if 'x' not in sample or 'y' not in sample:
            raise ValueError("ImageVoxelProcessor.infer_model_overrides requires processed sample keys: x, y")

        # Selects the required item from `sample` as the `x`.
        x = sample['x']
        y = sample['y']
        if not hasattr(x, 'shape') or not hasattr(y, 'shape'):
            raise ValueError("ImageVoxelProcessor.infer_model_overrides expects sample['x'] and sample['y'] to be tensor-like objects with shape")
        # Validates `x.ndim < 2` and raises `ValueError` when the input cannot be processed safely.
        if x.ndim < 2:
            raise ValueError("ImageVoxelProcessor.infer_model_overrides expects sample['x'] to include channel and spatial dimensions")

        task_cfg = (config or self.config).get("task", {})
        mode = str(task_cfg.get("mode", self.mode)).lower()
        # Stores the `temporal_strategy` for use by the remaining steps in this flow.
        temporal_strategy = str(task_cfg.get("temporal_strategy", "direct")).lower()

        if mode == "temporal" and temporal_strategy == "direct":
            if y.ndim != 5:
                raise ValueError(
                    # Completes the multi-line `ValueError` arguments or elements.
                    f"ImageVoxelProcessor temporal direct 期望时序 y 形状为五维张量，当前维度为 {y.ndim}"
                )

            time_axis = int(sample.get("meta", {}).get("time_axis", 1))
            channel_axis = 1 if time_axis == 0 else 0
            # Derives the `out_channels` from the current values for downstream processing.
            out_channels = int(y.shape[channel_axis]) * int(y.shape[time_axis])
            return {
                'in_channels': int(x.shape[0]),
                'out_channels': out_channels,
                # Completes the multi-line `tuple` arguments or elements.
                'img_size': tuple(x.shape[1:]),
            }

        if mode == "temporal" and temporal_strategy == "autoregressive":
            if y.ndim != 5:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(
                    f"ImageVoxelProcessor autoregressive 期望时序 y 形状为五维张量，当前维度为 {y.ndim}"
                )

            output_length = int(task_cfg.get("output_length", 1))
            # Validates `output_length != 1` and raises `NotImplementedError` when the input cannot be processed safely.
            if output_length != 1:
                raise NotImplementedError(
                    "ImageVoxelProcessor 当前仅支持 output_length=1 的 autoregressive 训练视图。"
                )

            # Converts the `input_length` to `int` so later checks use a consistent type.
            input_length = int(task_cfg.get("input_length", 1))
            time_axis = int(sample.get("meta", {}).get("time_axis", 1))
            channel_axis = 1 if time_axis == 0 else 0
            out_channels = int(y.shape[channel_axis])
            # Returns `` to the caller for the next stage.
            return {
                'in_channels': int(x.shape[0]) + input_length * out_channels,
                'out_channels': out_channels,
                'img_size': tuple(x.shape[1:]),
            # Completes the multi-line `collection` arguments or elements.
            }

        out_channels = int(y.shape[1]) if y.ndim == 5 else int(y.shape[0])
        return {
            'in_channels': int(x.shape[0]),
            # Completes the multi-line `collection` arguments or elements.
            'out_channels': out_channels,
            'img_size': tuple(x.shape[1:]),
        }

    def _get_output_field_channels(self) -> int:
        # Stores the `field_names` for use by the remaining steps in this flow.
        field_names = (
            self.config.get("task", {}).get("effective_variables")
            or self.config.get("task", {}).get("variables", [])
        )
        # Validates `not field_names` and raises `ValueError` when the input cannot be processed safely.
        if not field_names:
            raise ValueError("ImageVoxelProcessor 无法确定输出场通道数，请检查 task.variables 配置")
        return int(len(field_names))

    def _reshape_temporal_direct_target(
        # Continues the surrounding statement or collection definition.
        self,
        y: torch.Tensor,
        time_axis: int,
    ) -> torch.Tensor:
        # Stores the `y_time_first` for use by the remaining steps in this flow.
        y_time_first = self._move_time_axis_to_front(y, time_axis)
        return y_time_first.reshape(
            y_time_first.shape[0] * y_time_first.shape[1],
            *y_time_first.shape[2:],
        # Completes the multi-line `y_time_first.reshape` arguments or elements.
        )

    def _restore_temporal_direct_output(self, values: Any) -> Any:
        field_channels = self._get_output_field_channels()

        if isinstance(values, torch.Tensor):
            # Returns early when `values.ndim < 4` makes the remaining work inapplicable.
            if values.ndim < 4:
                return values
            folded_channels = int(values.shape[1]) if values.ndim >= 5 else int(values.shape[0])
            if folded_channels % field_channels != 0:
                # Returns `values` to the caller for the next stage.
                return values
            time_steps = folded_channels // field_channels
            if values.ndim == 4:
                return values.reshape(time_steps, field_channels, *values.shape[1:])
            # Returns early when `values.ndim == 5` makes the remaining work inapplicable.
            if values.ndim == 5:
                return values.reshape(values.shape[0], time_steps, field_channels, *values.shape[2:])
            return values

        array = np.asarray(values)
        # Returns early when `array.ndim < 4` makes the remaining work inapplicable.
        if array.ndim < 4:
            return array
        folded_channels = int(array.shape[1]) if array.ndim >= 5 else int(array.shape[0])
        if folded_channels % field_channels != 0:
            # Returns `array` to the caller for the next stage.
            return array
        time_steps = folded_channels // field_channels
        if array.ndim == 4:
            return array.reshape(time_steps, field_channels, *array.shape[1:])
        # Returns early when `array.ndim == 5` makes the remaining work inapplicable.
        if array.ndim == 5:
            return array.reshape(array.shape[0], time_steps, field_channels, *array.shape[2:])
        return array

    def _canonicalize_temporal_direct_eval_tensor(
        # Continues the surrounding statement or collection definition.
        self,
        values: Any,
        channels: int,
    ) -> np.ndarray:
        """将时序 direct 评估张量统一整理为 (batch, time, channel, ...)。"""
        # Stores the `array` for use by the remaining steps in this flow.
        array = np.asarray(values)
        if array.ndim != 6:
            return array
        if int(array.shape[2]) == int(channels):
            # Returns `array` to the caller for the next stage.
            return array
        if int(array.shape[1]) == int(channels):
            return np.swapaxes(array, 1, 2)
        return array

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _resolve_temporal_layout(sample: Dict[str, Any], tensor: torch.Tensor) -> Tuple[int, int]:
        """根据样本元信息解析时序张量的时间轴与通道轴。"""
        if tensor.ndim != 5:
            raise ValueError(f"时序张量维度必须为 5，当前为 {tensor.ndim}")

        # Stores the `meta` for use by the remaining steps in this flow.
        meta = sample.get("meta", {}) if isinstance(sample, dict) else {}
        raw_time_axis = meta.get("time_axis", 1)
        time_axis = 1 if raw_time_axis is None else int(raw_time_axis)
        if time_axis not in (0, 1):
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(f"ImageVoxelProcessor 仅支持 time_axis 为 0 或 1，当前为 {time_axis}")
        channel_axis = 1 if time_axis == 0 else 0
        return time_axis, channel_axis

    @staticmethod
    # 提取单个物理通道，返回形状统一为 (T, ...)。
    def _select_temporal_channel(
        tensor: torch.Tensor,
        channel_index: int,
        channel_axis: int,
    # Continues the surrounding statement or collection definition.
    ) -> torch.Tensor:
        """提取单个物理通道，返回形状统一为 (T, ...)。"""
        if channel_axis == 0:
            return tensor[int(channel_index), ...]
        return tensor[:, int(channel_index), ...]

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _assign_temporal_channel(
        tensor: torch.Tensor,
        channel_index: int,
        # Continues the surrounding statement or collection definition.
        channel_axis: int,
        channel_values: torch.Tensor,
    ) -> None:
        """将归一化后的单通道数据回写到原张量。"""
        if channel_axis == 0:
            # Stores the `]` for use by the remaining steps in this flow.
            tensor[int(channel_index), ...] = channel_values
        else:
            tensor[:, int(channel_index), ...] = channel_values

    def build_prediction_adapter(
        # Continues the surrounding statement or collection definition.
        self,
        reader,
        config: Dict,
        is_geometry_varying: bool,
        # Continues the surrounding statement or collection definition.
        resolve_reference_file,
        logger=None,
    ):
        from ...prediction.adapters import ImageVoxelPredictAdapter

        # Returns `` to the caller for the next stage.
        return ImageVoxelPredictAdapter(
            processor=self,
            reader=reader,
            config=config,
            # Completes the multi-line `ImageVoxelPredictAdapter` arguments or elements.
            is_geometry_varying=is_geometry_varying,
            resolve_reference_file=resolve_reference_file,
            logger=logger,
        )

    # Defines `build_direct_view` for build direct view logic.
    def build_direct_view(
        self,
        sample: Dict[str, Any],
    ) -> Any:
        # Selects the required item from `sample` as the `x`.
        x = sample["x"]
        y = sample["y"]
        meta = sample.get("meta", {})
        if (
            # Completes the multi-line `str(self.config.get('task', {}).get('mode', self.mode)).lower` arguments or elements.
            str(self.config.get("task", {}).get("mode", self.mode)).lower() == "temporal"
            and y.ndim == 5
        ):
            time_axis = int(meta.get("time_axis", 1))
            # Returns `x, self._reshape_temporal_direct_target(y, time_axis)` to the caller for the next stage.
            return x, self._reshape_temporal_direct_target(y, time_axis)
        return x, y

    def supports_fast_predict_path(self, is_geometry_varying: bool) -> bool:
        return self.mode == 'steady'

    # 固定几何稳态预测时，ImageVoxelProcessor 支持仅依赖 reference 的快路径。
    def supports_reference_only_predict_path(self) -> bool:
        """固定几何稳态预测时，ImageVoxelProcessor 支持仅依赖 reference 的快路径。"""
        task_cfg = self.config.get("task", {})
        return (
            str(task_cfg.get("mode", self.mode)).lower() == "steady"
            # Completes the multi-line `bool` arguments or elements.
            and not bool(task_cfg.get("is_geometry_varying", self.is_geometry_varying))
        )

    def supports_fixed_geometry_temporal_predict_path(self) -> bool:
        """固定几何下，ImageVoxelProcessor 支持时序 autoregressive 预测快路径。"""
        task_cfg = self.config.get("task", {})
        # Returns `` to the caller for the next stage.
        return (
            str(task_cfg.get("mode", self.mode)).lower() == "temporal"
            and str(task_cfg.get("temporal_strategy", "direct")).lower() == "autoregressive"
            and not bool(task_cfg.get("is_geometry_varying", self.is_geometry_varying))
        # Continues the surrounding statement or collection definition.
        )

    def requires_fixed_geometry_temporal_reference(self) -> bool:
        input_length = int(self.config.get("task", {}).get("input_length", 1))
        return input_length <= 1

    # Defines `requires_fixed_geometry_predict_writeback_reference` for requires fixed geometry predict writeback reference logic.
    def requires_fixed_geometry_predict_writeback_reference(self) -> bool:
        task_cfg = self.config.get("task", {})
        if (
            str(task_cfg.get("mode", self.mode)).lower() == "temporal"
            # Completes the multi-line `str(task_cfg.get('temporal_strategy', 'direct')).lower` arguments or elements.
            and str(task_cfg.get("temporal_strategy", "direct")).lower() == "autoregressive"
            and int(task_cfg.get("input_length", 1)) > 1
        ):
            return False
        # Returns `True` to the caller for the next stage.
        return True

    def build_fast_predict_dataset(
        self,
        samples_csv: str,
        # Continues the surrounding statement or collection definition.
        processed_dir: str,
        config: Dict[str, Any],
        norm_params: Optional[Dict[str, Any]],
    ):
        # Imports `DataPipeline` from `data.data_pipeline` for use in this module.
        from ...data.data_pipeline import DataPipeline
        from ...data.sample_statistics import (
            calculate_cumulative_coord_ranges,
            extract_coord_ranges_from_sample,
        # Continues the surrounding statement or collection definition.
        )
        from ...dataset import build_dataset
        from ...factories.reader_factory import ReaderFactory
        from ...readers.sample_map import SampleMap
        # Imports `resolve_path_from_base` from `utils.path_utils` for use in this module.
        from ...utils.path_utils import resolve_path_from_base

        sample_map = SampleMap(samples_csv)
        sample_ids = sample_map.get_all_samples()
        if not sample_ids:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("样本表为空，无法进行 ImageVoxel 预测")

        os.makedirs(processed_dir, exist_ok=True)
        base_dir = os.path.dirname(os.path.abspath(samples_csv))
        reader = ReaderFactory.get_reader(config, flag="predict")

        # Defines `load_prediction_sample` for load prediction sample logic.
        def load_prediction_sample(file_path: str, sample_id: int) -> BaseData:
            return reader.read(
                file_path,
                sample_order=sample_id,
                # Completes the multi-line `reader.read` arguments or elements.
                allow_missing_fields=True,
            )

        resolved_samples = []
        for sample_id in sample_ids:
            # Stores the `sample_file` for use by the remaining steps in this flow.
            sample_file = sample_map.get_file_path(sample_id)
            if not sample_file:
                continue
            resolved_samples.append(
                # Completes the multi-line `collection` arguments or elements.
                (sample_id, resolve_path_from_base(base_dir, sample_file))
            )

        if not resolved_samples:
            raise ValueError("ImageVoxel 预测输入构建失败，样本表中没有可解析文件")

        # Checks `self.cached_coord_ranges is not None` before entering this conditional branch.
        if self.cached_coord_ranges is not None:
            coord_ranges = self.cached_coord_ranges
        elif not self.is_geometry_varying:
            first_sample = load_prediction_sample(resolved_samples[0][1], resolved_samples[0][0])
            # Stores the `coord_ranges` for use by the remaining steps in this flow.
            coord_ranges = extract_coord_ranges_from_sample(first_sample)
        else:
            coord_ranges = calculate_cumulative_coord_ranges(
                load_prediction_sample,
                # Completes the multi-line `calculate_cumulative_coord_ranges` arguments or elements.
                [item[1] for item in resolved_samples],
                [item[0] for item in resolved_samples],
            )

        if coord_ranges is None:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("ImageVoxel 预测阶段无法确定坐标范围")

        processed_paths: List[str] = []
        success_ids: List[int] = []
        for sample_id, abs_path in resolved_samples:
            # Stores the current data for use by the remaining steps in this flow.
            data = load_prediction_sample(abs_path, sample_id)
            processed_data = self.process_input_only(data, coord_ranges=coord_ranges)
            processed_path = os.path.join(processed_dir, f"sample_{sample_id}.pt")
            self.save_to_pt(processed_data, processed_path)
            # Adds the current value to `processed_paths` for later aggregation.
            processed_paths.append(processed_path)
            success_ids.append(sample_id)

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
        # Imports `extract_coord_ranges_from_sample` from `data.sample_statistics` for use in this module.
        from ...data.sample_statistics import extract_coord_ranges_from_sample
        from ...dataset import build_dataset
        from ...readers.sample_map import SampleMap
        from ...utils.path_utils import resolve_path_from_base

        # Validates `not self.supports_fixed_geometry_temporal_predict_path()` and raises `NotImplementedError` when the input cannot be processed safely.
        if not self.supports_fixed_geometry_temporal_predict_path():
            raise NotImplementedError(
                "ImageVoxelProcessor 固定几何时序预测快路径仅支持 temporal autoregressive 模式。"
            )

        # Stores the `sample_map` for use by the remaining steps in this flow.
        sample_map = SampleMap(samples_csv)
        sample_ids = sample_map.get_all_samples()
        if not sample_ids:
            raise ValueError("样本表为空，无法构造 ImageVoxel 固定几何时序预测数据集")

        # Calls `os.makedirs` to perform the operation required at this stage.
        os.makedirs(processed_dir, exist_ok=True)
        base_dir = os.path.dirname(os.path.abspath(samples_csv))
        input_length = int(self.config.get("task", {}).get("input_length", 1))
        sample_loader = FieldSampleLoader(
            # Completes the multi-line `FieldSampleLoader` arguments or elements.
            reader=reader,
            mode=self.mode,
            allow_missing_fields=input_length <= 1,
            temporal_history_steps=input_length if input_length > 1 else None,
        # Completes the multi-line `FieldSampleLoader` arguments or elements.
        )

        reference_data = None
        resolved_samples = []
        for sample_id in sample_ids:
            # Stores the `sample_file` for use by the remaining steps in this flow.
            sample_file = sample_map.get_file_path(sample_id)
            if input_length <= 1:
                resolved_samples.append((sample_id, sample_file))
                continue
            # Validates `not sample_file` and raises `ValueError` when the input cannot be processed safely.
            if not sample_file:
                raise ValueError("K>1 的 ImageVoxel autoregressive 预测必须提供样本原始文件")
            resolved_samples.append(
                (sample_id, resolve_path_from_base(base_dir, sample_file))
            # Completes the multi-line `resolved_samples.append` arguments or elements.
            )

        if not resolved_samples:
            raise ValueError("ImageVoxel 固定几何时序预测输入构建失败，样本表中没有可解析文件")

        if self.cached_coord_ranges is not None:
            # Stores the `coord_ranges` for use by the remaining steps in this flow.
            coord_ranges = self.cached_coord_ranges
        else:
            if input_length <= 1:
                if not reference_file:
                    # Raises `ValueError` to report why this path cannot continue.
                    raise ValueError("ImageVoxel 固定几何 K=1 时序预测需要 reference 文件")
                try:
                    reference_data = reader.read(
                        reference_file,
                        # Completes the multi-line `reader.read` arguments or elements.
                        sample_order=sample_ids[0],
                        allow_missing_fields=True,
                    )
                except TypeError:
                    # Stores the `reference_data` for use by the remaining steps in this flow.
                    reference_data = reader.read(
                        reference_file,
                        sample_order=sample_ids[0],
                    )
                # Stores the `first_sample` for use by the remaining steps in this flow.
                first_sample = reference_data
            else:
                first_sample = sample_loader.load(resolved_samples[0][1], resolved_samples[0][0])
            coord_ranges = extract_coord_ranges_from_sample(first_sample)

        # Validates `coord_ranges is None` and raises `ValueError` when the input cannot be processed safely.
        if coord_ranges is None:
            raise ValueError("ImageVoxel 固定几何时序预测阶段无法确定坐标范围")

        processed_paths: List[str] = []
        success_ids: List[int] = []
        # Visits each item in `resolved_samples` and applies the current processing step.
        for sample_id, sample_ref in resolved_samples:
            if input_length <= 1:
                if reference_data is None:
                    raise ValueError("ImageVoxel K=1 预测缺少 reference 数据")
                # Stores the `sample_info` for use by the remaining steps in this flow.
                sample_info = dict(sample_map.get_sample_info(sample_id) or {})
                sample_info["sample_order"] = int(sample_id)
                sample_info.setdefault("file_path", "")
                data = BaseData(
                    # Completes the multi-line `reference_data.get_data` arguments or elements.
                    data=reference_data.get_data(),
                    task_type=reference_data.get_task_type(),
                    mode="temporal",
                    point_or_cell=reference_data.get_point_or_cell(),
                    # Completes the multi-line `BaseData` arguments or elements.
                    sample_map=sample_info,
                )
                processed_data = self._build_zero_autoregressive_predict_seed_sample(
                    data=data,
                    # Completes the multi-line `self._build_zero_autoregressive_predict_seed_sample` arguments or elements.
                    coord_ranges=coord_ranges,
                )
            else:
                data = sample_loader.load(sample_ref, sample_id)
                # Stores the `processed_data` for use by the remaining steps in this flow.
                processed_data = self._build_autoregressive_predict_seed_sample(
                    data=data,
                    coord_ranges=coord_ranges,
                )
            # Stores the `processed_path` for use by the remaining steps in this flow.
            processed_path = os.path.join(processed_dir, f"sample_{sample_id}.pt")
            self.save_to_pt(processed_data, processed_path)
            processed_paths.append(processed_path)
            success_ids.append(sample_id)

        # Validates `not processed_paths` and raises `ValueError` when the input cannot be processed safely.
        if not processed_paths:
            raise ValueError("ImageVoxel 固定几何时序预测输入构建失败，没有可用样本")

        if norm_params:
            norm_type = norm_params.get(
                # Completes the multi-line `norm_params.get` arguments or elements.
                "norm_type",
                config.get("processor", {}).get("norm_type", "standardization"),
            )
            DataPipeline.normalize_data_with_existing_params(
                # Completes the multi-line `DataPipeline.normalize_data_with_existing_params` arguments or elements.
                processed_paths=processed_paths,
                norm_params=norm_params,
                norm_type=norm_type,
                processor=self,
                # Completes the multi-line `DataPipeline.normalize_data_with_existing_params` arguments or elements.
                normalize_y=True,
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

    def build_reference_only_predict_dataset(
        self,
        reader,
        # Continues the surrounding statement or collection definition.
        samples_csv: str,
        processed_dir: str,
        config: Dict[str, Any],
        norm_params: Optional[Dict[str, Any]],
        # Continues the surrounding statement or collection definition.
        reference_file: str,
    ):
        """构建仅依赖 reference 和样本参数的 ImageVoxel 稳态预测数据集。"""
        from ...data.data_pipeline import DataPipeline
        from ...data.sample_statistics import extract_coord_ranges_from_sample
        # Imports `build_dataset` from `dataset` for use in this module.
        from ...dataset import build_dataset
        from ...readers.sample_map import SampleMap

        if not self.supports_reference_only_predict_path():
            raise NotImplementedError(
                # Completes the multi-line `NotImplementedError` arguments or elements.
                "ImageVoxelProcessor reference-only 预测快路径仅支持 fixed-geometry steady 模式。"
            )

        sample_map = SampleMap(samples_csv)
        sample_ids = sample_map.get_all_samples()
        # Validates `not sample_ids` and raises `ValueError` when the input cannot be processed safely.
        if not sample_ids:
            raise ValueError("样本表为空，无法构建 ImageVoxel reference-only 预测数据集")

        os.makedirs(processed_dir, exist_ok=True)
        try:
            # Stores the `reference_data` for use by the remaining steps in this flow.
            reference_data = reader.read(
                reference_file,
                sample_order=sample_ids[0],
                allow_missing_fields=True,
            # Completes the multi-line `reader.read` arguments or elements.
            )
        except TypeError:
            reference_data = reader.read(
                reference_file,
                # Completes the multi-line `reader.read` arguments or elements.
                sample_order=sample_ids[0],
            )

        if self.cached_coord_ranges is not None:
            coord_ranges = self.cached_coord_ranges
        # Continues the surrounding statement or collection definition.
        else:
            coord_ranges = extract_coord_ranges_from_sample(reference_data)
        if coord_ranges is None:
            raise ValueError("ImageVoxel reference-only 预测阶段无法确定坐标范围")

        # Initializes the `processed_paths` as a collection for values from this processing stage.
        processed_paths: List[str] = []
        success_ids: List[int] = []

        for sample_id in sample_ids:
            sample_info = dict(sample_map.get_sample_info(sample_id) or {})
            # Converts the `sample_info['sample_order']` to `int` so later checks use a consistent type.
            sample_info["sample_order"] = int(sample_id)
            sample_info.setdefault("file_path", "")
            sample_data = BaseData(
                data=reference_data.get_data(),
                # Completes the multi-line `reference_data.get_task_type` arguments or elements.
                task_type=reference_data.get_task_type(),
                mode=reference_data.get_mode(),
                point_or_cell=reference_data.get_point_or_cell(),
                sample_map=sample_info,
            # Completes the multi-line `BaseData` arguments or elements.
            )
            processed_data = self.process_input_only(sample_data, coord_ranges=coord_ranges)
            processed_path = os.path.join(processed_dir, f"sample_{sample_id}.pt")
            self.save_to_pt(processed_data, processed_path)
            # Adds the current value to `processed_paths` for later aggregation.
            processed_paths.append(processed_path)
            success_ids.append(sample_id)

        if not processed_paths:
            raise ValueError("ImageVoxel reference-only 预测输入构建失败，没有可用样本")

        # Checks `norm_params` before entering this conditional branch.
        if norm_params:
            norm_type = norm_params.get(
                "norm_type",
                config.get("processor", {}).get("norm_type", "standardization"),
            # Completes the multi-line `norm_params.get` arguments or elements.
            )
            DataPipeline.normalize_data_predict(
                processed_paths=processed_paths,
                norm_params=norm_params,
                # Completes the multi-line `DataPipeline.normalize_data_predict` arguments or elements.
                norm_type=norm_type,
                processor=self,
                processor_type=type(self).__name__,
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

    def build_criterion(self, loss_name: str):
        # Stores the `normalized` for use by the remaining steps in this flow.
        normalized = str(loss_name).lower()
        self._resolved_loss_name = normalized if normalized in {"mse", "mae", "huber"} else "mse"
        if normalized == "huber":
            criterion = (
                # Completes the multi-line `mask_huber` arguments or elements.
                lambda outputs, targets, inputs: mask_huber(
                    outputs,
                    targets,
                    self._extract_loss_mask(inputs, outputs),
                # Completes the multi-line `mask_huber` arguments or elements.
                )
            )
            setattr(criterion, "_tfai_loss_name", "huber")
            return criterion, "mask_huber"
        # Checks `normalized == "mae"` before entering this conditional branch.
        if normalized == "mae":
            criterion = (
                lambda outputs, targets, inputs: mask_mae(
                    outputs,
                    # Completes the multi-line `mask_mae` arguments or elements.
                    targets,
                    self._extract_loss_mask(inputs, outputs),
                )
            )
            # Calls `setattr` to perform the operation required at this stage.
            setattr(criterion, "_tfai_loss_name", "mae")
            return criterion, "mask_mae"
        criterion = (
            lambda outputs, targets, inputs: mask_mse(
                # Completes the multi-line `mask_mse` arguments or elements.
                outputs,
                targets,
                self._extract_loss_mask(inputs, outputs),
            )
        # Continues the surrounding statement or collection definition.
        )
        setattr(criterion, "_tfai_loss_name", "mse")
        return criterion, "mask_mse"

    def _extract_loss_mask(
        # Continues the surrounding statement or collection definition.
        self,
        inputs: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Validates `not isinstance(inputs, torch.Tensor) or inputs.ndim != 5` and raises `ValueError` when the input cannot be processed safely.
        if not isinstance(inputs, torch.Tensor) or inputs.ndim != 5:
            raise ValueError(
                f"ImageVoxelProcessor 损失掩码期望输入形状为 [B, C, D, H, W]，当前为 {getattr(inputs, 'shape', None)}"
            )

        # Converts the `static_channels` to `int` so later checks use a consistent type.
        static_channels = int(inputs.shape[1])
        task_cfg = self.config.get("task", {})
        if (
            str(task_cfg.get("mode", self.mode)).lower() == "temporal"
            # Completes the multi-line `str(task_cfg.get('temporal_strategy', 'direct')).lower` arguments or elements.
            and str(task_cfg.get("temporal_strategy", "direct")).lower() == "autoregressive"
            and outputs is not None
            and isinstance(outputs, torch.Tensor)
            and outputs.ndim >= 2
        # Continues the surrounding statement or collection definition.
        ):
            input_length = int(task_cfg.get("input_length", 1))
            history_channels = input_length * int(outputs.shape[1])
            static_channels = max(int(inputs.shape[1]) - history_channels, 1)

        # Derives the `mask_channel` from the current values for downstream processing.
        mask_channel = static_channels - 1
        return inputs[:, mask_channel : mask_channel + 1, ...]

    def _reshape_targets_for_weighting(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 5:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(f"ImageVoxelProcessor 关键区域加权期望 5 维输出，当前为 {values.ndim}")

        task_cfg = self.config.get("task", {})
        if (
            str(task_cfg.get("mode", self.mode)).lower() == "temporal"
            # Completes the multi-line `str(task_cfg.get('temporal_strategy', 'direct')).lower` arguments or elements.
            and str(task_cfg.get("temporal_strategy", "direct")).lower() == "direct"
        ):
            field_channels = self._get_output_field_channels()
            folded_channels = int(values.shape[1])
            # Validates `folded_channels % field_channels != 0` and raises `ValueError` when the input cannot be processed safely.
            if folded_channels % field_channels != 0:
                raise ValueError(
                    f"ImageVoxelProcessor 无法还原 temporal direct 通道布局："
                    f" folded_channels={folded_channels}, field_channels={field_channels}"
                # Completes the multi-line `ValueError` arguments or elements.
                )
            time_steps = folded_channels // field_channels
            return values.reshape(values.shape[0], time_steps, field_channels, *values.shape[2:])

        return values.unsqueeze(1)

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _compute_spatial_gradient_magnitude(values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4:
            raise ValueError(f"空间梯度计算期望输入形状为 [B, D, H, W]，当前为 {tuple(values.shape)}")

        # Stores the `gradient` for use by the remaining steps in this flow.
        gradient = torch.zeros_like(values, dtype=values.dtype)
        gradient[:, 1:, :, :] += torch.abs(values[:, 1:, :, :] - values[:, :-1, :, :])
        gradient[:, :-1, :, :] += torch.abs(values[:, 1:, :, :] - values[:, :-1, :, :])
        gradient[:, :, 1:, :] += torch.abs(values[:, :, 1:, :] - values[:, :, :-1, :])
        # Stores the `gradient[:, :, :-1, :]` for use by the remaining steps in this flow.
        gradient[:, :, :-1, :] += torch.abs(values[:, :, 1:, :] - values[:, :, :-1, :])
        gradient[:, :, :, 1:] += torch.abs(values[:, :, :, 1:] - values[:, :, :, :-1])
        gradient[:, :, :, :-1] += torch.abs(values[:, :, :, 1:] - values[:, :, :, :-1])
        return gradient / 6.0

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _smooth_spatial_weight_map(weight_map: torch.Tensor) -> torch.Tensor:
        if weight_map.ndim != 5:
            raise ValueError(f"平滑权重图期望输入形状为 [B, 1, D, H, W]，当前为 {tuple(weight_map.shape)}")
        # Selects the required item from `weight_map.shape` as the `spatial_shape`.
        spatial_shape = weight_map.shape[-3:]
        if min(spatial_shape) < 3:
            return weight_map
        return torch.nn.functional.avg_pool3d(
            # Completes the multi-line `torch.nn.functional.avg_pool3d` arguments or elements.
            weight_map,
            kernel_size=3,
            stride=1,
            padding=1,
        # Completes the multi-line `torch.nn.functional.avg_pool3d` arguments or elements.
        )

    def build_loss_weight(
        self,
        inputs: Any,
        # Continues the surrounding statement or collection definition.
        targets: Any,
        outputs: Any = None,
    ) -> Optional[torch.Tensor]:
        if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
            # Raises `TypeError` to report why this path cannot continue.
            raise TypeError("ImageVoxelProcessor 关键区域加权需要张量形式的 inputs/targets")

        mask = self._extract_loss_mask(inputs, outputs=outputs).float()
        reshaped_targets = self._reshape_targets_for_weighting(targets.float())
        if reshaped_targets.ndim != 6:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(
                f"ImageVoxelProcessor 关键区域加权期望还原后目标形状为 [B, T, C, D, H, W]，当前为 {tuple(reshaped_targets.shape)}"
            )

        batch_size, time_steps, _, depth, height, width = reshaped_targets.shape
        # Initializes the `spatial_shape` as a collection for values from this processing stage.
        spatial_shape = (batch_size, time_steps, depth, height, width)
        valid_mask = mask.reshape(batch_size, 1, depth, height, width).expand(-1, time_steps, -1, -1, -1)

        value_scores: List[torch.Tensor] = []
        gradient_scores: List[torch.Tensor] = []
        # Visits each item in `range(reshaped_targets.shape[2])` and applies the current processing step.
        for variable_index in range(reshaped_targets.shape[2]):
            variable_values = reshaped_targets[:, :, variable_index, ...].reshape(-1, depth, height, width)
            flattened_mask = valid_mask.reshape(-1, depth, height, width)
            value_scores.append(
                # Completes the multi-line `self._build_quantile_score(values=variable_values, valid_mask=flattened_mask).reshape` arguments or elements.
                self._build_quantile_score(
                    values=variable_values,
                    valid_mask=flattened_mask,
                ).reshape(spatial_shape)
            # Completes the multi-line `value_scores.append` arguments or elements.
            )
            gradient_values = self._compute_spatial_gradient_magnitude(variable_values)
            gradient_scores.append(
                self._build_quantile_score(
                    # Completes the multi-line `self._build_quantile_score(values=gradient_values, valid_mask=flattened_mask).reshape` arguments or elements.
                    values=gradient_values,
                    valid_mask=flattened_mask,
                ).reshape(spatial_shape)
            )

        # Stores the `shared_weight` for use by the remaining steps in this flow.
        shared_weight = self._compose_shared_weight_map(
            value_scores=value_scores,
            gradient_scores=gradient_scores,
            valid_mask=valid_mask,
        # Completes the multi-line `self._compose_shared_weight_map` arguments or elements.
        )
        shared_weight = self._smooth_spatial_weight_map(
            shared_weight.reshape(-1, 1, depth, height, width)
        ).reshape(spatial_shape)
        # Stores the `shared_weight` for use by the remaining steps in this flow.
        shared_weight = self._normalize_shared_weight_map(shared_weight, valid_mask=valid_mask)

        if reshaped_targets.shape[1] == 1:
            return shared_weight[:, :1, ...]

        expanded_weight = shared_weight.unsqueeze(2).expand(-1, -1, reshaped_targets.shape[2], -1, -1, -1)
        # Returns `expanded_weight.reshape(batch_size, -1, depth, height, width)` to the caller for the next stage.
        return expanded_weight.reshape(batch_size, -1, depth, height, width)

    def build_processing_context(self, **kwargs) -> Dict:
        stage = kwargs.get('stage')
        file_paths = kwargs.get('file_paths', [])
        # Reads `sample_ids` from the mapping into the sample identifiers with its declared fallback.
        sample_ids = kwargs.get('sample_ids', [])
        sample_loader = kwargs.get('sample_loader')
        cumulative_coord_range_loader = kwargs.get('cumulative_coord_range_loader')
        extract_coord_ranges_from_sample = kwargs.get('extract_coord_ranges_from_sample')

        # Stores the `local_feature_context` for use by the remaining steps in this flow.
        local_feature_context = self._build_local_feature_processing_context(**kwargs)

        if stage != 'train':
            return {
                'coord_ranges': self.cached_coord_ranges,
                # Completes the multi-line `collection` arguments or elements.
                'extract_coord_ranges_from_sample': extract_coord_ranges_from_sample,
                **local_feature_context,
            }

        coord_ranges = None
        # Checks `self.cached_coord_ranges is not None` before entering this conditional branch.
        if self.cached_coord_ranges is not None:
            coord_ranges = self.cached_coord_ranges
            _get_logger().info("使用处理器状态中的缓存坐标范围")
        elif not self.is_geometry_varying:
            # Records the current execution state to make progress and failures traceable.
            _get_logger().info("当前为固定几何，使用首个有效样本的坐标范围")
            first_data = None
            for file_path, sample_id in zip(file_paths, sample_ids):
                try:
                    # Stores the `first_data` for use by the remaining steps in this flow.
                    first_data = sample_loader(file_path, sample_id)
                    coord_ranges = extract_coord_ranges_from_sample(first_data)
                    break
                except Exception as exc:
                    # Records the current execution state to make progress and failures traceable.
                    _get_logger().warning(
                        "固定几何样本 %s 坐标范围探测失败，继续尝试后续样本: %s",
                        sample_id,
                        str(exc),
                    # Completes the multi-line `_get_logger().warning` arguments or elements.
                    )
                finally:
                    if first_data is not None:
                        del first_data
                        # Stores the `first_data` for use by the remaining steps in this flow.
                        first_data = None
        else:
            _get_logger().info("开始统计所有样本的累计坐标范围")
            coord_ranges = cumulative_coord_range_loader(file_paths, sample_ids)

        # Validates `coord_ranges is None` and raises `ValueError` when the input cannot be processed safely.
        if coord_ranges is None:
            raise ValueError('Failed to determine coordinate ranges for ImageVoxelProcessor')

        _get_logger().info(
            "坐标范围统计结果: X=%s Y=%s Z=%s",
            # Completes the multi-line `_get_logger().info` arguments or elements.
            coord_ranges[0],
            coord_ranges[1],
            coord_ranges[2],
        )
        # Returns `{'coord_ranges': coord_ranges, **local_feature_context}` to the caller for the next stage.
        return {'coord_ranges': coord_ranges, **local_feature_context}

    def process_single_sample(self, data: BaseData, context: Dict = None) -> Dict:
        context = context or {}
        coord_ranges = context.get('coord_ranges')
        # Checks `coord_ranges is None` before entering this conditional branch.
        if coord_ranges is None:
            extractor = context.get('extract_coord_ranges_from_sample')
            if extractor is not None:
                coord_ranges = extractor(data)
        # Returns `self.process(data, coord_ranges)` to the caller for the next stage.
        return self.process(data, coord_ranges)

    def process_predict_sample(
        self,
        data: BaseData,
        # Continues the surrounding statement or collection definition.
        context: Dict = None,
    ) -> Dict:
        context = context or {}
        coord_ranges = context.get('coord_ranges')
        # Checks `coord_ranges is None` before entering this conditional branch.
        if coord_ranges is None:
            extractor = context.get('extract_coord_ranges_from_sample')
            if extractor is not None:
                coord_ranges = extractor(data)
        # Checks `self.temporal_strategy == "autoregressive"` before entering this conditional branch.
        if self.temporal_strategy == "autoregressive":
            input_length = int(self.config.get("task", {}).get("input_length", 1))
            if input_length <= 1:
                return self._build_zero_autoregressive_predict_seed_sample(
                    # Completes the multi-line `self._build_zero_autoregressive_predict_seed_sample` arguments or elements.
                    data=data,
                    coord_ranges=coord_ranges,
                )
            return self._build_autoregressive_predict_seed_sample(
                # Completes the multi-line `self._build_autoregressive_predict_seed_sample` arguments or elements.
                data=data,
                coord_ranges=coord_ranges,
            )
        return self.process_input_only(data, coord_ranges=coord_ranges)

    # Defines `_build_local_feature_voxel_channels` for  build local feature voxel channels logic.
    def _build_local_feature_voxel_channels(
        self,
        data_dict: Dict[str, Any],
        target_location: str,
        # Continues the surrounding statement or collection definition.
        coord_source: Any,
        voxel_centers: np.ndarray,
        voxel_size,
        mask: np.ndarray,
        # Continues the surrounding statement or collection definition.
        feature_start: int,
    ) -> List[torch.Tensor]:
        self._last_local_feature_meta = {}
        if not self.local_feature_encoding:
            # Returns `[]` to the caller for the next stage.
            return []

        local_features, feature_names, feature_schema = self._build_local_feature_matrix(
            data_dict=data_dict,
            target_location=target_location,
        # Completes the multi-line `self._build_local_feature_matrix` arguments or elements.
        )
        source_coords = np.asarray(coord_source, dtype=np.float32)
        if source_coords.ndim != 2 or source_coords.shape[1] != 3:
            raise ValueError(
                # Completes the multi-line `ValueError` arguments or elements.
                "ImageVoxelProcessor 局部边界条件特征映射需要 (N, 3) 坐标，"
                f"当前为 {tuple(source_coords.shape)}"
            )
        if local_features.shape[0] != source_coords.shape[0]:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(
                "ImageVoxelProcessor 局部边界条件特征数量与映射源数量不一致："
                f"features={local_features.shape[0]}, source={source_coords.shape[0]}"
            )

        # Stores the `(_, nearest_indices)` for use by the remaining steps in this flow.
        _, nearest_indices = cKDTree(source_coords).query(voxel_centers, k=1)
        flat_features = np.asarray(local_features[nearest_indices], dtype=np.float32)
        mask_flat = np.asarray(mask, dtype=bool).reshape(-1)
        if mask_flat.shape[0] == flat_features.shape[0]:
            # Stores the `flat_features[~mask_flat, :]` for use by the remaining steps in this flow.
            flat_features[~mask_flat, :] = 0.0

        feature_grids = flat_features.T.reshape((len(feature_names),) + tuple(voxel_size))
        self._last_local_feature_meta = self._build_local_feature_meta_payload(
            schema=feature_schema,
            # Completes the multi-line `self._build_local_feature_meta_payload` arguments or elements.
            feature_start=feature_start,
            feature_location="voxel",
        )
        return [
            # Completes the multi-line `torch.tensor` arguments or elements.
            torch.tensor(feature_grid, dtype=torch.float32)
            for feature_grid in feature_grids
        ]

    def process_input_only(self, data: BaseData, coord_ranges) -> Dict:
        # Stores the `data_dict` for use by the remaining steps in this flow.
        data_dict = data.get_data()
        cell_coordinates = data_dict.get('CellCoordinates')
        # 第一阶段仅使用主模板几何（通常为 t=0），时序坐标变化若存在也不参与输入网格更新。
        point_coordinates = data_dict['PointCoordinates']
        element_connectivity = data_dict['ElementConnectivity']
        # Reads `ElementTypeIds` from the mapping into the `element_type_ids`.
        element_type_ids = data_dict.get('ElementTypeIds')
        point_or_cell = str(data.get_point_or_cell()).lower()

        if coord_ranges is None:
            raise ValueError("coord_ranges 参数必须传入，无法为预测阶段构造体素输入")

        # Stores the `use_cached_params` for use by the remaining steps in this flow.
        use_cached_params = self.cached_voxel_centers is not None
        if use_cached_params:
            voxel_centers = self.cached_voxel_centers
            x_grid = self.cached_x_grid
            # Stores the `y_grid` for use by the remaining steps in this flow.
            y_grid = self.cached_y_grid
            z_grid = self.cached_z_grid
            voxel_size = self.cached_voxel_size
            if not self.is_geometry_varying and self.cached_mask is not None:
                # Stores the `mask` for use by the remaining steps in this flow.
                mask = self.cached_mask
            else:
                mask = _compute_voxel_mask(
                    voxel_centers,
                    # Completes the multi-line `_compute_voxel_mask` arguments or elements.
                    point_coordinates,
                    element_connectivity,
                    element_type_ids=element_type_ids,
                    mesh_dimension=self.config.get("task", {}).get("mesh_dimension"),
                )
        # Continues the surrounding statement or collection definition.
        else:
            lengths = [max_val - min_val for (min_val, max_val) in coord_ranges]
            max_length = max(lengths)
            scale = self.base_resolution / max_length
            # Builds the `voxel_size` using the comprehension's filter and mapping rules.
            voxel_size = [max(1, int(length * scale)) for length in lengths]

            x_coords = np.linspace(coord_ranges[0][0], coord_ranges[0][1], voxel_size[0])
            y_coords = np.linspace(coord_ranges[1][0], coord_ranges[1][1], voxel_size[1])
            z_coords = np.linspace(coord_ranges[2][0], coord_ranges[2][1], voxel_size[2])
            # Stores the `(x_grid, y_grid, z_grid)` for use by the remaining steps in this flow.
            x_grid, y_grid, z_grid = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
            voxel_centers = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T
            mask = _compute_voxel_mask(
                voxel_centers,
                # Completes the multi-line `_compute_voxel_mask` arguments or elements.
                point_coordinates,
                element_connectivity,
                element_type_ids=element_type_ids,
                    mesh_dimension=self.config.get("task", {}).get("mesh_dimension"),
            )

            # Stores the `cached_mask` on the instance so other methods can reuse it.
            self.cached_mask = mask
            self.cached_voxel_centers = voxel_centers
            self.cached_coord_ranges = coord_ranges
            self.cached_x_grid = x_grid
            # Stores the `cached_y_grid` on the instance so other methods can reuse it.
            self.cached_y_grid = y_grid
            self.cached_z_grid = z_grid
            self.cached_voxel_size = voxel_size

        x_channels = [
            # Completes the multi-line `torch.tensor` arguments or elements.
            torch.tensor(x_grid, dtype=torch.float32),
            torch.tensor(y_grid, dtype=torch.float32),
            torch.tensor(z_grid, dtype=torch.float32),
        ]

        # Stores the `sample_map` for use by the remaining steps in this flow.
        sample_map = data.get_sample_map()
        if isinstance(sample_map, dict):
            for key, value in sample_map.items():
                if key not in _IMAGE_VOXEL_NON_PARAM_SAMPLE_KEYS and isinstance(value, (int, float)):
                    # Adds the current value to `x_channels` for later aggregation.
                    x_channels.append(torch.full(voxel_size, value, dtype=torch.float32))

        coord_source = point_coordinates if point_or_cell == "point" else cell_coordinates
        x_channels.extend(
            self._build_local_feature_voxel_channels(
                # Completes the multi-line `self._build_local_feature_voxel_channels` arguments or elements.
                data_dict=data_dict,
                target_location=point_or_cell,
                coord_source=coord_source,
                voxel_centers=voxel_centers,
                # Completes the multi-line `self._build_local_feature_voxel_channels` arguments or elements.
                voxel_size=voxel_size,
                mask=mask,
                feature_start=len(x_channels),
            )
        # Completes the multi-line `x_channels.extend` arguments or elements.
        )

        mask_grid = mask.reshape(voxel_size)
        x_channels.append(torch.tensor(mask_grid, dtype=torch.float32))

        x = torch.stack(x_channels, dim=0)
        # Stores the `field_names` for use by the remaining steps in this flow.
        field_names = self._get_prediction_field_names()
        y = self._build_dummy_target(voxel_size, len(field_names))
        meta = self._build_processed_sample_meta(data=data, y=y, time_axis=None)
        meta = self._with_local_feature_meta(meta, **self._last_local_feature_meta)
        # Checks `str(data.get_mode()).lower() == "temporal"` before entering this conditional branch.
        if str(data.get_mode()).lower() == "temporal":
            meta = self._build_temporal_direct_predict_meta(meta)
        else:
            meta = self._build_steady_input_only_meta(meta)
        # Returns `` to the caller for the next stage.
        return {
            'x': x,
            'y': y,
            'voxel_centers': voxel_centers,
            # Completes the multi-line `collection` arguments or elements.
            'mask': mask,
            'field_names': field_names,
            'meta': meta,
        }

    # Defines `_get_prediction_field_names` for  get prediction field names logic.
    def _get_prediction_field_names(self) -> List[str]:
        return list(
            self.config.get('task', {}).get('effective_variables')
            or self.config.get('task', {}).get('variables', [])
        # Completes the multi-line `list` arguments or elements.
        )

    @staticmethod
    def _build_dummy_target(voxel_size, channels: int) -> torch.Tensor:
        if channels <= 0:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("ImageVoxel 预测阶段无法确定输出通道数")
        return torch.zeros((channels,) + tuple(voxel_size), dtype=torch.float32)

    def _create_filtered_samples_csv(
        self,
        # Continues the surrounding statement or collection definition.
        source_csv: str,
        output_csv: str,
        keep_ids: List[int],
    ) -> str:
        # Imports `csv` for reuse in this module.
        import csv
        from ...readers.sample_map import SampleMap

        keep_set = set(keep_ids)
        sample_map = SampleMap(source_csv)
        # Builds the `all_ids` using the comprehension's filter and mapping rules.
        all_ids = [sid for sid in sample_map.get_all_samples() if sid in keep_set]
        if not all_ids:
            raise ValueError("过滤后样本为空，无法生成有效样本表")

        first_info = sample_map.get_sample_info(all_ids[0]) or {}
        # Derives the `fieldnames` from the current values for downstream processing.
        fieldnames = ['order'] + [k for k in first_info.keys()]

        output_dir = os.path.dirname(output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Scopes resource use so the context manager can release it on exit.
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for sample_id in all_ids:
                # Stores the `info` for use by the remaining steps in this flow.
                info = sample_map.get_sample_info(sample_id) or {}
                row = {'order': sample_id}
                row.update(info)
                writer.writerow(row)

        # Returns `output_csv` to the caller for the next stage.
        return output_csv

    def save_processing_state(
        self,
        stage: str,
        # Continues the surrounding statement or collection definition.
        processed_dir: str,
        processed_paths: List[str],
    ) -> None:
        if stage != 'train':
            # Ends this function and returns control to its caller.
            return
        processor_state_path = os.path.join(processed_dir, 'processor_state.pt')
        processor_state = self.export_state()
        atomic_torch_save(processor_state, processor_state_path)
        # Records the current execution state to make progress and failures traceable.
        _get_logger().info("处理器状态已保存: %s", processor_state_path)

    def denormalize_predictions(self, predictions, norm_params: Dict = None):
        if norm_params is None or 'y' not in norm_params:
            return predictions
        if isinstance(predictions, dict):
            return {key: self.denormalize_predictions(value, norm_params)
                    for key, value in predictions.items()}
        denorm = np.array(predictions, copy=True)
        # Per-sample maps are [C,D,H,W], batches [N,C,D,H,W].
        # Temporal direct batches add a leading time dimension after N.
        channel_axis = 0 if denorm.ndim == 4 else 1
        if denorm.ndim == 6 and self.mode == 'temporal':
            channel_axis = 2
        if denorm.ndim < 2:
            return denorm
        stats = norm_params['y']
        method = norm_params['norm_type']
        for channel in range(denorm.shape[channel_axis]):
            selection = [slice(None)] * denorm.ndim
            selection[channel_axis] = channel
            index = tuple(selection)
            if method == 'standardization' and channel < len(stats.get('mean', [])):
                denorm[index] = denorm[index] * (stats['std'][channel] + 1e-12) + stats['mean'][channel]
            elif method == 'min_max' and channel < len(stats.get('min', [])):
                denorm[index] = denorm[index] * (stats['max'][channel] - stats['min'][channel] + 1e-12) + stats['min'][channel]
        return denorm

    # Defines `normalize_output_tensor` for normalize output tensor logic.
    def normalize_output_tensor(
        self,
        outputs: Any,
        norm_params: Optional[Dict[str, Any]],
        # Continues the surrounding statement or collection definition.
        inputs: Any = None,
    ) -> Any:
        if norm_params is None or 'y' not in norm_params or not isinstance(outputs, torch.Tensor):
            return outputs

        # Stores the `normalized` for use by the remaining steps in this flow.
        normalized = outputs.float().clone()
        y_stats = norm_params['y']
        channel_axis = 1 if normalized.ndim >= 5 else 0
        mask = None
        # Checks `isinstance(inputs, torch.Tensor)` before entering this conditional branch.
        if isinstance(inputs, torch.Tensor):
            mask_axis = 1 if inputs.ndim >= 5 else 0
            mask_channel = int(inputs.shape[mask_axis]) - 1
            if mask_axis == 0:
                # Stores the `mask` for use by the remaining steps in this flow.
                mask = inputs[mask_channel, ...] > 0.5
            else:
                mask = inputs[:, mask_channel, ...] > 0.5

        channel_count = int(normalized.shape[channel_axis])
        # Visits each item in `range(channel_count)` and applies the current processing step.
        for c in range(channel_count):
            stats = {
                'mean': [y_stats['mean'][c]],
                'std': [y_stats['std'][c]],
                # Completes the multi-line `collection` arguments or elements.
                'min': [y_stats['min'][c]],
                'max': [y_stats['max'][c]],
            }
            if channel_axis == 0:
                # Selects the required item from `normalized` as the `channel`.
                channel = normalized[c, ...]
            else:
                channel = normalized[:, c, ...]

            if mask is not None and torch.any(mask):
                # Stores the `values` for use by the remaining steps in this flow.
                values = channel[mask].reshape(-1, 1)
                channel[mask] = self._apply_channel_norm(
                    values,
                    stats,
                    # Completes the multi-line `self._apply_channel_norm(values, stats, norm_params['norm_type']).reshape` arguments or elements.
                    norm_params['norm_type'],
                ).reshape(-1)
            else:
                shape = channel.shape
                # Stores the `channel` for use by the remaining steps in this flow.
                channel = self._apply_channel_norm(
                    channel.reshape(-1, 1),
                    stats,
                    norm_params['norm_type'],
                # Completes the multi-line `self._apply_channel_norm(channel.reshape(-1, 1), stats, norm_params['norm_type']).reshape` arguments or elements.
                ).reshape(shape)

            if channel_axis == 0:
                normalized[c, ...] = channel
            else:
                # Stores the `]` for use by the remaining steps in this flow.
                normalized[:, c, ...] = channel
        return normalized

    def denormalize_output_tensor(
        self,
        # Continues the surrounding statement or collection definition.
        outputs: Any,
        norm_params: Optional[Dict[str, Any]],
        inputs: Any = None,
    ) -> Any:
        # Returns early when `` makes the remaining work inapplicable.
        if norm_params is None or 'y' not in norm_params or not isinstance(outputs, torch.Tensor):
            return outputs

        denormalized = outputs.float().clone()
        y_stats = norm_params['y']
        # Stores the `channel_axis` for use by the remaining steps in this flow.
        channel_axis = 1 if denormalized.ndim >= 5 else 0
        mask = None
        if isinstance(inputs, torch.Tensor):
            mask_axis = 1 if inputs.ndim >= 5 else 0
            # Derives the `mask_channel` from the current values for downstream processing.
            mask_channel = int(inputs.shape[mask_axis]) - 1
            if mask_axis == 0:
                mask = inputs[mask_channel, ...] > 0.5
            else:
                # Stores the `mask` for use by the remaining steps in this flow.
                mask = inputs[:, mask_channel, ...] > 0.5

        channel_count = int(denormalized.shape[channel_axis])
        for c in range(channel_count):
            stats = {
                # Completes the multi-line `collection` arguments or elements.
                'mean': [y_stats['mean'][c]],
                'std': [y_stats['std'][c]],
                'min': [y_stats['min'][c]],
                'max': [y_stats['max'][c]],
            # Completes the multi-line `collection` arguments or elements.
            }
            if channel_axis == 0:
                channel = denormalized[c, ...]
            else:
                # Selects the required item from `denormalized` as the `channel`.
                channel = denormalized[:, c, ...]

            if mask is not None and torch.any(mask):
                values = channel[mask].reshape(-1, 1)
                channel[mask] = self._apply_channel_denorm(
                    # Completes the multi-line `self._apply_channel_denorm(values, stats, norm_params['norm_type']).reshape` arguments or elements.
                    values,
                    stats,
                    norm_params['norm_type'],
                ).reshape(-1)
            # Continues the surrounding statement or collection definition.
            else:
                shape = channel.shape
                channel = self._apply_channel_denorm(
                    channel.reshape(-1, 1),
                    # Completes the multi-line `self._apply_channel_denorm(channel.reshape(-1, 1), stats, norm_params['norm_type']).reshape` arguments or elements.
                    stats,
                    norm_params['norm_type'],
                ).reshape(shape)

            if channel_axis == 0:
                # Stores the `]` for use by the remaining steps in this flow.
                denormalized[c, ...] = channel
            else:
                denormalized[:, c, ...] = channel
        return denormalized

    # Defines `compute_train_loss` for compute train loss logic.
    def compute_train_loss(
        self,
        criterion,
        outputs: Any,
        # Continues the surrounding statement or collection definition.
        targets: Any,
        inputs: Any,
    ) -> torch.Tensor:
        if self._is_critical_region_weighting_enabled():
            # Returns `` to the caller for the next stage.
            return super().compute_train_loss(
                criterion=criterion,
                outputs=outputs,
                targets=targets,
                # Completes the multi-line `super().compute_train_loss` arguments or elements.
                inputs=inputs,
            )
        return criterion(outputs, targets, inputs)

    def prepare_eval_payload(
        # Continues the surrounding statement or collection definition.
        self,
        inputs: Any,
        outputs: Any,
        targets: Any,
    # Continues the surrounding statement or collection definition.
    ) -> Any:
        if (
            self.mode == 'temporal'
            and self.config.get('task', {}).get('temporal_strategy', 'direct') == 'direct'
        # Continues the surrounding statement or collection definition.
        ):
            return (
                self._restore_temporal_direct_output(outputs),
                self._restore_temporal_direct_output(targets),
            # Completes the multi-line `collection` arguments or elements.
            )
        return super().prepare_eval_payload(inputs=inputs, outputs=outputs, targets=targets)

    def collect_prediction_batch(
        self,
        # Continues the surrounding statement or collection definition.
        inputs: Any,
        outputs: Any,
    ) -> Any:
        if (
            # Continues the surrounding statement or collection definition.
            self.mode == 'temporal'
            and self.config.get('task', {}).get('temporal_strategy', 'direct') == 'direct'
        ):
            return self._to_numpy(self._restore_temporal_direct_output(outputs))
        # Returns `super().collect_prediction_batch(inputs=inputs, outputs=outputs)` to the caller for the next stage.
        return super().collect_prediction_batch(inputs=inputs, outputs=outputs)

    def compute_eval_loss(
        self,
        criterion,
        # Continues the surrounding statement or collection definition.
        outputs: Any,
        targets: Any,
        inputs: Any,
    ) -> torch.Tensor:
        # Returns early when `self._is_critical_region_weighting_enabled()` makes the remaining work inapplicable.
        if self._is_critical_region_weighting_enabled():
            return super().compute_eval_loss(
                criterion=criterion,
                outputs=outputs,
                # Completes the multi-line `super().compute_eval_loss` arguments or elements.
                targets=targets,
                inputs=inputs,
            )
        return criterion(outputs, targets, inputs)

    # Defines `collect_eval_batch` for collect eval batch logic.
    def collect_eval_batch(
        self,
        inputs: Any,
        outputs: Any,
        # Continues the surrounding statement or collection definition.
        targets: Any,
    ) -> Dict[str, Any]:
        payload = super().collect_eval_batch(inputs, outputs, targets)
        payload['mask'] = self._to_numpy(self._extract_loss_mask(inputs, outputs if isinstance(outputs, torch.Tensor) else None))
        # Returns `payload` to the caller for the next stage.
        return payload

    def merge_eval_batches(
        self,
        batch_payloads: List[Dict[str, Any]],
        # Completes the multi-line `collection` arguments or elements.
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        predictions = [np.asarray(payload['predictions']) for payload in batch_payloads]
        targets = [np.asarray(payload['targets']) for payload in batch_payloads]
        # Builds the `masks` using the comprehension's filter and mapping rules.
        masks = [np.asarray(payload['mask']) for payload in batch_payloads]

        if predictions:
            raw_predictions = np.concatenate(predictions, axis=0)
            raw_targets = np.concatenate(targets, axis=0)
            # Stores the `raw_masks` for use by the remaining steps in this flow.
            raw_masks = np.concatenate(masks, axis=0)
        else:
            raw_predictions = np.array([])
            raw_targets = np.array([])
            # Stores the `raw_masks` for use by the remaining steps in this flow.
            raw_masks = np.array([])

        norm_params = self._resolve_eval_norm_params(config)
        denormalized_predictions = self.denormalize_predictions(raw_predictions, norm_params)
        denormalized_targets = self.denormalize_predictions(raw_targets, norm_params)
        # Stores the `sample_predictions` for use by the remaining steps in this flow.
        sample_predictions = self._denormalize_sample_output_map(
            self._merge_sample_output_maps(
                batch_payloads=batch_payloads,
                payload_key='sample_predictions',
                # Completes the multi-line `self._merge_sample_output_maps` arguments or elements.
                fallback_key='predictions',
            ),
            config,
        )
        # Stores the `sample_targets` for use by the remaining steps in this flow.
        sample_targets = self._denormalize_sample_output_map(
            self._merge_sample_output_maps(
                batch_payloads=batch_payloads,
                payload_key='sample_targets',
                # Completes the multi-line `self._merge_sample_output_maps` arguments or elements.
                fallback_key='targets',
            ),
            config,
        )

        # Initializes the `scatter_by_var` as a collection for values from this processing stage.
        scatter_by_var: Dict[str, Dict[str, np.ndarray]] = {}
        overall_pred_parts = []
        overall_tgt_parts = []

        if (
            # Continues the surrounding statement or collection definition.
            denormalized_predictions.size > 0
            and raw_masks.size > 0
            and denormalized_predictions.ndim == 6
            and self.mode == 'temporal'
        # Continues the surrounding statement or collection definition.
        ):
            var_names = (
                config.get('task', {}).get('effective_variables')
                or config.get('task', {}).get('variables', [])
            # Continues the surrounding statement or collection definition.
            )
            expected_channels = int(len(var_names))
            if (
                expected_channels > 0
                # Completes the multi-line `config.get('task', {}).get` arguments or elements.
                and config.get('task', {}).get('temporal_strategy', 'direct') == 'direct'
            ):
                denormalized_predictions = self._canonicalize_temporal_direct_eval_tensor(
                    denormalized_predictions,
                    # Completes the multi-line `self._canonicalize_temporal_direct_eval_tensor` arguments or elements.
                    channels=expected_channels,
                )
                denormalized_targets = self._canonicalize_temporal_direct_eval_tensor(
                    denormalized_targets,
                    # Completes the multi-line `self._canonicalize_temporal_direct_eval_tensor` arguments or elements.
                    channels=expected_channels,
                )

            channels = denormalized_predictions.shape[2]
            if len(var_names) != channels:
                # Builds the `var_names` using the comprehension's filter and mapping rules.
                var_names = [f'field_{i}' for i in range(channels)]

            valid_mask = raw_masks[:, 0] > 0
            for c, var_name in enumerate(var_names):
                channel_pred = np.asarray(denormalized_predictions[:, :, c, ...])
                # Stores the `channel_tgt` for use by the remaining steps in this flow.
                channel_tgt = np.asarray(denormalized_targets[:, :, c, ...])
                broadcast_mask = np.broadcast_to(
                    valid_mask[:, np.newaxis, ...],
                    channel_pred.shape,
                # Completes the multi-line `np.broadcast_to` arguments or elements.
                )
                var_pred = np.asarray(channel_pred[broadcast_mask]).reshape(-1)
                var_tgt = np.asarray(channel_tgt[broadcast_mask]).reshape(-1)
                scatter_by_var[var_name] = {
                    # Completes the multi-line `collection` arguments or elements.
                    'predictions': var_pred,
                    'targets': var_tgt,
                }
                overall_pred_parts.append(var_pred)
                # Adds the current value to `overall_tgt_parts` for later aggregation.
                overall_tgt_parts.append(var_tgt)

            overall_predictions = (
                np.concatenate(overall_pred_parts, axis=0)
                if overall_pred_parts
                # Completes the multi-line `np.array` arguments or elements.
                else np.array([])
            )
            overall_targets = (
                np.concatenate(overall_tgt_parts, axis=0)
                # Continues the surrounding statement or collection definition.
                if overall_tgt_parts
                else np.array([])
            )

            return {
                # Completes the multi-line `collection` arguments or elements.
                'raw_predictions': raw_predictions,
                'raw_targets': raw_targets,
                'denormalized_predictions': denormalized_predictions,
                'denormalized_targets': denormalized_targets,
                # Completes the multi-line `collection` arguments or elements.
                'overall_predictions': overall_predictions,
                'overall_targets': overall_targets,
                'scatter_by_var': scatter_by_var,
                'sample_predictions': sample_predictions,
                # Completes the multi-line `collection` arguments or elements.
                'sample_targets': sample_targets,
            }

        if denormalized_predictions.size > 0 and raw_masks.size > 0:
            var_names = (
                # Completes the multi-line `config.get('task', {}).get` arguments or elements.
                config.get('task', {}).get('effective_variables')
                or config.get('task', {}).get('variables', [])
            )
            channels = denormalized_predictions.shape[1]
            # Checks `len(var_names) != channels` before entering this conditional branch.
            if len(var_names) != channels:
                var_names = [f'field_{i}' for i in range(channels)]

            valid_mask = raw_masks[:, 0] > 0
            for c, var_name in enumerate(var_names):
                # Stores the `var_pred` for use by the remaining steps in this flow.
                var_pred = np.asarray(denormalized_predictions[:, c][valid_mask]).reshape(-1)
                var_tgt = np.asarray(denormalized_targets[:, c][valid_mask]).reshape(-1)
                scatter_by_var[var_name] = {
                    'predictions': var_pred,
                    # Completes the multi-line `collection` arguments or elements.
                    'targets': var_tgt,
                }
                overall_pred_parts.append(var_pred)
                overall_tgt_parts.append(var_tgt)

        # Stores the `overall_predictions` for use by the remaining steps in this flow.
        overall_predictions = (
            np.concatenate(overall_pred_parts, axis=0)
            if overall_pred_parts
            else np.array([])
        # Continues the surrounding statement or collection definition.
        )
        overall_targets = (
            np.concatenate(overall_tgt_parts, axis=0)
            if overall_tgt_parts
            # Completes the multi-line `np.array` arguments or elements.
            else np.array([])
        )

        return {
            'raw_predictions': raw_predictions,
            # Completes the multi-line `collection` arguments or elements.
            'raw_targets': raw_targets,
            'denormalized_predictions': denormalized_predictions,
            'denormalized_targets': denormalized_targets,
            'overall_predictions': overall_predictions,
            # Completes the multi-line `collection` arguments or elements.
            'overall_targets': overall_targets,
            'scatter_by_var': scatter_by_var,
            'sample_predictions': sample_predictions,
            'sample_targets': sample_targets,
        # Completes the multi-line `collection` arguments or elements.
        }

    def compute_norm_params(
        self,
        train_samples: List[Dict],
        # Continues the surrounding statement or collection definition.
        norm_type: str,
        save_path: str = None,
    ) -> Dict:
        if not train_samples:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("ImageVoxelProcessor 训练样本为空，无法计算归一化参数")

        coord_values = []
        feature_values = []
        y_channel_values = None
        # Stores the `delta_channel_values` for use by the remaining steps in this flow.
        delta_channel_values = None
        mask_channel = None
        first_feature_channels = None
        first_local_indices = []

        # Visits each item in `train_samples` and applies the current processing step.
        for sample in train_samples:
            x = sample['x'].float()
            y = sample.get('y')
            if x.ndim != 4:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(f"ImageVoxelProcessor 的 x 维度应为4，当前为{x.ndim}")

            if x.shape[0] < 4:
                raise ValueError("ImageVoxelProcessor 的 x 至少应包含3个坐标通道和1个mask通道")

            mask_channel = x.shape[0] - 1
            # Stores the `mask` for use by the remaining steps in this flow.
            mask = x[mask_channel] > 0.5
            categorical_indices = {
                int(index)
                for index in sample.get("meta", {}).get("categorical_bc_indices", [])
            # Continues the surrounding statement or collection definition.
            }
            feature_channels = [
                index
                for index in range(3, int(mask_channel))
                # Continues the surrounding statement or collection definition.
                if index not in categorical_indices
            ]
            if first_feature_channels is None:
                first_feature_channels = list(feature_channels)
                # Stores the `first_local_indices` for use by the remaining steps in this flow.
                first_local_indices = sorted(categorical_indices)

            coords = x[:3].reshape(3, -1).T
            coord_values.append(coords)

            if mask.any() and feature_channels:
                # Stores the `params` for use by the remaining steps in this flow.
                params = x[feature_channels, ...][:, mask].T
                if params.numel() > 0:
                    feature_values.append(params)

            if y is not None:
                # Stores the `y` for use by the remaining steps in this flow.
                y = y.float()
                if y_channel_values is None:
                    if y.ndim == 5:
                        _, channel_dim = self._resolve_temporal_layout(sample, y)
                    # Continues the surrounding statement or collection definition.
                    else:
                        channel_dim = 0
                    num_channels = int(y.shape[channel_dim])
                    y_channel_values = [[] for _ in range(num_channels)]
                    # Checks `self.uses_autoregressive_increment()` before entering this conditional branch.
                    if self.uses_autoregressive_increment():
                        delta_channel_values = [[] for _ in range(num_channels)]

                if y.ndim == 5:
                    for c in range(num_channels):
                        # Stores the `channel` for use by the remaining steps in this flow.
                        channel = self._select_temporal_channel(y, c, channel_dim)
                        values = channel[:, mask].reshape(-1)
                        if values.numel() > 0:
                            y_channel_values[c].append(values)
                    # Checks `self.uses_autoregressive_increment()` before entering this conditional branch.
                    if self.uses_autoregressive_increment():
                        time_axis, _ = self._resolve_temporal_layout(sample, y)
                        sequence = torch.movedim(y, time_axis, 0)
                        if sequence.shape[0] >= 2:
                            # Visits each item in `range(num_channels)` and applies the current processing step.
                            for c in range(num_channels):
                                values = (sequence[1:, c, ...] - sequence[:-1, c, ...])[:, mask].reshape(-1)
                                if values.numel() > 0:
                                    delta_channel_values[c].append(values)
                # Checks `y.ndim == 4` before entering this conditional branch.
                elif y.ndim == 4:
                    for c in range(num_channels):
                        channel = y[c, ...]
                        values = channel[mask].reshape(-1)
                        # Checks `values.numel() > 0` before entering this conditional branch.
                        if values.numel() > 0:
                            y_channel_values[c].append(values)
                else:
                    raise ValueError(f"ImageVoxelProcessor 的 y 维度应为4或5，当前为{y.ndim}")

        # Initializes the normalization statistics as a collection for values from this processing stage.
        norm_params = {
            'processor_type': 'ImageVoxelProcessor',
            'norm_type': norm_type,
            'geometry': self._compute_geometry_params(torch.cat(coord_values, dim=0)),
            # Completes the multi-line `collection` arguments or elements.
            'x': {
                'feature_channels': list(first_feature_channels or []),
                'categorical_bc_indices': list(first_local_indices),
                'continuous_bc_indices': [
                    # Completes the multi-line `int` arguments or elements.
                    int(index)
                    for index in train_samples[0].get("meta", {}).get("continuous_bc_indices", [])
                ],
                'mask_channel': mask_channel,
            # Completes the multi-line `collection` arguments or elements.
            },
            'y': {},
            'meta': {
                'coord_channels': [0, 1, 2],
                # Completes the multi-line `collection` arguments or elements.
                'mask_channel': mask_channel,
            },
        }

        if feature_values:
            # Stores the `feature_stats` for use by the remaining steps in this flow.
            feature_stats = self._compute_channel_stats(torch.cat(feature_values, dim=0))
            norm_params['x'].update(feature_stats)

        if y_channel_values:
            y_stats = self._compute_channel_stats(
                # Completes the multi-line `torch.stack` arguments or elements.
                torch.stack([torch.cat(values, dim=0) for values in y_channel_values], dim=1)
            )
            norm_params['y'] = y_stats
        if self.uses_autoregressive_increment():
            # Validates `` and raises `ValueError` when the input cannot be processed safely.
            if not delta_channel_values or any(not values for values in delta_channel_values):
                raise ValueError("增量自回归需要至少一条含两个时间步的训练序列")
            norm_params['autoregressive_delta'] = self._finalize_delta_stats(
                self._compute_channel_stats(
                    # Completes the multi-line `torch.stack` arguments or elements.
                    torch.stack([torch.cat(values, dim=0) for values in delta_channel_values], dim=1)
                )
            )

        return norm_params

    # Defines `compute_norm_params_from_paths` for compute norm params from paths logic.
    def compute_norm_params_from_paths(
        self,
        train_processed_paths: List[str],
        norm_type: str,
        # Continues the surrounding statement or collection definition.
        save_path: str = None,
    ) -> Dict:
        if not train_processed_paths:
            raise ValueError("ImageVoxelProcessor training sample paths are empty")

        # Stores the `geometry_stats` for use by the remaining steps in this flow.
        geometry_stats = None
        feature_stats = None
        y_stats = None
        delta_stats = None
        # Stores the `mask_channel` for use by the remaining steps in this flow.
        mask_channel = None
        first_feature_channels = None
        first_local_indices = []

        for processed_path in train_processed_paths:
            # Stores the `sample` for use by the remaining steps in this flow.
            sample = load_torch(processed_path, weights_only=False)
            x = sample["x"].float()
            y = sample.get("y")
            if x.ndim != 4:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(f"ImageVoxelProcessor 鐨?x 缁村害搴斾负4锛屽綋鍓嶄负{x.ndim}")
            if x.shape[0] < 4:
                raise ValueError("ImageVoxelProcessor 鐨?x 鑷冲皯搴斿寘鍚?涓潗鏍囬€氶亾鍜?涓猰ask閫氶亾")

            mask_channel = x.shape[0] - 1
            # Stores the `mask` for use by the remaining steps in this flow.
            mask = x[mask_channel] > 0.5
            categorical_indices = {
                int(index)
                for index in sample.get("meta", {}).get("categorical_bc_indices", [])
            # Continues the surrounding statement or collection definition.
            }
            feature_channels = [
                index
                for index in range(3, int(mask_channel))
                # Continues the surrounding statement or collection definition.
                if index not in categorical_indices
            ]
            if first_feature_channels is None:
                first_feature_channels = list(feature_channels)
                # Stores the `first_local_indices` for use by the remaining steps in this flow.
                first_local_indices = sorted(categorical_indices)

            coords = x[:3].reshape(3, -1).T
            geometry_stats = self._update_running_geometry_stats(geometry_stats, coords)

            if mask.any() and feature_channels:
                # Stores the `params` for use by the remaining steps in this flow.
                params = x[feature_channels, ...][:, mask].T
                if params.numel() > 0:
                    feature_stats = self._update_running_channel_stats(feature_stats, params)

            if y is not None:
                # Stores the `y` for use by the remaining steps in this flow.
                y = y.float()
                if y.ndim == 5:
                    _, channel_dim = self._resolve_temporal_layout(sample, y)
                    num_channels = int(y.shape[channel_dim])
                    # Initializes the `channel_values` as a collection for values from this processing stage.
                    channel_values = []
                    for c in range(num_channels):
                        channel = self._select_temporal_channel(y, c, channel_dim)
                        values = channel[:, mask].reshape(-1, 1)
                        # Adds the current value to `channel_values` for later aggregation.
                        channel_values.append(values)
                    if channel_values:
                        merged = torch.cat(channel_values, dim=1)
                        y_stats = self._update_running_channel_stats(y_stats, merged)
                    # Checks `self.uses_autoregressive_increment()` before entering this conditional branch.
                    if self.uses_autoregressive_increment():
                        time_axis, _ = self._resolve_temporal_layout(sample, y)
                        sequence = torch.movedim(y, time_axis, 0)
                        if sequence.shape[0] >= 2:
                            # Builds the `delta_values` using the comprehension's filter and mapping rules.
                            delta_values = [
                                (sequence[1:, c, ...] - sequence[:-1, c, ...])[:, mask].reshape(-1, 1)
                                for c in range(num_channels)
                            ]
                            # Stores the `delta_stats` for use by the remaining steps in this flow.
                            delta_stats = self._update_running_channel_stats(
                                delta_stats,
                                torch.cat(delta_values, dim=1),
                            )
                # Checks `y.ndim == 4` before entering this conditional branch.
                elif y.ndim == 4:
                    num_channels = int(y.shape[0])
                    channel_values = []
                    for c in range(num_channels):
                        # Stores the `values` for use by the remaining steps in this flow.
                        values = y[c, ...][mask].reshape(-1, 1)
                        channel_values.append(values)
                    if channel_values:
                        merged = torch.cat(channel_values, dim=1)
                        # Stores the `y_stats` for use by the remaining steps in this flow.
                        y_stats = self._update_running_channel_stats(y_stats, merged)
                else:
                    raise ValueError(f"ImageVoxelProcessor 鐨?y 缁村害搴斾负4鎴?锛屽綋鍓嶄负{y.ndim}")

        if mask_channel is None:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("ImageVoxelProcessor could not infer mask_channel from processed samples")

        norm_params = {
            "processor_type": "ImageVoxelProcessor",
            "norm_type": norm_type,
            # Completes the multi-line `self._finalize_running_geometry_stats` arguments or elements.
            "geometry": self._finalize_running_geometry_stats(geometry_stats),
            "x": {
                "feature_channels": list(first_feature_channels or []),
                "categorical_bc_indices": list(first_local_indices),
                # Completes the multi-line `collection` arguments or elements.
                "continuous_bc_indices": [
                    int(index)
                    for index in sample.get("meta", {}).get("continuous_bc_indices", [])
                ],
                # Completes the multi-line `collection` arguments or elements.
                "mask_channel": mask_channel,
            },
            "y": {},
            "meta": {
                # Completes the multi-line `collection` arguments or elements.
                "coord_channels": [0, 1, 2],
                "mask_channel": mask_channel,
            },
        }

        # Stores the `feature_summary` for use by the remaining steps in this flow.
        feature_summary = self._finalize_running_channel_stats(feature_stats)
        if feature_summary:
            norm_params["x"].update(feature_summary)

        y_summary = self._finalize_running_channel_stats(y_stats)
        # Checks `y_summary` before entering this conditional branch.
        if y_summary:
            norm_params["y"] = y_summary
        if self.uses_autoregressive_increment():
            delta_summary = self._finalize_running_channel_stats(delta_stats)
            # Validates `not delta_summary` and raises `ValueError` when the input cannot be processed safely.
            if not delta_summary:
                raise ValueError("增量自回归需要至少一条含两个时间步的训练序列")
            norm_params["autoregressive_delta"] = self._finalize_delta_stats(delta_summary)

        return norm_params

    # Defines `normalize_processed_sample` for normalize processed sample logic.
    def normalize_processed_sample(
        self,
        sample: Dict,
        norm_params: Dict,
        # Continues the surrounding statement or collection definition.
        norm_type: str,
        normalize_y: bool = True,
    ) -> Dict:
        normalized = dict(sample)
        # Stores the `x` for use by the remaining steps in this flow.
        x = sample['x'].float().clone()
        geometry = norm_params['geometry']
        x_cfg = norm_params.get('x', {})
        mask_channel = int(x_cfg.get('mask_channel', x.shape[0] - 1))

        # Calls `self._validate_normalization_input_schema` to perform the operation required at this stage.
        self._validate_normalization_input_schema(x, x_cfg, mask_channel)

        coord_shape = x[:3].shape
        coords = x[:3].reshape(3, -1).T
        coords = self._normalize_coordinates(coords, geometry)
        # Stores the `x[:3]` for use by the remaining steps in this flow.
        x[:3] = coords.T.reshape(coord_shape)

        feature_channels = x_cfg.get('feature_channels', [])
        if feature_channels and all(k in x_cfg for k in ['mean', 'std', 'min', 'max']):
            mask = x[mask_channel] > 0.5
            # Selects the required item from `x` as the `features`.
            features = x[feature_channels, ...]
            if mask.any():
                features_masked = features[:, mask].T
                features_norm = self._apply_channel_norm(features_masked, x_cfg, norm_type)
                # Stores the `features[:, mask]` for use by the remaining steps in this flow.
                features[:, mask] = features_norm.T
                features[:, ~mask] = 0.0
                x[feature_channels, ...] = features

        normalized['x'] = x

        if normalize_y and 'y' in sample and norm_params.get('y'):
            # Stores the `y` for use by the remaining steps in this flow.
            y = sample['y'].float().clone()
            y_stats = norm_params['y']
            mask = x[mask_channel] > 0.5
            if y.ndim == 5:
                # Stores the `(_, channel_axis)` for use by the remaining steps in this flow.
                _, channel_axis = self._resolve_temporal_layout(sample, y)
                for c in range(int(y.shape[channel_axis])):
                    channel = self._select_temporal_channel(y, c, channel_axis)
                    target_shape = channel[:, mask].shape
                    # Stores the `values` for use by the remaining steps in this flow.
                    values = channel[:, mask].reshape(-1, 1)
                    normed = self._apply_channel_norm(
                        values,
                        {
                            # Completes the multi-line `collection` arguments or elements.
                            'mean': [y_stats['mean'][c]],
                            'std': [y_stats['std'][c]],
                            'min': [y_stats['min'][c]],
                            'max': [y_stats['max'][c]],
                        # Completes the multi-line `collection` arguments or elements.
                        },
                        norm_type,
                    ).reshape(target_shape)
                    channel[:, mask] = normed
                    # Calls `self._assign_temporal_channel` to perform the operation required at this stage.
                    self._assign_temporal_channel(y, c, channel_axis, channel)
            elif y.ndim == 4:
                for c in range(y.shape[0]):
                    channel = y[c, ...]
                    # Stores the `values` for use by the remaining steps in this flow.
                    values = channel[mask].reshape(-1, 1)
                    normed = self._apply_channel_norm(
                        values,
                        {
                            # Completes the multi-line `collection` arguments or elements.
                            'mean': [y_stats['mean'][c]],
                            'std': [y_stats['std'][c]],
                            'min': [y_stats['min'][c]],
                            'max': [y_stats['max'][c]],
                        # Completes the multi-line `collection` arguments or elements.
                        },
                        norm_type,
                    ).reshape(-1)
                    channel[mask] = normed
                    # Stores the `]` for use by the remaining steps in this flow.
                    y[c, ...] = channel
            else:
                raise ValueError(f"ImageVoxelProcessor 的 y 维度应为4或5，当前为{y.ndim}")
            normalized['y'] = y

        # Checks `normalize_y and 'y_seed' in sample and norm_params.get('y')` before entering this conditional branch.
        if normalize_y and 'y_seed' in sample and norm_params.get('y'):
            y_seed = sample['y_seed'].float().clone()
            y_stats = norm_params['y']
            mask = x[mask_channel] > 0.5
            # Checks `y_seed.ndim == 5` before entering this conditional branch.
            if y_seed.ndim == 5:
                _, channel_axis = self._resolve_temporal_layout(sample, y_seed)
                for c in range(int(y_seed.shape[channel_axis])):
                    channel = self._select_temporal_channel(y_seed, c, channel_axis)
                    # Stores the `target_shape` for use by the remaining steps in this flow.
                    target_shape = channel[:, mask].shape
                    values = channel[:, mask].reshape(-1, 1)
                    normed = self._apply_channel_norm(
                        values,
                        # Completes the multi-line `collection` arguments or elements.
                        {
                            'mean': [y_stats['mean'][c]],
                            'std': [y_stats['std'][c]],
                            'min': [y_stats['min'][c]],
                            # Completes the multi-line `collection` arguments or elements.
                            'max': [y_stats['max'][c]],
                        },
                        norm_type,
                    ).reshape(target_shape)
                    # Stores the `channel[:, mask]` for use by the remaining steps in this flow.
                    channel[:, mask] = normed
                    self._assign_temporal_channel(y_seed, c, channel_axis, channel)
            elif y_seed.ndim == 4:
                for c in range(y_seed.shape[0]):
                    # Selects the required item from `y_seed` as the `channel`.
                    channel = y_seed[c, ...]
                    values = channel[mask].reshape(-1, 1)
                    normed = self._apply_channel_norm(
                        values,
                        # Completes the multi-line `collection` arguments or elements.
                        {
                            'mean': [y_stats['mean'][c]],
                            'std': [y_stats['std'][c]],
                            'min': [y_stats['min'][c]],
                            # Completes the multi-line `collection` arguments or elements.
                            'max': [y_stats['max'][c]],
                        },
                        norm_type,
                    ).reshape(-1)
                    # Stores the `channel[mask]` for use by the remaining steps in this flow.
                    channel[mask] = normed
                    y_seed[c, ...] = channel
            else:
                raise ValueError(f"ImageVoxelProcessor 的 y_seed 维度应为4或5，当前为{y_seed.ndim}")
            # Stores the `normalized['y_seed']` for use by the remaining steps in this flow.
            normalized['y_seed'] = y_seed

        return normalized

    @staticmethod
    def _validate_normalization_input_schema(
        # Continues the surrounding statement or collection definition.
        x: torch.Tensor,
        x_cfg: Dict[str, Any],
        expected_mask_channel: int,
    ) -> None:
        """校验预测输入与训练期归一化通道 schema 是否一致。"""
        # Validates `x.ndim != 4` and raises `ValueError` when the input cannot be processed safely.
        if x.ndim != 4:
            raise ValueError(
                "ImageVoxelProcessor 归一化输入 x 应为 [C, D, H, W]，"
                f"当前形状为 {tuple(x.shape)}"
            # Completes the multi-line `ValueError` arguments or elements.
            )

        actual_channel_count = int(x.shape[0])
        actual_mask_channel = actual_channel_count - 1
        if expected_mask_channel != actual_mask_channel:
            # Derives the `expected_channel_count` from the current values for downstream processing.
            expected_channel_count = expected_mask_channel + 1
            raise ValueError(
                "ImageVoxel 预测输入通道 schema 与训练期不一致："
                f"训练期归一化参数要求 {expected_channel_count} 个输入通道，"
                # Completes the multi-line `ValueError` arguments or elements.
                f"mask_channel={expected_mask_channel}；当前样本只有 "
                f"{actual_channel_count} 个输入通道，mask_channel={actual_mask_channel}。"
                "请检查预测 samples.csv 的数值工况列及其类型、"
                "local_feature_encoding/field_bc 配置和 reference 文件是否与训练期一致。"
            # Completes the multi-line `ValueError` arguments or elements.
            )

        feature_channels = [int(index) for index in x_cfg.get('feature_channels', [])]
        invalid_feature_channels = [
            index
            # Continues the surrounding statement or collection definition.
            for index in feature_channels
            if index < 3 or index >= expected_mask_channel
        ]
        if invalid_feature_channels:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(
                "ImageVoxel 训练期归一化参数中的 feature_channels 无效："
                f"{invalid_feature_channels} 不在有效特征通道范围 [3, "
                f"{expected_mask_channel}) 内。"
            # Completes the multi-line `ValueError` arguments or elements.
            )

    def restore_autoregressive_state(
        self,
        last_state: torch.Tensor,
        # Continues the surrounding statement or collection definition.
        raw_output: torch.Tensor,
    ) -> torch.Tensor:
        if not self.uses_autoregressive_increment():
            return raw_output
        # Validates `last_state.ndim != 5 or raw_output.ndim != 5` and raises `ValueError` when the input cannot be processed safely.
        if last_state.ndim != 5 or raw_output.ndim != 5:
            raise ValueError("ImageVoxelProcessor 增量状态重建要求 [B, C, D, H, W] 张量")
        last_channels_last = last_state.movedim(1, -1)
        output_channels_last = raw_output.movedim(1, -1)
        # Returns `` to the caller for the next stage.
        return super().restore_autoregressive_state(
            last_channels_last, output_channels_last
        ).movedim(-1, 1)

    def restore_autoregressive_model_output(self, inputs: Any, raw_output: Any) -> Any:
        # Returns early when `not self.uses_autoregressive_increment()` makes the remaining work inapplicable.
        if not self.uses_autoregressive_increment():
            return raw_output
        if not isinstance(inputs, torch.Tensor) or not isinstance(raw_output, torch.Tensor):
            raise TypeError("ImageVoxelProcessor 增量自回归单步训练要求张量输入输出")
        # Stores the `output_step` for use by the remaining steps in this flow.
        output_step = raw_output[:, 0, ...] if raw_output.ndim == 6 and raw_output.shape[1] == 1 else raw_output
        if output_step.ndim != 5:
            raise ValueError("ImageVoxelProcessor 增量自回归仅支持单步模型输出")
        input_length = int(self.config.get("task", {}).get("input_length", 1))
        # Stores the `history` for use by the remaining steps in this flow.
        history = inputs[:, -input_length * output_step.shape[1]:, ...].reshape(
            inputs.shape[0], input_length, output_step.shape[1], *output_step.shape[2:]
        )
        state_step = self.restore_autoregressive_state(history[:, -1, ...], output_step)
        # Returns `state_step.unsqueeze(1) if raw_output.ndim == 6 else state_step` to the caller for the next stage.
        return state_step.unsqueeze(1) if raw_output.ndim == 6 else state_step

    def load_state(self, state: Dict) -> None:
        """从状态字典恢复处理器缓存。"""
        if not state:
            return
        # Stores the `cached_coord_ranges` on the instance so other methods can reuse it.
        self.cached_coord_ranges = state.get('cached_coord_ranges', None)
        self.cached_voxel_centers = state.get('cached_voxel_centers', None)
        self.cached_mask = state.get('cached_mask', None)
        self.cached_x_grid = state.get('cached_x_grid', None)
        # Stores the `cached_y_grid` on the instance so other methods can reuse it.
        self.cached_y_grid = state.get('cached_y_grid', None)
        self.cached_z_grid = state.get('cached_z_grid', None)
        self.cached_voxel_size = state.get('cached_voxel_size', None)
        self.local_feature_schema = state.get('local_feature_schema')
    
    # 处理场数据，返回图像体素化格式
    def process(self, data: BaseData, coord_ranges) -> Dict:
        """处理场数据，返回图像体素化格式
        
        Args:
            data: 场数据（BaseData对象）
            coord_ranges: 坐标范围，格式为 ((x_min, x_max), (y_min, y_max), (z_min, z_max))
            
        Returns:
            处理后的数据字典，包含x、y、voxel_centers、mask和field_names
        """
        # 获取场数据
        data_dict = data.get_data()
        point_coordinates = data_dict['PointCoordinates']
        cell_coordinates = data_dict['CellCoordinates']
        # Selects the required item from `data_dict` as the `element_connectivity`.
        element_connectivity = data_dict['ElementConnectivity']
        element_type_ids = data_dict.get('ElementTypeIds')
        fields = data_dict['Fields']
        
        # 从数据中读取point_or_cell参数
        point_or_cell = data.point_or_cell
        
        # 确保coord_ranges已传入
        # Validates `coord_ranges is None` and raises `ValueError` when the input cannot be processed safely.
        if coord_ranges is None:
            raise ValueError("coord_ranges参数必须传入，不支持从数据中自动计算")
        
        # 检查是否需要使用缓存的非mask参数
        use_cached_params = self.cached_voxel_centers is not None
        
        if use_cached_params:
            # 使用缓存的非mask参数
            # Records the current execution state to make progress and failures traceable.
            _get_logger().info("使用缓存的非 mask 参数")
            voxel_centers = self.cached_voxel_centers
            x_grid = self.cached_x_grid
            y_grid = self.cached_y_grid
            # Stores the `z_grid` for use by the remaining steps in this flow.
            z_grid = self.cached_z_grid
            voxel_size = self.cached_voxel_size
            
            # 只有当几何形状不变时才复用mask
            if not self.is_geometry_varying and self.cached_mask is not None:
                _get_logger().info("使用缓存的 mask")
                # Stores the `mask` for use by the remaining steps in this flow.
                mask = self.cached_mask
            else:
                # 几何形状变化时，重新计算mask
                _get_logger().info("重新计算 mask")
                mask = _compute_voxel_mask(
                    # Completes the multi-line `_compute_voxel_mask` arguments or elements.
                    voxel_centers,
                    point_coordinates,
                    element_connectivity,
                    element_type_ids=element_type_ids,
                    mesh_dimension=self.config.get("task", {}).get("mesh_dimension"),
                # Completes the multi-line `_compute_voxel_mask` arguments or elements.
                )
        else:
            # 计算各轴长度
            lengths = [max_val - min_val for (min_val, max_val) in coord_ranges]
            
            # 自适应分辨率计算：最长方向固定为base_resolution，其他方向按比例缩放
            max_length = max(lengths)
            # Derives the `scale` from the current values for downstream processing.
            scale = self.base_resolution / max_length
            voxel_size = [max(1, int(length * scale)) for length in lengths]
            
            # 生成规则网格和体素中心
            x_coords = np.linspace(coord_ranges[0][0], coord_ranges[0][1], voxel_size[0])
            y_coords = np.linspace(coord_ranges[1][0], coord_ranges[1][1], voxel_size[1])
            # Stores the `z_coords` for use by the remaining steps in this flow.
            z_coords = np.linspace(coord_ranges[2][0], coord_ranges[2][1], voxel_size[2])
            
            # 创建网格
            x_grid, y_grid, z_grid = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
            
            # 生成体素中心坐标
            voxel_centers = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T
            
            # 计算掩码：判断体素中心是否在域内
            _get_logger().info("首次计算 mask")
            # Stores the `mask` for use by the remaining steps in this flow.
            mask = _compute_voxel_mask(
                voxel_centers,
                point_coordinates,
                element_connectivity,
                # Completes the multi-line `_compute_voxel_mask` arguments or elements.
                element_type_ids=element_type_ids,
                    mesh_dimension=self.config.get("task", {}).get("mesh_dimension"),
            )
            
            # 缓存所有参数（包括mask）
            _get_logger().info("缓存体素化参数")
            self.cached_mask = mask
            # Stores the `cached_voxel_centers` on the instance so other methods can reuse it.
            self.cached_voxel_centers = voxel_centers
            self.cached_coord_ranges = coord_ranges
            self.cached_x_grid = x_grid
            self.cached_y_grid = y_grid
            # Stores the `cached_z_grid` on the instance so other methods can reuse it.
            self.cached_z_grid = z_grid
            self.cached_voxel_size = voxel_size
        
        # 获取场名称列表
        field_names = list(fields.keys())
        if self.local_feature_encoding == "vtk_based":
            # Builds the `field_names` using the comprehension's filter and mapping rules.
            field_names = [
                name
                for name in field_names
                if name not in self.field_bc
            # Continues the surrounding statement or collection definition.
            ]
        if not field_names:
            raise ValueError("ImageVoxelProcessor 未找到可作为预测目标的物理字段")
        
        # 初始化输出
        x_channels = []
        # Initializes the `y_channels` as a collection for values from this processing stage.
        y_channels = []
        
        # 添加坐标通道（原始坐标，不需要归一化）
        x_channels.append(torch.tensor(x_grid, dtype=torch.float32))
        x_channels.append(torch.tensor(y_grid, dtype=torch.float32))
        x_channels.append(torch.tensor(z_grid, dtype=torch.float32))
        
        # 从sample_map获取工况参数（支持多个参数）
        # Stores the `sample_map` for use by the remaining steps in this flow.
        sample_map = data.get_sample_map()
        condition_params = []
        
        # 检查sample_map中是否有工况参数
        if isinstance(sample_map, dict):
            # 从sample_map中直接获取所有数值类型的参数
            for key, value in sample_map.items():
                # Checks `` before entering this conditional branch.
                if key not in _IMAGE_VOXEL_NON_PARAM_SAMPLE_KEYS and isinstance(value, (int, float)):
                    condition_params.append(value)
        
        # 添加工况参数通道
        for param in condition_params:
            # 广播参数到整个体素空间
            param_channel = torch.full(voxel_size, param, dtype=torch.float32)
            # Adds the current value to `x_channels` for later aggregation.
            x_channels.append(param_channel)

        local_target_location = str(point_or_cell).lower()
        coord_source = point_coordinates if local_target_location == 'point' else cell_coordinates
        x_channels.extend(
            # Completes the multi-line `self._build_local_feature_voxel_channels` arguments or elements.
            self._build_local_feature_voxel_channels(
                data_dict=data_dict,
                target_location=local_target_location,
                coord_source=coord_source,
                # Completes the multi-line `self._build_local_feature_voxel_channels` arguments or elements.
                voxel_centers=voxel_centers,
                voxel_size=voxel_size,
                mask=mask,
                feature_start=len(x_channels),
            # Completes the multi-line `self._build_local_feature_voxel_channels` arguments or elements.
            )
        )
        
        # 添加mask通道
        mask_grid = mask.reshape(voxel_size)
        x_channels.append(torch.tensor(mask_grid, dtype=torch.float32))
        
        # 处理场数据
        # Checks `field_names` before entering this conditional branch.
        if field_names:
            # 对每个场进行处理
            for field_name in field_names:
                field_data = fields[field_name]
                
                # 根据point_or_cell选择正确的坐标作为插值源
                if point_or_cell == 'point':
                    # 点数据：使用点坐标作为插值源
                    # Stores the `coord_source` for use by the remaining steps in this flow.
                    coord_source = point_coordinates
                else:
                    # 单元数据：使用单元中心坐标作为插值源
                    coord_source = cell_coordinates
                
                # 检测时间步维度并进行插值
                if field_data.ndim == 1:
                    # 单个时间步
                    # Stores the `interpolated_data` for use by the remaining steps in this flow.
                    interpolated_data = _interpolate_field_to_voxel(
                        coord_source, field_data, voxel_centers, mask, k=8
                    )
                    # 重塑为体素网格形状
                    field_voxel = interpolated_data.reshape(voxel_size)
                # Checks `field_data.ndim == 2` before entering this conditional branch.
                elif field_data.ndim == 2:
                    # 多个时间步
                    num_time_steps = field_data.shape[0]
                    # 初始化时间步结果数组
                    time_steps_result = np.zeros((num_time_steps, len(voxel_centers)), dtype=field_data.dtype)
                    # 对每个时间步进行插值
                    for t in range(num_time_steps):
                        # Selects the required item from `field_data` as the `time_step_data`.
                        time_step_data = field_data[t]
                        interpolated_data = _interpolate_field_to_voxel(
                            coord_source, time_step_data, voxel_centers, mask, k=8
                        )
                        # Stores the `time_steps_result[t]` for use by the remaining steps in this flow.
                        time_steps_result[t] = interpolated_data
                    # 重塑为体素网格形状，包含时间步维度
                    field_voxel = time_steps_result.reshape((num_time_steps,) + tuple(voxel_size))
                else:
                    # 不支持的维度
                    raise ValueError(f"Unsupported field data dimension: {field_data.ndim}")
                
                # 所有场都作为输出场添加到y通道
                # Adds the current value to `y_channels` for later aggregation.
                y_channels.append(torch.tensor(field_voxel, dtype=torch.float32))
        
        # 构造最终的x和y
        x = torch.stack(x_channels, dim=0)
        y = torch.stack(y_channels, dim=0)
        meta = self._build_processed_sample_meta(
            # Completes the multi-line `self._build_processed_sample_meta` arguments or elements.
            data=data,
            y=y,
            time_axis=1 if y.ndim >= 5 else None,
        )
        # Stores the `meta` for use by the remaining steps in this flow.
        meta = self._with_local_feature_meta(meta, **self._last_local_feature_meta)
       
        # 构造处理后的数据字典，包含必要的信息用于保存vtk文件
        processed_data = {
            'x': x,
            'y': y,
            # Completes the multi-line `collection` arguments or elements.
            'voxel_centers': voxel_centers,
            'mask': mask,
            'field_names': field_names,
            'meta': meta,
        # Completes the multi-line `collection` arguments or elements.
        }
        
        return processed_data

    def build_autoregressive_view(
        self,
        # Completes the multi-line `collection` arguments or elements.
        sample: Dict[str, Any],
        start_t: int,
        input_length: int,
        output_length: int,
    # Continues the surrounding statement or collection definition.
    ) -> Any:
        x = sample["x"]
        y = sample["y"]
        meta = sample.get("meta", {})
        # Converts the `time_axis` to `int` so later checks use a consistent type.
        time_axis = int(meta.get("time_axis", 1))

        history = self._slice_temporal_tensor(y, start_t, start_t + input_length, time_axis)
        history = self._move_time_axis_to_front(history, time_axis)
        history = history.reshape(history.shape[0] * history.shape[1], *history.shape[2:])
        # Stores the `model_input` for use by the remaining steps in this flow.
        model_input = torch.cat([x, history], dim=0)

        future = self._slice_temporal_tensor(
            y,
            start_t + input_length,
            # Completes the multi-line `self._slice_temporal_tensor` arguments or elements.
            start_t + input_length + output_length,
            time_axis,
        )
        future = self._move_time_axis_to_front(future, time_axis)
        # Checks `output_length == 1` before entering this conditional branch.
        if output_length == 1:
            future = future[0]
        return model_input, future

    def _build_autoregressive_predict_seed_sample(
        # Continues the surrounding statement or collection definition.
        self,
        data: BaseData,
        coord_ranges,
    ) -> Dict:
        """构建仅包含初始 history 的 ImageVoxel autoregressive 预测样本。"""
        # Stores the `processed` for use by the remaining steps in this flow.
        processed = self.process(data, coord_ranges=coord_ranges)
        y = processed["y"]
        input_length = int(self.config.get("task", {}).get("input_length", 1))
        source_meta = processed.get("meta", {})
        # Converts the `time_axis` to `int` so later checks use a consistent type.
        time_axis = int(source_meta.get("time_axis", 1))
        total_time_steps = source_meta.get("total_time_steps")
        meta = self._build_temporal_rollout_seed_meta(
            source_meta,
            # Completes the multi-line `self._build_temporal_rollout_seed_meta` arguments or elements.
            seed_length=input_length,
            total_time_steps=int(total_time_steps or y.shape[time_axis]),
        )

        history = self._slice_temporal_tensor(y, 0, input_length, time_axis)
        # Stores the `history` for use by the remaining steps in this flow.
        history = self._move_time_axis_to_front(history, time_axis)
        if int(history.shape[0]) != input_length:
            raise ValueError(
                f"ImageVoxelProcessor 预测 history 长度不足："
                # Completes the multi-line `ValueError` arguments or elements.
                f" got={history.shape[0]}, expected={input_length}"
            )

        channels = int(history.shape[1])
        voxel_size = tuple(history.shape[2:])
        # Stores the `dummy_target` for use by the remaining steps in this flow.
        dummy_target = self._build_dummy_target(voxel_size, channels)
        output = dict(processed)
        output["y"] = dummy_target
        output["y_seed"] = history.clone()
        # Stores the `output['meta']` for use by the remaining steps in this flow.
        output["meta"] = meta
        return output

    def _build_zero_autoregressive_predict_seed_sample(
        self,
        # Continues the surrounding statement or collection definition.
        data: BaseData,
        coord_ranges,
    ) -> Dict:
        """构建使用零初始场的 ImageVoxel autoregressive 预测样本。"""
        processed = self.process_input_only(data, coord_ranges=coord_ranges)
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

        history = y_seed.float()
        # Validates `history.ndim != 5` and raises `ValueError` when the input cannot be processed safely.
        if history.ndim != 5:
            raise ValueError(
                f"ImageVoxelProcessor 预测 seed 期望 y_seed 形状为 [K, C, D, H, W]，当前为 {tuple(history.shape)}"
            )
        # Validates `int(history.shape[0]) != int(input_length)` and raises `ValueError` when the input cannot be processed safely.
        if int(history.shape[0]) != int(input_length):
            raise ValueError(
                f"ImageVoxelProcessor 预测 seed 的历史长度与 input_length 不一致："
                f" got={history.shape[0]}, expected={input_length}"
            # Completes the multi-line `ValueError` arguments or elements.
            )

        x = sample["x"]
        history_folded = history.reshape(history.shape[0] * history.shape[1], *history.shape[2:])
        return torch.cat([x, history_folded], dim=0)

    # 执行 ImageVoxelProcessor 的滚动预测，并沿通道维维护 history。
    def predict_autoregressive_rollout(
        self,
        model,
        inputs: Any,
        # Continues the surrounding statement or collection definition.
        rollout_steps: int,
    ) -> Any:
        """执行 ImageVoxelProcessor 的滚动预测，并沿通道维维护 history。"""
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("ImageVoxelProcessor autoregressive rollout 仅支持 torch.Tensor 输入")
        # Validates `inputs.dim() != 5` and raises `ValueError` when the input cannot be processed safely.
        if inputs.dim() != 5:
            raise ValueError(
                "ImageVoxelProcessor autoregressive rollout 要求输入形状为 [B, C, D, H, W]，"
                f"当前为 {tuple(inputs.shape)}"
            # Completes the multi-line `ValueError` arguments or elements.
            )

        steps = int(rollout_steps)
        if steps <= 0:
            raise ValueError(f"rollout_steps 必须为正整数，当前为 {rollout_steps}")

        # Converts the `input_length` to `int` so later checks use a consistent type.
        input_length = int(self.config.get("task", {}).get("input_length", 1))
        if input_length <= 0:
            raise ValueError(f"input_length 必须为正整数，当前为 {input_length}")

        if self._should_checkpoint_rollout(model):
            # Stores the `checkpoint_segments` for use by the remaining steps in this flow.
            checkpoint_segments = self._resolve_rollout_checkpoint_segments(
                rollout_steps=steps
            )
            if checkpoint_segments > 0:
                # Returns `` to the caller for the next stage.
                return self._predict_checkpointed_rollout(
                    model=model,
                    inputs=inputs,
                    rollout_steps=steps,
                    # Completes the multi-line `self._predict_checkpointed_rollout` arguments or elements.
                    input_length=input_length,
                    checkpoint_segments=checkpoint_segments,
                )

        current_input = inputs
        # Initializes the `batch_predictions` as a collection for values from this processing stage.
        batch_predictions = []

        for _ in range(steps):
            output = model(current_input)
            if output.dim() == 6 and output.shape[1] == 1:
                # Selects the required item from `output` as the `output_step`.
                output_step = output[:, 0, ...]
            elif output.dim() == 5:
                output_step = output
            else:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(
                    "ImageVoxelProcessor 当前仅支持单步 autoregressive 输出，"
                    f"模型输出形状为 {tuple(output.shape)}"
                )

            # Converts the `out_channels` to `int` so later checks use a consistent type.
            out_channels = int(output_step.shape[1])
            history_channels = input_length * out_channels
            static_channels = int(current_input.shape[1]) - history_channels
            if static_channels < 0:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(
                    "ImageVoxelProcessor 无法从输入通道推断静态上下文："
                    f"input_channels={current_input.shape[1]}, history_channels={history_channels}"
                )

            # Selects the required item from `current_input` as the `static_context`.
            static_context = current_input[:, :static_channels, ...]
            history = current_input[:, static_channels:, ...]
            history = history.reshape(
                history.shape[0],
                # Completes the multi-line `history.reshape` arguments or elements.
                input_length,
                out_channels,
                *history.shape[2:],
            )
            # Stores the `state_step` for use by the remaining steps in this flow.
            state_step = self.restore_autoregressive_state(history[:, -1, ...], output_step)
            next_history = torch.cat([history[:, 1:, ...], state_step.unsqueeze(1)], dim=1)
            next_history = next_history.reshape(next_history.shape[0], -1, *next_history.shape[3:])
            current_input = torch.cat([static_context, next_history], dim=1)
            # Adds the current value to `batch_predictions` for later aggregation.
            batch_predictions.append(state_step)

        return torch.stack(batch_predictions, dim=1)

    @staticmethod
    def _should_checkpoint_rollout(model) -> bool:
        """仅在训练期可微 rollout 中启用时间段重算。"""
        # Returns `` to the caller for the next stage.
        return bool(getattr(model, "training", False)) and torch.is_grad_enabled()

    def _resolve_rollout_checkpoint_segments(self, rollout_steps: int) -> int:
        """解析时间维检查点段数，0 表示不启用。"""
        raw_value = (
            self.config.get("training", {})
            # Completes the multi-line `collection` arguments or elements.
            .get("temporal_rollout", {})
            .get("checkpoint_segments", 0)
        )
        if isinstance(raw_value, bool):
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("training.temporal_rollout.checkpoint_segments 必须为非负整数")
        if isinstance(raw_value, float) and not raw_value.is_integer():
            raise ValueError("training.temporal_rollout.checkpoint_segments 必须为非负整数")
        try:
            # Converts the `checkpoint_segments` to `int` so later checks use a consistent type.
            checkpoint_segments = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "training.temporal_rollout.checkpoint_segments 必须为非负整数"
            # Completes the multi-line `ValueError` arguments or elements.
            ) from exc
        if checkpoint_segments < 0:
            raise ValueError("training.temporal_rollout.checkpoint_segments 必须为非负整数")
        if checkpoint_segments > rollout_steps:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(
                "training.temporal_rollout.checkpoint_segments 不能大于本次 rollout 步数"
            )
        return checkpoint_segments

    # Continues the surrounding statement or collection definition.
    @staticmethod
    @contextmanager
    def _preserve_batch_norm_running_stats(model):
        """重算后恢复 BatchNorm buffers，避免对运行统计量产生第二次写入。"""
        batch_norm_layers = [
            # Continues the surrounding statement or collection definition.
            module
            for module in model.modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            and bool(module.track_running_stats)
        # Continues the surrounding statement or collection definition.
        ]
        buffer_snapshots = [
            (
                module,
                # Completes the multi-line `module.running_mean.detach().clone` arguments or elements.
                None if module.running_mean is None else module.running_mean.detach().clone(),
                None if module.running_var is None else module.running_var.detach().clone(),
                None
                if module.num_batches_tracked is None
                # Completes the multi-line `module.num_batches_tracked.detach().clone` arguments or elements.
                else module.num_batches_tracked.detach().clone(),
            )
            for module in batch_norm_layers
        ]
        # Keeps the operation and its recovery path within the same exception boundary.
        try:
            yield
        finally:
            with torch.no_grad():
                # Visits each item in `buffer_snapshots` and applies the current processing step.
                for module, running_mean, running_var, num_batches_tracked in buffer_snapshots:
                    if running_mean is not None:
                        module.running_mean.copy_(running_mean)
                    if running_var is not None:
                        # Calls `module.running_var.copy_` to perform the operation required at this stage.
                        module.running_var.copy_(running_var)
                    if num_batches_tracked is not None:
                        module.num_batches_tracked.copy_(num_batches_tracked)

    def _checkpoint_context_fn(self, model):
        """为非重入 checkpoint 提供前向与反向重算上下文。"""
        # Continues the surrounding statement or collection definition.
        @contextmanager
        def forward_context():
            yield

        return forward_context(), self._preserve_batch_norm_running_stats(model)

    # 重放一段体素自由滚动，段边界仅保留输入张量。
    def _run_rollout_segment(
        self,
        model,
        current_input: torch.Tensor,
        # Continues the surrounding statement or collection definition.
        input_length: int,
        segment_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """重放一段体素自由滚动，段边界仅保留输入张量。"""
        predictions = []
        # Visits each item in `range(segment_steps)` and applies the current processing step.
        for _ in range(segment_steps):
            output = model(current_input)
            if output.dim() == 6 and output.shape[1] == 1:
                output_step = output[:, 0, ...]
            # Checks `output.dim() == 5` before entering this conditional branch.
            elif output.dim() == 5:
                output_step = output
            else:
                raise ValueError(
                    # Completes the multi-line `ValueError` arguments or elements.
                    "ImageVoxelProcessor 当前仅支持单步 autoregressive 输出，"
                    f"模型输出形状为 {tuple(output.shape)}"
                )

            out_channels = int(output_step.shape[1])
            # Derives the `history_channels` from the current values for downstream processing.
            history_channels = input_length * out_channels
            static_channels = int(current_input.shape[1]) - history_channels
            if static_channels < 0:
                raise ValueError(
                    # Completes the multi-line `ValueError` arguments or elements.
                    "ImageVoxelProcessor 无法从输入通道推断静态上下文："
                    f"input_channels={current_input.shape[1]}, history_channels={history_channels}"
                )

            static_context = current_input[:, :static_channels, ...]
            # Stores the `history` for use by the remaining steps in this flow.
            history = current_input[:, static_channels:, ...].reshape(
                current_input.shape[0],
                input_length,
                out_channels,
                # Completes the multi-line `current_input[:, static_channels:, ...].reshape` arguments or elements.
                *current_input.shape[2:],
            )
            state_step = self.restore_autoregressive_state(history[:, -1, ...], output_step)
            next_history = torch.cat([history[:, 1:, ...], state_step.unsqueeze(1)], dim=1)
            # Stores the `current_input` for use by the remaining steps in this flow.
            current_input = torch.cat(
                [
                    static_context,
                    next_history.reshape(next_history.shape[0], -1, *next_history.shape[3:]),
                # Completes the multi-line `collection` arguments or elements.
                ],
                dim=1,
            )
            predictions.append(state_step)

        # Returns `current_input, torch.stack(predictions, dim=1)` to the caller for the next stage.
        return current_input, torch.stack(predictions, dim=1)

    def _predict_checkpointed_rollout(
        self,
        model,
        # Continues the surrounding statement or collection definition.
        inputs: torch.Tensor,
        rollout_steps: int,
        input_length: int,
        checkpoint_segments: int,
    # Continues the surrounding statement or collection definition.
    ) -> torch.Tensor:
        """以时间段 checkpoint 重放体素自由滚动，避免保留完整 H 步激活。"""
        segment_size = (rollout_steps + checkpoint_segments - 1) // checkpoint_segments
        current_input = inputs
        predictions = []
        # Stores the `completed_steps` for use by the remaining steps in this flow.
        completed_steps = 0

        while completed_steps < rollout_steps:
            current_segment_steps = min(segment_size, rollout_steps - completed_steps)

            def run_segment(
                # Continues the surrounding statement or collection definition.
                segment_input: torch.Tensor,
                steps_in_segment: int = current_segment_steps,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                return self._run_rollout_segment(
                    # Completes the multi-line `self._run_rollout_segment` arguments or elements.
                    model=model,
                    current_input=segment_input,
                    input_length=input_length,
                    segment_steps=steps_in_segment,
                # Completes the multi-line `self._run_rollout_segment` arguments or elements.
                )

            current_input, segment_predictions = checkpoint(
                run_segment,
                current_input,
                # Completes the multi-line `checkpoint` arguments or elements.
                use_reentrant=False,
                context_fn=lambda: self._checkpoint_context_fn(model),
            )
            predictions.append(segment_predictions)
            # Stores the `completed_steps` for use by the remaining steps in this flow.
            completed_steps += current_segment_steps

        return torch.cat(predictions, dim=1)
    

    
    def save_to_pt(self, processed_data: Dict, save_path: str) -> None:
        """保存处理后的数据为pt文件
        
        Args:
            processed_data: 处理后的数据
            save_path: 保存路径
        """
        # 保存完整的数据，包括x、y、voxel_centers、mask和field_names
        output = dict(processed_data)
        # Calls `output.setdefault` to perform the operation required at this stage.
        output.setdefault('meta', {})
        atomic_torch_save(output, save_path)


def _extract_cell_corner_node_indices(cell_nodes, n_nodes, elem_type=None):
    """提取用于几何近似的角点节点索引，统一转换为 0-based。"""
    corner_nodes = cgnsmesh.extract_corner_nodes(
        # Completes the multi-line `cgnsmesh.extract_corner_nodes` arguments or elements.
        cell_nodes,
        elem_type=elem_type,
        prefer_volume=True,
    )
    # Stores the `corner_nodes` for use by the remaining steps in this flow.
    corner_nodes = np.asarray(corner_nodes, dtype=np.int64).reshape(-1)
    if corner_nodes.size == 0:
        return []

    if np.all(corner_nodes >= 1) and np.all(corner_nodes <= n_nodes):
        # Derives the `normalized` from the current values for downstream processing.
        normalized = corner_nodes - 1
    else:
        normalized = corner_nodes

    return [
        # Completes the multi-line `int` arguments or elements.
        int(node_idx)
        for node_idx in normalized
        if 0 <= int(node_idx) < n_nodes
    ]


# 构建节点到单元的映射
def _build_node_to_cells_map(conn, n_nodes, element_type_ids=None):
    """构建节点到单元的映射
    
    Args:
        conn: 单元连接数组
        n_nodes: 节点数量
        
    Returns:
        节点到单元的映射字典
    """
    return _build_node_to_cells_map_corner_aware(
        conn,
        n_nodes,
        # Completes the multi-line `_build_node_to_cells_map_corner_aware` arguments or elements.
        element_type_ids=element_type_ids,
    )

def _build_node_to_cells_map_corner_aware(conn, n_nodes, element_type_ids=None):
    """构建节点到单元的映射，高阶单元优先使用角点。"""
    node2cells = {}
    # Stores the `normalized_type_ids` for use by the remaining steps in this flow.
    normalized_type_ids = None
    if element_type_ids is not None:
        normalized_type_ids = np.asarray(element_type_ids, dtype=np.int32).reshape(-1)

    for ci, cell in enumerate(conn):
        # Stores the `elem_type` for use by the remaining steps in this flow.
        elem_type = None
        if normalized_type_ids is not None and ci < len(normalized_type_ids):
            elem_type = int(normalized_type_ids[ci])
        for node_idx in _extract_cell_corner_node_indices(cell, n_nodes, elem_type=elem_type):
            # Checks `node_idx not in node2cells` before entering this conditional branch.
            if node_idx not in node2cells:
                node2cells[node_idx] = []
            node2cells[node_idx].append(ci)

    return node2cells


# 判断点point是否在多面体polyhedron_coords内部。
def _point_in_polyhedron_legacy(point, polyhedron_coords):
    """
    判断点point是否在多面体polyhedron_coords内部。
    这里只做简单凸包判断，复杂网格建议用pyvista等库。
    """
    try:
        hull = Delaunay(polyhedron_coords)
        return hull.find_simplex(point) >= 0
    # Continues the surrounding statement or collection definition.
    except Exception:
        return False


def _compute_voxel_mask(
    points,
    # Continues the surrounding statement or collection definition.
    point_coordinates,
    element_connectivity,
    max_cells=8,
    show_progress=True,
    # Continues the surrounding statement or collection definition.
    element_type_ids=None,
    mesh_dimension=None,
):
    """
    批量判断points是否在单元内，返回mask数组。需要提供PointCoordinates和ElementConnectivity。
    """
    # 使用传入的PointCoordinates和ElementConnectivity
    coords = point_coordinates
    conn = element_connectivity
    # Stores the `coords` for use by the remaining steps in this flow.
    coords = np.asarray(coords, dtype=float)
    n_nodes = coords.shape[0]
    if mesh_dimension == 2:
        # Surface cells have no 3D interior. Sample a half-voxel-diagonal
        # band around the surface instead of testing tetrahedral containment.
        # Every original node has a grid center within this distance.
        grid = np.asarray(points, dtype=float)
        spacing = []
        for axis in range(3):
            values = np.unique(grid[:, axis])
            spacing.append(float(np.min(np.diff(values))) if len(values) > 1 else 0.0)
        radius = 0.5 * float(np.linalg.norm(spacing))
        distance, _ = cKDTree(coords).query(grid, k=1)
        return (distance <= radius + 1e-12).astype(int)
    
    # 构建节点到单元的映射
    node2cells = _build_node_to_cells_map_corner_aware(
        conn,
        # Completes the multi-line `_build_node_to_cells_map_corner_aware` arguments or elements.
        n_nodes,
        element_type_ids=element_type_ids,
    )
    # 创建KD树用于快速查找最近邻
    tree = cKDTree(coords)
    # 缓存单元凸包，避免重复构建Delaunay
    # Initializes the `cell_hulls` as a collection for values from this processing stage.
    cell_hulls = {}

    # 向量化查询所有点最近节点
    _, nearest_indices = tree.query(points, k=1)

    mask = np.zeros(len(points), dtype=int)
    base_iter = enumerate(points)
    # Stores the `iterator` for use by the remaining steps in this flow.
    iterator = (
        tqdm(base_iter, total=len(points), desc="mask计算")
        if show_progress
        else base_iter
    # Continues the surrounding statement or collection definition.
    )

    for i, pt in iterator:
        # 找到最近的节点
        nearest_idx = int(nearest_indices[i])
        # 获取该节点所属的单元
        connected_cells = node2cells.get(nearest_idx, [])[:max_cells]
        # Stores the `found` for use by the remaining steps in this flow.
        found = False
        # 检查点是否在这些单元内
        for ci in connected_cells:
            if ci not in cell_hulls:
                cell_nodes = conn[ci]
                # Stores the `elem_type` for use by the remaining steps in this flow.
                elem_type = None
                if element_type_ids is not None and ci < len(element_type_ids):
                    elem_type = int(element_type_ids[ci])
                cell_node_indices = _extract_cell_corner_node_indices(
                    # Completes the multi-line `_extract_cell_corner_node_indices` arguments or elements.
                    cell_nodes,
                    n_nodes,
                    elem_type=elem_type,
                )
                # Checks `not cell_node_indices` before entering this conditional branch.
                if not cell_node_indices:
                    cell_hulls[ci] = None
                else:
                    poly_coords = coords[cell_node_indices]
                    # Keeps the operation and its recovery path within the same exception boundary.
                    try:
                        cell_hulls[ci] = Delaunay(poly_coords)
                    except Exception:
                        cell_hulls[ci] = None

            # Selects the required item from `cell_hulls` as the `hull`.
            hull = cell_hulls[ci]
            if hull is not None and hull.find_simplex(pt) >= 0:
                found = True
                break
        # Stores the `mask[i]` for use by the remaining steps in this flow.
        mask[i] = 1 if found else 0
    return mask


def _interpolate_field_to_voxel(raw_coords, raw_field, points, mask, k=8):
    """
    对于mask为1的点，使用最近k个原始坐标的物理量加权插值，mask为0则物理量为0。
    向量化优化：批量插值，避免for循环。
    
    Args:
        raw_coords: 原始场的坐标，可以是节点坐标或单元中心坐标
        raw_field: 原始场数据，长度应与raw_coords匹配
        points: 要插值到的点（体素中心）
        mask: 标记点是否在域内的掩码
        k: 最近邻数量
        
    Returns:
        插值到体素中心的场数据
    """
    
    tree = cKDTree(raw_coords)
    # Stores the `values` for use by the remaining steps in this flow.
    values = np.zeros((len(points),) + raw_field.shape[1:], dtype=raw_field.dtype)
    idxs = np.where(mask == 1)[0]
    if len(idxs) == 0:
        return values

    # Stores the `k_use` for use by the remaining steps in this flow.
    k_use = min(k, len(raw_coords))
    dists, inds = tree.query(points[idxs], k=k_use)
    if k_use == 1:
        dists = dists[:, None]
        # Selects the required item from `inds` as the `inds`.
        inds = inds[:, None]

    safe_inds = np.clip(inds, 0, len(raw_coords) - 1)
    weights = 1 / (dists + 1e-12)
    weights = weights / weights.sum(axis=1, keepdims=True)

    # 向量化插值
    # Checks `raw_field.ndim == 1` before entering this conditional branch.
    if raw_field.ndim == 1:
        interp = np.sum(raw_field[safe_inds] * weights, axis=1)
    else:
        interp = np.einsum('nk,nk...->n...', weights, raw_field[safe_inds])

    # Stores the `values[idxs]` for use by the remaining steps in this flow.
    values[idxs] = interp
    return values
