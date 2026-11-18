import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
# Imports `torch` for reuse in this module.
import torch
from .field_processor import FieldProcessor
from ...data.base_data import BaseData
from ...utils.torch_utils import atomic_torch_save, load_torch

# Defines `MatrixProcessor` to group related data and behavior.
class MatrixProcessor(FieldProcessor):
    _NON_PARAM_SAMPLE_KEYS = {
        'file',
        'file_path',
        # Completes the multi-line `collection` arguments or elements.
        'sample_order',
        'order',
        'sample_id',
        'temporal_steps',
        # Completes the multi-line `collection` arguments or elements.
        'total_time_steps',
        'time_axis',
    }
    """矩阵格式场处理器
    
    将场数据处理为矩阵格式，专门用于PODNN（Proper Orthogonal Decomposition Neural Network）方法
    """
    
    def __init__(self, config: Dict):
        """初始化矩阵处理器
        
        Args:
            config: 处理器配置参数
        """
        # Calls `super().__init__` to perform the operation required at this stage.
        super().__init__(config)
        self.output_format = 'matrix'  # 输出格式设为矩阵
        self.fixed_energy_ratio = 0.99999  # 每个物理量统一模态能量占比阈值
        self.var_names = config['task'].get('effective_variables', config['task']['variables'])

        # POD状态
        # Stores the `pod_modes` on the instance so other methods can reuse it.
        self.pod_modes: Dict[str, torch.Tensor] = {}
        self.singular_values: Dict[str, torch.Tensor] = {}
        self.mean_fields: Dict[str, torch.Tensor] = {}
        self.modal_coefficients: Dict[str, torch.Tensor] = {}
        # Stores the `retained_energy_ratio` on the instance so other methods can reuse it.
        self.retained_energy_ratio: Dict[str, float] = {}
        self.coeff_sizes: Dict[str, int] = {}
        self.coeff_slices: Dict[str, List[int]] = {}
        self._eval_norm_params: Optional[Dict] = None

        # 收集的数据：按物理量存储快照列向量
        # Stores the `collected_data` on the instance so other methods can reuse it.
        self.collected_data: Dict[str, List[np.ndarray]] = {}
        self.n_dofs: Dict[str, int] = {}
        self.n_snapshots: int = 0

    def infer_model_overrides(self, sample: Dict, config: Dict = None) -> Dict:
        # Validates `not isinstance(sample, dict)` and raises `ValueError` when the input cannot be processed safely.
        if not isinstance(sample, dict):
            raise ValueError("MatrixProcessor.infer_model_overrides expects a dict sample payload")
        if 'x' not in sample or 'y' not in sample:
            raise ValueError("MatrixProcessor.infer_model_overrides requires processed sample keys: x, y")

        # Selects the required item from `sample` as the `x`.
        x = sample['x']
        y = sample['y']
        if not hasattr(x, 'numel') or not hasattr(y, 'numel') or not hasattr(y, 'shape'):
            raise ValueError("MatrixProcessor.infer_model_overrides expects sample['x'] and sample['y'] to be tensor-like objects")

        # Reads `task` from the mapping into the task settings with its declared fallback.
        task_cfg = (config or self.config).get("task", {})
        mode = str(task_cfg.get("mode", self.mode)).lower()
        temporal_strategy = str(task_cfg.get("temporal_strategy", "direct")).lower()

        if mode == "temporal" and temporal_strategy == "autoregressive":
            # Validates `y.ndim != 2` and raises `ValueError` when the input cannot be processed safely.
            if y.ndim != 2:
                raise ValueError(
                    f"MatrixProcessor autoregressive 期望时序 y 形状为 [T, C]，当前维度为 {y.ndim}"
                )

            # Converts the `input_length` to `int` so later checks use a consistent type.
            input_length = int(task_cfg.get("input_length", 1))
            output_length = int(task_cfg.get("output_length", 1))
            coeff_dim = int(y.shape[-1])

            if hasattr(x, "shape") and len(x.shape) >= 2 and int(x.shape[0]) >= input_length:
                # Converts the `in_channels` to `int` so later checks use a consistent type.
                in_channels = int(x[:input_length].numel())
            else:
                if x.ndim == 1:
                    static_dim = int(x.shape[0])
                # Checks `x.ndim == 2 and int(x.shape[0]) == 1` before entering this conditional branch.
                elif x.ndim == 2 and int(x.shape[0]) == 1:
                    static_dim = int(x.shape[-1])
                else:
                    raise ValueError(
                        # Completes the multi-line `ValueError` arguments or elements.
                        "MatrixProcessor autoregressive 无法从 x 推断输入维度，"
                        f"当前 x 形状为 {tuple(x.shape)}"
                    )
                in_channels = input_length * (static_dim + coeff_dim)

            # Initializes the `output_shape` as a collection for values from this processing stage.
            output_shape = [output_length, coeff_dim]
            return {
                'in_channels': in_channels,
                'out_channels': int(np.prod(output_shape)),
                # Completes the multi-line `collection` arguments or elements.
                'output_shape': output_shape,
            }

        return {
            'in_channels': int(x.numel()),
            # Completes the multi-line `int` arguments or elements.
            'out_channels': int(y.numel()),
            'output_shape': list(y.shape),
        }

    def _build_processed_payload(
        # Continues the surrounding statement or collection definition.
        self,
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        meta: Dict[str, Any],
        # Continues the surrounding statement or collection definition.
        y_seed: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """统一 MatrixProcessor 样本契约，仅使用 x/y。"""
        payload = {
            'x': x_tensor,
            # Completes the multi-line `collection` arguments or elements.
            'y': y_tensor,
            'meta': meta,
        }
        if y_seed is not None:
            # Stores the `payload['y_seed']` for use by the remaining steps in this flow.
            payload['y_seed'] = y_seed
        return payload

    def build_loss_weight(
        self,
        # Continues the surrounding statement or collection definition.
        inputs: Any,
        targets: Any,
        outputs: Any = None,
    ) -> Optional[torch.Tensor]:
        # Raises `NotImplementedError` to report why this path cannot continue.
        raise NotImplementedError(
            "MatrixProcessor 当前尚不支持关键区域加权。请先关闭 training.critical_region_weighting，"
            "或改用可在原始场空间构造权重图的处理链路。"
        )

    # Defines `_get_payload_x` for  get payload x logic.
    def _get_payload_x(self, sample: Dict[str, Any]) -> torch.Tensor:
        x_tensor = sample.get('x')
        if x_tensor is None:
            raise KeyError("MatrixProcessor 样本缺少 x，无法继续处理")
        # Returns `x_tensor` to the caller for the next stage.
        return x_tensor

    def _get_payload_y(self, sample: Dict[str, Any]) -> torch.Tensor:
        y_tensor = sample.get('y')
        if y_tensor is None:
            # Raises `KeyError` to report why this path cannot continue.
            raise KeyError("MatrixProcessor 样本缺少 y，无法继续处理")
        return y_tensor

    def build_prediction_adapter(
        self,
        # Continues the surrounding statement or collection definition.
        reader,
        config: Dict,
        is_geometry_varying: bool,
        resolve_reference_file,
        # Continues the surrounding statement or collection definition.
        logger=None,
    ):
        from ...prediction.adapters import MatrixPredictAdapter

        return MatrixPredictAdapter(
            # Completes the multi-line `MatrixPredictAdapter` arguments or elements.
            processor=self,
            reader=reader,
            config=config,
            is_geometry_varying=is_geometry_varying,
            # Completes the multi-line `MatrixPredictAdapter` arguments or elements.
            resolve_reference_file=resolve_reference_file,
            logger=logger,
        )

    def supports_fast_predict_path(self, is_geometry_varying: bool) -> bool:
        """几何固定时，MatrixProcessor 支持直接由样本参数构造预测输入。"""
        # Stores the `temporal_strategy` for use by the remaining steps in this flow.
        temporal_strategy = str(
            self.config.get("task", {}).get("temporal_strategy", "direct")
        ).lower()
        return temporal_strategy != 'autoregressive' and not is_geometry_varying

    # 固定几何下，MatrixProcessor 支持时序 autoregressive 预测快路径。
    def supports_fixed_geometry_temporal_predict_path(self) -> bool:
        """固定几何下，MatrixProcessor 支持时序 autoregressive 预测快路径。"""
        task_cfg = self.config.get("task", {})
        return (
            str(task_cfg.get("mode", self.mode)).lower() == "temporal"
            # Completes the multi-line `str(task_cfg.get('temporal_strategy', 'direct')).lower` arguments or elements.
            and str(task_cfg.get("temporal_strategy", "direct")).lower() == "autoregressive"
            and not bool(task_cfg.get("is_geometry_varying", self.is_geometry_varying))
        )

    def requires_fixed_geometry_temporal_reference(self) -> bool:
        # Returns `False` to the caller for the next stage.
        return False

    def requires_fixed_geometry_predict_writeback_reference(self) -> bool:
        task_cfg = self.config.get("task", {})
        if (
            # Completes the multi-line `str(task_cfg.get('mode', self.mode)).lower` arguments or elements.
            str(task_cfg.get("mode", self.mode)).lower() == "temporal"
            and str(task_cfg.get("temporal_strategy", "direct")).lower() == "autoregressive"
            and int(task_cfg.get("input_length", 1)) > 1
        ):
            # Returns `False` to the caller for the next stage.
            return False
        return True

    def requires_batch_preprocess(self) -> bool:
        return True

    # 构建 MatrixProcessor 预测快路径数据集。
    def build_fast_predict_dataset(
        self,
        samples_csv: str,
        processed_dir: str,
        # Completes the multi-line `collection` arguments or elements.
        config: Dict[str, Any],
        norm_params: Optional[Dict[str, Any]],
    ) -> Tuple[Any, str]:
        """构建 MatrixProcessor 预测快路径数据集。"""
        if self.temporal_strategy == 'autoregressive':
            # Raises `NotImplementedError` to report why this path cannot continue.
            raise NotImplementedError(
                "MatrixProcessor 快速预测路径当前仅支持 steady 模式。"
            )

        from ...data.data_pipeline import DataPipeline
        # Imports `build_dataset` from `dataset` for use in this module.
        from ...dataset import build_dataset
        from ...readers.sample_map import SampleMap

        sample_map = SampleMap(samples_csv)
        sample_ids = sample_map.get_all_samples()
        # Validates `not sample_ids` and raises `ValueError` when the input cannot be processed safely.
        if not sample_ids:
            raise ValueError("样本表为空，无法进行 POD 预测")

        os.makedirs(processed_dir, exist_ok=True)
        processed_paths: List[str] = []
        # Initializes the `success_ids` as a collection for values from this processing stage.
        success_ids: List[int] = []
        expected_dim = None

        for sample_id in sample_ids:
            sample_info = sample_map.get_sample_info(sample_id) or {}
            # Stores the `params` for use by the remaining steps in this flow.
            params = self._extract_numeric_sample_params(sample_info)

            if expected_dim is None:
                expected_dim = len(params)
                if expected_dim == 0:
                    # Raises `ValueError` to report why this path cannot continue.
                    raise ValueError("样本参数中没有数值字段，无法构造 POD 预测输入")
            elif len(params) != expected_dim:
                raise ValueError(
                    f"样本参数维度不一致: sample_id={sample_id}, "
                    # Completes the multi-line `len` arguments or elements.
                    f"got={len(params)}, expected={expected_dim}"
                )

            x_tensor = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
            y_tensor = self._build_dummy_target()
            # Stores the `processed_path` for use by the remaining steps in this flow.
            processed_path = os.path.join(processed_dir, f"sample_{sample_id}.pt")
            torch.save(
                {
                    "x": x_tensor,
                    # Completes the multi-line `collection` arguments or elements.
                    "y": y_tensor,
                    "meta": {
                        "sample_id": int(sample_id),
                        "processor_type": type(self).__name__,
                        # Completes the multi-line `collection` arguments or elements.
                        "is_temporal": self.mode == "temporal",
                        "total_time_steps": None,
                        "time_axis": 0 if self.mode == "temporal" else None,
                        "prediction_view": "direct_input_only" if self.mode == "temporal" else "steady_input_only",
                    # Completes the multi-line `collection` arguments or elements.
                    },
                },
                processed_path,
            )
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
            raise ValueError("未从检查点找到归一化参数，请确保训练时保存了归一化参数")

        # Stores the `effective_csv` for use by the remaining steps in this flow.
        effective_csv = samples_csv
        if len(success_ids) < len(sample_ids):
            filtered_csv = os.path.join(processed_dir, "effective_samples.csv")
            effective_csv = self._create_filtered_samples_csv(
                # Completes the multi-line `self._create_filtered_samples_csv` arguments or elements.
                samples_csv, filtered_csv, success_ids
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

    def build_fixed_geometry_temporal_predict_dataset(
        # Continues the surrounding statement or collection definition.
        self,
        reader,
        samples_csv: str,
        processed_dir: str,
        # Completes the multi-line `collection` arguments or elements.
        config: Dict[str, Any],
        norm_params: Optional[Dict[str, Any]],
        reference_file: Optional[str],
    ) -> Tuple[Any, str]:
        """构建固定几何时序 autoregressive 预测数据集。"""
        # Imports `DataPipeline` from `data.data_pipeline` for use in this module.
        from ...data.data_pipeline import DataPipeline
        from ...data.loaders import FieldSampleLoader
        from ...dataset import build_dataset
        from ...readers.sample_map import SampleMap
        # Imports `resolve_path_from_base` from `utils.path_utils` for use in this module.
        from ...utils.path_utils import resolve_path_from_base

        if not self.supports_fixed_geometry_temporal_predict_path():
            raise NotImplementedError(
                "MatrixProcessor 固定几何时序预测快路径仅支持 temporal autoregressive 模式。"
            # Completes the multi-line `NotImplementedError` arguments or elements.
            )

        if not self.pod_modes:
            raise ValueError("MatrixProcessor 尚未加载 POD 模态，无法构造固定几何时序预测输入")

        sample_map = SampleMap(samples_csv)
        # Stores the sample identifiers for use by the remaining steps in this flow.
        sample_ids = sample_map.get_all_samples()
        if not sample_ids:
            raise ValueError("样本表为空，无法构造固定几何时序预测数据集")

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

        # Initializes the `processed_paths` as a collection for values from this processing stage.
        processed_paths: List[str] = []
        success_ids: List[int] = []

        for sample_id in sample_ids:
            sample_file = sample_map.get_file_path(sample_id)
            # Stores the `sample_info` for use by the remaining steps in this flow.
            sample_info = dict(sample_map.get_sample_info(sample_id) or {})
            sample_info["sample_order"] = int(sample_id)
            sample_info.setdefault("file_path", sample_file or "")

            if input_length <= 1:
                # Stores the current data for use by the remaining steps in this flow.
                data = BaseData(
                    data={"Fields": {}},
                    task_type="field",
                    mode="temporal",
                    # Completes the multi-line `str` arguments or elements.
                    point_or_cell=str(self.config.get("task", {}).get("point_or_cell", "point")),
                    sample_map=sample_info,
                )
                processed_data = self._build_zero_autoregressive_predict_seed_sample(data)
            # Continues the surrounding statement or collection definition.
            else:
                if not sample_file:
                    raise ValueError("K>1 的 Matrix autoregressive 预测必须提供样本原始文件")
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
            raise ValueError("固定几何时序预测输入构建失败，没有可用样本")

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
            raise ValueError("未从检查点找到归一化参数，请确保训练时保存了归一化参数")

        effective_csv = samples_csv
        if len(success_ids) < len(sample_ids):
            # Stores the `filtered_csv` for use by the remaining steps in this flow.
            filtered_csv = os.path.join(processed_dir, "effective_samples.csv")
            effective_csv = self._create_filtered_samples_csv(
                samples_csv=samples_csv,
                output_csv=filtered_csv,
                # Completes the multi-line `self._create_filtered_samples_csv` arguments or elements.
                keep_ids=success_ids,
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

    def load_checkpoint_artifacts(self, checkpoint: Dict[str, Any]) -> None:
        """从 checkpoint 恢复 MatrixProcessor 所需的 POD 状态。"""
        # Reads `norm_params` from the mapping into the normalization statistics.
        norm_params = checkpoint.get("norm_params")
        if norm_params:
            self._eval_norm_params = norm_params
            self.set_normalization_params(norm_params)

        # Reads `modes` from the mapping into the `modes`.
        modes = checkpoint.get("modes")
        if modes:
            required = ["modes", "mean_fields", "var_names", "coeff_sizes", "coeff_slices"]
            missing = [k for k in required if k not in modes]
            # Validates `missing` and raises `ValueError` when the input cannot be processed safely.
            if missing:
                raise ValueError(f"checkpoint 中的 modes 缺少必要字段: {missing}")

            self.pod_modes = modes["modes"]
            self.mean_fields = modes["mean_fields"]
            # Stores the `var_names` on the instance so other methods can reuse it.
            self.var_names = modes["var_names"]
            self.coeff_sizes = modes["coeff_sizes"]
            self.coeff_slices = modes["coeff_slices"]

    def set_eval_artifacts(
        # Continues the surrounding statement or collection definition.
        self,
        norm_params: Optional[Dict[str, Any]] = None,
        modes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """显式注入评估阶段所需的归一化参数与 POD 状态。"""
        # Checks `norm_params is not None` before entering this conditional branch.
        if norm_params is not None:
            self._eval_norm_params = norm_params
            self.set_normalization_params(norm_params)
        if modes is not None:
            # Calls `self.load_checkpoint_artifacts` to perform the operation required at this stage.
            self.load_checkpoint_artifacts({"modes": modes})

    def process_loaded_samples(
        self,
        data_items: List[BaseData],
        # Continues the surrounding statement or collection definition.
        stage: str,
        context: Optional[Dict] = None,
    ) -> List[Dict]:
        if stage in {'train', 'evaluate'}:
            # Returns `self.process(data_items)` to the caller for the next stage.
            return self.process(data_items)
        return [self.process(data) for data in data_items]

    def denormalize_predictions(self, predictions, norm_params: Dict = None):
        if norm_params is None or 'y' not in norm_params:
            # Returns `predictions` to the caller for the next stage.
            return predictions

        if isinstance(predictions, dict):
            return {
                key: self.denormalize_predictions(value, norm_params)
                # Completes the multi-line `collection` arguments or elements.
                for key, value in predictions.items()
            }

        method = norm_params['norm_type']
        y_params = norm_params['y']
        # Stores the `denorm` for use by the remaining steps in this flow.
        denorm = np.array(predictions, copy=True)
        for _, item in y_params.items():
            start, end = item['slice']
            if method == 'standardization':
                # Selects the required item from `item` as the `mean`.
                mean = item['mean']
                std = item['std']
                denorm[..., start:end] = denorm[..., start:end] * (std + 1e-12) + mean
            elif method == 'min_max':
                # Selects the required item from `item` as the `min_v`.
                min_v = item['min']
                max_v = item['max']
                denorm[..., start:end] = denorm[..., start:end] * (max_v - min_v + 1e-12) + min_v
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

        # Selects the required item from `norm_params` as the `method`.
        method = norm_params['norm_type']
        normalized = outputs.float().clone()
        for _, item in norm_params['y'].items():
            start, end = item['slice']
            # Initializes the `stats` as a collection for values from this processing stage.
            stats = {
                'mean': [item['mean']],
                'std': [item['std']],
                'min': [item['min']],
                # Completes the multi-line `collection` arguments or elements.
                'max': [item['max']],
            }
            part = normalized[..., start:end]
            normalized[..., start:end] = self._apply_channel_norm(
                # Completes the multi-line `part.reshape` arguments or elements.
                part.reshape(-1, 1),
                stats,
                method,
            ).reshape(part.shape)
        # Returns `normalized` to the caller for the next stage.
        return normalized

    def denormalize_output_tensor(
        self,
        outputs: Any,
        # Completes the multi-line `collection` arguments or elements.
        norm_params: Optional[Dict[str, Any]],
        inputs: Any = None,
    ) -> Any:
        if norm_params is None or 'y' not in norm_params or not isinstance(outputs, torch.Tensor):
            # Returns `outputs` to the caller for the next stage.
            return outputs

        method = norm_params['norm_type']
        denormalized = outputs.float().clone()
        for _, item in norm_params['y'].items():
            # Selects the required item from `item` as the `(start, end)`.
            start, end = item['slice']
            stats = {
                'mean': [item['mean']],
                'std': [item['std']],
                # Completes the multi-line `collection` arguments or elements.
                'min': [item['min']],
                'max': [item['max']],
            }
            part = denormalized[..., start:end]
            # Stores the `, start:end]` for use by the remaining steps in this flow.
            denormalized[..., start:end] = self._apply_channel_denorm(
                part.reshape(-1, 1),
                stats,
                method,
            # Completes the multi-line `self._apply_channel_denorm(part.reshape(-1, 1), stats, method).reshape` arguments or elements.
            ).reshape(part.shape)
        return denormalized

    def merge_eval_batches(
        self,
        # Completes the multi-line `collection` arguments or elements.
        batch_payloads: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        if batch_payloads:
            # Stores the `raw_predictions` for use by the remaining steps in this flow.
            raw_predictions = np.concatenate(
                [np.asarray(payload['predictions']) for payload in batch_payloads],
                axis=0,
            )
            # Stores the `raw_targets` for use by the remaining steps in this flow.
            raw_targets = np.concatenate(
                [np.asarray(payload['targets']) for payload in batch_payloads],
                axis=0,
            )
        # Continues the surrounding statement or collection definition.
        else:
            raw_predictions = np.array([])
            raw_targets = np.array([])

        runtime_norm_params = config.get("__eval_norm_params__") if isinstance(config, dict) else None
        # Checks `runtime_norm_params is not None` before entering this conditional branch.
        if runtime_norm_params is not None:
            self._eval_norm_params = runtime_norm_params

        scatter_by_var: Dict[str, Dict[str, np.ndarray]] = {}
        denorm_predictions = np.array([])
        # Stores the `denorm_targets` for use by the remaining steps in this flow.
        denorm_targets = np.array([])
        if raw_predictions.size > 0 and raw_targets.size > 0:
            self._ensure_eval_artifacts_loaded(config)
            denorm_predictions = self.denormalize_predictions(
                # Completes the multi-line `self.denormalize_predictions` arguments or elements.
                predictions=raw_predictions,
                norm_params=self._eval_norm_params,
            )
            denorm_targets = self.denormalize_predictions(
                # Completes the multi-line `self.denormalize_predictions` arguments or elements.
                predictions=raw_targets,
                norm_params=self._eval_norm_params,
            )
            recon_predictions = self._reconstruct_eval_batch(denorm_predictions)
            # Stores the `recon_targets` for use by the remaining steps in this flow.
            recon_targets = self._reconstruct_eval_batch(denorm_targets)

            for var_name in recon_predictions.keys():
                scatter_by_var[var_name] = {
                    'predictions': np.asarray(recon_predictions[var_name]).reshape(-1),
                    # Completes the multi-line `np.asarray(recon_targets[var_name]).reshape` arguments or elements.
                    'targets': np.asarray(recon_targets[var_name]).reshape(-1),
                }

        overall_predictions = (
            np.concatenate(
                # Completes the multi-line `scatter_by_var.values` arguments or elements.
                [payload['predictions'] for payload in scatter_by_var.values()],
                axis=0,
            )
            if scatter_by_var
            # Completes the multi-line `np.array` arguments or elements.
            else np.array([])
        )
        overall_targets = (
            np.concatenate(
                # Completes the multi-line `scatter_by_var.values` arguments or elements.
                [payload['targets'] for payload in scatter_by_var.values()],
                axis=0,
            )
            if scatter_by_var
            # Completes the multi-line `np.array` arguments or elements.
            else np.array([])
        )

        sample_predictions = self._denormalize_sample_output_map(
            self._merge_sample_output_maps(
                # Completes the multi-line `self._merge_sample_output_maps` arguments or elements.
                batch_payloads=batch_payloads,
                payload_key='sample_predictions',
                fallback_key='predictions',
            ),
            # Completes the multi-line `collection` arguments or elements.
            {"__eval_norm_params__": self._eval_norm_params},
        )
        sample_targets = self._denormalize_sample_output_map(
            self._merge_sample_output_maps(
                # Completes the multi-line `self._merge_sample_output_maps` arguments or elements.
                batch_payloads=batch_payloads,
                payload_key='sample_targets',
                fallback_key='targets',
            ),
            # Completes the multi-line `collection` arguments or elements.
            {"__eval_norm_params__": self._eval_norm_params},
        )

        return {
            'raw_predictions': raw_predictions,
            # Completes the multi-line `collection` arguments or elements.
            'raw_targets': raw_targets,
            'denormalized_predictions': denorm_predictions,
            'denormalized_targets': denorm_targets,
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
            raise ValueError("MatrixProcessor 训练样本为空，无法计算归一化参数")

        x_values = []
        y_values = []
        for sample in train_samples:
            # Reads `x` from the mapping into the `x`.
            x = sample.get('x')
            y = sample.get('y')
            if x is not None:
                x_values.append(x.reshape(-1).float() if x.ndim == 1 else x.squeeze(0).reshape(-1).float())
            # Checks `y is not None` before entering this conditional branch.
            if y is not None:
                y_values.append(y.float())

        norm_params = {
            'processor_type': 'MatrixProcessor',
            # Completes the multi-line `collection` arguments or elements.
            'norm_type': norm_type,
            'geometry': {},
            'x': {},
            'y': {},
            # Completes the multi-line `collection` arguments or elements.
            'meta': {},
        }

        if x_values:
            x_stack = torch.stack(x_values, dim=0)
            # Stores the `norm_params['x']` for use by the remaining steps in this flow.
            norm_params['x'] = self._compute_channel_stats(x_stack)

        coeff_slices = self.coeff_slices
        var_names = self.var_names
        if not coeff_slices and save_path:
            # Stores the `modes_path` for use by the remaining steps in this flow.
            modes_path = os.path.join(os.path.dirname(save_path), 'modes.pt')
            if os.path.exists(modes_path):
                modes_data = load_torch(modes_path, map_location='cpu', weights_only=False)
                coeff_slices = modes_data.get('coeff_slices', {})
                # Reads `var_names` from the mapping into the `var_names` with its declared fallback.
                var_names = modes_data.get('var_names', var_names)

        if not coeff_slices:
            raise ValueError("MatrixProcessor 缺少 coeff_slices，无法按物理量计算 y 归一化参数")

        ordered_names = var_names if var_names else list(coeff_slices.keys())
        # Initializes the `y_by_variable` as a collection for values from this processing stage.
        y_by_variable = {}
        delta_by_variable = {}
        for var_name in ordered_names:
            if var_name not in coeff_slices:
                # Handles the data or control-flow case represented by this statement.
                continue
            start, end = coeff_slices[var_name]
            values = []
            for y in y_values:
                # Checks `y.ndim == 1` before entering this conditional branch.
                if y.ndim == 1:
                    values.append(y[start:end].reshape(-1))
                elif y.ndim == 2:
                    values.append(y[:, start:end].reshape(-1))
                # Continues the surrounding statement or collection definition.
                else:
                    raise ValueError(f"MatrixProcessor 的 y 维度应为1或2，当前为{y.ndim}")
            combined = torch.cat(values, dim=0).reshape(-1, 1)
            stats = self._compute_channel_stats(combined)
            # Initializes the `y_by_variable[var_name]` as a collection for values from this processing stage.
            y_by_variable[var_name] = {
                'slice': [int(start), int(end)],
                'mean': stats['mean'][0],
                'std': stats['std'][0],
                # Completes the multi-line `collection` arguments or elements.
                'min': stats['min'][0],
                'max': stats['max'][0],
            }
            if self.uses_autoregressive_increment():
                # Initializes the `delta_values` as a collection for values from this processing stage.
                delta_values = []
                for y in y_values:
                    if y.ndim == 2 and y.shape[0] >= 2:
                        delta_values.append((y[1:, start:end] - y[:-1, start:end]).reshape(-1, 1))
                # Validates `not delta_values` and raises `ValueError` when the input cannot be processed safely.
                if not delta_values:
                    raise ValueError("增量自回归需要至少一条含两个时间步的训练序列")
                delta_stats = self._compute_channel_stats(torch.cat(delta_values, dim=0))
                delta_by_variable[var_name] = {
                    # Completes the multi-line `collection` arguments or elements.
                    'slice': [int(start), int(end)],
                    'mean': delta_stats['mean'][0],
                    'std': max(float(delta_stats['std'][0]), 1e-12),
                    'min': delta_stats['min'][0],
                    # Completes the multi-line `collection` arguments or elements.
                    'max': delta_stats['max'][0],
                }

        norm_params['y'] = y_by_variable
        if delta_by_variable:
            # Initializes the `norm_params['autoregressive_delta']` as a collection for values from this processing stage.
            norm_params['autoregressive_delta'] = {
                'schema_version': 1,
                'by_variable': delta_by_variable,
            }
        # Initializes the `norm_params['meta']` as a collection for values from this processing stage.
        norm_params['meta'] = {
            'output_shape': self.config.get('model', {}).get('output_shape', []),
        }
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

        # Checks `'x' in sample and norm_params.get('x')` before entering this conditional branch.
        if 'x' in sample and norm_params.get('x'):
            x = sample['x'].float().clone()
            original_shape = x.shape
            x_2d = x.reshape(1, -1) if x.ndim == 1 else x.reshape(-1, original_shape[-1])
            # Stores the `x_norm` for use by the remaining steps in this flow.
            x_norm = self._apply_channel_norm(x_2d, norm_params['x'], norm_type)
            normalized['x'] = x_norm.reshape(original_shape)

        if normalize_y and 'y' in sample and norm_params.get('y'):
            y = sample['y'].float().clone()
            # Visits each item in `norm_params['y'].items()` and applies the current processing step.
            for _, item in norm_params['y'].items():
                start, end = item['slice']
                stats = {
                    'mean': [item['mean']],
                    # Completes the multi-line `collection` arguments or elements.
                    'std': [item['std']],
                    'min': [item['min']],
                    'max': [item['max']],
                }
                # Checks `y.ndim == 1` before entering this conditional branch.
                if y.ndim == 1:
                    y[start:end] = self._apply_channel_norm(y[start:end], stats, norm_type)
                elif y.ndim == 2:
                    part = self._apply_channel_norm(y[:, start:end].reshape(-1, 1), stats, norm_type)
                    # Stores the `y[:, start:end]` for use by the remaining steps in this flow.
                    y[:, start:end] = part.reshape(y.shape[0], end - start)
                else:
                    raise ValueError(f"MatrixProcessor 的 y 维度应为1或2，当前为{y.ndim}")
            normalized['y'] = y

        # Checks `normalize_y and 'y_seed' in sample and norm_params.get('y')` before entering this conditional branch.
        if normalize_y and 'y_seed' in sample and norm_params.get('y'):
            seed = sample['y_seed'].float().clone()
            for _, item in norm_params['y'].items():
                start, end = item['slice']
                # Initializes the `stats` as a collection for values from this processing stage.
                stats = {
                    'mean': [item['mean']],
                    'std': [item['std']],
                    'min': [item['min']],
                    # Completes the multi-line `collection` arguments or elements.
                    'max': [item['max']],
                }
                if seed.ndim == 1:
                    seed[start:end] = self._apply_channel_norm(seed[start:end], stats, norm_type)
                # Checks `seed.ndim == 2` before entering this conditional branch.
                elif seed.ndim == 2:
                    part = self._apply_channel_norm(seed[:, start:end].reshape(-1, 1), stats, norm_type)
                    seed[:, start:end] = part.reshape(seed.shape[0], end - start)
                else:
                    # Raises `ValueError` to report why this path cannot continue.
                    raise ValueError(f"MatrixProcessor 的 y_seed 维度应为1或2，当前为{seed.ndim}")
            normalized['y_seed'] = seed

        return normalized

    def restore_autoregressive_state(
        # Continues the surrounding statement or collection definition.
        self,
        last_state: torch.Tensor,
        raw_output: torch.Tensor,
    ) -> torch.Tensor:
        # Returns early when `not self.uses_autoregressive_increment()` makes the remaining work inapplicable.
        if not self.uses_autoregressive_increment():
            return raw_output
        norm_params = self._get_autoregressive_norm_params()
        norm_type = str(norm_params.get('norm_type', '')).lower()
        # Reads `by_variable` from the mapping into the `delta_by_variable` with its declared fallback.
        delta_by_variable = norm_params.get('autoregressive_delta', {}).get('by_variable', {})
        restored = raw_output.clone()
        for var_name, state_item in norm_params['y'].items():
            delta_item = delta_by_variable.get(var_name)
            # Validates `not delta_item` and raises `ValueError` when the input cannot be processed safely.
            if not delta_item:
                raise ValueError(f"增量自回归缺少变量 {var_name} 的统计参数")
            start, end = state_item['slice']
            if norm_type == 'standardization':
                # Stores the `delta_offset` for use by the remaining steps in this flow.
                delta_offset = torch.as_tensor(delta_item['mean'], device=raw_output.device, dtype=raw_output.dtype)
                delta_scale = torch.as_tensor(delta_item['std'], device=raw_output.device, dtype=raw_output.dtype)
                state_scale = torch.as_tensor(state_item['std'], device=raw_output.device, dtype=raw_output.dtype)
            elif norm_type == 'min_max':
                # Stores the `delta_offset` for use by the remaining steps in this flow.
                delta_offset = torch.as_tensor(delta_item['min'], device=raw_output.device, dtype=raw_output.dtype)
                delta_scale = torch.as_tensor(
                    float(delta_item['max']) - float(delta_item['min']),
                    device=raw_output.device,
                    # Completes the multi-line `torch.as_tensor` arguments or elements.
                    dtype=raw_output.dtype,
                )
                state_scale = torch.as_tensor(
                    float(state_item['max']) - float(state_item['min']),
                    # Completes the multi-line `torch.as_tensor` arguments or elements.
                    device=raw_output.device,
                    dtype=raw_output.dtype,
                )
            else:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(f"增量自回归不支持的归一化方式: {norm_type}")
            restored[..., start:end] = (
                last_state[..., start:end]
                + (raw_output[..., start:end] * delta_scale + delta_offset) / state_scale.clamp_min(1e-12)
            # Continues the surrounding statement or collection definition.
            )
        return restored

    def restore_autoregressive_model_output(self, inputs: Any, raw_output: Any) -> Any:
        if not self.uses_autoregressive_increment():
            # Returns `raw_output` to the caller for the next stage.
            return raw_output
        if not isinstance(inputs, torch.Tensor) or not isinstance(raw_output, torch.Tensor):
            raise TypeError("MatrixProcessor 增量自回归单步训练要求张量输入输出")
        if inputs.ndim != 3:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("MatrixProcessor 增量自回归单步训练要求输入形状为 [B, K, F]")
        output_step = raw_output[:, 0, :] if raw_output.ndim == 3 and raw_output.shape[1] == 1 else raw_output
        if output_step.ndim != 2:
            raise ValueError("MatrixProcessor 增量自回归仅支持单步模型输出")
        # Converts the `history_dim` to `int` so later checks use a consistent type.
        history_dim = int(output_step.shape[-1])
        history = inputs[..., -history_dim:]
        state_step = self.restore_autoregressive_state(history[:, -1, :], output_step)
        return state_step.unsqueeze(1) if raw_output.ndim == 3 else state_step
    
    # 按物理量收集场数据
    def collect_data(self, data: Union[BaseData, List[BaseData]]) -> None:
        """按物理量收集场数据
        
        Args:
            data: 场数据（BaseData对象）
        """
        if isinstance(data, list):
            for item in data:
                self.collect_data(item)
            # Ends this function and returns control to its caller.
            return

        data_dict = data.get_data()
        fields = data_dict.get('Fields', {})
        if not fields:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("BaseData中缺少Fields字段，无法进行POD收集")

        if not self.var_names:
            self.var_names = list(fields.keys())

        for var_name in self.var_names:
            # Stores the `field_array` for use by the remaining steps in this flow.
            field_array = self._get_field_array(fields, var_name)
            snapshots = self._extract_snapshots(field_array, data.get_mode())

            for snapshot in snapshots:
                if var_name not in self.collected_data:
                    # Initializes the `collected_data[var_name]` as a collection for values from this processing stage.
                    self.collected_data[var_name] = []

                if var_name not in self.n_dofs:
                    self.n_dofs[var_name] = snapshot.shape[0]
                elif self.n_dofs[var_name] != snapshot.shape[0]:
                    # Raises `ValueError` to report why this path cannot continue.
                    raise ValueError(
                        f"物理量{var_name}的自由度不一致: "
                        f"{self.n_dofs[var_name]} vs {snapshot.shape[0]}"
                    )

                # Adds the current value to `self.collected_data[var_name]` for later aggregation.
                self.collected_data[var_name].append(snapshot)

        # 以第一个物理量的快照数为准更新全局快照计数
        if self.var_names:
            first_var = self.var_names[0]
            self.n_snapshots = len(self.collected_data.get(first_var, []))

    # 提取单样本 POD 快照，供中断恢复时以分片形式持久化。
    def extract_pod_snapshot(self, data: BaseData) -> Dict[str, Any]:
        """提取单样本 POD 快照，供中断恢复时以分片形式持久化。"""
        data_dict = data.get_data()
        fields = data_dict.get("Fields", {})
        if not fields:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("BaseData 中缺少 Fields 字段，无法提取 POD 快照")

        var_names = list(self.var_names or fields.keys())
        snapshots_by_var: Dict[str, List[np.ndarray]] = {}
        n_dofs: Dict[str, int] = {}
        # Visits each item in `var_names` and applies the current processing step.
        for var_name in var_names:
            field_array = self._get_field_array(fields, var_name)
            snapshots = self._extract_snapshots(field_array, data.get_mode())
            normalized_snapshots = [np.asarray(snapshot) for snapshot in snapshots]
            # Validates `not normalized_snapshots` and raises `ValueError` when the input cannot be processed safely.
            if not normalized_snapshots:
                raise ValueError(f"物理量 {var_name} 未提取到 POD 快照")
            dof_count = int(normalized_snapshots[0].shape[0])
            if any(int(snapshot.shape[0]) != dof_count for snapshot in normalized_snapshots):
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(f"物理量 {var_name} 的单样本快照自由度不一致")
            snapshots_by_var[var_name] = normalized_snapshots
            n_dofs[var_name] = dof_count

        return {
            # Completes the multi-line `collection` arguments or elements.
            "schema_version": 1,
            "var_names": var_names,
            "n_dofs": n_dofs,
            "snapshots": snapshots_by_var,
        # Completes the multi-line `collection` arguments or elements.
        }

    def fit_pod_from_snapshot_paths(self, snapshot_paths: List[str]) -> None:
        """从持久化快照分片恢复全局 POD 模态。"""
        if not snapshot_paths:
            raise ValueError("POD 快照分片为空，无法执行模态拟合")

        # Stores the `collected_data` on the instance so other methods can reuse it.
        self.collected_data = {}
        self.n_dofs = {}
        self.n_snapshots = 0
        self.pod_modes = {}
        # Stores the `singular_values` on the instance so other methods can reuse it.
        self.singular_values = {}
        self.mean_fields = {}
        self.modal_coefficients = {}
        self.coeff_sizes = {}
        # Stores the `coeff_slices` on the instance so other methods can reuse it.
        self.coeff_slices = {}
        self.retained_energy_ratio = {}

        expected_var_names: Optional[List[str]] = None
        for snapshot_path in snapshot_paths:
            # Stores the `payload` for use by the remaining steps in this flow.
            payload = load_torch(snapshot_path, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict):
                raise ValueError(f"POD 快照分片格式非法: {snapshot_path}")
            var_names = [str(name) for name in payload.get("var_names", [])]
            # Reads `snapshots` from the mapping into the `snapshots_by_var` with its declared fallback.
            snapshots_by_var = payload.get("snapshots", {})
            n_dofs = payload.get("n_dofs", {})
            if not var_names or not isinstance(snapshots_by_var, dict):
                raise ValueError(f"POD 快照分片缺少必要字段: {snapshot_path}")
            # Checks `expected_var_names is None` before entering this conditional branch.
            if expected_var_names is None:
                expected_var_names = var_names
                self.var_names = list(var_names)
            elif var_names != expected_var_names:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(f"POD 快照分片变量顺序不一致: {snapshot_path}")

            for var_name in self.var_names:
                snapshots = snapshots_by_var.get(var_name)
                if not isinstance(snapshots, list) or not snapshots:
                    # Raises `ValueError` to report why this path cannot continue.
                    raise ValueError(f"POD 快照分片缺少物理量 {var_name}: {snapshot_path}")
                expected_dof = int(n_dofs.get(var_name, -1))
                if expected_dof <= 0:
                    raise ValueError(f"POD 快照分片缺少物理量 {var_name} 的自由度: {snapshot_path}")
                # Checks `var_name not in self.n_dofs` before entering this conditional branch.
                if var_name not in self.n_dofs:
                    self.n_dofs[var_name] = expected_dof
                elif self.n_dofs[var_name] != expected_dof:
                    raise ValueError(
                        # Completes the multi-line `ValueError` arguments or elements.
                        f"物理量 {var_name} 的 POD 快照自由度不一致: "
                        f"{self.n_dofs[var_name]} vs {expected_dof}"
                    )
                bucket = self.collected_data.setdefault(var_name, [])
                # Visits each item in `snapshots` and applies the current processing step.
                for snapshot in snapshots:
                    array = np.asarray(snapshot)
                    if int(array.shape[0]) != expected_dof:
                        raise ValueError(f"物理量 {var_name} 的 POD 快照维度非法: {snapshot_path}")
                    # Adds the current value to `bucket` for later aggregation.
                    bucket.append(array)

        self.n_snapshots = len(self.collected_data.get(self.var_names[0], []))
        self.perform_pod()

    def encode_sample_with_pod(self, data: BaseData) -> Dict:
        """使用已拟合或恢复的 POD 模态编码单样本。"""
        # Validates `not self.pod_modes` and raises `ValueError` when the input cannot be processed safely.
        if not self.pod_modes:
            raise ValueError("尚未拟合或恢复 POD 模态，无法编码样本")
        return self.process(data)
    

    
    def perform_pod(self) -> None:
        """执行SVD分解并计算模态
        """
        # Validates `not self.collected_data` and raises `ValueError` when the input cannot be processed safely.
        if not self.collected_data:
            raise ValueError("未收集到任何数据，无法执行POD")

        for var_name, snapshots in self.collected_data.items():
            if not snapshots:
                # Handles the data or control-flow case represented by this statement.
                continue

            matrix = np.column_stack(snapshots)  # (N_dof, n_snapshots)
            matrix_tensor = torch.tensor(matrix, dtype=torch.float32)

            mean_field = torch.mean(matrix_tensor, dim=1, keepdim=True)
            # Derives the `centered` from the current values for downstream processing.
            centered = matrix_tensor - mean_field

            # SVD分解
            u, s, vh = torch.linalg.svd(centered, full_matrices=False)

            var_rank = self._select_rank_by_energy(s, var_name, max_rank=u.shape[1])

            u_r = u[:, :var_rank]
            # Selects the required item from `s` as the `s_r`.
            s_r = s[:var_rank]
            vh_r = vh[:var_rank, :]

            # 模态系数 A = S * V^T
            coeff = torch.diag(s_r) @ vh_r

            self.pod_modes[var_name] = u_r
            # Stores the `singular_values[var_name]` for use by the remaining steps in this flow.
            self.singular_values[var_name] = s_r
            self.mean_fields[var_name] = mean_field.squeeze(1)
            self.modal_coefficients[var_name] = coeff
            self.coeff_sizes[var_name] = int(var_rank)

            # Stores the `total_energy` for use by the remaining steps in this flow.
            total_energy = torch.sum(s ** 2)
            kept_energy = torch.sum(s_r ** 2)
            if total_energy.item() > 0:
                self.retained_energy_ratio[var_name] = float((kept_energy / total_energy).item())
            # Continues the surrounding statement or collection definition.
            else:
                self.retained_energy_ratio[var_name] = 1.0

        self._build_coeff_slices()
    
    def save_pod_modes(self, save_path: str) -> None:
        """保存模态到文件
        
        Args:
            save_path: 保存路径
        """
        # Validates `not self.pod_modes` and raises `ValueError` when the input cannot be processed safely.
        if not self.pod_modes:
            raise ValueError("POD模态为空，无法保存")

        payload = {
            'modes': self.pod_modes,
            # Completes the multi-line `collection` arguments or elements.
            'mean_fields': self.mean_fields,
            'var_names': self.var_names,
            'coeff_sizes': self.coeff_sizes,
            'coeff_slices': self.coeff_slices,
        # Completes the multi-line `collection` arguments or elements.
        }
        atomic_torch_save(payload, save_path)

    def save_processing_state(
        self,
        # Continues the surrounding statement or collection definition.
        stage: str,
        processed_dir: str,
        processed_paths: List[str],
    ) -> None:
        """Persist POD artifacts required by normalization and evaluation."""
        # Returns early when `stage != 'train'` makes the remaining work inapplicable.
        if stage != 'train':
            return
        if not self.pod_modes or not self.mean_fields or not self.coeff_sizes:
            raise ValueError("MatrixProcessor missing POD artifacts after training preprocess")
        # Stores the `modes_path` for use by the remaining steps in this flow.
        modes_path = os.path.join(processed_dir, 'modes.pt')
        self.save_pod_modes(modes_path)
    
    def load_pod_modes(self, load_path: str) -> None:
        """从文件加载模态
        
        Args:
            load_path: 加载路径
        """
        payload = load_torch(load_path, map_location='cpu', weights_only=False)
        # Initializes the `required_keys` as a collection for values from this processing stage.
        required_keys = ['modes', 'mean_fields', 'var_names', 'coeff_sizes', 'coeff_slices']
        missing = [k for k in required_keys if k not in payload]
        if missing:
            raise ValueError(f"modes文件缺少必要键: {missing}")

        # Stores the `pod_modes` on the instance so other methods can reuse it.
        self.pod_modes = payload['modes']
        self.mean_fields = payload['mean_fields']
        self.var_names = payload['var_names']
        self.coeff_sizes = payload['coeff_sizes']
        # Stores the `coeff_slices` on the instance so other methods can reuse it.
        self.coeff_slices = payload['coeff_slices']
    
    def process(self, data: Union[BaseData, List[BaseData]]) -> Dict:
        """处理场数据，返回矩阵格式（模态系数）
        """

        # 训练阶段：先收集并执行POD，再逐样本处理
        if isinstance(data, list):            
            self.collect_data(data)
            # Checks `not self.pod_modes` before entering this conditional branch.
            if not self.pod_modes:
                self.perform_pod()
            return [self.process(item) for item in data]
        
        # 单样本处理逻辑
        if not self.pod_modes:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("尚未执行POD或未加载POD模态，无法处理样本")

        sample_params = self._extract_sample_params(data)
        x_tensor = torch.tensor(sample_params, dtype=torch.float32).unsqueeze(0)

        data_dict = data.get_data()
        # Reads `Fields` from the mapping into the `fields` with its declared fallback.
        fields = data_dict.get('Fields', {})

        if data.get_mode() == 'steady':
            coeff = self._compute_coefficients_for_time(fields, time_index=None)
            y_tensor = torch.tensor(coeff, dtype=torch.float32)
            # Stores the `processed_data` for use by the remaining steps in this flow.
            processed_data = self._build_processed_payload(
                x_tensor=x_tensor,
                y_tensor=y_tensor,
                meta=self._build_processed_sample_meta(data=data, y=y_tensor, time_axis=None),
            # Completes the multi-line `self._build_processed_payload` arguments or elements.
            )
            return processed_data

        # temporal: 统一输出完整时间序列系数，不在此处理direct/autoregressive策略
        time_steps = self._get_time_steps(fields)
        coeffs = []
        # Visits each item in `range(time_steps)` and applies the current processing step.
        for t in range(time_steps):
            coeffs.append(self._compute_coefficients_for_time(fields, time_index=t))
        y_tensor = torch.tensor(np.stack(coeffs), dtype=torch.float32)

        processed_data = self._build_processed_payload(
            # Completes the multi-line `self._build_processed_payload` arguments or elements.
            x_tensor=x_tensor,
            y_tensor=y_tensor,
            meta=self._build_processed_sample_meta(data=data, y=y_tensor, time_axis=0),
        )
        # Returns `processed_data` to the caller for the next stage.
        return processed_data

    def process_input_only(self, data: BaseData) -> Dict:
        """仅构造预测阶段输入x，并返回与数据集接口兼容的占位y。

        用于预测阶段跳过POD系数计算，仅保留参数输入。
        """
        x_tensor = self._build_input_tensor(data)
        y_tensor = self._build_dummy_target()
        # Stores the `meta` for use by the remaining steps in this flow.
        meta = self._build_processed_sample_meta(data=data, y=y_tensor, time_axis=None)
        if str(data.get_mode()).lower() == "temporal":
            meta = self._build_temporal_direct_predict_meta(meta)
        else:
            # Stores the `meta` for use by the remaining steps in this flow.
            meta = self._build_steady_input_only_meta(meta)
        return self._build_processed_payload(
            x_tensor=x_tensor,
            y_tensor=y_tensor,
            # Completes the multi-line `self._build_processed_payload` arguments or elements.
            meta=meta,
        )

    def process_predict_sample(
        self,
        # Continues the surrounding statement or collection definition.
        data: BaseData,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.temporal_strategy == "autoregressive":
            # Converts the `input_length` to `int` so later checks use a consistent type.
            input_length = int(self.config.get("task", {}).get("input_length", 1))
            if input_length <= 1:
                return self._build_zero_autoregressive_predict_seed_sample(data)
            return self._build_autoregressive_predict_seed_sample(data)
        # Returns `self.process_input_only(data)` to the caller for the next stage.
        return self.process_input_only(data)

    def _build_zero_autoregressive_predict_seed_sample(self, data: BaseData) -> Dict:
        """构建使用零初始场的 autoregressive 预测样本。"""
        processed = self.process_input_only(data)
        dummy_target = self._build_dummy_target()
        # Stores the `seed_tensor` for use by the remaining steps in this flow.
        seed_tensor = dummy_target.reshape(1, -1).clone()
        meta = self._build_temporal_rollout_seed_meta(
            processed.get("meta", {}),
            seed_length=1,
            # Completes the multi-line `self._build_temporal_rollout_seed_meta` arguments or elements.
            init_history_strategy="zero",
            total_time_steps=1,
        )
        return self._build_processed_payload(
            # Completes the multi-line `self._get_payload_x` arguments or elements.
            x_tensor=self._get_payload_x(processed),
            y_tensor=dummy_target,
            meta=meta,
            y_seed=seed_tensor,
        # Completes the multi-line `self._build_processed_payload` arguments or elements.
        )

    def _build_autoregressive_predict_seed_sample(self, data: BaseData) -> Dict:
        """构建仅包含初始 history 的 autoregressive 预测样本。"""
        processed = self.process(data)
        x_tensor = self._get_payload_x(processed)
        # Stores the `y_tensor` for use by the remaining steps in this flow.
        y_tensor = self._get_payload_y(processed)
        if y_tensor.ndim != 2:
            raise ValueError(
                f"MatrixProcessor autoregressive 预测样本期望完整 y 形状为 [T, C]，当前为 {tuple(y_tensor.shape)}"
            # Completes the multi-line `ValueError` arguments or elements.
            )

        input_length = int(self.config.get("task", {}).get("input_length", 1))
        if int(y_tensor.shape[0]) < input_length:
            raise ValueError(
                # Completes the multi-line `ValueError` arguments or elements.
                f"MatrixProcessor 预测 history 长度不足："
                f" total_steps={y_tensor.shape[0]}, input_length={input_length}"
            )

        seed_tensor = y_tensor[:input_length].clone()
        # Stores the `dummy_target` for use by the remaining steps in this flow.
        dummy_target = self._build_dummy_target()
        meta = self._build_temporal_rollout_seed_meta(
            processed.get("meta", {}),
            seed_length=input_length,
            # Completes the multi-line `int` arguments or elements.
            total_time_steps=int(y_tensor.shape[0]),
        )
        return self._build_processed_payload(
            x_tensor=x_tensor,
            # Completes the multi-line `self._build_processed_payload` arguments or elements.
            y_tensor=dummy_target,
            meta=meta,
            y_seed=seed_tensor,
        )

    # Defines `build_autoregressive_view` for build autoregressive view logic.
    def build_autoregressive_view(
        self,
        sample: Dict[str, Any],
        start_t: int,
        # Continues the surrounding statement or collection definition.
        input_length: int,
        output_length: int,
    ) -> Tuple[Any, Any]:
        x = sample["x"]
        # Selects the required item from `sample` as the `y`.
        y = sample["y"]
        meta = sample.get("meta", {})
        time_axis = int(meta.get("time_axis", 0))

        if hasattr(x, "shape") and len(x.shape) > 0 and int(x.shape[0]) >= start_t + input_length:
            # Selects the required item from `x` as the `x_seq`.
            x_seq = x[start_t : start_t + input_length]
        else:
            history = self._slice_temporal_tensor(y, start_t, start_t + input_length, time_axis)
            history = self._move_time_axis_to_front(history, time_axis)
            # Checks `x.ndim == 1` before entering this conditional branch.
            if x.ndim == 1:
                static_context = x.unsqueeze(0).expand(input_length, -1)
            elif x.ndim == 2 and x.shape[0] == 1:
                static_context = x.expand(input_length, -1)
            # Continues the surrounding statement or collection definition.
            else:
                raise ValueError(
                    "MatrixProcessor autoregressive 视图要求 x 为 [P] 或 [1, P]，"
                    f"当前形状为 {tuple(x.shape)}"
                # Completes the multi-line `ValueError` arguments or elements.
                )
            x_seq = torch.cat([static_context, history.reshape(input_length, -1)], dim=-1)

        y_seq = self._slice_temporal_tensor(
            y,
            # Completes the multi-line `self._slice_temporal_tensor` arguments or elements.
            start_t + input_length,
            start_t + input_length + output_length,
            time_axis,
        )
        # Stores the `y_seq` for use by the remaining steps in this flow.
        y_seq = self._move_time_axis_to_front(y_seq, time_axis)
        return x_seq, y_seq

    def build_autoregressive_predict_seed(
        self,
        # Completes the multi-line `collection` arguments or elements.
        sample: Dict[str, Any],
        input_length: int,
    ) -> Any:
        x = sample["x"]
        # Reads `y_seed` from the mapping into the `y_seed`.
        y_seed = sample.get("y_seed")
        if y_seed is None:
            return super().build_autoregressive_predict_seed(
                sample=sample,
                # Completes the multi-line `super().build_autoregressive_predict_seed` arguments or elements.
                input_length=input_length,
            )

        history = y_seed.float()
        if history.ndim != 2:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(
                f"MatrixProcessor 预测 seed 期望 y_seed 形状为 [K, C]，当前为 {tuple(history.shape)}"
            )
        if int(history.shape[0]) != int(input_length):
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(
                f"MatrixProcessor 预测 seed 的历史长度与 input_length 不一致："
                f" got={history.shape[0]}, expected={input_length}"
            )

        # Checks `x.ndim == 1` before entering this conditional branch.
        if x.ndim == 1:
            static_context = x.unsqueeze(0).expand(input_length, -1)
        elif x.ndim == 2 and x.shape[0] == 1:
            static_context = x.expand(input_length, -1)
        # Continues the surrounding statement or collection definition.
        else:
            raise ValueError(
                "MatrixProcessor 预测 seed 要求 x 为 [P] 或 [1, P]，"
                f"当前形状为 {tuple(x.shape)}"
            # Completes the multi-line `ValueError` arguments or elements.
            )
        return torch.cat([static_context, history.reshape(input_length, -1)], dim=-1)

    def predict_autoregressive_rollout(
        self,
        # Continues the surrounding statement or collection definition.
        model,
        inputs: Any,
        rollout_steps: int,
    ) -> Any:
        """执行 MatrixProcessor 的滚动预测，并显式维护静态上下文与 history。"""
        # Validates `not isinstance(inputs, torch.Tensor)` and raises `TypeError` when the input cannot be processed safely.
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("MatrixProcessor autoregressive rollout 仅支持 torch.Tensor 输入")
        if inputs.dim() != 3:
            raise ValueError(
                # Completes the multi-line `tuple` arguments or elements.
                f"MatrixProcessor autoregressive rollout 要求输入形状为 [B, T, F]，当前为 {tuple(inputs.shape)}"
            )

        steps = int(rollout_steps)
        if steps <= 0:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(f"rollout_steps 必须为正整数，当前为 {rollout_steps}")

        current_input = inputs
        batch_predictions = []

        for _ in range(steps):
            # Stores the output for use by the remaining steps in this flow.
            output = model(current_input)
            if output.dim() == 3 and output.shape[1] == 1:
                output_step = output[:, 0, :]
            elif output.dim() == 2:
                # Stores the `output_step` for use by the remaining steps in this flow.
                output_step = output
            else:
                raise ValueError(
                    "MatrixProcessor 当前仅支持单步 autoregressive 输出，"
                    # Completes the multi-line `tuple` arguments or elements.
                    f"模型输出形状为 {tuple(output.shape)}"
                )

            history_dim = int(output_step.shape[-1])
            feature_dim = int(current_input.shape[-1])
            # Derives the `static_dim` from the current values for downstream processing.
            static_dim = feature_dim - history_dim
            if static_dim < 0:
                raise ValueError(
                    f"MatrixProcessor 无法从输入维度推断静态特征数：feature_dim={feature_dim}, history_dim={history_dim}"
                # Completes the multi-line `ValueError` arguments or elements.
                )

            static_context = current_input[..., :static_dim] if static_dim > 0 else current_input[..., :0]
            history = current_input[..., static_dim:]
            state_step = self.restore_autoregressive_state(history[:, -1, :], output_step)
            # Stores the `next_history` for use by the remaining steps in this flow.
            next_history = torch.cat([history[:, 1:, :], state_step.unsqueeze(1)], dim=1)
            current_input = torch.cat([static_context, next_history], dim=-1)
            batch_predictions.append(state_step)

        return torch.stack(batch_predictions, dim=1)

    # 按照 MatrixProcessor 约定提取数值型样本参数。
    def _extract_numeric_sample_params(self, sample_info: Dict) -> List[float]:
        """按照 MatrixProcessor 约定提取数值型样本参数。"""
        params: List[float] = []
        for key in sorted(sample_info.keys()):
            if key.lower() in self._NON_PARAM_SAMPLE_KEYS:
                # Handles the data or control-flow case represented by this statement.
                continue
            value = sample_info[key]
            if isinstance(value, (int, float)):
                params.append(float(value))
        # Returns `params` to the caller for the next stage.
        return params

    def _create_filtered_samples_csv(
        self,
        source_csv: str,
        # Continues the surrounding statement or collection definition.
        output_csv: str,
        keep_ids: List[int],
    ) -> str:
        """按样本 ID 子集生成过滤后的样本表。"""
        import csv
        # Imports `SampleMap` from `readers.sample_map` for use in this module.
        from ...readers.sample_map import SampleMap

        keep_set = set(keep_ids)
        sample_map = SampleMap(source_csv)
        all_ids = [sid for sid in sample_map.get_all_samples() if sid in keep_set]
        # Validates `not all_ids` and raises `ValueError` when the input cannot be processed safely.
        if not all_ids:
            raise ValueError("过滤后样本为空，无法生成有效样本表")

        first_info = sample_map.get_sample_info(all_ids[0]) or {}
        fieldnames = ['order'] + [k for k in first_info.keys()]

        # Stores the output directory for use by the remaining steps in this flow.
        output_dir = os.path.dirname(output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            # Stores the `writer` for use by the remaining steps in this flow.
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for sample_id in all_ids:
                info = sample_map.get_sample_info(sample_id) or {}
                # Initializes the `row` as a collection for values from this processing stage.
                row = {'order': sample_id}
                row.update(info)
                writer.writerow(row)

        return output_csv

    # 保存处理后的数据为pt文件
    def save_to_pt(self, processed_data: Dict, save_path: str) -> None:
        """保存处理后的数据为pt文件

        Args:
            processed_data: 处理后的数据
            save_path: 保存路径
        """
        x_tensor = self._get_payload_x(processed_data)
        y_tensor = self._get_payload_y(processed_data)
        unified_data = {
            # Completes the multi-line `collection` arguments or elements.
            'x': x_tensor,
            'y': y_tensor,
            'meta': processed_data.get('meta', {}),
        }
        # Checks `'y_seed' in processed_data` before entering this conditional branch.
        if 'y_seed' in processed_data:
            unified_data['y_seed'] = processed_data['y_seed']
        atomic_torch_save(unified_data, save_path)

    def reconstruct(self, coefficients: Union[np.ndarray, torch.Tensor]) -> Dict[str, np.ndarray]:
        """根据预测得到的拼接模态系数重构物理场。

        Args:
            coefficients: 模态系数，支持以下形状：
                - (r_total,): 单时间步
                - (T, r_total): 多时间步

        Returns:
            物理场字典：
                - 单时间步: {var_name: (N_i,)}
                - 多时间步: {var_name: (T, N_i)}
        """
        # Validates `not self.pod_modes or not self.mean_fields` and raises `ValueError` when the input cannot be processed safely.
        if not self.pod_modes or not self.mean_fields:
            raise ValueError("尚未加载或计算POD模态，无法执行重构")

        if not self.coeff_slices:
            self._build_coeff_slices()

        # Stores the `coeff_np` for use by the remaining steps in this flow.
        coeff_np = coefficients.detach().cpu().numpy() if isinstance(coefficients, torch.Tensor) else np.asarray(coefficients)
        if coeff_np.ndim not in (1, 2):
            raise ValueError(f"coefficients维度必须为1或2，当前为{coeff_np.ndim}")

        expected_dim = self._get_total_coeff_dim()
        # Validates `coeff_np.shape[-1] != expected_dim` and raises `ValueError` when the input cannot be processed safely.
        if coeff_np.shape[-1] != expected_dim:
            raise ValueError(
                f"系数最后一维与期望不一致: got={coeff_np.shape[-1]}, expected={expected_dim}"
            )

        # Returns early when `coeff_np.ndim == 1` makes the remaining work inapplicable.
        if coeff_np.ndim == 1:
            return self._reconstruct_single_step(coeff_np)

        # (T, r_total)
        recon: Dict[str, List[np.ndarray]] = {name: [] for name in self.var_names}
        for t in range(coeff_np.shape[0]):
            # Stores the `step_result` for use by the remaining steps in this flow.
            step_result = self._reconstruct_single_step(coeff_np[t])
            for var_name in self.var_names:
                recon[var_name].append(step_result[var_name])

        return {name: np.stack(values, axis=0) for name, values in recon.items()}

    # Defines `_get_field_array` for  get field array logic.
    def _get_field_array(self, fields: Dict, var_name: str) -> np.ndarray:
        if var_name in fields:
            arr = fields[var_name]
        else:
            # 尝试忽略大小写匹配
            # Stores the `match` for use by the remaining steps in this flow.
            match = None
            for key in fields.keys():
                if key.lower() == var_name.lower():
                    match = key
                    # Handles the data or control-flow case represented by this statement.
                    break
            if match is None:
                raise KeyError(f"Fields中不存在物理量: {var_name}")
            arr = fields[match]

        # Checks `isinstance(arr, torch.Tensor)` before entering this conditional branch.
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        return np.asarray(arr)

    def _extract_snapshots(self, field_array: np.ndarray, mode: str) -> List[np.ndarray]:
        # Initializes the `snapshots` as a collection for values from this processing stage.
        snapshots: List[np.ndarray] = []
        if mode == 'steady':
            snapshots.append(self._flatten_field(field_array))
            return snapshots

        # temporal
        # Checks `field_array.ndim == 1` before entering this conditional branch.
        if field_array.ndim == 1:
            snapshots.append(self._flatten_field(field_array))
            return snapshots

        time_dim = field_array.shape[0]
        # Visits each item in `range(time_dim)` and applies the current processing step.
        for t in range(time_dim):
            snapshots.append(self._flatten_field(field_array[t]))
        return snapshots

    def _flatten_field(self, field_slice: np.ndarray) -> np.ndarray:
        # Returns `field_slice.reshape(-1).astype(np.float32)` to the caller for the next stage.
        return field_slice.reshape(-1).astype(np.float32)

    def _extract_sample_params(self, data: BaseData) -> List[float]:
        sample_map = data.get_sample_map() or {}
        params = []
        # Visits each item in `sorted(sample_map.keys())` and applies the current processing step.
        for key in sorted(sample_map.keys()):
            if key.lower() in self._NON_PARAM_SAMPLE_KEYS:
                continue
            value = sample_map[key]
            # Checks `isinstance(value, (int, float))` before entering this conditional branch.
            if isinstance(value, (int, float)):
                params.append(float(value))
        return params

    def _build_input_tensor(self, data: BaseData) -> torch.Tensor:
        # Stores the `sample_params` for use by the remaining steps in this flow.
        sample_params = self._extract_sample_params(data)
        return torch.tensor(sample_params, dtype=torch.float32).unsqueeze(0)

    def _build_dummy_target(self) -> torch.Tensor:
        output_shape = self.config.get('model', {}).get('output_shape', None)
        # Returns early when `output_shape is not None and len(output_shape) > 0` makes the remaining work inapplicable.
        if output_shape is not None and len(output_shape) > 0:
            return torch.zeros(tuple(output_shape), dtype=torch.float32)

        raise ValueError("预测阶段构造dummy_target失败：缺少model.output_shape配置")

    def _get_time_steps(self, fields: Dict) -> int:
        # Checks `not self.var_names` before entering this conditional branch.
        if not self.var_names:
            self.var_names = list(fields.keys())
        first_var = self.var_names[0]
        arr = self._get_field_array(fields, first_var)
        # Checks `self.mode == 'temporal'` before entering this conditional branch.
        if self.mode == 'temporal':
            # temporal模式下，常见形状为(T, N)或(T, N, C)
            if arr.ndim >= 2:
                return arr.shape[0]
            return 1

        # steady模式默认单时间步
        # Returns `1` to the caller for the next stage.
        return 1

    def _compute_coefficients_for_time(self, fields: Dict, time_index: Optional[int]) -> np.ndarray:
        coeffs = []
        for var_name in self.var_names:
            # Stores the `field_array` for use by the remaining steps in this flow.
            field_array = self._get_field_array(fields, var_name)
            if time_index is None:
                # steady或显式全量输入
                slice_array = field_array
            else:
                # temporal：按时间步取快照
                # Checks `field_array.ndim >= 2` before entering this conditional branch.
                if field_array.ndim >= 2:
                    slice_array = field_array[time_index]
                else:
                    slice_array = field_array

            # Stores the `snapshot` for use by the remaining steps in this flow.
            snapshot = self._flatten_field(slice_array)
            mean_field = self.mean_fields[var_name]
            modes = self.pod_modes[var_name]

            # 防御性检查，便于定位数据维度问题
            if snapshot.shape[0] != mean_field.shape[0]:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(
                    f"物理量{var_name}快照维度与均值场不一致: "
                    f"snapshot={snapshot.shape[0]}, mean={mean_field.shape[0]}"
                )

            # Stores the `snapshot_tensor` for use by the remaining steps in this flow.
            snapshot_tensor = torch.tensor(snapshot, dtype=torch.float32)
            centered = snapshot_tensor - mean_field
            coeff = torch.matmul(modes.t(), centered).detach().cpu().numpy()
            coeffs.append(coeff)

        # Returns `np.concatenate(coeffs, axis=0)` to the caller for the next stage.
        return np.concatenate(coeffs, axis=0)

    def _select_rank_by_energy(self, singular_values: torch.Tensor, var_name: str, max_rank: int) -> int:
        """根据固定能量占比选择POD模态数。

        所有物理量统一采用0.99999累计能量占比阈值。
        """
        sv = singular_values
        if sv.numel() == 0:
            # Returns `0` to the caller for the next stage.
            return 0

        ratio = float(self.fixed_energy_ratio)
        if ratio <= 0 or ratio > 1:
            raise ValueError(f"物理量{var_name}的fixed_energy_ratio必须在(0,1]区间，当前为{ratio}")

        # Derives the `energy` from the current values for downstream processing.
        energy = sv ** 2
        total = torch.sum(energy)
        if total.item() <= 0:
            return 1

        # Derives the `cumulative` from the current values for downstream processing.
        cumulative = torch.cumsum(energy, dim=0) / total
        # 最小满足累计能量占比的模态数
        rank = int(torch.searchsorted(cumulative, torch.tensor(ratio, dtype=cumulative.dtype), right=False).item()) + 1
        return max(1, min(rank, max_rank))

    def _reconstruct_single_step(self, coeff_vector: np.ndarray) -> Dict[str, np.ndarray]:
        """重构单时间步物理场。"""
        # Initializes the `result` as a collection for values from this processing stage.
        result: Dict[str, np.ndarray] = {}
        for var_name in self.var_names:
            if var_name not in self.coeff_slices:
                raise KeyError(f"缺少物理量{var_name}的系数切片信息")

            # Selects the required item from `self.coeff_slices` as the `(start, end)`.
            start, end = self.coeff_slices[var_name]
            coeff_var = coeff_vector[start:end]

            modes = self.pod_modes[var_name]                 # (N_i, r_i)
            mean_field = self.mean_fields[var_name]          # (N_i,)

            # Stores the `coeff_t` for use by the remaining steps in this flow.
            coeff_t = torch.tensor(coeff_var, dtype=torch.float32)
            recon = torch.matmul(modes, coeff_t) + mean_field
            result[var_name] = recon.detach().cpu().numpy()

        return result

    # 获取拼接后总系数维度。
    def _get_total_coeff_dim(self) -> int:
        """获取拼接后总系数维度。"""
        if not self.coeff_sizes:
            raise ValueError("coeff_sizes为空，无法确定总系数维度")
        return int(sum(self.coeff_sizes.values()))

    # 构建拼接系数向量中各物理量的切片范围。
    def _build_coeff_slices(self) -> None:
        """构建拼接系数向量中各物理量的切片范围。"""
        slices: Dict[str, List[int]] = {}
        start = 0
        for var_name in self.var_names:
            # Converts the `size` to `int` so later checks use a consistent type.
            size = int(self.coeff_sizes.get(var_name, 0))
            end = start + size
            slices[var_name] = [start, end]
            start = end
        # Stores the `coeff_slices` on the instance so other methods can reuse it.
        self.coeff_slices = slices

    def _ensure_eval_artifacts_loaded(self, config: Dict[str, Any]) -> None:
        """按需加载评估阶段所需的归一化参数与 POD 模态。"""
        if self._eval_norm_params is not None and self.pod_modes and self.mean_fields:
            return

        # Selects the required item from `config['general']` as the `processed_dir`.
        processed_dir = config['general']['train_processed_dir']
        norm_path = os.path.join(processed_dir, 'norm_params.pt')
        modes_path = os.path.join(processed_dir, 'modes.pt')

        if not os.path.exists(norm_path):
            # Raises `FileNotFoundError` to report why this path cannot continue.
            raise FileNotFoundError(f"未找到归一化参数文件: {norm_path}")
        if not os.path.exists(modes_path):
            raise FileNotFoundError(f"未找到 POD 模态文件: {modes_path}")

        self._eval_norm_params = load_torch(norm_path, map_location='cpu', weights_only=False)
        # Checks `not self.pod_modes or not self.mean_fields` before entering this conditional branch.
        if not self.pod_modes or not self.mean_fields:
            self.load_pod_modes(modes_path)

    def _reconstruct_eval_batch(self, coeff_batch: np.ndarray) -> Dict[str, np.ndarray]:
        """将一批系数重构为按物理量分组的数据。"""
        by_var: Dict[str, List[np.ndarray]] = {}
        # Visits each item in `coeff_batch` and applies the current processing step.
        for sample_coeff in coeff_batch:
            recon = self.reconstruct(sample_coeff)
            for var_name in self.var_names:
                if var_name not in recon:
                    # Handles the data or control-flow case represented by this statement.
                    continue
                by_var.setdefault(var_name, []).append(np.asarray(recon[var_name]))

        if not by_var:
            raise ValueError("POD 重构结果为空，无法执行评估")

        # Returns `` to the caller for the next stage.
        return {name: np.stack(values, axis=0) for name, values in by_var.items()}
