import copy
import hashlib
import logging
# Imports `Callable, Dict, List` from `typing` for use in this module.
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import math
import os
import time

# Imports `matplotlib` for reuse in this module.
import matplotlib
import torch
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Imports `FuncFormatter` from `matplotlib.ticker` for use in this module.
from matplotlib.ticker import FuncFormatter
from torch.utils.data import Dataset

from tfai.training.optimization import build_lr_scheduler, restore_lr_scheduler
from tfai.dataset import DataLoader
from tfai.dataset.dataset_builder import should_use_graph_partition
# Imports `capture_rng_state, restore_rng_state` from `tfai.utils.random_utils` for use in this module.
from tfai.utils.random_utils import capture_rng_state, restore_rng_state
from tfai.utils.runtime_paths import sanitize_config_for_checkpoint
from tfai.utils.torch_utils import atomic_torch_save, load_torch, resolve_torch_device


def get_processor_checkpoint_artifact_policy(processor) -> Dict[str, bool]:
    """返回当前处理器在 checkpoint 中需要携带的附属产物策略。"""
    # Stores the `processor_type` for use by the remaining steps in this flow.
    processor_type = type(processor).__name__ if processor is not None else ""
    return {
        "modes": processor_type == "MatrixProcessor",
        "processor_state": bool(processor is not None and hasattr(processor, "load_state")),
    # Completes the multi-line `collection` arguments or elements.
    }


class _IndexedDataset(Dataset):
    """以固定索引顺序提供数据集视图，并保留 DataLoader 所需的类型标记。"""

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        # Stores the `indices` on the instance so other methods can reuse it.
        self.indices = [int(index) for index in indices]
        for attribute in ("data_kind", "model_type", "sample_view"):
            if hasattr(dataset, attribute):
                setattr(self, attribute, getattr(dataset, attribute))

    # Defines `__len__` for   len   logic.
    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]

    # Defines `get_sample_meta` for get sample meta logic.
    def get_sample_meta(self, index: int) -> Dict:
        if not hasattr(self.dataset, "get_sample_meta"):
            return {}
        return self.dataset.get_sample_meta(self.indices[index]) or {}


# Defines `Trainer` to group related data and behavior.
class Trainer:
    def __init__(
        self,
        model,
        # Continues the surrounding statement or collection definition.
        criterion: callable,
        optimizer: callable,
        config,
        processor=None,
        # Continues the surrounding statement or collection definition.
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        # Stores the `optimizer` on the instance so other methods can reuse it.
        self.optimizer = optimizer
        self.config = config
        self.processor = processor
        self.logger = logger or logging.getLogger(__name__)
        # Stores the `processed_data_path` on the instance so other methods can reuse it.
        self.processed_data_path = config["general"]["train_processed_dir"]
        self.training_params = config["training"]
        self.lr_scheduler = build_lr_scheduler(self.optimizer, self.training_params)
        self.task_config = config["task"]

        self.batch_size = int(self.training_params["batch_size"])
        # Validates `self.batch_size <= 0` and raises `ValueError` when the input cannot be processed safely.
        if self.batch_size <= 0:
            raise ValueError(f"training.batch_size must be >= 1, got {self.batch_size}")
        self.epochs = self._parse_optional_positive_int(
            self.training_params.get("epochs"),
            # Completes the multi-line `self._parse_optional_positive_int` arguments or elements.
            "training.epochs",
        )
        self.max_steps = self._parse_optional_positive_int(
            self.training_params.get("max_steps"),
            # Completes the multi-line `self._parse_optional_positive_int` arguments or elements.
            "training.max_steps",
        )
        if self.epochs is None and self.max_steps is None:
            raise ValueError("training.epochs 或 training.max_steps 至少需要提供一个正整数")
        # Stores the `validate_every` on the instance so other methods can reuse it.
        self.validate_every = self._parse_optional_positive_int(
            self.training_params.get("validate_every"),
            "training.validate_every",
        )
        # Stores the `validation_max_batches` on the instance so other methods can reuse it.
        self.validation_max_batches = self._parse_validation_max_batches(
            self.training_params.get("validation_max_batches")
        )
        loader_cfg = self.training_params.get("data_loader", {})
        # Stores the `train_num_workers` on the instance so other methods can reuse it.
        self.train_num_workers = int(loader_cfg.get("num_workers", 0) or 0)
        self.valid_num_workers = int(loader_cfg.get("valid_num_workers", 0) or 0)
        processor_type = type(processor).__name__ if processor is not None else None
        self.graph_use_partition = should_use_graph_partition(
            # Completes the multi-line `should_use_graph_partition` arguments or elements.
            config=self.config,
            processor_type=processor_type,
            processor=processor,
        )
        # Stores the `graph_partition_count` on the instance so other methods can reuse it.
        self.graph_partition_count = int(self.training_params.get("graph_partition_count", 1) or 1)
        if self.graph_use_partition and self.graph_partition_count < 2:
            raise ValueError("training.graph_partition_count must be >= 2 when partition training is enabled")
        self.full_graph_equivalent_batch_size = self.batch_size
        # Stores the `train_batch_size` on the instance so other methods can reuse it.
        self.train_batch_size = self._resolve_train_batch_size()
        self.valid_batch_size = self.batch_size

        early_stopping_config = self.training_params.get("early_stopping", {})
        self.early_stopping = int(early_stopping_config.get("patience", 0) or 0)
        # Validates `self.early_stopping < 0` and raises `ValueError` when the input cannot be processed safely.
        if self.early_stopping < 0:
            raise ValueError("training.early_stopping.patience must be >= 0")
        self.patience = self.early_stopping
        self.monitor = str(early_stopping_config.get("monitor", "val_loss"))
        # Stores the `early_stopping_counter` on the instance so other methods can reuse it.
        self.early_stopping_counter = 0
        # 当前最佳模型固定按验证损失选择，和计划内验证及早停口径保持一致。
        self.best_score = float("inf")

        self.loss_curve_log_scale = bool(self.training_params.get("loss_curve_log_scale", False))

        self.task_type = self.task_config["type"]
        # Stores the `mode` on the instance so other methods can reuse it.
        self.mode = self.task_config["mode"]
        self.total_time_steps = self.task_config.get("total_time_steps")
        self.time_step_size = self.task_config.get("time_step_size")
        self.temporal_strategy = self.task_config["temporal_strategy"]
        # Stores the `input_length` on the instance so other methods can reuse it.
        self.input_length = self.task_config["input_length"]
        self.output_length = self.task_config["output_length"]
        self.autoregressive_prediction_mode = self._resolve_effective_autoregressive_prediction_mode(
            task_config=self.task_config,
            # Completes the multi-line `getattr` arguments or elements.
            declared_mode=getattr(self.processor, "autoregressive_prediction_mode", None),
        )
        self.temporal_rollout = self._resolve_temporal_rollout_config()
        if self.graph_use_partition:
            # Validates `` and raises `NotImplementedError` when the input cannot be processed safely.
            if str(self.config.get("model", {}).get("type", "")).lower() != "gcn":
                raise NotImplementedError("graph_partition 首版仅支持 GCN（MeshGraphNet）模型")

        self.device = resolve_torch_device(self.config)
        self.model.to(self.device)

        # Stores the path on the instance so other methods can reuse it.
        self.path = os.path.join(self.config["general"]["experiment"], "train")
        self.checkpoint_path = config["model"]["checkpoint_path"]
        resume_config = self.training_params.get("resume", {})
        self.resume_path = resume_config.get("path", os.path.join(self.path, "last_checkpoint.pt"))
        # Stores the `seed` on the instance so other methods can reuse it.
        self.seed = int(self.config.get("general", {}).get("seed", 42))
        self.start_epoch = 0
        self.global_step = 0
        self.data_epoch = 0
        # Stores the `batch_in_epoch` on the instance so other methods can reuse it.
        self.batch_in_epoch = 0
        self.steps_per_epoch = 0
        self.resolved_max_steps: Optional[int] = None
        self.resolved_validate_every: Optional[int] = None
        # Stores the `last_validation_step` on the instance so other methods can reuse it.
        self.last_validation_step = 0
        self.validation_checks = 0
        self.validation_manifest: Optional[Dict] = None
        self.train_manifest: Optional[Dict] = None
        # Stores the `resume_exact` on the instance so other methods can reuse it.
        self.resume_exact = True
        self._loaded_steps_per_epoch: Optional[int] = None
        self._loaded_step_checkpoint = False
        self._resume_from_legacy_checkpoint = False
        # Stores the `history` on the instance so other methods can reuse it.
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_units": [],
            # Completes the multi-line `collection` arguments or elements.
            "val_units": [],
            "schedule_unit": "step",
            "train_loss_scope": self.temporal_rollout["loss_scope"],
            "val_loss_scope": self.temporal_rollout["loss_scope"],
            # Completes the multi-line `collection` arguments or elements.
            "rollout_horizon": self.temporal_rollout["horizon"],
        }
        self.train_units: List[int] = []
        self.val_units: List[int] = []
        # Stores the `training_terminated` on the instance so other methods can reuse it.
        self.training_terminated = False
        self.progress_start_timestamp = 0
        self.progress_history: List[Dict] = []
        self.train_loader_generator = torch.Generator()
        # Calls `self.train_loader_generator.manual_seed` to perform the operation required at this stage.
        self.train_loader_generator.manual_seed(self.seed)
        os.makedirs(self.path, exist_ok=True)

    @staticmethod
    def _parse_optional_positive_int(value, field_name: str) -> Optional[int]:
        """解析可空的正整数配置，避免布尔值被静默当作数字。"""
        # Returns early when `value is None` makes the remaining work inapplicable.
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError(f"{field_name} 必须为正整数或 null")
        # Validates `isinstance(value, float) and not value.is_integer()` and raises `ValueError` when the input cannot be processed safely.
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"{field_name} 必须为正整数或 null")
        try:
            parsed = int(value)
        # Completes the multi-line `collection` arguments or elements.
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} 必须为正整数或 null") from exc
        if parsed <= 0:
            raise ValueError(f"{field_name} 必须为正整数或 null")
        # Returns `parsed` to the caller for the next stage.
        return parsed

    @staticmethod
    def _parse_validation_max_batches(value) -> Optional[int]:
        """解析验证批次数上限；空值和非正数均表示完整验证。"""
        if value is None:
            # Returns `None` to the caller for the next stage.
            return None
        if isinstance(value, bool):
            raise ValueError("training.validation_max_batches 必须为整数或 null")
        if isinstance(value, float) and not value.is_integer():
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("training.validation_max_batches 必须为整数或 null")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("training.validation_max_batches 必须为整数或 null") from exc
        return parsed if parsed > 0 else None

    def _resolve_train_batch_size(self) -> int:
        """将配置中的 batch_size 统一解释为完整样本等价批大小。"""
        return self.batch_size

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _is_temporal_autoregressive_task(task_config: Optional[Dict]) -> bool:
        """判断任务是否实际进入时序自回归链路。"""
        task_config = task_config or {}
        return (
            # Completes the multi-line `str(task_config.get('mode', '')).lower` arguments or elements.
            str(task_config.get("mode", "")).lower() == "temporal"
            and str(task_config.get("temporal_strategy", "direct")).lower()
            == "autoregressive"
        )

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _resolve_effective_autoregressive_prediction_mode(
        task_config: Optional[Dict],
        declared_mode: Optional[str] = None,
    # Continues the surrounding statement or collection definition.
    ) -> str:
        """仅为时序自回归任务保留增量预测语义。"""
        task_config = task_config or {}
        if not Trainer._is_temporal_autoregressive_task(task_config):
            return "state"
        # Returns `` to the caller for the next stage.
        return str(
            declared_mode
            if declared_mode is not None
            else task_config.get("autoregressive_prediction_mode", "state")
        # Completes the multi-line `str(declared_mode if declared_mode is not None else task_config.get('autoregressive_prediction_mode', 'state')).lower` arguments or elements.
        ).lower()

    @staticmethod
    def _disabled_temporal_rollout_config() -> Dict:
        """返回非时序自回归任务使用的规范化禁用配置。"""
        return {
            # Completes the multi-line `collection` arguments or elements.
            "enabled": False,
            "horizon": 1,
            "checkpoint_segments": 0,
            "horizon_weighting": "uniform",
            # Completes the multi-line `collection` arguments or elements.
            "horizon_weight_gamma": 1.0,
            "training_noise": False,
            "training_noise_alpha": 0.01,
            "boundary_state_override": False,
            # Completes the multi-line `collection` arguments or elements.
            "feedback_mode": "single_step",
            "loss_scope": "single_step",
        }

    def _resolve_temporal_rollout_config(self) -> Dict:
        """解析时序自由滚动配置，并兼容已废弃的 enabled 字段。"""
        # Stores the `raw_config` for use by the remaining steps in this flow.
        raw_config = self.training_params.get("temporal_rollout", {}) or {}
        if not self._is_temporal_autoregressive_task(self.task_config):
            return self._disabled_temporal_rollout_config()

        horizon = self._parse_optional_positive_int(
            # Completes the multi-line `raw_config.get` arguments or elements.
            raw_config.get("horizon", 1),
            "training.temporal_rollout.horizon",
        )
        training_noise = raw_config.get("training_noise", False)
        # Reads `training_noise_alpha` from the mapping into the `training_noise_alpha` with its declared fallback.
        training_noise_alpha = raw_config.get("training_noise_alpha", 0.01)
        boundary_state_override = raw_config.get("boundary_state_override", False)
        if not isinstance(training_noise, bool):
            raise ValueError("training.temporal_rollout.training_noise 必须为布尔值")
        # Validates `isinstance(training_noise_alpha, bool)` and raises `ValueError` when the input cannot be processed safely.
        if isinstance(training_noise_alpha, bool):
            raise ValueError("training.temporal_rollout.training_noise_alpha 必须为数值")
        try:
            training_noise_alpha = float(training_noise_alpha)
        # Completes the multi-line `collection` arguments or elements.
        except (TypeError, ValueError) as exc:
            raise ValueError("training.temporal_rollout.training_noise_alpha 必须为数值") from exc
        if not math.isfinite(training_noise_alpha) or not 0.0 < training_noise_alpha <= 0.1:
            raise ValueError(
                # Completes the multi-line `ValueError` arguments or elements.
                "training.temporal_rollout.training_noise_alpha 必须位于 (0, 0.1]"
            )
        if not isinstance(boundary_state_override, bool):
            raise ValueError("training.temporal_rollout.boundary_state_override 必须为布尔值")
        # Checks `"enabled" in raw_config` before entering this conditional branch.
        if "enabled" in raw_config:
            self.logger.warning(
                "training.temporal_rollout.enabled 已废弃，现由 horizon 及扩展功能自动解析，请删除该字段"
            )

        # Stores the `rollout_enabled` for use by the remaining steps in this flow.
        rollout_enabled = int(horizon) > 1 or training_noise or boundary_state_override
        if rollout_enabled and int(self.output_length) != 1:
            raise ValueError("时序自由滚动、训练噪声和边界状态回写要求 task.output_length=1")
        if training_noise and type(self.processor).__name__ not in {
            # Completes the multi-line `collection` arguments or elements.
            "GraphProcessor",
            "PointCloudProcessor",
            "TransolverProcessor",
        }:
            # Raises `NotImplementedError` to report why this path cannot continue.
            raise NotImplementedError(
                "training_noise 首版仅支持 GraphProcessor、PointCloudProcessor 或 TransolverProcessor"
            )
        if boundary_state_override and type(self.processor).__name__ not in {
            # Completes the multi-line `collection` arguments or elements.
            "GraphProcessor",
            "PointCloudProcessor",
            "TransolverProcessor",
        }:
            # Raises `NotImplementedError` to report why this path cannot continue.
            raise NotImplementedError(
                "boundary_state_override 首版仅支持 GraphProcessor、PointCloudProcessor 或 TransolverProcessor"
            )
        if boundary_state_override and not getattr(self.processor, "local_feature_encoding", False):
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(
                "boundary_state_override 需要 local_feature_encoding=cgns_based 或 vtk_based"
            )
        checkpoint_segments = raw_config.get("checkpoint_segments", 0)
        # Validates `isinstance(checkpoint_segments, bool)` and raises `ValueError` when the input cannot be processed safely.
        if isinstance(checkpoint_segments, bool):
            raise ValueError("training.temporal_rollout.checkpoint_segments 必须为非负整数")
        if isinstance(checkpoint_segments, float) and not checkpoint_segments.is_integer():
            raise ValueError("training.temporal_rollout.checkpoint_segments 必须为非负整数")
        # Keeps the operation and its recovery path within the same exception boundary.
        try:
            checkpoint_segments = int(checkpoint_segments)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                # Completes the multi-line `ValueError` arguments or elements.
                "training.temporal_rollout.checkpoint_segments 必须为非负整数"
            ) from exc
        if checkpoint_segments < 0:
            raise ValueError("training.temporal_rollout.checkpoint_segments 必须为非负整数")
        # Validates `checkpoint_segments > horizon` and raises `ValueError` when the input cannot be processed safely.
        if checkpoint_segments > horizon:
            raise ValueError(
                "training.temporal_rollout.checkpoint_segments 不能大于 horizon"
            )
        # Stores the `weighting` for use by the remaining steps in this flow.
        weighting = str(raw_config.get("horizon_weighting", "uniform")).lower()
        if weighting != "uniform":
            raise ValueError("时序多步自由滚动首版仅支持 horizon_weighting=uniform")
        gamma = raw_config.get("horizon_weight_gamma", 1.0)
        # Keeps the operation and its recovery path within the same exception boundary.
        try:
            gamma = float(gamma)
        except (TypeError, ValueError) as exc:
            raise ValueError("training.temporal_rollout.horizon_weight_gamma 必须为数值") from exc
        # Returns `` to the caller for the next stage.
        return {
            "enabled": bool(rollout_enabled),
            "horizon": int(horizon),
            "checkpoint_segments": checkpoint_segments,
            # Completes the multi-line `collection` arguments or elements.
            "horizon_weighting": weighting,
            "horizon_weight_gamma": gamma,
            "training_noise": training_noise,
            "training_noise_alpha": training_noise_alpha,
            # Completes the multi-line `collection` arguments or elements.
            "boundary_state_override": boundary_state_override,
            "feedback_mode": "free_rollout" if int(horizon) > 1 else "single_step",
            "loss_scope": (
                "autoregressive_free_rollout_horizon"
                # Completes the multi-line `int` arguments or elements.
                if int(horizon) > 1
                else "single_step"
            ),
        }

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _move_to_device(value, device):
        """递归迁移张量或含张量的输入结构，支持 Scalar 滚动视图。"""
        if isinstance(value, torch.Tensor):
            return value.to(device)
        # Returns early when `isinstance(value, dict)` makes the remaining work inapplicable.
        if isinstance(value, dict):
            return {key: Trainer._move_to_device(item, device) for key, item in value.items()}
        if isinstance(value, list):
            return [Trainer._move_to_device(item, device) for item in value]
        # Returns early when `isinstance(value, tuple)` makes the remaining work inapplicable.
        if isinstance(value, tuple):
            return tuple(Trainer._move_to_device(item, device) for item in value)
        to_method = getattr(value, "to", None)
        return to_method(device) if callable(to_method) else value

    # 执行 H 步自由滚动，并沿时间维使用现有 processor 损失。
    def _compute_rollout_batch_loss(self, inputs, targets, *, training: bool):
        """执行 H 步自由滚动，并沿时间维使用现有 processor 损失。"""
        rollout_fn = getattr(self.processor, "predict_autoregressive_rollout", None)
        if not callable(rollout_fn):
            raise NotImplementedError("当前 processor 不支持 autoregressive 自由滚动训练")
        # Stores the `rollout_inputs` for use by the remaining steps in this flow.
        rollout_inputs = inputs
        if training and self.temporal_rollout.get("training_noise", False):
            perturb_fn = getattr(self.processor, "perturb_autoregressive_history", None)
            if not callable(perturb_fn):
                # Raises `NotImplementedError` to report why this path cannot continue.
                raise NotImplementedError("当前 processor 不支持 autoregressive 训练噪声")
            noise_generator = torch.Generator()
            noise_generator.manual_seed(self.seed + 104729 * (int(self.global_step) + 1))
            perturb_kwargs = {
                # Completes the multi-line `collection` arguments or elements.
                "generator": noise_generator,
                "alpha": self.temporal_rollout["training_noise_alpha"],
            }
            if self.graph_use_partition:
                # Derives the `perturb_kwargs['global_node_noise_seed']` from the current values for downstream processing.
                perturb_kwargs["global_node_noise_seed"] = (
                    self.seed + 104729 * (int(self.global_step) + 1)
                )
            rollout_inputs = perturb_fn(inputs, **perturb_kwargs)
        # Stores the `outputs` for use by the remaining steps in this flow.
        outputs = rollout_fn(
            model=self.model,
            inputs=rollout_inputs,
            rollout_steps=self.temporal_rollout["horizon"],
        # Completes the multi-line `rollout_fn` arguments or elements.
        )
        aligned_targets = targets
        if isinstance(outputs, torch.Tensor) and isinstance(targets, torch.Tensor):
            if outputs.ndim == targets.ndim + 1:
                # Stores the `aligned_targets` for use by the remaining steps in this flow.
                aligned_targets = targets.unsqueeze(1)
            if outputs.shape != aligned_targets.shape:
                raise ValueError(
                    "时序自由滚动预测与目标形状不一致："
                    # Completes the multi-line `tuple` arguments or elements.
                    f"pred={tuple(outputs.shape)}, target={tuple(aligned_targets.shape)}"
                )
        if self.processor is not None:
            compute_loss = (
                # Continues the surrounding statement or collection definition.
                self.processor.compute_train_loss
                if training else self.processor.compute_eval_loss
            )
            return compute_loss(
                # Completes the multi-line `compute_loss` arguments or elements.
                criterion=self.criterion,
                outputs=outputs,
                targets=aligned_targets,
                inputs=rollout_inputs,
            # Completes the multi-line `compute_loss` arguments or elements.
            )
        return self.criterion(outputs, aligned_targets)

    @staticmethod
    def _effective_loss_element_count(inputs, targets) -> int:
        """返回参与当前损失聚合的监督元素数量。"""
        # Stores the `owner_mask` for use by the remaining steps in this flow.
        owner_mask = getattr(inputs, "owner_mask", None)
        override_mask = getattr(inputs, "boundary_state_override_mask", None)
        if owner_mask is None and override_mask is None:
            return int(targets.numel())
        # Checks `targets.ndim == 3` before entering this conditional branch.
        if targets.ndim == 3:
            multiplier = int(targets.shape[1])
        elif targets.ndim == 2:
            multiplier = 1
        # Continues the surrounding statement or collection definition.
        else:
            raise ValueError("损失掩码仅支持二维或三维目标张量")
        if owner_mask is not None:
            if not isinstance(owner_mask, torch.Tensor) or owner_mask.ndim != 1:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError("owner_mask 必须为 [N]")
            owner_node_count = int(owner_mask.numel())
            if targets.shape[0] == owner_node_count:
                node_dim = 0
            # Checks `targets.ndim == 3 and targets.shape[1] == owner_node_count` before entering this conditional branch.
            elif targets.ndim == 3 and targets.shape[1] == owner_node_count:
                node_dim = 1
            else:
                raise ValueError("owner_mask 与目标的节点维度不一致")
            # Stores the `element_per_node` for use by the remaining steps in this flow.
            element_per_node = math.prod(
                int(size) for axis, size in enumerate(targets.shape) if axis != node_dim
            )
            owner_count = int(owner_mask.sum().item()) * element_per_node
            # Checks `override_mask is None` before entering this conditional branch.
            if override_mask is None:
                if owner_count <= 0:
                    raise ValueError("owner_mask 排除了全部监督目标")
                return owner_count
        # Validates `` and raises `ValueError` when the input cannot be processed safely.
        if not isinstance(override_mask, torch.Tensor) or override_mask.ndim != 2:
            raise ValueError("boundary_state_override_mask 必须为 [N, C]")
        valid_mask = ~override_mask
        if owner_mask is not None:
            # Derives the valid-item mask from the current values for downstream processing.
            valid_mask = valid_mask & owner_mask.to(device=valid_mask.device, dtype=torch.bool).unsqueeze(-1)
        valid_count = int(valid_mask.sum().item()) * multiplier
        if valid_count <= 0:
            raise ValueError("boundary_state_override 排除了全部监督目标，无法聚合损失")
        # Returns `valid_count` to the caller for the next stage.
        return valid_count

    @staticmethod
    def _find_full_sample_dataset(dataset: Dataset) -> Optional[Dataset]:
        """沿 dataset 包装链定位可读取完整时序样本的基础数据集。"""
        visited = set()
        # Stores the `current` for use by the remaining steps in this flow.
        current = dataset
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if hasattr(current, "get_processed_sample"):
                # Returns `current` to the caller for the next stage.
                return current
            current = getattr(current, "base_dataset", None) or getattr(current, "dataset", None)
        return None

    def _prepare_boundary_state_override(self, train_dataset: Dataset) -> None:
        """只从训练集拟合静态边界状态规则，并持久化到处理器状态。"""
        # Returns early when `not self.temporal_rollout.get("boundary_state_override", False)` makes the remaining work inapplicable.
        if not self.temporal_rollout.get("boundary_state_override", False):
            return
        if self.processor is None:
            raise ValueError("boundary_state_override 需要 FieldProcessor")
        # Stores the `fit_schema` for use by the remaining steps in this flow.
        fit_schema = getattr(self.processor, "fit_boundary_state_override_schema", None)
        if not callable(fit_schema):
            raise NotImplementedError("当前 processor 不支持 boundary_state_override")
        full_dataset = self._find_full_sample_dataset(train_dataset)
        # Validates `full_dataset is None` and raises `ValueError` when the input cannot be processed safely.
        if full_dataset is None:
            raise ValueError("boundary_state_override 无法定位训练集的完整时序样本")
        samples = [
            full_dataset.get_processed_sample(index)
            # Completes the multi-line `range` arguments or elements.
            for index in range(len(full_dataset))
        ]
        schema = fit_schema(samples)
        os.makedirs(self.processed_data_path, exist_ok=True)
        # Stores the `processor_state_path` for use by the remaining steps in this flow.
        processor_state_path = os.path.join(self.processed_data_path, "processor_state.pt")
        export_state = getattr(self.processor, "export_state", None)
        if not callable(export_state):
            raise NotImplementedError("当前 processor 无法持久化 boundary_state_override 规则")
        # Calls `atomic_torch_save` to perform the operation required at this stage.
        atomic_torch_save(export_state(), processor_state_path)
        group_count = len(schema.get("groups", {}))
        self.logger.info(
            "已从训练集拟合静态边界状态回写规则: groups=%s output_dim=%s",
            # Completes the multi-line `self.logger.info` arguments or elements.
            group_count,
            schema.get("output_dim"),
        )

    def _compute_train_batch_loss(self, inputs, targets):
        """计算单个训练批次的损失。"""
        # Returns early when `self.temporal_rollout["enabled"]` makes the remaining work inapplicable.
        if self.temporal_rollout["enabled"]:
            return self._compute_rollout_batch_loss(inputs, targets, training=True)
        outputs = self.model(inputs)
        if self.processor is not None:
            # Stores the `restore_output` for use by the remaining steps in this flow.
            restore_output = getattr(self.processor, "restore_autoregressive_model_output", None)
            if callable(restore_output):
                outputs = restore_output(inputs, outputs)
            return self.processor.compute_train_loss(
                # Completes the multi-line `self.processor.compute_train_loss` arguments or elements.
                criterion=self.criterion,
                outputs=outputs,
                targets=targets,
                inputs=inputs,
            # Completes the multi-line `self.processor.compute_train_loss` arguments or elements.
            )
        return self.criterion(outputs, targets)

    def _compute_eval_batch_loss(self, inputs, targets):
        """计算单个验证批次的损失。"""
        if self.temporal_rollout["enabled"]:
            # Returns `` to the caller for the next stage.
            return self._compute_rollout_batch_loss(inputs, targets, training=False)
        outputs = self.model(inputs)
        if self.processor is not None:
            restore_output = getattr(self.processor, "restore_autoregressive_model_output", None)
            # Checks `callable(restore_output)` before entering this conditional branch.
            if callable(restore_output):
                outputs = restore_output(inputs, outputs)
            return self.processor.compute_eval_loss(
                criterion=self.criterion,
                # Completes the multi-line `self.processor.compute_eval_loss` arguments or elements.
                outputs=outputs,
                targets=targets,
                inputs=inputs,
            )
        # Returns `self.criterion(outputs, targets)` to the caller for the next stage.
        return self.criterion(outputs, targets)

    def _build_train_loader(self, train_dataset: Dataset, data_epoch: int) -> DataLoader:
        """为指定数据轮次构建可重建顺序的训练加载器。"""
        epoch_generator = torch.Generator()
        epoch_generator.manual_seed(self.seed + int(data_epoch))
        # Returns `` to the caller for the next stage.
        return DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            # Completes the multi-line `DataLoader` arguments or elements.
            num_workers=self.train_num_workers,
            generator=epoch_generator,
        )

    @staticmethod
    # 基于稳定元数据构建轻量数据集指纹，不读取完整样本内容。
    def _build_dataset_fingerprint(dataset: Dataset) -> str:
        """基于稳定元数据构建轻量数据集指纹，不读取完整样本内容。"""
        digest = hashlib.sha256()
        visited = set()

        def update(value) -> None:
            # Adds the current value to `digest` for later aggregation.
            digest.update(str(value).encode("utf-8", errors="replace"))
            digest.update(b"\0")

        def update_sequence(name: str, values) -> None:
            try:
                # Stores the `value_count` for use by the remaining steps in this flow.
                value_count = len(values)
            except (TypeError, ValueError):
                return

            update(name)
            # Calls `update` to perform the operation required at this stage.
            update(value_count)
            if value_count <= 0:
                return

            sample_count = min(value_count, 64)
            # Checks `sample_count == 1` before entering this conditional branch.
            if sample_count == 1:
                positions = (0,)
            else:
                positions = tuple(
                    # Completes the multi-line `int` arguments or elements.
                    int(position * (value_count - 1) / (sample_count - 1))
                    for position in range(sample_count)
                )
            for position in positions:
                # Keeps the operation and its recovery path within the same exception boundary.
                try:
                    update(position)
                    update(values[position])
                except (IndexError, KeyError, TypeError):
                    # Calls `update` to perform the operation required at this stage.
                    update("unavailable")

        def update_tensor(name: str, tensor: torch.Tensor) -> None:
            update(name)
            update(tuple(tensor.shape))
            # Calls `update` to perform the operation required at this stage.
            update(tensor.dtype)
            try:
                flat_tensor = tensor.detach().reshape(-1)
                value_count = int(flat_tensor.numel())
                # Calls `update` to perform the operation required at this stage.
                update(value_count)
                if value_count <= 0:
                    return
                for position in sorted({0, value_count // 2, value_count - 1}):
                    # Calls `update` to perform the operation required at this stage.
                    update(position)
                    update(flat_tensor[position].item())
            except (RuntimeError, TypeError, ValueError):
                update("value_unavailable")

        # Defines `update_dataset` for update dataset logic.
        def update_dataset(current_dataset, depth: int = 0) -> None:
            if current_dataset is None or depth > 2:
                return
            identity = id(current_dataset)
            # Checks `identity in visited` before entering this conditional branch.
            if identity in visited:
                update("recursive_dataset")
                return
            visited.add(identity)

            # Calls `update` to perform the operation required at this stage.
            update(f"{type(current_dataset).__module__}.{type(current_dataset).__qualname__}")
            try:
                update(len(current_dataset))
            except (TypeError, ValueError):
                # Calls `update` to perform the operation required at this stage.
                update("length_unavailable")

            for attribute in (
                "sample_view",
                "data_kind",
                # Completes the multi-line `collection` arguments or elements.
                "model_type",
                "mode",
                "temporal_strategy",
            ):
                # Calls `update` to perform the operation required at this stage.
                update(attribute)
                update(getattr(current_dataset, attribute, ""))

            for attribute in (
                "sample_ids",
                # Completes the multi-line `collection` arguments or elements.
                "file_paths",
                "window_indexes",
                "partition_indexes",
                "indices",
                # Completes the multi-line `collection` arguments or elements.
                "sample_entries",
            ):
                values = getattr(current_dataset, attribute, None)
                if values is not None:
                    # Calls `update_sequence` to perform the operation required at this stage.
                    update_sequence(attribute, values)

            tensors = getattr(current_dataset, "tensors", None)
            if isinstance(tensors, (list, tuple)):
                for index, tensor in enumerate(tensors):
                    # Checks `isinstance(tensor, torch.Tensor)` before entering this conditional branch.
                    if isinstance(tensor, torch.Tensor):
                        update_tensor(f"tensor_{index}", tensor)
                    else:
                        update(f"tensor_{index}")
                        # Calls `update` to perform the operation required at this stage.
                        update(type(tensor))

            update_dataset(getattr(current_dataset, "base_dataset", None), depth + 1)
            update_dataset(getattr(current_dataset, "dataset", None), depth + 1)

        update_dataset(dataset)
        # Returns `digest.hexdigest()` to the caller for the next stage.
        return digest.hexdigest()

    def _build_train_manifest(self, train_dataset: Dataset) -> Dict:
        """记录中途续训所需的训练顺序前提。"""
        return {
            "fingerprint": self._build_dataset_fingerprint(train_dataset),
            # Completes the multi-line `collection` arguments or elements.
            "seed": self.seed,
            "batch_size": self.batch_size,
            "train_batch_size": self.train_batch_size,
            "graph_use_partition": self.graph_use_partition,
            # Completes the multi-line `collection` arguments or elements.
            "graph_partition_count": self.graph_partition_count,
            "num_workers": self.train_num_workers,
        }

    def _resolve_train_manifest(self, train_dataset: Dataset) -> List[str]:
        """校验训练顺序前提，返回与 checkpoint 不一致的字段。"""
        # Stores the `expected` for use by the remaining steps in this flow.
        expected = self._build_train_manifest(train_dataset)
        loaded = self.train_manifest
        if loaded is None:
            self.train_manifest = expected
            # Returns `["train_manifest"] if self._loaded_step_checkpoint else []` to the caller for the next stage.
            return ["train_manifest"] if self._loaded_step_checkpoint else []

        if not isinstance(loaded, dict):
            self.train_manifest = expected
            return ["train_manifest"]

        # Initializes the `fields` as a collection for values from this processing stage.
        fields = (
            "fingerprint",
            "seed",
            "batch_size",
            # Completes the multi-line `collection` arguments or elements.
            "train_batch_size",
            "graph_use_partition",
            "graph_partition_count",
            "num_workers",
        # Completes the multi-line `collection` arguments or elements.
        )
        mismatches = [
            field
            for field in fields
            # Completes the multi-line `loaded.get` arguments or elements.
            if loaded.get(field) != expected.get(field)
        ]
        self.train_manifest = expected
        return mismatches

    # 在中途续训前提变化时放弃跳过 batch，改从当前数据轮次开头恢复。
    def _mark_resume_inexact(self, reasons: Sequence[str]) -> None:
        """在中途续训前提变化时放弃跳过 batch，改从当前数据轮次开头恢复。"""
        self.resume_exact = False
        reason_text = ", ".join(str(reason) for reason in reasons)
        if self.batch_in_epoch > 0:
            # Records the current execution state to make progress and failures traceable.
            self.logger.warning(
                "续训前提不一致（%s），将从数据轮次 %s 开头恢复",
                reason_text,
                self.data_epoch + 1,
            # Completes the multi-line `self.logger.warning` arguments or elements.
            )
            self.batch_in_epoch = 0
        else:
            self.logger.warning(
                # Completes the multi-line `self.logger.warning` arguments or elements.
                "续训前提不一致（%s），当前位于数据轮次边界，标记为非精确恢复",
                reason_text,
            )
        self.start_epoch = self.data_epoch

    # 构建一次性固定验证清单，供每个验证周期复用。
    def _build_validation_manifest(self, valid_dataset: Dataset) -> Dict:
        """构建一次性固定验证清单，供每个验证周期复用。"""
        dataset_size = len(valid_dataset)
        if dataset_size <= 0:
            raise ValueError("验证集为空，无法执行验证")

        # Returns early when `self.validation_max_batches is None` makes the remaining work inapplicable.
        if self.validation_max_batches is None:
            return {
                "scope": "full",
                "max_batches": None,
                # Completes the multi-line `collection` arguments or elements.
                "valid_batch_size": self.valid_batch_size,
                "seed": self.seed,
                "indices": [],
                "fingerprint": self._build_validation_fingerprint(valid_dataset),
            # Completes the multi-line `collection` arguments or elements.
            }

        index_generator = torch.Generator()
        index_generator.manual_seed(self.seed)
        indices = torch.randperm(dataset_size, generator=index_generator).tolist()
        # Stores the `max_items` for use by the remaining steps in this flow.
        max_items = min(dataset_size, self.validation_max_batches * self.valid_batch_size)
        indices = indices[:max_items]
        if not indices:
            raise ValueError("validation_max_batches 产生了空的快速验证子集")
        # Returns `` to the caller for the next stage.
        return {
            "scope": "fixed_fast_subset",
            "max_batches": self.validation_max_batches,
            "valid_batch_size": self.valid_batch_size,
            # Completes the multi-line `collection` arguments or elements.
            "seed": self.seed,
            "indices": indices,
            "fingerprint": self._build_validation_fingerprint(valid_dataset),
        }

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _build_validation_fingerprint(valid_dataset: Dataset) -> str:
        """记录轻量数据集指纹，防止续训时静默替换快速验证范围。"""
        return Trainer._build_dataset_fingerprint(valid_dataset)

    def _resolve_validation_manifest(self, valid_dataset: Dataset) -> Dict:
        """恢复或新建验证清单，并校验续训范围保持不变。"""
        # Stores the `expected` for use by the remaining steps in this flow.
        expected = self._build_validation_manifest(valid_dataset)
        if self.validation_manifest is None:
            self.validation_manifest = expected
            return self.validation_manifest

        # Stores the `loaded` for use by the remaining steps in this flow.
        loaded = self.validation_manifest
        if loaded.get("fingerprint") != expected["fingerprint"]:
            raise ValueError("续训验证集与 checkpoint 中的验证清单不一致，无法继续训练")
        if loaded.get("scope") != expected["scope"]:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("续训 validation_max_batches 与 checkpoint 中的验证清单不一致")
        if loaded.get("max_batches") != expected["max_batches"]:
            raise ValueError("续训 validation_max_batches 与 checkpoint 中的验证清单不一致")
        loaded_valid_batch_size = loaded.get("valid_batch_size")
        # Validates `` and raises `ValueError` when the input cannot be processed safely.
        if loaded_valid_batch_size is None and loaded.get("scope") == "fixed_fast_subset":
            raise ValueError(
                "checkpoint 中的快速验证清单缺少 valid_batch_size，无法安全恢复验证"
            )
        # Keeps the operation and its recovery path within the same exception boundary.
        try:
            loaded_batch_size_value = (
                int(loaded_valid_batch_size)
                if loaded_valid_batch_size is not None
                # Continues the surrounding statement or collection definition.
                else None
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoint 中的 valid_batch_size 无效，无法安全恢复验证") from exc
        # Validates `` and raises `ValueError` when the input cannot be processed safely.
        if (
            loaded_batch_size_value is not None
            and loaded_batch_size_value != expected["valid_batch_size"]
        ):
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("续训验证 batch_size 与 checkpoint 中的验证清单不一致")
        if loaded.get("scope") == "fixed_fast_subset":
            indices = [int(index) for index in loaded.get("indices", [])]
            if not indices or min(indices) < 0 or max(indices) >= len(valid_dataset):
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError("checkpoint 中的快速验证清单无效")
            self.validation_manifest = {
                **loaded,
                "indices": indices,
                # Completes the multi-line `int` arguments or elements.
                "seed": int(loaded.get("seed", self.seed)),
            }
        else:
            self.validation_manifest = expected
        # Returns `self.validation_manifest` to the caller for the next stage.
        return self.validation_manifest

    def _build_valid_loader(self, valid_dataset: Dataset) -> DataLoader:
        """按验证清单构建验证加载器，快速验证不在每轮重新随机抽样。"""
        manifest = self._resolve_validation_manifest(valid_dataset)
        dataset_view: Dataset = valid_dataset
        # Checks `manifest["scope"] == "fixed_fast_subset"` before entering this conditional branch.
        if manifest["scope"] == "fixed_fast_subset":
            dataset_view = _IndexedDataset(valid_dataset, manifest["indices"])
        return DataLoader(
            dataset_view,
            # Completes the multi-line `DataLoader` arguments or elements.
            batch_size=self.valid_batch_size,
            shuffle=False,
            num_workers=self.valid_num_workers,
        )

    # 在得到训练加载器长度后解析统一 step 调度参数。
    def _resolve_schedule(self, train_dataset: Dataset) -> None:
        """在得到训练加载器长度后解析统一 step 调度参数。"""
        self.steps_per_epoch = len(
            DataLoader(
                train_dataset,
                # Completes the multi-line `DataLoader` arguments or elements.
                batch_size=self.train_batch_size,
                shuffle=False,
                num_workers=self.train_num_workers,
            )
        # Completes the multi-line `len` arguments or elements.
        )
        if self.steps_per_epoch <= 0:
            raise ValueError("训练集为空，无法执行训练")

        saved_steps_per_epoch = self._loaded_steps_per_epoch or self.steps_per_epoch
        # Checks `self.batch_in_epoch >= saved_steps_per_epoch` before entering this conditional branch.
        if self.batch_in_epoch >= saved_steps_per_epoch:
            if self.batch_in_epoch > saved_steps_per_epoch:
                self.resume_exact = False
                self.logger.warning(
                    # Completes the multi-line `self.logger.warning` arguments or elements.
                    "checkpoint 的 batch_in_epoch=%s 超出保存时数据轮次批次数=%s，将从下一数据轮次恢复",
                    self.batch_in_epoch,
                    saved_steps_per_epoch,
                )
            # Stores the `data_epoch` on the instance so other methods can reuse it.
            self.data_epoch += 1
            self.batch_in_epoch = 0
            self.start_epoch = self.data_epoch

        resume_mismatches = self._resolve_train_manifest(train_dataset)
        # Checks `` before entering this conditional branch.
        if self._loaded_steps_per_epoch is not None and self._loaded_steps_per_epoch != self.steps_per_epoch:
            resume_mismatches.append("steps_per_epoch")

        if resume_mismatches:
            self.resume_exact = False
            # Calls `self._mark_resume_inexact` to perform the operation required at this stage.
            self._mark_resume_inexact(resume_mismatches)
        self._loaded_steps_per_epoch = self.steps_per_epoch

        self.resolved_max_steps = (
            self.max_steps
            # Continues the surrounding statement or collection definition.
            if self.max_steps is not None
            else int(self.epochs) * self.steps_per_epoch
        )
        self.resolved_validate_every = (
            # Continues the surrounding statement or collection definition.
            self.validate_every
            if self.validate_every is not None
            else self.steps_per_epoch
        )

        # Checks `self._resume_from_legacy_checkpoint` before entering this conditional branch.
        if self._resume_from_legacy_checkpoint:
            self.global_step = self.start_epoch * self.steps_per_epoch
            self.data_epoch = self.start_epoch
            self.batch_in_epoch = 0
            # Stores the `last_validation_step` on the instance so other methods can reuse it.
            self.last_validation_step = self.global_step
            self.validation_checks = len(self.history.get("val_loss", []))
            self._resume_from_legacy_checkpoint = False

        if self.global_step > self.resolved_max_steps:
            # Records the current execution state to make progress and failures traceable.
            self.logger.warning(
                "checkpoint 的 global_step=%s 超过当前 max_steps=%s，训练不会继续执行",
                self.global_step,
                self.resolved_max_steps,
            # Completes the multi-line `self.logger.warning` arguments or elements.
            )

        self.logger.info(
            "训练调度已解析: max_steps=%s validate_every=%s steps_per_epoch=%s 预计数据轮次=%s",
            self.resolved_max_steps,
            # Completes the multi-line `self.logger.info` arguments or elements.
            self.resolved_validate_every,
            self.steps_per_epoch,
            math.ceil(self.resolved_max_steps / self.steps_per_epoch),
        )

    # 执行一次完整或固定快速验证；收到终止信号时返回空值。
    def _run_validation(
        self,
        valid_loader: DataLoader,
        terminate_training_checker: Optional[Callable[[], bool]],
    # Continues the surrounding statement or collection definition.
    ) -> Optional[float]:
        """执行一次完整或固定快速验证；收到终止信号时返回空值。"""
        self.model.eval()
        val_loss_sum = 0.0
        val_weight_sum = 0

        # Scopes resource use so the context manager can release it on exit.
        with torch.no_grad():
            for inputs, targets in valid_loader:
                if self._should_terminate_training(terminate_training_checker):
                    self.training_terminated = True
                    # Records the current execution state to make progress and failures traceable.
                    self.logger.warning("验证过程中检测到终止信号，停止训练")
                    return None

                inputs = self._move_to_device(inputs, self.device)
                targets = self._move_to_device(targets, self.device)
                # Stores the `loss` for use by the remaining steps in this flow.
                loss = self._compute_eval_batch_loss(inputs, targets)
                loss_value = float(loss.item())
                if not math.isfinite(loss_value):
                    raise ValueError("验证损失不是有限数值，已停止训练且未更新历史或检查点")
                # Stores the `batch_weight` for use by the remaining steps in this flow.
                batch_weight = self._effective_loss_element_count(inputs, targets)
                val_loss_sum += loss_value * batch_weight
                val_weight_sum += batch_weight

        if val_weight_sum == 0:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("验证集目标元素数量为零，无法计算平均损失")
        val_loss = val_loss_sum / val_weight_sum
        if not math.isfinite(val_loss):
            raise ValueError("验证平均损失不是有限数值，已停止训练且未更新历史或检查点")
        # Returns `val_loss` to the caller for the next stage.
        return val_loss

    def _run_validation_cycle(
        self,
        valid_loader: DataLoader,
        # Completes the multi-line `collection` arguments or elements.
        progress_callback: Optional[Callable[[Dict], None]],
        terminate_training_checker: Optional[Callable[[], bool]],
        interval_train_loss_sum: float,
        interval_train_weight_sum: int,
    # Continues the surrounding statement or collection definition.
    ) -> bool:
        """完成验证、checkpoint、回调和早停判断，返回是否应停止训练。"""
        if interval_train_weight_sum <= 0:
            raise ValueError("训练区间目标元素数量为零，无法计算平均损失")

        train_loss = interval_train_loss_sum / interval_train_weight_sum
        # Validates `not math.isfinite(train_loss)` and raises `ValueError` when the input cannot be processed safely.
        if not math.isfinite(train_loss):
            raise ValueError("训练平均损失不是有限数值，已停止训练且未更新历史或检查点")

        val_loss = self._run_validation(valid_loader, terminate_training_checker)
        if val_loss is None:
            # Returns `True` to the caller for the next stage.
            return True

        if not math.isfinite(val_loss):
            raise ValueError("验证损失不是有限数值，已停止训练且未更新历史或检查点")
        interval_step_count = self.global_step - self.last_validation_step
        # Adds the current value to `self.history['train_loss']` for later aggregation.
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.train_units.append(self.global_step)
        self.val_units.append(self.global_step)
        # Adds the current value to `self.history['train_units']` for later aggregation.
        self.history["train_units"].append(self.global_step)
        self.history["val_units"].append(self.global_step)
        self.validation_checks += 1
        self.last_validation_step = self.global_step

        # Records the current execution state to make progress and failures traceable.
        self.logger.info(
            "验证周期完成: 验证周期=%s/%s global_step=%s data_epoch=%s train_loss=%.6f val_loss=%.6f",
            self.validation_checks,
            self._estimated_validation_checks(),
            # Completes the multi-line `self.logger.info` arguments or elements.
            self.global_step,
            self.data_epoch + 1,
            train_loss,
            val_loss,
        # Completes the multi-line `self.logger.info` arguments or elements.
        )
        if self.lr_scheduler is not None:
            old_lr = [group["lr"] for group in self.optimizer.param_groups]
            self.lr_scheduler.step(val_loss)
            new_lr = [group["lr"] for group in self.optimizer.param_groups]
            self.logger.info("学习率调度: step=%s lr=%s", self.global_step, new_lr)
            if old_lr != new_lr:
                self.logger.info("验证平台期降低学习率: %s -> %s", old_lr, new_lr)
        self._plot_loss_curve(self.history, self.validation_checks)

        if progress_callback is not None:
            self._record_progress(
                # Completes the multi-line `self._record_progress` arguments or elements.
                epoch=self.validation_checks,
                total_step=interval_step_count,
                train_loss=train_loss,
                val_loss=val_loss,
            # Completes the multi-line `self._record_progress` arguments or elements.
            )

        improved = val_loss < self.best_score
        if improved:
            self.best_score = val_loss
            # Stores the `early_stopping_counter` on the instance so other methods can reuse it.
            self.early_stopping_counter = 0
            self.save_checkpoint(
                path=self.checkpoint_path,
                epoch=self.data_epoch,
                # Completes the multi-line `self.save_checkpoint` arguments or elements.
                val_loss=val_loss,
            )
        elif self.early_stopping:
            self.early_stopping_counter += 1

        # Calls `self.save_checkpoint` to perform the operation required at this stage.
        self.save_checkpoint(
            path=self.resume_path,
            epoch=self.data_epoch,
            val_loss=val_loss,
        # Completes the multi-line `self.save_checkpoint` arguments or elements.
        )
        self._emit_progress(progress_callback=progress_callback)

        if self.early_stopping and self.early_stopping_counter >= self.patience:
            self.logger.info(
                # Completes the multi-line `self.logger.info` arguments or elements.
                "触发早停: 验证周期=%s patience=%s",
                self.validation_checks,
                self.patience,
            )
            # Returns `True` to the caller for the next stage.
            return True
        return False

    def _estimated_validation_checks(self) -> int:
        """返回未早停时的计划验证周期总数。"""
        if not self.resolved_max_steps or not self.resolved_validate_every:
            # Returns `0` to the caller for the next stage.
            return 0
        return math.ceil(self.resolved_max_steps / self.resolved_validate_every)

    def _run_partition_logical_batch(self, logical_batch) -> Tuple[float, int]:
        """对 B 个完整图的全部 owner + halo 子图累积梯度，再执行一次更新。"""
        partitions = [item for sample in logical_batch for item in sample]
        # Validates `not partitions` and raises `ValueError` when the input cannot be processed safely.
        if not partitions:
            raise ValueError("图分块逻辑 batch 不能为空")
        prepared = [
            (self._move_to_device(inputs, self.device), self._move_to_device(targets, self.device))
            # Completes the multi-line `collection` arguments or elements.
            for inputs, targets in partitions
        ]
        weights = [self._effective_loss_element_count(inputs, targets) for inputs, targets in prepared]
        total_weight = sum(weights)
        # Validates `total_weight <= 0` and raises `ValueError` when the input cannot be processed safely.
        if total_weight <= 0:
            raise ValueError("图分块逻辑 batch 没有有效监督元素")

        self.optimizer.zero_grad()
        loss_sum = 0.0
        # Visits each item in `zip(prepared, weights)` and applies the current processing step.
        for (inputs, targets), weight in zip(prepared, weights):
            loss = self._compute_train_batch_loss(inputs, targets)
            loss_value = float(loss.item())
            if not math.isfinite(loss_value):
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError("训练损失不是有限数值，已停止训练且未更新参数或检查点")
            (loss * (float(weight) / float(total_weight))).backward()
            loss_sum += loss_value * weight
        self.optimizer.step()
        # Returns `loss_sum / total_weight, total_weight` to the caller for the next stage.
        return loss_sum / total_weight, total_weight

    def train(
        self,
        train_dataset: Dataset,
        # Continues the surrounding statement or collection definition.
        valid_dataset: Dataset,
        progress_callback: Optional[Callable[[Dict], None]] = None,
        terminate_training_checker: Optional[Callable[[], bool]] = None,
    ) -> Dict:
        # Stores the `training_terminated` on the instance so other methods can reuse it.
        self.training_terminated = False
        if self.graph_use_partition:
            self.logger.info(
                "批大小语义: batch_size(full_graph_equivalent)=%s effective_train_batch_size(subgraphs)=%s valid_batch_size(full_graph)=%s",
                # Completes the multi-line `self.logger.info` arguments or elements.
                self.full_graph_equivalent_batch_size,
                self.train_batch_size,
                self.valid_batch_size,
            )
        # Continues the surrounding statement or collection definition.
        else:
            self.logger.info(
                "批大小语义: train_batch_size(full_graph)=%s valid_batch_size(full_graph)=%s",
                self.train_batch_size,
                # Completes the multi-line `self.logger.info` arguments or elements.
                self.valid_batch_size,
            )

        history = self.history
        if self.progress_start_timestamp <= 0:
            # Stores the `progress_start_timestamp` on the instance so other methods can reuse it.
            self.progress_start_timestamp = int(time.time_ns() // 1_000_000)

        self._prepare_boundary_state_override(train_dataset)
        self._resolve_schedule(train_dataset)
        valid_loader = self._build_valid_loader(valid_dataset)

        # Checks `self.global_step >= self.resolved_max_steps` before entering this conditional branch.
        if self.global_step >= self.resolved_max_steps:
            self.logger.info(
                "续训检查点已达到目标步数: global_step=%s max_steps=%s",
                self.global_step,
                # Completes the multi-line `self.logger.info` arguments or elements.
                self.resolved_max_steps,
            )
            return history

        if self._should_terminate_training(terminate_training_checker):
            # Stores the `training_terminated` on the instance so other methods can reuse it.
            self.training_terminated = True
            self.logger.warning("检测到提前终止训练信号，跳过训练并返回当前历史")
            return history

        interval_train_loss_sum = 0.0
        # Stores the `interval_train_weight_sum` for use by the remaining steps in this flow.
        interval_train_weight_sum = 0

        while self.global_step < self.resolved_max_steps:
            if self._should_terminate_training(terminate_training_checker):
                self.training_terminated = True
                # Records the current execution state to make progress and failures traceable.
                self.logger.warning("在数据轮次 %s 开始前检测到终止信号，停止训练", self.data_epoch + 1)
                return history

            self.model.train()
            train_loader = self._build_train_loader(train_dataset, self.data_epoch)
            # Stores the `resume_batch_in_epoch` for use by the remaining steps in this flow.
            resume_batch_in_epoch = self.batch_in_epoch

            for batch_index, batch in enumerate(train_loader):
                if batch_index < self.batch_in_epoch:
                    continue
                # Checks `self._should_terminate_training(terminate_training_checker)` before entering this conditional branch.
                if self._should_terminate_training(terminate_training_checker):
                    self.training_terminated = True
                    self.logger.warning(
                        "在数据轮次 %s 的训练批次开始前检测到终止信号，停止训练",
                        # Completes the multi-line `self.logger.warning` arguments or elements.
                        self.data_epoch + 1,
                    )
                    return history

                if self.graph_use_partition:
                    # Stores the `(loss_value, batch_weight)` for use by the remaining steps in this flow.
                    loss_value, batch_weight = self._run_partition_logical_batch(batch)
                else:
                    inputs, targets = batch
                    inputs = self._move_to_device(inputs, self.device)
                    # Stores the target tensors for use by the remaining steps in this flow.
                    targets = self._move_to_device(targets, self.device)
                    self.optimizer.zero_grad()
                    loss = self._compute_train_batch_loss(inputs, targets)
                    loss_value = float(loss.item())
                    # Validates `not math.isfinite(loss_value)` and raises `ValueError` when the input cannot be processed safely.
                    if not math.isfinite(loss_value):
                        raise ValueError("训练损失不是有限数值，已停止训练且未更新历史或检查点")
                    loss.backward()
                    self.optimizer.step()
                    # Stores the `batch_weight` for use by the remaining steps in this flow.
                    batch_weight = self._effective_loss_element_count(inputs, targets)
                interval_train_loss_sum += loss_value * batch_weight
                interval_train_weight_sum += batch_weight
                self.global_step += 1
                # Stores the `batch_in_epoch` on the instance so other methods can reuse it.
                self.batch_in_epoch = batch_index + 1

                self.logger.info(
                    "训练步骤完成: global_step=%s/%s data_epoch=%s batch=%s/%s batch_loss=%.6f",
                    self.global_step,
                    # Completes the multi-line `self.logger.info` arguments or elements.
                    self.resolved_max_steps,
                    self.data_epoch + 1,
                    self.batch_in_epoch,
                    len(train_loader),
                    # Completes the multi-line `self.logger.info` arguments or elements.
                    loss_value,
                )

                reached_interval = self.global_step % self.resolved_validate_every == 0
                reached_final_step = self.global_step >= self.resolved_max_steps
                # Checks `reached_interval or reached_final_step` before entering this conditional branch.
                if reached_interval or reached_final_step:
                    if self._run_validation_cycle(
                        valid_loader=valid_loader,
                        progress_callback=progress_callback,
                        # Completes the multi-line `self._run_validation_cycle` arguments or elements.
                        terminate_training_checker=terminate_training_checker,
                        interval_train_loss_sum=interval_train_loss_sum,
                        interval_train_weight_sum=interval_train_weight_sum,
                    ):
                        # Returns `history` to the caller for the next stage.
                        return history
                    interval_train_loss_sum = 0.0
                    interval_train_weight_sum = 0
                    self.model.train()

                # Checks `self.global_step >= self.resolved_max_steps` before entering this conditional branch.
                if self.global_step >= self.resolved_max_steps:
                    break

            if resume_batch_in_epoch > len(train_loader):
                self.resume_exact = False
                # Records the current execution state to make progress and failures traceable.
                self.logger.warning(
                    "checkpoint 的 batch_in_epoch=%s 超出当前数据轮次批次数=%s，将从下一数据轮次恢复",
                    resume_batch_in_epoch,
                    len(train_loader),
                # Completes the multi-line `self.logger.warning` arguments or elements.
                )

            if self.batch_in_epoch >= len(train_loader):
                self.data_epoch += 1
                self.start_epoch = self.data_epoch
                # Stores the `batch_in_epoch` on the instance so other methods can reuse it.
                self.batch_in_epoch = 0

            if self.global_step >= self.resolved_max_steps:
                break

        return history

    # 轮询外部终止信号，支持训练过程中安全退出。
    def _should_terminate_training(
        self,
        terminate_training_checker: Optional[Callable[[], bool]],
    ) -> bool:
        """轮询外部终止信号，支持训练过程中安全退出。"""
        # Returns early when `terminate_training_checker is None` makes the remaining work inapplicable.
        if terminate_training_checker is None:
            return False
        return bool(terminate_training_checker())

    def _record_progress(
        # Continues the surrounding statement or collection definition.
        self,
        epoch: int,
        total_step: int,
        train_loss: float,
        # Continues the surrounding statement or collection definition.
        val_loss: float,
    ) -> None:
        """记录一次验证完成后的进度，供 checkpoint 与外部回调共同使用。"""
        current_timestamp = int(time.time_ns() // 1_000_000)
        current_lr = None
        # Checks `getattr(self.optimizer, "param_groups", None)` before entering this conditional branch.
        if getattr(self.optimizer, "param_groups", None):
            current_lr = self.optimizer.param_groups[0].get("lr")

        progress_item = {
            "epoch": int(epoch),
            # Completes the multi-line `int` arguments or elements.
            "numEpochs": int(self._estimated_validation_checks()),
            "totalStep": int(total_step),
            "loss": float(train_loss),
            "valLoss": None if val_loss is None else float(val_loss),
            # Completes the multi-line `float` arguments or elements.
            "lr": None if current_lr is None else float(current_lr),
            "timestamp": current_timestamp,
            "stepLoss": {},
        }
        # Adds the current value to `self.progress_history` for later aggregation.
        self.progress_history.append(progress_item)

    def _emit_progress(
        self,
        progress_callback: Optional[Callable[[Dict], None]],
    # Continues the surrounding statement or collection definition.
    ) -> None:
        """在每个验证周期完成后向外部回调累计训练进度。"""
        if progress_callback is None:
            return

        progress = {
            # Completes the multi-line `list` arguments or elements.
            "data": list(self.progress_history),
            "size": len(self.progress_history),
            "start": int(self.progress_start_timestamp),
        }
        # Calls `progress_callback` to perform the operation required at this stage.
        progress_callback(progress)

    def save_checkpoint(self, path: str, epoch: int = 0, val_loss: float = 0.0) -> None:
        artifact_policy = get_processor_checkpoint_artifact_policy(self.processor)

        norm_params = None
        # Stores the `norm_save_path` for use by the remaining steps in this flow.
        norm_save_path = os.path.join(self.processed_data_path, "norm_params.pt")
        if os.path.exists(norm_save_path):
            norm_params = load_torch(norm_save_path, weights_only=False)

        modes = None
        # Checks `artifact_policy["modes"]` before entering this conditional branch.
        if artifact_policy["modes"]:
            modes_save_path = os.path.join(self.processed_data_path, "modes.pt")
            if os.path.exists(modes_save_path):
                modes = load_torch(modes_save_path, weights_only=False)

        # Stores the `processor_state` for use by the remaining steps in this flow.
        processor_state = None
        if artifact_policy["processor_state"]:
            processor_state_path = os.path.join(self.processed_data_path, "processor_state.pt")
            if os.path.exists(processor_state_path):
                # Stores the `processor_state` for use by the remaining steps in this flow.
                processor_state = load_torch(processor_state_path, weights_only=False)

        variable_manifest = {
            "logical_variables": self.config.get("task", {}).get("logical_variables", []),
            "effective_variables": self.config.get("task", {}).get("effective_variables", []),
            # Completes the multi-line `self.config.get('task', {}).get` arguments or elements.
            "field_component_map": self.config.get("task", {}).get("field_component_map", {}),
            "variable_catalog": self.config.get("task", {}).get("variable_catalog", {}),
            "effective_variable_sources": self.config.get("task", {}).get("effective_variable_sources", {}),
            "point_or_cell": self.config.get("task", {}).get("point_or_cell"),
            # Completes the multi-line `self.config.get('task', {}).get` arguments or elements.
            "field_location_map": self.config.get("task", {}).get("field_location_map", {}),
            "projected_point_or_cell": self.config.get("task", {}).get("projected_point_or_cell"),
            "mixed_field_policy": self.config.get("task", {}).get("mixed_field_policy"),
            "projection_method": self.config.get("task", {}).get("projection_method"),
            # Completes the multi-line `self.config.get('task', {}).get` arguments or elements.
            "mesh_dimension": self.config.get("task", {}).get("mesh_dimension"),
            "geometry_regime": self.config.get("task", {}).get("geometry_regime"),
            "geometry_motion": self.config.get("task", {}).get("geometry_motion", {}),
            "geometry_representation": self.config.get("task", {}).get("geometry_representation"),
            # Completes the multi-line `self.config.get('task', {}).get` arguments or elements.
            "point_feature_fields": self.config.get("task", {}).get("point_feature_fields", []),
            "point_target_fields": self.config.get("task", {}).get("point_target_fields", []),
            "displacement_logical_variable": self.config.get("task", {}).get("displacement_logical_variable"),
            "displacement_effective_variables": self.config.get("task", {}).get("displacement_effective_variables", []),
        # Completes the multi-line `collection` arguments or elements.
        }
        history_snapshot = {
            "train_loss": list(self.history.get("train_loss", [])),
            "val_loss": list(self.history.get("val_loss", [])),
            # Completes the multi-line `list` arguments or elements.
            "train_units": list(self.train_units),
            "val_units": list(self.val_units),
            "schedule_unit": "step",
            "train_loss_scope": self.history.get("train_loss_scope", "single_step"),
            # Completes the multi-line `self.history.get` arguments or elements.
            "val_loss_scope": self.history.get("val_loss_scope", "single_step"),
            "rollout_horizon": self.history.get("rollout_horizon", 1),
        }

        atomic_torch_save(
            # Completes the multi-line `collection` arguments or elements.
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": None if self.lr_scheduler is None else self.lr_scheduler.state_dict(),
                "epoch": epoch,
                # Completes the multi-line `collection` arguments or elements.
                "val_loss": val_loss,
                "best_score": self.best_score,
                "early_stopping_counter": self.early_stopping_counter,
                "history": history_snapshot,
                # Completes the multi-line `copy.deepcopy` arguments or elements.
                "progress_history": copy.deepcopy(self.progress_history),
                "progress_start_timestamp": self.progress_start_timestamp,
                "rng_state": capture_rng_state(),
                "train_loader_generator_state": self.train_loader_generator.get_state(),
                # Completes the multi-line `collection` arguments or elements.
                "global_step": self.global_step,
                "data_epoch": self.data_epoch,
                "batch_in_epoch": self.batch_in_epoch,
                "steps_per_epoch": self.steps_per_epoch,
                # Completes the multi-line `collection` arguments or elements.
                "resolved_max_steps": self.resolved_max_steps,
                "resolved_validate_every": self.resolved_validate_every,
                "last_validation_step": self.last_validation_step,
                "validation_checks": self.validation_checks,
                # Completes the multi-line `copy.deepcopy` arguments or elements.
                "validation_manifest": copy.deepcopy(self.validation_manifest),
                "train_manifest": copy.deepcopy(self.train_manifest),
                "temporal_rollout": copy.deepcopy(self.temporal_rollout),
                "autoregressive_prediction_mode": self.autoregressive_prediction_mode,
                # Completes the multi-line `collection` arguments or elements.
                "resume_exact": self.resume_exact,
                "sampler_state": {
                    "strategy": "seed_plus_data_epoch",
                    "seed": self.seed,
                    # Completes the multi-line `collection` arguments or elements.
                    "data_epoch": self.data_epoch,
                    "batch_in_epoch": self.batch_in_epoch,
                },
                "config": sanitize_config_for_checkpoint(self.config),
                # Completes the multi-line `collection` arguments or elements.
                "norm_params": norm_params,
                "processor_state": processor_state,
                "modes": modes,
                "variable_manifest": variable_manifest,
            # Completes the multi-line `collection` arguments or elements.
            },
            path,
        )

    def update_checkpoint_evaluation(
        # Continues the surrounding statement or collection definition.
        self,
        path: str,
        metrics: Optional[Dict] = None,
        split: str = "valid",
        # Continues the surrounding statement or collection definition.
        extra_info: Optional[Dict] = None,
        evaluate_result: Optional[Dict] = None,
        field_name: Optional[str] = None,
    ) -> None:
        # Stores the `checkpoint` for use by the remaining steps in this flow.
        checkpoint = load_torch(path, map_location="cpu", weights_only=False)
        normalized_field_name = str(field_name or "").strip()
        if not normalized_field_name:
            normalized_split = str(split or "").strip().lower()
            # Checks `normalized_split == "train"` before entering this conditional branch.
            if normalized_split == "train":
                normalized_field_name = "evaluate_result_train"
            elif normalized_split == "valid":
                normalized_field_name = "evaluate_result_valid"
            # Continues the surrounding statement or collection definition.
            else:
                normalized_field_name = "evaluate_result_test"

        payload = copy.deepcopy(evaluate_result) if evaluate_result is not None else {
            "metrics": copy.deepcopy(metrics or {}),
            # Completes the multi-line `collection` arguments or elements.
            "scatter_data": {
                "overall": {"predictions": [], "targets": []},
                "by_variable": {},
            },
            # Completes the multi-line `collection` arguments or elements.
            "samples_evaluated": [],
        }
        checkpoint[normalized_field_name] = payload
        if extra_info is not None:
            # Stores the `checkpoint['extra_info']` for use by the remaining steps in this flow.
            checkpoint["extra_info"] = copy.deepcopy(extra_info)
        atomic_torch_save(checkpoint, path)

    def load_checkpoint(self, path: str) -> Dict:
        # RNG and DataLoader generator states must remain CPU ByteTensors.
        # Model/optimizer load_state_dict moves parameter states to their devices.
        checkpoint = load_torch(path, map_location="cpu", weights_only=False)
        # Stores the `saved_config` for use by the remaining steps in this flow.
        saved_config = checkpoint.get("config") or {}
        saved_task_config = saved_config.get("task") if isinstance(saved_config, dict) else {}
        saved_prediction_mode = self._resolve_effective_autoregressive_prediction_mode(
            task_config=saved_task_config,
            # Completes the multi-line `checkpoint.get` arguments or elements.
            declared_mode=checkpoint.get("autoregressive_prediction_mode", "state"),
        )
        current_prediction_mode = self.autoregressive_prediction_mode
        if saved_prediction_mode != current_prediction_mode:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(
                "checkpoint 的 autoregressive_prediction_mode 与当前训练目标不一致，无法继续训练"
            )
        saved_rollout = checkpoint.get("temporal_rollout")
        # Checks `isinstance(saved_rollout, dict)` before entering this conditional branch.
        if isinstance(saved_rollout, dict):
            # 旧 checkpoint 未记录该字段时等价于首版默认强度。
            saved_rollout = dict(saved_rollout)
            saved_rollout.setdefault("training_noise_alpha", 0.01)
            if not self._is_temporal_autoregressive_task(saved_task_config):
                # Stores the `saved_rollout` for use by the remaining steps in this flow.
                saved_rollout = self._disabled_temporal_rollout_config()
        if saved_rollout is not None and saved_rollout != self.temporal_rollout:
            raise ValueError(
                "checkpoint 的 temporal_rollout 配置与当前训练目标不一致，无法继续训练"
            # Completes the multi-line `ValueError` arguments or elements.
            )
        if self.temporal_rollout["enabled"] and saved_rollout is None:
            raise ValueError(
                "checkpoint 缺少 temporal_rollout 配置，无法安全恢复多步自由滚动训练"
            # Completes the multi-line `ValueError` arguments or elements.
            )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        restore_lr_scheduler(self.lr_scheduler, checkpoint)

        self.best_score = checkpoint.get("best_score", checkpoint.get("val_loss", self.best_score))
        # Stores the `early_stopping_counter` on the instance so other methods can reuse it.
        self.early_stopping_counter = int(checkpoint.get("early_stopping_counter", 0))

        history = checkpoint.get("history", {})
        self.history = {
            "train_loss": list(history.get("train_loss", [])),
            # Completes the multi-line `list` arguments or elements.
            "val_loss": list(history.get("val_loss", [])),
            "train_units": [int(value) for value in history.get("train_units", [])],
            "val_units": [int(value) for value in history.get("val_units", [])],
            "schedule_unit": str(history.get("schedule_unit", "step")),
            # Completes the multi-line `str` arguments or elements.
            "train_loss_scope": str(history.get("train_loss_scope", self.temporal_rollout["loss_scope"])),
            "val_loss_scope": str(history.get("val_loss_scope", self.temporal_rollout["loss_scope"])),
            "rollout_horizon": int(history.get("rollout_horizon", self.temporal_rollout["horizon"])),
        }
        # Stores the `train_units` on the instance so other methods can reuse it.
        self.train_units = list(self.history["train_units"])
        self.val_units = list(self.history["val_units"])

        saved_progress_history = checkpoint.get("progress_history", [])
        if isinstance(saved_progress_history, list):
            # Stores the `progress_history` on the instance so other methods can reuse it.
            self.progress_history = [
                dict(item)
                for item in saved_progress_history
                if isinstance(item, dict)
            # Continues the surrounding statement or collection definition.
            ]
        else:
            self.progress_history = []
        saved_progress_start = checkpoint.get("progress_start_timestamp", 0)
        # Keeps the operation and its recovery path within the same exception boundary.
        try:
            self.progress_start_timestamp = int(saved_progress_start or 0)
        except (TypeError, ValueError):
            self.progress_start_timestamp = 0

        # Checks `"global_step" in checkpoint` before entering this conditional branch.
        if "global_step" in checkpoint:
            self._loaded_step_checkpoint = True
            self.global_step = int(checkpoint.get("global_step", 0))
            self.data_epoch = int(checkpoint.get("data_epoch", checkpoint.get("epoch", 0)))
            # Stores the `batch_in_epoch` on the instance so other methods can reuse it.
            self.batch_in_epoch = int(checkpoint.get("batch_in_epoch", 0))
            self._loaded_steps_per_epoch = int(checkpoint.get("steps_per_epoch", 0) or 0) or None
            self.start_epoch = self.data_epoch
            if self._loaded_steps_per_epoch and self.batch_in_epoch >= self._loaded_steps_per_epoch:
                # Stores the `start_epoch` on the instance so other methods can reuse it.
                self.start_epoch += 1
            self.resolved_max_steps = checkpoint.get("resolved_max_steps")
            self.resolved_validate_every = checkpoint.get("resolved_validate_every")
            self.last_validation_step = int(checkpoint.get("last_validation_step", self.global_step))
            # Stores the `validation_checks` on the instance so other methods can reuse it.
            self.validation_checks = int(
                checkpoint.get("validation_checks", len(self.history["val_loss"]))
            )
            manifest = checkpoint.get("validation_manifest")
            # Stores the `validation_manifest` on the instance so other methods can reuse it.
            self.validation_manifest = copy.deepcopy(manifest) if manifest else None
            train_manifest = checkpoint.get("train_manifest")
            self.train_manifest = copy.deepcopy(train_manifest) if isinstance(train_manifest, dict) else None
            self.resume_exact = bool(checkpoint.get("resume_exact", True))
            # Stores the `_resume_from_legacy_checkpoint` on the instance so other methods can reuse it.
            self._resume_from_legacy_checkpoint = False
            if len(self.train_units) != len(self.history["train_loss"]):
                self.train_units = []
                self.history["train_units"] = []
            # Checks `len(self.val_units) != len(self.history["val_loss"])` before entering this conditional branch.
            if len(self.val_units) != len(self.history["val_loss"]):
                self.val_units = []
                self.history["val_units"] = []
        else:
            # Stores the `_loaded_step_checkpoint` on the instance so other methods can reuse it.
            self._loaded_step_checkpoint = False
            self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
            self.data_epoch = self.start_epoch
            self.batch_in_epoch = 0
            # Stores the `global_step` on the instance so other methods can reuse it.
            self.global_step = 0
            self.last_validation_step = 0
            self.validation_checks = len(self.history["val_loss"])
            self.resume_exact = False
            # Stores the `_resume_from_legacy_checkpoint` on the instance so other methods can reuse it.
            self._resume_from_legacy_checkpoint = True
            self.logger.warning("加载旧版 checkpoint，将在解析当前训练集后以轮次边界方式续训")

        generator_state = checkpoint.get("train_loader_generator_state")
        if generator_state is not None:
            # Calls `self.train_loader_generator.set_state` to perform the operation required at this stage.
            self.train_loader_generator.set_state(generator_state)

        restore_rng_state(checkpoint.get("rng_state"))
        return checkpoint

    @staticmethod
    # Defines `_format_log_loss_tick` for  format log loss tick logic.
    def _format_log_loss_tick(value: float, _position: int) -> str:
        if value <= 0 or not math.isfinite(value):
            return ""
        return f"{value:.1e}"

    # Defines `_plot_loss_curve` for  plot loss curve logic.
    def _plot_loss_curve(self, history: Dict, epoch: int) -> None:
        os.makedirs(self.path, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 8))
        train_losses = history["train_loss"]
        # Selects the required item from `history` as the `val_losses`.
        val_losses = history["val_loss"]
        # Plot one point per completed validation cycle. The stored units are
        # optimizer steps, but the training curve is indexed by validation cycle.
        train_units = list(range(1, len(train_losses) + 1))
        val_units = list(range(1, len(val_losses) + 1))
        ax.set_xticks(range(1, max(len(train_losses), len(val_losses)) + 1))

        ax.plot(train_units, train_losses, label="Train Loss", color="blue", linewidth=2)
        # Calls `ax.plot` to perform the operation required at this stage.
        ax.plot(val_units, val_losses, label="Val Loss", color="green", linewidth=2)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_xlabel("Training Epoch", fontsize=22)
        ax.set_ylabel("Loss", fontsize=22)

        # Checks `self.loss_curve_log_scale` before entering this conditional branch.
        if self.loss_curve_log_scale:
            ax.set_title("Training and Validation Loss (Log Scale)", fontsize=24, fontweight="bold")
        else:
            ax.set_title("Training and Validation Loss", fontsize=24, fontweight="bold")

        # Calls `ax.tick_params` to perform the operation required at this stage.
        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=20)
        ax.legend(loc="best", fontsize=20)

        if self.loss_curve_log_scale:
            # Calls `ax.set_yscale` to perform the operation required at this stage.
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(self._format_log_loss_tick))

        fig.tight_layout()
        fig.savefig(os.path.join(self.path, "loss_curve.png"), dpi=150)
        # Calls `plt.close` to perform the operation required at this stage.
        plt.close(fig)


class ResidualStageTrainer(Trainer):
    """用于第二阶段残差学习的训练器。"""

    def __init__(
        self,
        # Continues the surrounding statement or collection definition.
        model,
        criterion: callable,
        optimizer: callable,
        config,
        # Continues the surrounding statement or collection definition.
        processor=None,
        teacher_model=None,
        residual_norm_params=None,
        logger: Optional[logging.Logger] = None,
        # Continues the surrounding statement or collection definition.
        residual_validation_uses_combined_output: bool = True,
    ) -> None:
        if teacher_model is None:
            raise ValueError("残差训练器缺少第一阶段 teacher_model。")

        # Calls `super().__init__` to perform the operation required at this stage.
        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            # Completes the multi-line `super().__init__` arguments or elements.
            config=config,
            processor=processor,
            logger=logger,
        )
        # Stores the `teacher_model` on the instance so other methods can reuse it.
        self.teacher_model = teacher_model
        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        for parameter in self.teacher_model.parameters():
            # Calls `parameter.requires_grad_` to perform the operation required at this stage.
            parameter.requires_grad_(False)
        self.residual_norm_params = residual_norm_params
        self.residual_validation_uses_combined_output = bool(
            residual_validation_uses_combined_output
        # Completes the multi-line `bool` arguments or elements.
        )

    def _normalize_residual_targets(self, residual_targets, inputs):
        if (
            not self.residual_norm_params
            # Continues the surrounding statement or collection definition.
            or self.processor is None
            or not hasattr(self.processor, "normalize_output_tensor")
        ):
            return residual_targets
        # Returns `` to the caller for the next stage.
        return self.processor.normalize_output_tensor(
            residual_targets,
            self.residual_norm_params,
            inputs=inputs,
        # Completes the multi-line `self.processor.normalize_output_tensor` arguments or elements.
        )

    def _denormalize_stage2_outputs(self, outputs, inputs):
        if (
            not self.residual_norm_params
            # Continues the surrounding statement or collection definition.
            or self.processor is None
            or not hasattr(self.processor, "denormalize_output_tensor")
        ):
            return outputs
        # Returns `` to the caller for the next stage.
        return self.processor.denormalize_output_tensor(
            outputs,
            self.residual_norm_params,
            inputs=inputs,
        # Completes the multi-line `self.processor.denormalize_output_tensor` arguments or elements.
        )

    def _compute_train_batch_loss(self, inputs, targets):
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        # Derives the `residual_targets` from the current values for downstream processing.
        residual_targets = targets - teacher_outputs.detach()
        residual_targets = self._normalize_residual_targets(residual_targets, inputs)
        outputs = self.model(inputs)
        if self.processor is not None:
            # Returns `` to the caller for the next stage.
            return self.processor.compute_train_loss(
                criterion=self.criterion,
                outputs=outputs,
                targets=residual_targets,
                # Completes the multi-line `self.processor.compute_train_loss` arguments or elements.
                inputs=inputs,
            )
        return self.criterion(outputs, residual_targets)

    def _compute_eval_batch_loss(self, inputs, targets):
        # Scopes resource use so the context manager can release it on exit.
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        outputs = self.model(inputs)
        if self.residual_validation_uses_combined_output:
            # Derives the `outputs` from the current values for downstream processing.
            outputs = teacher_outputs.detach() + self._denormalize_stage2_outputs(outputs, inputs)

        if self.processor is not None:
            return self.processor.compute_eval_loss(
                criterion=self.criterion,
                # Completes the multi-line `self.processor.compute_eval_loss` arguments or elements.
                outputs=outputs,
                targets=targets,
                inputs=inputs,
            )
        # Returns `self.criterion(outputs, targets)` to the caller for the next stage.
        return self.criterion(outputs, targets)
