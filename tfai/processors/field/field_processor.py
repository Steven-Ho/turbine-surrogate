from abc import ABC, abstractmethod
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Imports `numpy` for reuse in this module.
import numpy as np
import torch
from torch.utils.data import Dataset

from ...data.base_data import BaseData
# Imports `build_basic_criterion, weighted_huber, weighted_mae` from `training.losses` for use in this module.
from ...training.losses import (
    build_basic_criterion,
    weighted_huber,
    weighted_mae,
    # Continues the surrounding statement or collection definition.
    weighted_mse,
)
from ...utils.boundary_feature_encoding import (
    build_boundary_feature_matrix,
    # Continues the surrounding statement or collection definition.
    build_field_bc_feature_matrix,
    fit_field_bc_schema,
    normalize_field_bc_names,
)
# Imports `atomic_torch_save, load_torch` from `utils.torch_utils` for use in this module.
from ...utils.torch_utils import atomic_torch_save, load_torch


class FieldProcessor(ABC):
    """场数据处理器基类。

    所有 field 任务处理器都继承自该类，并共享以下能力：
    - 原始样本处理与保存
    - 训练前后所需的运行时接口
    - 归一化、反归一化
    - 预测与评估阶段的辅助协议
    """

    def __init__(self, config: Dict):
        """初始化场处理器。"""
        self.config = config
        # Stores the `mode` on the instance so other methods can reuse it.
        self.mode = config["task"]["mode"]
        self.temporal_strategy = config["task"].get("temporal_strategy", "direct")
        self.is_geometry_varying = config["task"]["is_geometry_varying"]
        self.local_feature_encoding = self._parse_local_feature_encoding(config["task"])
        # Stores the `field_bc` on the instance so other methods can reuse it.
        self.field_bc = (
            normalize_field_bc_names(config["task"].get("field_bc"))
            if self.local_feature_encoding == "vtk_based"
            else []
        # Continues the surrounding statement or collection definition.
        )
        self.local_feature_schema: Optional[Dict[str, Any]] = None
        self.grid_type = config["task"].get("grid_type")
        self._resolved_loss_name = "mse"
        # Stores the `autoregressive_prediction_mode` on the instance so other methods can reuse it.
        self.autoregressive_prediction_mode = self._resolve_autoregressive_prediction_mode()
        self._autoregressive_norm_params: Optional[Dict[str, Any]] = None
        # 仅由训练集拟合；验证、评估和预测阶段只读取该规则。
        self.boundary_state_override_schema: Optional[Dict[str, Any]] = None

    def _resolve_autoregressive_prediction_mode(self) -> str:
        """解析时序自回归模型输出语义，并拒绝尚未覆盖的处理器组合。"""
        # Reads `task` from the mapping into the task settings with its declared fallback.
        task_cfg = self.config.get("task", {})
        is_autoregressive = (
            str(task_cfg.get("mode", "")).lower() == "temporal"
            and str(task_cfg.get("temporal_strategy", "direct")).lower()
            # Continues the surrounding statement or collection definition.
            == "autoregressive"
        )
        # 该参数只描述时序自回归链路的输出；其他任务必须保持状态预测语义。
        if not is_autoregressive:
            return "state"

        # Stores the `mode` for use by the remaining steps in this flow.
        mode = str(task_cfg.get("autoregressive_prediction_mode", "state")).strip().lower()
        if mode not in {"state", "increment"}:
            raise ValueError("task.autoregressive_prediction_mode 仅支持 state 或 increment")
        if mode == "state":
            # Returns `mode` to the caller for the next stage.
            return mode

        if int(task_cfg.get("output_length", 1)) != 1:
            raise ValueError(
                "autoregressive_prediction_mode=increment 仅支持 temporal autoregressive 且 output_length=1"
            # Completes the multi-line `ValueError` arguments or elements.
            )
        if type(self).__name__ not in {
            "MatrixProcessor",
            "GraphProcessor",
            # Completes the multi-line `collection` arguments or elements.
            "PointCloudProcessor",
            "ImageVoxelProcessor",
        }:
            raise NotImplementedError(
                # Completes the multi-line `type` arguments or elements.
                f"{type(self).__name__} 暂不支持 autoregressive_prediction_mode=increment"
            )
        geometry_regime = str(task_cfg.get("geometry_regime", "")).strip().lower()
        coordinate_convention = str(
            # Completes the multi-line `task_cfg.get` arguments or elements.
            task_cfg.get("displacement_coordinate_convention", "")
        ).strip().lower()
        if (
            type(self).__name__ == "GraphProcessor"
            # Continues the surrounding statement or collection definition.
            and geometry_regime == "displaced_static_topology"
            and coordinate_convention not in {"", "reference_absolute"}
        ):
            raise ValueError(
                # Completes the multi-line `ValueError` arguments or elements.
                "运动网格增量自回归仅支持 displacement_coordinate_convention=reference_absolute"
            )
        return mode

    def uses_autoregressive_increment(self) -> bool:
        # Returns `self.autoregressive_prediction_mode == "increment"` to the caller for the next stage.
        return self.autoregressive_prediction_mode == "increment"

    def set_normalization_params(self, norm_params: Optional[Dict[str, Any]]) -> None:
        """注入当前运行使用的归一化参数，供增量状态重建复用。"""
        self._autoregressive_norm_params = norm_params

    def _get_autoregressive_norm_params(self) -> Dict[str, Any]:
        # Stores the normalization statistics for use by the remaining steps in this flow.
        norm_params = self._autoregressive_norm_params
        if not norm_params and hasattr(self, "_eval_norm_params"):
            norm_params = getattr(self, "_eval_norm_params")
        if not norm_params and isinstance(self.config, dict):
            # Reads `__eval_norm_params__` from the mapping into the normalization statistics.
            norm_params = self.config.get("__eval_norm_params__")
        if not isinstance(norm_params, dict):
            raise ValueError("增量自回归缺少归一化参数")
        if "y" not in norm_params or "autoregressive_delta" not in norm_params:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("增量自回归归一化参数缺少 y 或 autoregressive_delta")
        return norm_params

    @staticmethod
    def _reshape_channel_stats(values: Any, reference: torch.Tensor) -> torch.Tensor:
        # Stores the `stats` for use by the remaining steps in this flow.
        stats = torch.as_tensor(values, device=reference.device, dtype=reference.dtype)
        if stats.ndim != 1 or stats.numel() != reference.shape[-1]:
            raise ValueError(
                f"增量统计通道数不匹配：stats={tuple(stats.shape)}，输出={tuple(reference.shape)}"
            # Completes the multi-line `ValueError` arguments or elements.
            )
        return stats.reshape(*([1] * (reference.ndim - 1)), -1)

    def restore_autoregressive_state(
        self,
        # Continues the surrounding statement or collection definition.
        last_state: torch.Tensor,
        raw_output: torch.Tensor,
    ) -> torch.Tensor:
        """将模型原始输出恢复为与历史相同坐标中的下一绝对状态。"""
        if not self.uses_autoregressive_increment():
            # Returns `raw_output` to the caller for the next stage.
            return raw_output

        norm_params = self._get_autoregressive_norm_params()
        y_stats = norm_params["y"]
        delta_stats = norm_params["autoregressive_delta"]
        # Stores the `norm_type` for use by the remaining steps in this flow.
        norm_type = str(norm_params.get("norm_type", "")).lower()
        if norm_type == "standardization":
            state_scale = self._reshape_channel_stats(y_stats.get("std"), raw_output)
            delta_offset = self._reshape_channel_stats(delta_stats.get("mean"), raw_output)
            # Stores the `delta_scale` for use by the remaining steps in this flow.
            delta_scale = self._reshape_channel_stats(delta_stats.get("std"), raw_output)
        elif norm_type == "min_max":
            state_min = self._reshape_channel_stats(y_stats.get("min"), raw_output)
            state_max = self._reshape_channel_stats(y_stats.get("max"), raw_output)
            # Stores the `delta_offset` for use by the remaining steps in this flow.
            delta_offset = self._reshape_channel_stats(delta_stats.get("min"), raw_output)
            delta_max = self._reshape_channel_stats(delta_stats.get("max"), raw_output)
            state_scale = state_max - state_min
            delta_scale = delta_max - delta_offset
        # Continues the surrounding statement or collection definition.
        else:
            raise ValueError(f"增量自回归不支持的归一化方式: {norm_type}")
        return last_state + (raw_output * delta_scale + delta_offset) / state_scale.clamp_min(1e-12)

    def restore_autoregressive_model_output(self, inputs: Any, raw_output: Any) -> Any:
        """单步训练路径的处理器扩展点；未实现时拒绝增量模式。"""
        # Validates `self.uses_autoregressive_increment()` and raises `NotImplementedError` when the input cannot be processed safely.
        if self.uses_autoregressive_increment():
            raise NotImplementedError(
                f"{type(self).__name__} 尚未实现增量自回归单步状态重建"
            )
        # Returns `raw_output` to the caller for the next stage.
        return raw_output

    def build_boundary_mask(
        self,
        node_features: torch.Tensor,
        # Completes the multi-line `collection` arguments or elements.
        meta: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        """从局部边界条件特征构建与目标位置对齐的边界掩码。"""
        if not isinstance(node_features, torch.Tensor) or node_features.ndim != 2:
            raise TypeError("构建 boundary_mask 需要二维节点特征张量")
        # Validates `not isinstance(meta, dict)` and raises `ValueError` when the input cannot be processed safely.
        if not isinstance(meta, dict):
            raise ValueError("training_noise 需要样本携带局部边界条件元信息")

        if self.local_feature_encoding == "cgns_based":
            names = list(meta.get("feature_bc_names", []))
            # Stores the `indices` for use by the remaining steps in this flow.
            indices = list(meta.get("feature_bc_indices", []))
            try:
                feature_index = indices[names.index("bc_is_boundary")]
            except (ValueError, IndexError) as exc:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError("cgns_based 缺少 bc_is_boundary 特征，无法启用训练噪声") from exc
            return node_features[:, int(feature_index)] > 0

        if self.local_feature_encoding == "vtk_based":
            categorical_indices = [int(index) for index in meta.get("categorical_bc_indices", [])]
            # Validates `not categorical_indices` and raises `ValueError` when the input cannot be processed safely.
            if not categorical_indices:
                raise ValueError(
                    "vtk_based 缺少离散 field_bc 边界类别，无法启用训练噪声"
                )
            # Returns `node_features[:, categorical_indices].abs().sum(dim=1) > 0` to the caller for the next stage.
            return node_features[:, categorical_indices].abs().sum(dim=1) > 0

        raise ValueError(
            "training_noise 仅支持带 cgns_based 或 vtk_based 局部边界条件的处理器"
        )

    # 向图或点云 rollout 视图附加统一的边界掩码。
    def attach_boundary_mask(
        self,
        view: Any,
        *,
        # Completes the multi-line `collection` arguments or elements.
        meta: Optional[Dict[str, Any]],
        boundary_mask: Optional[torch.Tensor] = None,
    ) -> Any:
        """向图或点云 rollout 视图附加统一的边界掩码。"""
        if not hasattr(view, "x"):
            # Raises `TypeError` to report why this path cannot continue.
            raise TypeError("边界掩码仅支持含 x 属性的图或点云视图")
        if boundary_mask is None:
            boundary_mask = self.build_boundary_mask(view.x, meta)
        if not isinstance(boundary_mask, torch.Tensor) or boundary_mask.ndim != 1:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("boundary_mask 必须为一维张量")
        if int(boundary_mask.shape[0]) != int(view.x.shape[0]):
            raise ValueError("boundary_mask 与 rollout 视图节点/点数量不一致")
        view.boundary_mask = boundary_mask.to(device=view.x.device, dtype=torch.bool)
        # Returns `view` to the caller for the next stage.
        return view

    def build_boundary_entity_codes(
        self,
        node_features: torch.Tensor,
        # Completes the multi-line `collection` arguments or elements.
        meta: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        """以离散局部边界条件签名构建稳定的边界实体分组编码。

        CGNS 当前公开的是边界类型通道，VTK 当前公开的是离散 field_bc 通道。
        因而首版以这些离散通道的组合为分组依据；同一签名中的任一节点或样本
        不是静态状态时，整组均不会被回写，保证行为保守。
        """
        boundary_mask = self.build_boundary_mask(node_features, meta)
        if self.local_feature_encoding == "cgns_based":
            # Builds the `indices` using the comprehension's filter and mapping rules.
            indices = [int(index) for index in meta.get("feature_bc_indices", [])]
            names = list(meta.get("feature_bc_names", []))
            type_indices = [
                indices[index]
                # Completes the multi-line `collection` arguments or elements.
                for index, name in enumerate(names)
                if str(name) != "bc_is_boundary" and index < len(indices)
            ]
        else:
            # Builds the `type_indices` using the comprehension's filter and mapping rules.
            type_indices = [int(index) for index in meta.get("categorical_bc_indices", [])]
        if not type_indices:
            raise ValueError("boundary_state_override 缺少可用于分组的离散边界条件特征")

        active = node_features[:, type_indices] > 0
        # Stores the `bits` for use by the remaining steps in this flow.
        bits = (2 ** torch.arange(
            active.shape[1], device=active.device, dtype=torch.long
        )).reshape(1, -1)
        codes = (active.to(torch.long) * bits).sum(dim=1)
        # 0 专用于非边界；边界但没有类型通道的节点使用 1，避免与非边界混淆。
        # Stores the `codes` for use by the remaining steps in this flow.
        codes = torch.where(boundary_mask & (codes == 0), torch.ones_like(codes), codes)
        return codes.to(dtype=torch.long)

    def fit_boundary_state_override_schema(
        self,
        # Completes the multi-line `collection` arguments or elements.
        samples: List[Dict[str, Any]],
        *,
        tolerance: float = 1.0e-6,
    ) -> Dict[str, Any]:
        """只根据训练样本拟合静态边界状态规则。"""
        # Validates `not samples` and raises `ValueError` when the input cannot be processed safely.
        if not samples:
            raise ValueError("boundary_state_override 需要至少一个训练样本")
        if tolerance < 0:
            raise ValueError("boundary_state_override 的静态判定容差不能为负数")

        # Initializes the `group_static` as a collection for values from this processing stage.
        group_static: Dict[int, torch.Tensor] = {}
        output_dim: Optional[int] = None
        for sample in samples:
            y = sample.get("y")
            # Reads `boundary_mask` from the mapping into the boundary mask.
            boundary_mask = sample.get("boundary_mask")
            boundary_codes = sample.get("boundary_entity_codes")
            if boundary_mask is None or boundary_codes is None:
                sample_x = sample.get("x")
                # Stores the `node_features` for use by the remaining steps in this flow.
                node_features = getattr(sample_x, "x", None)
                meta = sample.get("meta", {})
                if isinstance(node_features, torch.Tensor):
                    # 兼容首版字段落盘前已生成的处理后样本；离散边界通道不参与归一化。
                    boundary_mask = self.build_boundary_mask(node_features, meta)
                    # Stores the `boundary_codes` for use by the remaining steps in this flow.
                    boundary_codes = self.build_boundary_entity_codes(node_features, meta)
            if not isinstance(y, torch.Tensor) or y.ndim != 3:
                raise ValueError("boundary_state_override 仅支持形状为 [T, N, C] 的时序目标")
            if not isinstance(boundary_mask, torch.Tensor) or boundary_mask.ndim != 1:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError("boundary_state_override 缺少与节点/点对齐的 boundary_mask")
            if not isinstance(boundary_codes, torch.Tensor) or boundary_codes.ndim != 1:
                raise ValueError("boundary_state_override 缺少边界实体分组编码")
            if y.shape[1] != boundary_mask.shape[0] or y.shape[1] != boundary_codes.shape[0]:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError("boundary_state_override 的目标、边界掩码和实体编码长度不一致")
            if y.shape[0] < 2:
                raise ValueError("boundary_state_override 需要每个训练样本至少包含两个时间步")
            if output_dim is None:
                # Converts the `output_dim` to `int` so later checks use a consistent type.
                output_dim = int(y.shape[-1])
            elif output_dim != int(y.shape[-1]):
                raise ValueError("boundary_state_override 的训练样本目标通道数不一致")

            delta = (y - y[:1]).abs().amax(dim=0)
            # Stores the `is_static` for use by the remaining steps in this flow.
            is_static = delta <= float(tolerance)
            for code in torch.unique(boundary_codes[boundary_mask]).tolist():
                code_mask = boundary_mask & (boundary_codes == int(code))
                code_static = is_static[code_mask].all(dim=0).cpu()
                # Checks `int(code) in group_static` before entering this conditional branch.
                if int(code) in group_static:
                    group_static[int(code)] &= code_static
                else:
                    group_static[int(code)] = code_static

        # Validates `output_dim is None or not group_static` and raises `ValueError` when the input cannot be processed safely.
        if output_dim is None or not group_static:
            raise ValueError("boundary_state_override 未在训练集发现可用的边界节点/点")
        schema = {
            "version": 1,
            # Completes the multi-line `collection` arguments or elements.
            "grouping": "categorical_boundary_signature",
            "tolerance_normalized": float(tolerance),
            "output_dim": output_dim,
            "groups": {
                # Completes the multi-line `str` arguments or elements.
                str(code): values.to(dtype=torch.bool).tolist()
                for code, values in sorted(group_static.items())
            },
        }
        # Stores the `boundary_state_override_schema` on the instance so other methods can reuse it.
        self.boundary_state_override_schema = schema
        return schema

    def attach_boundary_state_override(
        self,
        # Continues the surrounding statement or collection definition.
        view: Any,
        *,
        history_last_state: torch.Tensor,
        boundary_entity_codes: Optional[torch.Tensor],
    # Continues the surrounding statement or collection definition.
    ) -> Any:
        """将训练集拟合的静态规则附到单个滚动视图。"""
        schema = self.boundary_state_override_schema
        if schema is None:
            raise ValueError("boundary_state_override 尚未从训练集拟合规则")
        # Stores the boundary mask for use by the remaining steps in this flow.
        boundary_mask = getattr(view, "boundary_mask", None)
        if not isinstance(boundary_mask, torch.Tensor) or boundary_mask.ndim != 1:
            raise ValueError("boundary_state_override 需要先附加 boundary_mask")
        if not isinstance(boundary_entity_codes, torch.Tensor) or boundary_entity_codes.ndim != 1:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("boundary_state_override 缺少边界实体分组编码")
        if history_last_state.ndim != 2:
            raise ValueError("boundary_state_override 的初始状态必须为 [N, C]")
        if (
            # Continues the surrounding statement or collection definition.
            boundary_mask.shape[0] != history_last_state.shape[0]
            or boundary_entity_codes.shape[0] != history_last_state.shape[0]
        ):
            raise ValueError("boundary_state_override 的节点/点数量不一致")
        # Converts the `output_dim` to `int` so later checks use a consistent type.
        output_dim = int(schema.get("output_dim", 0))
        if output_dim != int(history_last_state.shape[-1]):
            raise ValueError("boundary_state_override 规则与当前模型输出通道数不一致")

        override_mask = torch.zeros_like(history_last_state, dtype=torch.bool)
        # Visits each item in `schema.get("groups", {}).items()` and applies the current processing step.
        for raw_code, components in schema.get("groups", {}).items():
            component_mask = torch.as_tensor(
                components, device=history_last_state.device, dtype=torch.bool
            )
            # Validates `component_mask.numel() != output_dim` and raises `ValueError` when the input cannot be processed safely.
            if component_mask.numel() != output_dim:
                raise ValueError("boundary_state_override 规则中的通道掩码长度无效")
            node_mask = boundary_mask & (boundary_entity_codes.to(boundary_mask.device) == int(raw_code))
            override_mask[node_mask] = component_mask
        # Stores the `boundary_state_override_mask` for use by the remaining steps in this flow.
        view.boundary_state_override_mask = override_mask
        # 只保留 seed 最后一个历史状态；绝不读取未来真值。
        view.boundary_state_override_values = history_last_state.clone()
        return view

    @staticmethod
    # 在模型输出转为绝对状态后，回写已知静态边界状态。
    def apply_boundary_state_override(inputs: Any, state_step: torch.Tensor) -> torch.Tensor:
        """在模型输出转为绝对状态后，回写已知静态边界状态。"""
        mask = getattr(inputs, "boundary_state_override_mask", None)
        values = getattr(inputs, "boundary_state_override_values", None)
        if mask is None and values is None:
            # Returns `state_step` to the caller for the next stage.
            return state_step
        if not isinstance(mask, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise ValueError("boundary_state_override 缺少掩码或初始状态值")
        if mask.shape != state_step.shape or values.shape != state_step.shape:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("boundary_state_override 的掩码、初始状态和值预测形状不一致")
        return torch.where(mask.to(device=state_step.device), values.to(device=state_step.device), state_step)

    def perturb_autoregressive_history(
        self,
        # Continues the surrounding statement or collection definition.
        inputs: Any,
        *,
        generator: torch.Generator,
        alpha: float = 0.01,
        # Continues the surrounding statement or collection definition.
        global_node_noise_seed: Optional[int] = None,
    ) -> Any:
        """扰动历史状态；启用回写时仅跳过被回写的节点分量。"""
        if not hasattr(inputs, "x") or not isinstance(inputs.x, torch.Tensor):
            raise TypeError("训练噪声仅支持图或点云 autoregressive 输入")
        # Validates `alpha <= 0` and raises `ValueError` when the input cannot be processed safely.
        if alpha <= 0:
            raise ValueError("训练噪声幅度必须为正数")

        norm_params = self._autoregressive_norm_params
        norm_type = str((norm_params or {}).get("norm_type", "")).lower()
        # Validates `norm_type not in {"standardization", "min_max"}` and raises `ValueError` when the input cannot be processed safely.
        if norm_type not in {"standardization", "min_max"}:
            raise ValueError(
                "training_noise 仅支持具有 standardization 或 min_max 归一化统计的目标变量"
            )

        # Stores the `perturbed` for use by the remaining steps in this flow.
        perturbed = inputs.clone()
        out_channels = int(self.config.get("model", {}).get("output_dim", 0))
        if out_channels <= 0:
            target_tensor = getattr(inputs, "y", None)
            # Checks `` before entering this conditional branch.
            if isinstance(target_tensor, torch.Tensor) and target_tensor.ndim >= 2:
                out_channels = int(target_tensor.shape[-1])
        if out_channels <= 0:
            raise ValueError("训练噪声无法从模型或监督目标确定输出通道数")
        # Converts the `input_length` to `int` so later checks use a consistent type.
        input_length = int(self.config.get("task", {}).get("input_length", 1))
        history_channels = input_length * out_channels
        static_dim = int(perturbed.x.shape[1]) - history_channels
        if static_dim < 0:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("训练噪声无法从输入特征中解析历史状态通道")

        history = perturbed.x[:, static_dim:].reshape(
            perturbed.x.shape[0], input_length, out_channels
        )
        # Checks `global_node_noise_seed is None` before entering this conditional branch.
        if global_node_noise_seed is None:
            innovation = torch.randn(
                history.shape,
                generator=generator,
                # Completes the multi-line `torch.randn` arguments or elements.
                device="cpu",
                dtype=history.dtype,
            )
        else:
            # Stores the `global_entity_id` for use by the remaining steps in this flow.
            global_entity_id = getattr(perturbed, "global_entity_id", None)
            if not isinstance(global_entity_id, torch.Tensor) or global_entity_id.ndim != 1:
                raise ValueError("图分块 training_noise 需要一维 global_entity_id")
            if int(global_entity_id.numel()) != int(history.shape[0]):
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError("global_entity_id 与训练噪声节点数不一致")
            sample_id = getattr(perturbed, "sample_id", 0)
            if isinstance(sample_id, torch.Tensor):
                if sample_id.numel() <= 0:
                    # Raises `ValueError` to report why this path cannot continue.
                    raise ValueError("sample_id 不能为空")
                sample_id = int(sample_id.reshape(-1)[0].item())
            else:
                sample_id = int(sample_id)
            # Derives the `modulus` from the current values for downstream processing.
            modulus = (1 << 63) - 1
            innovation = torch.empty(history.shape, device="cpu", dtype=history.dtype)
            for node_index, entity_id in enumerate(global_entity_id.detach().cpu().tolist()):
                node_generator = torch.Generator(device="cpu")
                # Derives the `node_seed` from the current values for downstream processing.
                node_seed = (
                    int(global_node_noise_seed)
                    + 6_364_136_223_846_793_005 * (int(entity_id) + 1)
                    + 1_442_695_040_888_963_407 * (sample_id + 1)
                # Continues the surrounding statement or collection definition.
                ) % modulus
                node_generator.manual_seed(node_seed)
                innovation[node_index] = torch.randn(
                    (input_length, out_channels),
                    # Completes the multi-line `torch.randn` arguments or elements.
                    generator=node_generator,
                    device="cpu",
                    dtype=history.dtype,
                )
        # Stores the `innovation` for use by the remaining steps in this flow.
        innovation = innovation.to(history.device)
        innovation = innovation * (float(alpha) / float(input_length) ** 0.5)
        random_walk_noise = innovation.cumsum(dim=1)
        override_mask = getattr(perturbed, "boundary_state_override_mask", None)
        # Checks `override_mask is not None` before entering this conditional branch.
        if override_mask is not None:
            if not isinstance(override_mask, torch.Tensor) or override_mask.ndim != 2:
                raise ValueError("boundary_state_override_mask 必须为 [N, C]")
            if override_mask.shape != (history.shape[0], history.shape[-1]):
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError("boundary_state_override_mask 与历史状态形状不一致")
            random_walk_noise = random_walk_noise * (~override_mask).unsqueeze(1).to(
                device=history.device,
                dtype=history.dtype,
            # Completes the multi-line `(~override_mask).unsqueeze(1).to` arguments or elements.
            )
        history = history + random_walk_noise
        perturbed.x = torch.cat(
            [perturbed.x[:, :static_dim], history.reshape(history.shape[0], -1)], dim=1
        # Completes the multi-line `torch.cat` arguments or elements.
        )
        return perturbed

    @staticmethod
    def _parse_local_feature_encoding(task_cfg: Dict[str, Any]) -> Union[bool, str]:
        # Reads `local_feature_encoding` from the mapping into the `raw_value` with its declared fallback.
        raw_value = task_cfg.get("local_feature_encoding", False)
        if raw_value is False or raw_value is None:
            return False
        if raw_value is True:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(
                "local_feature_encoding 不再支持 true，请使用 false、cgns_based 或 vtk_based"
            )
        if isinstance(raw_value, str):
            # Stores the `normalized` for use by the remaining steps in this flow.
            normalized = raw_value.strip().lower()
            if normalized in {"false", "none", "off", "disabled"}:
                return False
            if normalized in {"cgns_based", "vtk_based"}:
                # Returns `normalized` to the caller for the next stage.
                return normalized
        raise ValueError("local_feature_encoding 仅支持 false、cgns_based 或 vtk_based")

    @abstractmethod
    def process(self, data: Union[BaseData, List[BaseData]], coord_ranges=None) -> Dict:
        """处理场数据并返回统一的中间样本格式。"""
        # Handles the data or control-flow case represented by this statement.
        pass

    @abstractmethod
    def save_to_pt(self, processed_data: Dict, save_path: str) -> None:
        """将处理后的样本保存为 `.pt` 文件。"""
        pass

    # 根据一个已处理样本推断模型输入输出相关参数。
    def infer_model_overrides(
        self, sample: Dict[str, Any], config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """根据一个已处理样本推断模型输入输出相关参数。"""
        raise NotImplementedError(
            # Completes the multi-line `type` arguments or elements.
            f"{type(self).__name__} 尚未实现 infer_model_overrides 接口"
        )

    def build_prediction_adapter(
        self,
        # Continues the surrounding statement or collection definition.
        reader,
        config: Dict,
        is_geometry_varying: bool,
        resolve_reference_file: Callable[[], str],
        # Continues the surrounding statement or collection definition.
        logger=None,
    ):
        """构建 Predictor 使用的预测适配器。"""
        raise NotImplementedError(
            f"{type(self).__name__} 尚未实现 build_prediction_adapter 接口"
        # Completes the multi-line `NotImplementedError` arguments or elements.
        )

    def supports_fast_predict_path(self, is_geometry_varying: bool) -> bool:
        """声明当前处理器是否支持绕过原始场读取的预测快路径。"""
        return False

    def supports_fixed_geometry_temporal_predict_path(self) -> bool:
        """声明当前处理器是否支持固定几何时序预测快路径。"""
        # Returns `False` to the caller for the next stage.
        return False

    def supports_reference_only_predict_path(self) -> bool:
        """声明当前处理器是否支持仅依赖 reference 的稳态预测快路径。"""
        return False

    def requires_fixed_geometry_temporal_reference(self) -> bool:
        """声明固定几何时序预测快路径是否必须提供 reference 文件。"""
        # Returns `False` to the caller for the next stage.
        return False

    def requires_fixed_geometry_predict_writeback_reference(self) -> bool:
        """声明固定几何预测结果写回阶段是否必须依赖共享 reference 模板。"""
        return True

    def build_fast_predict_dataset(
        # Continues the surrounding statement or collection definition.
        self,
        samples_csv: str,
        processed_dir: str,
        config: Dict[str, Any],
        # Completes the multi-line `collection` arguments or elements.
        norm_params: Optional[Dict[str, Any]],
    ) -> Tuple[Dataset, str]:
        """构建预测快路径所需的数据集。"""
        raise NotImplementedError(
            f"{type(self).__name__} 尚未实现 build_fast_predict_dataset 接口"
        # Completes the multi-line `NotImplementedError` arguments or elements.
        )

    def build_fixed_geometry_temporal_predict_dataset(
        self,
        reader: Any,
        # Continues the surrounding statement or collection definition.
        samples_csv: str,
        processed_dir: str,
        config: Dict[str, Any],
        norm_params: Optional[Dict[str, Any]],
        # Continues the surrounding statement or collection definition.
        reference_file: Optional[str],
    ) -> Tuple[Dataset, str]:
        """构建固定几何时序预测所需的数据集。"""
        raise NotImplementedError(
            f"{type(self).__name__} 尚未实现 build_fixed_geometry_temporal_predict_dataset 接口"
        # Completes the multi-line `NotImplementedError` arguments or elements.
        )

    def build_reference_only_predict_dataset(
        self,
        reader: Any,
        # Continues the surrounding statement or collection definition.
        samples_csv: str,
        processed_dir: str,
        config: Dict[str, Any],
        norm_params: Optional[Dict[str, Any]],
        # Continues the surrounding statement or collection definition.
        reference_file: str,
    ) -> Tuple[Dataset, str]:
        """构建仅依赖 reference 和样本参数的稳态预测数据集。"""
        raise NotImplementedError(
            f"{type(self).__name__} 灏氭湭瀹炵幇 build_reference_only_predict_dataset 鎺ュ彛"
        # Completes the multi-line `NotImplementedError` arguments or elements.
        )

    def build_direct_view(
        self,
        sample: Dict[str, Any],
    # Completes the multi-line `collection` arguments or elements.
    ) -> Tuple[Any, Any]:
        """构建 direct 任务视图，默认直接返回完整 `x / y`。"""
        return sample["x"], sample["y"]

    def build_direct_predict_view(
        self,
        # Completes the multi-line `collection` arguments or elements.
        sample: Dict[str, Any],
    ) -> Any:
        """构建 direct 预测视图，默认仅返回输入 x。"""
        return sample["x"]

    def build_autoregressive_view(
        # Continues the surrounding statement or collection definition.
        self,
        sample: Dict[str, Any],
        start_t: int,
        input_length: int,
        # Continues the surrounding statement or collection definition.
        output_length: int,
    ) -> Tuple[Any, Any]:
        """构建 autoregressive 任务视图。"""
        raise NotImplementedError(
            f"{type(self).__name__} 灏氭湭瀹炵幇 build_autoregressive_view 鎺ュ彛"
        # Completes the multi-line `NotImplementedError` arguments or elements.
        )

    def build_autoregressive_predict_seed(
        self,
        sample: Dict[str, Any],
        # Continues the surrounding statement or collection definition.
        input_length: int,
    ) -> Any:
        """构建 autoregressive 预测 seed，默认退回到起始训练窗口。"""
        model_input, _ = self.build_autoregressive_view(
            sample=sample,
            # Completes the multi-line `self.build_autoregressive_view` arguments or elements.
            start_t=0,
            input_length=input_length,
            output_length=1,
        )
        # Returns `model_input` to the caller for the next stage.
        return model_input

    def build_autoregressive_rollout_view(
        self,
        sample: Dict[str, Any],
        # Continues the surrounding statement or collection definition.
        start_t: int,
        input_length: int,
        rollout_steps: int,
    ) -> Tuple[Any, Any]:
        """构建自由滚动的初始输入与连续真实目标。"""
        # Returns `` to the caller for the next stage.
        return self.build_autoregressive_view(
            sample=sample,
            start_t=start_t,
            input_length=input_length,
            # Completes the multi-line `self.build_autoregressive_view` arguments or elements.
            output_length=rollout_steps,
        )

    def predict_autoregressive_rollout(
        self,
        # Continues the surrounding statement or collection definition.
        model: Callable,
        inputs: Any,
        rollout_steps: int,
    ) -> Any:
        """执行默认的 autoregressive 滚动预测，兼容现有张量输入语义。"""
        # Validates `not isinstance(inputs, torch.Tensor)` and raises `TypeError` when the input cannot be processed safely.
        if not isinstance(inputs, torch.Tensor):
            raise TypeError(
                f"{type(self).__name__} 默认 autoregressive rollout 仅支持 torch.Tensor 输入"
            )
        # Validates `inputs.dim() < 3` and raises `ValueError` when the input cannot be processed safely.
        if inputs.dim() < 3:
            raise ValueError(
                f"autoregressive 模式要求输入维度 >= 3，当前为 {inputs.dim()}"
            )

        # Converts the `steps` to `int` so later checks use a consistent type.
        steps = int(rollout_steps)
        if steps <= 0:
            raise ValueError(f"rollout_steps 必须为正整数，当前为 {rollout_steps}")

        batch_predictions = []
        # Stores the `current_input` for use by the remaining steps in this flow.
        current_input = inputs

        for step in range(steps):
            output = model(current_input)
            batch_predictions.append(output)

            # Checks `step < steps - 1` before entering this conditional branch.
            if step < steps - 1:
                current_input = torch.cat(
                    [current_input[:, 1:], output.unsqueeze(1)],
                    dim=1,
                # Completes the multi-line `torch.cat` arguments or elements.
                )

        if len(batch_predictions) == 1:
            return batch_predictions[0]
        return torch.cat(batch_predictions, dim=1)

    # 在 Evaluator 聚合前，对单批次输出做轻量整理。
    def prepare_eval_payload(
        self,
        inputs: Any,
        outputs: Any,
        # Continues the surrounding statement or collection definition.
        targets: Any,
    ) -> Tuple[Any, Any]:
        """在 Evaluator 聚合前，对单批次输出做轻量整理。"""
        return outputs, targets

    def build_processing_context(self, **kwargs) -> Dict[str, Any]:
        """在样本处理开始前构建处理器侧上下文。"""
        # Returns `self._build_local_feature_processing_context(**kwargs)` to the caller for the next stage.
        return self._build_local_feature_processing_context(**kwargs)

    def build_criterion(self, loss_name: str):
        """Build the processor-appropriate criterion."""
        criterion, resolved_loss_name = build_basic_criterion(loss_name)
        self._resolved_loss_name = resolved_loss_name
        # Returns `criterion, resolved_loss_name` to the caller for the next stage.
        return criterion, resolved_loss_name

    def build_loss_weight(
        self,
        inputs: Any,
        # Continues the surrounding statement or collection definition.
        targets: Any,
        outputs: Any = None,
    ) -> Optional[torch.Tensor]:
        """构建损失权重，默认不启用额外加权。"""
        return None

    # Defines `_is_critical_region_weighting_enabled` for  is critical region weighting enabled logic.
    def _is_critical_region_weighting_enabled(self) -> bool:
        training_cfg = self.config.get("training", {})
        return bool(training_cfg.get("critical_region_weighting", False))

    def _resolve_loss_name(self, criterion: Callable) -> str:
        # Stores the `resolved` for use by the remaining steps in this flow.
        resolved = getattr(criterion, "_tfai_loss_name", None) or self._resolved_loss_name
        if resolved:
            return str(resolved).lower()
        return str(self.config.get("training", {}).get("loss_function", "mse")).lower()

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _compute_quantile_threshold(
        values: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        # Continues the surrounding statement or collection definition.
        quantile: float = 0.90,
    ) -> torch.Tensor:
        if values.ndim < 2:
            raise ValueError(f"values 维度至少应为 2，当前为 {values.ndim}")

        # Stores the `flattened_values` for use by the remaining steps in this flow.
        flattened_values = values.reshape(values.shape[0], -1)
        flattened_mask = None
        if valid_mask is not None:
            flattened_mask = valid_mask.reshape(valid_mask.shape[0], -1) > 0

        # Initializes the `thresholds` as a collection for values from this processing stage.
        thresholds: List[torch.Tensor] = []
        for sample_index in range(flattened_values.shape[0]):
            sample_values = flattened_values[sample_index]
            if flattened_mask is not None:
                # Selects the required item from `sample_values` as the `sample_values`.
                sample_values = sample_values[flattened_mask[sample_index]]
            sample_values = sample_values[torch.isfinite(sample_values)]

            if sample_values.numel() == 0:
                thresholds.append(
                    # Completes the multi-line `torch.tensor` arguments or elements.
                    torch.tensor(0.0, device=values.device, dtype=values.dtype)
                )
                continue

            thresholds.append(
                # Completes the multi-line `torch.quantile(sample_values, q=float(quantile)).to` arguments or elements.
                torch.quantile(
                    sample_values,
                    q=float(quantile),
                ).to(device=values.device, dtype=values.dtype)
            # Completes the multi-line `thresholds.append` arguments or elements.
            )

        return torch.stack(thresholds, dim=0)

    @classmethod
    def _build_quantile_score(
        # Continues the surrounding statement or collection definition.
        cls,
        values: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        quantile: float = 0.90,
        # Continues the surrounding statement or collection definition.
        sharpness: float = 10.0,
    ) -> torch.Tensor:
        magnitude = torch.abs(values.float())
        thresholds = cls._compute_quantile_threshold(
            # Completes the multi-line `cls._compute_quantile_threshold` arguments or elements.
            values=magnitude,
            valid_mask=valid_mask,
            quantile=quantile,
        )
        # Stores the `thresholds` for use by the remaining steps in this flow.
        thresholds = thresholds.reshape(-1, *([1] * (magnitude.ndim - 1)))
        scale = torch.clamp(thresholds.abs(), min=1.0e-6)
        score = torch.sigmoid(sharpness * (magnitude / scale - 1.0))
        zero_threshold_mask = thresholds.abs() <= 1.0e-6
        # Checks `zero_threshold_mask.any()` before entering this conditional branch.
        if zero_threshold_mask.any():
            zero_threshold_mask = zero_threshold_mask.expand_as(score)
            score = torch.where(zero_threshold_mask, (magnitude > 0).to(score.dtype), score)
        if valid_mask is not None:
            # Derives the `score` from the current values for downstream processing.
            score = score * (valid_mask > 0).to(score.dtype)
        return score

    @staticmethod
    def _normalize_shared_weight_map(
        # Continues the surrounding statement or collection definition.
        weight_map: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normalized = weight_map.float()
        # Checks `valid_mask is not None` before entering this conditional branch.
        if valid_mask is not None:
            mask = (valid_mask > 0).to(dtype=normalized.dtype, device=normalized.device)
            normalized = normalized * mask
            mask_sum = mask.reshape(mask.shape[0], -1).sum(dim=1, keepdim=True)
            # Stores the `weight_sum` for use by the remaining steps in this flow.
            weight_sum = normalized.reshape(normalized.shape[0], -1).sum(dim=1, keepdim=True)
            safe_mean = torch.where(mask_sum > 0, weight_sum / torch.clamp(mask_sum, min=1.0), torch.ones_like(mask_sum))
        else:
            safe_mean = normalized.reshape(normalized.shape[0], -1).mean(dim=1, keepdim=True)
            # Stores the `safe_mean` for use by the remaining steps in this flow.
            safe_mean = torch.clamp(safe_mean, min=1.0e-6)

        safe_mean = safe_mean.reshape(-1, *([1] * (normalized.ndim - 1)))
        normalized = normalized / torch.clamp(safe_mean, min=1.0e-6)
        normalized = torch.clamp(normalized, min=0.5, max=5.0)
        # Checks `valid_mask is not None` before entering this conditional branch.
        if valid_mask is not None:
            normalized = normalized * (valid_mask > 0).to(dtype=normalized.dtype, device=normalized.device)
        return normalized

    @classmethod
    # Defines `_compose_shared_weight_map` for  compose shared weight map logic.
    def _compose_shared_weight_map(
        cls,
        value_scores: List[torch.Tensor],
        gradient_scores: List[torch.Tensor],
        # Continues the surrounding statement or collection definition.
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not value_scores or not gradient_scores:
            raise ValueError("共享权重图构建失败：缺少高值项或高梯度项")

        # Stores the `value_map` for use by the remaining steps in this flow.
        value_map = torch.stack(value_scores, dim=0).amax(dim=0)
        gradient_map = torch.stack(gradient_scores, dim=0).amax(dim=0)
        shared_weight = 1.0 + 2.0 * value_map + 2.0 * gradient_map
        return cls._normalize_shared_weight_map(shared_weight, valid_mask=valid_mask)

    # Defines `_compute_weighted_loss` for  compute weighted loss logic.
    def _compute_weighted_loss(
        self,
        loss_name: str,
        outputs: torch.Tensor,
        # Continues the surrounding statement or collection definition.
        targets: torch.Tensor,
        weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        normalized_name = str(loss_name).lower()
        # Returns early when `normalized_name == "mae"` makes the remaining work inapplicable.
        if normalized_name == "mae":
            return weighted_mae(outputs, targets, weight)
        if normalized_name == "huber":
            return weighted_huber(outputs, targets, weight, delta=1.0)
        # Returns `weighted_mse(outputs, targets, weight)` to the caller for the next stage.
        return weighted_mse(outputs, targets, weight)

    def supports_partition_training(self) -> bool:
        """声明当前处理器是否支持 partition 子图训练。"""
        return False

    def requires_batch_preprocess(self) -> bool:
        """Return whether preprocessing must consume the full batch at once."""
        # Returns `False` to the caller for the next stage.
        return False

    def process_single_sample(
        self,
        data: BaseData,
        # Completes the multi-line `collection` arguments or elements.
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """处理单个样本，默认直接转调 `process()`。"""
        return self.process(data)

    def process_predict_sample(
        # Continues the surrounding statement or collection definition.
        self,
        data: BaseData,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """澶勭悊棰勬祴闃舵鍗曚釜鏍锋湰锛岄粯璁ゅ鐢ㄥ父瑙勫崟鏍锋湰澶勭悊銆?"""
        # Returns `self.process_single_sample(data, context=context)` to the caller for the next stage.
        return self.process_single_sample(data, context=context)

    def process_loaded_samples(
        self,
        data_items: List[BaseData],
        # Continues the surrounding statement or collection definition.
        stage: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """批量处理已加载样本，默认逐个处理。"""
        context = context or {}
        # Returns `` to the caller for the next stage.
        return [self.process_single_sample(data, context=context) for data in data_items]

    def save_processing_state(
        self,
        stage: str,
        # Continues the surrounding statement or collection definition.
        processed_dir: str,
        processed_paths: List[str],
    ) -> None:
        """在样本处理结束后持久化处理器状态。"""
        del processed_paths
        # Checks `stage == "train" and self.local_feature_schema` before entering this conditional branch.
        if stage == "train" and self.local_feature_schema:
            os.makedirs(processed_dir, exist_ok=True)
            atomic_torch_save(self.export_state(), os.path.join(processed_dir, "processor_state.pt"))
        return None

    # 对处理器输出做反归一化。
    def denormalize_predictions(
        self,
        predictions: Any,
        norm_params: Optional[Dict[str, Any]],
    # Continues the surrounding statement or collection definition.
    ) -> Any:
        """对处理器输出做反归一化。"""
        return predictions

    def _resolve_eval_norm_params(
        self,
        # Completes the multi-line `collection` arguments or elements.
        config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        runtime_config = config or self.config or {}
        if isinstance(runtime_config, dict):
            # Reads `__eval_norm_params__` from the mapping into the `runtime_norm_params`.
            runtime_norm_params = runtime_config.get("__eval_norm_params__")
            if runtime_norm_params is not None:
                return runtime_norm_params
            general_cfg = runtime_config.get("general", {})
        # Continues the surrounding statement or collection definition.
        else:
            general_cfg = {}

        processed_dir = str(general_cfg.get("train_processed_dir") or "").strip()
        if not processed_dir:
            # Returns `None` to the caller for the next stage.
            return None

        norm_path = os.path.join(processed_dir, "norm_params.pt")
        if not os.path.exists(norm_path):
            return None
        # Returns `load_torch(norm_path, map_location="cpu", weights_only=False)` to the caller for the next stage.
        return load_torch(norm_path, map_location="cpu", weights_only=False)

    def _denormalize_sample_output_map(
        self,
        sample_output_map: Dict[int, np.ndarray],
        # Completes the multi-line `collection` arguments or elements.
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[int, np.ndarray]:
        if not sample_output_map:
            return {}
        # Stores the normalization statistics for use by the remaining steps in this flow.
        norm_params = self._resolve_eval_norm_params(config)
        denormalized = self.denormalize_predictions(sample_output_map, norm_params)
        if isinstance(denormalized, dict):
            return denormalized
        # Returns `sample_output_map` to the caller for the next stage.
        return sample_output_map

    def load_checkpoint_artifacts(self, checkpoint: Dict[str, Any]) -> None:
        """从 checkpoint 恢复处理器侧状态。"""
        processor_state = checkpoint.get("processor_state")
        if processor_state and hasattr(self, "load_state"):
            # Calls `self.load_state` to perform the operation required at this stage.
            self.load_state(processor_state)

    def compute_train_loss(
        self,
        criterion: Callable,
        # Continues the surrounding statement or collection definition.
        outputs: Any,
        targets: Any,
        inputs: Any,
    ) -> torch.Tensor:
        """计算训练阶段损失，默认直接调用 criterion(outputs, targets)。"""
        # Stores the `override_mask` for use by the remaining steps in this flow.
        override_mask = getattr(inputs, "boundary_state_override_mask", None)
        owner_mask = getattr(inputs, "owner_mask", None)
        if override_mask is not None or owner_mask is not None:
            if not isinstance(outputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
                # Raises `TypeError` to report why this path cannot continue.
                raise TypeError("boundary_state_override 的损失掩码仅支持张量输出")
            if override_mask is None:
                valid_mask = torch.ones_like(outputs, dtype=outputs.dtype)
            else:
                # Validates `` and raises `ValueError` when the input cannot be processed safely.
                if not isinstance(override_mask, torch.Tensor) or override_mask.ndim != 2:
                    raise ValueError("boundary_state_override_mask 必须为 [N, C]")
                valid_mask = (~override_mask).to(device=outputs.device, dtype=outputs.dtype)
            while valid_mask.ndim < outputs.ndim:
                # Stores the valid-item mask for use by the remaining steps in this flow.
                valid_mask = valid_mask.unsqueeze(1)
            try:
                valid_mask = valid_mask.expand_as(outputs)
            except RuntimeError as exc:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError("boundary_state_override_mask 与损失输出形状不一致")
            if owner_mask is not None:
                if not isinstance(owner_mask, torch.Tensor) or owner_mask.ndim != 1:
                    raise ValueError("owner_mask 必须为 [N]")
                # Converts the `owner_node_count` to `int` so later checks use a consistent type.
                owner_node_count = int(owner_mask.numel())
                if outputs.shape[0] == owner_node_count:
                    owner_node_dim = 0
                elif outputs.ndim == 3 and outputs.shape[1] == owner_node_count:
                    # Stores the `owner_node_dim` for use by the remaining steps in this flow.
                    owner_node_dim = 1
                else:
                    raise ValueError("owner_mask 与损失输出的节点维度不一致")
                owner_shape = [1] * outputs.ndim
                # Stores the `owner_shape[owner_node_dim]` for use by the remaining steps in this flow.
                owner_shape[owner_node_dim] = owner_node_count
                owner_weight = owner_mask.to(device=outputs.device, dtype=outputs.dtype).reshape(owner_shape)
                try:
                    valid_mask = valid_mask * owner_weight.expand_as(outputs)
                # Continues the surrounding statement or collection definition.
                except RuntimeError as exc:
                    raise ValueError("owner_mask 与损失输出的节点维度不一致") from exc
            loss_name = self._resolve_loss_name(criterion)
            if loss_name == "mae":
                # Stores the `loss_map` for use by the remaining steps in this flow.
                loss_map = torch.abs(outputs - targets)
            elif loss_name == "huber":
                error = torch.abs(outputs - targets)
                quadratic = torch.clamp(error, max=1.0)
                # Derives the `loss_map` from the current values for downstream processing.
                loss_map = 0.5 * quadratic ** 2 + (error - quadratic)
            else:
                loss_map = (outputs - targets) ** 2
            if self._is_critical_region_weighting_enabled():
                # Scopes resource use so the context manager can release it on exit.
                with torch.no_grad():
                    weight = self.build_loss_weight(
                        inputs=inputs,
                        targets=targets,
                        # Completes the multi-line `self.build_loss_weight` arguments or elements.
                        outputs=outputs,
                    )
                if weight is not None:
                    weight = weight.to(device=outputs.device, dtype=outputs.dtype)
                    # Repeats the current step while `weight.ndim < outputs.ndim` remains true.
                    while weight.ndim < outputs.ndim:
                        weight = weight.unsqueeze(1)
                    valid_mask = valid_mask * weight
            denominator = valid_mask.sum()
            # Validates `denominator <= 0` and raises `ValueError` when the input cannot be processed safely.
            if denominator <= 0:
                raise ValueError("boundary_state_override 排除了全部监督目标，无法计算损失")
            return (loss_map * valid_mask).sum() / denominator

        if self._is_critical_region_weighting_enabled():
            # Validates `` and raises `TypeError` when the input cannot be processed safely.
            if not isinstance(outputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
                raise TypeError("关键区域加权当前仅支持张量形式的场损失")
            # 权重图仅依赖样本统计，不需要参与反向传播。
            with torch.no_grad():
                weight = self.build_loss_weight(inputs=inputs, targets=targets, outputs=outputs)
            # Returns `` to the caller for the next stage.
            return self._compute_weighted_loss(
                loss_name=self._resolve_loss_name(criterion),
                outputs=outputs,
                targets=targets,
                # Completes the multi-line `self._compute_weighted_loss` arguments or elements.
                weight=weight,
            )
        return criterion(outputs, targets)

    def compute_eval_loss(
        # Continues the surrounding statement or collection definition.
        self,
        criterion: Callable,
        outputs: Any,
        targets: Any,
        # Continues the surrounding statement or collection definition.
        inputs: Any,
    ) -> torch.Tensor:
        """计算验证阶段损失，默认与训练阶段保持一致。"""
        return self.compute_train_loss(
            criterion=criterion,
            # Completes the multi-line `self.compute_train_loss` arguments or elements.
            outputs=outputs,
            targets=targets,
            inputs=inputs,
        )

    # 收集单个 batch 的评估载荷。
    def collect_eval_batch(
        self,
        inputs: Any,
        outputs: Any,
        # Continues the surrounding statement or collection definition.
        targets: Any,
    ) -> Dict[str, Any]:
        """收集单个 batch 的评估载荷。"""
        payload = {
            "predictions": self._to_numpy(outputs),
            # Completes the multi-line `self._to_numpy` arguments or elements.
            "targets": self._to_numpy(targets),
        }
        override_mask = getattr(inputs, "boundary_state_override_mask", None)
        if override_mask is not None:
            # Validates `` and raises `TypeError` when the input cannot be processed safely.
            if not isinstance(outputs, torch.Tensor) or not isinstance(override_mask, torch.Tensor):
                raise TypeError("boundary_state_override 的评估掩码仅支持张量输出")
            metric_mask = (~override_mask).to(device=outputs.device, dtype=torch.bool)
            while metric_mask.ndim < outputs.ndim:
                # Stores the `metric_mask` for use by the remaining steps in this flow.
                metric_mask = metric_mask.unsqueeze(1)
            try:
                payload["metric_mask"] = self._to_numpy(metric_mask.expand_as(outputs))
            except RuntimeError as exc:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError("boundary_state_override_mask 与评估输出形状不一致") from exc
        return payload

    @staticmethod
    def _build_sample_output_map(
        # Continues the surrounding statement or collection definition.
        sample_ids: Optional[List[int]],
        values: Any,
    ) -> Dict[int, np.ndarray]:
        if not sample_ids:
            # Returns `{}` to the caller for the next stage.
            return {}

        array = np.asarray(values)
        if array.ndim == 0 or len(array) != len(sample_ids):
            return {}

        # Returns `` to the caller for the next stage.
        return {
            int(sample_id): np.asarray(array[index])
            for index, sample_id in enumerate(sample_ids)
        }

    # Continues the surrounding statement or collection definition.
    @classmethod
    def _merge_sample_output_maps(
        cls,
        batch_payloads: List[Dict[str, Any]],
        # Continues the surrounding statement or collection definition.
        payload_key: str,
        fallback_key: str,
    ) -> Dict[int, np.ndarray]:
        merged: Dict[int, np.ndarray] = {}
        # Visits each item in `batch_payloads` and applies the current processing step.
        for payload in batch_payloads:
            if payload_key in payload:
                merged.update(payload.get(payload_key, {}))
                continue

            # Reads `sample_ids` from the mapping into the sample identifiers.
            sample_ids = payload.get("sample_ids")
            merged.update(
                cls._build_sample_output_map(
                    sample_ids=sample_ids,
                    # Completes the multi-line `payload.get` arguments or elements.
                    values=payload.get(fallback_key),
                )
            )
        return merged

    # 收集单个预测 batch 的输出。
    def collect_prediction_batch(
        self,
        inputs: Any,
        outputs: Any,
    # Continues the surrounding statement or collection definition.
    ) -> Any:
        """收集单个预测 batch 的输出。"""
        return self._to_numpy(outputs)

    def merge_prediction_batches(
        self,
        # Continues the surrounding statement or collection definition.
        batch_outputs: List[Any],
    ) -> Any:
        """合并预测 batch 输出，默认按 batch 维拼接。"""
        if not batch_outputs:
            return np.array([])
        # Builds the `arrays` using the comprehension's filter and mapping rules.
        arrays = [np.asarray(item) for item in batch_outputs]
        return np.concatenate(arrays, axis=0)

    def merge_eval_batches(
        self,
        # Completes the multi-line `collection` arguments or elements.
        batch_payloads: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """合并评估 batch，默认按 batch 维直接拼接。"""
        predictions = [payload["predictions"] for payload in batch_payloads]
        # Builds the target tensors using the comprehension's filter and mapping rules.
        targets = [payload["targets"] for payload in batch_payloads]

        metric_masks = [payload.get("metric_mask") for payload in batch_payloads]
        if predictions:
            merged_predictions = np.concatenate(predictions, axis=0)
            # Stores the `merged_targets` for use by the remaining steps in this flow.
            merged_targets = np.concatenate(targets, axis=0)
            merged_metric_mask = (
                np.concatenate(metric_masks, axis=0)
                if all(mask is not None for mask in metric_masks)
                # Continues the surrounding statement or collection definition.
                else None
            )
        else:
            merged_predictions = np.array([])
            # Stores the `merged_targets` for use by the remaining steps in this flow.
            merged_targets = np.array([])
            merged_metric_mask = None

        if merged_metric_mask is not None:
            if merged_metric_mask.shape != merged_predictions.shape:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError("评估掩码与合并后的预测形状不一致")
            overall_predictions = np.asarray(merged_predictions)[merged_metric_mask]
            overall_targets = np.asarray(merged_targets)[merged_metric_mask]
        else:
            # Stores the `overall_predictions` for use by the remaining steps in this flow.
            overall_predictions = np.asarray(merged_predictions).reshape(-1)
            overall_targets = np.asarray(merged_targets).reshape(-1)

        return {
            "raw_predictions": merged_predictions,
            # Completes the multi-line `collection` arguments or elements.
            "raw_targets": merged_targets,
            "metric_mask": merged_metric_mask,
            "overall_predictions": overall_predictions,
            "overall_targets": overall_targets,
            # Completes the multi-line `collection` arguments or elements.
            "scatter_by_var": {},
            "sample_predictions": self._merge_sample_output_maps(
                batch_payloads=batch_payloads,
                payload_key="sample_predictions",
                # Completes the multi-line `self._merge_sample_output_maps` arguments or elements.
                fallback_key="predictions",
            ),
            "sample_targets": self._merge_sample_output_maps(
                batch_payloads=batch_payloads,
                # Completes the multi-line `self._merge_sample_output_maps` arguments or elements.
                payload_key="sample_targets",
                fallback_key="targets",
            ),
        }

    # 根据训练样本统计归一化参数。
    def compute_norm_params(
        self,
        train_samples: List[Dict[str, Any]],
        norm_type: str,
        # Continues the surrounding statement or collection definition.
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """根据训练样本统计归一化参数。"""
        raise NotImplementedError(
            f"{type(self).__name__} 尚未实现 compute_norm_params 接口"
        # Completes the multi-line `NotImplementedError` arguments or elements.
        )

    def compute_norm_params_from_paths(
        self,
        train_processed_paths: List[str],
        # Continues the surrounding statement or collection definition.
        norm_type: str,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Default path-based fallback for processors without streaming statistics."""
        train_samples = [
            # Completes the multi-line `load_torch` arguments or elements.
            load_torch(processed_path, weights_only=False)
            for processed_path in train_processed_paths
        ]
        return self.compute_norm_params(
            # Completes the multi-line `self.compute_norm_params` arguments or elements.
            train_samples=train_samples,
            norm_type=norm_type,
            save_path=save_path,
        )

    # 对单个已处理样本做归一化。
    def normalize_processed_sample(
        self,
        sample: Dict[str, Any],
        norm_params: Dict[str, Any],
        # Continues the surrounding statement or collection definition.
        norm_type: str,
        normalize_y: bool = True,
    ) -> Dict[str, Any]:
        """对单个已处理样本做归一化。"""
        raise NotImplementedError(
            # Completes the multi-line `type` arguments or elements.
            f"{type(self).__name__} 尚未实现 normalize_processed_sample 接口"
        )

    def normalize_predict_sample(
        self,
        # Completes the multi-line `collection` arguments or elements.
        sample: Dict[str, Any],
        norm_params: Dict[str, Any],
        norm_type: str,
    ) -> Dict[str, Any]:
        """预测阶段只归一化输入，不归一化目标。"""
        # Returns `` to the caller for the next stage.
        return self.normalize_processed_sample(
            sample=sample,
            norm_params=norm_params,
            norm_type=norm_type,
            # Completes the multi-line `self.normalize_processed_sample` arguments or elements.
            normalize_y=False,
        )

    def normalize_output_tensor(
        self,
        # Continues the surrounding statement or collection definition.
        outputs: Any,
        norm_params: Optional[Dict[str, Any]],
        inputs: Any = None,
    ) -> Any:
        """对模型输出张量做归一化，默认保持原样。"""
        # Returns `outputs` to the caller for the next stage.
        return outputs

    def denormalize_output_tensor(
        self,
        outputs: Any,
        # Completes the multi-line `collection` arguments or elements.
        norm_params: Optional[Dict[str, Any]],
        inputs: Any = None,
    ) -> Any:
        """对模型输出张量做反归一化，默认保持原样。"""
        return outputs

    # Defines `export_state` for export state logic.
    def export_state(self) -> Dict[str, Any]:
        return {
            "processor_type": type(self).__name__,
            "local_feature_encoding": self.local_feature_encoding,
            # Completes the multi-line `list` arguments or elements.
            "field_bc": list(self.field_bc),
            "local_feature_schema": self.local_feature_schema,
            "boundary_state_override_schema": self.boundary_state_override_schema,
        }

    # Defines `load_state` for load state logic.
    def load_state(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self.local_feature_schema = state.get("local_feature_schema")
        # Stores the `boundary_state_override_schema` on the instance so other methods can reuse it.
        self.boundary_state_override_schema = state.get("boundary_state_override_schema")

    def _build_local_feature_processing_context(self, **kwargs) -> Dict[str, Any]:
        if not self.local_feature_encoding:
            return {}

        # Checks `self.local_feature_encoding == "cgns_based"` before entering this conditional branch.
        if self.local_feature_encoding == "cgns_based":
            self.local_feature_schema = {
                "local_feature_encoding": "cgns_based",
                "feature_bc_names": [
                    # Completes the multi-line `collection` arguments or elements.
                    "bc_is_boundary",
                    "bc_type_wall",
                    "bc_type_inlet",
                    "bc_type_outlet",
                    # Completes the multi-line `collection` arguments or elements.
                    "bc_type_symmetry",
                    "bc_type_interface",
                    "bc_type_other",
                ],
                # Completes the multi-line `list` arguments or elements.
                "feature_bc_offsets": list(range(7)),
                "categorical_bc_offsets": list(range(7)),
                "continuous_bc_offsets": [],
            }
            # Returns `{"local_feature_schema": self.local_feature_schema}` to the caller for the next stage.
            return {"local_feature_schema": self.local_feature_schema}

        if self.local_feature_encoding != "vtk_based":
            raise ValueError("local_feature_encoding 仅支持 false、cgns_based 或 vtk_based")

        if self.local_feature_schema is not None:
            # Returns `{"local_feature_schema": self.local_feature_schema}` to the caller for the next stage.
            return {"local_feature_schema": self.local_feature_schema}

        stage = kwargs.get("stage")
        if stage != "train":
            raise ValueError("vtk_based 需要先从训练阶段加载或拟合 field_bc schema")

        # Reads `sample_loader` from the mapping into the `sample_loader`.
        sample_loader = kwargs.get("sample_loader")
        local_feature_sample_loader = kwargs.get("local_feature_sample_loader") or sample_loader
        file_paths = kwargs.get("file_paths") or []
        sample_ids = kwargs.get("sample_ids") or []
        # Reads `logger` from the mapping into the run logger.
        logger = kwargs.get("logger")
        if local_feature_sample_loader is None:
            raise ValueError("vtk_based 拟合 field_bc schema 需要 sample_loader")

        if logger is not None:
            # Records the current execution state to make progress and failures traceable.
            logger.info(
                "开始拟合 vtk_based 局部边界条件 schema: samples=%s field_bc=%s",
                len(file_paths),
                self.field_bc,
            # Completes the multi-line `logger.info` arguments or elements.
            )

        def iter_data_dicts():
            for index, (file_path, sample_id) in enumerate(zip(file_paths, sample_ids), start=1):
                if logger is not None:
                    # Records the current execution state to make progress and failures traceable.
                    logger.info(
                        "拟合局部边界条件 schema: %s/%s sample_id=%s",
                        index,
                        len(file_paths),
                        # Completes the multi-line `logger.info` arguments or elements.
                        sample_id,
                    )
                data = local_feature_sample_loader(file_path, sample_id)
                try:
                    # Handles the data or control-flow case represented by this statement.
                    yield data.get_data()
                finally:
                    del data

        self.local_feature_schema = fit_field_bc_schema(
            # Completes the multi-line `iter_data_dicts` arguments or elements.
            data_items=iter_data_dicts(),
            field_bc_names=self.field_bc,
        )
        if logger is not None:
            # Records the current execution state to make progress and failures traceable.
            logger.info(
                "vtk_based 局部边界条件 schema 拟合完成: feature_count=%s feature_names=%s",
                len(self.local_feature_schema.get("feature_bc_names", [])),
                self.local_feature_schema.get("feature_bc_names", []),
            # Completes the multi-line `logger.info` arguments or elements.
            )
        return {"local_feature_schema": self.local_feature_schema}

    def _build_local_feature_matrix(
        self,
        # Completes the multi-line `collection` arguments or elements.
        data_dict: Dict[str, Any],
        target_location: str,
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        if not self.local_feature_encoding:
            # Returns `np.zeros((0, 0), dtype=np.float32), [], {}` to the caller for the next stage.
            return np.zeros((0, 0), dtype=np.float32), [], {}

        if self.local_feature_encoding == "cgns_based":
            features, feature_names, source = build_boundary_feature_matrix(
                data_dict=data_dict,
                # Completes the multi-line `build_boundary_feature_matrix` arguments or elements.
                target_location=target_location,
            )
            if source == "none":
                raise ValueError(
                    # Completes the multi-line `ValueError` arguments or elements.
                    "local_feature_encoding=cgns_based 时未在样本中解析到可用边界条件"
                )
            schema = {
                "local_feature_encoding": "cgns_based",
                # Completes the multi-line `collection` arguments or elements.
                "local_feature_source": source,
                "feature_bc_names": feature_names,
                "feature_bc_offsets": list(range(len(feature_names))),
                "categorical_bc_offsets": list(range(len(feature_names))),
                # Completes the multi-line `collection` arguments or elements.
                "continuous_bc_offsets": [],
            }
            return features, feature_names, schema

        if self.local_feature_encoding == "vtk_based":
            # Validates `self.local_feature_schema is None` and raises `ValueError` when the input cannot be processed safely.
            if self.local_feature_schema is None:
                raise ValueError("vtk_based 缺少训练阶段拟合的 field_bc schema")
            features, feature_names, schema = build_field_bc_feature_matrix(
                data_dict=data_dict,
                # Completes the multi-line `build_field_bc_feature_matrix` arguments or elements.
                target_location=target_location,
                schema=self.local_feature_schema,
            )
            return features, feature_names, schema

        # Raises `ValueError` to report why this path cannot continue.
        raise ValueError("local_feature_encoding 仅支持 false、cgns_based 或 vtk_based")

    @staticmethod
    def _offsets_to_indices(offsets: List[int], feature_start: int) -> List[int]:
        return [feature_start + int(offset) for offset in offsets]

    # Defines `_build_local_feature_meta_payload` for  build local feature meta payload logic.
    def _build_local_feature_meta_payload(
        self,
        schema: Dict[str, Any],
        feature_start: int,
        # Continues the surrounding statement or collection definition.
        feature_location: Optional[str],
    ) -> Dict[str, Any]:
        if not schema:
            return {}

        # Stores the `feature_names` for use by the remaining steps in this flow.
        feature_names = list(schema.get("feature_bc_names", []))
        feature_offsets = list(schema.get("feature_bc_offsets", range(len(feature_names))))
        categorical_offsets = list(schema.get("categorical_bc_offsets", []))
        continuous_offsets = list(schema.get("continuous_bc_offsets", []))
        # Stores the `feature_indices` for use by the remaining steps in this flow.
        feature_indices = self._offsets_to_indices(feature_offsets, feature_start)
        categorical_indices = self._offsets_to_indices(categorical_offsets, feature_start)
        continuous_indices = self._offsets_to_indices(continuous_offsets, feature_start)

        fields_schema = []
        # Visits each item in `schema.get("fields_bc_schema", [])` and applies the current processing step.
        for field_schema in schema.get("fields_bc_schema", []):
            item = dict(field_schema)
            item["feature_bc_indices"] = self._offsets_to_indices(
                list(item.get("feature_bc_offsets", [])),
                # Completes the multi-line `self._offsets_to_indices` arguments or elements.
                feature_start,
            )
            item["categorical_bc_indices"] = self._offsets_to_indices(
                list(item.get("categorical_bc_offsets", [])),
                # Completes the multi-line `self._offsets_to_indices` arguments or elements.
                feature_start,
            )
            item["continuous_bc_indices"] = self._offsets_to_indices(
                list(item.get("continuous_bc_offsets", [])),
                # Completes the multi-line `self._offsets_to_indices` arguments or elements.
                feature_start,
            )
            fields_schema.append(item)

        return {
            # Completes the multi-line `collection` arguments or elements.
            "feature_names": feature_names,
            "feature_indices": feature_indices,
            "feature_source": schema.get("local_feature_source")
            or schema.get("local_feature_encoding")
            # Completes the multi-line `collection` arguments or elements.
            or "none",
            "feature_location": feature_location,
            "feature_bc_names": feature_names,
            "feature_bc_indices": feature_indices,
            # Completes the multi-line `collection` arguments or elements.
            "categorical_bc_indices": categorical_indices,
            "continuous_bc_indices": continuous_indices,
            "field_bc": list(schema.get("field_bc", self.field_bc)),
            "field_bc_location": schema.get("field_bc_location"),
            # Completes the multi-line `collection` arguments or elements.
            "fields_bc_schema": fields_schema,
        }

    @staticmethod
    def _extract_sample_id_from_data(data: BaseData) -> int:
        """从 BaseData 中提取样本编号。"""
        # Stores the `sample_map` for use by the remaining steps in this flow.
        sample_map = data.get_sample_map() or {}
        sample_id = sample_map.get("sample_order", sample_map.get("sample_id", 0))
        try:
            return int(sample_id)
        # Completes the multi-line `collection` arguments or elements.
        except (TypeError, ValueError):
            return 0

    def _build_processed_sample_meta(
        self,
        # Continues the surrounding statement or collection definition.
        data: BaseData,
        y: Any,
        time_axis: Optional[int] = None,
    ) -> Dict[str, Any]:
        """构建统一的 processed sample 元信息。"""
        # Stores the `total_time_steps` for use by the remaining steps in this flow.
        total_time_steps = None
        if data.get_mode() == "temporal":
            total_time_steps = self._infer_total_time_steps(y=y, time_axis=time_axis)
        sample_map = data.get_sample_map() or {}
        # Reads `task` from the mapping into the task settings with its declared fallback.
        task_cfg = self.config.get("task", {})
        source_geometry_regime = data.get_geometry_regime()
        configured_geometry_regime = str(
            task_cfg.get("geometry_regime") or ""
        # Completes the multi-line `str(task_cfg.get('geometry_regime') or '').strip().lower` arguments or elements.
        ).strip().lower()
        geometry_regime = source_geometry_regime
        if configured_geometry_regime == "displaced_static_topology":
            # 预测仅提供参考网格时，源文件本身不会携带完整时序运动元信息。
            geometry_regime = configured_geometry_regime

        # Stores the `displacement_effective_variables` for use by the remaining steps in this flow.
        displacement_effective_variables = list(
            sample_map.get("displacement_effective_variables")
            or task_cfg.get("displacement_effective_variables")
            or []
        # Completes the multi-line `list` arguments or elements.
        )
        return {
            "processor_type": type(self).__name__,
            "sample_id": self._extract_sample_id_from_data(data),
            # Completes the multi-line `data.get_mode` arguments or elements.
            "is_temporal": data.get_mode() == "temporal",
            "total_time_steps": total_time_steps,
            "time_axis": int(time_axis) if time_axis is not None else None,
            "geometry_regime": geometry_regime,
            # Completes the multi-line `collection` arguments or elements.
            "target_source": "source",
            "displacement_logical_variable": sample_map.get("displacement_logical_name")
            or sample_map.get("displacement_logical_variable")
            or task_cfg.get("displacement_logical_variable"),
            # Completes the multi-line `collection` arguments or elements.
            "displacement_effective_variables": displacement_effective_variables,
            "displacement_source": sample_map.get("displacement_source")
            or task_cfg.get("displacement_source"),
            "displacement_coordinate_convention": (
                # Completes the multi-line `collection` arguments or elements.
                "reference_absolute"
                if geometry_regime == "displaced_static_topology"
                else None
            ),
        # Completes the multi-line `collection` arguments or elements.
        }

    def _with_local_feature_meta(
        self,
        meta: Dict[str, Any],
        # Continues the surrounding statement or collection definition.
        *,
        feature_names: Optional[List[str]] = None,
        feature_indices: Optional[List[int]] = None,
        feature_source: str = "none",
        # Continues the surrounding statement or collection definition.
        feature_location: Optional[str] = None,
        feature_bc_names: Optional[List[str]] = None,
        feature_bc_indices: Optional[List[int]] = None,
        categorical_bc_indices: Optional[List[int]] = None,
        # Continues the surrounding statement or collection definition.
        continuous_bc_indices: Optional[List[int]] = None,
        field_bc: Optional[List[str]] = None,
        field_bc_location: Optional[str] = None,
        fields_bc_schema: Optional[List[Dict[str, Any]]] = None,
    # Completes the multi-line `collection` arguments or elements.
    ) -> Dict[str, Any]:
        """补充局部边界条件特征元信息。"""
        merged = dict(meta or {})
        enabled = bool(self.local_feature_encoding and feature_names)
        merged["local_feature_encoding"] = self.local_feature_encoding if enabled else False
        # Stores the `merged['local_feature_names']` for use by the remaining steps in this flow.
        merged["local_feature_names"] = list(feature_names or [])
        merged["local_feature_indices"] = [int(index) for index in (feature_indices or [])]
        merged["local_feature_source"] = str(feature_source or "none")
        merged["local_feature_location"] = feature_location
        # Stores the `bc_names` for use by the remaining steps in this flow.
        bc_names = list(feature_bc_names if feature_bc_names is not None else (feature_names or []))
        bc_indices = [
            int(index)
            for index in (feature_bc_indices if feature_bc_indices is not None else (feature_indices or []))
        # Continues the surrounding statement or collection definition.
        ]
        merged["feature_bc_names"] = bc_names
        merged["feature_bc_indices"] = bc_indices
        merged["categorical_bc_indices"] = [int(index) for index in (categorical_bc_indices or [])]
        # Builds the `merged['continuous_bc_indices']` using the comprehension's filter and mapping rules.
        merged["continuous_bc_indices"] = [int(index) for index in (continuous_bc_indices or [])]
        merged["field_bc"] = list(field_bc or self.field_bc)
        merged["field_bc_location"] = field_bc_location
        merged["fields_bc_schema"] = list(fields_bc_schema or [])
        # Returns `merged` to the caller for the next stage.
        return merged

    @staticmethod
    def _build_temporal_contract_meta(
        meta: Optional[Dict[str, Any]] = None,
        # Continues the surrounding statement or collection definition.
        *,
        sample_view: Optional[str] = None,
        prediction_view: Optional[str] = None,
        total_time_steps: Optional[int] = None,
        # Continues the surrounding statement or collection definition.
        time_axis: Optional[int] = None,
        seed_length: Optional[int] = None,
        init_history_strategy: Optional[str] = None,
        target_source: Optional[str] = None,
    # Completes the multi-line `collection` arguments or elements.
    ) -> Dict[str, Any]:
        """为时序样本补齐统一的契约元信息。"""
        merged: Dict[str, Any] = dict(meta or {})
        merged.setdefault("is_temporal", True)
        merged.setdefault("sample_view", None)
        # Calls `merged.setdefault` to perform the operation required at this stage.
        merged.setdefault("prediction_view", None)
        merged.setdefault("total_time_steps", None)
        merged.setdefault("time_axis", None)
        merged.setdefault("seed_length", None)
        # Calls `merged.setdefault` to perform the operation required at this stage.
        merged.setdefault("init_history_strategy", None)

        if sample_view is not None:
            merged["sample_view"] = sample_view
        if prediction_view is not None:
            # Stores the `merged['prediction_view']` for use by the remaining steps in this flow.
            merged["prediction_view"] = prediction_view
        if total_time_steps is not None:
            merged["total_time_steps"] = int(total_time_steps)
        if time_axis is not None:
            # Converts the `merged['time_axis']` to `int` so later checks use a consistent type.
            merged["time_axis"] = int(time_axis)
        if seed_length is not None:
            merged["seed_length"] = int(seed_length)
        if init_history_strategy is not None:
            # Stores the `merged['init_history_strategy']` for use by the remaining steps in this flow.
            merged["init_history_strategy"] = init_history_strategy
        if target_source is not None:
            merged["target_source"] = target_source
        return merged

    # 构建 temporal direct 输入专用元信息。
    def _build_temporal_direct_predict_meta(
        self,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """构建 temporal direct 输入专用元信息。"""
        # Returns `` to the caller for the next stage.
        return self._build_temporal_contract_meta(
            meta,
            sample_view="full",
            prediction_view="direct_input_only",
            # Completes the multi-line `self._build_temporal_contract_meta` arguments or elements.
            target_source="dummy",
        )

    @staticmethod
    def _build_steady_input_only_meta(
        # Completes the multi-line `collection` arguments or elements.
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """构建 steady 输入专用元信息。"""
        merged: Dict[str, Any] = dict(meta or {})
        merged["prediction_view"] = "steady_input_only"
        # Stores the `merged['target_source']` for use by the remaining steps in this flow.
        merged["target_source"] = "dummy"
        return merged

    def _build_temporal_rollout_seed_meta(
        self,
        # Completes the multi-line `collection` arguments or elements.
        meta: Optional[Dict[str, Any]] = None,
        *,
        seed_length: int,
        total_time_steps: Optional[int] = None,
        # Continues the surrounding statement or collection definition.
        init_history_strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """构建 temporal autoregressive rollout seed 元信息。"""
        return self._build_temporal_contract_meta(
            meta,
            # Completes the multi-line `self._build_temporal_contract_meta` arguments or elements.
            sample_view="temporal_rollout_seed",
            prediction_view="autoregressive_seed_only",
            target_source="seed",
            seed_length=seed_length,
            # Completes the multi-line `self._build_temporal_contract_meta` arguments or elements.
            total_time_steps=total_time_steps,
            init_history_strategy=init_history_strategy,
        )

    @staticmethod
    # 根据目标张量推断总时间步数。
    def _infer_total_time_steps(
        y: Any,
        time_axis: Optional[int],
    ) -> Optional[int]:
        """根据目标张量推断总时间步数。"""
        # Returns early when `time_axis is None or not hasattr(y, "shape")` makes the remaining work inapplicable.
        if time_axis is None or not hasattr(y, "shape"):
            return None
        try:
            axis = int(time_axis)
            # Returns `int(y.shape[axis])` to the caller for the next stage.
            return int(y.shape[axis])
        except (IndexError, TypeError, ValueError):
            return None

    @staticmethod
    # 按显式时间轴切片时序张量。
    def _slice_temporal_tensor(
        tensor: Any,
        start_t: int,
        end_t: int,
        # Continues the surrounding statement or collection definition.
        time_axis: int,
    ) -> Any:
        """按显式时间轴切片时序张量。"""
        slices = [slice(None)] * tensor.ndim
        slices[int(time_axis)] = slice(int(start_t), int(end_t))
        # Returns `tensor[tuple(slices)]` to the caller for the next stage.
        return tensor[tuple(slices)]

    @staticmethod
    def _move_time_axis_to_front(
        tensor: Any,
        # Continues the surrounding statement or collection definition.
        time_axis: int,
    ) -> Any:
        """将时间轴移动到最前面，便于拼装 history。"""
        if int(time_axis) == 0:
            return tensor
        # Returns `torch.movedim(tensor, int(time_axis), 0)` to the caller for the next stage.
        return torch.movedim(tensor, int(time_axis), 0)

    @staticmethod
    def _compute_channel_stats(values: torch.Tensor) -> Dict[str, List[float]]:
        """按通道统计 mean/std/min/max。"""
        values = values.float()
        # Checks `values.ndim == 1` before entering this conditional branch.
        if values.ndim == 1:
            values = values.unsqueeze(1)
        mean = values.mean(dim=0)
        if int(values.shape[0]) > 1:
            # Stores the `std` for use by the remaining steps in this flow.
            std = values.std(dim=0, unbiased=False)
        else:
            std = torch.zeros_like(mean)
        std = torch.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0)
        # Returns `` to the caller for the next stage.
        return {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": values.min(dim=0).values.tolist(),
            # Completes the multi-line `values.max(dim=0).values.tolist` arguments or elements.
            "max": values.max(dim=0).values.tolist(),
        }

    @staticmethod
    def _finalize_delta_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
        """为增量归一化保留可逆统计，并避免常量通道的零尺度。"""
        # Stores the `finalized` for use by the remaining steps in this flow.
        finalized = dict(stats)
        finalized["std"] = [max(float(value), 1e-12) for value in finalized.get("std", [])]
        finalized["schema_version"] = 1
        return finalized

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _init_running_channel_stats(num_channels: int) -> Dict[str, Any]:
        return {
            "count": 0,
            # Completes the multi-line `torch.zeros` arguments or elements.
            "sum": torch.zeros(num_channels, dtype=torch.float64),
            "sum_sq": torch.zeros(num_channels, dtype=torch.float64),
            "min": torch.full((num_channels,), float("inf"), dtype=torch.float64),
            "max": torch.full((num_channels,), float("-inf"), dtype=torch.float64),
        # Completes the multi-line `collection` arguments or elements.
        }

    @staticmethod
    def _update_running_channel_stats(
        stats: Optional[Dict[str, Any]],
        # Continues the surrounding statement or collection definition.
        values: torch.Tensor,
    ) -> Dict[str, Any]:
        values = values.float()
        if values.numel() == 0:
            # Returns `stats` to the caller for the next stage.
            return stats
        if values.ndim == 1:
            values = values.unsqueeze(1)

        if stats is None:
            # Stores the `stats` for use by the remaining steps in this flow.
            stats = FieldProcessor._init_running_channel_stats(values.shape[1])

        values64 = values.to(dtype=torch.float64)
        stats["count"] += int(values64.shape[0])
        stats["sum"] += values64.sum(dim=0)
        # Stores the `stats['sum_sq']` for use by the remaining steps in this flow.
        stats["sum_sq"] += (values64 * values64).sum(dim=0)
        stats["min"] = torch.minimum(stats["min"], values64.min(dim=0).values)
        stats["max"] = torch.maximum(stats["max"], values64.max(dim=0).values)
        return stats

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _finalize_running_channel_stats(stats: Optional[Dict[str, Any]]) -> Dict[str, List[float]]:
        if not stats or int(stats.get("count", 0)) <= 0:
            return {}

        # Converts the `count` to `int` so later checks use a consistent type.
        count = int(stats["count"])
        mean = stats["sum"] / count
        if count > 1:
            variance = (stats["sum_sq"] - (stats["sum"] * stats["sum"]) / count) / (count - 1)
            # Stores the `variance` for use by the remaining steps in this flow.
            variance = torch.clamp(variance, min=0.0)
            std = torch.sqrt(variance)
        else:
            std = torch.zeros_like(mean)

        # Returns `` to the caller for the next stage.
        return {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": stats["min"].tolist(),
            # Completes the multi-line `stats['max'].tolist` arguments or elements.
            "max": stats["max"].tolist(),
        }

    @staticmethod
    def _init_running_geometry_stats() -> Dict[str, Any]:
        # Returns `` to the caller for the next stage.
        return {
            "min": torch.full((3,), float("inf"), dtype=torch.float64),
            "max": torch.full((3,), float("-inf"), dtype=torch.float64),
        }

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _update_running_geometry_stats(
        stats: Optional[Dict[str, Any]],
        coords: torch.Tensor,
    # Completes the multi-line `collection` arguments or elements.
    ) -> Dict[str, Any]:
        coords = coords.float()
        if coords.numel() == 0:
            return stats
        # Validates `coords.ndim != 2 or coords.shape[1] != 3` and raises `ValueError` when the input cannot be processed safely.
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"鍧愭爣寮犻噺褰㈢姸搴斾负 (N, 3)锛屽綋鍓嶄负 {tuple(coords.shape)}")

        if stats is None:
            stats = FieldProcessor._init_running_geometry_stats()

        # Stores the `coords64` for use by the remaining steps in this flow.
        coords64 = coords.to(dtype=torch.float64)
        stats["min"] = torch.minimum(stats["min"], coords64.min(dim=0).values)
        stats["max"] = torch.maximum(stats["max"], coords64.max(dim=0).values)
        return stats

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _finalize_running_geometry_stats(stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not stats:
            return {}
        # Returns `` to the caller for the next stage.
        return {
            "coord_min": stats["min"].tolist(),
            "coord_max": stats["max"].tolist(),
            "coord_mode": "minmax_neg1_pos1",
        # Completes the multi-line `collection` arguments or elements.
        }

    @staticmethod
    def _apply_channel_norm(
        values: torch.Tensor, stats: Dict[str, Any], norm_type: str
    # Continues the surrounding statement or collection definition.
    ) -> torch.Tensor:
        """对特征张量应用按通道归一化。"""
        values = values.float()
        if values.numel() == 0:
            return values

        # Checks `values.ndim == 1` before entering this conditional branch.
        if values.ndim == 1:
            values_2d = values.unsqueeze(1)
            squeeze_back = True
        else:
            # Stores the `values_2d` for use by the remaining steps in this flow.
            values_2d = values
            squeeze_back = False

        if norm_type == "standardization":
            mean = torch.tensor(
                # Completes the multi-line `torch.tensor(stats['mean'], dtype=torch.float32, device=values_2d.device).reshape` arguments or elements.
                stats["mean"], dtype=torch.float32, device=values_2d.device
            ).reshape(1, -1)
            std = torch.tensor(
                stats["std"], dtype=torch.float32, device=values_2d.device
            # Completes the multi-line `torch.tensor(stats['std'], dtype=torch.float32, device=values_2d.device).reshape` arguments or elements.
            ).reshape(1, -1)
            std = torch.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0)
            safe_std = torch.where(std.abs() < 1e-12, torch.ones_like(std), std)
            normalized = (values_2d - mean) / safe_std
        # Checks `norm_type == "min_max"` before entering this conditional branch.
        elif norm_type == "min_max":
            min_val = torch.tensor(
                stats["min"], dtype=torch.float32, device=values_2d.device
            ).reshape(1, -1)
            # Stores the `max_val` for use by the remaining steps in this flow.
            max_val = torch.tensor(
                stats["max"], dtype=torch.float32, device=values_2d.device
            ).reshape(1, -1)
            min_val = torch.nan_to_num(min_val, nan=0.0, posinf=0.0, neginf=0.0)
            # Stores the `max_val` for use by the remaining steps in this flow.
            max_val = torch.nan_to_num(max_val, nan=0.0, posinf=0.0, neginf=0.0)
            scale = torch.nan_to_num(
                max_val - min_val,
                nan=0.0,
                # Completes the multi-line `torch.nan_to_num` arguments or elements.
                posinf=0.0,
                neginf=0.0,
            )
            safe_scale = torch.where(scale.abs() < 1e-12, torch.ones_like(scale), scale)
            # Derives the `normalized` from the current values for downstream processing.
            normalized = (values_2d - min_val) / safe_scale
        else:
            raise ValueError(f"不支持的归一化方式: {norm_type}")

        if squeeze_back:
            # Returns `normalized.squeeze(1)` to the caller for the next stage.
            return normalized.squeeze(1)
        return normalized

    @staticmethod
    def _compute_geometry_params(coords: torch.Tensor) -> Dict[str, Any]:
        """统计几何坐标归一化所需的范围参数。"""
        # Stores the `coords` for use by the remaining steps in this flow.
        coords = coords.float()
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"坐标张量形状应为 (N, 3)，当前为 {tuple(coords.shape)}")
        return {
            # Completes the multi-line `coords.min(dim=0).values.tolist` arguments or elements.
            "coord_min": coords.min(dim=0).values.tolist(),
            "coord_max": coords.max(dim=0).values.tolist(),
            "coord_mode": "minmax_neg1_pos1",
        }

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _normalize_coordinates(coords: torch.Tensor, geometry_params: Dict[str, Any]) -> torch.Tensor:
        """将坐标缩放到 [-1, 1]。"""
        coords = coords.float()
        coord_min = torch.tensor(
            # Completes the multi-line `torch.tensor(geometry_params['coord_min'], dtype=torch.float32, device=coords.device).reshape` arguments or elements.
            geometry_params["coord_min"], dtype=torch.float32, device=coords.device
        ).reshape(1, 3)
        coord_max = torch.tensor(
            geometry_params["coord_max"], dtype=torch.float32, device=coords.device
        # Completes the multi-line `torch.tensor(geometry_params['coord_max'], dtype=torch.float32, device=coords.device).reshape` arguments or elements.
        ).reshape(1, 3)
        scaled = (coords - coord_min) / (coord_max - coord_min + 1e-12)
        return scaled * 2.0 - 1.0

    @staticmethod
    # 对特征张量应用按通道反归一化。
    def _apply_channel_denorm(
        values: torch.Tensor, stats: Dict[str, Any], norm_type: str
    ) -> torch.Tensor:
        """对特征张量应用按通道反归一化。"""
        values = values.float()
        # Returns early when `values.numel() == 0` makes the remaining work inapplicable.
        if values.numel() == 0:
            return values

        if values.ndim == 1:
            values_2d = values.unsqueeze(1)
            # Stores the `squeeze_back` for use by the remaining steps in this flow.
            squeeze_back = True
        else:
            values_2d = values
            squeeze_back = False

        # Checks `norm_type == "standardization"` before entering this conditional branch.
        if norm_type == "standardization":
            mean = torch.tensor(
                stats["mean"], dtype=torch.float32, device=values_2d.device
            ).reshape(1, -1)
            # Stores the `std` for use by the remaining steps in this flow.
            std = torch.tensor(
                stats["std"], dtype=torch.float32, device=values_2d.device
            ).reshape(1, -1)
            std = torch.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0)
            # Stores the `safe_std` for use by the remaining steps in this flow.
            safe_std = torch.where(std.abs() < 1e-12, torch.ones_like(std), std)
            denormalized = values_2d * safe_std + mean
        elif norm_type == "min_max":
            min_val = torch.tensor(
                # Completes the multi-line `torch.tensor(stats['min'], dtype=torch.float32, device=values_2d.device).reshape` arguments or elements.
                stats["min"], dtype=torch.float32, device=values_2d.device
            ).reshape(1, -1)
            max_val = torch.tensor(
                stats["max"], dtype=torch.float32, device=values_2d.device
            # Completes the multi-line `torch.tensor(stats['max'], dtype=torch.float32, device=values_2d.device).reshape` arguments or elements.
            ).reshape(1, -1)
            min_val = torch.nan_to_num(min_val, nan=0.0, posinf=0.0, neginf=0.0)
            max_val = torch.nan_to_num(max_val, nan=0.0, posinf=0.0, neginf=0.0)
            scale = torch.nan_to_num(
                # Completes the multi-line `torch.nan_to_num` arguments or elements.
                max_val - min_val,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            # Completes the multi-line `torch.nan_to_num` arguments or elements.
            )
            safe_scale = torch.where(scale.abs() < 1e-12, torch.ones_like(scale), scale)
            denormalized = values_2d * safe_scale + min_val
        else:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(f"不支持的归一化方式: {norm_type}")

        if squeeze_back:
            return denormalized.squeeze(1)
        return denormalized

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _to_numpy(value: Any) -> Any:
        """将常见张量安全转换为 numpy，其他类型保持原样。"""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        # Returns `value` to the caller for the next stage.
        return value
