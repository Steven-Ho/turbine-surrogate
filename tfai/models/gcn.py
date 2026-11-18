"""基于 MeshGraphNet 消息传递的图场预测模型。"""

from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
# Imports `torch.nn.functional` for reuse in this module.
import torch.nn.functional as functional
from torch_geometric.utils import scatter


class _MeshGraphMLP(nn.Module):
    """与 MeshGraphNet 对齐的 MLP：隐藏层激活，输出后可选 LayerNorm。"""

    def __init__(
        # Continues the surrounding statement or collection definition.
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        # Continues the surrounding statement or collection definition.
        hidden_layers: int = 2,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        # Validates `hidden_layers < 1` and raises `ValueError` when the input cannot be processed safely.
        if hidden_layers < 1:
            raise ValueError("MeshGraph MLP 的 hidden_layers 必须不小于 1")

        self.layers = nn.ModuleList()
        current_dim = int(input_dim)
        # Visits each item in `range(hidden_layers)` and applies the current processing step.
        for layer_index in range(hidden_layers):
            next_dim = hidden_dim
            self.layers.append(nn.Linear(current_dim, next_dim))
            current_dim = next_dim
        # Stores the output on the instance so other methods can reuse it.
        self.output = nn.Linear(int(current_dim), output_dim)
        self.activation = nn.ReLU(inplace=False)
        self.norm = nn.LayerNorm(output_dim) if use_layer_norm else nn.Identity()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # Visits each item in `self.layers` and applies the current processing step.
        for layer in self.layers:
            values = self.activation(layer(values))
        return self.norm(self.output(values))

    def forward_from_linear_parts(self, parts: Iterable[torch.Tensor]) -> torch.Tensor:
        """以首层线性投影之和等价替代特征拼接，避免构造大边消息临时张量。"""
        # Stores the `values` for use by the remaining steps in this flow.
        values = tuple(parts)
        if not values:
            raise ValueError("MeshGraph MLP 至少需要一个首层输入分量")
        if not self.layers:
            # Raises `RuntimeError` to report why this path cannot continue.
            raise RuntimeError("MeshGraph MLP 缺少首个线性层")

        first_layer = self.layers[0]
        input_width = sum(int(item.shape[-1]) for item in values)
        if input_width != first_layer.in_features:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(
                "MeshGraph MLP 首层输入维度不匹配："
                f"期望 {first_layer.in_features}，实际 {input_width}"
            )
        # Selects the required item from `values[0].shape` as the `leading_shape`.
        leading_shape = values[0].shape[:-1]
        if any(item.shape[:-1] != leading_shape for item in values):
            raise ValueError("MeshGraph MLP 的各首层输入分量必须具有相同的前导维度")

        offset = 0
        # Stores the output for use by the remaining steps in this flow.
        output = None
        for index, item in enumerate(values):
            next_offset = offset + int(item.shape[-1])
            bias = first_layer.bias if index == 0 else None
            # Stores the `projected` for use by the remaining steps in this flow.
            projected = functional.linear(
                item,
                first_layer.weight[:, offset:next_offset],
                bias,
            # Completes the multi-line `functional.linear` arguments or elements.
            )
            output = projected if output is None else output + projected
            offset = next_offset

        assert output is not None
        # Stores the `values` for use by the remaining steps in this flow.
        values = self.activation(output)
        for layer in self.layers[1:]:
            values = self.activation(layer(values))
        return self.norm(self.output(values))

    # Defines `reset_parameters` for reset parameters logic.
    def reset_parameters(self) -> None:
        for layer in self.layers:
            layer.reset_parameters()
        self.output.reset_parameters()
        # Checks `hasattr(self.norm, "reset_parameters")` before entering this conditional branch.
        if hasattr(self.norm, "reset_parameters"):
            self.norm.reset_parameters()


class _MeshGraphBlock(nn.Module):
    """先更新边 latent，再聚合入边消息并更新节点 latent。"""

    def __init__(self, hidden_dim: int) -> None:
        # Calls `super().__init__` to perform the operation required at this stage.
        super().__init__()
        self.edge_mlp = _MeshGraphMLP(
            input_dim=hidden_dim * 3,
            output_dim=hidden_dim,
            # Completes the multi-line `_MeshGraphMLP` arguments or elements.
            hidden_dim=hidden_dim,
            use_layer_norm=True,
        )
        self.node_mlp = _MeshGraphMLP(
            # Completes the multi-line `_MeshGraphMLP` arguments or elements.
            input_dim=hidden_dim * 2,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            use_layer_norm=True,
        # Completes the multi-line `_MeshGraphMLP` arguments or elements.
        )

    def forward(
        self,
        node_features: torch.Tensor,
        # Continues the surrounding statement or collection definition.
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        source_nodes = edge_index[0]
        # Selects the required item from `edge_index` as the `receiver_nodes`.
        receiver_nodes = edge_index[1]
        edge_delta = self.edge_mlp.forward_from_linear_parts(
            (
                edge_features,
                # Completes the multi-line `collection` arguments or elements.
                node_features[source_nodes],
                node_features[receiver_nodes],
            )
        )
        # Derives the `next_edge_features` from the current values for downstream processing.
        next_edge_features = edge_features + edge_delta
        aggregated_messages = scatter(
            next_edge_features,
            receiver_nodes,
            # Completes the multi-line `scatter` arguments or elements.
            dim=0,
            dim_size=node_features.shape[0],
            reduce="sum",
        )
        # Stores the `node_delta` for use by the remaining steps in this flow.
        node_delta = self.node_mlp(
            torch.cat([node_features, aggregated_messages], dim=-1)
        )
        return node_features + node_delta, next_edge_features

    # Defines `reset_parameters` for reset parameters logic.
    def reset_parameters(self) -> None:
        self.edge_mlp.reset_parameters()
        self.node_mlp.reset_parameters()


class GCN(nn.Module):
    """无 world edge 的 MeshGraphNet 风格图场预测模型。

    模型名称保持为 GCN，以兼容既有工厂和数据集路由；内部使用节点/边编码器、
    残差消息传递 processor 与节点解码器。边输入固定为 GraphProcessor 生成的四维几何特征。
    """

    # Initializes the `_SCALE_CONFIG` as a collection for values from this processing stage.
    _SCALE_CONFIG = {
        "small": {"processor_layers": 5, "hidden_size": 64},
        "base": {"processor_layers": 10, "hidden_size": 128},
        "large": {"processor_layers": 15, "hidden_size": 128},
    # Completes the multi-line `collection` arguments or elements.
    }
    _MLP_HIDDEN_LAYERS = 2
    _EDGE_FEATURE_DIM = 4

    def __init__(self, config: Dict, input_dim: int, output_dim: int):
        # Calls `super().__init__` to perform the operation required at this stage.
        super().__init__()
        self.config = config

        scale = config["model"]["scale"]
        if scale not in self._SCALE_CONFIG:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(
                f"不支持的 GCN scale: {scale}，可选值为 small、base、large"
            )
        if input_dim <= 0 or output_dim <= 0:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(f"input_dim={input_dim}，output_dim={output_dim} 必须大于 0")

        scale_cfg = self._SCALE_CONFIG[scale]
        self.scale = scale
        self.input_dim = int(input_dim)
        # Stores the `output_dim` on the instance so other methods can reuse it.
        self.output_dim = int(output_dim)
        self.hidden_size = int(scale_cfg["hidden_size"])
        self.processor_layers = int(scale_cfg["processor_layers"])
        self.edge_feature_dim = self._EDGE_FEATURE_DIM

        # Stores the `node_encoder` on the instance so other methods can reuse it.
        self.node_encoder = _MeshGraphMLP(
            input_dim=self.input_dim,
            output_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            # Completes the multi-line `_MeshGraphMLP` arguments or elements.
            hidden_layers=self._MLP_HIDDEN_LAYERS,
            use_layer_norm=True,
        )
        self.edge_encoder = _MeshGraphMLP(
            # Completes the multi-line `_MeshGraphMLP` arguments or elements.
            input_dim=self.edge_feature_dim,
            output_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            hidden_layers=self._MLP_HIDDEN_LAYERS,
            # Completes the multi-line `_MeshGraphMLP` arguments or elements.
            use_layer_norm=True,
        )
        self.processor = nn.ModuleList(
            _MeshGraphBlock(self.hidden_size)
            # Completes the multi-line `range` arguments or elements.
            for _ in range(self.processor_layers)
        )
        self.node_decoder = _MeshGraphMLP(
            input_dim=self.hidden_size,
            # Completes the multi-line `_MeshGraphMLP` arguments or elements.
            output_dim=self.output_dim,
            hidden_dim=self.hidden_size,
            hidden_layers=self._MLP_HIDDEN_LAYERS,
            use_layer_norm=False,
        # Completes the multi-line `_MeshGraphMLP` arguments or elements.
        )

    def _validate_graph_input(
        self,
        data,
    # Completes the multi-line `collection` arguments or elements.
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """校验并返回节点、拓扑与边特征，防止退化为忽略物理边信息的 GCN。"""
        node_features = getattr(data, "x", None)
        edge_index = getattr(data, "edge_index", None)
        edge_features = getattr(data, "edge_attr", None)
        # Validates `` and raises `ValueError` when the input cannot be processed safely.
        if not isinstance(node_features, torch.Tensor) or node_features.ndim != 2:
            raise ValueError("GCN 要求 data.x 为 [num_nodes, num_features] 张量")
        if node_features.shape[-1] != self.input_dim:
            raise ValueError(
                # Completes the multi-line `ValueError` arguments or elements.
                f"GCN 节点特征维度不匹配：期望 {self.input_dim}，实际 {node_features.shape[-1]}"
            )
        if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("GCN 要求 data.edge_index 形状为 [2, num_edges]")
        # Validates `` and raises `ValueError` when the input cannot be processed safely.
        if not isinstance(edge_features, torch.Tensor) or edge_features.ndim != 2:
            raise ValueError("MeshGraphNet 风格 GCN 要求 data.edge_attr 为 [num_edges, edge_dim] 张量")
        if edge_features.shape[0] != edge_index.shape[1]:
            raise ValueError(
                # Completes the multi-line `ValueError` arguments or elements.
                "GCN 的 edge_attr 行数必须与 edge_index 的边数一致："
                f"edge_attr={edge_features.shape[0]}，edge_index={edge_index.shape[1]}"
            )
        if edge_features.shape[1] <= 0:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("GCN 的 edge_attr 必须至少包含一个边特征通道")
        if edge_index.numel() and (
            edge_index.min().item() < 0 or edge_index.max().item() >= node_features.shape[0]
        ):
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("GCN 的 edge_index 存在越界节点索引")

        if edge_features.shape[1] != self.edge_feature_dim:
            raise ValueError(
                f"GCN 边特征维度必须为 {self.edge_feature_dim}，"
                # Completes the multi-line `ValueError` arguments or elements.
                f"实际 {edge_features.shape[1]}"
            )
        return node_features, edge_index.long(), edge_features

    def forward(self, data):
        # Stores the `(node_features, edge_index, edge_features)` for use by the remaining steps in this flow.
        node_features, edge_index, edge_features = self._validate_graph_input(data)
        node_features = self.node_encoder(node_features)
        edge_features = self.edge_encoder(edge_features)

        for block in self.processor:
            # Stores the `(node_features, edge_features)` for use by the remaining steps in this flow.
            node_features, edge_features = block(
                node_features,
                edge_features,
                edge_index,
            # Completes the multi-line `block` arguments or elements.
            )
        return self.node_decoder(node_features)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """拒绝加载旧 GCNConv 权重，避免同名模型发生静默结构错配。"""
        legacy_prefixes = ("input_proj.", "gcn_layers.", "skip_projs.", "batch_norms.", "output_proj.")
        # Validates `any(str(key).startswith(legacy_prefixes) for key in state_dict)` and raises `ValueError` when the input cannot be processed safely.
        if any(str(key).startswith(legacy_prefixes) for key in state_dict):
            raise ValueError(
                "检测到旧版 GCNConv checkpoint；当前 GCN 已改为 MeshGraphNet 结构，"
                "旧 checkpoint 不兼容，请重新训练。"
            # Completes the multi-line `ValueError` arguments or elements.
            )
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def reset_parameters(self) -> None:
        self.node_encoder.reset_parameters()
        # Calls `self.edge_encoder.reset_parameters` to perform the operation required at this stage.
        self.edge_encoder.reset_parameters()
        for block in self.processor:
            block.reset_parameters()
        self.node_decoder.reset_parameters()
