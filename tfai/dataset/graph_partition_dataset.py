import math
from collections import deque
from typing import Dict, List, Tuple

# Imports `torch` for reuse in this module.
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as PyGData
from torch_geometric.utils import subgraph as pyg_subgraph

# Imports `FullSampleDataset` from `tfai.dataset.full_sample_dataset` for use in this module.
from tfai.dataset.full_sample_dataset import FullSampleDataset


class GraphPartitionDataset(Dataset):
    """将一个完整图样本转换为一组 owner + halo 子图。

    一个 ``__getitem__`` 仍然代表一个完整图样本；其中的所有分块必须在同一次
    参数更新前反向传播，避免旧实现将分块错误地当成相互独立的训练样本。
    """

    data_kind = "graph"
    sample_view = "graph_partition"
    # Stores the `partition_format_version` for use by the remaining steps in this flow.
    partition_format_version = 2

    def __init__(self, base_dataset: Dataset, config: Dict) -> None:
        self.base_dataset = base_dataset
        self.config = config
        # Stores the `model_type` on the instance so other methods can reuse it.
        self.model_type = getattr(base_dataset, "model_type", config["model"]["type"].lower())
        self.mode = config["task"]["mode"].lower()
        self.temporal_strategy = config["task"]["temporal_strategy"].lower()
        self.is_geometry_varying = bool(config["task"]["is_geometry_varying"])

        # Reads `training` from the mapping into the `training_cfg` with its declared fallback.
        training_cfg = config.get("training", {})
        self.graph_use_partition = bool(training_cfg.get("graph_use_partition", False))
        self.partition_count = int(training_cfg.get("graph_partition_count", 1) or 1)
        # 多步自由滚动会逐步扩大依赖域，需覆盖 horizon × 单步消息传递感受野。
        self.halo_hops = self.resolve_halo_hops(config)

        # Validates `not self.graph_use_partition` and raises `ValueError` when the input cannot be processed safely.
        if not self.graph_use_partition:
            raise ValueError("GraphPartitionDataset requires training.graph_use_partition=true")
        if self.partition_count < 2:
            raise ValueError("training.graph_partition_count must be >= 2 when partition training is enabled")
        # Validates `getattr(base_dataset, "data_kind", None) != "graph"` and raises `TypeError` when the input cannot be processed safely.
        if getattr(base_dataset, "data_kind", None) != "graph":
            raise TypeError("GraphPartitionDataset expects a graph-like base dataset")

        self._partition_cache: Dict[str, List[torch.Tensor]] = {}

    @classmethod
    # Defines `from_csv` for from csv logic.
    def from_csv(cls, csv_path: str, config: Dict, data_dir: str) -> "GraphPartitionDataset":
        base_dataset = FullSampleDataset.from_csv(csv_path=csv_path, config=config, data_dir=data_dir)
        return cls(base_dataset=base_dataset, config=config)

    @staticmethod
    # 解析首版 MeshGraphNet 的精确局部感受野。
    def resolve_halo_hops(config: Dict) -> int:
        """解析首版 MeshGraphNet 的精确局部感受野。"""
        model_type = str(config.get("model", {}).get("type", "")).lower()
        if model_type != "gcn":
            raise NotImplementedError("graph_partition 首版仅支持 GCN（MeshGraphNet）模型")
        # Stores the `scale` for use by the remaining steps in this flow.
        scale = str(config.get("model", {}).get("scale", "base")).lower()
        layers_by_scale = {"small": 5, "base": 10, "large": 15}
        if scale not in layers_by_scale:
            raise ValueError(f"不支持的 GCN scale: {scale}")
        # Stores the `rollout_cfg` for use by the remaining steps in this flow.
        rollout_cfg = config.get("training", {}).get("temporal_rollout", {}) or {}
        is_autoregressive = (
            str(config.get("task", {}).get("mode", "")).lower() == "temporal"
            and str(config.get("task", {}).get("temporal_strategy", "")).lower() == "autoregressive"
        # Continues the surrounding statement or collection definition.
        )
        horizon = rollout_cfg.get("horizon", 1) if is_autoregressive else 1
        if isinstance(horizon, bool):
            raise ValueError("training.temporal_rollout.horizon 必须为正整数")
        # Keeps the operation and its recovery path within the same exception boundary.
        try:
            horizon = int(horizon)
        except (TypeError, ValueError) as exc:
            raise ValueError("training.temporal_rollout.horizon 必须为正整数") from exc
        # Validates `horizon <= 0` and raises `ValueError` when the input cannot be processed safely.
        if horizon <= 0:
            raise ValueError("training.temporal_rollout.horizon 必须为正整数")
        return layers_by_scale[scale] * horizon

    def __len__(self) -> int:
        # Returns `len(self.base_dataset)` to the caller for the next stage.
        return len(self.base_dataset)

    def __getitem__(self, base_idx: int):
        graph, y = self.base_dataset[base_idx]
        if not isinstance(graph, PyGData):
            # Raises `TypeError` to report why this path cannot continue.
            raise TypeError(f"Expected PyG Data from base dataset, got {type(graph)}")

        partition_groups = self._get_partition_node_groups(graph, cache_key=self._cache_key(base_idx))
        logical_sample = []
        for partition_id, owner_node_ids in enumerate(partition_groups):
            # Stores the `(subgraph, subgraph_y)` for use by the remaining steps in this flow.
            subgraph, subgraph_y = self._extract_partition_subgraph(
                graph=graph,
                y=y,
                owner_node_ids=owner_node_ids,
                # Completes the multi-line `self._extract_partition_subgraph` arguments or elements.
                halo_hops=self.halo_hops,
                partition_id=partition_id,
            )
            logical_sample.append((subgraph, subgraph_y))
        # Returns `logical_sample` to the caller for the next stage.
        return logical_sample

    def get_sample_meta(self, idx: int) -> Dict:
        meta = {}
        if hasattr(self.base_dataset, "get_sample_meta"):
            # Adds the current value to `meta` for later aggregation.
            meta.update(self.base_dataset.get_sample_meta(idx))
        meta.update({"base_index": int(idx), "sample_view": self.sample_view})
        return meta

    def _cache_key(self, base_idx: int) -> str:
        # Returns early when `not self.is_geometry_varying` makes the remaining work inapplicable.
        if not self.is_geometry_varying:
            return "__fixed_geometry__"
        # 变几何样本不能仅按文件路径复用分块：同一文件中的不同时刻也可能拥有不同拓扑。
        return f"__sample_{int(base_idx)}__"

    def _get_partition_node_groups(self, graph: PyGData, cache_key: str) -> List[torch.Tensor]:
        # Checks `cache_key not in self._partition_cache` before entering this conditional branch.
        if cache_key not in self._partition_cache:
            self._partition_cache[cache_key] = self.build_partition_groups(
                graph=graph,
                partition_count=self.partition_count,
            # Completes the multi-line `self.build_partition_groups` arguments or elements.
            )
        return self._partition_cache[cache_key]

    @staticmethod
    def build_partition_groups(graph: PyGData, partition_count: int) -> List[torch.Tensor]:
        # Converts the `num_nodes` to `int` so later checks use a consistent type.
        num_nodes = int(graph.num_nodes)
        if partition_count > num_nodes:
            raise ValueError(
                f"training.graph_partition_count={partition_count} exceeds graph node count={num_nodes}"
            # Completes the multi-line `ValueError` arguments or elements.
            )
        adjacency = GraphPartitionDataset._build_adjacency(graph)
        remaining = set(range(num_nodes))
        groups: List[torch.Tensor] = []
        # Visits each item in `range(partition_count)` and applies the current processing step.
        for partition_id in range(partition_count):
            target_size = max(1, math.ceil(len(remaining) / (partition_count - partition_id)))
            groups.append(torch.tensor(sorted(GraphPartitionDataset._take_connected_nodes(
                adjacency, remaining, target_size
            # Completes the multi-line `groups.append` arguments or elements.
            )), dtype=torch.long))
        if remaining:
            groups[-1] = torch.cat([groups[-1], torch.tensor(sorted(remaining), dtype=torch.long)])
        return groups

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _build_adjacency(graph: PyGData) -> List[List[int]]:
        num_nodes = int(graph.num_nodes)
        adjacency = [[] for _ in range(num_nodes)]
        # Stores the graph edge indices for use by the remaining steps in this flow.
        edge_index = graph.edge_index.detach().cpu()
        for edge_id in range(int(edge_index.shape[1])):
            src, dst = int(edge_index[0, edge_id]), int(edge_index[1, edge_id])
            if 0 <= src < num_nodes and 0 <= dst < num_nodes:
                # Adds the current value to `adjacency[src]` for later aggregation.
                adjacency[src].append(dst)
                if src != dst:
                    adjacency[dst].append(src)
        return adjacency

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _take_connected_nodes(adjacency: List[List[int]], remaining: set, target_size: int) -> List[int]:
        chosen: List[int] = []
        while remaining and len(chosen) < target_size:
            # Stores the `seed` for use by the remaining steps in this flow.
            seed = min(remaining)
            queue, queued = deque([seed]), {seed}
            while queue and len(chosen) < target_size:
                node = queue.popleft()
                # Checks `node not in remaining` before entering this conditional branch.
                if node not in remaining:
                    continue
                remaining.remove(node)
                chosen.append(node)
                # Visits each item in `adjacency[node]` and applies the current processing step.
                for neighbor in adjacency[node]:
                    if neighbor in remaining and neighbor not in queued:
                        queue.append(neighbor)
                        queued.add(neighbor)
        # Returns `chosen` to the caller for the next stage.
        return chosen

    @staticmethod
    def _expand_halo_nodes(graph: PyGData, owner_node_ids: torch.Tensor, halo_hops: int) -> torch.Tensor:
        adjacency = GraphPartitionDataset._build_adjacency(graph)
        # Builds the `visited` using the comprehension's filter and mapping rules.
        visited = {int(node) for node in owner_node_ids.detach().cpu().tolist()}
        frontier = set(visited)
        for _ in range(int(halo_hops)):
            next_frontier = {neighbor for node in frontier for neighbor in adjacency[node]} - visited
            # Checks `not next_frontier` before entering this conditional branch.
            if not next_frontier:
                break
            visited.update(next_frontier)
            frontier = next_frontier
        # Returns `` to the caller for the next stage.
        return torch.tensor(sorted(visited), dtype=torch.long, device=owner_node_ids.device)

    @staticmethod
    def _slice_target(y: torch.Tensor, node_ids: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if y.ndim == 2:
            # Returns `y[node_ids]` to the caller for the next stage.
            return y[node_ids]
        if y.ndim == 3:
            if y.shape[0] == num_nodes:
                return y[node_ids]
            # Returns early when `y.shape[1] == num_nodes` makes the remaining work inapplicable.
            if y.shape[1] == num_nodes:
                return y[:, node_ids]
        raise ValueError(f"无法识别图目标的节点维度: y.shape={tuple(y.shape)}, num_nodes={num_nodes}")

    @staticmethod
    # Defines `_extract_partition_subgraph` for  extract partition subgraph logic.
    def _extract_partition_subgraph(
        graph: PyGData,
        y: torch.Tensor,
        owner_node_ids: torch.Tensor,
        # Continues the surrounding statement or collection definition.
        halo_hops: int,
        partition_id: int,
    ) -> Tuple[PyGData, torch.Tensor]:
        node_ids = GraphPartitionDataset._expand_halo_nodes(graph, owner_node_ids, halo_hops)
        # Stores the `edge_attr` for use by the remaining steps in this flow.
        edge_attr = getattr(graph, "edge_attr", None)
        sub_edge_index, sub_edge_attr, edge_mask = pyg_subgraph(
            subset=node_ids,
            edge_index=graph.edge_index,
            # Completes the multi-line `pyg_subgraph` arguments or elements.
            edge_attr=edge_attr,
            relabel_nodes=True,
            num_nodes=int(graph.num_nodes),
            return_edge_mask=True,
        # Completes the multi-line `pyg_subgraph` arguments or elements.
        )
        subgraph = PyGData(x=graph.x[node_ids], edge_index=sub_edge_index)
        if sub_edge_attr is not None:
            subgraph.edge_attr = sub_edge_attr

        # Initializes the `(num_nodes, num_edges)` as a collection for values from this processing stage.
        num_nodes, num_edges = int(graph.num_nodes), int(graph.edge_index.shape[1])
        for key in graph.keys():
            if key in {"x", "edge_index", "edge_attr", "y", "num_nodes"}:
                continue
            # Stores the `value` for use by the remaining steps in this flow.
            value = getattr(graph, key)
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == num_nodes:
                setattr(subgraph, key, value[node_ids])
            elif torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == num_edges:
                # Calls `setattr` to perform the operation required at this stage.
                setattr(subgraph, key, value[edge_mask])
            else:
                setattr(subgraph, key, value.clone() if torch.is_tensor(value) else value)

        owner_set = set(int(node) for node in owner_node_ids.detach().cpu().tolist())
        # Stores the `owner_mask` for use by the remaining steps in this flow.
        subgraph.owner_mask = torch.tensor(
            [int(node) in owner_set for node in node_ids.detach().cpu().tolist()],
            dtype=torch.bool,
            device=node_ids.device,
        # Completes the multi-line `torch.tensor` arguments or elements.
        )
        subgraph.partition_id = torch.tensor([int(partition_id)], dtype=torch.long)
        subgraph.owner_node_count = int(owner_node_ids.numel())
        subgraph.local_node_count = int(node_ids.numel())
        # Converts the `num_full_graph_nodes` to `int` so later checks use a consistent type.
        subgraph.num_full_graph_nodes = int(getattr(graph, "num_full_graph_nodes", graph.num_nodes))
        if not hasattr(subgraph, "global_entity_id"):
            subgraph.global_entity_id = node_ids.clone()
        subgraph.field_location = getattr(graph, "field_location", "point")
        # Returns `` to the caller for the next stage.
        return subgraph, GraphPartitionDataset._slice_target(y, node_ids, int(graph.num_nodes))
