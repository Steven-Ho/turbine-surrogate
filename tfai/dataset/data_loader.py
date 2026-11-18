import copy
from dataclasses import dataclass
from typing import Any, Optional, Callable, Iterable, Tuple, List
# Imports `torch` for reuse in this module.
import torch
import torch.utils.data as torch_data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Data as PyGData, Batch


# Continues the surrounding statement or collection definition.
@dataclass
class TransolverBatch:
    """不同节点数样本补齐后的通用点集批次。"""

    x: torch.Tensor
    pos: torch.Tensor
    # Stores the valid-item mask for use by the remaining steps in this flow.
    valid_mask: torch.Tensor
    node_counts: torch.Tensor
    sample_id: torch.Tensor
    boundary_mask: Optional[torch.Tensor] = None
    # Stores the `boundary_entity_codes` for use by the remaining steps in this flow.
    boundary_entity_codes: Optional[torch.Tensor] = None
    boundary_state_override_mask: Optional[torch.Tensor] = None
    boundary_state_override_values: Optional[torch.Tensor] = None
    time: Optional[torch.Tensor] = None

    # 按 PyG 数据对象语义迁移批次中的全部张量。
    def to(self, device, non_blocking: bool = False):
        """按 PyG 数据对象语义迁移批次中的全部张量。"""
        values = {}
        for key, value in self.__dict__.items():
            values[key] = (
                # Completes the multi-line `value.to` arguments or elements.
                value.to(device=device, non_blocking=non_blocking)
                if isinstance(value, torch.Tensor)
                else value
            )
        # Returns `type(self)(**values)` to the caller for the next stage.
        return type(self)(**values)

    def clone(self):
        """复制张量字段，供自回归 rollout 安全更新历史状态。"""
        values = {}
        for key, value in self.__dict__.items():
            # Stores the `values[key]` for use by the remaining steps in this flow.
            values[key] = value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        return type(self)(**values)


def _resolve_set_sample_input(item: Any) -> dict:
    """读取处理后点集样本的标准字典输入。"""
    if not isinstance(item, dict):
        # Raises `TypeError` to report why this path cannot continue.
        raise TypeError("Transolver batch 输入必须为字典")
    features = item.get("x", item.get("features"))
    pos = item.get("pos")
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        # Raises `ValueError` to report why this path cannot continue.
        raise ValueError("Transolver 单样本 x 必须为 [N, C] 张量")
    if not isinstance(pos, torch.Tensor) or pos.ndim != 2 or pos.shape[0] != features.shape[0]:
        raise ValueError("Transolver 单样本 pos 必须为与 x 对齐的 [N, D] 张量")
    return item


# Defines `_copy_optional_1d` for  copy optional 1d logic.
def _copy_optional_1d(
    batch_tensor: torch.Tensor,
    item: dict,
    key: str,
    # Continues the surrounding statement or collection definition.
    batch_index: int,
    node_count: int,
) -> None:
    values = item.get(key)
    # Returns early when `values is None` makes the remaining work inapplicable.
    if values is None:
        return
    if not isinstance(values, torch.Tensor) or values.ndim != 1 or values.numel() != node_count:
        raise ValueError(f"Transolver 单样本 {key} 必须为 [N] 张量")
    # Stores the `batch_tensor[batch_index, :node_count]` for use by the remaining steps in this flow.
    batch_tensor[batch_index, :node_count] = values.to(dtype=batch_tensor.dtype)


def _copy_optional_2d(
    batch_tensor: torch.Tensor,
    item: dict,
    # Continues the surrounding statement or collection definition.
    key: str,
    batch_index: int,
    node_count: int,
) -> None:
    # Stores the `values` for use by the remaining steps in this flow.
    values = item.get(key)
    if values is None:
        return
    if not isinstance(values, torch.Tensor) or values.ndim != 2 or values.shape[0] != node_count:
        # Raises `ValueError` to report why this path cannot continue.
        raise ValueError(f"Transolver 单样本 {key} 必须为 [N, C] 张量")
    if values.shape[1] != batch_tensor.shape[-1]:
        raise ValueError(f"Transolver 单样本 {key} 通道数不一致")
    batch_tensor[batch_index, :node_count] = values.to(dtype=batch_tensor.dtype)


# 按样本动态补齐节点维度，并保留物理边界与样本标识。
def collate_transolver_batch(batch: List[Tuple]) -> Tuple[TransolverBatch, torch.Tensor]:
    """按样本动态补齐节点维度，并保留物理边界与样本标识。"""
    if not batch:
        raise ValueError("Transolver batch 不能为空")

    inputs, targets = zip(*batch)
    # Builds the `items` using the comprehension's filter and mapping rules.
    items = [_resolve_set_sample_input(item) for item in inputs]
    if not all(isinstance(target, torch.Tensor) for target in targets):
        raise TypeError("Transolver batch 目标必须为张量")
    target_ndim = targets[0].ndim
    # Validates `` and raises `ValueError` when the input cannot be processed safely.
    if target_ndim not in {2, 3} or any(target.ndim != target_ndim for target in targets):
        raise ValueError("Transolver 目标仅支持 [N, C] 或 [T, N, C]，且同 batch 维度一致")

    node_counts = [int(item["x"].shape[0]) for item in items]
    feature_dim = int(items[0]["x"].shape[1])
    # Converts the `pos_dim` to `int` so later checks use a consistent type.
    pos_dim = int(items[0]["pos"].shape[1])
    if any(item["x"].shape[1] != feature_dim or item["pos"].shape[1] != pos_dim for item in items):
        raise ValueError("Transolver batch 中的特征或坐标通道数不一致")
    if any(int(target.shape[-1]) != int(targets[0].shape[-1]) for target in targets):
        # Raises `ValueError` to report why this path cannot continue.
        raise ValueError("Transolver batch 中目标通道数不一致")
    if target_ndim == 3 and any(target.shape[0] != targets[0].shape[0] for target in targets):
        raise ValueError("Transolver 时序 batch 中时间步数不一致")

    batch_size = len(items)
    # Stores the `max_nodes` for use by the remaining steps in this flow.
    max_nodes = max(node_counts)
    device = items[0]["x"].device
    dtype = items[0]["x"].dtype
    x = torch.zeros((batch_size, max_nodes, feature_dim), dtype=dtype, device=device)
    # Stores the `pos` for use by the remaining steps in this flow.
    pos = torch.zeros((batch_size, max_nodes, pos_dim), dtype=items[0]["pos"].dtype, device=device)
    valid_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool, device=device)
    sample_ids = torch.zeros((batch_size,), dtype=torch.long, device=device)

    has_boundary_mask = any(item.get("boundary_mask") is not None for item in items)
    # Stores the `has_boundary_codes` for use by the remaining steps in this flow.
    has_boundary_codes = any(item.get("boundary_entity_codes") is not None for item in items)
    has_override_mask = any(item.get("boundary_state_override_mask") is not None for item in items)
    has_override_values = any(item.get("boundary_state_override_values") is not None for item in items)
    has_time = any(item.get("time") is not None for item in items)
    # Stores the boundary mask for use by the remaining steps in this flow.
    boundary_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool, device=device) if has_boundary_mask else None
    boundary_codes = torch.zeros((batch_size, max_nodes), dtype=torch.long, device=device) if has_boundary_codes else None

    target_channels = int(targets[0].shape[-1])
    override_mask = torch.zeros((batch_size, max_nodes, target_channels), dtype=torch.bool, device=device) if has_override_mask else None
    # Stores the `override_values` for use by the remaining steps in this flow.
    override_values = torch.zeros((batch_size, max_nodes, target_channels), dtype=targets[0].dtype, device=device) if has_override_values else None
    time = torch.zeros((batch_size,), dtype=dtype, device=device) if has_time else None
    target_shape = (
        (batch_size, max_nodes, target_channels)
        # Continues the surrounding statement or collection definition.
        if target_ndim == 2
        else (batch_size, int(targets[0].shape[0]), max_nodes, target_channels)
    )
    target_batch = torch.zeros(target_shape, dtype=targets[0].dtype, device=targets[0].device)

    # Visits each item in `enumerate(zip(items, targets, node_counts))` and applies the current processing step.
    for index, (item, target, node_count) in enumerate(zip(items, targets, node_counts)):
        if target_ndim == 2 and target.shape[0] != node_count:
            raise ValueError("Transolver 稳态目标节点数与输入不一致")
        if target_ndim == 3 and target.shape[1] != node_count:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("Transolver 时序目标节点数与输入不一致")
        x[index, :node_count] = item["x"]
        pos[index, :node_count] = item["pos"]
        valid_mask[index, :node_count] = True
        # Checks `target_ndim == 2` before entering this conditional branch.
        if target_ndim == 2:
            target_batch[index, :node_count] = target
        else:
            target_batch[index, :, :node_count] = target
        # Reads `sample_id` from the mapping into the `raw_sample_id` with its declared fallback.
        raw_sample_id = item.get("sample_id", 0)
        sample_ids[index] = int(raw_sample_id.item()) if isinstance(raw_sample_id, torch.Tensor) else int(raw_sample_id)
        if boundary_mask is not None:
            _copy_optional_1d(boundary_mask, item, "boundary_mask", index, node_count)
        # Checks `boundary_codes is not None` before entering this conditional branch.
        if boundary_codes is not None:
            _copy_optional_1d(boundary_codes, item, "boundary_entity_codes", index, node_count)
        if override_mask is not None:
            _copy_optional_2d(override_mask, item, "boundary_state_override_mask", index, node_count)
        # Checks `override_values is not None` before entering this conditional branch.
        if override_values is not None:
            _copy_optional_2d(override_values, item, "boundary_state_override_values", index, node_count)
        if time is not None:
            raw_time = item.get("time")
            # Validates `raw_time is None` and raises `ValueError` when the input cannot be processed safely.
            if raw_time is None:
                raise ValueError("Transolver batch 中启用时间嵌入时每个样本都必须提供 time")
            time_value = torch.as_tensor(raw_time, dtype=dtype, device=device)
            if time_value.numel() != 1:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError("Transolver 单样本 time 必须为标量")
            time[index] = time_value.reshape(())

    return TransolverBatch(
        x=x,
        # Completes the multi-line `collection` arguments or elements.
        pos=pos,
        valid_mask=valid_mask,
        node_counts=torch.tensor(node_counts, dtype=torch.long, device=device),
        sample_id=sample_ids,
        # Completes the multi-line `collection` arguments or elements.
        boundary_mask=boundary_mask,
        boundary_entity_codes=boundary_codes,
        boundary_state_override_mask=override_mask,
        boundary_state_override_values=override_values,
        # Completes the multi-line `collection` arguments or elements.
        time=time,
    ), target_batch


def collate_gcn_batch(batch: List[Tuple]) -> Tuple:
    """GCN模型的自定义collate函数
    
    x和y都按第0维（节点维度）拼接，保持一致性。
    
    Args:
        batch: List[Tuple[PyGData, Tensor]]
        
    Returns:
        Tuple[Batch, Tensor]
            - batch_x: PyG Batch对象
            - batch_y: Tensor (total_nodes, output_dim)
    """
    if not batch:
        # Raises `ValueError` to report why this path cannot continue.
        raise ValueError("Batch不能为空")
    
    x_list = []
    y_list = []
    y_ndim = None
    # Stores the `node_dim` for use by the remaining steps in this flow.
    node_dim = None
    
    for x, y in batch:
        if not isinstance(x, PyGData):
            raise TypeError(f"期望PyGData类型，得到 {type(x)}")
        # Validates `not isinstance(y, torch.Tensor)` and raises `TypeError` when the input cannot be processed safely.
        if not isinstance(y, torch.Tensor):
            raise TypeError(f"期望Tensor类型，得到 {type(y)}")

        if y_ndim is None:
            y_ndim = y.ndim
        # Validates `y.ndim != y_ndim` and raises `ValueError` when the input cannot be processed safely.
        elif y.ndim != y_ndim:
            raise ValueError(f"GCN batch中y维度不一致: 期望{y_ndim}，得到{y.ndim}")

        # 对时序场景自动识别节点维：支持 (num_nodes, m) / (num_nodes, t, m) / (t, num_nodes, m)
        if y.ndim == 2:
            current_node_dim = 0
        # Checks `y.ndim == 3` before entering this conditional branch.
        elif y.ndim == 3:
            if y.shape[0] == x.num_nodes:
                current_node_dim = 0
            elif y.shape[1] == x.num_nodes:
                # Stores the `current_node_dim` for use by the remaining steps in this flow.
                current_node_dim = 1
            else:
                raise ValueError(
                    f"无法识别y中的节点维，y.shape={tuple(y.shape)}, num_nodes={x.num_nodes}"
                # Completes the multi-line `ValueError` arguments or elements.
                )
        else:
            raise ValueError(f"当前仅支持y为2维或3维，得到{y.ndim}维")

        if node_dim is None:
            # Stores the `node_dim` for use by the remaining steps in this flow.
            node_dim = current_node_dim
        elif current_node_dim != node_dim:
            raise ValueError(
                f"GCN batch中样本的节点维不一致: 期望node_dim={node_dim}，得到{current_node_dim}"
            # Completes the multi-line `ValueError` arguments or elements.
            )

        x_list.append(x)
        y_list.append(y)
    
    # x: PyG Batch合并（自动按node维度拼接）
    batch_x = Batch.from_data_list(x_list)
    
    # y: 按节点维拼接（稳态/时序统一）
    # - 稳态: (num_nodes, m) -> dim=0
    # - 时序: (num_nodes, t, m) -> dim=0 或 (t, num_nodes, m) -> dim=1
    # Stores the `batch_y` for use by the remaining steps in this flow.
    batch_y = torch.cat(y_list, dim=node_dim)

    # 将监督目标挂载到图对象，便于模型内部进行残差统计等监督相关逻辑
    batch_x.y = batch_y
    
    
    return batch_x, batch_y


def collate_graph_partition_logical_batch(batch: List[List[Tuple]]) -> List[List[Tuple]]:
    """保留完整图到 owner + halo 分块的层级，供 Trainer 统一执行一次优化。"""
    # Validates `not batch` and raises `ValueError` when the input cannot be processed safely.
    if not batch:
        raise ValueError("图分块逻辑 batch 不能为空")
    if any(not isinstance(sample, list) or not sample for sample in batch):
        raise TypeError("图分块数据集必须为每个完整图返回非空分块列表")
    # Returns `batch` to the caller for the next stage.
    return batch


def infer_data_kind(dataset) -> str:
    """Infer whether the dataset yields graph samples or dense tensors."""
    data_kind = getattr(dataset, 'data_kind', None)
    if data_kind in {'graph', 'tensor', 'set'}:
        # Returns `data_kind` to the caller for the next stage.
        return data_kind

    model_type = getattr(dataset, 'model_type', '').lower()
    if model_type in ['gcn', 'graph_unet', 'graph_transformer']:
        return 'graph'
    # Returns early when `model_type in ['transolver', 'linearno']` makes the remaining work inapplicable.
    if model_type in ['transolver', 'linearno']:
        return 'set'

    return 'tensor'


class DataLoader:
    """数据加载器
    
    用于批量加载数据，支持并行处理。
    根据模型类型选择合适的dataloader：对于mlp和cnn使用pytorch的dataloader，对于gcn使用pyg的dataloader。
    """
    
    # 初始化数据加载器
    def __init__(self, 
                 dataset, 
                 batch_size: int = 32, 
                 shuffle: bool = True, 
                 # Continues the surrounding statement or collection definition.
                 num_workers: int = 0,
                 generator: Optional[torch.Generator] = None):
        """初始化数据加载器
        
        Args:
            dataset: 数据集对象
            batch_size: 批量大小
            shuffle: 是否打乱数据
            num_workers: 并行加载的线程数
        """
        self.dataset = dataset
        self.batch_size = batch_size
        # Stores the `shuffle` on the instance so other methods can reuse it.
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.generator = generator
        
        # 根据模型类型选择合适的dataloader
        if getattr(dataset, "sample_view", None) == "graph_partition":
            # Stores the `dataloader` on the instance so other methods can reuse it.
            self.dataloader = torch_data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                # Completes the multi-line `torch_data.DataLoader` arguments or elements.
                num_workers=num_workers,
                collate_fn=collate_graph_partition_logical_batch,
                generator=generator,
            )
        # Checks `infer_data_kind(dataset) == 'graph'` before entering this conditional branch.
        elif infer_data_kind(dataset) == 'graph':
            # 对于GCN模型，DataLoader + 自定义collate_fn
            self.dataloader = torch_data.DataLoader(
                dataset, 
                batch_size=batch_size, 
                # Completes the multi-line `torch_data.DataLoader` arguments or elements.
                shuffle=shuffle, 
                num_workers=num_workers,
                collate_fn=collate_gcn_batch,
                generator=generator,
            # Completes the multi-line `torch_data.DataLoader` arguments or elements.
            )
        elif infer_data_kind(dataset) == 'set':
            self.dataloader = torch_data.DataLoader(
                dataset,
                # Completes the multi-line `torch_data.DataLoader` arguments or elements.
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=collate_transolver_batch,
                # Completes the multi-line `torch_data.DataLoader` arguments or elements.
                generator=generator,
            )
        else:
            # 其他模型，使用PyTorch的原生DataLoader
            self.dataloader = torch_data.DataLoader(
                # Completes the multi-line `torch_data.DataLoader` arguments or elements.
                dataset, 
                batch_size=batch_size, 
                shuffle=shuffle, 
                num_workers=num_workers,
                # Completes the multi-line `torch_data.DataLoader` arguments or elements.
                generator=generator,
            )
    

    
    def __iter__(self) -> Iterable:
        """返回数据加载器迭代器
        
        Returns:
            数据加载器迭代器
        """
        return iter(self.dataloader)
    
    # 返回数据加载器长度
    def __len__(self) -> int:
        """返回数据加载器长度
        
        Returns:
            批次数目
        """
        return len(self.dataloader)
    
    def get_dataset(self):
        """获取数据集
        
        Returns:
            数据集对象
        """
        return self.dataset
