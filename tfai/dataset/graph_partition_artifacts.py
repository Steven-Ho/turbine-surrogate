import logging
import os
import shutil
# Imports `Path` from `pathlib` for use in this module.
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data as PyGData

# Imports `GraphPartitionDataset` from `tfai.dataset.graph_partition_dataset` for use in this module.
from tfai.dataset.graph_partition_dataset import GraphPartitionDataset
from tfai.training.preprocess_cache import build_preprocess_signature
from tfai.utils.torch_utils import load_torch


GRAPH_PARTITION_ARTIFACT_DIRNAME = "train_graph_partitions_v2"
# Stores the `GRAPH_PARTITION_MANIFEST_FILENAME` for use by the remaining steps in this flow.
GRAPH_PARTITION_MANIFEST_FILENAME = "graph_partition_manifest.pt"
GRAPH_PARTITION_FORMAT_VERSION = 2


def get_graph_partition_artifact_dir(processed_dir: str) -> str:
    return str(Path(processed_dir) / GRAPH_PARTITION_ARTIFACT_DIRNAME)


# Defines `get_graph_partition_manifest_path` for get graph partition manifest path logic.
def get_graph_partition_manifest_path(processed_dir: str) -> str:
    return str(Path(get_graph_partition_artifact_dir(processed_dir)) / GRAPH_PARTITION_MANIFEST_FILENAME)


def _build_source_signature_item(source_meta: Dict[str, Any], source_index: int) -> Dict[str, Any]:
    return {
        # Completes the multi-line `int` arguments or elements.
        "source_index": int(source_index),
        "sample_id": source_meta.get("sample_id"),
        "base_index": source_meta.get("base_index"),
        "start_t": source_meta.get("start_t"),
        # Completes the multi-line `source_meta.get` arguments or elements.
        "file_path": source_meta.get("file_path"),
        "sample_view": source_meta.get("sample_view"),
    }


def _normalize_source_signature_list(source_dataset) -> List[Dict[str, Any]]:
    # Initializes the `source_items` as a collection for values from this processing stage.
    source_items: List[Dict[str, Any]] = []
    for source_index in range(len(source_dataset)):
        source_meta = {}
        if hasattr(source_dataset, "get_sample_meta"):
            # Stores the `source_meta` for use by the remaining steps in this flow.
            source_meta = source_dataset.get_sample_meta(source_index) or {}
        source_items.append(_build_source_signature_item(source_meta=source_meta, source_index=source_index))
    return source_items


def build_graph_partition_preprocess_signature(config: Dict, processor_type: str) -> Dict[str, Any]:
    """构建图分块离线产物依赖的预处理签名。"""
    # Stores the `signature` for use by the remaining steps in this flow.
    signature = build_preprocess_signature(config=config, processor_type=processor_type)
    signature["graph_partition_format_version"] = GRAPH_PARTITION_FORMAT_VERSION
    signature["graph_partition_halo_hops"] = GraphPartitionDataset.resolve_halo_hops(config)
    return signature


# Defines `_build_partition_payload` for  build partition payload logic.
def _build_partition_payload(
    subgraph: PyGData,
    subgraph_y: torch.Tensor,
    source_meta: Dict[str, Any],
    # Continues the surrounding statement or collection definition.
    source_index: int,
    partition_id: int,
    partition_count: int,
) -> Dict[str, Any]:
    # Stores the `meta` for use by the remaining steps in this flow.
    meta = dict(source_meta or {})
    meta["sample_view"] = "graph_partition"
    meta["source_sample_view"] = source_meta.get("sample_view", "full")
    meta["source_index"] = int(source_index)
    # Converts the `meta['partition_id']` to `int` so later checks use a consistent type.
    meta["partition_id"] = int(partition_id)
    meta["partition_count"] = int(partition_count)
    meta["base_sample_id"] = source_meta.get("sample_id")
    return {
        # Completes the multi-line `collection` arguments or elements.
        "x": subgraph,
        "y": subgraph_y,
        "meta": meta,
    }


# Defines `load_graph_partition_manifest` for load graph partition manifest logic.
def load_graph_partition_manifest(processed_dir: str) -> Dict[str, Any]:
    manifest_path = get_graph_partition_manifest_path(processed_dir)
    if not os.path.exists(manifest_path):
        return {}
    # Keeps the operation and its recovery path within the same exception boundary.
    try:
        payload = load_torch(manifest_path, weights_only=False)
    except Exception:
        return {}
    # Returns `payload if isinstance(payload, dict) else {}` to the caller for the next stage.
    return payload if isinstance(payload, dict) else {}


def validate_graph_partition_manifest(
    processed_dir: str,
    partition_count: int,
    # Completes the multi-line `collection` arguments or elements.
    source_items: List[Dict[str, Any]],
    preprocess_signature: Dict[str, Any],
) -> Tuple[bool, str, Dict[str, Any]]:
    manifest = load_graph_partition_manifest(processed_dir)
    # Returns early when `not manifest` makes the remaining work inapplicable.
    if not manifest:
        return False, "图分块离线产物缺失或不可读", {}

    if int(manifest.get("format_version", 0) or 0) != GRAPH_PARTITION_FORMAT_VERSION:
        return False, "图分块离线产物格式已更新，需要重新生成", {}

    # Returns early when `` makes the remaining work inapplicable.
    if int(manifest.get("partition_count", 0) or 0) != int(partition_count):
        return False, "图分块离线产物与当前 partition_count 不一致", {}

    manifest_source_items = list(manifest.get("source_items") or [])
    if manifest_source_items != source_items:
        # Returns `False, "图分块离线产物与当前训练样本视图不一致", {}` to the caller for the next stage.
        return False, "图分块离线产物与当前训练样本视图不一致", {}

    manifest_preprocess_signature = manifest.get("preprocess_signature") or {}
    if manifest_preprocess_signature != preprocess_signature:
        return False, "图分块离线产物与当前预处理配置不一致", {}

    # Stores the `sample_entries` for use by the remaining steps in this flow.
    sample_entries = list(manifest.get("sample_entries") or [])
    if not sample_entries:
        return False, "图分块离线产物清单为空", {}

    missing_paths = [
        # Completes the multi-line `entry.get` arguments or elements.
        entry.get("processed_path")
        for entry in sample_entries
        if not entry.get("processed_path") or not Path(str(entry.get("processed_path"))).exists()
    ]
    # Returns early when `missing_paths` makes the remaining work inapplicable.
    if missing_paths:
        return False, "图分块离线产物文件缺失", {}

    return True, "图分块离线产物校验通过", manifest


def save_graph_partition_manifest(
    # Continues the surrounding statement or collection definition.
    processed_dir: str,
    partition_count: int,
    source_items: List[Dict[str, Any]],
    sample_entries: List[Dict[str, Any]],
    # Completes the multi-line `collection` arguments or elements.
    preprocess_signature: Dict[str, Any],
) -> str:
    manifest_path = get_graph_partition_manifest_path(processed_dir)
    payload = {
        # Completes the multi-line `collection` arguments or elements.
        "format_version": GRAPH_PARTITION_FORMAT_VERSION,
        "partition_count": int(partition_count),
        "source_items": source_items,
        "sample_entries": sample_entries,
        # Completes the multi-line `collection` arguments or elements.
        "preprocess_signature": preprocess_signature,
    }
    torch.save(payload, manifest_path)
    return manifest_path


# 从离线落盘的子图产物构建训练数据集。
class DiskGraphPartitionDataset(torch.utils.data.Dataset):
    """从离线落盘的子图产物构建训练数据集。"""

    sample_view = "graph_partition"

    def __init__(
        self,
        # Continues the surrounding statement or collection definition.
        config: Dict,
        sample_entries: List[Dict[str, Any]],
        partition_count: int,
        processor: Optional[Any] = None,
    # Continues the surrounding statement or collection definition.
    ) -> None:
        del config, processor
        self.partition_count = int(partition_count)
        self.data_kind = "graph"
        # Stores the `_groups` on the instance so other methods can reuse it.
        self._groups: Dict[int, List[Dict[str, Any]]] = {}
        for entry in sample_entries:
            source_index = int(entry.get("source_index", -1))
            if source_index < 0:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError("图分块离线产物缺少 source_index")
            self._groups.setdefault(source_index, []).append(dict(entry))
        self._source_indexes = sorted(self._groups)
        for entries in self._groups.values():
            # Calls `entries.sort` to perform the operation required at this stage.
            entries.sort(key=lambda item: int(item.get("partition_id", -1)))
            if len(entries) != self.partition_count:
                raise ValueError("图分块离线产物的分块数量不完整")

    def __len__(self) -> int:
        # Returns `len(self._source_indexes)` to the caller for the next stage.
        return len(self._source_indexes)

    def __getitem__(self, idx: int):
        logical_sample = []
        for entry in self._groups[self._source_indexes[idx]]:
            # Stores the `payload` for use by the remaining steps in this flow.
            payload = load_torch(str(entry["processed_path"]), weights_only=False)
            logical_sample.append((payload["x"], payload["y"]))
        return logical_sample

    def get_sample_meta(self, idx: int) -> Dict[str, Any]:
        # Selects the required item from `self._groups[self._source_indexes[idx]]` as the `entry`.
        entry = self._groups[self._source_indexes[idx]][0]
        return {
            "source_index": self._source_indexes[idx],
            "sample_id": entry.get("base_sample_id"),
            # Completes the multi-line `collection` arguments or elements.
            "sample_view": self.sample_view,
        }

    @classmethod
    def from_manifest(
        # Continues the surrounding statement or collection definition.
        cls,
        config: Dict,
        processed_dir: str,
        processor: Optional[Any] = None,
    # Continues the surrounding statement or collection definition.
    ) -> "DiskGraphPartitionDataset":
        manifest = load_graph_partition_manifest(processed_dir)
        sample_entries = list(manifest.get("sample_entries") or [])
        partition_count = int(manifest.get("partition_count", 1) or 1)
        # Returns `` to the caller for the next stage.
        return cls(
            config=config,
            sample_entries=sample_entries,
            partition_count=partition_count,
            # Completes the multi-line `cls` arguments or elements.
            processor=processor,
        )


def ensure_graph_partition_artifacts(
    source_dataset,
    # Continues the surrounding statement or collection definition.
    config: Dict,
    processed_dir: str,
    processor_type: str,
    logger: Optional[logging.Logger] = None,
# Completes the multi-line `collection` arguments or elements.
) -> Dict[str, Any]:
    training_cfg = config.get("training", {})
    partition_count = int(training_cfg.get("graph_partition_count", 1) or 1)
    if partition_count < 2:
        # Raises `ValueError` to report why this path cannot continue.
        raise ValueError("training.graph_partition_count must be >= 2 when partition training is enabled")

    source_items = _normalize_source_signature_list(source_dataset)
    preprocess_signature = build_graph_partition_preprocess_signature(
        config=config,
        # Completes the multi-line `build_graph_partition_preprocess_signature` arguments or elements.
        processor_type=processor_type,
    )
    manifest_ok, manifest_status, manifest = validate_graph_partition_manifest(
        processed_dir=processed_dir,
        # Completes the multi-line `validate_graph_partition_manifest` arguments or elements.
        partition_count=partition_count,
        source_items=source_items,
        preprocess_signature=preprocess_signature,
    )
    # Returns early when `manifest_ok` makes the remaining work inapplicable.
    if manifest_ok:
        return {
            "built": False,
            "status": manifest_status,
            # Completes the multi-line `len` arguments or elements.
            "partition_sample_count": len(list(manifest.get("sample_entries") or [])),
            "artifact_dir": get_graph_partition_artifact_dir(processed_dir),
            "manifest_path": get_graph_partition_manifest_path(processed_dir),
        }

    # Stores the `artifact_dir` for use by the remaining steps in this flow.
    artifact_dir = Path(get_graph_partition_artifact_dir(processed_dir))
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Checks `logger is not None` before entering this conditional branch.
    if logger is not None:
        logger.info(
            "开始生成离线图分块训练产物: source_samples=%s partition_count=%s",
            len(source_dataset),
            # Completes the multi-line `logger.info` arguments or elements.
            partition_count,
        )

    sample_entries: List[Dict[str, Any]] = []
    partition_sample_id = 0

    # Visits each item in `range(len(source_dataset))` and applies the current processing step.
    for source_index in range(len(source_dataset)):
        graph, y = source_dataset[source_index]
        if not isinstance(graph, PyGData):
            raise TypeError(f"离线图分块要求训练样本输出 PyG Data，当前得到 {type(graph)}")

        # Initializes the `source_meta` as a collection for values from this processing stage.
        source_meta = {}
        if hasattr(source_dataset, "get_sample_meta"):
            source_meta = source_dataset.get_sample_meta(source_index) or {}

        partition_groups = GraphPartitionDataset.build_partition_groups(
            # Completes the multi-line `GraphPartitionDataset.build_partition_groups` arguments or elements.
            graph=graph,
            partition_count=partition_count,
        )
        halo_hops = GraphPartitionDataset.resolve_halo_hops(config)
        # Visits each item in `enumerate(partition_groups)` and applies the current processing step.
        for partition_id, owner_node_ids in enumerate(partition_groups):
            subgraph, subgraph_y = GraphPartitionDataset._extract_partition_subgraph(
                graph=graph,
                y=y,
                # Completes the multi-line `GraphPartitionDataset._extract_partition_subgraph` arguments or elements.
                owner_node_ids=owner_node_ids,
                halo_hops=halo_hops,
                partition_id=partition_id,
            )
            # Derives the `partition_path` from the current values for downstream processing.
            partition_path = artifact_dir / f"partition_{source_index}_{partition_id}.pt"
            payload = _build_partition_payload(
                subgraph=subgraph,
                subgraph_y=subgraph_y,
                # Completes the multi-line `_build_partition_payload` arguments or elements.
                source_meta=source_meta,
                source_index=source_index,
                partition_id=partition_id,
                partition_count=partition_count,
            # Completes the multi-line `_build_partition_payload` arguments or elements.
            )
            torch.save(payload, partition_path)
            sample_entries.append(
                {
                    # Completes the multi-line `int` arguments or elements.
                    "sample_id": int(partition_sample_id),
                    "processed_path": str(partition_path),
                    "source_file": source_meta.get("file_path"),
                    "source_index": int(source_index),
                    # Completes the multi-line `int` arguments or elements.
                    "partition_id": int(partition_id),
                    "base_sample_id": source_meta.get("sample_id"),
                }
            )
            # Stores the `partition_sample_id` for use by the remaining steps in this flow.
            partition_sample_id += 1

    manifest_path = save_graph_partition_manifest(
        processed_dir=processed_dir,
        partition_count=partition_count,
        # Completes the multi-line `save_graph_partition_manifest` arguments or elements.
        source_items=source_items,
        sample_entries=sample_entries,
        preprocess_signature=preprocess_signature,
    )

    # Checks `logger is not None` before entering this conditional branch.
    if logger is not None:
        logger.info(
            "离线图分块训练产物已生成: partition_samples=%s artifact_dir=%s",
            len(sample_entries),
            # Completes the multi-line `str` arguments or elements.
            str(artifact_dir),
        )

    return {
        "built": True,
        # Completes the multi-line `collection` arguments or elements.
        "status": "离线图分块训练产物已生成",
        "partition_sample_count": len(sample_entries),
        "artifact_dir": str(artifact_dir),
        "manifest_path": manifest_path,
    # Completes the multi-line `collection` arguments or elements.
    }
