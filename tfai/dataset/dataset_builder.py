from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from tfai.dataset.full_sample_dataset import FullSampleDataset
# Imports `DiskGraphPartitionDataset, build_graph_partition_preprocess_signature, ensure_graph_partition_artifacts` from `tfai.dataset.graph_partition_artifacts` for use in this module.
from tfai.dataset.graph_partition_artifacts import (
    DiskGraphPartitionDataset,
    build_graph_partition_preprocess_signature,
    ensure_graph_partition_artifacts,
    # Continues the surrounding statement or collection definition.
    validate_graph_partition_manifest,
)
from tfai.dataset.graph_partition_dataset import GraphPartitionDataset


def build_base_dataset(
    # Continues the surrounding statement or collection definition.
    config: Dict,
    csv_path: str,
    data_dir: str,
    processor: Optional[Any] = None,
    # Continues the surrounding statement or collection definition.
    phase: str = "train",
):
    """Build the base full-sample dataset from processed files."""
    return FullSampleDataset.from_csv(
        csv_path=csv_path,
        # Completes the multi-line `FullSampleDataset.from_csv` arguments or elements.
        config=config,
        data_dir=data_dir,
        processor=processor,
        phase=phase,
    # Completes the multi-line `FullSampleDataset.from_csv` arguments or elements.
    )


def _resolve_processor_type(processor_type: Optional[str], processor: Optional[Any]) -> str:
    if processor is not None:
        return type(processor).__name__
    # Validates `processor_type is None` and raises `ValueError` when the input cannot be processed safely.
    if processor_type is None:
        raise ValueError("build_dataset requires either processor or processor_type")
    return processor_type


def should_use_graph_partition(
    # Continues the surrounding statement or collection definition.
    config: Dict,
    processor_type: Optional[str],
    processor: Optional[Any] = None,
) -> bool:
    """Return whether graph partition training should be enabled for the current processor."""
    # Reads `training` from the mapping into the `training_cfg` with its declared fallback.
    training_cfg = config.get("training", {})
    if not bool(training_cfg.get("graph_use_partition", False)):
        return False

    resolved_processor_type = _resolve_processor_type(processor_type, processor)
    # Returns early when `processor is not None` makes the remaining work inapplicable.
    if processor is not None:
        return bool(processor.supports_partition_training())
    return resolved_processor_type == "GraphProcessor"


def build_dataset(
    # Continues the surrounding statement or collection definition.
    config: Dict,
    processor_type: Optional[str],
    csv_path: str,
    data_dir: str,
    # Continues the surrounding statement or collection definition.
    phase: str,
    processor: Optional[Any] = None,
):
    """Compose dataset views for the requested phase."""
    dataset = build_partition_source_dataset(
        # Completes the multi-line `build_partition_source_dataset` arguments or elements.
        config=config,
        processor_type=processor_type,
        csv_path=csv_path,
        data_dir=data_dir,
        # Completes the multi-line `build_partition_source_dataset` arguments or elements.
        phase=phase,
        processor=processor,
    )

    if phase != "train":
        # Returns `dataset` to the caller for the next stage.
        return dataset

    resolved_processor_type = _resolve_processor_type(processor_type, processor)
    graph_use_partition = should_use_graph_partition(
        config=config,
        # Completes the multi-line `should_use_graph_partition` arguments or elements.
        processor_type=resolved_processor_type,
        processor=processor,
    )
    if not graph_use_partition:
        # Returns `dataset` to the caller for the next stage.
        return dataset

    rollout_cfg = config.get("training", {}).get("temporal_rollout", {}) or {}
    task_cfg = config.get("task", {}) or {}
    is_temporal_autoregressive = (
        # Completes the multi-line `str(task_cfg.get('mode', '')).lower` arguments or elements.
        str(task_cfg.get("mode", "")).lower() == "temporal"
        and str(task_cfg.get("temporal_strategy", "direct")).lower() == "autoregressive"
    )
    # 状态回写规则在 Trainer 启动后由训练集拟合；离线子图早于该时刻生成，无法安全携带该规则。
    if is_temporal_autoregressive and bool(rollout_cfg.get("boundary_state_override", False)):
        # Stores the `partition_dataset` for use by the remaining steps in this flow.
        partition_dataset = GraphPartitionDataset(base_dataset=dataset, config=config)
        setattr(
            partition_dataset,
            "temporal_direct_total_time_steps",
            # Completes the multi-line `getattr` arguments or elements.
            getattr(dataset, "temporal_direct_total_time_steps", None),
        )
        return partition_dataset

    partition_count = int(config.get("training", {}).get("graph_partition_count", 1) or 1)
    # Stores the `preprocess_signature` for use by the remaining steps in this flow.
    preprocess_signature = build_graph_partition_preprocess_signature(
        config=config,
        processor_type=resolved_processor_type,
    )
    # Initializes the `source_items` as a collection for values from this processing stage.
    source_items = []
    if hasattr(dataset, "get_sample_meta"):
        for source_index in range(len(dataset)):
            source_meta = dataset.get_sample_meta(source_index) or {}
            # Adds the current value to `source_items` for later aggregation.
            source_items.append(
                {
                    "source_index": int(source_index),
                    "sample_id": source_meta.get("sample_id"),
                    # Completes the multi-line `source_meta.get` arguments or elements.
                    "base_index": source_meta.get("base_index"),
                    "start_t": source_meta.get("start_t"),
                    "file_path": source_meta.get("file_path"),
                    "sample_view": source_meta.get("sample_view"),
                # Completes the multi-line `collection` arguments or elements.
                }
            )
    manifest_ok, _, _ = validate_graph_partition_manifest(
        processed_dir=data_dir,
        # Completes the multi-line `validate_graph_partition_manifest` arguments or elements.
        partition_count=partition_count,
        source_items=source_items,
        preprocess_signature=preprocess_signature,
    )
    # Checks `manifest_ok` before entering this conditional branch.
    if manifest_ok:
        partition_dataset = DiskGraphPartitionDataset.from_manifest(
            config=config,
            processed_dir=data_dir,
            # Completes the multi-line `DiskGraphPartitionDataset.from_manifest` arguments or elements.
            processor=processor,
        )
        setattr(
            partition_dataset,
            # Completes the multi-line `setattr` arguments or elements.
            "temporal_direct_total_time_steps",
            getattr(dataset, "temporal_direct_total_time_steps", None),
        )
        return partition_dataset

    # Stores the `partition_dataset` for use by the remaining steps in this flow.
    partition_dataset = GraphPartitionDataset(base_dataset=dataset, config=config)
    setattr(
        partition_dataset,
        "temporal_direct_total_time_steps",
        # Completes the multi-line `getattr` arguments or elements.
        getattr(dataset, "temporal_direct_total_time_steps", None),
    )
    return partition_dataset


def _validate_temporal_direct_dataset_steps(
    # Continues the surrounding statement or collection definition.
    config: Dict,
    datasets: Tuple[object, ...],
) -> None:
    task_cfg = config.get("task", {})
    # Returns early when `str(task_cfg.get("mode", "")).lower() != "temporal"` makes the remaining work inapplicable.
    if str(task_cfg.get("mode", "")).lower() != "temporal":
        return
    if str(task_cfg.get("temporal_strategy", "")).lower() != "direct":
        return

    # Stores the `expected_steps` for use by the remaining steps in this flow.
    expected_steps = None
    for dataset in datasets:
        total_steps = getattr(dataset, "temporal_direct_total_time_steps", None)
        if total_steps is None:
            # Handles the data or control-flow case represented by this statement.
            continue
        if expected_steps is None:
            expected_steps = int(total_steps)
            continue
        # Validates `int(total_steps) != int(expected_steps)` and raises `ValueError` when the input cannot be processed safely.
        if int(total_steps) != int(expected_steps):
            raise ValueError(
                "时序 direct 模式要求所有训练/验证/测试样本时间步数一致："
                f"got={total_steps}, expected={expected_steps}"
            # Completes the multi-line `ValueError` arguments or elements.
            )


def build_partition_source_dataset(
    config: Dict,
    processor_type: Optional[str],
    # Continues the surrounding statement or collection definition.
    csv_path: str,
    data_dir: str,
    phase: str,
    processor: Optional[Any] = None,
# Continues the surrounding statement or collection definition.
):
    """Build the training/evaluation source view before graph partition expansion."""
    dataset = build_base_dataset(
        config=config,
        csv_path=csv_path,
        # Completes the multi-line `build_base_dataset` arguments or elements.
        data_dir=data_dir,
        processor=processor,
        phase=phase,
    )

    # Returns `dataset` to the caller for the next stage.
    return dataset


def ensure_train_graph_partition_artifacts(
    config: Dict,
    processor_type: Optional[str],
    # Continues the surrounding statement or collection definition.
    train_csv: str,
    processed_dir: str,
    processor: Optional[Any] = None,
    logger=None,
# Completes the multi-line `collection` arguments or elements.
) -> Optional[Dict[str, Any]]:
    """Generate and validate offline graph partition artifacts for the training split."""
    resolved_processor_type = _resolve_processor_type(processor_type, processor)
    graph_use_partition = should_use_graph_partition(
        config=config,
        # Completes the multi-line `should_use_graph_partition` arguments or elements.
        processor_type=resolved_processor_type,
        processor=processor,
    )
    if not graph_use_partition:
        # Returns `None` to the caller for the next stage.
        return None
    rollout_cfg = config.get("training", {}).get("temporal_rollout", {}) or {}
    task_cfg = config.get("task", {}) or {}
    is_temporal_autoregressive = (
        # Completes the multi-line `str(task_cfg.get('mode', '')).lower` arguments or elements.
        str(task_cfg.get("mode", "")).lower() == "temporal"
        and str(task_cfg.get("temporal_strategy", "direct")).lower() == "autoregressive"
    )
    if is_temporal_autoregressive and bool(rollout_cfg.get("boundary_state_override", False)):
        # Checks `logger is not None` before entering this conditional branch.
        if logger is not None:
            logger.info("boundary_state_override 需要训练启动后拟合边界规则，跳过离线图分块产物")
        return {
            "built": False,
            # Completes the multi-line `collection` arguments or elements.
            "status": "boundary_state_override 使用训练时动态图分块视图",
        }

    source_dataset = build_partition_source_dataset(
        config=config,
        # Completes the multi-line `build_partition_source_dataset` arguments or elements.
        processor_type=resolved_processor_type,
        csv_path=train_csv,
        data_dir=_resolve_phase_data_dir(config, processed_dir, "train"),
        phase="train",
        # Completes the multi-line `build_partition_source_dataset` arguments or elements.
        processor=processor,
    )
    return ensure_graph_partition_artifacts(
        source_dataset=source_dataset,
        # Completes the multi-line `ensure_graph_partition_artifacts` arguments or elements.
        config=config,
        processed_dir=_resolve_phase_data_dir(config, processed_dir, "train"),
        processor_type=resolved_processor_type,
        logger=logger,
    # Completes the multi-line `ensure_graph_partition_artifacts` arguments or elements.
    )


def _resolve_phase_data_dir(config: Dict, processed_dir: str, phase: str) -> str:
    task_type = config["task"]["type"].lower()
    if task_type == "scalar" and phase in {"train", "valid", "test"}:
        # Returns `str(Path(processed_dir) / phase)` to the caller for the next stage.
        return str(Path(processed_dir) / phase)
    return processed_dir


def build_phase_datasets(
    config: Dict,
    # Continues the surrounding statement or collection definition.
    processor_type: Optional[str],
    train_csv: str,
    valid_csv: str,
    test_csv: str,
    # Continues the surrounding statement or collection definition.
    processed_dir: str,
    processor: Optional[Any] = None,
) -> Tuple[object, object, object]:
    """Build train/valid/test datasets with phase-aware composition."""
    train_dataset = build_dataset(
        # Completes the multi-line `build_dataset` arguments or elements.
        config=config,
        processor_type=processor_type,
        csv_path=train_csv,
        data_dir=_resolve_phase_data_dir(config, processed_dir, "train"),
        # Completes the multi-line `build_dataset` arguments or elements.
        phase="train",
        processor=processor,
    )
    valid_dataset = build_dataset(
        # Completes the multi-line `build_dataset` arguments or elements.
        config=config,
        processor_type=processor_type,
        csv_path=valid_csv,
        data_dir=_resolve_phase_data_dir(config, processed_dir, "valid"),
        # Completes the multi-line `build_dataset` arguments or elements.
        phase="valid",
        processor=processor,
    )
    test_dataset = build_dataset(
        # Completes the multi-line `build_dataset` arguments or elements.
        config=config,
        processor_type=processor_type,
        csv_path=test_csv,
        data_dir=_resolve_phase_data_dir(config, processed_dir, "test"),
        # Completes the multi-line `build_dataset` arguments or elements.
        phase="test",
        processor=processor,
    )
    _validate_temporal_direct_dataset_steps(
        # Completes the multi-line `_validate_temporal_direct_dataset_steps` arguments or elements.
        config=config,
        datasets=(train_dataset, valid_dataset, test_dataset),
    )
    return train_dataset, valid_dataset, test_dataset


# Build train/valid datasets for the standalone training flow.
def build_train_valid_datasets(
    config: Dict,
    processor_type: Optional[str],
    train_csv: str,
    # Continues the surrounding statement or collection definition.
    valid_csv: str,
    processed_dir: str,
    processor: Optional[Any] = None,
) -> Tuple[object, object]:
    """Build train/valid datasets for the standalone training flow."""
    # Stores the `train_dataset` for use by the remaining steps in this flow.
    train_dataset = build_dataset(
        config=config,
        processor_type=processor_type,
        csv_path=train_csv,
        # Completes the multi-line `_resolve_phase_data_dir` arguments or elements.
        data_dir=_resolve_phase_data_dir(config, processed_dir, "train"),
        phase="train",
        processor=processor,
    )
    # Stores the `valid_dataset` for use by the remaining steps in this flow.
    valid_dataset = build_dataset(
        config=config,
        processor_type=processor_type,
        csv_path=valid_csv,
        # Completes the multi-line `_resolve_phase_data_dir` arguments or elements.
        data_dir=_resolve_phase_data_dir(config, processed_dir, "valid"),
        phase="valid",
        processor=processor,
    )
    # Calls `_validate_temporal_direct_dataset_steps` to perform the operation required at this stage.
    _validate_temporal_direct_dataset_steps(
        config=config,
        datasets=(train_dataset, valid_dataset),
    )
    # Returns `train_dataset, valid_dataset` to the caller for the next stage.
    return train_dataset, valid_dataset
