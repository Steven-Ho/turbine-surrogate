import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Imports `torch` for reuse in this module.
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as PyGData

from tfai.readers.sample_map import SampleMap
# Imports `load_torch` from `tfai.utils.torch_utils` for use in this module.
from tfai.utils.torch_utils import load_torch


def is_pyg_data(obj: Any) -> bool:
    """Return whether the payload behaves like a PyG graph sample."""
    return isinstance(obj, PyGData) or (hasattr(obj, "x") and hasattr(obj, "edge_index"))


def is_set_data(obj: Any) -> bool:
    """判断处理后载荷是否为 Transolver 系列的点集输入。"""
    # Returns `` to the caller for the next stage.
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("x"), torch.Tensor)
        and isinstance(obj.get("pos"), torch.Tensor)
    # Continues the surrounding statement or collection definition.
    )


def _coerce_positive_int(value: Any) -> Optional[int]:
    """将配置或元数据中的时间步数规范为正整数。"""
    if value is None:
        return None
    # Keeps the operation and its recovery path within the same exception boundary.
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    # Returns `resolved if resolved > 0 else None` to the caller for the next stage.
    return resolved if resolved > 0 else None


class FullSampleDataset(Dataset):
    """Dataset that loads one processed `.pt` file as one full sample view."""

    sample_view = "full"

    def __init__(
        # Continues the surrounding statement or collection definition.
        self,
        config: Dict,
        csv_path: Optional[str],
        data_dir: Optional[str],
        # Completes the multi-line `collection` arguments or elements.
        sample_entries: Optional[List[Dict[str, Any]]] = None,
        processor: Optional[Any] = None,
        phase: str = "train",
    ) -> None:
        # Stores the runtime configuration on the instance so other methods can reuse it.
        self.config = config
        self.csv_path = csv_path or ""
        self.data_dir = data_dir or ""
        self.sample_entries = sample_entries or []
        # Stores the data processor on the instance so other methods can reuse it.
        self.processor = processor
        self.phase = str(phase).lower()
        self.temporal_direct_total_time_steps: Optional[int] = None

        model_cfg = config["model"]
        # Selects the required item from `config` as the task settings.
        task_cfg = config["task"]

        self.model_type = model_cfg["type"].lower()
        self.mode = task_cfg["mode"].lower()
        self.temporal_strategy = task_cfg["temporal_strategy"].lower()
        # Stores the `output_length` on the instance so other methods can reuse it.
        self.output_length = int(task_cfg["output_length"])
        self.task_type = task_cfg["type"].lower()

        if self.sample_entries:
            self.sample_ids = [int(entry["sample_id"]) for entry in self.sample_entries]
            # Stores the sample file paths on the instance so other methods can reuse it.
            self.file_paths = [str(entry["processed_path"]) for entry in self.sample_entries]
            if not self.file_paths:
                raise ValueError("No processed samples were provided")
        elif self.task_type == "scalar":
            # Stores the sample file paths on the instance so other methods can reuse it.
            self.file_paths = sorted(
                str(path) for path in Path(self.data_dir).glob("sample_*.pt")
            )
            if not self.file_paths:
                # Raises `ValueError` to report why this path cannot continue.
                raise ValueError(f"No processed scalar samples found in {self.data_dir}")

            self.sample_ids = []
            for file_path in self.file_paths:
                file_name = Path(file_path).name
                # Keeps the operation and its recovery path within the same exception boundary.
                try:
                    sample_id = int(file_name[len("sample_") : -3])
                except ValueError:
                    sample_id = len(self.sample_ids)
                # Adds the current value to `self.sample_ids` for later aggregation.
                self.sample_ids.append(sample_id)
        else:
            sample_map = SampleMap(self.csv_path)
            self.sample_ids = sample_map.get_all_samples()
            # Validates `not self.sample_ids` and raises `ValueError` when the input cannot be processed safely.
            if not self.sample_ids:
                raise ValueError(f"No samples found in {self.csv_path}")

            self.file_paths = [
                os.path.join(self.data_dir, f"sample_{sample_id}.pt")
                # Continues the surrounding statement or collection definition.
                for sample_id in self.sample_ids
            ]

        first_payload = self._load_processed_payload(self.file_paths[0])
        first_x = first_payload["x"]
        # Stores the `data_kind` on the instance so other methods can reuse it.
        self.data_kind = (
            "graph" if is_pyg_data(first_x) else "set" if is_set_data(first_x) else "tensor"
        )
        self._validate_temporal_direct_time_steps(first_payload)

    # Continues the surrounding statement or collection definition.
    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        # Continues the surrounding statement or collection definition.
        config: Dict,
        data_dir: str,
        processor: Optional[Any] = None,
        phase: str = "train",
    # Continues the surrounding statement or collection definition.
    ) -> "FullSampleDataset":
        return cls(
            config=config,
            csv_path=csv_path,
            # Completes the multi-line `cls` arguments or elements.
            data_dir=data_dir,
            processor=processor,
            phase=phase,
        )

    # Continues the surrounding statement or collection definition.
    @classmethod
    def from_sample_entries(
        cls,
        config: Dict,
        # Completes the multi-line `collection` arguments or elements.
        sample_entries: List[Dict[str, Any]],
        processor: Optional[Any] = None,
        phase: str = "train",
    ) -> "FullSampleDataset":
        # Returns `` to the caller for the next stage.
        return cls(
            config=config,
            csv_path="",
            data_dir="",
            # Completes the multi-line `cls` arguments or elements.
            sample_entries=sample_entries,
            processor=processor,
            phase=phase,
        )

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _infer_sample_id_from_path(file_path: str) -> int:
        file_name = Path(file_path).name
        if file_name.startswith("sample_") and file_name.endswith(".pt"):
            # Selects the required item from `file_name` as the `raw_id`.
            raw_id = file_name[len("sample_") : -3]
            try:
                return int(raw_id)
            except ValueError:
                # Returns `0` to the caller for the next stage.
                return 0
        return 0

    @classmethod
    def _normalize_payload(
        # Continues the surrounding statement or collection definition.
        cls,
        file_path: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Validates `"x" not in payload` and raises `ValueError` when the input cannot be processed safely.
        if "x" not in payload:
            raise ValueError(f"Invalid processed sample format: {file_path}")
        x = payload["x"]

        if "y" not in payload:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(f"Invalid processed sample format: {file_path}")
        y = payload["y"]

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
        # Continues the surrounding statement or collection definition.
        else:
            y = y.float()

        meta = payload.get("meta", {})
        if not isinstance(meta, dict):
            # Initializes the `meta` as a collection for values from this processing stage.
            meta = {}
        meta = dict(meta)
        meta.setdefault("sample_id", cls._infer_sample_id_from_path(file_path))
        meta.setdefault("processor_type", None)
        # Calls `meta.setdefault` to perform the operation required at this stage.
        meta.setdefault("is_temporal", False)
        meta.setdefault("total_time_steps", None)
        meta.setdefault("time_axis", None)
        normalized = dict(payload)
        # Stores the `normalized['x']` for use by the remaining steps in this flow.
        normalized["x"] = x
        normalized["y"] = y
        normalized["meta"] = meta
        if "y_seed" in normalized:
            # Selects the required item from `normalized` as the `y_seed`.
            y_seed = normalized["y_seed"]
            if not isinstance(y_seed, torch.Tensor):
                y_seed = torch.tensor(y_seed, dtype=torch.float32)
            else:
                # Stores the `y_seed` for use by the remaining steps in this flow.
                y_seed = y_seed.float()
            normalized["y_seed"] = y_seed
        return normalized

    @classmethod
    # Defines `_load_processed_payload` for  load processed payload logic.
    def _load_processed_payload(cls, file_path: str) -> Dict[str, Any]:
        payload = load_torch(file_path, weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid processed sample format: {file_path}")
        # Returns `cls._normalize_payload(file_path=file_path, payload=payload)` to the caller for the next stage.
        return cls._normalize_payload(file_path=file_path, payload=payload)

    @staticmethod
    def infer_payload_total_time_steps(payload: Dict[str, Any]) -> Optional[int]:
        """从处理后样本中推断真实时序步数。"""
        meta = payload.get("meta", {})
        # Checks `not isinstance(meta, dict)` before entering this conditional branch.
        if not isinstance(meta, dict):
            meta = {}

        total_steps = _coerce_positive_int(meta.get("total_time_steps"))
        if total_steps is not None:
            # Returns `total_steps` to the caller for the next stage.
            return total_steps

        y_seed = payload.get("y_seed")
        if y_seed is not None and hasattr(y_seed, "shape") and len(y_seed.shape) > 0:
            return int(y_seed.shape[0])

        # Reads `y` from the mapping into the `y`.
        y = payload.get("y")
        time_axis = meta.get("time_axis")
        if time_axis is None or not hasattr(y, "shape"):
            return None

        # Keeps the operation and its recovery path within the same exception boundary.
        try:
            axis = int(time_axis)
            return int(y.shape[axis])
        except (IndexError, TypeError, ValueError):
            # Returns `None` to the caller for the next stage.
            return None

    def _validate_temporal_direct_time_steps(self, first_payload: Dict[str, Any]) -> None:
        """校验 direct 时序任务的样本时间步数一致性。"""
        if self.phase == "predict":
            return
        # Returns early when `self.mode != "temporal" or self.temporal_strategy != "direct"` makes the remaining work inapplicable.
        if self.mode != "temporal" or self.temporal_strategy != "direct":
            return

        expected_steps = self.infer_payload_total_time_steps(first_payload)
        if expected_steps is None:
            # 兼容旧版处理样本：缺少时间轴元数据时无法可靠推断时间维，跳过一致性校验。
            # Ends this function and returns control to its caller.
            return
        self.temporal_direct_total_time_steps = int(expected_steps)

        for index, file_path in enumerate(self.file_paths[1:], start=1):
            payload = self._load_processed_payload(file_path)
            # Stores the `total_steps` for use by the remaining steps in this flow.
            total_steps = self.infer_payload_total_time_steps(payload)
            if total_steps is None:
                # 同一数据集混入旧版样本时无法确认时间步一致性，保留原有训练兼容性。
                return
            if int(total_steps) != int(expected_steps):
                # Reads `sample_id` from the mapping into the sample identifier with its declared fallback.
                sample_id = payload.get("meta", {}).get("sample_id", self.sample_ids[index])
                raise ValueError(
                    "时序 direct 模式要求所有样本时间步数一致："
                    f"sample_id={sample_id}, got={total_steps}, expected={expected_steps}"
                # Completes the multi-line `ValueError` arguments or elements.
                )

    def get_processed_sample(self, idx: int) -> Dict[str, Any]:
        file_path = self.file_paths[idx]
        return self._load_processed_payload(file_path)

    # Defines `__len__` for   len   logic.
    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[Any, torch.Tensor]:
        payload = self.get_processed_sample(idx)
        # Selects the required item from `payload` as the `x`.
        x = payload["x"]
        y = payload["y"]
        if (
            self.processor is not None
            # Continues the surrounding statement or collection definition.
            and self.mode == "temporal"
            and self.temporal_strategy == "direct"
        ):
            prediction_view = payload.get("meta", {}).get("prediction_view")
            # Checks `prediction_view == "direct_input_only"` before entering this conditional branch.
            if prediction_view == "direct_input_only":
                build_direct_predict_view = getattr(self.processor, "build_direct_predict_view", None)
                if callable(build_direct_predict_view):
                    return build_direct_predict_view(payload), payload["y"]
            # Stores the `build_direct_view` for use by the remaining steps in this flow.
            build_direct_view = getattr(self.processor, "build_direct_view", None)
            if callable(build_direct_view):
                return build_direct_view(payload)
        return x, y

    # Defines `get_sample_meta` for get sample meta logic.
    def get_sample_meta(self, idx: int) -> Dict[str, Any]:
        payload_meta = dict(self.get_processed_sample(idx).get("meta", {}))
        if self.sample_entries:
            entry = dict(self.sample_entries[idx])
            # Initializes the `meta` as a collection for values from this processing stage.
            meta = {
                "sample_id": self.sample_ids[idx],
                "file_path": self.file_paths[idx],
                "processed_path": entry.get("processed_path"),
                # Completes the multi-line `entry.get` arguments or elements.
                "source_file": entry.get("source_file"),
                "data_kind": self.data_kind,
                "sample_view": self.sample_view,
                "mode": self.mode,
                # Completes the multi-line `collection` arguments or elements.
                "temporal_strategy": self.temporal_strategy,
            }
            meta.update(payload_meta)
            return meta
        # Initializes the `meta` as a collection for values from this processing stage.
        meta = {
            "sample_id": self.sample_ids[idx],
            "file_path": self.file_paths[idx],
            "data_kind": self.data_kind,
            # Completes the multi-line `collection` arguments or elements.
            "sample_view": self.sample_view,
            "mode": self.mode,
            "temporal_strategy": self.temporal_strategy,
        }
        # Adds the current value to `meta` for later aggregation.
        meta.update(payload_meta)
        return meta
