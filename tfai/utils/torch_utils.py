import inspect
import os
import tempfile
import warnings
from typing import Any, Optional

import torch


def _supports_weights_only() -> bool:
    """Return True when current torch.load supports the weights_only argument."""
    try:
        return 'weights_only' in inspect.signature(torch.load).parameters
    except (TypeError, ValueError):
        return False


_HAS_WEIGHTS_ONLY = _supports_weights_only()


def load_torch(path: str, map_location: Optional[Any] = None, weights_only: Optional[bool] = None, **kwargs):
    """Compatibility wrapper for torch.load across torch versions.

    - On newer torch versions, forwards `weights_only` when provided.
    - On older torch versions, silently drops `weights_only` to avoid TypeError.
    """
    load_kwargs = dict(kwargs)
    if map_location is not None:
        load_kwargs['map_location'] = map_location
    if weights_only is not None and _HAS_WEIGHTS_ONLY:
        load_kwargs['weights_only'] = weights_only

    return torch.load(path, **load_kwargs)


def atomic_torch_save(obj: Any, path: str) -> None:
    """将对象以原子方式写入磁盘，避免中断时留下损坏文件。"""
    target_path = os.path.abspath(str(path))
    target_dir = os.path.dirname(target_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(
        # 暂存文件不复用目标文件名或扩展名，避免深层实验目录下临时路径超过 Windows 限制。
        prefix="t",
        suffix="",
        dir=target_dir or None,
    )
    os.close(fd)

    try:
        torch.save(obj, temp_path)
        os.replace(temp_path, target_path)
    except Exception:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
        raise


def resolve_torch_device(config: Optional[dict] = None) -> torch.device:
    """Resolve runtime device from config.general.device.

    Supported values:
    - "auto" / missing: use CUDA when available, else CPU
    - "cpu"
    - "cuda"
    - "cuda:<index>"
    """
    general_cfg = (config or {}).get("general", {}) if isinstance(config, dict) else {}
    requested = str(general_cfg.get("device", "auto") or "auto").strip().lower()

    if requested in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        device = torch.device(requested)
    except Exception as exc:
        raise ValueError(f"Invalid general.device value: {requested}") from exc

    if device.type == "cuda":
        if not torch.cuda.is_available():
            warnings.warn(
                f"general.device={requested} requested CUDA, but CUDA is not available. Falling back to CPU.",
                RuntimeWarning,
            )
            return torch.device("cpu")
        if device.index is not None:
            device_count = torch.cuda.device_count()
            if device.index < 0 or device.index >= device_count:
                raise ValueError(
                    f"general.device={requested} is out of range; available CUDA device count is {device_count}."
                )

    return device
