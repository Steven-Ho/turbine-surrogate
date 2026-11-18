import ctypes
import math
import os
import sys
from typing import Dict, List, Optional
from ctypes import wintypes


def _validate_cpu_affinity_ratio(cpu_affinity_ratio: Optional[float]) -> Optional[float]:
    """校验 CPU 亲和性比例配置。"""
    if cpu_affinity_ratio is None:
        return None

    ratio = float(cpu_affinity_ratio)
    if ratio <= 0 or ratio > 1:
        raise ValueError(
            f"general.cpu_affinity_ratio 必须位于 (0, 1] 区间，当前值为: {cpu_affinity_ratio}"
        )
    return ratio


def _list_affinity_cpu_ids_windows() -> List[int]:
    """读取 Windows 当前进程可用的 CPU 亲和性列表。"""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    mask_type = ctypes.c_size_t
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetProcessAffinityMask.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(mask_type),
        ctypes.POINTER(mask_type),
    ]
    kernel32.GetProcessAffinityMask.restype = wintypes.BOOL

    handle = kernel32.GetCurrentProcess()
    process_mask = mask_type()
    system_mask = mask_type()

    success = kernel32.GetProcessAffinityMask(
        handle,
        ctypes.byref(process_mask),
        ctypes.byref(system_mask),
    )
    if not success:
        error_code = ctypes.get_last_error()
        raise OSError(error_code, "读取当前进程 CPU 亲和性失败")

    cpu_ids: List[int] = []
    bitmask = int(process_mask.value)
    bit_index = 0
    while bitmask:
        if bitmask & 1:
            cpu_ids.append(bit_index)
        bitmask >>= 1
        bit_index += 1
    return cpu_ids


def _set_affinity_cpu_ids_windows(cpu_ids: List[int]) -> None:
    """在 Windows 上设置当前进程 CPU 亲和性。"""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    kernel32.SetProcessAffinityMask.restype = wintypes.BOOL

    handle = kernel32.GetCurrentProcess()

    affinity_mask = 0
    for cpu_id in cpu_ids:
        affinity_mask |= 1 << int(cpu_id)

    success = kernel32.SetProcessAffinityMask(handle, ctypes.c_size_t(affinity_mask))
    if not success:
        error_code = ctypes.get_last_error()
        raise OSError(error_code, "设置当前进程 CPU 亲和性失败")


def _list_affinity_cpu_ids() -> List[int]:
    """获取当前进程可使用的 CPU 编号列表。"""
    if hasattr(os, "sched_getaffinity"):
        return sorted(int(cpu_id) for cpu_id in os.sched_getaffinity(0))

    if sys.platform.startswith("win"):
        return _list_affinity_cpu_ids_windows()

    raise NotImplementedError("当前平台暂不支持读取 CPU 亲和性")


def _set_affinity_cpu_ids(cpu_ids: List[int]) -> None:
    """设置当前进程可使用的 CPU 编号列表。"""
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(int(cpu_id) for cpu_id in cpu_ids))
        return

    if sys.platform.startswith("win"):
        _set_affinity_cpu_ids_windows(cpu_ids)
        return

    raise NotImplementedError("当前平台暂不支持设置 CPU 亲和性")


def get_current_cpu_affinity_ids() -> List[int]:
    """返回当前进程可用的 CPU 编号列表。"""
    return _list_affinity_cpu_ids()


def restore_cpu_affinity(cpu_ids: Optional[List[int]]) -> bool:
    """将当前进程的 CPU 亲和性恢复到指定集合。"""
    if cpu_ids is None:
        return False

    normalized_cpu_ids = sorted({int(cpu_id) for cpu_id in cpu_ids})
    if not normalized_cpu_ids:
        raise ValueError("cpu_ids 不能为空，无法恢复 CPU 亲和性")

    current_cpu_ids = _list_affinity_cpu_ids()
    if current_cpu_ids == normalized_cpu_ids:
        return False

    _set_affinity_cpu_ids(normalized_cpu_ids)
    return True


def apply_cpu_affinity_limit(cpu_affinity_ratio: Optional[float]) -> Optional[Dict]:
    """按比例限制当前进程可使用的 CPU 核心数。"""
    ratio = _validate_cpu_affinity_ratio(cpu_affinity_ratio)
    if ratio is None:
        return None

    available_cpu_ids = _list_affinity_cpu_ids()
    if not available_cpu_ids:
        raise RuntimeError("未能获取当前进程可用的 CPU 列表")

    limited_cpu_count = max(1, math.floor(len(available_cpu_ids) * ratio))
    selected_cpu_ids = available_cpu_ids[:limited_cpu_count]

    if selected_cpu_ids != available_cpu_ids:
        _set_affinity_cpu_ids(selected_cpu_ids)

    return {
        "ratio": float(ratio),
        "available_cpu_count": len(available_cpu_ids),
        "limited_cpu_count": len(selected_cpu_ids),
        "cpu_ids": list(selected_cpu_ids),
        "applied": selected_cpu_ids != available_cpu_ids,
    }
