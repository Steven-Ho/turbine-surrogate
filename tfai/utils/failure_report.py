"""失败摘要与降级报告工具。"""

import copy
from typing import Dict, Iterable, List, Optional


def build_failure_entry(
    stage: str,
    component: str,
    message: str,
    error: Optional[BaseException] = None,
    details: Optional[Dict] = None,
) -> Dict:
    """构建结构化失败记录。"""
    entry = {
        "stage": str(stage),
        "component": str(component),
        "message": str(message),
    }
    if error is not None:
        entry["error_type"] = type(error).__name__
        entry["error_message"] = str(error)
    if details:
        entry["details"] = copy.deepcopy(details)
    return entry


def summarize_writeback_failures(
    failures: Iterable[Dict],
    *,
    strict: bool,
    expected_count: int,
    success_count: int,
) -> Dict:
    """汇总写回阶段的成功、失败与严格模式状态。"""
    failures_list: List[Dict] = [copy.deepcopy(item) for item in failures]
    expected_count = max(int(expected_count), 0)
    success_count = max(int(success_count), 0)
    failed_count = max(expected_count - success_count, 0)

    if not failures_list and expected_count == 0:
        status = "idle"
    elif not failures_list:
        status = "success"
    elif success_count <= 0:
        status = "failed"
    else:
        status = "partial"

    return {
        "strict": bool(strict),
        "status": status,
        "expected_count": expected_count,
        "success_count": success_count,
        "failed_count": failed_count,
        "failures": failures_list,
    }
