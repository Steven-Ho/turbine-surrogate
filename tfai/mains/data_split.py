"""训练/测试数据划分接口。"""

import copy
import math
import random
# Imports `Dict, List, Optional` from `typing` for use in this module.
from typing import Dict, List, Optional, Tuple


def _validate_ratio(name: str, value: float) -> None:
    """校验划分比例是否合法。"""
    if not 0 < value < 1:
        raise ValueError(f"{name} 必须位于 0 和 1 之间，当前为 {value}")


# 按配置将样本列表划分为训练集与测试集。
def split_train_test(
    samples_all: List[Dict],
    config: Dict,
    test_ratio: Optional[float] = None,
# Completes the multi-line `collection` arguments or elements.
) -> Tuple[List[Dict], List[Dict]]:
    """按配置将样本列表划分为训练集与测试集。"""
    if not samples_all:
        raise ValueError("samples_all 不能为空")

    training_cfg = config.setdefault("training", {})
    # Stores the `general_cfg` for use by the remaining steps in this flow.
    general_cfg = config.setdefault("general", {})
    if test_ratio is not None:
        training_cfg["test_ratio"] = test_ratio
    test_ratio = float(training_cfg.get("test_ratio", 0.1))
    # Converts the `seed` to `int` so later checks use a consistent type.
    seed = int(general_cfg.get("seed", 42))
    _validate_ratio("training.test_ratio", test_ratio)

    if len(samples_all) < 2:
        single_split = [copy.deepcopy(item) for item in samples_all]
        # Returns `single_split, [copy.deepcopy(item) for item in samples_all]` to the caller for the next stage.
        return single_split, [copy.deepcopy(item) for item in samples_all]

    sample_count = len(samples_all)
    test_count = max(1, math.ceil(sample_count * test_ratio))
    if test_count >= sample_count:
        # Raises `ValueError` to report why this path cannot continue.
        raise ValueError(
            f"training.test_ratio={test_ratio} 对 {sample_count} 个样本过大，训练集至少需要保留 1 个样本"
        )

    rng = random.Random(seed)
    # Stores the `selected_test_indices` for use by the remaining steps in this flow.
    selected_test_indices = set(rng.sample(list(range(sample_count)), test_count))

    samples_train: List[Dict] = []
    samples_test: List[Dict] = []
    for index, item in enumerate(samples_all):
        # Stores the `copied` for use by the remaining steps in this flow.
        copied = copy.deepcopy(item)
        if index in selected_test_indices:
            samples_test.append(copied)
        else:
            # Adds the current value to `samples_train` for later aggregation.
            samples_train.append(copied)

    return samples_train, samples_test
