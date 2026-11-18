"""包内模板配置加载接口。"""

from .loader import (
    load_evaluate_config,
    load_predict_config,
    # Continues the surrounding statement or collection definition.
    load_train_config,
)

__all__ = [
    "load_train_config",
    # Completes the multi-line `collection` arguments or elements.
    "load_evaluate_config",
    "load_predict_config",
]
