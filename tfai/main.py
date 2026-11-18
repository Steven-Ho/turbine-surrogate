"""已弃用的旧入口兼容模块。

新代码请从 ``tfai.mains`` 或顶层 ``tfai`` 导入包接口。
"""

from tfai.mains.legacy import (
    run_evaluate_from_config as run_evaluate,
    run_predict_from_config as run_predict,
    # Continues the surrounding statement or collection definition.
    run_train_from_config as run_train,
)

__all__ = ["run_train", "run_evaluate", "run_predict"]
