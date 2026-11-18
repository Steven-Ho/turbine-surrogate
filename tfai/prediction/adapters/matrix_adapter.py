from __future__ import annotations

from typing import Dict, List, Optional
import os
# Imports `numpy` for reuse in this module.
import numpy as np
from .base_adapter import BasePredictAdapter


class MatrixPredictAdapter(BasePredictAdapter):
    """MatrixProcessor 预测后处理适配器。"""

    def postprocess_prediction(
        # Continues the surrounding statement or collection definition.
        self,
        prediction: np.ndarray,
        sample_id: int,
        sample_file: str,
    # Completes the multi-line `collection` arguments or elements.
    ) -> Optional[Dict[str, np.ndarray]]:
        """POD重构物理场。"""
        if not hasattr(self.processor, "reconstruct"):
            self._log("warning", f"样本 {sample_id} 的 MatrixProcessor 未实现 `reconstruct` 方法")
            return None

        # Keeps the operation and its recovery path within the same exception boundary.
        try:
            # POD重构
            reconstructed_fields = self.processor.reconstruct(prediction)
            return reconstructed_fields
        except Exception as e:
            # Calls `self._log` to perform the operation required at this stage.
            self._log("error", f"样本 {sample_id} POD 重构失败: {str(e)}")
            return None



