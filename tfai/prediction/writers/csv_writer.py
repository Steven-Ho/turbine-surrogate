from __future__ import annotations

import csv
import logging
# Imports `os` for reuse in this module.
import os
from typing import Dict, Optional

import numpy as np

from tfai.utils.logger import Logger


# 负责将预测结果写回 CSV 文件。
class CSVPredictionWriter:
    """负责将预测结果写回 CSV 文件。"""

    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger

    # Defines `_log` for  log logic.
    def _log(self, level: str, message: str) -> None:
        logger = self.logger
        if logger is None:
            logger = Logger.get_first_configured_logger(
                # Completes the multi-line `collection` arguments or elements.
                ["predict", "evaluate", "train"],
                fallback_name=__name__,
            )
        log_fn = getattr(logger, level, logger.info)
        # Calls `log_fn` to perform the operation required at this stage.
        log_fn(message)

    def _get_write_variables(self):
        """仅从 `task.effective_variables` 解析写回变量顺序。"""
        task_cfg = self.config.get("task", {}) if isinstance(self.config, dict) else {}
        effective = task_cfg.get("effective_variables", [])

        # Returns early when `effective` makes the remaining work inapplicable.
        if effective:
            return list(effective)

        raise ValueError("写回结果时必须提供 task.effective_variables")

    def write(
        # Continues the surrounding statement or collection definition.
        self,
        sample_id: int,
        sample_file: str,
        processed_fields: Dict[str, np.ndarray],
        # Continues the surrounding statement or collection definition.
        output_dir: str,
        output_basename: Optional[str] = None,
    ) -> Optional[str]:
        """将预测结果写回 CSV。"""
        basename = output_basename or f"predicted_{sample_id}"
        # Stores the `output_file` for use by the remaining steps in this flow.
        output_file = os.path.join(output_dir, f"{basename}.csv")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        variables = self._get_write_variables()
        data_arrays = []
        # Initializes the `headers` as a collection for values from this processing stage.
        headers = []
        for var_name in variables:
            if var_name in processed_fields:
                field_data = np.asarray(processed_fields[var_name]).flatten()
                # Adds the current value to `data_arrays` for later aggregation.
                data_arrays.append(field_data)
                headers.append(var_name)

        if not data_arrays:
            self._log("warning", f"样本 {sample_id} 没有可写回的 CSV 变量")
            # Returns `None` to the caller for the next stage.
            return None

        data_matrix = np.column_stack(data_arrays)
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Calls `writer.writerow` to perform the operation required at this stage.
            writer.writerow(headers)
            writer.writerows(data_matrix)

        self._log("info", f"样本 {sample_id} 已写回 {len(headers)} 个变量到 CSV")
        return output_file
