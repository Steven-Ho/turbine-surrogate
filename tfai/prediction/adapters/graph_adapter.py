from __future__ import annotations

import os
import re
# Imports `Dict, List, Optional` from `typing` for use in this module.
from typing import Dict, List, Optional

import numpy as np

from tfai.readers.sample_map import SampleMap
from tfai.utils.path_utils import resolve_path_from_base

# Imports `BasePredictAdapter` from `base_adapter` for use in this module.
from .base_adapter import BasePredictAdapter


class GraphPredictAdapter(BasePredictAdapter):
    """GraphProcessor 预测后处理适配器。"""

    @staticmethod
    def _build_source_file_index(source_sample_files: List[str]) -> Dict[int, str]:
        """基于文件名中的 `sample_<id>` 片段构建样本文件索引。"""
        # Initializes the `indexed_files` as a collection for values from this processing stage.
        indexed_files: Dict[int, str] = {}
        for path in source_sample_files:
            match = re.search(r"sample_(\d+)\.", os.path.basename(path))
            if not match:
                # Handles the data or control-flow case represented by this statement.
                continue
            sample_id = int(match.group(1))
            indexed_files.setdefault(sample_id, path)
        return indexed_files

    # 处理按样本字典组织的预测结果并写回原始格式文件。
    def process_outputs(
        self,
        predictions: Dict[int, np.ndarray],
        samples_csv: str,
        # Continues the surrounding statement or collection definition.
        output_dir: str,
        source_sample_files: List[str],
    ) -> List[str]:
        """处理按样本字典组织的预测结果并写回原始格式文件。"""
        if not isinstance(predictions, dict):
            # Returns `` to the caller for the next stage.
            return super().process_outputs(
                predictions,
                samples_csv,
                output_dir,
                # Completes the multi-line `super().process_outputs` arguments or elements.
                source_sample_files,
            )

        os.makedirs(output_dir, exist_ok=True)
        output_paths: List[str] = []

        # Stores the `sample_map` for use by the remaining steps in this flow.
        sample_map = SampleMap(samples_csv)
        base_dir = os.path.dirname(os.path.abspath(samples_csv))
        fallback_source_index = self._build_source_file_index(source_sample_files)

        for sid, pred in predictions.items():
            # Checks `sid is None` before entering this conditional branch.
            if sid is None:
                self._log("warning", "跳过无效样本 ID")
                continue

            sample_id = int(sid)
            # Stores the file path for use by the remaining steps in this flow.
            file_path = sample_map.get_file_path(sample_id)
            resolved_source = ""
            if not file_path:
                file_path = fallback_source_index.get(sample_id)
            # Checks `file_path` before entering this conditional branch.
            if file_path:
                resolved_source = resolve_path_from_base(base_dir, file_path)

            sample_file = self._get_sample_file(
                index=0,
                # Completes the multi-line `self._get_sample_file` arguments or elements.
                sample_id=sample_id,
                sample_files=[resolved_source] if resolved_source else [],
            )
            if not sample_file:
                # Checks `not resolved_source` before entering this conditional branch.
                if not resolved_source:
                    self._log("warning", f"样本 {sample_id} 找不到源文件路径，已跳过")
                continue

            try:
                # Stores the `processed_fields` for use by the remaining steps in this flow.
                processed_fields = self.postprocess_prediction(
                    pred,
                    sample_id,
                    sample_file,
                # Completes the multi-line `self.postprocess_prediction` arguments or elements.
                )
                if processed_fields is None:
                    self._log("warning", f"样本 {sample_id} 后处理失败，已跳过")
                    continue

                # Adds the current value to `output_paths` for later aggregation.
                output_paths.extend(
                    self._write_processed_fields(
                        sample_id=sample_id,
                        sample_file=sample_file,
                        # Completes the multi-line `self._write_processed_fields` arguments or elements.
                        processed_fields=processed_fields,
                        output_dir=output_dir,
                    )
                )
            # Continues the surrounding statement or collection definition.
            except Exception as exc:
                self._log("error", f"样本 {sample_id} 写回失败: {str(exc)}", exc_info=True)

        return output_paths

    def postprocess_prediction(
        # Continues the surrounding statement or collection definition.
        self,
        prediction: np.ndarray,
        sample_id: int,
        sample_file: str,
    # Completes the multi-line `collection` arguments or elements.
    ) -> Optional[Dict[str, np.ndarray]]:
        """将图节点级预测映射回字段字典。"""
        del sample_file

        field_names = list(
            self.config.get("task", {}).get("effective_variables")
            # Completes the multi-line `self.config.get('task', {}).get` arguments or elements.
            or self.config.get("task", {}).get("variables", [])
        )
        if not field_names:
            self._log("warning", f"样本 {sample_id} 缺少 task.effective_variables，无法映射预测结果")
            # Returns `None` to the caller for the next stage.
            return None

        processed: Dict[str, np.ndarray] = {}

        if prediction.ndim == 2:
            _, channel_count = prediction.shape
            # Visits each item in `enumerate(field_names)` and applies the current processing step.
            for channel_idx, name in enumerate(field_names):
                if channel_idx < channel_count:
                    processed[name] = prediction[:, channel_idx]
            return processed

        # Checks `prediction.ndim == 3` before entering this conditional branch.
        if prediction.ndim == 3:
            _, _, channel_count = prediction.shape
            for channel_idx, name in enumerate(field_names):
                if channel_idx < channel_count:
                    # Selects the required item from `prediction` as the `processed[name]`.
                    processed[name] = prediction[:, :, channel_idx]
            return processed

        self._log("warning", f"样本 {sample_id} 预测维度不支持: {prediction.shape}")
        return None
