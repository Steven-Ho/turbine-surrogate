from __future__ import annotations

from typing import Dict, Optional

import numpy as np
# Imports `cKDTree` from `scipy.spatial` for use in this module.
from scipy.spatial import cKDTree

from .base_adapter import BasePredictAdapter


class ImageVoxelPredictAdapter(BasePredictAdapter):
    """ImageVoxelProcessor 预测后处理适配器。"""

    def __init__(self, *args, **kwargs):
        # Calls `super().__init__` to perform the operation required at this stage.
        super().__init__(*args, **kwargs)
        self._cached_original_coords: Optional[np.ndarray] = None
        self._cached_original_source: Optional[str] = None
        self._cached_interpolation_plan: Optional[Dict[str, np.ndarray]] = None

    def _get_original_coords(self, sample_id: int, sample_file: str) -> Optional[np.ndarray]:
        resolved_source = self._resolve_writeback_template_source(sample_file, sample_id)
        if resolved_source:
            # Stores the `sample_file` for use by the remaining steps in this flow.
            sample_file = resolved_source

        if (not self.is_geometry_varying and self._cached_original_coords is not None
                and self._cached_original_source == sample_file):
            return self._cached_original_coords
        # A fixed training grid does not imply every test mesh has identical
        # coordinates. Invalidate the inverse map whenever the source changes.
        self._cached_interpolation_plan = None
        data = self.reader.read(
            sample_file,
            sample_order=sample_id,
            # Completes the multi-line `self.reader.read` arguments or elements.
            allow_missing_fields=True,
        )
        data_dict = data.get_data()

        point_or_cell = str(
            # Completes the multi-line `self.config.get('task', {}).get` arguments or elements.
            self.config.get("task", {}).get("point_or_cell", data.point_or_cell)
        ).strip().lower()
        if point_or_cell == "point" and "PointCoordinates" in data_dict:
            original_coords = data_dict["PointCoordinates"]
        # Checks `point_or_cell == "cell" and "CellCoordinates" in data_dict` before entering this conditional branch.
        elif point_or_cell == "cell" and "CellCoordinates" in data_dict:
            original_coords = data_dict["CellCoordinates"]
        else:
            self._log("warning", f"样本 {sample_id} 无法找到原始网格坐标")
            # Returns `None` to the caller for the next stage.
            return None

        if not self.is_geometry_varying:
            self._cached_original_coords = original_coords
            self._cached_original_source = sample_file

        return original_coords

    # Defines `_build_interpolation_plan` for  build interpolation plan logic.
    def _build_interpolation_plan(
        self,
        voxel_centers: np.ndarray,
        mask: np.ndarray,
        # Continues the surrounding statement or collection definition.
        original_coords: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        valid_voxel_centers = voxel_centers[mask == 1]
        if len(valid_voxel_centers) == 0:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("没有可用于反插值的有效体素中心")

        tree = cKDTree(valid_voxel_centers)
        k_use = min(8, len(valid_voxel_centers))
        dists, inds = tree.query(original_coords, k=k_use)
        # Checks `k_use == 1` before entering this conditional branch.
        if k_use == 1:
            dists = dists[:, None]
            inds = inds[:, None]
        weights = 1 / (dists + 1e-12)
        # Derives the `weights` from the current values for downstream processing.
        weights = weights / weights.sum(axis=1, keepdims=True)
        return {
            "inds": inds,
            "weights": weights,
            # Completes the multi-line `collection` arguments or elements.
            "valid_mask": (mask == 1),
        }

    def postprocess_prediction(
        self,
        # Continues the surrounding statement or collection definition.
        prediction: np.ndarray,
        sample_id: int,
        sample_file: str,
    ) -> Optional[Dict[str, np.ndarray]]:
        """将体素预测插值回原始网格。"""
        # Checks `getattr(self.processor, "cached_voxel_centers", None) is None` before entering this conditional branch.
        if getattr(self.processor, "cached_voxel_centers", None) is None:
            self._log("warning", f"样本 {sample_id} 缺少 `cached_voxel_centers`")
            return None

        voxel_centers = self.processor.cached_voxel_centers
        # Stores the `mask` for use by the remaining steps in this flow.
        mask = self.processor.cached_mask

        original_coords = self._get_original_coords(sample_id, sample_file)
        if original_coords is None:
            return None

        # Stores the `interpolated_data` for use by the remaining steps in this flow.
        interpolated_data = self._interpolate_voxel_to_original(
            prediction,
            voxel_centers,
            mask,
            # Completes the multi-line `self._interpolate_voxel_to_original` arguments or elements.
            original_coords,
        )

        field_names = list(
            self.config.get("task", {}).get("effective_variables")
            # Completes the multi-line `self.config.get('task', {}).get` arguments or elements.
            or self.config.get("task", {}).get("variables", [])
        )
        if not field_names:
            self._log("warning", f"样本 {sample_id} 缺少 task.effective_variables，无法映射预测结果")
            # Returns `None` to the caller for the next stage.
            return None

        processed_fields: Dict[str, np.ndarray] = {}
        if interpolated_data.ndim == 2:
            for j, field_name in enumerate(field_names):
                # Checks `j < interpolated_data.shape[1]` before entering this conditional branch.
                if j < interpolated_data.shape[1]:
                    processed_fields[field_name] = interpolated_data[:, j]
        elif interpolated_data.ndim == 3:
            for j, field_name in enumerate(field_names):
                # Checks `j < interpolated_data.shape[2]` before entering this conditional branch.
                if j < interpolated_data.shape[2]:
                    processed_fields[field_name] = interpolated_data[:, :, j]
        else:
            self._log(
                # Completes the multi-line `self._log` arguments or elements.
                "warning",
                f"样本 {sample_id} 的体素插值结果维度不支持: {interpolated_data.shape}",
            )
            return None

        # Returns `processed_fields` to the caller for the next stage.
        return processed_fields

    def _interpolate_single_step(
        self,
        prediction: np.ndarray,
        # Continues the surrounding statement or collection definition.
        valid_mask: np.ndarray,
        inds: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """对单个时间步的体素结果做插值，返回 [M, C]。"""
        # Validates `prediction.ndim != 4` and raises `ValueError` when the input cannot be processed safely.
        if prediction.ndim != 4:
            raise ValueError(f"单步体素预测应为 [C, D, H, W]，当前为 {prediction.shape}")

        voxel_pred = prediction.reshape(prediction.shape[0], -1)[:, valid_mask]
        interpolated = np.sum(voxel_pred[:, inds] * weights, axis=2)
        # Returns `interpolated.T` to the caller for the next stage.
        return interpolated.T

    def _interpolate_voxel_to_original(
        self,
        prediction: np.ndarray,
        # Continues the surrounding statement or collection definition.
        voxel_centers: np.ndarray,
        mask: np.ndarray,
        original_coords: np.ndarray,
    ) -> np.ndarray:
        """将体素网格结果插值回原始网格。"""
        # Checks `` before entering this conditional branch.
        if self.is_geometry_varying or self._cached_interpolation_plan is None:
            plan = self._build_interpolation_plan(voxel_centers, mask, original_coords)
            if not self.is_geometry_varying:
                self._cached_interpolation_plan = plan
        # Continues the surrounding statement or collection definition.
        else:
            plan = self._cached_interpolation_plan

        inds = plan["inds"]
        weights = plan["weights"]
        # Selects the required item from `plan` as the valid-item mask.
        valid_mask = plan["valid_mask"]

        if prediction.ndim == 4:
            return self._interpolate_single_step(
                prediction=prediction,
                # Completes the multi-line `self._interpolate_single_step` arguments or elements.
                valid_mask=valid_mask,
                inds=inds,
                weights=weights,
            )

        # Checks `prediction.ndim == 5` before entering this conditional branch.
        if prediction.ndim == 5:
            step_results = []
            for step_prediction in prediction:
                step_results.append(
                    # Completes the multi-line `self._interpolate_single_step` arguments or elements.
                    self._interpolate_single_step(
                        prediction=step_prediction,
                        valid_mask=valid_mask,
                        inds=inds,
                        # Completes the multi-line `self._interpolate_single_step` arguments or elements.
                        weights=weights,
                    )
                )
            return np.stack(step_results, axis=0)

        # Stores the `interpolated` for use by the remaining steps in this flow.
        interpolated = np.sum(prediction[inds] * weights[:, :, np.newaxis], axis=1)
        return interpolated
