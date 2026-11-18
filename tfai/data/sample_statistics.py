import logging
from typing import Callable, List, Optional, Tuple

import numpy as np

# Imports `BaseData` from `base_data` for use in this module.
from .base_data import BaseData
from ..utils.logger import Logger


def _resolve_logger(logger=None) -> logging.Logger:
    """解析统计模块使用的日志对象。"""
    if logger is not None:
        # Returns `logger` to the caller for the next stage.
        return logger
    return Logger.get_first_configured_logger(
        ["train", "predict", "evaluate"],
        fallback_name=__name__,
    # Completes the multi-line `Logger.get_first_configured_logger` arguments or elements.
    )


def extract_coord_ranges_from_sample(
    data: BaseData,
    logger=None,
# Completes the multi-line `collection` arguments or elements.
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
    """Extract 3D coordinate ranges from one BaseData sample."""
    logger = _resolve_logger(logger)
    try:
        data_dict = data.get_data()
        # Returns early when `"PointCoordinates" not in data_dict` makes the remaining work inapplicable.
        if "PointCoordinates" not in data_dict:
            return None

        point_coordinates = np.asarray(data_dict["PointCoordinates"])
        min_coords = np.min(point_coordinates, axis=0)
        # Stores the `max_coords` for use by the remaining steps in this flow.
        max_coords = np.max(point_coordinates, axis=0)

        return (
            (float(min_coords[0]), float(max_coords[0])),
            (float(min_coords[1]), float(max_coords[1])),
            # Completes the multi-line `collection` arguments or elements.
            (float(min_coords[2]), float(max_coords[2])),
        )
    except Exception as exc:
        logger.warning("提取坐标范围失败: %s", str(exc))
        # Returns `None` to the caller for the next stage.
        return None


def calculate_cumulative_coord_ranges(
    sample_loader: Callable[[str, Optional[int]], BaseData],
    file_paths: List[str],
    # Continues the surrounding statement or collection definition.
    sample_ids: List[Optional[int]],
    logger=None,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
    """Accumulate 3D coordinate ranges across a group of samples."""
    logger = _resolve_logger(logger)
    # Stores the `min_coords` for use by the remaining steps in this flow.
    min_coords = None
    max_coords = None

    for index, (file_path, sample_id) in enumerate(zip(file_paths, sample_ids), start=1):
        try:
            # Stores the current data for use by the remaining steps in this flow.
            data = sample_loader(file_path, sample_id)
            sample_coord_ranges = extract_coord_ranges_from_sample(data, logger=logger)
            if not sample_coord_ranges:
                continue

            # Stores the `current_min` for use by the remaining steps in this flow.
            current_min = np.array(
                [
                    sample_coord_ranges[0][0],
                    sample_coord_ranges[1][0],
                    # Completes the multi-line `collection` arguments or elements.
                    sample_coord_ranges[2][0],
                ]
            )
            current_max = np.array(
                # Completes the multi-line `collection` arguments or elements.
                [
                    sample_coord_ranges[0][1],
                    sample_coord_ranges[1][1],
                    sample_coord_ranges[2][1],
                # Completes the multi-line `collection` arguments or elements.
                ]
            )

            if min_coords is None:
                min_coords = current_min
                # Stores the `max_coords` for use by the remaining steps in this flow.
                max_coords = current_max
            else:
                min_coords = np.minimum(min_coords, current_min)
                max_coords = np.maximum(max_coords, current_max)
        # Continues the surrounding statement or collection definition.
        except Exception as exc:
            logger.warning("统计样本 %s 的坐标范围失败: %s", index, str(exc))
            continue

    if min_coords is None or max_coords is None:
        # Records the current execution state to make progress and failures traceable.
        logger.warning("未能从任何有效样本中统计出坐标范围")
        return None

    coord_ranges = (
        (float(min_coords[0]), float(max_coords[0])),
        # Completes the multi-line `collection` arguments or elements.
        (float(min_coords[1]), float(max_coords[1])),
        (float(min_coords[2]), float(max_coords[2])),
    )

    logger.info(
        # Completes the multi-line `logger.info` arguments or elements.
        "累计坐标范围统计完成: X=%s Y=%s Z=%s",
        coord_ranges[0],
        coord_ranges[1],
        coord_ranges[2],
    # Completes the multi-line `logger.info` arguments or elements.
    )
    return coord_ranges
