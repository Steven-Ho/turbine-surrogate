from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn


# Defines `_align_weight` for  align weight logic.
def _align_weight(weight: Optional[torch.Tensor], reference: torch.Tensor) -> Optional[torch.Tensor]:
    if weight is None:
        return None

    aligned = weight.to(device=reference.device, dtype=reference.dtype)
    # Repeats the current step while `aligned.ndim < reference.ndim` remains true.
    while aligned.ndim < reference.ndim:
        aligned = aligned.unsqueeze(1)
    return aligned


def _weighted_reduce(
    # Continues the surrounding statement or collection definition.
    loss_map: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    aligned_weight = _align_weight(weight, loss_map)
    # Returns early when `aligned_weight is None` makes the remaining work inapplicable.
    if aligned_weight is None:
        return loss_map.mean()

    weighted_loss = loss_map * aligned_weight
    weight_sum = aligned_weight.sum()
    # Returns early when `weight_sum <= 0` makes the remaining work inapplicable.
    if weight_sum <= 0:
        return torch.tensor(0.0, device=loss_map.device, dtype=loss_map.dtype)
    return weighted_loss.sum() / weight_sum


def _huber_loss_map(
    # Continues the surrounding statement or collection definition.
    outputs: torch.Tensor,
    targets: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    # Stores the `abs_error` for use by the remaining steps in this flow.
    abs_error = torch.abs(outputs - targets)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    return 0.5 * quadratic ** 2 + delta * linear


# 计算可选加权的 MSE。
def weighted_mse(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
# Continues the surrounding statement or collection definition.
) -> torch.Tensor:
    """计算可选加权的 MSE。"""
    loss_map = (outputs - targets) ** 2
    return _weighted_reduce(loss_map, weight)


def weighted_mae(
    # Continues the surrounding statement or collection definition.
    outputs: torch.Tensor,
    targets: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """计算可选加权的 MAE。"""
    # Stores the `loss_map` for use by the remaining steps in this flow.
    loss_map = torch.abs(outputs - targets)
    return _weighted_reduce(loss_map, weight)


def weighted_huber(
    outputs: torch.Tensor,
    # Continues the surrounding statement or collection definition.
    targets: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    delta: float = 1.0,
) -> torch.Tensor:
    """计算可选加权的 Huber 损失。"""
    # Stores the `loss_map` for use by the remaining steps in this flow.
    loss_map = _huber_loss_map(outputs, targets, delta=delta)
    return _weighted_reduce(loss_map, weight)


def mask_mse(outputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute MSE only on masked spatial locations."""
    return weighted_mse(outputs, targets, mask)


# Compute MAE only on masked spatial locations.
def mask_mae(outputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute MAE only on masked spatial locations."""
    return weighted_mae(outputs, targets, mask)


def mask_huber(
    outputs: torch.Tensor,
    # Continues the surrounding statement or collection definition.
    targets: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Compute Huber loss only on masked spatial locations."""
    # Returns `weighted_huber(outputs, targets, mask, delta=delta)` to the caller for the next stage.
    return weighted_huber(outputs, targets, mask, delta=delta)


def build_basic_criterion(loss_name: str) -> Tuple[Callable, str]:
    """Build a standard criterion and return its resolved display name."""
    normalized = str(loss_name).lower()
    if normalized == "huber":
        # Stores the `criterion` for use by the remaining steps in this flow.
        criterion = nn.HuberLoss(delta=1.0)
        setattr(criterion, "_tfai_loss_name", "huber")
        return criterion, "huber"
    if normalized == "mae":
        # Stores the `criterion` for use by the remaining steps in this flow.
        criterion = nn.L1Loss()
        setattr(criterion, "_tfai_loss_name", "mae")
        return criterion, "mae"
    if normalized == "mse":
        # Stores the `criterion` for use by the remaining steps in this flow.
        criterion = nn.MSELoss()
        setattr(criterion, "_tfai_loss_name", "mse")
        return criterion, "mse"
    criterion = nn.MSELoss()
    # Calls `setattr` to perform the operation required at this stage.
    setattr(criterion, "_tfai_loss_name", "mse")
    return criterion, "mse"
