"""启元发行包的单模型 checkpoint 恢复工具。"""

from typing import Dict

from tfai.factories.model_factory import ModelFactory
from tfai.utils.qiyuan_constraints import validate_qiyuan_config


RESIDUAL_ADDITIVE_TWO_STAGE_BUNDLE = "residual_additive_two_stage"


def is_residual_additive_bundle(checkpoint: Dict) -> bool:
    """判断 checkpoint 是否为不支持的多阶段残差模型。"""
    return (
        isinstance(checkpoint, dict)
        and str(checkpoint.get("model_bundle_type", "")).strip().lower()
        == RESIDUAL_ADDITIVE_TWO_STAGE_BUNDLE
    )


def build_model_from_checkpoint(checkpoint: Dict, config: Dict):
    """从单模型 checkpoint 恢复启元支持的模型。"""
    model_type = validate_qiyuan_config(
        config,
        model_type=(config.get("model", {}) or {}).get("type", ""),
    )
    if is_residual_additive_bundle(checkpoint):
        raise ValueError("启元发行包不支持多阶段残差模型 checkpoint")
    if "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint 缺少 model_state_dict，无法恢复模型")

    model = ModelFactory.create_model(
        model_type=model_type,
        task_type="field",
        config=config,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, model_type
