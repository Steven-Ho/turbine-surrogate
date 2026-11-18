from typing import Dict

from tfai.utils.qiyuan_constraints import (
    SUPPORTED_STEADY_MODELS,
    # Continues the surrounding statement or collection definition.
    validate_qiyuan_config,
)


class ModelFactory:
    """启元发行包的稳态场模型工厂。"""

    SUPPORTED_MODELS = SUPPORTED_STEADY_MODELS

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def get_model(model_type: str, task_type: str, config: Dict):
        """校验并返回规范化模型名称。"""
        if str(task_type).strip().lower() != "field":
            raise ValueError("启元发行包仅支持 field 场任务")
        # Returns `validate_qiyuan_config(config, model_type=model_type)` to the caller for the next stage.
        return validate_qiyuan_config(config, model_type=model_type)

    @staticmethod
    def create_model(model_type: str, task_type: str, config: Dict):
        """根据稳态场配置创建模型实例。"""
        model_type = ModelFactory.get_model(model_type, task_type, config)
        # Selects the required item from `config` as the model settings.
        model_cfg = config["model"]

        if model_type == "UNet":
            from tfai.models.unet import UNet

            return UNet(
                # Completes the multi-line `UNet` arguments or elements.
                in_channels=model_cfg["in_channels"],
                n_classes=model_cfg["out_channels"],
                img_size=model_cfg["img_size"],
                model_scale=model_cfg["scale"],
            # Completes the multi-line `UNet` arguments or elements.
            )
        if model_type == "GCN":
            from tfai.models.gcn import GCN

            return GCN(
                # Completes the multi-line `GCN` arguments or elements.
                input_dim=model_cfg["in_channels"],
                output_dim=model_cfg["out_channels"],
                config=config,
            )
        # Checks `model_type == "Graph_UNet"` before entering this conditional branch.
        if model_type == "Graph_UNet":
            from tfai.models.graph_unet import GraphUNet

            return GraphUNet(
                input_dim=model_cfg["in_channels"],
                # Completes the multi-line `GraphUNet` arguments or elements.
                output_dim=model_cfg["out_channels"],
                config=config,
            )
        if model_type == "Graph_Transformer":
            # Imports `GraphTransformer` from `tfai.models.graph_transformer` for use in this module.
            from tfai.models.graph_transformer import GraphTransformer

            return GraphTransformer(
                input_dim=model_cfg["in_channels"],
                output_dim=model_cfg["out_channels"],
                # Completes the multi-line `GraphTransformer` arguments or elements.
                config=config,
            )
        if model_type in {"Transolver_3", "GeoTransolver"}:
            from tfai.models.transolver_adapters import Transolver3, GeoTransolver
            cls = Transolver3 if model_type == "Transolver_3" else GeoTransolver
            return cls(config=config,input_dim=model_cfg["in_channels"],output_dim=model_cfg["out_channels"])
        if model_type == "POD":
            from tfai.models.pod import POD

            # Returns `` to the caller for the next stage.
            return POD(
                in_channels=model_cfg["in_channels"],
                out_channels=model_cfg["out_channels"],
                model_scale=model_cfg["scale"],
                # Completes the multi-line `POD` arguments or elements.
                config=config,
                output_shape=tuple(model_cfg["output_shape"]),
            )

        raise ValueError(f"不支持的模型类型: {model_type}")
