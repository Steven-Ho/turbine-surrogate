from typing import Dict, List, Optional, Tuple
import torch.nn as nn
import math

# Defines `POD` to group related data and behavior.
class POD(nn.Module):
    def __init__(
        self,
        in_channels: int,
        # Continues the surrounding statement or collection definition.
        out_channels: int,
        model_scale: str,
        config: Dict,
        output_shape: Optional[Tuple[int, ...]] = None,
    # Continues the surrounding statement or collection definition.
    ):
        """初始化POD模型（本质为MLP）
        
        Args:
            in_channels: 输入通道/维度（由外部传入）
            out_channels: 输出通道/维度（由外部传入）
            model_scale: 模型规模，small/base/large
            config: 模型配置参数
            output_shape: 目标输出形状（在训练主程序中根据实际y.shape传入）
        """
        super().__init__()
        self.config = config
        self.model_scale = model_scale

        # 仅根据scale设定MLP结构
        # Checks `self.model_scale == 'small'` before entering this conditional branch.
        if self.model_scale == 'small':
            self.hidden_layers: List[int] = [64, 128, 256, 256, 256]
        
        elif self.model_scale == 'base':
            self.hidden_layers = [128, 256, 512, 512, 512]
            
        # Checks `self.model_scale == 'large'` before entering this conditional branch.
        elif self.model_scale == 'large':
            self.hidden_layers = [256, 512, 1024, 1024, 1024]
            
        else:
            raise ValueError(f"Invalid model_scale: {self.model_scale}. Expected 'small', 'base', or 'large'.")

        # Stores the `activation` on the instance so other methods can reuse it.
        self.activation = 'relu'
        self.input_dim = int(in_channels)
        self.output_dim = int(out_channels)
        self.output_shape = tuple(output_shape) if output_shape is not None else None
        # Checks `self.output_shape is not None` before entering this conditional branch.
        if self.output_shape is not None:
            expected = math.prod(self.output_shape)
            if expected != self.output_dim:
                raise ValueError(
                    # Completes the multi-line `ValueError` arguments or elements.
                    f"POD output_shape元素乘积({expected})与out_channels({self.output_dim})不一致"
                )

        # 构建网络
        layers: List[nn.Module] = []
        prev_dim = self.input_dim
        # Visits each item in `self.hidden_layers` and applies the current processing step.
        for hidden_dim in self.hidden_layers:
            hidden_dim = int(hidden_dim)
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(self.activation))
            # Stores the `prev_dim` for use by the remaining steps in this flow.
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        # Stores the `_` for use by the remaining steps in this flow.
        _ = name
        return nn.ReLU()
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量，常见形状为 (batch_size, input_dim) 或 (batch_size, 1, input_dim)
            
        Returns:
            输出张量，形状为 (batch_size, output_dim) 或 (batch_size, *output_shape)
        """
        if x.ndim == 1:
            # Stores the `x` for use by the remaining steps in this flow.
            x = x.unsqueeze(0)

        # 展平除batch外维度，确保输入为2D
        x = x.reshape(x.shape[0], -1)

        if x.shape[1] != self.input_dim:
            raise ValueError(f"POD输入维度不匹配: got {x.shape[1]}, expected {self.input_dim}")

        # Stores the `y` for use by the remaining steps in this flow.
        y = self.net(x)
        if self.output_shape is not None:
            y = y.reshape(y.shape[0], *self.output_shape)
        return y
