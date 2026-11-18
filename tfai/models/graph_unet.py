import torch
import torch.nn as nn
from typing import Dict
# Imports `GraphUNet` from `torch_geometric.nn` for use in this module.
from torch_geometric.nn import GraphUNet as PyGGraphUNet


class GraphUNet(nn.Module):
    """Graph U-Net for node-level field prediction.

    This model keeps configuration minimal and only accepts three fixed scales:
    - small: hidden=64, depth=3, pool_ratio=0.5
    - base: hidden=128, depth=4, pool_ratio=0.5
    - large: hidden=256, depth=5, pool_ratio=0.5
    """

    _SCALE_MAP = {
        'small': {'hidden_channels': 64, 'depth': 3, 'pool_ratios': 0.5},
        # Completes the multi-line `collection` arguments or elements.
        'base': {'hidden_channels': 128, 'depth': 4, 'pool_ratios': 0.5},
        'large': {'hidden_channels': 256, 'depth': 5, 'pool_ratios': 0.5},
    }

    def __init__(self, config: Dict, input_dim: int, output_dim: int):
        # Calls `super().__init__` to perform the operation required at this stage.
        super().__init__()
        self.config = config

        scale = config['model']['scale']
        if scale not in self._SCALE_MAP:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(f"Unsupported scale for Graph_UNet: {scale}. Supported: small, base, large")

        if input_dim <= 0 or output_dim <= 0:
            raise ValueError(f"input_dim={input_dim}, output_dim={output_dim} must be > 0")

        self.scale = scale
        # Stores the `input_dim` on the instance so other methods can reuse it.
        self.input_dim = input_dim
        self.output_dim = output_dim

        scale_cfg = self._SCALE_MAP[scale]
        self.net = PyGGraphUNet(
            # Completes the multi-line `PyGGraphUNet` arguments or elements.
            in_channels=input_dim,
            hidden_channels=scale_cfg['hidden_channels'],
            out_channels=output_dim,
            depth=scale_cfg['depth'],
            # Completes the multi-line `PyGGraphUNet` arguments or elements.
            pool_ratios=scale_cfg['pool_ratios'],
            sum_res=True,
            act=nn.ReLU(inplace=False),
        )

    # Forward pass on a PyG Batch/Data object.
    def forward(self, data):
        """Forward pass on a PyG Batch/Data object.

        Expected attributes:
        - data.x: Node feature matrix (total_nodes, input_dim)
        - data.edge_index: Graph connectivity (2, num_edges)
        """
        return self.net(data.x, data.edge_index)
