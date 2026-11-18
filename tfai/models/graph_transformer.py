import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

# Imports `TransformerConv` from `torch_geometric.nn` for use in this module.
from torch_geometric.nn import TransformerConv

class GraphTransformer(nn.Module):
    """Graph Transformer for node-level field prediction.

    Fixed scale mapping:
    - small: hidden=64, heads=4, layers=3
    - base: hidden=128, heads=8, layers=4
    - large: hidden=256, heads=8, layers=6
    """

    _SCALE_MAP = {
        'small': {'hidden_channels': 64, 'heads': 4, 'num_layers': 3},
        # Completes the multi-line `collection` arguments or elements.
        'base': {'hidden_channels': 128, 'heads': 8, 'num_layers': 4},
        'large': {'hidden_channels': 256, 'heads': 8, 'num_layers': 6},
    }

    def __init__(self, config: Dict, input_dim: int, output_dim: int):
        # Calls `super().__init__` to perform the operation required at this stage.
        super().__init__()
        self.config = config

        scale = config['model']['scale']
        if scale not in self._SCALE_MAP:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(
                f"Unsupported scale for Graph_Transformer: {scale}. Supported: small, base, large"
            )

        if input_dim <= 0 or output_dim <= 0:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(f"input_dim={input_dim}, output_dim={output_dim} must be > 0")

        cfg = self._SCALE_MAP[scale]
        hidden = cfg['hidden_channels']
        heads = cfg['heads']
        # Selects the required item from `cfg` as the `num_layers`.
        num_layers = cfg['num_layers']

        if hidden % heads != 0:
            raise ValueError(f"hidden_channels={hidden} must be divisible by heads={heads}")

        self.scale = scale
        # Stores the `input_dim` on the instance so other methods can reuse it.
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden = hidden
        self.heads = heads
        # Stores the `num_layers` on the instance so other methods can reuse it.
        self.num_layers = num_layers
        self.dropout_rate = 0.0
        self.edge_dim = 4

        self.input_proj = nn.Linear(input_dim, hidden)
        # Stores the `convs` on the instance so other methods can reuse it.
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        out_per_head = hidden // heads
        for _ in range(num_layers):
            # Adds the current value to `self.convs` for later aggregation.
            self.convs.append(
                TransformerConv(
                    in_channels=hidden,
                    out_channels=out_per_head,
                    # Completes the multi-line `TransformerConv` arguments or elements.
                    heads=heads,
                    concat=True,
                    dropout=self.dropout_rate,
                    edge_dim=self.edge_dim,
                # Completes the multi-line `TransformerConv` arguments or elements.
                )
            )
            self.norms.append(nn.LayerNorm(hidden))

        self.output_proj = nn.Linear(hidden, output_dim)

    # Forward pass on a PyG Batch/Data object.
    def forward(self, data):
        """Forward pass on a PyG Batch/Data object."""
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        if edge_attr is None:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError("GraphTransformer forward requires edge_attr, but it is missing")
        if edge_attr.ndim != 2 or edge_attr.shape[1] != self.edge_dim:
            raise ValueError(
                f"GraphTransformer edge_attr shape must be (num_edges, {self.edge_dim}), got {tuple(edge_attr.shape)}"
            # Completes the multi-line `ValueError` arguments or elements.
            )

        x = self.input_proj(x)

        for conv, norm in zip(self.convs, self.norms):
            residual = x
            # Stores the `x` for use by the remaining steps in this flow.
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(residual + x)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Returns `self.output_proj(x)` to the caller for the next stage.
        return self.output_proj(x)

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        for conv in self.convs:
            # Calls `conv.reset_parameters` to perform the operation required at this stage.
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
        self.output_proj.reset_parameters()
