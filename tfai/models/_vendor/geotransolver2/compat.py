"""Single-device compatibility helpers; no PhysicsNeMo or NVIDIA imports."""
import torch
from torch import nn


class Float:
    @classmethod
    def __class_getitem__(cls, key):
        return torch.Tensor


class ModelMetaData:
    pass


class Module(nn.Module):
    def __init__(self, meta=None):
        super().__init__()
        self.meta = meta


def check_version_spec(name, *args, **kwargs):
    if name != 'transformer_engine':
        raise ValueError(f'Unsupported optional backend check: {name}')
    return False


def get_activation(name):
    activations = {'gelu': nn.GELU, 'relu': nn.ReLU, 'silu': nn.SiLU,
                   'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'elu': nn.ELU}
    return activations[name.lower()]()


class Replicate:
    def __init__(self):
        raise RuntimeError('This GeoTransolver port is single-device; ShardTensor is unsupported')
