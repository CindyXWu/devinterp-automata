from enum import Enum
from functools import partial
import math
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor


class ActivationType(Enum):
    RELU = "relu"
    TANH = "tanh"
    SILU = "silu"
    GELU = "gelu"
    SIGMA_GELU = "sigma_gelu"


def get_activation(activation_type: ActivationType, **activation_kwargs):
    if activation_type == ActivationType.RELU:
        return partial(nn.ReLU, **activation_kwargs)
    elif activation_type == ActivationType.TANH:
        return partial(nn.Tanh, **activation_kwargs)
    elif activation_type == ActivationType.SILU:
        return partial(nn.SiLU, **activation_kwargs)
    elif activation_type == ActivationType.GELU:
        return partial(nn.GELU, **activation_kwargs)
    elif activation_type == ActivationType.SIGMA_GELU:
        return partial(SigmaGeLU, **activation_kwargs)
    else:
        raise ValueError(f"Unknown activation type: {activation_type}")


class SigmaGeLU(nn.Module):
    def __init__(self, sigma: float = 1.0, approximate: bool = True):
        super().__init__()
        self.sigma = sigma
        self.approximate = "tanh" if approximate else "none"

    def forward(self, x: Tensor) -> Tensor:
        # return (
        #     0.5 * self.sigma * nn.functional.gelu(x / self.sigma, approximate=self.approximate)
        #     + 0.5 * x
        #     + (self.sigma / (2 * math.sqrt(math.pi))) * torch.exp(-(x / self.sigma) ** 2)
        # )
        return (
            0.5 * x * torch.erf(x / self.sigma)
            + 0.5 * x
            + (self.sigma / (2 * math.sqrt(math.pi))) * torch.exp(-(x / self.sigma) ** 2)
        )

    def extra_repr(self):
        return f"sigma={self.sigma}"