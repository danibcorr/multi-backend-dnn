"""PyTorch Linear layer."""

# 3pps
import torch
from torch import nn

# Own modules
from illia.base.nn.base import LinearBase


class Linear(LinearBase, nn.Module):
    """Fully-connected layer backed by PyTorch."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        nn.Module.__init__(self)
        self._linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._linear(x)
