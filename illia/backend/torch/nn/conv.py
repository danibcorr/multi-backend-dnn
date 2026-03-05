"""PyTorch Conv2d layer."""

# 3pps
import torch
from torch import nn

# Own modules
from illia.base.nn.base import Conv2dBase


class Conv2d(Conv2dBase, nn.Module):
    """2-D convolution layer backed by PyTorch."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        self._conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._conv(x)
