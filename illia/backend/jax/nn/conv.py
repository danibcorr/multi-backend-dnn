"""JAX/Flax Conv2d layer."""

# 3pps
import jax
from flax import nnx

# Own modules
from illia.backend.base.nn.base import Conv2dBase


class Conv2d(Conv2dBase, nnx.Module):
    """2-D convolution layer backed by JAX/Flax."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: str = "VALID",
        bias: bool = True,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)

        self._conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            use_bias=bias,
            rngs=rngs,
        )

    def forward(self, x: jax.Array) -> jax.Array:
        return self._conv(x)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.forward(x)
