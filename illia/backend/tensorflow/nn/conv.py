"""TensorFlow Conv2d layer."""

# 3pps
import tensorflow as tf
from keras import layers

# Own modules
from illia.backend.base.nn.base import Conv2dBase


class Conv2d(Conv2dBase, layers.Layer):
    """2-D convolution layer backed by TensorFlow/Keras."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: str = "valid",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self._conv = layers.Conv2D(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            use_bias=bias,
            data_format="channels_first",
        )

    def forward(self, x: tf.Tensor) -> tf.Tensor:
        return self._conv(x)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.forward(x)
