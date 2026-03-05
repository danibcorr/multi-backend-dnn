"""TensorFlow Linear (Dense) layer."""

# 3pps
import tensorflow as tf
from keras import layers

# Own modules
from illia.base.nn.base import LinearBase


class Linear(LinearBase, layers.Layer):
    """Fully-connected layer backed by TensorFlow/Keras."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self._dense = layers.Dense(
            out_features, use_bias=bias, input_shape=(in_features,)
        )

    def forward(self, x: tf.Tensor) -> tf.Tensor:
        return self._dense(x)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.forward(x)
