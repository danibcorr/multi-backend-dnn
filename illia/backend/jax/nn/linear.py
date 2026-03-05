"""JAX/Flax Linear layer."""

# 3pps
import jax
from flax import nnx

# Own modules
from illia.base.nn.base import LinearBase


class Linear(LinearBase, nnx.Module):
    """Fully-connected layer backed by JAX/Flax."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:
        self._linear = nnx.Linear(in_features, out_features, use_bias=bias, rngs=rngs)

    def forward(self, x: jax.Array) -> jax.Array:
        return self._linear(x)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.forward(x)
