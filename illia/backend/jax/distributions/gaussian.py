"""JAX/Flax learnable Gaussian distribution."""

# 3pps
import jax
import jax.numpy as jnp
from flax import nnx

# Own modules
from illia.base.distributions.base import DistributionBase


class GaussianDistribution(DistributionBase, nnx.Module):
    """Learnable Gaussian with diagonal covariance.

    The standard deviation is derived from *rho* via softplus to ensure
    positivity.  KL divergence can be estimated from ``log_prob`` differences.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        mu_prior: float = 0.0,
        std_prior: float = 0.1,
        mu_init: float = 0.0,
        rho_init: float = -7.0,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:
        self.shape = shape
        self.mu_prior = mu_prior
        self.std_prior = std_prior

        self.mu = nnx.Param(mu_init + 0.1 * jax.random.normal(rngs.params(), shape))
        self.rho = nnx.Param(rho_init + 0.1 * jax.random.normal(rngs.params(), shape))

    def sample(self, rngs: nnx.Rngs = nnx.Rngs(0)) -> jax.Array:
        """Draw a reparameterised sample."""
        eps = jax.random.normal(rngs.params(), self.rho.shape)
        sigma = jnp.log1p(jnp.exp(jnp.asarray(self.rho)))
        return self.mu + sigma * eps

    def log_prob(self, x: jax.Array | None = None) -> jax.Array:
        """Log-probability of *x* (sampled internally when *None*)."""
        if x is None:
            x = self.sample()

        pi = jnp.acos(jnp.zeros(1)) * 2
        log_norm = -jnp.log(jnp.sqrt(2 * pi))

        log_prior = (
            log_norm
            - jnp.log(self.std_prior)
            - ((x - self.mu_prior) ** 2) / (2 * self.std_prior**2)
            - 0.5
        )

        sigma = jnp.log1p(jnp.exp(jnp.asarray(self.rho)))
        log_posterior = (
            log_norm - jnp.log(sigma) - ((x - self.mu) ** 2) / (2 * sigma**2) - 0.5
        )

        return log_posterior.sum() - log_prior.sum()

    @property
    def num_params(self) -> int:
        """Total number of learnable parameters."""
        return len(self.mu.reshape(-1))

    def __call__(self) -> jax.Array:
        """Forward pass: draw a sample."""
        return self.sample()
