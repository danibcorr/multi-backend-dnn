"""TensorFlow learnable Gaussian distribution."""

# Standard libraries
import math

# 3pps
import keras
import tensorflow as tf
from keras import saving

# Own modules
from illia.base.distributions.base import DistributionBase


@saving.register_keras_serializable(package="illia", name="GaussianDistribution")
class GaussianDistribution(DistributionBase, keras.layers.Layer):
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
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.shape_ = shape
        self.mu_prior_value = mu_prior
        self.std_prior_value = std_prior
        self.mu_init = mu_init
        self.rho_init = rho_init

        self.build(shape)

    def build(self, input_shape: tf.TensorShape) -> None:
        """Create trainable and non-trainable weights."""
        self.mu_prior = self.add_weight(
            shape=(),
            initializer=tf.constant_initializer(self.mu_prior_value),
            trainable=False,
            name="mu_prior",
        )
        self.std_prior = self.add_weight(
            shape=(),
            initializer=tf.constant_initializer(self.std_prior_value),
            trainable=False,
            name="std_prior",
        )
        self.mu = self.add_weight(
            shape=self.shape_,
            initializer=tf.random_normal_initializer(mean=self.mu_init, stddev=0.1),
            trainable=True,
            name="mu",
        )
        self.rho = self.add_weight(
            shape=self.shape_,
            initializer=tf.random_normal_initializer(mean=self.rho_init, stddev=0.1),
            trainable=True,
            name="rho",
        )
        super().build(input_shape)

    def get_config(self) -> dict:
        """Serialisation config for Keras save/load."""
        return {
            **super().get_config(),
            "shape": self.shape_,
            "mu_prior": self.mu_prior_value,
            "std_prior": self.std_prior_value,
            "mu_init": self.mu_init,
            "rho_init": self.rho_init,
        }

    def sample(self) -> tf.Tensor:
        """Draw a reparameterised sample."""
        eps = tf.random.normal(shape=self.rho.shape)
        sigma = tf.math.log1p(tf.math.exp(self.rho))
        return self.mu + sigma * eps

    def log_prob(self, x: tf.Tensor | None = None) -> tf.Tensor:
        """Log-probability of *x* (sampled internally when *None*)."""
        if x is None:
            x = self.sample()

        pi = tf.convert_to_tensor(math.pi)
        log_norm = -tf.math.log(tf.math.sqrt(2 * pi))

        log_prior = (
            log_norm
            - tf.math.log(self.std_prior)
            - ((x - self.mu_prior) ** 2) / (2 * self.std_prior**2)
            - 0.5
        )

        sigma = tf.math.log1p(tf.math.exp(self.rho))
        log_posterior = (
            log_norm - tf.math.log(sigma) - ((x - self.mu) ** 2) / (2 * sigma**2) - 0.5
        )

        return tf.math.reduce_sum(log_posterior) - tf.math.reduce_sum(log_prior)

    @property
    def num_params(self) -> int:
        """Total number of learnable parameters."""
        return int(tf.size(self.mu))
