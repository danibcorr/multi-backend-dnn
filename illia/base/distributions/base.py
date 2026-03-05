"""Abstract base for probabilistic distributions."""

# Standard libraries
from abc import ABC, abstractmethod


class DistributionBase(ABC):
    """Interface that every backend distribution must implement."""

    @abstractmethod
    def sample(self):
        """Draw a sample from the distribution."""

    @abstractmethod
    def log_prob(self, x=None):
        """Compute the log-probability of *x* (or of a fresh sample)."""

    @property
    @abstractmethod
    def num_params(self) -> int:
        """Return the number of learnable parameters."""
