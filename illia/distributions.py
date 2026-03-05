"""Probability distributions — re-exported from the active backend."""

# Own modules
from illia import __get_backend__, _check_backend_switch, _lock_backend

_check_backend_switch()
_lock_backend()

if __get_backend__ == "torch":
    # Own modules
    from illia.backend.torch.distributions import GaussianDistribution
elif __get_backend__ == "tensorflow":
    # Own modules
    from illia.backend.tensorflow.distributions import GaussianDistribution
elif __get_backend__ == "jax":
    # Own modules
    from illia.backend.jax.distributions import GaussianDistribution

__all__ = ["GaussianDistribution"]
