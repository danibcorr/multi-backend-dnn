"""Public ``illia.distributions`` classes – backend-agnostic."""

# Own modules
from illia import BackendManager


class GaussianDistribution:
    """Learnable Gaussian distribution. Delegates to the active backend."""

    def __new__(cls, *args, **kwargs):
        impl = getattr(BackendManager.get_module("distributions"), "GaussianDistribution")
        return impl(*args, **kwargs)


__all__ = ["GaussianDistribution"]
