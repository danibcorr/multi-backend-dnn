"""Public ``illia.nn`` layer classes – backend-agnostic."""

# Own modules
from illia import BackendManager


class Linear:
    """Fully-connected layer. Delegates to the active backend."""

    def __new__(cls, *args, **kwargs):
        impl = getattr(BackendManager.get_module("nn"), "Linear")
        return impl(*args, **kwargs)


class Conv2d:
    """2-D convolution layer. Delegates to the active backend."""

    def __new__(cls, *args, **kwargs):
        impl = getattr(BackendManager.get_module("nn"), "Conv2d")
        return impl(*args, **kwargs)


__all__ = ["Conv2d", "Linear"]
