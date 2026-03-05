"""Neural network layers — re-exported from the active backend."""

# Own modules
from illia import __get_backend__, _check_backend_switch, _lock_backend

_check_backend_switch()
_lock_backend()

if __get_backend__ == "torch":
    # Own modules
    from illia.backend.torch.nn import Conv2d, Linear
elif __get_backend__ == "tensorflow":
    # Own modules
    from illia.backend.tensorflow.nn import Conv2d, Linear
elif __get_backend__ == "jax":
    # Own modules
    from illia.backend.jax.nn import Conv2d, Linear

__all__ = ["Conv2d", "Linear"]
