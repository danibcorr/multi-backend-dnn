# Standard libraries
import importlib
import importlib.metadata
import os
import pkgutil
from functools import lru_cache
from typing import Final


__available_dnn_backends__: Final[tuple[str, ...]] = ("torch", "tensorflow", "jax")
__default_backend__: Final[str] = "torch"

__get_backend__: str = os.environ.get("ILLIA_BACKEND", __default_backend__).lower()
if __get_backend__ not in __available_dnn_backends__:
    raise ValueError(
        f"Unsupported backend '{__get_backend__}', "
        f"choose from {__available_dnn_backends__}."
    )

__version__: str = importlib.metadata.version("illia")


@lru_cache
def _discover_capabilities(backend: str) -> dict[str, frozenset[str]]:
    """Import ``illia.backend.<backend>`` and inspect each subpackage's ``__all__``."""
    base = f"illia.backend.{backend}"
    try:
        pkg = importlib.import_module(base)
    except ImportError:
        return {}

    caps: dict[str, frozenset[str]] = {}
    for finder, name, ispkg in pkgutil.iter_modules(pkg.__path__):
        if not ispkg:
            continue
        try:
            mod = importlib.import_module(f"{base}.{name}")
        except ImportError:
            continue
        all_names = getattr(mod, "__all__", None)
        if all_names:
            caps[name] = frozenset(all_names)
    return caps


def is_class_available(class_name: str, module_type: str, backend: str | None = None) -> bool:
    """Check if *class_name* exists in *module_type* for the given (or active) backend."""
    return class_name in _discover_capabilities(backend or __get_backend__).get(
        module_type, frozenset()
    )


def get_available_classes(module_type: str, backend: str | None = None) -> frozenset[str]:
    """Return all class names registered for *module_type* in the given (or active) backend."""
    return _discover_capabilities(backend or __get_backend__).get(
        module_type, frozenset()
    )


def get_backends_for_class(class_name: str, module_type: str) -> tuple[str, ...]:
    """Return every backend that provides *class_name* in *module_type*."""
    return tuple(
        b for b in __available_dnn_backends__
        if class_name in _discover_capabilities(b).get(module_type, frozenset())
    )

_backend_locked: bool = False


def _lock_backend() -> None:
    """Mark the backend as locked (called on first submodule import)."""
    global _backend_locked
    _backend_locked = True


def _check_backend_switch() -> None:
    """Raise if the env var changed after the backend was already loaded."""
    if not _backend_locked:
        return
    current_env = os.environ.get("ILLIA_BACKEND", __default_backend__).lower()
    if current_env != __get_backend__:
        raise RuntimeError(
            f"Backend already loaded as '{__get_backend__}'. "
            f"Cannot switch to '{current_env}'. Restart the process to change backends."
        )


__all__ = [
    "__default_backend__",
    "__get_backend__",
    "__available_dnn_backends__",
    "__version__",
    "is_class_available",
    "get_available_classes",
    "get_backends_for_class",
]
