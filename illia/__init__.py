# Standard libraries
import importlib
import os
from typing import Any, Optional
from functools import lru_cache

# Own modules
from illia.support import DEFAULT_BACKEND, ENV_OS_NAME, _BACKEND_DIR


@lru_cache
def _discover_backends() -> dict[str, list[str]]:
    """_summary_

    Returns:
        dict[str, list[str]]: _description_
    """

    result: dict[str, list[str]] = {}

    if not _BACKEND_DIR.is_dir():
        return result

    for backend in sorted(_BACKEND_DIR.iterdir()):
        if not backend.is_dir() or backend.name.startswith("_"):
            continue

        modules = [
            f"illia.backend.{backend.name}.{sub.name}"
            for sub in sorted(backend.iterdir())
            if sub.is_dir() and not sub.name.startswith("_")
        ]

        if modules:
            result[backend.name] = modules

    return result


class BackendManager:
    """Manages backend selection, validation and module loading."""

    _active_backend: Optional[str] = None
    _module_cache: dict[str, Any] = {}
    _backend_modules: dict[str, list[str]] = _discover_backends()

    @classmethod
    def get_backend(cls, backend_name: Optional[str] = None) -> str:
        """Resolve, validate and lock the active backend."""

        if backend_name is None:
            env_backend = os.environ.get(ENV_OS_NAME)
            backend_name = env_backend or cls._active_backend or DEFAULT_BACKEND

        if backend_name not in cls._backend_modules:
            raise ImportError(
                f"Backend '{backend_name}' doesn't exist. "
                f"Available: {list(cls._backend_modules.keys())}."
            )

        if cls._active_backend and cls._active_backend != backend_name:
            raise RuntimeError(
                f"Already using '{cls._active_backend}'. "
                f"Can't switch to '{backend_name}'. Restart to change."
            )

        cls._active_backend = backend_name
        return cls._active_backend

    @classmethod
    def get_available_backends(cls) -> list[str]:
        """Return all discovered backend names."""

        return list(cls._backend_modules.keys())

    @classmethod
    def is_backend_available(cls, backend: str) -> bool:
        """Check whether *backend* was discovered."""

        return backend.lower() in {b.lower() for b in cls._backend_modules}

    @classmethod
    def get_module(cls, module_type: str, backend: Optional[str] = None) -> Any:
        """Import and return a backend sub-module (``nn``, ``distributions``, …)."""

        backend = cls.get_backend(backend)
        dotted = f"illia.backend.{backend}.{module_type}"

        if dotted not in cls._module_cache:
            cls._module_cache[dotted] = importlib.import_module(dotted)

        return cls._module_cache[dotted]


is_backend_available = BackendManager.is_backend_available

BackendManager.get_backend()
