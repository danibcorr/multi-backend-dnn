# Standard libraries
import pathlib
from typing import Final

ENV_OS_NAME: Final[str] = "ILLIA_BACKEND"
DEFAULT_BACKEND: Final[str] = "torch"
_BACKEND_DIR: Final[str] = pathlib.Path(__file__).parent / "backend"
