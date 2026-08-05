from __future__ import annotations
import logging

from .startup import TaoStartup
from .core import (
    TaoCommandError,
    TaoInitializationError,
    TaoSharedLibraryNotFoundError,
    configure_logging,
)
from .errors import (
    TaoException,
    filter_tao_messages,
    filter_tao_messages_context,
    get_log_mode,
    set_log_mode,
)
from .subproc import (
    AnyTao,
    SubprocessTao,
    TaoInitResult,
    parallel_subprocess_taos,
)
from .tao import Tao

logging.getLogger("pytao").addHandler(logging.NullHandler())

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "AnyTao",
    "SubprocessTao",
    "Tao",
    "TaoCommandError",
    "TaoException",
    "TaoInitResult",
    "TaoInitializationError",
    "TaoSharedLibraryNotFoundError",
    "TaoStartup",
    "configure_logging",
    "filter_tao_messages",
    "filter_tao_messages_context",
    "get_log_mode",
    "parallel_subprocess_taos",
    "set_log_mode",
]
