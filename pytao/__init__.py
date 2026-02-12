"""
pytao is the python interface to tao.  Contains backend implementations in both
ctypes and pexpect.  The gui package supports a GUI interface to tao, in
place of the tao command line interface, with matplotlib plotting capabilities.
pytao also has some pre-defined constructs for dealing with data from tao
in the util package.
"""

from .core import (
    TaoCommandError,
    TaoInitializationError,
    TaoSharedLibraryNotFoundError,
    TaoStartup,
)
from .errors import TaoException, filter_tao_messages, filter_tao_messages_context
from .subproc import AnyTao, SubprocessTao
from .tao import Tao
from .tao_interface import tao_interface
from .tao_pexpect import tao_io

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
    "TaoInitializationError",
    "TaoSharedLibraryNotFoundError",
    "TaoStartup",
    "filter_tao_messages",
    "filter_tao_messages_context",
    "tao_interface",
    "tao_io",
]
