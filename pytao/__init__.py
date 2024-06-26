"""
pytao is the python interface to tao.  Contains backend implementations in both
ctypes and pexpect.  The gui package supports a GUI interface to tao, in
place of the tao command line interface, with matplotlib plotting capabilities.
pytao also has some pre-defined constructs for dealing with data from tao
in the util package.
"""

from .interface_commands import Tao
from .subproc import SubprocessTao
from .tao_ctypes import TaoModel, run_tao
from .tao_ctypes.evaluate import evaluate_tao
from .tao_interface import tao_interface
from .tao_pexpect import tao_io

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "tao_io",
    "TaoModel",
    "Tao",
    "run_tao",
    "evaluate_tao",
    "tao_interface",
    "SubprocessTao",
]
