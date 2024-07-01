"""
This package provides the pexpect implementation of the interface between
tao and python.  The ctypes implementaion is much faster and should be
used if possible.
"""

from .tao_pipe import tao_io

__all__ = ["tao_io"]
