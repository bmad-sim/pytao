import ctypes
import logging
import os
import pickle
import subprocess
import sys
import threading
import traceback
import typing

import numpy as np

from .interface_commands import Tao

logger = logging.getLogger(__name__)


class TaoDisconnectedError(Exception):
    """The Tao subprocess quit, crashed, or otherwise disconnected."""

    pass


def read_length_from_pipe(pipe) -> int:
    """
    Get the buffer length from the pipe.

    Parameters
    ----------
    pipe : file-like object

    Returns
    -------
    int
        Buffer length.

    Raises
    ------
    TaoDisconnectedError
        If the pipe is closed and a length cannot be read.
    """
    length_bytes = pipe.read(ctypes.sizeof(ctypes.c_uint32))
    if not length_bytes:
        logger.debug("<- disconnected")
        raise TaoDisconnectedError()
    length = ctypes.c_uint32.from_buffer(bytearray(length_bytes))
    return length.value


def read_pickled_data(pipe) -> dict:
    """
    Read pickled data from the pipe.

    Parameters
    ----------
    pipe : file-like object

    Returns
    -------
    dict
        Deserialized data.  Numpy arrays are handled automatically.

    Raises
    ------
    TaoDisconnectedError
        If the pipe is closed and a length cannot be read.
    """
    length = read_length_from_pipe(pipe)
    buffer = pipe.read(length)
    if len(buffer) != length:
        logger.debug("<- disconnected")
        raise TaoDisconnectedError()
    data = pickle.loads(buffer)
    res = data.get("result", None)
    if isinstance(res, dict) and "__type__" in res:
        data["result"] = dict_to_array(res)
    logger.debug(f"<- {data}")
    return data


def write_pickled_data(pipe, data):
    """
    Read pickled data from the pipe.

    Parameters
    ----------
    pipe : file-like object, or file
    data :
        Picklable data
    """
    logger.debug(f"-> {data}")
    message_bytes = pickle.dumps(data) + b"\n"
    to_write = bytes(ctypes.c_uint32(len(message_bytes))) + message_bytes
    if isinstance(pipe, int):
        os.write(pipe, to_write)
    else:
        pipe.write(to_write)
        pipe.flush()


class SerializedArray(typing.TypedDict):
    """Representation of a serialized numpy array as a picklable dict."""

    __type__: typing.Literal["array"]
    shape: typing.Tuple[int, ...]
    dtype: np.dtype
    data: bytes


def array_to_dict(arr: np.ndarray) -> SerializedArray:
    return {
        "__type__": "array",
        "shape": arr.shape,
        "dtype": arr.dtype,
        "data": arr.tobytes(),
    }


def dict_to_array(data: SerializedArray) -> np.ndarray:
    """Deserialize a SerializedArray back to a np.ndarray."""
    assert isinstance(data, dict)
    assert data["__type__"] == "array"
    arr = np.frombuffer(bytearray(data["data"]), dtype=data["dtype"])
    return arr.reshape(data["shape"])


def _get_result(value, raises: bool = True):
    """
    Pick out the result data from the subprocess return value.

    Parameters
    ----------
    value : dict
        The deserialized dictionary result from the subprocess.
    raises : bool, optional
        Raise on errors.
    """
    if not isinstance(value, dict):
        raise ValueError(f"Unexpected result type: {type(value)}")
    if "result" in value:
        return value["result"]
    if "error" in value:
        error = value.get("error")
        error_cls = value.get("error_cls")
        tb = value.get("traceback", "")
        if raises:
            ex = RuntimeError(f"Tao in subprocess raised {error_cls}: {error}")
            if hasattr(ex, "add_note") and callable(ex.add_note):
                ex.add_note(f"Subprocess {tb}")
            raise ex
        logger.error(f"Tao in subprocess raised an error {error_cls}:\n{tb}")


class _TaoPipe:
    """Tao subprocess Pipe helper."""

    def __init__(self):
        self._subproc = None
        self._read_pipe = None
        self._monitor_thread = None
        self._subproc_in_fd = None
        self._subproc_out_fd = None

    def close(self):
        """Close the pipe."""
        if self._subproc_out_fd is None:
            return

        try:
            self.send_receive("quit", "", raises=False)
        except TaoDisconnectedError:
            pass

    def _tao_subprocess_monitor(self, subproc: subprocess.Popen):
        """Subprocess monitor thread.  Cleans up after the subprocess ends."""
        if subproc is None:
            return
        in_fd = self._subproc_in_fd
        out_fd = self._subproc_out_fd
        try:
            code = subproc.wait()
            if code != 0:
                logger.warning(f"Subprocess exited with error code {code}")
            else:
                logger.debug("Subprocess exited without error")
            if in_fd is not None:
                os.close(in_fd)
            if out_fd is not None:
                os.close(out_fd)
        finally:
            self._subproc = None
            self._read_pipe = None
            self._subproc_in_fd = None
            self._subproc_out_fd = None
            self._monitor_thread = None

    def _send(self, cmd: str, argument: str):
        """Send `cmd` with `argument` over the pipe."""
        if self._subproc is None:
            self._subproc = self._init_subprocess()

        assert self._subproc.stdin is not None
        return write_pickled_data(
            self._subproc.stdin,
            {"command": cmd, "args": [argument]},
        )

    def _receive(self):
        """Read back the command result from the subprocess."""
        return read_pickled_data(self._read_pipe)

    def _init_subprocess(self) -> subprocess.Popen:
        """Initialize the Tao subprocess, the pipe, and monitor thread."""
        logger.debug("Initializing Tao subprocess")
        assert self._subproc is None
        self._subproc_in_fd, self._subproc_out_fd = os.pipe()
        subproc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "pytao.subproc_main",
                str(self._subproc_out_fd),
            ],
            stdin=subprocess.PIPE,
            pass_fds=[self._subproc_out_fd],
        )
        self._read_pipe = os.fdopen(self._subproc_in_fd, "rb")
        assert subproc.stdin is not None
        self._monitor_thread = threading.Thread(
            target=self._tao_subprocess_monitor,
            args=(subproc,),
            daemon=True,
        )
        self._monitor_thread.start()
        return subproc

    def send_receive(self, cmd: str, argument: str, raises: bool):
        """
        Send a command and receive a result through the pipe.

        Parameters
        ----------
        cmd : one of {"init", "cmd", "cmd_real", "cmd_integer", "quit"}
            The command class to send.
        argument : str
            The argument to send to the command.
        raises : bool
            Raise if the subprocess command raises or disconnects.
        """
        self._send(cmd, argument)
        return _get_result(self._receive(), raises=raises)


class SubprocessTao(Tao):
    """
    Subprocess helper for Tao.

    This special version of the `Tao` class executes a Python subprocess which
    interacts with Tao through ctypes.

    This can be used exactly as the normal `Tao` object with the primary added
    benefit that Fortran crashes will not affect the main Python process.
    """

    def __init__(self, *args, **kwargs):
        self._pipe = _TaoPipe()
        super().__init__(*args, **kwargs)

    def close_subprocess(self):
        self._pipe.close()

    def init(self, cmd):
        """Initialize Tao with the given `cmd`."""
        return self._pipe.send_receive("init", cmd, raises=True)

    def cmd(self, cmd, raises=True):
        """Runs a command, and returns the output."""
        return self._pipe.send_receive("cmd", cmd, raises=raises)

    def cmd_real(self, cmd, raises=True):
        """Runs a command, and returns a floating point array."""
        return self._pipe.send_receive("cmd_real", cmd, raises=raises)

    def cmd_integer(self, cmd, raises=True):
        """Runs a command, and returns an integer array."""
        return self._pipe.send_receive("cmd_integer", cmd, raises=raises)
