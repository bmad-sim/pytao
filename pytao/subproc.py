from __future__ import annotations

import ctypes
import dataclasses
import logging
import os
import pickle
import subprocess
import sys
import threading
import typing
from typing import Any, List, Union, cast

import numpy as np
from bokeh.core.property.singletons import Optional
from typing_extensions import Literal, NotRequired, TypedDict, override

from .interface_commands import Tao, TaoStartup
from .tao_ctypes.core import TaoCommandError
from .tao_ctypes.util import error_filter_context

logger = logging.getLogger(__name__)

AnyTao = Union[Tao, "SubprocessTao"]

Command = Literal["quit", "init", "cmd", "cmd_real", "cmd_integer"]


class SubprocessRequest(TypedDict):
    command: Command
    arg: str
    capture_ctx: NotRequired[dict]


class SubprocessErrorResult(TypedDict):
    error: str
    error_cls: str
    traceback: str
    tao_output: str


class SubprocessSuccessResult(TypedDict):
    result: Any
    tao_output: str


SubprocessResult = Union[SubprocessSuccessResult, SubprocessErrorResult]


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


def read_pickled_data(pipe):
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


def _get_result(value: SubprocessResult, raises: bool = True):
    """
    Pick out the result data from the subprocess return value.

    Parameters
    ----------
    value : dict
        The deserialized dictionary result from the subprocess.
    raises : bool, optional
        Raise on errors.

    Returns
    -------
    result :
        The deserialized result of the command from the subprocess.
    output : list of str
        Tao's raw output.
    """
    if not isinstance(value, dict):
        raise ValueError(f"Unexpected result type: {type(value)}")

    if "result" in value:
        return value["result"]
    if "error" in value:
        error = value.get("error")
        error_cls = value.get("error_cls")
        tb = value.get("traceback", "")
        tao_output = value.get("tao_output", "")
        if raises:
            ex = TaoCommandError(
                f"Tao in subprocess raised {error_cls}: {error}",
                tao_output=tao_output,
            )
            if hasattr(ex, "add_note") and callable(ex.add_note):
                ex.add_note(f"Subprocess {tb}")
            raise ex
        logger.error(f"Tao in subprocess raised an error {error_cls}:\n{tb}")
        return tao_output

    raise RuntimeError(f"Unexpected pipe response: {value}")


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
        in_fd = self._subproc_in_fd
        out_fd = self._subproc_out_fd
        try:
            code = subproc.wait()
            if code != 0:
                logger.warning(f"Subprocess exited with error code {code}")
            else:
                logger.debug("Subprocess exited without error")
            for fd in (in_fd, out_fd):
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
        finally:
            self._subproc = None
            self._read_pipe = None
            self._subproc_in_fd = None
            self._subproc_out_fd = None
            self._monitor_thread = None

    def _send(self, cmd: Command, argument: str):
        """Send `cmd` with `argument` over the pipe."""
        if self._subproc is None:
            self._subproc = self._init_subprocess()

        assert self._subproc.stdin is not None
        req: SubprocessRequest = {"command": cmd, "arg": argument}
        ctx = error_filter_context.get()
        if ctx is not None:
            req["capture_ctx"] = dataclasses.asdict(ctx)
        return write_pickled_data(self._subproc.stdin, req)

    def _receive(self) -> SubprocessResult:
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

    def send_receive(self, cmd: Command, argument: str, raises: bool):
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

        Returns
        -------
        tao_output : str
            Raw Tao output for the command.
        result : Any
            The final deserialized result of the command - which may be an ndarray,
            a list of dictionaries, a list of strings, and so on.
        """
        try:
            self._send(cmd, argument)
            received = self._receive()
            result = _get_result(received, raises=raises)
            return received["tao_output"], result
        except BrokenPipeError:
            raise TaoCommandError(
                f"Tao command {cmd}({argument!r}) was unable to complete as the subprocess "
                f"has already exited."
            )


class SubprocessTao(Tao):
    """
    Subprocess helper for Tao.

    This special version of the `Tao` class executes a Python subprocess which
    interacts with Tao through ctypes.

    This can be used exactly as the normal `Tao` object with the primary added
    benefit that Fortran crashes will not affect the main Python process.

    For full parameter information, see the `Tao` class documentation.

    Usage
    -----

    When creating many `SubprocessTao` objects, ensure to close the subprocess
    when done with it.  This can be done manually:

        >>> tao.close_subprocess()

    Or automatically by way of a context manager:

        >>> with SubprocessTao(init_file="$ACC_ROOT_DIR/bmad-doc/tao_examples/cbeta_cell/tao.init", plot=True) as tao:
        ...     tao.plot("floor")
    """

    def __init__(self, *args, **kwargs):
        self._pipe = _TaoPipe()
        try:
            super().__init__(*args, **kwargs)
        except Exception:
            # In case we don't make a usable SubprocessTao object, close the
            # subprocess so it doesn't linger.
            try:
                self.close_subprocess()
            except Exception:
                pass
            raise

    @property
    def subprocess_alive(self) -> bool:
        """Subprocess is still running."""
        if not self._pipe._subproc:
            return False
        # No exit code -> is still running
        return self._pipe._subproc.poll() is None

    def close_subprocess(self) -> None:
        """Close the Tao subprocess."""
        self._pipe.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close_subprocess()

    def __del__(self) -> None:
        try:
            self.close_subprocess()
        except Exception:
            pass

    def _send_command_through_pipe(self, command: Command, tao_cmdline: str, raises: bool):
        output, result = self._pipe.send_receive(command, tao_cmdline, raises=raises)
        output_lines = output.splitlines()
        self._last_output = output_lines
        return result

    @override
    def _init(self, startup: TaoStartup):
        self._reset_graph_managers()
        return self._send_command_through_pipe("init", startup.tao_init, raises=True)

    @override
    def cmd(self, cmd: str, raises: bool = True) -> List[str]:
        """Runs a command, and returns the output."""
        res = self._send_command_through_pipe("cmd", cmd, raises=raises)
        return cast(List[str], res)

    @override
    def cmd_real(self, cmd: str, raises: bool = True) -> Optional[np.ndarray]:
        """Runs a command, and returns a floating point array."""
        res = self._send_command_through_pipe("cmd_real", cmd, raises=raises)
        return cast(Optional[np.ndarray], res)

    @override
    def cmd_integer(self, cmd: str, raises: bool = True) -> Optional[np.ndarray]:
        """Runs a command, and returns an integer array."""
        res = self._send_command_through_pipe("cmd_integer", cmd, raises=raises)
        return cast(Optional[np.ndarray], res)
