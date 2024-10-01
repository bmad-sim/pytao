from __future__ import annotations

import ctypes
import dataclasses
import io
import logging
import os
import pickle
import queue
import subprocess
import sys
import tempfile
import threading
import typing
from typing import Any, Callable, Dict, List, Optional, Union, cast

import numpy as np
from typing_extensions import Literal, NotRequired, TypedDict, override

from .interface_commands import Tao, TaoStartup
from .tao_ctypes.core import TaoCommandError
from .tao_ctypes.util import error_filter_context

logger = logging.getLogger(__name__)

AnyTao = Union[Tao, "SubprocessTao"]

Command = Literal["quit", "init", "cmd", "cmd_real", "cmd_integer", "function"]
SupportedKwarg = Union[float, int, str, bool, bytes, dict, list, tuple, set, np.ndarray]


class SubprocessRequest(TypedDict):
    command: Command
    arg: str
    kwargs: NotRequired[Dict[str, Any]]
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
        data["result"] = dict_to_array(cast(SerializedArray, res))
    logger.debug(f"<- {data}")
    return data


def write_pickled_data(file, data):
    """
    Write pickled data to the file (named pipe / FIFO).

    Parameters
    ----------
    file : file-like object
    data :
        Picklable data
    """
    logger.debug(f"-> {data}")
    message_bytes = pickle.dumps(data) + b"\n"
    to_write = bytes(ctypes.c_uint32(len(message_bytes))) + message_bytes
    file.write(to_write)
    file.flush()


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


def deserialize_value(value: SupportedKwarg):
    if isinstance(value, (float, int, str, bytes, bool)):
        return value
    if isinstance(value, dict):
        if "__type__" in value:
            return dict_to_array(cast(SerializedArray, value))
        return {key: deserialize_value(value) for key, value in value.items()}
    if isinstance(value, (list, tuple, set)):
        return type(value)(deserialize_value(v) for v in value)
    if isinstance(value, np.ndarray):
        return value
    raise ValueError(f"Unsupported data type for deserialization: {type(value)}")


def serialize_value(value: Any):
    if isinstance(value, (float, int, str, bytes)):
        return value
    if isinstance(value, dict):
        return {key: serialize_value(value) for key, value in value.items()}
    if isinstance(value, np.ndarray):
        return array_to_dict(value)
    if isinstance(value, (list, tuple, set)):
        return type(value)(serialize_value(v) for v in value)
    raise ValueError(f"Unsupported data type for serialization: {type(value)}")


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
    """
    Tao subprocess Pipe helper.

    This corresponds to a single Tao subprocess.  For a new subprocess,
    instantiate another `_TaoPipe` instance.
    """

    _init_queue: queue.Queue
    _subproc: subprocess.Popen
    _fifo: Optional[io.BufferedReader]
    _subprocess_monitor_thread: Optional[threading.Thread]

    def __init__(self):
        self._init_queue = queue.Queue(maxsize=1)
        self._subproc = self._init_subprocess()

    @property
    def alive(self) -> bool:
        """Subprocess is still running."""
        # No exit code -> is still running
        return self._subproc.poll() is None

    def close(self):
        """Close the pipe."""
        try:
            self.send_receive("quit", "", raises=False)
        except TaoDisconnectedError:
            pass

    def _tao_subprocess(self):
        """Subprocess monitor thread.  Cleans up after the subprocess ends."""

        with tempfile.TemporaryDirectory(suffix="_pytao_subproc") as tempdir:
            fifo_path = os.path.join(tempdir, "fifo")
            os.mkfifo(fifo_path, mode=0o600)
            try:
                subproc = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "pytao.subproc_main",
                        fifo_path,
                    ],
                    stdin=subprocess.PIPE,
                )
            except Exception as ex:
                # Report the exception back to the main thread so it can be
                # re-raised appropriately.
                self._init_queue.put(ex)
                raise

            with open(fifo_path, "rb") as self._fifo:
                self._subproc = subproc
                self._init_queue.put(subproc)

                assert subproc.stdin is not None
                try:
                    code = subproc.wait()
                    if code != 0:
                        logger.warning(f"Subprocess exited with error code {code}")
                    else:
                        logger.debug("Subprocess exited without error")
                finally:
                    self._subprocess_monitor_thread = None
                    self._fifo = None

    def _send(self, cmd: Command, argument: str, **kwargs: SupportedKwarg):
        """Send `cmd` with `argument` over the pipe."""
        assert self._subproc.stdin is not None
        req: SubprocessRequest = {"command": cmd, "arg": argument}
        ctx = error_filter_context.get()
        if kwargs:
            req["kwargs"] = serialize_value(kwargs)
        if ctx is not None:
            req["capture_ctx"] = dataclasses.asdict(ctx)
        return write_pickled_data(self._subproc.stdin, req)

    def _receive(self) -> SubprocessResult:
        """Read back the command result from the subprocess."""
        if self._fifo is None:
            raise TaoDisconnectedError("The SubprocessTao pipe is closed")
        return read_pickled_data(self._fifo)

    def _init_subprocess(self) -> subprocess.Popen:
        """Initialize the Tao subprocess, the pipe, and monitor thread."""
        logger.debug("Initializing Tao subprocess")
        self._subprocess_monitor_thread = threading.Thread(
            target=self._tao_subprocess,
            daemon=True,
        )
        self._subprocess_monitor_thread.start()
        start = self._init_queue.get()
        if not isinstance(start, subprocess.Popen):
            if isinstance(start, Exception):
                raise start
            else:
                raise NotImplementedError(
                    f"Failed to start Tao subprocess, unknown error: {type(start).__name__}"
                )
        self._subproc = start

        return self._subproc

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

    def send_receive_custom(self, func: Callable, kwargs: Dict[str, SupportedKwarg]):
        """
        Run a custom function in the subprocess and retrieve its result.

        Parameters
        ----------
        cmd : one of {"init", "cmd", "cmd_real", "cmd_integer", "quit", "function"}
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
        if not callable(func):
            raise ValueError(f"Object of type{type(func).__name__} is not callable")
        if not func.__module__ or func.__module__ == "__main__":
            raise ValueError(f"Function {func.__name__} is not in an importable module")
        if func.__name__ == "__lambda__":
            raise ValueError(f"Function {func.__name__} is a lambda function")
        try:
            self._send("function", f"{func.__module__}.{func.__name__}", **kwargs)
            received = self._receive()
            result = _get_result(received, raises=True)
            return deserialize_value(result)
        except BrokenPipeError:
            raise TaoCommandError(
                f"Function {func.__name__}() was unable to complete as the subprocess "
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

    _subproc_pipe_: Optional[_TaoPipe]

    def __init__(self, *args, **kwargs):
        self._subproc_pipe_ = None
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
        if not self._subproc_pipe_:
            return False
        return self._subproc_pipe_.alive

    def close_subprocess(self) -> None:
        """Close the Tao subprocess."""
        if self._subproc_pipe_ is not None:
            self._subproc_pipe_.close()
        self._subproc_pipe_ = None

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
        if not self.subprocess_alive:
            raise TaoDisconnectedError(
                "Tao subprocess is no longer running. Make a new `SubprocessTao` "
                "object or reinitialize with `.init()`."
            )
        assert self._subproc_pipe_ is not None
        output, result = self._subproc_pipe_.send_receive(command, tao_cmdline, raises=raises)
        output_lines = output.splitlines()
        self._last_output = output_lines
        return result

    def subprocess_call(self, func: Callable, **kwargs):
        """
        Run a custom function in the subprocess.

        The function must be readily importable by Python and not a dynamically
        created function or `lambda`.  The first argument passed will be the
        `tao` object, and the remainder of the arguments are user-specified by
        keyword only.
        """
        if not self.subprocess_alive:
            raise TaoDisconnectedError(
                "Tao subprocess is no longer running. Make a new `SubprocessTao` "
                "object or reinitialize with `.init()`."
            )
        assert self._subproc_pipe_ is not None
        return self._subproc_pipe_.send_receive_custom(func, kwargs)

    @override
    def _init(self, startup: TaoStartup):
        self._reset_graph_managers()
        if not self.subprocess_alive:
            logger.debug("Reinitializing Tao subprocess")
            self._subproc_pipe_ = _TaoPipe()

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
