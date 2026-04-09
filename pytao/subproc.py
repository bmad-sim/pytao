from __future__ import annotations

import contextlib
import io
import logging
import os
import pickle
import queue
import struct
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Literal, Optional, Union, cast

import numpy as np
from typing_extensions import NotRequired, TypedDict, override

from .errors import TaoCommandError, TaoInitializationError
from .startup import TaoStartup
from .tao import Tao

logger = logging.getLogger(__name__)

AnyTao = Union[Tao, "SubprocessTao"]

Command = Literal[
    "quit",
    "init",
    "cmd",
    "cmd_real",
    "cmd_integer",
    "function",
]
SupportedKwarg = Union[float, int, str, bool, bytes, dict, list, tuple, set, np.ndarray]


class SubprocessRequest(TypedDict):
    command: Command
    arg: str
    kwargs: NotRequired[dict[str, Any]]


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


def write_pickled_data(file, data):
    """Write pickled data through the pipe."""
    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.flush()


def read_pickled_data(pipe):
    """Read pickled data from the pipe."""
    try:
        return pickle.load(pipe)
    except EOFError:
        raise TaoDisconnectedError(
            "Subprocess data transfer reached end of file (EOF)"
        ) from None


def _get_result(
    value: SubprocessResult,
    propagate_exceptions: bool = True,
    initializing: bool = False,
):
    """
    Pick out the result data from the subprocess return value.

    Parameters
    ----------
    value : dict
        The deserialized dictionary result from the subprocess.
    propagate_exceptions : bool, optional
        Re-raise Python exceptions from the subprocess (e.g.,
        TaoInitializationError, uninitialized Tao). This does NOT
        control Tao-level error checking of output lines — that is
        handled separately by ``_check_output_lines`` in the parent.
    initializing : bool, optional
        If True, re-raised exceptions use TaoInitializationError.

    Returns
    -------
    result :
        The deserialized result of the command from the subprocess.
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
        if propagate_exceptions:
            err_cls = TaoInitializationError if initializing else TaoCommandError
            ex = err_cls(
                f"Tao in subprocess raised {error_cls}: {error}",
                tao_output=tao_output,
            )
            if hasattr(ex, "add_note") and callable(ex.add_note):
                ex.add_note(f"Subprocess {tb}")
            raise ex
        logger.error(f"Tao in subprocess raised an error {error_cls}:\n{tb}")
        return tao_output

    raise RuntimeError(f"Unexpected pipe response: {value}")


_BEAM_TRACK_SHM_FMT = "i"
_BEAM_TRACK_SHM_SIZE = struct.calcsize(_BEAM_TRACK_SHM_FMT)


class _TaoPipe:
    """
    Tao subprocess Pipe helper.

    This corresponds to a single Tao subprocess.  For a new subprocess,
    instantiate another `_TaoPipe` instance.
    """

    _init_queue: queue.Queue
    _subproc: subprocess.Popen
    _fifo: io.BufferedReader | None
    _subprocess_monitor_thread: threading.Thread | None
    _subprocess_env: dict[str, str]
    _beam_track_shm: SharedMemory

    def __init__(self, env: dict[str, str]):
        self._init_queue = queue.Queue(maxsize=1)
        self._subprocess_env = env.copy()
        self._beam_track_shm = SharedMemory(create=True, size=_BEAM_TRACK_SHM_SIZE)
        struct.pack_into(_BEAM_TRACK_SHM_FMT, self._beam_track_shm.buf, 0, -1)
        self._subproc = self._init_subprocess()

    @property
    def alive(self) -> bool:
        """Subprocess is still running."""
        # No exit code -> is still running
        return self._subproc.poll() is None

    def close(self) -> None:
        """Close the pipe."""
        try:
            self.send_receive("quit", "", propagate_exceptions=False)
        except TaoDisconnectedError:
            pass
        self._close_beam_track_shm()

    def close_forcefully(self) -> None:
        """Close the pipe."""
        self._subproc.terminate()
        self._close_beam_track_shm()

    def _close_beam_track_shm(self) -> None:
        self._beam_track_shm.close()
        self._beam_track_shm.unlink()

    def read_beam_track_element(self) -> int:
        """Read the active beam track element index from shared memory."""
        return struct.unpack_from(_BEAM_TRACK_SHM_FMT, self._beam_track_shm.buf, 0)[0]

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
                        self._beam_track_shm.name,
                    ],
                    stdin=subprocess.PIPE,
                    env=self._subprocess_env,
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
        if kwargs:
            req["kwargs"] = kwargs
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

    def send_receive(self, cmd: Command, argument: str, propagate_exceptions: bool = True):
        """
        Send a command and receive a result through the pipe.

        Parameters
        ----------
        cmd : one of {"init", "cmd", "cmd_real", "cmd_integer", "quit", "function"}
            The command class to send.
        argument : str
            The argument to send to the command.
        propagate_exceptions : bool
            Re-raise Python exceptions from the subprocess.

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
            result = _get_result(
                received,
                propagate_exceptions=propagate_exceptions,
                initializing=cmd == "init",
            )
            return received["tao_output"], result
        except BrokenPipeError:
            if cmd == "quit":
                return "", None
            raise TaoCommandError(
                f"Tao command {cmd}({argument!r}) was unable to complete as the subprocess "
                f"has already exited."
            )

    def send_receive_custom(self, func: Callable, kwargs: dict[str, SupportedKwarg]):
        """
        Run a custom function in the subprocess and retrieve its result.

        Parameters
        ----------
        func: callable
            The function to call.
        kwargs : dict of str to SupportedKwarg
            Keyword arguments for the function.

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
            return _get_result(received, propagate_exceptions=True)
        except BrokenPipeError:
            raise TaoCommandError(
                f"Function {func.__name__}() was unable to complete as the subprocess "
                f"has already exited."
            )


@contextlib.contextmanager
def subprocess_timeout_context(
    taos: list[SubprocessTao],
    timeout: float,
    *,
    timeout_hook: Callable[[], None] | None = None,
):
    """
    Context manager to set a timeout for a block of SubprocessTao calls.

    Note that there is no possibility for a graceful timeout. In the event
    of a timeout, all subprocesses will be terminated.

    Parameters
    ----------
    taos : list of SubprocessTao
    timeout : float
        The timeout duration in seconds.
    timeout_hook : callable, optional
        An alternative hook to call when the timeouts occur.
        This replaces the built-in subprocess-closing hook.

    Yields
    ------
    None
        Yields control back to the calling context.

    Raises
    ------
    TimeoutError
        If the block of code does not execute within `when` seconds.
        The Tao subprocesses are forcefully terminated at this point.
    """
    timed_out = False

    def monitor():
        if evt.wait(timeout):
            return

        nonlocal timed_out
        timed_out = True

        if timeout_hook is not None:
            timeout_hook()
        else:
            for tao in taos:
                try:
                    tao.close_subprocess(force=True)
                except Exception:
                    logger.debug("Subprocess close fail", exc_info=True)

    evt = threading.Event()
    monitor_thread = threading.Thread(daemon=True, target=monitor)
    monitor_thread.start()
    try:
        yield
    except TaoDisconnectedError:
        if not timed_out:
            # Tao disconnected, but not due to our timeout
            raise
        # Otherwise, the reason for disconnection was our timeout closure.

    evt.set()
    monitor_thread.join()
    if timed_out:
        raise TimeoutError(
            f"Operation timed out after {timeout} seconds. Closing Tao subprocesses."
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

    To add a new environment variable in addition to the parent process
    environment:

        >>> import os
        >>> with SubprocessTao(init_file="...", env={**os.environ, "NEW_VAR": "NEW_VALUE"}) as tao:
        ...     print(tao.version())

    Parameters
    ----------
    env : dict[str, str] or None, optional
        Environment variables to use for the subprocess.  If None, defaults to
        `os.environ`.

    Attributes
    ----------
    subprocess_env : dict[str, str]
        Environment variables to use for the subprocess.  It is recommended to
        use a new `SubprocessTao` instance in order to update these environment
        variables. However, while this dictionary may be updated in place, it
        will only be applied after the next subprocess starts and initializes.
        That is, `tao.close_subprocess()` and `tao.init()`.
    """

    _subproc_pipe_: _TaoPipe | None

    def __init__(self, *args, env: dict[str, str] | None = None, **kwargs):
        self._subproc_pipe_ = None
        self.subprocess_env = dict(env if env is not None else os.environ)

        try:
            super().__init__(*args, **kwargs)
        except Exception:
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

    def close_subprocess(self, *, force: bool = False) -> None:
        """Close the Tao subprocess."""
        if self._subproc_pipe_ is not None:
            if force:
                self._subproc_pipe_.close_forcefully()
            else:
                self._subproc_pipe_.close()
        self._subproc_pipe_ = None

    @contextlib.contextmanager
    def timeout(self, when: float):
        """
        Context manager to set a timeout for a block of SubprocessTao calls.

        Note that there is no possibility for a graceful timeout. In the event
        of a timeout, the subprocess will be terminated.

        Parameters
        ----------
        when : float
            The timeout duration in seconds.

        Yields
        ------
        None
            Yields control back to the calling context.

        Raises
        ------
        TimeoutError
            If the block of code does not execute within `when` seconds.
            The Tao subprocess is forcefully terminated at this point.
        """
        with subprocess_timeout_context([self], timeout=when):
            yield

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
        """
        Send a command to the subprocess and process the result.

        The subprocess always runs Tao commands with ``raises=False``. All
        Tao-level error checking is done here in the parent process via
        ``_check_output_lines``, which respects the parent's
        ``error_filter_context``.

        Uncaught errors in the subprocess will be propagated regardless of
        ``raises``.
        """
        if not self.subprocess_alive:
            raise TaoDisconnectedError(
                "Tao subprocess is no longer running. Make a new `SubprocessTao` "
                "object or reinitialize with `.init()`."
            )
        assert self._subproc_pipe_ is not None
        output, result = self._subproc_pipe_.send_receive(command, tao_cmdline)
        output_lines = output.splitlines()
        self._last_output = output_lines

        if command == "init":
            raw_output, messages = self._check_output_lines(
                tao_cmdline, output_lines, raises=False
            )
            for msg in messages:
                self._log(tao_cmdline, msg)
        else:
            raw_output, messages = self._check_output_lines(
                tao_cmdline, output_lines, raises=raises
            )
            for msg in messages:
                self._log(tao_cmdline, msg)
            if command == "cmd":
                result = raw_output

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

    def _init_backend(self, startup: TaoStartup):
        # Backend initialization: this is the hook for either Tao or SubprocessTao
        # to do its thing.
        self._reset_graph_managers()
        if not self.subprocess_alive:
            logger.debug("Reinitializing Tao subprocess")
            self._subproc_pipe_ = _TaoPipe(env=self.subprocess_env)

        return self._send_command_through_pipe("init", startup.tao_init, raises=True)

    @override
    def cmd(self, cmd: str, raises: bool = True) -> list[str]:
        """Runs a command, and returns the output."""
        return self._send_command_through_pipe("cmd", cmd, raises=raises)

    @override
    def cmd_real(self, cmd: str, raises: bool = True) -> np.ndarray | None:
        """Runs a command, and returns a floating point array."""
        return cast(
            Optional[np.ndarray],
            self._send_command_through_pipe("cmd_real", cmd, raises=raises),
        )

    @override
    def cmd_integer(self, cmd: str, raises: bool = True) -> np.ndarray | None:
        """Runs a command, and returns an integer array."""
        return cast(
            Optional[np.ndarray],
            self._send_command_through_pipe("cmd_integer", cmd, raises=raises),
        )

    def get_active_beam_track_element(self) -> int:
        """Get the active element index being tracked via shared memory."""
        assert self._subproc_pipe_ is not None
        return self._subproc_pipe_.read_beam_track_element()
