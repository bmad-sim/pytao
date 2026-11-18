from __future__ import annotations

import contextlib
import logging
import os
import pickle
import signal
import struct
import sys
import threading
import traceback
import warnings

from ._shmem_compat import SharedMemory
from .core import TaoCommandError
from .errors import TaoForkError
from .fork import fork_map_payloads
from .subproc import (
    _BEAM_TRACK_SHM_FMT,
    SubprocessErrorResult,
    SubprocessRequest,
    SubprocessResult,
    SubprocessSuccessResult,
    TaoDisconnectedError,
    read_pickled_data,
    write_pickled_data,
)
from .tao import Tao
from .util import import_by_name

logger = logging.getLogger(__name__)


def _beam_track_writer(
    tao: Tao,
    shm: SharedMemory,
    stop_event: threading.Event,
    rate: float = 0.05,
) -> None:
    """Daemon thread that writes the active beam track element to shared memory."""
    try:
        while not stop_event.is_set():
            idx = tao.so_lib.tao_c_get_beam_track_element()
            struct.pack_into(_BEAM_TRACK_SHM_FMT, shm.buf, 0, idx)
            stop_event.wait(rate)
    finally:
        shm.close()


class _Terminated(BaseException):
    """Raised on SIGTERM so that `fork_map` unwinds and kills its children."""


@contextlib.contextmanager
def _terminate_unwinds():
    def handler(signum, _frame):
        raise _Terminated(signum)

    previous = signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _reset_forked_child(output_fifo_fd: int) -> None:
    """
    Detach a forked child from this process's request/response pipes.

    An open FIFO copy would delay the parent's EOF; an open stdin could
    swallow requests.
    """
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    with contextlib.suppress(OSError):
        os.close(output_fifo_fd)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)


def _tao_subprocess(output_fifo_filename: str, beam_track_shm_name: str) -> None:
    logger.debug("Tao subprocess handler started")

    tao = None
    beam_track_stop = threading.Event()

    def run_custom_function(message: SubprocessRequest, *_):
        func_name = message["arg"]
        func = import_by_name(func_name)
        kwargs = message.get("kwargs", {})
        return func(tao, **kwargs)

    def run_fork_map(message: SubprocessRequest):
        kwargs = message.get("kwargs", {})
        func = pickle.loads(kwargs["func"])
        items = pickle.loads(kwargs["items"])

        try:
            with warnings.catch_warnings(), _terminate_unwinds():
                # Only the lock-free beam-track thread is running; fork is safe.
                warnings.filterwarnings(
                    "ignore", message=".*multi-threaded.*", category=DeprecationWarning
                )
                return fork_map_payloads(
                    tao,
                    func,
                    items,
                    max_workers=kwargs["max_workers"],
                    timeout=kwargs["timeout"],
                    raises=kwargs["raises"],
                )
        except (TaoForkError, TimeoutError) as ex:
            # Returned, not raised, so the parent re-raises the same type.
            return ex

    def run_tao_command(message: SubprocessRequest):
        nonlocal tao

        command = message["command"]
        arg = message["arg"]
        if command == "quit":
            sys.exit(0)

        if command == "init":
            if tao is None:
                tao = Tao(arg)
                shm = SharedMemory(name=beam_track_shm_name, create=False, track=False)
                threading.Thread(
                    daemon=True,
                    target=_beam_track_writer,
                    args=(tao, shm, beam_track_stop),
                ).start()
                return tao.init_output
            return tao.init(arg)

        if tao is None:
            raise TaoCommandError("Tao object not yet initialized")

        if command == "function":
            tao._last_output = []
            return run_custom_function(message, arg)

        if command == "fork_map":
            tao._last_output = []
            return run_fork_map(message)

        if command == "cmd":
            res = tao.cmd(arg, raises=False)  # sets _last_output
            return res

        if command == "cmd_real":
            tao.so_lib.tao_c_command(arg.encode("utf-8"))
            res = tao._read_array(float)
            tao.get_output()  # sets _last_output, resets array
            return res

        if command == "cmd_integer":
            tao.so_lib.tao_c_command(arg.encode("utf-8"))
            res = tao._read_array(int)
            tao.get_output()  # sets _last_output, resets array
            return res

        raise RuntimeError(f"Unexpected Tao subprocess command: {command}")

    def make_response(message) -> SubprocessResult:
        try:
            result = run_tao_command(message)
            success: SubprocessSuccessResult = {
                "result": result,
                "tao_output": "\n".join(getattr(tao, "_last_output", "")),
            }
            return success
        except Exception as ex:
            error: SubprocessErrorResult = {
                "error": str(ex),
                "error_cls": type(ex).__name__,
                "traceback": traceback.format_exc(),
                "tao_output": getattr(ex, "tao_output", ""),
            }
            return error
        finally:
            if tao is not None:
                tao.reset_output()

    with open(output_fifo_filename, "wb") as output_fifo:
        fifo_fd = output_fifo.fileno()
        os.register_at_fork(after_in_child=lambda: _reset_forked_child(fifo_fd))
        while True:
            message = read_pickled_data(sys.stdin.buffer)
            write_pickled_data(output_fifo, make_response(message))


if __name__ == "__main__":
    try:
        output_fifo_filename = sys.argv[1]
        beam_track_shm_name = sys.argv[2]
    except (IndexError, ValueError):
        print(
            f"Usage: {sys.executable} {__file__} (output_fifo) (beam_track_shm_name)",
            file=sys.stderr,
        )
        exit(1)

    try:
        _tao_subprocess(output_fifo_filename, beam_track_shm_name)
    except (TaoDisconnectedError, OSError):
        exit(1)
    except KeyboardInterrupt:
        logger.debug("Caught KeyboardInterrupt; exiting.")
        exit(0)
    except _Terminated:
        # Children are reaped; now die by SIGTERM as if it had never been caught.
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTERM)
