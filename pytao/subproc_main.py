from __future__ import annotations

import logging
import struct
import sys
import threading
import traceback
from multiprocessing.shared_memory import SharedMemory

from .core import TaoCommandError
from .subproc import (
    SubprocessErrorResult,
    SubprocessRequest,
    SubprocessResult,
    SubprocessSuccessResult,
    TaoDisconnectedError,
    _BEAM_TRACK_SHM_FMT,
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


def _tao_subprocess(output_fifo_filename: str, beam_track_shm_name: str) -> None:
    logger.debug("Tao subprocess handler started")

    tao = None
    beam_track_stop = threading.Event()

    def run_custom_function(message: SubprocessRequest, *_):
        func_name = message["arg"]
        func = import_by_name(func_name)
        kwargs = message.get("kwargs", {})
        return func(tao, **kwargs)

    def run_tao_command(message: SubprocessRequest):
        nonlocal tao

        command = message["command"]
        arg = message["arg"]
        if command == "quit":
            sys.exit(0)

        if command == "init":
            if tao is None:
                tao = Tao(arg)
                shm = SharedMemory(name=beam_track_shm_name, create=False)
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
