from __future__ import annotations

import logging
import sys
import traceback

from .core import TaoCommandError
from .subproc import (
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


def _tao_subprocess(output_fifo_filename: str) -> None:
    logger.debug("Tao subprocess handler started")

    tao = None

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
                return tao.init_output
            return tao.init(arg)

        if tao is None:
            raise TaoCommandError("Tao object not yet initialized")

        if command == "get_active_beam_track_element":
            tao._last_output = []
            return tao.get_active_beam_track_element()

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
    except (IndexError, ValueError):
        print(
            f"Usage: {sys.executable} {__file__} (output_file_descriptor)",
            file=sys.stderr,
        )
        exit(1)

    try:
        _tao_subprocess(output_fifo_filename)
    except (TaoDisconnectedError, OSError):
        exit(1)
    except KeyboardInterrupt:
        logger.debug("Caught KeyboardInterrupt; exiting.")
        exit(0)
