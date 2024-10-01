from __future__ import annotations

import contextlib
from functools import partial
import logging
import sys
import traceback

import numpy as np

from .interface_commands import Tao
from .subproc import (
    SubprocessErrorResult,
    SubprocessRequest,
    SubprocessResult,
    SubprocessSuccessResult,
    TaoDisconnectedError,
    array_to_dict,
    deserialize_value,
    read_pickled_data,
    write_pickled_data,
)
from .tao_ctypes.core import TaoCommandError
from .tao_ctypes.util import filter_tao_messages_context, import_by_name

logger = logging.getLogger(__name__)


def _tao_subprocess(output_fifo_filename: str) -> None:
    logger.debug("Tao subprocess handler started")

    tao = None

    def run_custom_function(message: SubprocessRequest, *_):
        func_name = message["arg"]
        func = import_by_name(func_name)
        kwargs = deserialize_value(message.get("kwargs", {}))
        return func(tao, **kwargs)

    def run_tao_command(message: SubprocessRequest):
        nonlocal tao

        command = message["command"]
        if command == "quit":
            sys.exit(0)

        arg = message.get("arg", None)
        capture_context_options = message.get("capture_ctx", None)
        if not capture_context_options:
            capture_context = contextlib.nullcontext()
        else:
            capture_context = filter_tao_messages_context(**capture_context_options)

        with capture_context:
            if command == "init":
                if tao is None:
                    tao = Tao(arg)
                    return tao.init_output
                return tao.init(arg)

            if tao is None:
                raise TaoCommandError("Tao object not yet initialized")

            try:
                func = {
                    "cmd": tao.cmd,
                    "cmd_real": tao.cmd_real,
                    "cmd_integer": tao.cmd_integer,
                    "function": partial(run_custom_function, message),
                }[command]
            except KeyError:
                raise RuntimeError(f"Unexpected Tao subprocess command: {command}")

            return func(arg)

    def make_response(message) -> SubprocessResult:
        try:
            raw_result = run_tao_command(message)
            if isinstance(raw_result, np.ndarray):
                raw_result = array_to_dict(raw_result)

            success: SubprocessSuccessResult = {
                "result": raw_result,
                "tao_output": "\n".join(getattr(tao, "last_output", "")),
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
