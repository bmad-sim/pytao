import logging
import os
import sys
import traceback

import numpy as np

from .tao_ctypes.core import TaoCommandError
from .interface_commands import Tao
from .subproc import array_to_dict, read_pickled_data, write_pickled_data

logger = logging.getLogger(__name__)


def _tao_subprocess(output_fd: int) -> None:
    logger.debug("Tao subprocess handler started")

    tao = None

    with os.fdopen(output_fd, "wb") as output_pipe:
        while True:
            message = read_pickled_data(sys.stdin.buffer)
            try:
                command = message["command"]
                if command == "quit":
                    sys.exit(0)

                if command == "init":
                    if tao is None:
                        tao = Tao(*message["args"])
                        output = tao.init_output
                    else:
                        output = tao.init(*message["args"])
                elif tao is None:
                    raise TaoCommandError("Tao object not yet initialized")
                elif command == "cmd":
                    output = tao.cmd(*message["args"])
                elif command == "cmd_real":
                    output = tao.cmd_real(*message["args"])
                elif command == "cmd_integer":
                    output = tao.cmd_integer(*message["args"])
                else:
                    output = "unknown command"
            except Exception as ex:
                write_pickled_data(
                    output_pipe,
                    {
                        "error": str(ex),
                        "error_cls": type(ex).__name__,
                        "traceback": traceback.format_exc(),
                        "tao_output": getattr(ex, "tao_output", ""),
                    },
                )
            else:
                if isinstance(output, np.ndarray):
                    output = array_to_dict(output)

                write_pickled_data(output_pipe, {"result": output})


if __name__ == "__main__":
    try:
        output_fd = int(sys.argv[1])
    except (IndexError, ValueError):
        print(
            f"Usage: {sys.executable} {__file__} (output_file_descriptor)",
            file=sys.stderr,
        )
        exit(1)

    try:
        _tao_subprocess(output_fd)
    except OSError:
        exit(1)
    except KeyboardInterrupt:
        logger.debug("Caught KeyboardInterrupt; exiting.")
        exit(0)
