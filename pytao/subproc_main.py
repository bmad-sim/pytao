import os
import logging
import sys
import traceback

import numpy as np

from .interface_commands import Tao
from .subproc import read_pickled_data, write_pickled_data, array_to_dict


logger = logging.getLogger(__name__)


def _tao_subprocess(output_fd: int) -> None:
    logger.debug("Tao subprocess handler started")

    tao = Tao()

    with os.fdopen(output_fd, "wb") as output_pipe:
        while True:
            message = read_pickled_data(sys.stdin.buffer)
            try:
                command = message["command"]
                if command == "init":
                    output = tao.init(*message["args"])
                elif command == "cmd":
                    output = tao.cmd(*message["args"])
                elif command == "cmd_real":
                    output = tao.cmd_real(*message["args"])
                elif command == "cmd_integer":
                    output = tao.cmd_integer(*message["args"])
                elif command == "quit":
                    sys.exit(0)
                else:
                    output = "unknown command"
            except Exception as ex:
                write_pickled_data(
                    output_pipe,
                    {
                        "error": str(ex),
                        "error_cls": type(ex).__name__,
                        "traceback": traceback.format_exc(),
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
    _tao_subprocess(output_fd)
