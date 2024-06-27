import sys

from .subproc import _tao_subprocess


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
