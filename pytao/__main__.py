from __future__ import annotations
import argparse
import logging
import os
import sys
from types import SimpleNamespace

from .interface_commands import Tao
from .subproc import SubprocessTao

logger = logging.getLogger("pytao")


class PytaoArgs(SimpleNamespace):
    pyplot: str | None
    pyscript: str | None
    pycommand: str | None
    pylog: str | None
    pysubprocess: bool


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PyTAO command line interface")
    parser.add_argument(
        "--pyplot",
        choices=["mpl", "bokeh"],
        help="Select plotting backend: Matplotlib (mpl) or bokeh",
    )
    parser.add_argument(
        "--pyscript",
        type=str,
        help="Python script filename to run at startup. May use `tao` object.",
    )
    parser.add_argument(
        "--pycommand",
        type=str,
        help="Python command to run at startup. May use `tao` object.",
    )
    parser.add_argument(
        "--pylog",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level.",
    )
    parser.add_argument(
        "--pysubprocess",
        action="store_true",
        help="Launch Tao in a subprocess.",
    )
    return parser


def split_pytao_tao_args(args: list[str]) -> tuple[PytaoArgs, str]:
    parser = create_argparser()
    pytao, tao = parser.parse_known_intermixed_args(args, namespace=PytaoArgs())
    return pytao, " ".join(tao)


def main():
    try:
        import IPython
    except ImportError as ex:
        print(f"IPython unavailable ({ex}); pytao interactive mode unavailable.")
        exit(1)

    python_args, init_args = split_pytao_tao_args(sys.argv[1:])

    startup_message = f"Initializing Tao object with: {init_args}"
    print("-" * len(startup_message))
    print(startup_message)
    print()
    print("Type `tao.` and hit tab to see available commands.")
    print("-" * len(startup_message))

    plot = os.environ.get("PYTAO_PLOT", python_args.pyplot or "tao").lower()

    tao_cls = SubprocessTao if python_args.pysubprocess else Tao
    tao = tao_cls(init=init_args, plot=plot)

    user_ns: dict = {"tao": tao}
    if plot == "mpl":
        import matplotlib.pyplot as plt

        user_ns["plt"] = plt
        plt.ion()
        print()
        print("* Matplotlib mode configured. Pyplot available as `plt`. *")

    if python_args.pylog:
        logger.setLevel(python_args.pylog)
        logging.basicConfig()

    ipy_argv = ["--no-banner"]
    if python_args.pycommand:
        ipy_argv.append("-c")
        ipy_argv.append(python_args.pycommand)
    if python_args.pyscript:
        ipy_argv.append("-i")
        ipy_argv.append(python_args.pyscript)

    if len(ipy_argv) > 1:
        logger.debug("Initializing IPython with: %s", ipy_argv)
    return IPython.start_ipython(user_ns=user_ns, argv=ipy_argv)


if __name__ == "__main__":
    main()
