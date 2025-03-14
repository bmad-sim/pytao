from __future__ import annotations
import argparse
import code
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
    parser = argparse.ArgumentParser(description="PyTao command line interface")
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


def init(ipython: bool):
    python_args, init_args = split_pytao_tao_args(sys.argv[1:])

    startup_message = f"Initializing Tao object with: {init_args}"
    print("-" * len(startup_message))
    print(startup_message)
    print()

    if ipython:
        print("Type `tao.` and hit tab to see available commands.")
    else:
        print("The `tao` object is available.")
        print("Tab completion not available in basic mode.")
        print("To enable tab completion, install IPython: pip install ipython")

    print("-" * len(startup_message))

    plot = os.environ.get("PYTAO_PLOT", python_args.pyplot or "tao").lower()

    tao_cls = SubprocessTao if python_args.pysubprocess else Tao
    tao = tao_cls(init=init_args, plot=plot)

    user_ns = {"tao": tao}
    if plot == "mpl":
        import matplotlib.pyplot as plt

        user_ns["plt"] = plt
        plt.ion()
        print()
        print("* Matplotlib mode configured. Pyplot available as `plt`. *")

    if python_args.pylog:
        logger.setLevel(python_args.pylog)
        logging.basicConfig()
    return python_args, user_ns


def main_python():
    python_args, user_ns = init(ipython=False)

    # Handle command or script execution
    if python_args.pycommand:
        exec(python_args.pycommand, user_ns)

    if python_args.pyscript:
        with open(python_args.pyscript) as f:
            script_content = f.read()
        exec(script_content, user_ns)

    # If no command or script specified or if script should be followed by interactive mode
    if not (python_args.pycommand or python_args.pyscript) or python_args.pyscript:
        console = code.InteractiveConsole(locals=user_ns)
        console.interact(banner="")

    return 0


def main_ipython():
    import IPython

    python_args, user_ns = init(ipython=True)

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


def main():
    try:
        import IPython  # noqa
    except ImportError:
        main_python()
    else:
        main_ipython()
