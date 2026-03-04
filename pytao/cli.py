from __future__ import annotations

import argparse
import code
import copy
import logging
import os
import sys
from typing import Any

from pydantic import ConfigDict, dataclasses

from pytao.errors import TaoInitializationError

from .startup import TaoArgumentParser, TaoStartup, create_tao_cli_parser

logger = logging.getLogger("pytao")


@dataclasses.dataclass(config=ConfigDict(extra="forbid", validate_assignment=True))
class PytaoArgs(TaoStartup):
    pycommand: str | None = None
    pylog: str | None = None
    pyplot: str | None = None
    pyprefix: str = "`"
    pyscript: str | None = None

    pytao: bool = False
    pyinteractive: bool = True
    pyquiet: bool = False
    pysubprocess: bool = True


DESCRIPTION = ""


def create_argparser() -> argparse.ArgumentParser:
    parser = TaoArgumentParser(
        description="PyTao command-line interface",
        epilog=DESCRIPTION,
        prog="pytao",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    create_tao_cli_parser(parser)

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
        "--pyno-interactive",
        action="store_false",
        dest="pyinteractive",
        help="After running `pycommand`, do not enter interactive mode.",
    )
    parser.add_argument(
        "--pyquiet",
        action="store_true",
        help="Do not show any PyTao banner or welcome messages.",
    )
    parser.add_argument(
        "--pytao",
        action="store_true",
        help="Go straight to interactive Tao mode first.",
    )
    parser.add_argument(
        "--pylog",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level.",
    )
    parser.add_argument(
        "--pyprefix",
        default="`",
        help=(
            "Default prefix for the input text transformer. Every IPython line "
            "that starts with this character will turn into a `tao.cmd()` line."
        ),
    )
    parser.add_argument(
        "--pyno-subprocess",
        action="store_false",
        dest="pysubprocess",
        help="Do not launch Tao in a subprocess.",
    )
    return parser


def _get_implied_init_args(init_args: PytaoArgs) -> PytaoArgs:
    if not init_args.lattice_file and not init_args.init_file:
        init_args = copy.deepcopy(init_args)
        init_args.init_file = "tao.init"

    return init_args


def print_header(ipython: bool, startup_message: str, plot: str = "") -> None:
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

    if plot == "mpl":
        print()
        print("* Matplotlib mode configured. Pyplot available as `plt`. *")

    elif plot == "bokeh":
        print()
        print("* Bokeh mode configured. *")
        print()
        print(
            "No interactive window is available with Bokeh. To save plots for viewing with a browser: "
        )
        print("  tao.plot('beta', save='beta.html')`")


def init(argv, ipython: bool):
    parser = create_argparser()
    args = parser.parse_args(argv, namespace=PytaoArgs())

    plot = os.environ.get("PYTAO_PLOT", args.pyplot or "tao").lower()

    args = _get_implied_init_args(args)
    if not args.pyquiet:
        startup_message = f"Initializing Tao object with: {args.tao_init}"
        print_header(ipython=ipython, startup_message=startup_message, plot=plot)

    if args.pylog:
        logger.setLevel(args.pylog)
        logging.basicConfig()

    try:
        tao = args.run(use_subprocess=args.pysubprocess)
    except TaoInitializationError as ex:
        if "Tao will not be able to initialize with the following settings:" in str(ex):
            create_argparser().print_help()
            sys.exit(1)
        raise

    user_ns: dict[str, Any] = {"tao": tao}
    if plot == "mpl":
        import matplotlib.pyplot as plt

        user_ns["plt"] = plt
        plt.ion()

    if args.command:
        for line in tao.cmd(args.command, raises=False):
            print(line)

    return args, user_ns


def main_python():
    args, user_ns = init(sys.argv, ipython=False)

    # Handle command or script execution
    if args.pycommand:
        exec(args.pycommand, user_ns)

    if args.pyscript:
        with open(args.pyscript) as f:
            script_content = f.read()
        exec(script_content, user_ns)

    console = code.InteractiveConsole(locals=user_ns)
    console.interact(banner="")


def main_ipython():
    import IPython
    from traitlets.config import Config

    args, user_ns = init(sys.argv[1:], ipython=True)

    ipy_argv = ["--no-banner"]
    if args.pyinteractive:
        ipy_argv.append("-i")
    if args.pycommand:
        ipy_argv.append("-c")
        ipy_argv.append(args.pycommand)
    if args.pyscript:
        ipy_argv.append(args.pyscript)

    if len(ipy_argv) > 1:
        logger.debug("Initializing IPython with: %s", ipy_argv)

    conf = Config()
    conf.InteractiveShellApp.exec_lines = ["tao.register_cell_magic()"]

    if args.pyprefix:
        conf.InteractiveShellApp.exec_lines.append(
            f"tao.register_input_transformer({args.pyprefix!r})"
        )

    if args.pytao:
        conf.InteractiveShellApp.exec_lines.append("tao.shell()")

    return IPython.start_ipython(config=conf, user_ns=user_ns, argv=ipy_argv)


def main():
    try:
        import IPython  # noqa
    except ImportError:
        main_python()
    else:
        main_ipython()
