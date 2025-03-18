from __future__ import annotations
import argparse
import code
import dataclasses
import logging
import os
import sys

from pytao.tao_ctypes.util import TaoInitializationError

from .interface_commands import Tao
from .subproc import SubprocessTao

logger = logging.getLogger("pytao")


@dataclasses.dataclass
class PytaoArgs:
    pycommand: str | None = None
    pylog: str | None = None
    pyplot: str | None = None
    pyprefix: str = "`"
    pyscript: str | None = None

    pytao: bool = False
    pyinteractive: bool = True
    pyquiet: bool = False
    pysubprocess: bool = True


DESCRIPTION = """

Tao options (use -- to negate any option):
  -beam_file <file_name>               File containing the tao_beam_init namelist.
  -beam_init_position_file <file_name> File containing initial particle positions.
  -building_wall_file <file_name>      Define the building tunnel wall
  -command <command_string>            Commands to run after startup file commands
  -data_file <file_name>               Define data for plotting and optimization
  -debug                               Debug mode for Wizards
  -disable_smooth_line_calc            Disable the smooth line calc used in plotting
  -external_plotting                   Tells Tao that plotting is done externally to Tao.
  -geometry <width>x<height>           Plot window geometry (pixels)
  -help                                Display this list of command line options
  -hook_init_file <file_name>          Init file for hook routines (Default = tao_hook.init)
  -init_file <file_name>               Tao init file
  -lattice_file <file_name>            Bmad lattice file
  -log_startup                         Write startup debugging info
  -no_stopping                         For debugging: Prevents Tao from exiting on errors
  -noinit                              Do not use Tao init file.
  -noplot                              Do not open a plotting window
  -nostartup                           Do not open a startup command file
  -no_rad_int                          Do not do any radiation integrals calculations.
  -plot_file <file_name>               Plotting initialization file
  -prompt_color <color>                Set color of prompt string. Default is blue.
  -reverse                             Reverse lattice element order?
  -rf_on                               Use "--rf_on" to turn off RF (default is now RF on)
  -quiet <level>                       Suppress terminal output when running a command file?
                                        Levels: "all" (default), "warnings".
  -slice_lattice <ele_list>            Discards elements from lattice that are not in the list
  -start_branch_at <ele_name>          Start lattice branch at element.
  -startup_file <file_name>            Commands to run after parsing Tao init file
  -symbol_import                       Import symbols defined in lattice files(s)?
  -var_file <file_name>                Define variables for plotting and optimization
"""


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PyTao command-line interface",
        epilog=DESCRIPTION,
        prog="pytao",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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


def split_pytao_tao_args(args: list[str]) -> tuple[PytaoArgs, str]:
    parser = create_argparser()
    pytao, tao = parser.parse_known_intermixed_args(args, namespace=PytaoArgs())
    return pytao, " ".join(tao)


def _get_implied_init_args(init_args: str) -> str:
    tao_init_parts = init_args.split()
    can_init = any(part.startswith(flag) for part in tao_init_parts for flag in {"-i", "-la"})
    if can_init:
        return init_args
    return f"{init_args.strip()} -init tao.init"


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


def init(ipython: bool):
    python_args, init_args = split_pytao_tao_args(sys.argv[1:])

    plot = os.environ.get("PYTAO_PLOT", python_args.pyplot or "tao").lower()

    if not python_args.pyquiet:
        implied_init_args = _get_implied_init_args(init_args)
        startup_message = f"Initializing Tao object with: {implied_init_args}"
        print_header(ipython=ipython, startup_message=startup_message, plot=plot)

    tao_cls = SubprocessTao if python_args.pysubprocess else Tao

    try:
        tao = tao_cls(init=init_args, plot=plot)
    except TaoInitializationError as ex:
        if "Tao will not be able to initialize with the following settings:" in str(ex):
            create_argparser().print_help()
            sys.exit(1)
        raise

    user_ns = {"tao": tao}
    if plot == "mpl":
        import matplotlib.pyplot as plt

        user_ns["plt"] = plt
        plt.ion()

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

    console = code.InteractiveConsole(locals=user_ns)
    console.interact(banner="")


def main_ipython():
    import IPython
    from traitlets.config import Config

    python_args, user_ns = init(ipython=True)

    ipy_argv = ["--no-banner"]
    if python_args.pyinteractive:
        ipy_argv.append("-i")
    if python_args.pycommand:
        ipy_argv.append("-c")
        ipy_argv.append(python_args.pycommand)
    if python_args.pyscript:
        ipy_argv.append(python_args.pyscript)

    if len(ipy_argv) > 1:
        logger.debug("Initializing IPython with: %s", ipy_argv)

    conf = Config()
    conf.InteractiveShellApp.exec_lines = ["tao.register_cell_magic()"]

    if python_args.pyprefix:
        conf.InteractiveShellApp.exec_lines.append(
            f"tao.register_input_transformer({python_args.pyprefix!r})"
        )

    if python_args.pytao:
        conf.InteractiveShellApp.exec_lines.append("tao.shell()")

    return IPython.start_ipython(config=conf, user_ns=user_ns, argv=ipy_argv)


def main():
    try:
        import IPython  # noqa
    except ImportError:
        main_python()
    else:
        main_ipython()
