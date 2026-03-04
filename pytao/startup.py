from __future__ import annotations

import contextlib
import logging
import pathlib
import shlex
from dataclasses import InitVar, asdict
from typing import TYPE_CHECKING, Any, Literal, Union

import pydantic
from pydantic import ConfigDict, TypeAdapter, dataclasses

if TYPE_CHECKING:
    from .subproc import SubprocessTao
    from .tao import Tao

    AnyTao = Union[Tao, SubprocessTao]

import argparse
import sys
from collections.abc import Sequence

from typing_extensions import override

logger = logging.getLogger(__name__)

AnyPath = Union[pathlib.Path, str]
Quiet = Literal["all", "warnings"]


class TaoArgumentParser(argparse.ArgumentParser):
    """
    Custom ArgumentParser that supports abbreviation/prefixing
    for single-dash '-' arguments.
    """

    @override
    def parse_known_args(  # type: ignore
        self,
        args: list[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> tuple[argparse.Namespace, list[str]]:
        """
        Parses the arguments after expanding any unambiguous prefixes.

        Parameters
        ----------
        args : list[str] | None, optional
            List of command line argument strings. Defaults to ``sys.argv[1:]``.
        namespace : argparse.Namespace | None, optional
            A namespace object to populate. Defaults to None.

        Returns
        -------
        tuple[argparse.Namespace, list[str]]
            A tuple of the populated namespace and the remaining unknown arguments.
        """
        if args is None:
            args = sys.argv[1:]

        valid_opts: list[str] = [
            opt for action in self._actions for opt in action.option_strings
        ]

        expanded_args: list[str] = []

        for arg in args:
            if arg.startswith("-") and arg not in ("-", "--"):
                # Extract potential inline value like -la=file.lat
                opt_str, eq, val_str = arg.partition("=")

                # Look for prefix matches
                matches = [opt for opt in valid_opts if opt.startswith(opt_str)]

                if opt_str in matches:
                    expanded_args.append(arg)
                elif len(matches) == 1:
                    # Unambiguous match
                    new_arg = matches[0] + (eq + val_str if eq else "")
                    expanded_args.append(new_arg)
                else:
                    # Zero match or ambiguous -> error
                    expanded_args.append(arg)
            else:
                # Positional argument or bare hyphen
                expanded_args.append(arg)

        return super().parse_known_args(expanded_args, namespace)  # type: ignore


def create_tao_cli_parser(
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    """
    Creates an ArgumentParser configured for Tao command line options.

    The parser expects a single hyphen `-` for standard options and
    allows double hyphens `--` to negate boolean options.

    Returns
    -------
    argparse.ArgumentParser
    """
    if parser is None:
        parser = TaoArgumentParser(
            description="Tao command line parser.",
            add_help=False,
            formatter_class=argparse.RawTextHelpFormatter,
        )

    def add_bool_arg(name: str, help_text: str, default: bool = False):
        # e.g., name = "debug" generates "-debug" and "--debug"
        parser.add_argument(
            f"-{name}",
            action="store_true",
            default=default,
            dest=name,
            help=f"{help_text} --{name} to negate.",
        )
        parser.add_argument(
            f"--{name}",
            action="store_false",
            dest=name,
            help=argparse.SUPPRESS,  # hide from --help to reduce clutter
        )

    parser.add_argument(
        "-beam_file", type=str, help="File containing the tao_beam_init namelist."
    )
    parser.add_argument(
        "-beam_init_position_file",
        type=str,
        help="File containing initial particle positions.",
    )
    parser.add_argument(
        "-building_wall_file", type=str, help="Define the building tunnel wall."
    )
    parser.add_argument(
        "-command", type=str, help="Commands to run after startup file commands."
    )
    parser.add_argument(
        "-data_file", type=str, help="Define data for plotting and optimization."
    )
    parser.add_argument(
        "-geometry", type=str, help="Plot window geometry (pixels), e.g., 800x600."
    )
    parser.add_argument(
        "-hook_init_file",
        type=str,
        help="Init file for hook routines (Default = tao_hook.init).",
    )
    parser.add_argument("-init_file", type=str, help="Tao init file.")
    parser.add_argument("-lattice_file", type=str, help="Bmad lattice file.")
    parser.add_argument("-plot_file", type=str, help="Plotting initialization file.")
    parser.add_argument(
        "-prompt_color",
        type=str,
        default="blue",
        help="Set color of prompt string. Default is blue.",
    )

    # -quiet expects specific levels.
    parser.add_argument(
        "-quiet",
        type=str,
        nargs="?",
        const="all",
        default=None,
        choices=["all", "warnings"],
        help="Suppress terminal output. Levels: 'all' (default if given), 'warnings'.",
    )

    parser.add_argument(
        "-slice_lattice",
        type=str,
        help="Discards elements from lattice that are not in the list.",
    )
    parser.add_argument("-start_branch_at", type=str, help="Start lattice branch at element.")
    parser.add_argument(
        "-startup_file", type=str, help="Commands to run after parsing Tao init file."
    )
    parser.add_argument(
        "-var_file", type=str, help="Define variables for plotting and optimization."
    )

    add_bool_arg("debug", "Debug mode for Wizards.")
    add_bool_arg("disable_smooth_line_calc", "Disable the smooth line calc used in plotting.")
    add_bool_arg("external_plotting", "Tells Tao that plotting is done externally to Tao.")
    add_bool_arg("log_startup", "Write startup debugging info.")
    add_bool_arg("no_stopping", "For debugging: Prevents Tao from exiting on errors.")
    add_bool_arg("noinit", "Do not use Tao init file.")
    add_bool_arg("noplot", "Do not open a plotting window.")
    add_bool_arg("nostartup", "Do not open a startup command file.")
    add_bool_arg("no_rad_int", "Do not do any radiation integrals calculations.")
    add_bool_arg("reverse", "Reverse lattice element order?")
    add_bool_arg("symbol_import", "Import symbols defined in lattice files(s)?")

    add_bool_arg("rf_on", "Use '--rf_on' to turn off RF (default is now RF on).", default=True)

    parser.add_argument(
        "-help", action="help", help="Display this list of command line options."
    )

    return parser


def parse_tao_args(args: Sequence[str] | None = None) -> TaoStartup:
    """
    Parses Tao CLI args and returns a convenient dataclass representation.

    Parameters
    ----------
    args : Sequence[str] | None
        List of command line arguments. Falls back to sys.argv if None.

    Returns
    -------
    TaoArgs
        Parsed arguments mapped to the TaoArgs dataclass.
    """
    parser = create_tao_cli_parser()
    return parser.parse_args(args, namespace=TaoStartup())


def make_tao_init(init: str, *, quiet: bool | Quiet = False, **kwargs) -> str:
    """
    Make Tao init string based on optional flags/command-line arguments.

    Parameters
    ----------
    init : str
        The user-specified init string.
    **kwargs :
        Command-line switches without the leading `-`,
        mapped to their respective values.
        Only added as an argument if not empty and not False.
    """
    result = shlex.split(init)

    if quiet is True:
        quiet = "all"

    for name, value in kwargs.items():
        switch = f"-{name}"
        if switch in result:
            continue
        if not value:
            continue
        result.append(switch)
        if value not in {True, False}:
            result.append(str(value))
    return shlex.join(result)


@dataclasses.dataclass(config=ConfigDict(extra="forbid", validate_assignment=True))
class TaoStartup:
    """
    All Tao startup settings.

    Attributes
    ----------
    init : str, optional
        Initialization string for Tao.  Same as the tao command-line, including
        "-init" and such.  Shell variables in `init` strings will be expanded
        by Tao.  For example, an `init` string containing `$HOME` would be
        replaced by your home directory.
    so_lib : str, optional
        Path to the Tao shared library.  Auto-detected if not specified.
    plot : str, bool, optional
        Use pytao's plotting mechanism with matplotlib or bokeh, if available.
        If `True`, pytao will pick an appropriate plotting backend.
        If `False` or "tao", Tao plotting will be used. (Default)
        If "mpl", the pytao matplotlib plotting backend will be selected.
        If "bokeh", the pytao Bokeh plotting backend will be selected.
    metadata : dict[str, Any], optional
        User-specified metadata about this startup.  Not passed to Tao.
    env : dict[str, str], optional
        Environment variables to set when initializing a new subprocess Tao.
        Not used for the standard in-process `Tao` class.
    beam_file : str or pathlib.Path, default=None
        File containing the tao_beam_init namelist.
    beam_init_position_file : pathlib.Path or str, default=None
        File containing initial particle positions.
    building_wall_file : str or pathlib.Path, default=None
        Define the building tunnel wall
    command : str, optional
        Commands to run after startup file commands
    data_file : str or pathlib.Path, default=None
        Define data for plotting and optimization
    debug : bool, default=False
        Debug mode for Wizards
    disable_smooth_line_calc : bool, default=False
        Disable the smooth line calc used in plotting
    external_plotting : bool, default=False
        Tells Tao that plotting is done externally to Tao.
    geometry : "wxh" or (width, height) tuple, optional
        Plot window geometry (pixels)
    hook_init_file :  pathlib.Path or str, default=None
        Init file for hook routines (Default = tao_hook.init)
    init_file : str or pathlib.Path, default=None
        Tao init file
    lattice_file : str or pathlib.Path, default=None
        Bmad lattice file
    log_startup : bool, default=False
        Write startup debugging info
    no_stopping : bool, default=False
        For debugging : Prevents Tao from exiting on errors
    noinit : bool, default=False
        Do not use Tao init file.
    noplot : bool, default=False
        Do not open a plotting window
    nostartup : bool, default=False
        Do not open a startup command file
    no_rad_int : bool, default=False
        Do not do any radiation integrals calculations.
    plot_file : str or pathlib.Path, default=None
        Plotting initialization file
    prompt_color : str, optional
        Set color of prompt string. Default is blue.
    reverse : bool, default=False
        Reverse lattice element order?
    rf_on : bool, default=False
        Use "--rf_on" to turn off RF (default is now RF on)
    quiet : bool or "all" or "warnings", default=False
        Suppress terminal output when running a command file.
        For backward compatibility, True is equivalent to "all".
    slice_lattice : str, optional
        Discards elements from lattice that are not in the list
    start_branch_at : str, optional
        Start lattice branch at element.
    startup_file : str or pathlib.Path, default=None
        Commands to run after parsing Tao init file
    symbol_import : bool, default=False
        Import symbols defined in lattice files(s)?
    var_file : str or pathlib.Path, default=None
        Define variables for plotting and optimization
    """

    # General case 'init' string:
    init: InitVar[str] = ""

    # Tao ctypes-specific - shared library location.
    so_lib: str = pydantic.Field(default="", kw_only=False)

    # pytao specific
    plot: str | bool = "tao"
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)
    env: dict[str, str] | None = None  # only for subprocesses

    # All remaining flags:
    beam_file: AnyPath | None = None
    beam_init_position_file: AnyPath | None = None
    building_wall_file: AnyPath | None = None
    command: str = ""
    data_file: AnyPath | None = None
    debug: bool = False
    disable_smooth_line_calc: bool = False
    external_plotting: bool = False
    geometry: str | tuple[int, int] = ""
    hook_init_file: AnyPath | None = "tao_hook.init"
    init_file: AnyPath | None = None
    lattice_file: AnyPath | None = None
    log_startup: bool = False
    no_stopping: bool = False
    noinit: bool = False
    noplot: bool = False
    nostartup: bool = False
    no_rad_int: bool = False
    plot_file: AnyPath | None = None
    prompt_color: str = "blue"
    reverse: bool = False
    rf_on: bool = True
    quiet: bool | Quiet = False
    slice_lattice: str = ""
    start_branch_at: str = ""
    startup_file: AnyPath | None = None
    symbol_import: bool = False
    var_file: AnyPath | None = None

    def __post_init__(self, init: str):
        if not init:
            return

        cls = type(self)
        parsed = cls.from_cli_args(shlex.split(init))
        changes = TypeAdapter(cls).dump_python(parsed, exclude_defaults=True)
        for key, value in changes.items():
            setattr(self, key, value)

    @classmethod
    def from_cli_args(cls, args: list[str] | None = None, exit_on_error: bool = False):
        parser = create_tao_cli_parser()
        parser.exit_on_error = exit_on_error
        return parser.parse_args(args, namespace=cls())

    @property
    def tao_class_params(self) -> dict[str, Any]:
        """Parameters used to initialize Tao or make a new Tao instance."""
        # init_parts = self.init.split()
        params = {
            key: value
            for key, value in asdict(self).items()
            if value != getattr(type(self), key, None)
        }
        # params["init"] = self.init
        params.pop("metadata")
        params.pop("env", None)

        geometry = params.get("geometry", "")
        if not isinstance(geometry, str):
            width, height = geometry
            params["geometry"] = f"{width}x{height}"
        return params

    @property
    def can_initialize(self) -> bool:
        """
        Can Tao be initialized with these settings?

        Tao requires one or more of the following to be initialized:

        * `-init_file` to specify the initialization file.
        * `-lattice_file` to specify the lattice file.

        These are commonly shortened to `-init` or `-lat`.  Tao accepts
        shortened flags if they are not ambiguous.
        """
        tao_init_parts = self.tao_init.split()
        return any(part.startswith(flag) for part in tao_init_parts for flag in {"-i", "-la"})

    @property
    def tao_init(self) -> str:
        """Tao.init() command string."""
        params = self.tao_class_params
        # For tao.init(), we throw away Tao class-specific things:
        params.pop("so_lib", None)
        params.pop("plot", None)
        return make_tao_init("", **params)

    def run(self, use_subprocess: bool = False) -> AnyTao:
        """Create a new Tao instance and run it using these settings."""
        params = self.tao_class_params
        if use_subprocess:
            from .subproc import SubprocessTao

            return SubprocessTao(**params, env=self.env)

        from .tao import Tao

        return Tao(**params)

    @contextlib.contextmanager
    def run_context(self, use_subprocess: bool = False):
        """
        Create a new Tao instance and run it using these settings in a context manager.

        Yields
        ------
        Tao
            Tao instance.
        """
        tao = self.run(use_subprocess=use_subprocess)

        try:
            yield tao
        finally:
            from .subproc import SubprocessTao

            if isinstance(tao, SubprocessTao):
                tao.close_subprocess()
