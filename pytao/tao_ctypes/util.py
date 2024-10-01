from __future__ import annotations

import contextlib
import contextvars
import importlib
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

import numpy as np
from typing_extensions import Literal

TaoMessageLevel = Literal[
    "INFO",  # Informational message
    "SUCCESS",  # Successful completion notice
    "WARNING",  # General warning
    "ERROR",  # General error
    "FATAL",  # Fatal error - cannot continue computations, reset will be attempted
    "ABORT",  # Severe error which will lead to an abort
    "MESSAGE",  # An important message
]

all_message_levels = ("INFO", "SUCCESS", "WARNING", "ERROR", "FATAL", "ABORT", "MESSAGE")
error_message_levels = ("ERROR", "FATAL", "ABORT")


error_filter_context: contextvars.ContextVar[Optional[TaoErrorFilterContext]] = (
    contextvars.ContextVar("error_filter_context", default=None)
)


class TaoException(Exception):
    pass


class TaoExceptionWithOutput(TaoException):
    """
    A Tao Exception that includes command-line output.

    This may include one or more messages from specific Tao functions.

    Attributes
    ----------
    tao_output : str
        The raw output from Tao.
    """

    tao_output: str

    def __init__(self, message: str, tao_output: str = ""):
        super().__init__(message)
        self.tao_output = tao_output

    @property
    def errors(self) -> List[TaoMessage]:
        """All Tao messages marked error, fatal, or abort."""
        return self.get_errors()

    @property
    def messages(self) -> List[TaoMessage]:
        """All Tao messages found from any level."""
        return self.get_messages(all_message_levels)

    def get_errors(
        self,
        exclude_functions: Iterable[str] = (),
    ) -> List[TaoMessage]:
        """
        Get all Tao messages marked error, fatal, or abort.

        Parameters
        ----------
        exclude_functions : Iterable[str]
            Tao function names to ignore in the listing.

        Returns
        -------
        List[TaoMessage]
        """
        return self.get_messages(
            levels=error_message_levels,
            exclude_functions=exclude_functions,
        )

    def get_messages(
        self,
        levels: Iterable[TaoMessageLevel] = (),
        exclude_functions: Iterable[str] = (),
    ) -> List[TaoMessage]:
        """
        Get all Tao messages marked with the specified levels.

        Parameters
        ----------
        levels : iterable of TaoMessageLevel, optional
            If unspecified, defaults to `all_message_levels`:
            ``("INFO", "SUCCESS", "WARNING", "ERROR", "FATAL", "ABORT", "MESSAGE")``
        exclude_functions : Iterable[str]
            Tao function names to ignore in the listing.

        Returns
        -------
        List[TaoMessage]
        """
        if not levels:
            levels = all_message_levels
        _, messages = capture_messages_from_functions(
            self.tao_output.splitlines(), levels=levels
        )

        return [message for message in messages if message.function not in exclude_functions]


class TaoInitializationError(TaoExceptionWithOutput, RuntimeError):
    """
    A Tao error that happened during initialization.

    See `.tao_output` for the raw output text from Tao. This may include one or
    more messages from specific Tao functions, which are accessible through
    `.messages` or `.errors`.

    Attributes
    ----------
    tao_output : str
        The raw output from Tao.
    """

    tao_output: str


class TaoSharedLibraryNotFoundError(TaoException, RuntimeError):
    """The Tao shared library (i.e., libtao.so) was not found."""

    pass


class TaoCommandError(TaoExceptionWithOutput, RuntimeError):
    """
    A Tao error that happened during the course of running a command.

    See `.tao_output` for the raw output text from Tao. This may include one or
    more messages from specific Tao functions, which are accessible through
    `.messages` or `.errors`.

    Attributes
    ----------
    tao_output : str
        The raw output from Tao.
    """

    tao_output: str


CaptureByLevel = Dict[TaoMessageLevel, FrozenSet[str]]


@dataclass
class TaoErrorFilterContext:
    """
    The state of pytao's error capture context.

    Messages that match these filter settings will **not** be considered
    errors when processing Tao's text output.

    Attributes
    ----------
    functions : FrozenSet[str]
        Tao Fortran function names to exclude.
    by_level : Dict[TaoMessageLevel, FrozenSet[str]]
        Message-level specific Tao Fortran function names to exclude.
    by_command : Dict[str, FrozenSet[str]]
        Based on the Tao command used, exclude messages from these Tao Fortran
        functions.
    """

    functions: FrozenSet[str] = field(default_factory=frozenset)
    by_level: CaptureByLevel = field(default_factory=dict)
    by_command: Dict[str, FrozenSet[str]] = field(default_factory=dict)

    @classmethod
    def from_user(
        cls,
        functions: Optional[Iterable[str]] = None,
        by_level: Optional[Dict[TaoMessageLevel, Iterable[str]]] = None,
        by_command: Optional[Dict[str, Iterable[str]]] = None,
    ) -> TaoErrorFilterContext:
        def fix_by_level(items: Dict[TaoMessageLevel, Iterable[str]]) -> CaptureByLevel:
            return {level: frozenset(functions) for level, functions in items.items()}

        return cls(
            functions=frozenset(functions or set()),
            by_level=fix_by_level(by_level or {}),
            by_command={key: frozenset(value) for key, value in (by_command or {}).items()},
        )

    def check_output(self, cmd: str, lines: List[str]):
        """
        Check Tao output using the filter context.

        Parameters
        ----------
        cmd : str
            The Tao command used to get the output.
        lines : List[str]
            The Tao output lines.

        Returns
        -------
        list of TaoMessage
            Messages found in the output, excluding those filtered out.
        """
        cmd = cmd.strip()
        if not cmd:
            by_command = {}
        else:
            by_command = self.by_command.get(cmd.split()[0].lower(), frozenset())

        def should_include(message: TaoMessage) -> bool:
            return (
                message.function not in self.functions
                and message.function not in self.by_level.get(message.level, ())
                and message.function not in by_command
            )

        messages = [
            message
            for message in capture_messages_from_functions(lines)[1]
            if should_include(message)
        ]

        errors = [message for message in messages if message.level in error_message_levels]
        if errors:
            functions = ", ".join(sorted(set(error.function for error in errors)))
            error_lines = "\n\n".join(
                "\n".join(
                    (
                        f"{error.level.capitalize()} in {error.function}:",
                        textwrap.indent(error.message, "  "),
                    )
                )
                for error in errors
            )
            raise TaoCommandError(
                f"Command: {cmd} causes errors in the function(s): {functions}\n\n{error_lines}",
                tao_output="\n".join(lines),
            )
        return messages


@dataclass
class TaoMessage:
    """A Tao message from `out_io`."""

    level: TaoMessageLevel
    function: str
    message: str

    @property
    def level_number(self) -> int:
        return all_message_levels.index(self.level)


def filter_tao_messages(
    *,
    functions: Optional[Iterable[str]] = None,
    by_level: Optional[Dict[TaoMessageLevel, Iterable[str]]] = None,
    by_command: Optional[Dict[str, Iterable[str]]] = None,
) -> TaoErrorFilterContext:
    """
    Filter out Tao messages originating from specific Fortran functions.

    Messages that match these filter settings will **not** be considered
    errors when processing Tao's text output.

    Consider using `filter_tao_messages_context` for limiting the scope of the
    filter.

    Parameters
    ----------
    functions : Iterable[str], optional
        Tao Fortran function names to exclude.
    by_level : Dict[TaoMessageLevel, Iterable[str]], optional
        Message-level specific Tao Fortran function names to exclude.
        For example, to exclude errors from ``some_func`` when they are at
        the ``"ERROR"`` severity, this could be:
        ``by_level={"ERROR": ["some_func"]}``.
    by_command : Dict[str, Iterable[str]], optional
        Based on the Tao command used, exclude messages from these Tao Fortran
        functions.
        For example, to exclude errors from ``some_func`` that appear when
        Tao's ``show`` command is run, this could be
        ``by_command={"show": ["some_func"]}``.
    """

    ctx = TaoErrorFilterContext.from_user(
        functions=functions,
        by_level=by_level,
        by_command=by_command,
    )
    error_filter_context.set(ctx)
    return ctx


@contextlib.contextmanager
def filter_tao_messages_context(
    *,
    functions: Optional[Iterable[str]] = None,
    by_level: Optional[Dict[TaoMessageLevel, Iterable[str]]] = None,
    by_command: Optional[Dict[str, Iterable[str]]] = None,
):
    """
    Filter out Tao messages originating from specific Fortran functions.

    Messages that match these filter settings will **not** be considered
    errors when processing Tao's text output.

    Parameters
    ----------
    functions : Iterable[str], optional
        Tao Fortran function names to exclude.
    by_level : Dict[TaoMessageLevel, Iterable[str]], optional
        Message-level specific Tao Fortran function names to exclude.
        For example, to exclude errors from ``some_func`` when they are at
        the ``"ERROR"`` severity, this could be:
        ``by_level={"ERROR": ["some_func"]}``.
    by_command : Dict[str, Iterable[str]], optional
        Based on the Tao command used, exclude messages from these Tao Fortran
        functions.
        For example, to exclude errors from ``some_func`` that appear when
        Tao's ``show`` command is run, this could be
        ``by_command={"show": ["some_func"]}``.
    """
    ctx = TaoErrorFilterContext.from_user(
        functions=functions,
        by_level=by_level,
        by_command=by_command,
    )
    prev = error_filter_context.get()
    try:
        error_filter_context.set(ctx)
        yield ctx
    finally:
        error_filter_context.set(prev)


def capture_messages_from_functions(
    lines: List[str],
    levels: Iterable[TaoMessageLevel] = all_message_levels,
) -> Tuple[List[str], List[TaoMessage]]:
    """
    Capture Tao output text lines.

    Parameters
    ----------
    lines : List[str]
        Lines from Tao.
    levels : one or more of {"INFO", "SUCCESS", "WARNING", "ERROR", "FATAL", "ABORT"}
        Message levels to capture.

    Returns
    -------
    filtered_lines : str
    """
    out_lines = []
    message = None
    messages: List[TaoMessage] = []
    for line in lines:
        if message is not None:
            if not line.strip() or line[0].isspace():
                # Lines starting with space are part of the message
                message.message = "\n".join((message.message, line.lstrip())).strip()
                continue
            message = None

        for level in levels:
            # Two possible formats:
            # * [LEVEL] function:
            # * [LEVEL | date-time] function:
            if line.startswith(f"[{level}") and line.endswith(":"):
                function = line.split()[-1].rstrip(":")
                message = TaoMessage(level=level, function=function, message="")
                messages.append(message)
                break
        else:
            out_lines.append(line)

    return out_lines, messages


def filter_output_lines(lines: List[str], exclude: Set[str]) -> List[str]:
    """
    Filter Tao output text lines.

    Parameters
    ----------
    lines : List[str]
        Lines from Tao.
    exclude : Set[str]
        Function names to exclude.

    Returns
    -------
    List[str]
        Lines filtered, excluding those pertaining to the requested functions.
    """
    removing_block = False
    out_lines = []
    for line in lines:
        if removing_block:
            if not line.strip():
                # Empty line -> skip
                continue
            if line[0].isspace():
                # Lines starting with spaces in a skipped block are ignored
                continue
            removing_block = False

        if not line.startswith("[ERROR"):
            out_lines.append(line)
            continue

        for func in exclude:
            if line.endswith(f"{func}:"):
                removing_block = True
                break
        else:
            out_lines.append(line)

    return out_lines


def error_in_lines(lines):
    """
    Checks '[ERROR', '[CRITICAL', '[FATAL' found in
    lines, and returns a string of info if something is found.
    Otherwise, '' is returned.

    """
    for i, line in enumerate(lines):
        err = error_in_line(line)
        if err:
            info = "\n".join(lines[i:])
            return f"{err} detected: {info}"

    return ""


def error_in_line(line):
    """
    Returns True if the line contains: '[ERROR', '[CRITICAL', '[FATAL'
    """
    for chars in ["[ERROR", "[CRITICAL", "[FATAL"]:
        if chars in line:
            return chars[1:]
    return ""


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def parse_bool(s):
    x = s.upper()[0]
    if x == "T":
        return True
    elif x == "F":
        return False
    else:
        raise ValueError("Unknown bool: " + s)


def parse_tao_lat_ele_list(lines):
    """
    returns mapping of names to index

    TODO: if elements are duplicated, this returns only the last one.

    Example:
    ixlist = parse_tao_lat_ele_list(tao.cmd('python lat_ele_list 1@0'))
    """
    ix = {}
    for line in lines:
        index, name = line.split(";")
        ix[name] = int(index)
    return ix


def parse_pytype(type, val):
    """
    Parses the various types from tao_python_cmd


    """

    # Handle
    if isinstance(val, list):
        if len(val) == 1:
            val = val[0]

    if type in [
        "STR",
        "ENUM",
        "FILE",
        "CRYSTAL",
        "COMPONENT",
        "DAT_TYPE",
        "DAT_TYPE_Z",
        "SPECIES",
        "ELE_PARAM",
    ]:
        return val

    if type == "LOGIC":
        return parse_bool(val)

    if type in ["INT", "INUM"]:
        return int(val)

    if type == "REAL":
        return float(val)

    if type == "INT_ARR":
        return np.array(val).astype(int)

    if type == "REAL_ARR":
        return np.array(val).astype(float)

    if type == "COMPLEX":
        return complex(*val)

    if type == "STRUCT":
        return {name: parse_pytype(t1, v1) for name, t1, v1 in chunks(val, 3)}

    # Not found
    raise ValueError("Unknown type: " + type)


def parse_tao_python_data1(line, clean_key=True):
    """
    Parses most common data output from a Tao>python command
    <component_name>;<type>;<is_variable>;<component_value>

    and returns a dict
    Example:
        eta_x;REAL;F;  9.0969865321048662E+00
    parses to:
        {'eta_x':9.0969865321048662E+00}

    If clean key, the key will be cleaned up by replacing '.' with '_' for use as class attributes.

    See: tao_python_cmd.f90
    """
    dat = {}

    sline = line.split(";")
    name, type, setable = sline[0:3]
    component_value = sline[3:]

    # Parse
    dat = parse_pytype(type, component_value)

    if clean_key:
        name = name.replace(".", "_")

    return {name: dat}


def parse_tao_python_data(lines, clean_key=True):
    """
    returns dict with data
    """
    dat = {}
    for line in lines:
        dat.update(parse_tao_python_data1(line, clean_key))

    return dat


def simple_lat_table(tao, ix_universe=1, ix_branch=0, which="model", who="twiss"):
    """
    Takes the tao object, and returns columns of parameters associated with lattice elements
     "which" is one of:
       model
       base
       design
     and "who" is one of:
       general         ! ele%xxx compnents where xxx is "simple" component (not a structure nor an array, nor allocatable, nor pointer).
       parameters      ! parameters in ele%value array
       multipole       ! nonzero multipole components.
       floor           ! floor coordinates.
       twiss           ! twiss parameters at exit end.
       orbit           ! orbit at exit end.
     Example:


    """
    # Form list of ele names
    cmd = "python lat_ele_list " + str(ix_universe) + "@" + str(ix_branch)
    lines = tao.cmd(cmd)
    # initialize
    ele_table = {}
    for x in lines:
        ix, name = x.split(";")
        # Single element information
        cmd = (
            "python lat_ele1 "
            + str(ix_universe)
            + "@"
            + str(ix_branch)
            + ">>"
            + str(ix)
            + "|"
            + which
            + " "
            + who
        )
        lines2 = tao.cmd(cmd)
        # Parse, setting types correctly
        ele = parse_tao_python_data(lines2)
        # Add name and index
        ele["name"] = name
        ele["ix_ele"] = int(ix)

        # Add data to columns
        for key in ele:
            if key not in ele_table:
                ele_table[key] = [ele[key]]
            else:
                ele_table[key].append(ele[key])

        # Stop at the end ele
        if name == "END":
            break
    return ele_table


def import_by_name(clsname: str):
    """
    Import the given object by name.

    Parameters
    ----------
    clsname : str
        The module path to find the class e.g.
        ``"pytao.Tao"``
    """
    module, cls = clsname.rsplit(".", 1)
    if module not in sys.modules:
        importlib.import_module(module)

    mod = sys.modules[module]
    try:
        return getattr(mod, cls)
    except AttributeError:
        raise ImportError(f"Unable to import {clsname!r} from module {module!r}")
