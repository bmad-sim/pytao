from __future__ import annotations

import contextlib
import contextvars
import logging
import textwrap
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Literal, Optional, Set, Tuple

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


CaptureByLevel = Dict[TaoMessageLevel, FrozenSet[str]]


def split_error_messages(
    messages: List[TaoMessage],
) -> Tuple[List[TaoMessage], List[TaoMessage]]:
    regular, errors = [], []
    for msg in messages:
        if msg.level in error_message_levels:
            errors.append(msg)
        else:
            regular.append(msg)

    return regular, errors


def raise_for_error_messages(cmd: str, lines: List[str], errors: List[TaoMessage]):
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
        f"Command: {cmd!r} causes errors in the function(s): {functions}\n\n{error_lines}",
        tao_output="\n".join(lines),
    )


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

    def filter_messages(self, cmd: str, all_messages: list[TaoMessage]):
        """
        Remove context-filtered messages from the list.

        Parameters
        ----------
        cmd : str
            The Tao command used to get the output.
        all_messages : list[TaoMessage]

        Returns
        -------
        list of TaoMessage
            Filtered tao messages.
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

        return [message for message in all_messages if should_include(message)]

    def check_output(self, cmd: str, lines: List[str]):
        lines, all_messages = capture_messages_from_functions(lines)
        messages = self.filter_messages(cmd, all_messages)

        _regular, errors = split_error_messages(messages)
        if errors:
            raise_for_error_messages(cmd, lines, errors)


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


error_filter_context: contextvars.ContextVar[Optional[TaoErrorFilterContext]] = (
    contextvars.ContextVar("error_filter_context", default=None)
)


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


@dataclass
class TaoMessage:
    """A Tao message from `out_io`."""

    level: TaoMessageLevel
    function: str
    message: str

    @property
    def level_number(self) -> int:
        return all_message_levels.index(self.level)

    def __str__(self) -> str:
        return f"[{self.level} {self.function}] {self.message}"

    @property
    def log_level(self) -> int:
        if self.level in error_message_levels:
            return logging.ERROR
        return logging.DEBUG


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
