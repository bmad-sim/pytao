from __future__ import annotations

import ctypes
import logging
import os
import pathlib
import textwrap
from ctypes.util import find_library
from typing import TYPE_CHECKING, Literal, Union

import numpy as np

from .errors import (
    TaoCommandError,
    TaoInitializationError,
    TaoMessage,
    TaoSharedLibraryNotFoundError,
    capture_messages_from_functions,
    error_filter_context,
    error_message_levels,
    raise_for_error_messages,
)
from .util import parsers as _pytao_parsers
from .util.parameters import tao_parameter_dict

if TYPE_CHECKING:
    from .subproc import SubprocessTao
    from .tao import Tao

    AnyTao = Union[Tao, SubprocessTao]

logger = logging.getLogger(__name__)
AnyPath = Union[pathlib.Path, str]
Quiet = Literal["all", "warnings"]


def ipython_shell(tao: AnyTao) -> None:
    """Spawn an interactive Tao shell using IPython."""
    from IPython.terminal.embed import InteractiveShellEmbed
    from IPython.terminal.prompts import Prompts, Token
    from traitlets.config.loader import Config

    class TaoPrompt(Prompts):
        def in_prompt_tokens(self, cli=None):
            return [(Token.Prompt, "Tao> ")]

    cfg = Config()
    cfg.TerminalInteractiveShell.prompts_class = TaoPrompt
    cfg.TerminalInteractiveShell.confirm_exit = False

    # Remove standard completion stuff from Jedi:
    cfg.IPCompleter.use_jedi = False

    cfg.IPCompleter.disable_matchers = [
        "IPCompleter.latex_name_matcher",
        # "IPCompleter.unicode_name_matcher",
        "back_latex_name_matcher",
        "back_unicode_name_matcher",
        # "IPCompleter.fwd_unicode_matcher",
        "IPCompleter.magic_config_matcher",
        "IPCompleter.magic_color_matcher",
        # "IPCompleter.custom_completer_matcher",
        "IPCompleter.dict_key_matcher",
        "IPCompleter.magic_matcher",
        "IPCompleter.python_matcher",
        "IPCompleter.file_matcher",  # <- TODO can we use this in a restricted way?
        "IPCompleter.python_func_kw_matcher",
    ]

    pytao_config = pathlib.Path("~/.config/.pytao").expanduser()
    pytao_config.mkdir(parents=True, exist_ok=True)

    cfg.HistoryManager.hist_file = pytao_config / "pytao_shell_history.db"

    # TODO: can we integrate with Tao's history?
    # try:
    #     with open(pathlib.Path("~/.history_tao")) as fp:
    #         tao_history = fp.readlines()
    # except IOError:
    #     tao_history = []

    ipshell = InteractiveShellEmbed(
        config=cfg,
        banner1="Entering Tao interactive shell. Type commands as in Tao.",
    )

    def tao_top_level_completer(ipython, event):
        """Tab completion for top-level Tao commands."""
        options = list(tao._autocomplete_usage_)

        parts = [part.lower() for part in event.line.lstrip().split()]

        if " ".join(parts[:2]) in {"change ele", "set ele", "show ele"}:
            return tao.lat_list(f"{event.symbol}*", "ele.name")

        cmd = parts[0] if parts else None

        if cmd not in tao._autocomplete_usage_:
            return [opt for opt in options if opt.startswith(event.symbol.lower())]

        level = len(parts)
        return [
            option.split(" ")[level]
            for option, _help in tao._autocomplete_usage_[cmd]
            if option.count(" ") > level and "{" not in option and "<" not in option
        ]

    ipshell.set_hook("complete_command", tao_top_level_completer, re_key="^")

    print("Type 'exit', 'quit', or press Ctrl-D on a blank line to return to IPython mode.")

    def preprocess_line(line: str) -> str:
        line = line.strip()
        if line.lower() in ["exit()", "quit()", "history"] or not line:
            return line
        if line.startswith("get_ipython"):
            return line

        res = tao.cmd(line, raises=False)
        if isinstance(res, str):
            res = [res]

        print("\n".join(res))
        return ""

    def preprocess_lines(lines: list[str]) -> list[str]:
        lines = [preprocess_line(line) for line in lines]
        return [preprocess_line(line) for line in lines]

    ipshell.input_transformers_post.append(preprocess_lines)

    ipshell()


def simple_shell(tao: Tao) -> None:
    print("Entering Tao interactive shell.")
    print("Type 'exit', 'quit', or press Ctrl-D on a blank line to return to Python mode.")

    while True:
        try:
            cmd = input("Tao> ")
            if cmd.lower() in ("exit", "quit"):
                break
            result = tao.cmd(cmd)
            res = tao.cmd(cmd, raises=False)
            if isinstance(res, str):
                res = [res]
            if res:
                print("\n".join(result))
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            print(f"Error: {e}")


def register_input_transformer(prefix: str = "`"):
    """
    Register IPython magic convenience transform.

    Parameters
    ----------
    prefix : str, optional
        The leading character(s) for the command prefix.
    """
    import IPython

    ip = IPython.get_ipython()

    if ip is None:
        return  # Not in an IPython environment

    # Define our transformer function
    def tao_transform(lines: list[str]) -> list[str]:
        """Transform lines that start with the prefix."""
        transformed_lines = []
        for line in lines:
            if line.strip() == prefix:
                transformed_lines.append("tao.shell()")
            elif line.startswith(prefix):
                cmd = line[len(prefix) :].strip()
                transformed_lines.append(
                    f"""
                    print("\\n".join(tao.cmd({cmd!r})))
                    """.strip()
                )
                logger.debug(f"Transformed: {cmd} -> {transformed_lines[-1]}")
            else:
                transformed_lines.append(line)
        return transformed_lines

    # Register the transformer in the IPython instance
    ip.input_transformers_post.append(tao_transform)


def _tao_line_cell_magic(tao_instance: Tao, line: str, cell: str | None = None):
    """
    Execute Tao commands in IPython as line or cell magic.

    This function is used to implement the %tao line magic and %%tao cell magic
    in IPython. It sends commands to a Tao instance and prints the results.

    Parameters
    ----------
    tao_instance : Tao
        The Tao instance registered with the line/cell magic.
    line : str
        The line content when used as line magic, or the line after %%tao when
        used as cell magic. In cell magic, this can optionally specify a
        different Tao instance to use.
    cell : str or None, default=None
        The cell content when used as cell magic.
        If None, function operates as line magic.

    Returns
    -------
    None
        Results are printed rather than returned.

    Raises
    ------
    ValueError
        If the specified tao_instance_name in cell magic is invalid or not a TaoCore instance.
    AssertionError
        If not running in an IPython environment.
    """
    from IPython import get_ipython

    ipy = get_ipython()
    assert ipy is not None

    if cell is None:
        cmds = [line.format(**ipy.user_ns)]
    else:
        if line.strip():
            try:
                (tao_instance_name,) = line.strip().split()
                tao_instance = ipy.user_ns[tao_instance_name]
                if not isinstance(tao_instance, TaoCore):
                    raise ValueError("Not a valid Tao instance")
            except Exception as ex:
                raise ValueError(
                    f"Usage: %%tao [tao_instance_name]\n"
                    f"If specified, tao_instance_name must be the variable name of a 'Tao' instance. "
                    f"({ex.__class__.__name__}: {ex})"
                )

        cell = cell.format(**ipy.user_ns)
        cmds = cell.split("\n")

    for cmd in cmds:
        if not cmd.strip():
            continue

        header = f"Tao> {cmd}"
        print("-" * len(header))
        print(header)
        res = tao_instance.cmd(cmd)
        for line in res:
            print(line)


class TaoCore:
    """
    Class to run and interact with Tao. Requires libtao shared object.

    Setup:

    import os
    import sys
    TAO_PYTHON_DIR=os.environ['ACC_ROOT_DIR'] + '/tao/python'
    sys.path.insert(0, TAO_PYTHON_DIR)

    import tao_ctypes
    tao = tao_ctypes.Tao("command line args here...")
    """

    _init_output: list[str]
    _last_output: list[str]
    so_lib: ctypes.CDLL
    so_lib_file: str
    _ctypes_initialized_: bool = False

    def _init_shared_library(self, so_lib: str) -> None:
        self.so_lib, self.so_lib_file = init_libtao(user_path=so_lib)

    @property
    def init_output(self) -> list[str]:
        """Output from the latest Tao initialization."""
        return list(self._init_output)

    @property
    def last_output(self) -> list[str]:
        return list(self._last_output)

    def get_output(self, reset=True) -> list[str]:
        """
        Returns a list of output strings.

        Parameters
        ----------
        reset : bool, default=True
            Reset the internal Tao buffers after getting the output.

        Returns
        -------
        list of str
            Tao output text lines.
        """
        n_lines = self.so_lib.tao_c_out_io_buffer_num_lines()
        lines = [
            self.so_lib.tao_c_out_io_buffer_get_line(i).decode("utf-8")
            for i in range(1, n_lines + 1)
        ]

        self._last_output = lines
        if reset:
            self.so_lib.tao_c_out_io_buffer_reset()
        return lines

    def reset_output(self):
        """
        Resets all output buffers
        """
        self.so_lib.tao_c_out_io_buffer_reset()

    def _init_or_reinit(self, cmd: str) -> tuple[int, list[str]]:
        """
        Initialize (or reinitialize) Tao with `cmd`.

        Parameters
        ----------
        cmd : str
            The command to (re)initialize Tao with.

        Returns
        -------
        errno : int
            Tao's reported error number on first initialization.
        list of str
            Tao initialization output text.
        """
        if not TaoCore._ctypes_initialized_:
            logger.debug("Initializing Tao.")
            logger.debug(f"Tao> {cmd}")
            errno = self.so_lib.tao_c_init_tao(cmd.encode("utf-8"))
            if errno == 0:
                # Only mark it initialized on the first actual success.
                TaoCore._ctypes_initialized_ = True
        else:
            logger.debug("Re-initializing Tao.")

            reinit_cmd = f"reinit tao -clear {cmd}"
            logger.debug(f"Tao> {reinit_cmd}")

            errno = self.so_lib.tao_c_command(reinit_cmd.encode("utf-8"))

        output = self.get_output()

        self._init_output = output
        return errno, output

    def _init_or_raise(self, cmd: str) -> list[str]:
        """
        Initialize (or reinitialize) Tao with `cmd`.

        Parameters
        ----------
        cmd : str
            The command to (re)initialize Tao with.

        Returns
        -------
        list of str
            Tao initialization output text.

        Raises
        ------
        TaoInitializationError
        """
        errno, output = self._init_or_reinit(cmd)
        if errno != 0:
            message = textwrap.indent("\n".join(output), "  ")
            raise TaoInitializationError(
                (
                    f"Tao initialization reported an unrecoverable error (code={errno}):\n"
                    f"Tao> {cmd}\n"
                    f"\n\n{message}"
                ),
                tao_output="\n".join(output),
            )
        return output

    def _check_output_lines(
        self, cmd: str, lines: list[str], raises: bool = False
    ) -> tuple[list[str], list[TaoMessage]]:
        """
        Check Tao output for errors respecting the current capture context.

        Parameters
        ----------
        cmd : str
            The Tao command which returned output `lines`.
        lines : List[str]
            Tao output lines for `cmd`.
        """
        ctx = error_filter_context.get()
        lines, all_messages = capture_messages_from_functions(lines)
        if ctx is not None:
            all_messages = ctx.filter_messages(cmd, all_messages)

        errors = [msg for msg in all_messages if msg.level in error_message_levels]
        if errors and raises:
            raise_for_error_messages(cmd, lines, errors)

        return lines, all_messages

    def _execute(
        self,
        cmd: str,
        as_dict: bool = True,
        raises: bool = True,
        method_name=None,
        cmd_type: Literal["string_list", "real_array", "integer_array"] = "string_list",
    ):
        """

        A wrapper to handle commonly used options when running a command through tao.

        Parameters
        ----------
        cmd : str
            The command to run
        as_dict : bool, optional
            Return string data as a dict? by default True
        raises : bool, optional
            Raise exception on tao errors? by default True
        method_name : str/None, optional
            Name of the caller. Required for custom parsers for commands, by
            default None
        cmd_type : str, optional
            The type of data returned by tao in its common memory, by default
            "string_list"

        Returns
        -------
        Any
        Result from running tao. The type of data depends on configuration, but is generally a list of strings, a dict, or a
        numpy array.
        """

        if cmd_type == "real_array":
            raw_output = self.cmd_real(cmd, raises=raises)
        elif cmd_type == "integer_array":
            raw_output = self.cmd_integer(cmd, raises=raises)
        else:
            cmd_output = self.cmd(cmd, raises=False)
            raw_output, messages = self._check_output_lines(cmd, cmd_output, raises=raises)

            for msg in messages:
                self._log(cmd, msg)

        special_parser = getattr(_pytao_parsers, f"parse_{method_name}", None)
        try:
            if special_parser and callable(special_parser):
                return special_parser(raw_output, cmd=cmd)
            if isinstance(raw_output, np.ndarray):
                return raw_output
            if as_dict:
                return _pytao_parsers.parse_tao_python_data(raw_output)
            return tao_parameter_dict(raw_output)
        except Exception as ex:
            if raises:
                setattr(ex, "tao_output", raw_output)
                raise
            logger.exception(
                "Failed to parse string data with custom parser. Returning raw value."
            )
            return raw_output

    def cmd(self, cmd, raises=True) -> list[str]:
        """
        Runs a command, and returns the text output.

        Parameters
        ----------
        cmd : str
            Command string
        remove_messages : bool, default=False
            Filter out Tao error and status messages.
        raises : bool, default=True
            Raise an exception of [ERROR or [FATAL is detected in the output

        Returns
        -------
        list of str
            Output lines.
        """

        logger.debug(f"Tao> {cmd}")

        self.so_lib.tao_c_command(cmd.encode("utf-8"))

        try:
            lines = self.get_output(reset=False)
            if not raises:
                return lines
            self._check_output_lines(cmd, lines, raises=True)
            return lines
        finally:
            self.reset_output()

    def _get_array(
        self,
        cmd: str,
        dtype: type[float] | type[int],
        raises: bool,
    ) -> np.ndarray | None:
        """
        Get an array directly from Tao (without string parsing).

        Does not reset the output buffer.  The caller is expected to handle
        this.

        Parameters
        ----------
        cmd : str
            The command used to retrieve this data.
        dtype : type
            The data type - float or int.
        raises : bool
            Raise TaoCommandError if an error is detected in the text output.

        Returns
        -------
        np.ndarray or None
            The array.  None is only returned if `raises=False` and an error
            is detected in the output.
        """
        if dtype is float:
            ctypes_type = ctypes.c_double
            num_elements = self.so_lib.tao_c_real_array_size()
            get_array = self.so_lib.tao_c_get_real_array
        elif dtype is int:
            ctypes_type = ctypes.c_int32
            num_elements = self.so_lib.tao_c_integer_array_size()
            get_array = self.so_lib.tao_c_get_integer_array
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        if num_elements == 0:
            return np.array([], dtype=dtype)

        try:
            self._check_output_lines(cmd, self.get_output(reset=False), raises=True)
        except TaoCommandError:
            if raises:
                raise
            return None

        # This is a pointer to the scratch space of (num_elements * dtype)
        get_array.restype = ctypes.POINTER(ctypes_type * num_elements)
        ptr = ctypes.addressof(get_array().contents)

        # Extract array data
        array = np.ctypeslib.as_array((ctypes_type * num_elements).from_address(ptr))
        return array.copy()

    def cmd_real(self, cmd: str, raises: bool = True) -> np.ndarray | None:
        """
        Get real array output.

        Only python commands that load the real array buffer can be used with this method.

        Parameters
        ----------
        cmd : str
        raises : bool, default=True
        """

        logger.debug(f"Tao> {cmd}")

        self.so_lib.tao_c_command(cmd.encode("utf-8"))
        try:
            return self._get_array(cmd=cmd, dtype=float, raises=raises)
        finally:
            self.reset_output()

    def _maybe_raise_for_message(self, cmd: str, lines: list[str], messages: list[TaoMessage]):
        if any(msg.level in error_message_levels for msg in messages):
            error_lines = "\n".join(
                str(msg) for msg in messages if msg.level in error_message_levels
            )
            raise TaoCommandError(
                "\n".join(
                    (
                        f"Command: {cmd!r} caused these error(s). Suppress this with 'raises=False':",
                        f"\n{error_lines}",
                    )
                ),
                tao_output="\n".join(lines),
            )

    def _log(self, cmd: str, message: TaoMessage) -> None:
        logger.log(message.log_level, str(message))

    def cmd_integer(self, cmd: str, raises: bool = True) -> np.ndarray | None:
        """
        Get integer array output.

        Only python commands that load the real array buffer can be used with this method.

        Parameters
        ----------
        cmd : str
        raises : bool, default=True
        """
        logger.debug(f"Tao> {cmd}")

        self.so_lib.tao_c_command(cmd.encode("utf-8"))
        try:
            return self._get_array(cmd=cmd, dtype=int, raises=raises)
        finally:
            self.reset_output()

    def register_input_transformer(self, prefix: str) -> None:
        """
        Registers an IPython input text transformer. Every IPython line
        that starts with `prefix` character(s) will turn into a `tao.cmd()` line.

        Examples
        --------
        >>> %tao sho lat

        >>> %%tao
        ... sho lat
        """

        register_input_transformer()

    def register_cell_magic(self):
        """
        Registers a cell magic in Jupyter notebooks.

        Examples
        --------
        >>> %tao sho lat

        >>> %%tao
        ... sho lat
        """

        from IPython.core.magic import register_line_cell_magic

        @register_line_cell_magic
        def tao(line, cell=None):
            _tao_line_cell_magic(
                tao_instance=self,  # type: ignore
                line=line,
                cell=cell,
            )

        del tao

    def shell(self) -> None:
        """
        Start an interactive shell with a 'Tao>' prompt.

        Uses IPython if available, otherwise falls back to standard Python input loop.
        """
        try:
            ipython_shell(
                tao=self,  # type: ignore
            )
        except ImportError:
            simple_shell(
                tao=self,  # type: ignore
            )


def find_libtao(base_dir):
    """
    Searches base_for for an appropriate libtao shared library.
    """
    for lib in ["libtao.so", "libtao.dylib", "libtao.dll"]:
        so_lib_file = os.path.join(base_dir, lib)
        if os.path.exists(so_lib_file):
            return so_lib_file
    return None


def auto_discovery_libtao():
    """
    Use system loader to try and find libtao.
    """
    # Find tao library regardless of suffix (ie .so, .dylib, etc) and load
    so_lib_file = find_library("tao")
    so_lib = ctypes.CDLL(so_lib_file) if so_lib_file is not None else None
    return so_lib, so_lib_file


def _configure_cdll(so_lib: ctypes.CDLL) -> None:
    """Configure return types for specific exported functions."""
    so_lib.tao_c_out_io_buffer_get_line.restype = ctypes.c_char_p
    so_lib.tao_c_out_io_buffer_reset.restype = None


def init_libtao(user_path: str = "") -> tuple[ctypes.CDLL, str]:
    """
    Find the libtao shared library and initialize it.

    Parameters
    ----------
    user_path : str, optional
        User-specified path to the library.

    Returns
    -------
    ctypes.CDLL
        The shared library instance.
    str
        Path to the shared library.
    """
    so_lib_file = None
    user_path = user_path or os.getenv("PYTAO_LIB_PATH", "")
    if user_path:
        if os.path.isdir(user_path):
            so_lib_file = find_libtao(user_path)
        else:
            so_lib_file = user_path
    else:
        ACC_ROOT_DIR = os.getenv("ACC_ROOT_DIR", "")
        if ACC_ROOT_DIR:
            BASE_DIR = os.path.join(ACC_ROOT_DIR, "production", "lib")
            so_lib_file = find_libtao(BASE_DIR)

    # Library was found from ACC_ROOT_DIR environment variable
    if so_lib_file:
        so_lib = ctypes.CDLL(so_lib_file)
        _configure_cdll(so_lib)
        return so_lib, so_lib_file

    # Try loading from system path
    # Find shared library from path and load it (finds regardless of extensions like .so or .dylib)
    so_lib, so_lib_file = auto_discovery_libtao()

    if so_lib_file is None or so_lib is None:
        raise TaoSharedLibraryNotFoundError("Shared object libtao library not found.")

    _configure_cdll(so_lib)
    return so_lib, so_lib_file
