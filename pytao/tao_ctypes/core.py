from __future__ import annotations

import ctypes
import logging
import os
import pathlib
import shutil
import tempfile
import textwrap
from ctypes.util import find_library
from typing import TYPE_CHECKING, List, Optional, Tuple, Type, Union

import numpy as np

from .. import tao_ctypes
from ..util.parameters import tao_parameter_dict
from . import util
from .tools import full_path
from .util import (
    TaoCommandError,
    TaoInitializationError,
    TaoMessage,
    TaoSharedLibraryNotFoundError,
)

if TYPE_CHECKING:
    from ..interface_commands import Tao

logger = logging.getLogger(__name__)


def ipython_shell(tao: Tao) -> None:
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


def _tao_line_cell_magic(tao_instance: Tao, line: str, cell: Optional[str] = None):
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

    _init_output: List[str]
    _last_output: List[str]
    so_lib: ctypes.CDLL
    so_lib_file: str

    def __init__(self, init="", so_lib=""):
        self._init_shared_library(so_lib=so_lib)

        if not init:
            raise TaoInitializationError(
                "pytao now requires an `init` string in order to initialize a new Tao object."
            )
        self._init_output = self.init(init)
        try:
            self.register_cell_magic()
        except Exception:
            pass

    def _init_shared_library(self, so_lib: str) -> None:
        self.so_lib, self.so_lib_file = init_libtao(user_path=so_lib)

    @property
    def init_output(self) -> List[str]:
        """Output from the latest Tao initialization."""
        return list(self._init_output)

    @property
    def last_output(self) -> List[str]:
        return list(self._last_output)

    def get_output(self, reset=True) -> List[str]:
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

    def _init_or_reinit(self, cmd: str) -> Tuple[int, List[str]]:
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
        if not tao_ctypes.initialized:
            logger.debug("Initializing Tao.")
            logger.debug(f"Tao> {cmd}")
            errno = self.so_lib.tao_c_init_tao(cmd.encode("utf-8"))
            if errno == 0:
                # Only mark it initialized on the first actual success.
                tao_ctypes.initialized = True
        else:
            logger.debug("Re-initializing Tao.")

            reinit_cmd = f"reinit tao -clear {cmd}"
            logger.debug(f"Tao> {reinit_cmd}")

            errno = self.so_lib.tao_c_command(reinit_cmd.encode("utf-8"))

        output = self.get_output()

        self._init_output = output
        return errno, output

    def _init_or_raise(self, cmd: str) -> List[str]:
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

    def _check_output_lines(self, cmd: str, lines: List[str]) -> Optional[List[TaoMessage]]:
        """
        Check Tao output for errors respecting the current capture context.

        Parameters
        ----------
        cmd : str
            The Tao command which returned output `lines`.
        lines : List[str]
            Tao output lines for `cmd`.
        """
        ctx = util.error_filter_context.get()
        if ctx is not None:
            return ctx.check_output(cmd, lines)

        err = util.error_in_lines(lines)
        if err:
            raise TaoCommandError(
                f"Command: {cmd} causes error: {err}",
                tao_output="\n".join(lines),
            )

    def cmd(self, cmd, raises=True) -> List[str]:
        """
        Runs a command, and returns the text output.

        Parameters
        ----------
        cmd : str
            Command string
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
            if not raises:
                return self.get_output(reset=False)
            lines = self.get_output(reset=False)
            self._check_output_lines(cmd, lines)
            return lines
        finally:
            self.reset_output()

    def _get_array(
        self,
        cmd: str,
        dtype: Union[Type[float], Type[int]],
        raises: bool,
    ) -> Optional[np.ndarray]:
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
            self._check_output_lines(cmd, self.get_output(reset=False))
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

    def cmd_real(self, cmd: str, raises: bool = True) -> Optional[np.ndarray]:
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

    def cmd_integer(self, cmd: str, raises: bool = True) -> Optional[np.ndarray]:
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


def init_libtao(user_path: str = "") -> Tuple[ctypes.CDLL, str]:
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
    if user_path:
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


# ----------------------------------------------------------------------


class TaoModel(TaoCore):
    """
    Base class for setting up a Tao model in a directory. Builds upon the Tao class.

    If use_tempdir==True, then the input_file and its directory will be copied to a temporary directory.
    If workdir is given, then this temporary directory will be placed in workdir.
    """

    def __init__(
        self,
        input_file="tao.init",
        ploton=True,
        use_tempdir=True,
        workdir=None,
        verbose=True,
        so_lib="",  # Passed onto Tao superclass
        auto_configure=True,  # Should be disables if inheriting.
    ):
        # NOTE: SUPER is being called from configure(...)

        # Save init

        self.original_input_file = input_file
        self.ploton = ploton
        self.use_tempdir = use_tempdir
        self.workdir = workdir
        if workdir:
            assert os.path.exists(workdir), "workdir does not exist: " + workdir

        self.verbose = verbose
        self.so_lib = so_lib

        # Run control
        self.finished = False
        self.configured = False

        if os.path.exists(os.path.expandvars(input_file)):
            f = full_path(input_file)
            self.original_path, self.original_input_file = os.path.split(
                f
            )  # Get original path, filename
            if auto_configure:
                self.configure()
        else:
            self.vprint("Warning: Input file does not exist. Cannot configure.")

    def configure(self):
        # Set paths
        if self.use_tempdir:
            # Need to attach this to the object. Otherwise it will go out of scope.
            self.tempdir = tempfile.TemporaryDirectory(dir=self.workdir)
            # Make yet another directory to overcome the limitations of shutil.copytree
            self.path = full_path(os.path.join(self.tempdir.name, "tao/"))
            # Copy everything in original_path
            shutil.copytree(self.original_path, self.path, symlinks=True)
        else:
            # Work in place
            self.path = self.original_path

        self.input_file = os.path.join(self.path, self.original_input_file)

        self.vprint("Initialized Tao with " + self.input_file)

        # Set up Tao library
        super().__init__(init=self.init_line(), so_lib=self.so_lib)

        self.configured = True

    def init_line(self):
        line = "-init " + self.input_file
        if self.ploton:
            line += " --noplot"
        else:
            line += " -noplot"
        return line

    def reinit(self):
        line = "reinit tao " + self.init_line()
        self.cmd(line)
        self.vprint("Re-initialized with " + line)

    def vprint(self, *args, **kwargs):
        # Verbose print
        if self.verbose:
            print(*args, **kwargs)

    # ---------------------------------
    # Conveniences

    @property
    def globals(self):
        """
        Returns dict of tao parameters.
        Note that the name of this function cannot be named 'global'
        """

        dat = self.cmd("python global")
        return tao_parameter_dict(dat)

    # ---------------------------------
    # [] for set command

    def __setitem__(self, key, item):
        """
        Issue a set command separated by :

        Example:
            TaoModel['global:track_type'] = 'beam'
        will issue command:
            set global track_type = beam
        """

        cmd = form_set_command(key, item, delim=":")
        self.vprint(cmd)
        self.cmd(cmd)

    # ---------------------------------
    def evaluate(self, expression):
        """
        Example:
            .evaluate('lat::orbit.x[beginning:end]')
        Returns an np.array of floats
        """

        return tao_object_evaluate(self, expression)

    # ---------------------------------
    def __str__(self):
        s = "Tao Model initialized from: " + self.original_path
        s += "\n Working in path: " + self.path
        return s

    def init(self, cmd: str) -> List[str]:
        return self._init_or_raise(cmd)


# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# Helper functions


def tao_object_evaluate(tao_object, expression):
    """
    Evaluates an expression and returns

    Example expressions:
        beam::norm_emit.x[end]        # returns a single float
        lat::orbit.x[beginning:end]   # returns an np array of floats
    """

    cmd = f"python evaluate {expression}"
    res = tao_object.cmd(cmd)

    # Cast to float
    vals = [x.split(";")[1] for x in res]

    try:
        fvals = np.asarray(vals, dtype=np.float)
    except Exception:
        fvals = vals

    # Return single value, or array
    if len(fvals) == 1:
        return fvals[0]
    return fvals


def form_set_command(s, value, delim=":"):
    """
    Forms a set command string that is separated by delim.

    Splits into three parts:
    command:what:attribute

    If 'what' had delim inside, the comma should preserve that.

    Example:
    >>>form_set_command('ele:BEG:END:a', 1.23)
    'set ele BEG:END a = 1.23'

    """
    x = s.split(delim)

    cmd0 = x[0]
    what = ":".join(x[1:-1])
    att = x[-1]
    cmd = f"set {cmd0} {what} {att} = {value}"

    # cmd = 'set '+' '.join(x) + f' = {value}'

    return cmd


def apply_settings(tao_object, settings):
    """
    Applies multiple settings to a tao object.
    Checks for lattice_calc_on and plot_on, and temporarily disables these for speed.
    """

    cmds = []

    # Save these
    plot_on = tao_object.globals["plot_on"].value
    lattice_calc_on = tao_object.globals["lattice_calc_on"].value

    if plot_on:
        cmds.append("set global plot_on = F")
    if lattice_calc_on:
        cmds.append("set global lattice_calc_on = F")

    for k, v in settings.items():
        cmd = form_set_command(k, v)
        cmds.append(cmd)

    # Restore
    if lattice_calc_on:
        cmds.append("set global lattice_calc_on = T")

    if plot_on:
        cmds.append("set global plot_on = T")

    for cmd in cmds:
        tao_object.vprint(cmd)
        tao_object.cmd(cmd, raises=True)

    return cmds


# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# Helper functions


def run_tao(
    settings=None,
    run_commands=["set global track_type=single"],
    input_file="tao.init",
    ploton=False,
    workdir=None,
    so_lib="",
    verbose=False,
):
    """
    Creates an LCLSTaoModel object, applies settings, and runs the beam.
    """

    assert os.path.exists(input_file), f"Tao input file does not exist: {input_file}"

    M = TaoModel(
        input_file=input_file,
        ploton=ploton,
        use_tempdir=True,
        workdir=workdir,
        verbose=verbose,
        so_lib=so_lib,  # Passed onto Tao superclass
        auto_configure=True,
    )  # Should be disables if inheriting.

    # Move to local dir, so call commands work
    init_dir = os.getcwd()
    os.chdir(M.path)

    try:
        if settings:
            apply_settings(M, settings)

        for command in run_commands:
            if verbose:
                print("run command:", command)
            M.cmd(command, raises=True)

    finally:
        # Return to init_dir
        os.chdir(init_dir)

    return M
