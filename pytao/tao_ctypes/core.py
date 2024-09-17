import ctypes
import logging
import os
import shutil
import tempfile
import textwrap
from ctypes.util import find_library
from typing import List, Optional, Tuple, Type, Union

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

logger = logging.getLogger(__name__)


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
            logger.debug(f"Initializing Tao with: {cmd!r}")
            errno = self.so_lib.tao_c_init_tao(cmd.encode("utf-8"))
            tao_ctypes.initialized = True
            output = self.get_output()
        else:
            errno = 0
            output = self.cmd(f"reinit tao -clear {cmd}", raises=False)

        self._init_output = output
        return errno, output

    def init(self, cmd: str) -> List[str]:
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
        try:
            self._check_output_lines(cmd=f"init {cmd}", lines=output)
        except TaoCommandError as ex:
            message = textwrap.indent("\n".join(output), "  ")
            raise TaoInitializationError(str(ex), tao_output="\n".join(output)) from None

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

    def register_cell_magic(self):
        """
        Registers a cell magic in Jupyter notebooks.

        Example:

            %%tao
            sho lat
        """

        from IPython.core.magic import register_cell_magic

        @register_cell_magic
        def tao(line, cell):
            cell = cell.format(**globals())
            cmds = cell.split("\n")
            for c in cmds:
                print("-------------------------")
                print("Tao> " + c)
                res = self.cmd(c)
                for line in res:
                    print(line)

        del tao


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
