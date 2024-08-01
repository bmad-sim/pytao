from __future__ import annotations
import contextlib
import datetime
import logging
import pathlib
import numpy as np
import typing

from dataclasses import asdict
from pydantic import ConfigDict, dataclasses
from typing import Any, Dict, List, Optional, Tuple, Union
from typing_extensions import override

from .tao_ctypes.core import TaoCore
from .tao_ctypes.util import parse_tao_python_data
from .util.command import make_tao_init
from .util.parameters import tao_parameter_dict
from .util import parsers as _pytao_parsers
from .plotting.util import select_graph_manager_class

if typing.TYPE_CHECKING:
    from .plotting.bokeh import BokehGraphManager, NotebookGraphManager  # noqa: F401
    from .plotting.plot import MatplotlibGraphManager

    from .subproc import SubprocessTao

    AnyTao = Union["Tao", SubprocessTao]


logger = logging.getLogger(__name__)
AnyPath = Union[pathlib.Path, str]


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
    quiet : bool, default=False
        Suppress terminal output when running a command file?
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

    init: str = dataclasses.Field(default="", kw_only=False)
    so_lib: str = dataclasses.Field(default="", kw_only=False)

    metadata: Dict[str, Any] = dataclasses.Field(default_factory=dict)
    plot: Union[str, bool] = "tao"
    beam_file: Optional[AnyPath] = None
    beam_init_position_file: Optional[AnyPath] = None
    building_wall_file: Optional[AnyPath] = None
    command: str = ""
    data_file: Optional[AnyPath] = None
    debug: bool = False
    disable_smooth_line_calc: bool = False
    external_plotting: bool = False
    geometry: Union[str, Tuple[float, float]] = ""
    hook_init_file: Optional[AnyPath] = None
    init_file: Optional[AnyPath] = None
    lattice_file: Optional[AnyPath] = None
    log_startup: bool = False
    no_stopping: bool = False
    noinit: bool = False
    noplot: bool = False
    nostartup: bool = False
    no_rad_int: bool = False
    plot_file: Optional[AnyPath] = None
    prompt_color: str = ""
    reverse: bool = False
    rf_on: bool = False
    quiet: bool = False
    slice_lattice: str = ""
    start_branch_at: str = ""
    startup_file: Optional[AnyPath] = None
    symbol_import: bool = False
    var_file: Optional[AnyPath] = None

    @property
    def run_parameters(self) -> Dict[str, Any]:
        """Parameters used to initialize Tao or make a new Tao instance."""
        params = {
            key: value
            for key, value in asdict(self).items()
            if value != getattr(type(self), key, None)
        }
        params.setdefault("init", "")
        params.pop("metadata")
        return params

    @property
    def tao_init(self) -> str:
        """Tao.init() command string."""
        params = self.run_parameters
        # For tao.init(), we throw away Tao class-specific things:
        params.pop("so_lib", None)
        params.pop("plot", None)
        return make_tao_init(**params)

    def run(self, use_subprocess: bool = False) -> AnyTao:
        """Create a new Tao instance and run it using these settings."""
        params = self.run_parameters
        if use_subprocess:
            from .subproc import SubprocessTao

            return SubprocessTao(**params)
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


class Tao(TaoCore):
    """
    Communicate with Tao using ctypes.

    Parameters
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
    quiet : bool, default=False
        Suppress terminal output when running a command file?
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

    plot_backend_name: Optional[str]
    _graph_managers: dict
    _min_tao_version = datetime.datetime(2024, 7, 26)

    def __init__(
        self,
        init: str = "",
        so_lib: str = "",
        *,
        plot: Union[str, bool] = "tao",
        beam_file: Optional[AnyPath] = None,
        beam_init_position_file: Optional[AnyPath] = None,
        building_wall_file: Optional[AnyPath] = None,
        command: str = "",
        data_file: Optional[AnyPath] = None,
        debug: bool = False,
        disable_smooth_line_calc: bool = False,
        external_plotting: bool = False,
        geometry: Union[str, Tuple[float, float]] = "",
        hook_init_file: Optional[AnyPath] = None,
        init_file: Optional[AnyPath] = None,
        lattice_file: Optional[AnyPath] = None,
        log_startup: bool = False,
        no_stopping: bool = False,
        noinit: bool = False,
        noplot: bool = False,
        nostartup: bool = False,
        no_rad_int: bool = False,
        plot_file: Optional[AnyPath] = None,
        prompt_color: str = "",
        reverse: bool = False,
        rf_on: bool = False,
        quiet: bool = False,
        slice_lattice: str = "",
        start_branch_at: str = "",
        startup_file: Optional[AnyPath] = None,
        symbol_import: bool = False,
        var_file: Optional[AnyPath] = None,
    ):
        self.plot_backend_name = None
        self._graph_managers = {}
        self._tao_version_checked = False
        super().__init__(init="", so_lib=so_lib)
        self.init(
            cmd=init,
            plot=plot,
            beam_file=beam_file,
            beam_init_position_file=beam_init_position_file,
            building_wall_file=building_wall_file,
            command=command,
            data_file=data_file,
            debug=debug,
            disable_smooth_line_calc=disable_smooth_line_calc,
            external_plotting=external_plotting,
            geometry=geometry,
            hook_init_file=hook_init_file,
            init_file=init_file,
            lattice_file=lattice_file,
            log_startup=log_startup,
            no_stopping=no_stopping,
            noinit=noinit,
            noplot=noplot,
            nostartup=nostartup,
            no_rad_int=no_rad_int,
            plot_file=plot_file,
            prompt_color=prompt_color,
            reverse=reverse,
            rf_on=rf_on,
            quiet=quiet,
            slice_lattice=slice_lattice,
            start_branch_at=start_branch_at,
            startup_file=startup_file,
            symbol_import=symbol_import,
            var_file=var_file,
        )

    @override
    def init(
        self,
        cmd: str = "",
        *,
        plot: Union[str, bool] = "tao",
        beam_file: Optional[AnyPath] = None,
        beam_init_position_file: Optional[AnyPath] = None,
        building_wall_file: Optional[AnyPath] = None,
        command: str = "",
        data_file: Optional[AnyPath] = None,
        debug: bool = False,
        disable_smooth_line_calc: bool = False,
        external_plotting: bool = False,
        geometry: Union[str, Tuple[float, float]] = "",
        hook_init_file: Optional[AnyPath] = None,
        init_file: Optional[AnyPath] = None,
        lattice_file: Optional[AnyPath] = None,
        log_startup: bool = False,
        no_stopping: bool = False,
        noinit: bool = False,
        noplot: bool = False,
        nostartup: bool = False,
        no_rad_int: bool = False,
        plot_file: Optional[AnyPath] = None,
        prompt_color: str = "",
        reverse: bool = False,
        rf_on: bool = False,
        quiet: bool = False,
        slice_lattice: str = "",
        start_branch_at: str = "",
        startup_file: Optional[AnyPath] = None,
        symbol_import: bool = False,
        var_file: Optional[AnyPath] = None,
    ) -> None:
        """(Re-)Initialize Tao with the given command."""
        if plot in {"mpl", "bokeh"}:
            self.plot_backend_name = plot
        else:
            self.plot_backend_name = None

        use_pytao_plotting = plot in {"mpl", "bokeh", True}

        self.init_settings = TaoStartup(
            init=cmd,
            plot=plot,
            beam_file=beam_file,
            beam_init_position_file=beam_init_position_file,
            building_wall_file=building_wall_file,
            command=command,
            data_file=data_file,
            debug=debug,
            disable_smooth_line_calc=disable_smooth_line_calc,
            external_plotting=use_pytao_plotting or external_plotting,
            geometry=geometry,
            hook_init_file=hook_init_file,
            init_file=init_file,
            lattice_file=lattice_file,
            log_startup=log_startup,
            no_stopping=no_stopping,
            noinit=noinit,
            noplot=use_pytao_plotting or noplot,
            nostartup=nostartup,
            no_rad_int=no_rad_int,
            plot_file=plot_file,
            prompt_color=prompt_color,
            reverse=reverse,
            rf_on=rf_on,
            quiet=quiet,
            slice_lattice=slice_lattice,
            start_branch_at=start_branch_at,
            startup_file=startup_file,
            symbol_import=symbol_import,
            var_file=var_file,
        )

        init_cmd = self.init_settings.tao_init
        if "-init" in init_cmd or "-lat" in init_cmd:
            self._init(self.init_settings)
            if not self._tao_version_checked:
                self._tao_version_checked = True
                self._check_tao_version()

    def _check_tao_version(self):
        version = self.version()
        if version is None:
            # Don't continue to warn about failing to parse the version
            return

        if version.date() < self._min_tao_version.date():
            logger.warning(
                f"Installed Tao version is lower than pytao's recommended and tested version. "
                f"\n   You have Tao version: {version.date()}"
                f"\n   Recommended version:  {self._min_tao_version.date()}"
                f"\nSome features may not work as expected.  Please upgrade bmad."
            )

    def _init(self, startup: TaoStartup):
        return super().init(startup.tao_init)

    def __execute(
        self,
        cmd: str,
        as_dict: bool = True,
        raises: bool = True,
        method_name=None,
        cmd_type: str = "string_list",
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
        func_for_type = {
            "string_list": self.cmd,
            "real_array": self.cmd_real,
            "integer_array": self.cmd_integer,
        }
        func = func_for_type.get(cmd_type, self.cmd)
        raw_output = func(cmd, raises=raises)
        special_parser = getattr(_pytao_parsers, f"parse_{method_name}", None)
        try:
            if special_parser and callable(special_parser):
                return special_parser(raw_output, cmd=cmd)
            if "string" not in cmd_type:
                return raw_output
            if as_dict:
                return parse_tao_python_data(raw_output)
            return tao_parameter_dict(raw_output)
        except Exception:
            if raises:
                raise
            logger.exception(
                "Failed to parse string data with custom parser. Returning raw value."
            )
            return raw_output

    def bunch_data(self, ele_id, *, which="model", ix_bunch=1, verbose=False):
        """
        Returns bunch data in openPMD-beamphysics format/notation.

        Notes
        -----
        Note that Tao's 'write beam' will also write a proper h5 file in this format.

        Expected usage:
            data = bunch_data(tao, 'end')
            from pmd_beamphysics import ParticleGroup
            P = ParicleGroup(data=data)


        Returns
        -------
        data : dict
            dict of arrays, with keys 'x', 'px', 'y', 'py', 't', 'pz',
            'status', 'weight', 'z', 'species'


        Examples
        --------
        Example: 1
        init: $ACC_ROOT_DIR/tao/examples/csr_beam_tracking/tao.init
        args:
        ele_id: end
        which: model
        ix_bunch: 1

        """

        # Get species
        stats = self.bunch_params(ele_id, which=which, verbose=verbose)
        species = stats["species"]

        dat = {}
        for coordinate in ["x", "px", "y", "py", "t", "pz", "p0c", "charge", "state"]:
            dat[coordinate] = self.bunch1(
                ele_id,
                coordinate=coordinate,
                which=which,
                ix_bunch=ix_bunch,
                verbose=verbose,
            )

        # Remove normalizations
        p0c = dat.pop("p0c")

        dat["status"] = dat.pop("state")
        dat["weight"] = dat.pop("charge")

        # px from Bmad is px/p0c
        # pz from Bmad is delta = p/p0c -1.
        # pz = sqrt( (delta+1)**2 -px**2 -py**2)*p0c
        dat["pz"] = np.sqrt((dat["pz"] + 1) ** 2 - dat["px"] ** 2 - dat["py"] ** 2) * p0c
        dat["px"] = dat["px"] * p0c
        dat["py"] = dat["py"] * p0c

        # z = 0 by definition
        dat["z"] = np.full(len(dat["x"]), 0)

        dat["species"] = species.lower()

        return dat

    def version(self) -> Optional[datetime.datetime]:
        """Get the date-coded version."""
        cmd = "show version"
        return _pytao_parsers.parse_show_version(self.cmd(cmd), cmd=cmd)

    def plot_page(self):
        """Get plot page parameters."""
        cmd = "show plot_page"
        return _pytao_parsers.parse_show_plot_page(self.cmd(cmd), cmd=cmd)

    def _get_graph_manager_by_key(self, key: str):
        graph_manager = self._graph_managers.get(key, None)

        if graph_manager is None:
            if key == "bokeh":
                from .plotting.bokeh import select_graph_manager_class

                cls = select_graph_manager_class()
            elif key == "mpl":
                from .plotting import MatplotlibGraphManager as cls

            else:
                raise NotImplementedError(key)

            graph_manager = cls(self)
            self._graph_managers[key] = graph_manager
        return graph_manager

    @property
    def matplotlib(self):
        """Get a matplotlib graph manager."""
        return self._get_graph_manager_by_key("mpl")

    @property
    def bokeh(self):
        """Get a matplotlib graph manager."""
        return self._get_graph_manager_by_key("bokeh")

    @property
    def plot_manager(
        self,
    ) -> Union[BokehGraphManager, NotebookGraphManager, MatplotlibGraphManager]:
        """
        The currently-configured plot graph manager.

        This can be configured at initialization time by specifying
        `plot="mpl"`, for example.
        This may also be reconfigured by changing the attribute
        `plot_backend_name`.
        """
        return self._get_graph_manager_by_key(self.plot_backend_name or "mpl")

    def _get_user_specified_backend(self, backend: Optional[str]):
        if backend is None:
            backend = self.plot_backend_name or select_graph_manager_class()._key_

        if not self.init_settings.external_plotting:
            raise RuntimeError(
                "Tao was not configured for external plotting, which pytao requires. "
                "Please re-initialize Tao and set `plot=True` (or specify a backend). "
                "For example: tao.init(..., plot=True)"
            )

        if backend not in {"mpl", "bokeh"}:
            raise ValueError(f"Unsupported backend: {backend}")

        return self._get_graph_manager_by_key(backend)

    def plot(
        self,
        graph_name: Optional[Union[str, List[str]]] = None,
        *,
        region_name: Optional[str] = None,
        include_layout: bool = True,
        width: Optional[int] = None,
        height: Optional[int] = None,
        layout_height: Optional[int] = None,
        share_x: Optional[bool] = None,
        backend: Optional[str] = None,
        reuse: bool = True,
        grid: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> None:
        """
        Make a plot with the provided backend.

        Plot a graph, or all placed graphs.

        To plot a specific graph, specify `graph_name` (optionally `region_name`).
        The default is to plot all placed graphs.

        For full details on available parameters, see the specific backend's
        graph manager. For example:

        In [1]: tao.bokeh.plot?
        In [2]: tao.matplotlib.plot?

        Parameters
        ----------
        graph_name : str or list[str]
            Graph name or names.
        region_name : str, optional
            Graph region name.  Chosen automatically if not specified.
        include_layout : bool, optional
            Include a layout plot at the bottom, if not already placed and if
            appropriate (i.e., another plot uses longitudinal coordinates on
            the x-axis).
        width : int, optional
            Width of each plot.
        height : int, optional
            Height of each plot.
        layout_height : int, optional
            Height of the layout plot.
        share_x : bool or None, default=None
            Share x-axes where sensible (`None`) or force sharing x-axes (True)
            for all plots.
        save : pathlib.Path or str, optional
            Save the plot to the given filename.
        xlim : (float, float), optional
            X axis limits.
        ylim : (float, float), optional
            Y axis limits.
        reuse : bool, default=True
            If an existing plot of the given template type exists, reuse the
            existing plot region rather than selecting a new empty region.
        grid : (nrows, ncols), optional
            If multiple graph names are specified, the plots will be placed
            in a grid according to this parameter.  The default is to have
            stacked plots if this parameter is unspecified.
        backend : {"bokeh", "mpl"}, optional
            The backend to use.  Auto-detects Jupyter and availability of bokeh
            to select a backend.

        Returns
        -------
        None
            To gain access to the resulting plot objects, use the backend's
            `plot` method directly.
        """
        manager = self._get_user_specified_backend(backend)

        if width is not None:
            kwargs["width"] = width
        if height is not None:
            kwargs["height"] = height
        if layout_height is not None:
            kwargs["layout_height"] = layout_height
        if share_x is not None:
            kwargs["share_x"] = share_x

        if not graph_name:
            self.last_plot = manager.plot_all(
                include_layout=include_layout,
                reuse=reuse,
                **kwargs,
            )
        elif not isinstance(graph_name, str):
            graph_names = list(graph_name)
            grid = grid or (len(graph_names), 1)
            self.last_plot = manager.plot_grid(
                graph_names=graph_names,
                grid=grid,
                reuse=reuse,
                include_layout=include_layout,
                **kwargs,
            )
        else:
            self.last_plot = manager.plot(
                region_name=region_name,
                graph_name=graph_name,
                include_layout=include_layout,
                reuse=reuse,
                **kwargs,
            )

    def plot_field(
        self,
        ele_id: str,
        *,
        colormap: Optional[str] = None,
        radius: float = 0.015,
        num_points: int = 100,
        backend: Optional[str] = None,
        **kwargs,
    ):
        """
        Plot field information for a given element.

        Parameters
        ----------
        ele_id : str
            Element ID.
        colormap : str, optional
            Colormap for the plot.
            Matplotlib defaults to "PRGn_r", and bokeh defaults to "Magma256".
        radius : float, default=0.015
            Radius.
        num_points : int, default=100
            Number of data points.
        backend : {"bokeh", "mpl"}, optional
            The backend to use.  Auto-detects Jupyter and availability of bokeh
            to select a backend.
        """
        manager = self._get_user_specified_backend(backend)
        self.last_plot = manager.plot_field(
            ele_id,
            colormap=colormap,
            radius=radius,
            num_points=num_points,
            **kwargs,
        )
