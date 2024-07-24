from __future__ import annotations
import logging
import pathlib
import numpy as np
import typing

from typing import Optional, List, Tuple, Union


from .tao_ctypes.core import TaoCore
from .tao_ctypes.util import parse_tao_python_data
from .util.command import make_tao_init
from .util.parameters import tao_parameter_dict
from .util import parsers as _pytao_parsers
from .plotting.util import select_graph_manager_class

if typing.TYPE_CHECKING:
    from .plotting.bokeh import BokehGraphManager, NotebookGraphManager  # noqa: F401
    from .plotting.plot import MatplotlibGraphManager


logger = logging.getLogger(__name__)
AnyPath = Union[pathlib.Path, str]


class Tao(TaoCore):
    """
    Communicate with Tao using ctypes.

    Parameters
    ----------
    init : str, optional
        Initialization string for Tao.  Same as the tao command-line, including
        "-init" and such.
    so_lib : str, optional
        Path to the Tao shared library.  Auto-detected if not specified.
    expand_vars : bool, optional
        Expand shell variables in `init` strings.  For example,
        an `init` string of `$HOME` would be replaced by your home directory.
    plot : str, bool, optional
        Use pytao's plotting mechanism with matplotlib or bokeh, if available.
        If `True`, pytao will pick an appropriate plotting backend.
        If `False` or "tao", Tao plotting will be used. (Default)
        If "mpl", the pytao matplotlib plotting backend will be selected.
        If "bokeh", the pytao Bokeh plotting backend will be selected.

    beam_file: AnyPath, default=None
        File containing the tao_beam_init namelist.
    beam_init_position_file: pathlib.Path or str, default=None
        File containing initial particle positions.
    building_wall_file: AnyPath, default=None
        Define the building tunnel wall
    command: str, optional
        Commands to run after startup file commands
    data_file: AnyPath, default=None
        Define data for plotting and optimization
    debug: bool, default=False
        Debug mode for Wizards
    disable_smooth_line_calc: bool, default=False
        Disable the smooth line calc used in plotting
    external_plotting: bool, default=False
        Tells Tao that plotting is done externally to Tao.
    geometry: "wxh" or (width, height) tuple, optional
        Plot window geometry (pixels)
    hook_init_file:  pathlib.Path or str, default=None
        Init file for hook routines (Default = tao_hook.init)
    init_file: AnyPath, default=None
        Tao init file
    lattice_file: AnyPath, default=None
        Bmad lattice file
    log_startup: bool, default=False
        Write startup debugging info
    no_stopping: bool, default=False
        For debugging: Prevents Tao from exiting on errors
    noinit: bool, default=False
        Do not use Tao init file.
    noplot: bool, default=False
        Do not open a plotting window
    nostartup: bool, default=False
        Do not open a startup command file
    no_rad_int: bool, default=False
        Do not do any radiation integrals calculations.
    plot_file: AnyPath, default=None
        Plotting initialization file
    prompt_color: str, optional
        Set color of prompt string. Default is blue.
    reverse: bool, default=False
        Reverse lattice element order?
    rf_on: bool, default=False
        Use "--rf_on" to turn off RF (default is now RF on)
    quiet: bool, default=False
        Suppress terminal output when running a command file?
    slice_lattice: str, optional
        Discards elements from lattice that are not in the list
    start_branch_at: str, optional
        Start lattice branch at element.
    startup_file: AnyPath, default=None
        Commands to run after parsing Tao init file
    symbol_import: bool, default=False
        Import symbols defined in lattice files(s)?
    var_file: AnyPath, default=None
        Define variables for plotting and optimization
    """

    plot_backend_name: Optional[str]
    _use_pytao_plotting: bool
    _graph_managers: dict

    def __init__(
        self,
        init: str = "",
        so_lib: str = "",
        *,
        expand_vars: bool = True,
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
        if plot == "tao":
            plot = False

        init = make_tao_init(
            init,
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

        self._use_pytao_plotting = plot in {"mpl", "bokeh", True}
        if plot is not None and self._use_pytao_plotting:
            self.plot_backend_name = plot
        else:
            self.plot_backend_name = None

        self._graph_managers = {}
        super().__init__(init=init, so_lib=so_lib, expand_vars=expand_vars)

    def init(self, cmd: str) -> List[str]:
        """(Re-)Initialize Tao with the given command."""
        if self._use_pytao_plotting:
            cmd = make_tao_init(cmd, noplot=True, external_plotting=True)

        return super().init(cmd)

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

    def plot(
        self,
        graph_name: Optional[str] = None,
        *,
        region_name: Optional[str] = None,
        include_layout: bool = True,
        place: bool = True,
        update: bool = True,
        width: Optional[int] = None,
        height: Optional[int] = None,
        layout_height: Optional[int] = None,
        share_x: Optional[bool] = None,
        backend: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Make a plot with the provided backend.

        Plot a graph, region, or all placed graphs.

        To plot a specific graph, specify `graph_name` (optionally `region_name`).
        To plot a specific region, specify `region_name`.
        To plot all placed graphs, specify neither.

        For full details on available parameters, see the specific backend's
        graph manager. For example:

        In [1]: tao.bokeh.plot?
        In [2]: tao.matplotlib.plot?

        Parameters
        ----------
        graph_name : str, optional
            Graph name.
        region_name : str, optional
            Graph region name.
        include_layout : bool, optional
            Include a layout plot at the bottom, if not already placed and if
            appropriate (i.e., another plot uses longitudinal coordinates on
            the x-axis).
        place : bool, default=True
            Place all requested plots prior to continuing.
        update : bool, default=True
            Query Tao to update relevant graphs prior to plotting.
        width : int, optional
            Width of each plot.
        height : int, optional
            Height of each plot.
        layout_height : int, optional
            Height of the layout plot.
        share_x : bool or None, default=None
            Share x-axes where sensible (`None`) or force sharing x-axes (True)
            for all plots.
        backend : {"bokeh", "mpl"}, optional
            The backend to use.  Auto-detects Jupyter and availability of bokeh
            to select a backend.

        Returns
        -------
        None
            To gain access to the resulting plot objects, use the backend's
            `plot` method directly.
        """
        if backend is None:
            backend = self.plot_backend_name or select_graph_manager_class()._key_

        if backend not in {"mpl", "bokeh"}:
            raise ValueError(f"Unsupported backend: {backend}")

        if width is not None:
            kwargs["width"] = width
        if height is not None:
            kwargs["height"] = height
        if layout_height is not None:
            kwargs["layout_height"] = layout_height
        if share_x is not None:
            kwargs["share_x"] = share_x
        manager = self._get_graph_manager_by_key(backend)
        manager.plot(
            region_name=region_name,
            graph_name=graph_name,
            include_layout=include_layout,
            place=place,
            update=update,
            **kwargs,
        )
