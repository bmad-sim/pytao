import logging
import numpy as np

from typing import Optional


from .tao_ctypes.core import TaoCore
from .tao_ctypes.util import parse_tao_python_data
from .util.parameters import tao_parameter_dict
from .util import parsers as _pytao_parsers
from .plotting.util import select_graph_manager_class


logger = logging.getLogger(__name__)


class Tao(TaoCore):
    plot_backend: Optional[str] = None

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

    def plot(
        self,
        region_name: Optional[str] = None,
        graph_name: Optional[str] = None,
        *,
        include_layout: bool = True,
        place: bool = True,
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

        To plot a specific graph, specify `region_name` and `graph_name`.
        To plot a specific region, specify `region_name`.
        To plot all placed graphs, specify neither.

        For full details on available parameters, see the specific backend's
        graph manager. For example:

        In [1]: tao.bokeh.plot?
        In [2]: tao.matplotlib.plot?

        Parameters
        ----------
        region_name : str, optional
            Graph region name.
        graph_name : str, optional
            Graph name.
        include_layout : bool
            Include a layout plot at the bottom, if not already placed and if
            appropriate (i.e., another plot uses longitudinal coordinates on
            the x-axis).
        place : bool
            Place all requested plots prior to continuing.
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
            backend = self.plot_backend or select_graph_manager_class()._key_

        if backend not in {"mpl", "bokeh"}:
            raise ValueError(f"Unsupported backend: {backend}")

        manager = self._get_graph_manager_by_key(backend)
        manager.plot(
            region_name=region_name,
            graph_name=graph_name,
            include_layout=include_layout,
            place=place,
            width=width,
            height=height,
            layout_height=layout_height,
            share_x=share_x,
        )
