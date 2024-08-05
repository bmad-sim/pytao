from __future__ import annotations

import logging
import pathlib
import time
from typing import ClassVar, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from .curves import TaoCurveSettings
from .fields import ElementField
from .plot import (
    GraphManager,
    UnsupportedGraphError,
    LatticeLayoutGraph,
)
from .settings import TaoGraphSettings


logger = logging.getLogger(__name__)


class _Defaults:
    layout_height: float = 0.5
    colormap: str = "PRGn_r"


def set_defaults(
    layout_height: Optional[float] = None,
    colormap: Optional[str] = None,
    figsize: Optional[float] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    dpi: Optional[int] = None,
):
    if layout_height is not None:
        _Defaults.layout_height = layout_height
    if colormap is not None:
        _Defaults.colormap = colormap
    if figsize is not None:
        matplotlib.rcParams["figure.figsize"] = figsize
    if width and height:
        matplotlib.rcParams["figure.figsize"] = (width, height)
    if dpi is not None:
        matplotlib.rcParams["figure.dpi"] = dpi


class MatplotlibGraphManager(GraphManager):
    """Matplotlib backend graph manager."""

    _key_: ClassVar[str] = "mpl"

    def plot_grid(
        self,
        graph_names: List[str],
        grid: Tuple[int, int],
        *,
        include_layout: bool = False,
        figsize: Optional[Tuple[float, float]] = None,
        tight_layout: bool = True,
        share_x: Union[bool, Literal["row", "col", "all"]] = "col",
        layout_height: Optional[float] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        xlim: Optional[List[Optional[Tuple[float, float]]]] = None,
        ylim: Optional[List[Optional[Tuple[float, float]]]] = None,
        curves: Optional[List[Dict[int, TaoCurveSettings]]] = None,
        settings: Optional[List[TaoGraphSettings]] = None,
        save: Union[bool, str, pathlib.Path, None] = None,
        axes: Optional[List[List[matplotlib.axes.Axes]]] = None,
    ):
        """
        Plot graphs on a grid with Matplotlib.

        Parameters
        ----------
        graph_names : list of str
            Graph names.
        grid : (nrows, ncols), optional
            Grid the provided graphs into this many rows and columns.
        include_layout : bool, default=False
            Include a layout plot at the bottom of each column.
        tight_layout : bool, default=True
            Apply a tight layout with matplotlib.
        figsize : (float, float), optional
            Figure size. Alternative to specifying `width` and `height`
            separately.  This takes precedence over `width` and `height`.
            Defaults to Matplotlib's `rcParams["figure.figsize"]``.
        width : float, optional
            Width of the whole plot.
        height : float, optional
            Height of the whole plot.
        layout_height : int, optional
            Normalized height of the layout plot - assuming regular plots are
            of height 1.  Default is 0.5 which is configurable with `set_defaults`.
        share_x : bool or None, default=None
            Share x-axes where sensible (`None`) or force sharing x-axes (True)
            for all plots.
        xlim : list of (float, float), optional
            X axis limits for each graph.
        ylim : list of (float, float), optional
            Y axis limits for each graph.
        curves : list of Dict[int, TaoCurveSettings], optional
            One dictionary per graph, with each dictionary mapping the curve
            index to curve settings. These settings will be applied to the
            placed graphs prior to plotting.
        settings : list of TaoGraphSettings, optional
            Graph customization settings, per graph.
        save : pathlib.Path or str, optional
            Save the plot to the given filename.

        Returns
        -------
        list of graphs
            List of plotted graphs.
        matplotlib.Figure
            To gain access to the resulting plot objects, use the backend's
            `plot` method directly.
        List[List[matplotlib.axes.Axes]]
            Gridded axes, accessible with `grid[row][col]`.
        """

        graphs = self.prepare_grid_by_names(
            graph_names=graph_names,
            curves=curves,
            settings=settings,
        )
        nrows, ncols = grid
        height_ratios = None

        if figsize is None and width and height:
            figsize = (width, height)

        if include_layout:
            layout_height = layout_height or _Defaults.layout_height
            empty_graph_count = nrows * ncols - len(graph_names)
            if empty_graph_count < ncols:
                # Add a row for the layout
                nrows += 1
            height_ratios = [1] * (nrows - 1) + [layout_height]

        if axes is not None:
            tight_layout = False
            fig = None
        else:
            fig, gs = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                sharex=share_x,
                figsize=figsize,
                squeeze=False,
                height_ratios=height_ratios,
            )
            axes = [gs[row, :] for row in range(nrows)]
            for row in axes:
                for ax in row:
                    ax.set_axis_off()

        xlim = xlim or [None]
        if len(xlim) < len(graphs):
            xlim.extend([xlim[-1]] * (len(graphs) - len(xlim)))

        ylim = ylim or [None]
        if len(ylim) < len(graphs):
            ylim.extend([ylim[-1]] * (len(graphs) - len(ylim)))

        rows_cols = [(row, col) for row in range(nrows) for col in range(ncols)]

        for graph, xl, yl, (row, col) in zip(graphs, xlim, ylim, rows_cols):
            ax = axes[row][col]
            try:
                graph.plot(ax)
            except UnsupportedGraphError:
                continue

            ax.set_axis_on()
            if xl is not None:
                ax.set_xlim(*xl)
            if yl is not None:
                ax.set_ylim(*yl)

        if include_layout:
            layout_graph = self.lattice_layout_graph
            for col in range(ncols):
                ax = axes[-1][col]
                layout_graph.plot(ax)
                ax.set_axis_on()

        if tight_layout and fig is not None:
            fig.tight_layout()

        if save and fig is not None:
            title = graphs[0].title or f"plot-{time.time()}"
            if save is True:
                save = f"{title}.png"
            logger.info(f"Saving plot to {save!r}")
            fig.savefig(save)

        return graphs, fig, axes

    def plot(
        self,
        graph_name: str,
        *,
        region_name: Optional[str] = None,
        include_layout: bool = True,
        tight_layout: bool = True,
        width: Optional[float] = None,
        height: Optional[float] = None,
        layout_height: Optional[float] = None,
        figsize: Optional[Tuple[float, float]] = None,
        share_x: bool = True,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        save: Union[bool, str, pathlib.Path, None] = None,
        settings: Optional[TaoGraphSettings] = None,
        curves: Optional[Dict[int, TaoCurveSettings]] = None,
        axes: Optional[List[matplotlib.axes.Axes]] = None,
    ):
        """
        Plot a graph with Matplotlib.

        Parameters
        ----------
        graph_name : str
            Graph template name.
        region_name : str, optional
            Graph region name.
        include_layout : bool, optional
            Include a layout plot at the bottom, if not already placed and if
            appropriate (i.e., another plot uses longitudinal coordinates on
            the x-axis).
        tight_layout : bool, default=True
            Apply a tight layout with matplotlib.
        figsize : (float, float), optional
            Figure size. Alternative to specifying `width` and `height`
            separately.  This takes precedence over `width` and `height`.
            Defaults to Matplotlib's `rcParams["figure.figsize"]``.
        width : float, optional
            Width of the whole plot.
        height : float, optional
            Height of the whole plot.
        layout_height : float, optional
            Normalized height of the layout plot - assuming regular plots are
            of height 1.  Default is 0.5 which is configurable with `set_defaults`.
        share_x : bool or None, default=None
            Share x-axes where sensible (`None`) or force sharing x-axes (True)
            for all plots.
        xlim : (float, float), optional
            X axis limits.
        ylim : (float, float), optional
            Y axis limits.
        save : pathlib.Path or str, optional
            Save the plot to the given filename.
        curves : Dict[int, TaoCurveSettings], optional
            Dictionary of curve index to curve settings. These settings will be
            applied to the placed graph prior to plotting.
        settings : TaoGraphSettings, optional
            Graph customization settings.

        Returns
        -------
        list of graphs
            List of plotted graphs.
        matplotlib.Figure
            To gain access to the resulting plot objects, use the backend's
            `plot` method directly.
        List[matplotlib.axes.Axes]
        """
        graphs = self.prepare_graphs_by_name(
            graph_name=graph_name,
            region_name=region_name,
            curves=curves,
            settings=settings,
        )
        if not graphs:
            return None

        if figsize is None and width and height:
            figsize = (width, height)

        if (
            include_layout
            and not any(isinstance(graph, LatticeLayoutGraph) for graph in graphs)
            and any(graph.is_s_plot for graph in graphs)
        ):
            layout_graph = self.lattice_layout_graph
            graphs.append(layout_graph)
        else:
            include_layout = False

        if axes is not None:
            if len(axes) != len(graphs):
                raise ValueError(
                    f"Not enough axes provided. Expected {len(graphs)}, got {len(axes)}"
                )
            fig = axes[0].figure
        else:
            if include_layout:
                layout_height = layout_height or _Defaults.layout_height
                fig, gs = plt.subplots(
                    nrows=len(graphs),
                    ncols=1,
                    sharex=share_x,
                    height_ratios=[1] * (len(graphs) - 1) + [layout_height],
                    figsize=figsize,
                    squeeze=False,
                )
            else:
                fig, gs = plt.subplots(
                    nrows=len(graphs),
                    ncols=1,
                    sharex=share_x,
                    figsize=figsize,
                    squeeze=False,
                )
            axes = gs[:, 0]
            assert axes is not None

        if include_layout:
            layout_graph = self.lattice_layout_graph

        for ax, graph in zip(axes, graphs):
            try:
                graph.plot(ax)
            except UnsupportedGraphError:
                continue

            if xlim is not None:
                ax.set_xlim(xlim)

            if ylim is not None:
                if not isinstance(graph, LatticeLayoutGraph) or len(graphs) == 1:
                    ax.set_ylim(ylim)

        if fig is not None:
            if tight_layout:
                fig.tight_layout()

            if save:
                title = graphs[0].title or f"plot-{time.time()}"
                if save is True:
                    save = f"{title}.png"
                logger.info(f"Saving plot to {save!r}")
                fig.savefig(save)

        return graphs, fig, axes

    __call__ = plot

    def plot_field(
        self,
        ele_id: str,
        *,
        colormap: Optional[str] = None,
        radius: float = 0.015,
        num_points: int = 100,
        figsize: Optional[Tuple[int, int]] = None,
        width: int = 4,
        height: int = 4,
        x_scale: float = 1e3,
        ax: Optional[matplotlib.axes.Axes] = None,
        save: Union[bool, str, pathlib.Path, None] = None,
    ):
        """
        Plot field information for a given element.

        Parameters
        ----------
        ele_id : str
            Element ID.
        colormap : str, optional
            Colormap for the plot.
            Matplotlib defaults to "PRGn_r", and bokeh defaults to "".
        radius : float, default=0.015
            Radius.
        num_points : int, default=100
            Number of data points.
        ax : matplotlib.axes.Axes, optional
            The axes to place the plot in.
        save : pathlib.Path or str, optional
            Save the plot to the given filename.
        """
        user_specified_axis = ax is not None

        if figsize is None and width and height:
            figsize = (width, height)

        if ax is None:
            _, ax = plt.subplots(figsize=(width, height))
        assert ax is not None

        colormap = colormap or _Defaults.colormap

        field = ElementField.from_tao(self.tao, ele_id, num_points=num_points, radius=radius)
        mesh = ax.pcolormesh(
            np.asarray(field.s),
            np.asarray(field.x) * x_scale,
            np.asarray(field.by),
            # vmin=min_field,
            # vmax=max_field,
            cmap=colormap,
        )
        fig = ax.figure
        if fig is not None:
            if not user_specified_axis:
                fig.colorbar(mesh)

            if save:
                if save is True:
                    save = f"{ele_id}_field.png"
                if not pathlib.Path(save).suffix:
                    save = f"{save}.png"
                logger.info(f"Saving plot to {save!r}")
                fig.savefig(save)

        return field, fig, ax
