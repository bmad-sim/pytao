from __future__ import annotations

import functools
import logging
import pathlib
import time
from typing import ClassVar, Dict, List, Literal, Optional, Sequence, Tuple, Union

import matplotlib.axes
import matplotlib.axis
import matplotlib.cm
import matplotlib.collections
import matplotlib.patches
import matplotlib.path
import matplotlib.pyplot as plt
import matplotlib.text
import matplotlib.ticker
import numpy as np

from . import floor_plan_shapes, layout_shapes, pgplot
from .curves import PlotCurveLine, PlotCurveSymbols, PlotHistogram, TaoCurveSettings
from .fields import ElementField
from .patches import (
    PlotPatch,
    PlotPatchArc,
    PlotPatchCircle,
    PlotPatchEllipse,
    PlotPatchPolygon,
    PlotPatchRectangle,
    PlotPatchSbend,
)
from .plot import (
    AnyGraph,
    BasicGraph,
    FloorPlanGraph,
    GraphManager,
    LatticeLayoutGraph,
    PlotAnnotation,
    PlotCurve,
    UnsupportedGraphError,
)
from .settings import TaoGraphSettings
from .types import Limit, OptionalLimit, Point
from .util import fix_grid_limits

logger = logging.getLogger(__name__)


class _Defaults:
    layout_height: float = 0.5
    colormap: str = "PRGn_r"


def set_defaults(
    layout_height: Optional[float] = None,
    colormap: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
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

    info = {key: value for key, value in vars(_Defaults).items() if not key.startswith("_")}
    info["figsize"] = matplotlib.rcParams["figure.figsize"]
    info["dpi"] = matplotlib.rcParams["figure.dpi"]
    return info


def setup_matplotlib_ticks(
    graph: AnyGraph,
    ax: matplotlib.axes.Axes,
    user_xlim: Optional[Limit],
    user_ylim: Optional[Limit],
) -> None:
    if user_xlim is None:
        _setup_matplotlib_xticks(graph, ax)
    else:
        ax.set_xlim(user_xlim)

    if user_ylim is None:
        _setup_matplotlib_yticks(graph, ax)
    else:
        ax.set_ylim(user_ylim)


def _fix_limits(lim: Point, pad_factor: float = 0.0) -> Point:
    low, high = lim
    if np.isclose(low, 0.0) and np.isclose(high, 0.0):
        # TODO: matplotlib can sometimes get in a bad spot trying to plot empty data
        # with very small limits
        return (-0.001, 0.001)
    return (low - abs(low * pad_factor), high + abs(high * pad_factor))


def _setup_matplotlib_xticks(graph: AnyGraph, ax: matplotlib.axes.Axes):
    """Configure ticks on the provided matplotlib x-axis."""
    ax.set_xlim(_fix_limits(graph.xlim))

    xlim = ax.get_xlim()
    if graph.info["x_minor_div"] > 0:
        ax.xaxis.set_minor_locator(
            matplotlib.ticker.AutoMinorLocator(graph.info["x_minor_div"])
        )
        ax.tick_params(axis="x", which="minor", length=4, color="black")

    if graph.info["x_major_div_nominal"] > 2:
        ticks = np.linspace(*xlim, graph.info["x_major_div_nominal"])
        ax.set_xticks(ticks)


def _setup_matplotlib_yticks(graph: AnyGraph, ax: matplotlib.axes.Axes):
    """Configure ticks on the provided matplotlib y-axis."""
    ax.set_ylim(_fix_limits(graph.ylim))
    ylim = ax.get_ylim()
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.tick_params(axis="y", which="minor", length=4, color="black")
    if graph.info["y_major_div_nominal"] > 2:
        ax.set_yticks(np.linspace(*ylim, graph.info["y_major_div_nominal"]))


def setup_matplotlib_axis(graph: AnyGraph, ax: matplotlib.axes.Axes):
    """Configure limits, title, and basic info for the given axes."""
    if not graph.show_axes:
        ax.set_axis_off()

    ax.set_title(pgplot.mpl_string(graph.title))
    ax.set_xlabel(pgplot.mpl_string(graph.xlabel))
    ax.set_ylabel(pgplot.mpl_string(graph.ylabel))
    ax.set_axisbelow(True)

    if graph.draw_grid:
        ax.grid(graph.draw_grid, which="major", axis="both")


def get_figsize(
    figsize: Optional[Tuple[float, float]] = None,
    width: Optional[float] = None,
    height: Optional[float] = None,
):
    if figsize is not None:
        return figsize

    if width or height:
        return (
            width or plt.rcParams["figure.figsize"][0],
            height or plt.rcParams["figure.figsize"][1],
        )
    return None


def plot_annotation(annotation: PlotAnnotation, ax: matplotlib.axes.Axes):
    return ax.annotate(
        xy=(annotation.x, annotation.y),
        text=pgplot.mpl_string(annotation.text),
        horizontalalignment=annotation.horizontalalignment,
        verticalalignment=annotation.verticalalignment,
        clip_on=annotation.clip_on,
        color=pgplot.mpl_color(annotation.color),
        rotation=annotation.rotation,
        rotation_mode=annotation.rotation_mode,
        fontsize=8,
    )


def plot_curve_line(
    curve: PlotCurveLine,
    ax: matplotlib.axes.Axes,
    label: Optional[str] = None,
):
    return ax.plot(
        curve.xs,
        curve.ys,
        color=pgplot.mpl_color(curve.color or "black"),
        linestyle=curve.linestyle,
        linewidth=curve.linewidth,
        label=label,
    )


def plot_curve_symbols(
    curve: PlotCurveSymbols,
    ax: matplotlib.axes.Axes,
    label: Optional[str] = None,
):
    return ax.plot(
        curve.xs,
        curve.ys,
        color=pgplot.mpl_color(curve.color),
        markerfacecolor=curve.markerfacecolor,
        markersize=curve.markersize,
        marker=pgplot.symbols.get(curve.marker, "."),
        markeredgewidth=curve.markeredgewidth,
        linewidth=curve.linewidth,
        label=label,
    )


def plot_histogram(
    hist: PlotHistogram,
    ax: matplotlib.axes.Axes,
):
    return ax.hist(
        hist.xs,
        bins=hist.bins,
        weights=hist.weights,
        histtype=hist.histtype,
        color=pgplot.mpl_color(hist.color),
    )


def plot_curve(curve: PlotCurve, ax: matplotlib.axes.Axes):
    res = []
    if curve.line is not None:
        res.append(
            plot_curve_line(
                curve.line,
                ax,
                label=pgplot.mpl_string(curve.legend_label),
            )
        )
    if curve.symbol is not None:
        res.append(
            plot_curve_symbols(
                curve.symbol,
                ax,
                label=pgplot.mpl_string(curve.legend_label) if curve.line is None else None,
            )
        )
    if curve.histogram is not None:
        res.append(plot_histogram(curve.histogram, ax))
    for patch in curve.patches or []:
        res.append(plot_patch(patch, ax))
    return res


def patch_to_mpl(patch: PlotPatch):
    if isinstance(patch, PlotPatchRectangle):
        return matplotlib.patches.Rectangle(
            xy=patch.xy,
            width=patch.width,
            height=patch.height,
            angle=patch.angle,
            rotation_point=patch.rotation_point,
            **patch._patch_args,
        )
    if isinstance(patch, PlotPatchArc):
        return matplotlib.patches.Arc(
            xy=patch.xy,
            width=patch.width,
            height=patch.height,
            angle=patch.angle,
            theta1=patch.theta1,
            theta2=patch.theta2,
            **patch._patch_args,
        )
    if isinstance(patch, PlotPatchCircle):
        return matplotlib.patches.Circle(
            xy=patch.xy,
            radius=patch.radius,
            **patch._patch_args,
        )
    if isinstance(patch, PlotPatchPolygon):
        return matplotlib.patches.Polygon(
            xy=patch.vertices,
            **patch._patch_args,
        )

    if isinstance(patch, PlotPatchEllipse):
        return matplotlib.patches.Ellipse(
            xy=patch.xy,
            width=patch.width,
            height=patch.height,
            angle=patch.angle,
            **patch._patch_args,
        )
    if isinstance(patch, PlotPatchSbend):
        codes = [
            matplotlib.path.Path.MOVETO,
            matplotlib.path.Path.CURVE3,
            matplotlib.path.Path.CURVE3,
            matplotlib.path.Path.LINETO,
            matplotlib.path.Path.CURVE3,
            matplotlib.path.Path.CURVE3,
            matplotlib.path.Path.CLOSEPOLY,
        ]
        vertices = [
            patch.spline1[0],
            patch.spline1[1],
            patch.spline1[2],
            patch.spline2[0],
            patch.spline2[1],
            patch.spline2[2],
            patch.spline1[0],
        ]
        return matplotlib.patches.PathPatch(
            matplotlib.path.Path(vertices, codes),
            facecolor="green",
            alpha=0.5,
        )

    raise NotImplementedError(f"Unsupported patch type: {type(patch).__name__}")


def plot_patch(patch: PlotPatch, ax: matplotlib.axes.Axes):
    mpl = patch_to_mpl(patch)
    ax.add_patch(mpl)
    return mpl


def plot_layout_shape(shape: layout_shapes.AnyLayoutShape, ax: matplotlib.axes.Axes):
    if isinstance(shape, layout_shapes.LayoutWrappedShape):
        ax.add_collection(
            matplotlib.collections.LineCollection(
                [[(x, y) for x, y in zip(line[0], line[1])] for line in shape.lines],
                colors=pgplot.mpl_color(shape.color),
                linewidths=shape.line_width,
            )
        )
    else:
        lines = shape.lines
        if lines:
            ax.add_collection(
                matplotlib.collections.LineCollection(
                    lines,
                    colors=pgplot.mpl_color(shape.color),
                    linewidths=shape.line_width,
                )
            )
        for patch in shape.to_patches():
            plot_patch(patch, ax)


def plot_floor_plan_shape(shape: floor_plan_shapes.Shape, ax: matplotlib.axes.Axes):
    for line in shape.to_lines():
        plot_curve_line(line, ax)
    if not isinstance(shape, floor_plan_shapes.Box):
        for patch in shape.to_patches():
            plot_patch(patch, ax)


def plot(graph: AnyGraph, ax: Optional[matplotlib.axes.Axes] = None) -> matplotlib.axes.Axes:
    if ax is None:
        _, ax = plt.subplots()

    assert ax is not None

    if isinstance(graph, BasicGraph):
        for curve in graph.curves:
            assert not curve.info["use_y2"], "TODO: y2 support"
            plot_curve(curve, ax)

        if graph.draw_legend and any(curve.legend_label for curve in graph.curves):
            ax.legend()

    elif isinstance(graph, LatticeLayoutGraph):
        ax.axhline(y=0, color="Black", linewidth=1)

        for elem in graph.elements:
            if elem.shape is not None:
                plot_layout_shape(elem.shape, ax)
            # ax.add_collection(
            #     matplotlib.collections.LineCollection(
            #         elem.lines,
            #         colors=pgplot.mpl_color(elem.color),
            #         linewidths=elem.width,
            #     )
            # )
            # for patch in elem.patches:
            #     plot_patch(patch, ax)
            for annotation in elem.annotations:
                plot_annotation(annotation, ax)

        # Invisible line to give the lat layout enough vertical space.
        # Without this, the tops and bottoms of shapes could be cut off
        # ax.plot([0, 0], [-1.7 * self.y_max, 1.3 * self.y_max], alpha=0)
        ax.yaxis.set_visible(False)

        # ax.set_xticks([elem.info["ele_s_start"] for elem in self.elements])
        # ax.set_xticklabels([elem.info["label_name"] for elem in self.elements], rotation=90)
        ax.grid(visible=False)
    elif isinstance(graph, FloorPlanGraph):
        ax.set_aspect("equal")
        for elem in graph.elements:
            if elem.shape is not None:
                plot_floor_plan_shape(elem.shape, ax)
            for annotation in elem.annotations:
                plot_annotation(annotation, ax)

        for line in graph.building_walls.lines:
            plot_curve_line(line, ax)
        for patch in graph.building_walls.patches:
            plot_patch(patch, ax)
        if graph.floor_orbits is not None:
            plot_curve_symbols(graph.floor_orbits.curve, ax)
    else:
        raise NotImplementedError(f"Unsupported graph for matplotlib: {type(graph)}")

    setup_matplotlib_axis(graph, ax)
    return ax


class MatplotlibGraphManager(GraphManager):
    """Matplotlib backend graph manager."""

    _key_: ClassVar[str] = "mpl"

    @functools.wraps(set_defaults)
    def configure(self, **kwargs):
        return set_defaults(**kwargs)

    def plot_grid(
        self,
        templates: List[str],
        grid: Tuple[int, int],
        *,
        include_layout: bool = False,
        figsize: Optional[Tuple[float, float]] = None,
        tight_layout: bool = True,
        share_x: Union[bool, Literal["row", "col", "all"]] = "col",
        layout_height: Optional[float] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        xlim: Union[OptionalLimit, Sequence[OptionalLimit]] = None,
        ylim: Union[OptionalLimit, Sequence[OptionalLimit]] = None,
        curves: Optional[List[Dict[int, TaoCurveSettings]]] = None,
        settings: Optional[List[TaoGraphSettings]] = None,
        save: Union[bool, str, pathlib.Path, None] = None,
        axes: Optional[List[List[matplotlib.axes.Axes]]] = None,
    ):
        """
        Plot graphs on a grid with Matplotlib.

        Parameters
        ----------
        templates : list of str
            Graph template names.
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
        share_x : bool, "row", "col", "all", default="col"
            Share all x-axes (`True` or "all"), share x-axes in rows ("row") or
            in columns ("col").
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
            template_names=templates,
            curves=curves,
            settings=settings,
            xlim=xlim,
            ylim=ylim,
        )
        nrows, ncols = grid
        height_ratios = None

        figsize = get_figsize(figsize, width, height)

        if include_layout:
            layout_height = layout_height or _Defaults.layout_height
            empty_graph_count = nrows * ncols - len(templates)
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
            axes = [list(gs[row, :]) for row in range(nrows)]
            for row in axes:
                for ax in row:
                    ax.set_axis_off()

        all_xlim = fix_grid_limits(xlim, num_graphs=len(graphs))
        all_ylim = fix_grid_limits(ylim, num_graphs=len(graphs))

        rows_cols = [(row, col) for row in range(nrows) for col in range(ncols)]

        for graph, xl, yl, (row, col) in zip(graphs, all_xlim, all_ylim, rows_cols):
            ax = axes[row][col]
            try:
                plot(graph, ax)
            except UnsupportedGraphError:
                continue

            ax.set_axis_on()
            setup_matplotlib_ticks(graph, ax, user_xlim=xl, user_ylim=yl)

        if include_layout:
            layout_graph = self.lattice_layout_graph
            for col in range(ncols):
                ax = axes[-1][col]
                plot(layout_graph, ax)
                ax.set_axis_on()

                xl = None
                if share_x in {"all", "col", True} and nrows > 1:
                    try:
                        xl = axes[0][col].get_xlim()
                    except IndexError:
                        pass

                setup_matplotlib_ticks(layout_graph, ax, user_xlim=xl, user_ylim=None)

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
        template: str,
        *,
        region_name: Optional[str] = None,
        include_layout: bool = True,
        tight_layout: bool = True,
        width: Optional[float] = None,
        height: Optional[float] = None,
        layout_height: Optional[float] = None,
        figsize: Optional[Tuple[float, float]] = None,
        share_x: bool = True,
        xlim: Optional[Limit] = None,
        ylim: Optional[Limit] = None,
        save: Union[bool, str, pathlib.Path, None] = None,
        settings: Optional[TaoGraphSettings] = None,
        curves: Optional[Dict[int, TaoCurveSettings]] = None,
        axes: Optional[List[matplotlib.axes.Axes]] = None,
    ):
        """
        Plot a graph with Matplotlib.

        Parameters
        ----------
        template : str
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
        share_x : bool, default=True
            Share x-axes for all plots.
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
            template_name=template,
            region_name=region_name,
            curves=curves,
            settings=settings,
            xlim=xlim,
            ylim=ylim,
        )
        if not graphs:
            raise UnsupportedGraphError(f"No supported plots from this template: {template}")

        figsize = get_figsize(figsize, width, height)

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
            axes = list(gs[:, 0])
            assert axes is not None

        if include_layout:
            layout_graph = self.lattice_layout_graph

        for ax, graph in zip(axes, graphs):
            try:
                plot(graph, ax)
            except UnsupportedGraphError:
                continue

            if isinstance(graph, LatticeLayoutGraph) and len(graphs) > 1:
                # Do not set ylimits if the user specifically requested a layout graph
                yl = None
            else:
                yl = ylim

            setup_matplotlib_ticks(graph, ax, user_xlim=xlim, user_ylim=yl)

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

    def plot_field(
        self,
        ele_id: str,
        *,
        colormap: Optional[str] = None,
        radius: float = 0.015,
        num_points: int = 100,
        figsize: Optional[Tuple[float, float]] = None,
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

        figsize = get_figsize(figsize, width, height)

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
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
