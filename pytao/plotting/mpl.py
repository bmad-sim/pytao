from __future__ import annotations

import functools
import logging
import pathlib
import time
from collections.abc import Sequence
from typing import ClassVar, Literal

import matplotlib.axes
import matplotlib.axis
import matplotlib.cm
import matplotlib.collections
import matplotlib.legend
import matplotlib.lines
import matplotlib.patches
import matplotlib.path
import matplotlib.pyplot as plt
import matplotlib.text
import matplotlib.ticker
import numpy as np

from . import floor_plan_shapes, layout_shapes, pgplot
from .curves import PlotCurveLine, PlotCurveSymbols, PlotHistogram, TaoCurveSettings
from .ele_methods import ElementMethodsPlotData, color_for_value, is_garbage_value
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
    line_width_scale: float = 0.5
    floor_line_width_scale: float = 0.5
    colormap: str = "PRGn_r"


def set_defaults(
    layout_height: float | None = None,
    colormap: str | None = None,
    line_width_scale: float | None = None,
    floor_line_width_scale: float | None = None,
    figsize: tuple[float, float] | None = None,
    width: float | None = None,
    height: float | None = None,
    dpi: int | None = None,
):
    """
    Set default values for Matplotlib plot settings.

    Parameters
    ----------
    layout_height : float, optional
        Height of the layout. Default is 0.5.
    colormap : str, optional
        Colormap to use for plotting. Default is "PRGn_r".
    line_width_scale : float, optional
        Scale factor for line widths, excluding floor plan lines. Default is 0.5.
    floor_line_width_scale : float, optional
        Scale factor for floor plan line widths. Default is 0.5.
    figsize : tuple of float, optional
        Size of the figure (width, height). Default is as-configured in matplotlib rcParams.
    width : float, optional
        Width of the figure in inches. Default is as-configured in matplotlib rcParams.
    height : float, optional
        Height of the figure in inches. Default is as-configured in matplotlib rcParams.
    dpi : int, optional
        Dots per inch for the figure. Default is as-configured in matplotlib rcParams.
    """

    if layout_height is not None:
        _Defaults.layout_height = layout_height
    if colormap is not None:
        _Defaults.colormap = colormap
    if line_width_scale is not None:
        _Defaults.line_width_scale = line_width_scale
    if floor_line_width_scale is not None:
        _Defaults.floor_line_width_scale = floor_line_width_scale
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
    user_xlim: Limit | None,
    user_ylim: Limit | None,
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
    figsize: tuple[float, float] | None = None,
    width: float | None = None,
    height: float | None = None,
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
    label: str | None = None,
    line_width_scale: float = 1.0,
):
    return ax.plot(
        curve.xs,
        curve.ys,
        color=pgplot.mpl_color(curve.color or "black"),
        linestyle=curve.linestyle,
        linewidth=curve.linewidth * line_width_scale,
        label=label,
    )


def plot_curve_symbols(
    curve: PlotCurveSymbols,
    ax: matplotlib.axes.Axes,
    label: str | None = None,
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


def plot_curve(curve: PlotCurve, ax: matplotlib.axes.Axes, line_width_scale: float = 1.0):
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
        res.append(plot_patch(patch, ax, line_width_scale=line_width_scale))
    return res


def patch_to_mpl(patch: PlotPatch, line_width_scale: float = 1.0):
    patch_args = patch._patch_args
    if patch_args["linewidth"] is not None:
        patch_args["linewidth"] *= line_width_scale

    if isinstance(patch, PlotPatchRectangle):
        return matplotlib.patches.Rectangle(
            xy=patch.xy,
            width=patch.width,
            height=patch.height,
            angle=patch.angle,
            rotation_point=patch.rotation_point,
            **patch_args,
        )
    if isinstance(patch, PlotPatchArc):
        return matplotlib.patches.Arc(
            xy=patch.xy,
            width=patch.width,
            height=patch.height,
            angle=patch.angle,
            theta1=patch.theta1,
            theta2=patch.theta2,
            **patch_args,
        )
    if isinstance(patch, PlotPatchCircle):
        return matplotlib.patches.Circle(
            xy=patch.xy,
            radius=patch.radius,
            **patch_args,
        )
    if isinstance(patch, PlotPatchPolygon):
        return matplotlib.patches.Polygon(
            xy=patch.vertices,
            **patch_args,
        )

    if isinstance(patch, PlotPatchEllipse):
        return matplotlib.patches.Ellipse(
            xy=patch.xy,
            width=patch.width,
            height=patch.height,
            angle=patch.angle,
            **patch_args,
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
            # facecolor="green",
            # alpha=0.5,
            **patch_args,
        )

    raise NotImplementedError(f"Unsupported patch type: {type(patch).__name__}")


def plot_patch(patch: PlotPatch, ax: matplotlib.axes.Axes, line_width_scale: float = 1.0):
    mpl = patch_to_mpl(patch, line_width_scale=line_width_scale)
    ax.add_patch(mpl)
    return mpl


def plot_layout_shape(
    shape: layout_shapes.AnyLayoutShape,
    ax: matplotlib.axes.Axes,
    line_width_scale: float | None = None,
):
    if line_width_scale is None:
        line_width_scale = _Defaults.line_width_scale

    if isinstance(shape, layout_shapes.LayoutWrappedShape):
        ax.add_collection(
            matplotlib.collections.LineCollection(
                [[(x, y) for x, y in zip(line[0], line[1])] for line in shape.lines],
                colors=pgplot.mpl_color(shape.color),
                linewidths=shape.line_width * line_width_scale,
            )
        )
    else:
        lines = shape.lines
        if lines:
            ax.add_collection(
                matplotlib.collections.LineCollection(
                    lines,
                    colors=pgplot.mpl_color(shape.color),
                    linewidths=shape.line_width * line_width_scale,
                )
            )
        for patch in shape.to_patches():
            plot_patch(patch, ax, line_width_scale=line_width_scale)


def plot_floor_plan_shape(
    shape: floor_plan_shapes.Shape,
    ax: matplotlib.axes.Axes,
    line_width_scale: float | None = None,
):
    if line_width_scale is None:
        line_width_scale = _Defaults.floor_line_width_scale

    for line in shape.to_lines():
        plot_curve_line(line, ax, line_width_scale=line_width_scale)
    if not isinstance(shape, floor_plan_shapes.Box):
        for patch in shape.to_patches():
            plot_patch(patch, ax, line_width_scale=line_width_scale)


def plot(graph: AnyGraph, ax: matplotlib.axes.Axes | None = None) -> matplotlib.axes.Axes:
    if ax is None:
        _, ax = plt.subplots()

    assert ax is not None

    if isinstance(graph, BasicGraph):
        for curve in graph.curves:
            if curve.info["use_y2"]:
                raise NotImplementedError("y2 support")

            plot_curve(curve, ax, line_width_scale=_Defaults.line_width_scale)

        if graph.draw_legend and any(curve.legend_label for curve in graph.curves):
            ax.legend()

    elif isinstance(graph, LatticeLayoutGraph):
        ax.axhline(y=0, color="Black", linewidth=1)

        for elem in graph.elements:
            if elem.shape is not None:
                plot_layout_shape(elem.shape, ax, line_width_scale=_Defaults.line_width_scale)
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
                plot_floor_plan_shape(
                    elem.shape,
                    ax,
                    line_width_scale=_Defaults.floor_line_width_scale,
                )
            for annotation in elem.annotations:
                plot_annotation(annotation, ax)

        for line in graph.building_walls.lines:
            plot_curve_line(line, ax, line_width_scale=_Defaults.floor_line_width_scale)
        for patch in graph.building_walls.patches:
            plot_patch(patch, ax, line_width_scale=_Defaults.floor_line_width_scale)
        if graph.floor_orbits is not None:
            plot_curve_symbols(graph.floor_orbits.curve, ax)
    else:
        raise NotImplementedError(f"Unsupported graph for matplotlib: {type(graph)}")

    setup_matplotlib_axis(graph, ax)
    return ax


def _draw_method_lanes(
    data: ElementMethodsPlotData,
    columns: list[str],
    ax: matplotlib.axes.Axes,
    lane_height: float,
) -> None:
    """
    Draw one horizontal lane of colored blocks per method column.

    Contiguous elements with the same value are merged into a single block;
    per-element blocks would show antialiasing seams as vertical stripes on
    large lattices.
    """
    for lane, col in enumerate(columns):
        runs = data.value_runs(col)
        for value in sorted({value for _, _, value in runs}):
            color = color_for_value(value)
            garbage = is_garbage_value(value)
            spans = [
                (data.s_start[first], data.s_end[last] - data.s_start[first])
                for first, last, run_value in runs
                if run_value == value and data.s_end[last] > data.s_start[first]
            ]
            if spans:
                ax.broken_barh(
                    spans,
                    (lane - lane_height / 2, lane_height),
                    facecolors=color,
                    edgecolors="black" if garbage else "none",
                    hatch="///" if garbage else None,
                )
            zero_length = [
                data.s_start[first]
                for first, last, run_value in runs
                if run_value == value and data.s_end[last] <= data.s_start[first]
            ]
            if zero_length:
                ax.vlines(
                    zero_length,
                    lane - lane_height / 2,
                    lane + lane_height / 2,
                    colors=color,
                    linewidths=1.0,
                )

    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels(columns)
    ax.set_ylim(len(columns) - 0.5, -0.5)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)


def _draw_method_transition_names(
    data: ElementMethodsPlotData,
    columns: list[str],
    ax: matplotlib.axes.Axes,
) -> None:
    """Mark per-lane method transitions and label the elements on either side."""

    def draw_name(name: str, value: str, side: int):
        color = color_for_value(value)
        if color in ("white", "#00000000"):
            # I haven't settled on the color scheme yet
            color = "black"
        ax.annotate(
            name,
            xy=(boundary, lane),
            xytext=(3 * side, 0),
            textcoords="offset points",
            rotation=90,
            rotation_mode="anchor",
            ha="center",
            va="bottom" if side < 0 else "top",
            fontsize="x-small",
            color=color,
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
            },
            clip_on=True,
            zorder=4,
        )

    for lane, col in enumerate(columns):
        for idx, before, after in data.value_transitions(col):
            boundary = data.s_end[idx]
            ax.vlines(
                boundary,
                lane - 0.5,
                lane + 0.5,
                colors="black",
                linewidths=0.75,
                zorder=3,
            )
            draw_name(data.names[idx], before, -1)
            draw_name(data.names[idx + 1], after, 1)


def _draw_lane_legends(
    data: ElementMethodsPlotData,
    columns: list[str],
    ax: matplotlib.axes.Axes,
) -> None:
    """
    Add a small legend of each lane's values next to that lane.

    Related global settings are attached to their lanes: the space charge mesh
    to `space_charge_method`, and the CSR mesh/binning to `csr_method`.
    """
    lane_info = data.settings_summary

    for lane, col in enumerate(columns):
        handles: list[matplotlib.patches.Patch | matplotlib.lines.Line2D] = [
            matplotlib.patches.Patch(
                facecolor=color_for_value(value),
                edgecolor="black" if is_garbage_value(value) else "none",
                hatch="///" if is_garbage_value(value) else None,
                label=value,
            )
            for value in sorted({v for v in data.methods[col] if v is not None})
        ]
        handles.extend(
            matplotlib.lines.Line2D([], [], linestyle="none", label=line)
            for line in lane_info.get(col, [])
        )
        if not handles:
            continue

        legend = matplotlib.legend.Legend(
            ax,
            handles,
            [handle.get_label() for handle in handles],
            loc="center left",
            # Anchor at the lane's vertical center, just right of the axes.
            bbox_to_anchor=(1.01, (len(columns) - 0.5 - lane) / len(columns)),
            frameon=False,
            fontsize="x-small",
            ncols=2,
            handlelength=1.0,
            handletextpad=0.4,
            columnspacing=0.8,
            borderaxespad=0.0,
        )
        ax.add_artist(legend)
        # add_artist clips to the axes patch, which would hide the legend and
        # exclude it from tight-bbox calculations.
        legend.set_clip_on(False)


def _draw_csr_ds_step(data: ElementMethodsPlotData, ax: matplotlib.axes.Axes) -> None:
    """Draw `csr_ds_step` vs s, colored by each element's `csr_method`."""
    segments = data.csr_ds_step_segments()
    if segments:
        s_start, s_end, steps, csr_methods = zip(*segments)
        colors = [
            color_for_value(value) if value is not None else "#888888" for value in csr_methods
        ]
        ax.hlines(steps, s_start, s_end, colors=colors, linewidths=2.0)
    ax.set_ylabel("csr_ds_step [m]")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)


class MatplotlibGraphManager(GraphManager):
    """Matplotlib backend graph manager."""

    _key_: ClassVar[str] = "mpl"

    @functools.wraps(set_defaults)
    def configure(self, **kwargs):
        return set_defaults(**kwargs)

    def plot_grid(
        self,
        templates: list[str],
        grid: tuple[int, int],
        *,
        include_layout: bool = False,
        figsize: tuple[float, float] | None = None,
        tight_layout: bool = True,
        share_x: bool | Literal["row", "col", "all"] = "col",
        layout_height: float | None = None,
        width: float | None = None,
        height: float | None = None,
        xlim: OptionalLimit | Sequence[OptionalLimit] = None,
        ylim: OptionalLimit | Sequence[OptionalLimit] = None,
        curves: list[dict[int, TaoCurveSettings]] | None = None,
        settings: list[TaoGraphSettings] | None = None,
        save: bool | str | pathlib.Path | None = None,
        axes: list[list[matplotlib.axes.Axes]] | None = None,
        ix_uni: int | None = None,
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
        ix_uni : int, optional
            Plot data from this universe for every graph in the grid.

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
            ix_uni=ix_uni,
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
            layout_graph = self.get_lattice_layout_graph(ix_uni=ix_uni)
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
        region_name: str | None = None,
        include_layout: bool = True,
        tight_layout: bool = True,
        width: float | None = None,
        height: float | None = None,
        layout_height: float | None = None,
        figsize: tuple[float, float] | None = None,
        share_x: bool = True,
        xlim: Limit | None = None,
        ylim: Limit | None = None,
        save: bool | str | pathlib.Path | None = None,
        settings: TaoGraphSettings | None = None,
        curves: dict[int, TaoCurveSettings] | None = None,
        axes: list[matplotlib.axes.Axes] | None = None,
        ix_uni: int | None = None,
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
        ix_uni : int, optional
            Plot data from this universe.

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
            ix_uni=ix_uni,
        )
        if not graphs:
            raise UnsupportedGraphError(f"No supported plots from this template: {template}")

        figsize = get_figsize(figsize, width, height)

        if (
            include_layout
            and not any(isinstance(graph, LatticeLayoutGraph) for graph in graphs)
            and any(graph.is_s_plot for graph in graphs)
        ):
            graphs = [
                *graphs,
                self.get_lattice_layout_graph(ix_uni=ix_uni),
            ]
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
        colormap: str | None = None,
        radius: float = 0.015,
        num_points: int = 100,
        figsize: tuple[float, float] | None = None,
        width: int = 4,
        height: int = 4,
        x_scale: float = 1e3,
        ax: matplotlib.axes.Axes | None = None,
        save: bool | str | pathlib.Path | None = None,
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

    def plot_ele_methods(
        self,
        data: ElementMethodsPlotData,
        *,
        columns: Sequence[str] | None = None,
        show_names: bool = True,
        show_csr_ds_step: bool | None = None,
        include_layout: bool = True,
        lane_height: float = 0.8,
        figsize: tuple[float, float] | None = None,
        width: float | None = None,
        height: float | None = None,
        layout_height: float | None = None,
        ax: matplotlib.axes.Axes | None = None,
        save: bool | str | pathlib.Path | None = None,
    ):
        """
        Plot element method settings as categorical lanes along the beamline.

        Each method (e.g., `tracking_method`) becomes a horizontal lane, with
        each element drawn as a block spanning its longitudinal extent, colored
        by the method value.

        Parameters
        ----------
        data : ElementMethodsPlotData
            Per-element method data, gathered via
            `ElementMethodsPlotData.from_tao`.
        columns : sequence of str, optional
            Method columns to plot, in order.  Defaults to all categorical
            columns with data for the selected elements.
        show_names : bool, default=True
            Label method transitions with the element names before and after
            the transition point.
        show_csr_ds_step : bool, optional
            Add a subplot of `csr_ds_step` vs s.  The default (`None`) shows
            it only when CSR is active for at least one selected element.
        include_layout : bool, default=True
            Include a lattice layout plot at the bottom.
        lane_height : float, default=0.8
            Height of each lane's blocks, where lanes are spaced 1.0 apart.
        figsize : (float, float), optional
            Figure size.  Takes precedence over `width` and `height`.
        width : float, optional
            Width of the whole plot.
        height : float, optional
            Height of the whole plot.
        layout_height : float, optional
            Normalized height of the layout plot - assuming the lane plot is
            of height 1.  Defaults to about half the height of a single lane
            (`0.5 / len(columns)`).
        ax : matplotlib.axes.Axes, optional
            The axes to place the lanes in.  Only supported with
            `include_layout=False` and `show_csr_ds_step=False`.
        save : pathlib.Path or str, optional
            Save the plot to the given filename.

        Returns
        -------
        ElementMethodsPlotData
        matplotlib.figure.Figure
        list of matplotlib.axes.Axes
        """
        columns = data.validate_columns(columns)

        if show_csr_ds_step is None:
            show_csr_ds_step = data.csr_on

        nrows = 1 + int(show_csr_ds_step) + int(include_layout)
        if ax is not None:
            if nrows > 1:
                raise ValueError(
                    "A user-specified axis is only supported with "
                    "include_layout=False and show_csr_ds_step=False"
                )
            fig = ax.figure
            axes = [ax]
        else:
            layout_height = layout_height or 0.5 / len(columns)

            if figsize is None and width is None and height is None:
                lanes_inches = max(2.0, 0.45 * len(columns) + 1.2)
                figsize = (
                    12.0,
                    lanes_inches
                    + (1.5 if show_csr_ds_step else 0.0)
                    + (lanes_inches * layout_height if include_layout else 0.0),
                )
            else:
                figsize = get_figsize(figsize, width, height)

            height_ratios = [1.0]
            if show_csr_ds_step:
                height_ratios.append(0.4)
            if include_layout:
                height_ratios.append(layout_height)

            fig, gs = plt.subplots(
                nrows=nrows,
                ncols=1,
                sharex=True,
                height_ratios=height_ratios,
                figsize=figsize,
                squeeze=False,
            )
            axes = list(gs[:, 0])

        lanes_ax = axes[0]
        _draw_method_lanes(data, columns, lanes_ax, lane_height)
        if show_names:
            _draw_method_transition_names(data, columns, lanes_ax)
        _draw_lane_legends(data, columns, lanes_ax)

        if show_csr_ds_step:
            _draw_csr_ds_step(data, axes[1])

        if include_layout:
            plot(self.lattice_layout_graph, ax=axes[-1])

        lanes_ax.set_xlim(min(data.s_start), max(data.s_end))
        axes[-1].set_xlabel("s [m]")

        if fig is not None:
            if ax is None:
                fig.tight_layout()

            if save:
                if save is True:
                    save = "ele_methods.png"
                if not pathlib.Path(save).suffix:
                    save = f"{save}.png"
                logger.info(f"Saving plot to {save!r}")
                fig.savefig(save, bbox_inches="tight")

        return data, fig, axes
