from __future__ import annotations

import functools
import logging
import math
import pathlib
import typing
from abc import ABC, abstractmethod
from typing import (
    Any,
    ClassVar,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import plotly.graph_objects as go
import plotly.subplots
from plotly.graph_objs import Figure

try:
    import plotly.io
    import plotly.offline as pyo

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import ipywidgets as widgets
    from IPython.display import display

    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False

from ..interface_commands import AnyPath
from ..tao_ctypes.core import TaoCommandError
from .curves import CurveIndexToCurve, TaoCurveSettings
from .fields import ElementField
from .plot import AnyGraph
from .settings import TaoGraphSettings
from .types import FloatVariableInfo
from .util import Limit, OptionalLimit

if typing.TYPE_CHECKING:
    from plotly.graph_objs import Figure

    from .. import Tao


from . import floor_plan_shapes, pgplot, util
from .curves import PlotCurveLine, PlotCurveSymbols
from .layout_shapes import LayoutShape
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
    BasicGraph,
    FloorPlanGraph,
    GraphBase,
    GraphManager,
    LatticeLayoutGraph,
    PlotCurve,
    UnsupportedGraphError,
)

if typing.TYPE_CHECKING:
    from .. import Tao

logger = logging.getLogger(__name__)


def plotly_color(color: str | None) -> str:
    """Convert color name to Plotly-compatible format."""
    color_map = {
        "red": "#FF0000",
        "blue": "#0000FF",
        "green": "#008000",
        "black": "#000000",
        "white": "#FFFFFF",
        "yellow": "#FFFF00",
        "cyan": "#00FFFF",
        "magenta": "#FF00FF",
        "orange": "#FFA500",
        "purple": "#800080",
        "brown": "#A52A2A",
        "pink": "#FFC0CB",
        "gray": "#808080",
        "grey": "#808080",
    }
    color = (color or "black").lower().replace("_", "")
    return color_map.get(color, color)


class _PlotlyDefaults:
    """
    Defaults used for Plotly plots internally.

    To change these values, use `set_plotly_defaults`.
    """

    width: int = 800
    height: int = 800
    stacked_height: int = 200
    layout_height: int = 100
    colorscale: str = "Viridis"
    show_toolbar: bool = True
    grid_spacing: float = 0.02
    limit_scale_factor: float = 1.01
    max_data_points: int = 10_000
    variables_per_row: int = 2
    show_sliders: bool = True
    line_width_scale: float = 0.5
    floor_line_width_scale: float = 0.5
    marker_size_scale: float = 1.0

    @classmethod
    def get_size_for_class(
        cls,
        typ: Type[AnyPlotlyGraph],
        user_width: Optional[int] = None,
        user_height: Optional[int] = None,
    ) -> Tuple[int, int]:
        default = {
            PlotlyBasicGraph: (cls.width, cls.height),
            PlotlyLatticeLayoutGraph: (cls.width, cls.layout_height),
            PlotlyFloorPlanGraph: (cls.width, cls.height),
        }[typ]
        return (user_width or default[0], user_height or default[1])


def set_plotly_defaults(
    width: Optional[int] = None,
    height: Optional[int] = None,
    stacked_height: Optional[int] = None,
    layout_height: Optional[int] = None,
    colorscale: Optional[str] = None,
    show_toolbar: Optional[bool] = None,
    grid_spacing: Optional[float] = None,
    limit_scale_factor: Optional[float] = None,
    max_data_points: Optional[int] = None,
    variables_per_row: Optional[int] = None,
    show_sliders: Optional[bool] = None,
    line_width_scale: Optional[float] = None,
    floor_line_width_scale: Optional[float] = None,
    marker_size_scale: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Change defaults used for Plotly plots.

    Parameters
    ----------
    width : int, optional
        Plot default width.
    height : int, optional
        Plot default height.
    stacked_height : int, optional
        Stacked plot default height (`plot_grid`)
    layout_height : int, optional
        Layout plot height.
    colorscale : str, optional
        Colorscale for `plot_field`.
    show_toolbar : bool, optional
        Show Plotly toolbar on each plot.
    grid_spacing : float, optional
        Spacing between subplots in grid layout.
    limit_scale_factor : float, default=1.01
        View limits from Tao are scaled by this factor.
    max_data_points : int, optional
        Maximum number of data points to show in the slider.
    variables_per_row : int, default=2
        Variables to list per row when in single mode.
    show_sliders : bool, default=True
        Show sliders alongside the spinners in single mode.
    line_width_scale : float, default=0.5
        Plot line width scaling factor applied to Tao's line width.
    floor_line_width_scale : float, default=0.5
        Floor plan line width scaling factor applied to Tao's line width.
    marker_size_scale : float, default=1.0
        Marker size scaling factor.
    """

    if width is not None:
        _PlotlyDefaults.width = int(width)
    if height is not None:
        _PlotlyDefaults.height = int(height)
    if stacked_height is not None:
        _PlotlyDefaults.stacked_height = int(stacked_height)
    if layout_height is not None:
        _PlotlyDefaults.layout_height = int(layout_height)
    if colorscale is not None:
        _PlotlyDefaults.colorscale = colorscale
    if show_toolbar is not None:
        _PlotlyDefaults.show_toolbar = bool(show_toolbar)
    if grid_spacing is not None:
        _PlotlyDefaults.grid_spacing = float(grid_spacing)
    if limit_scale_factor is not None:
        _PlotlyDefaults.limit_scale_factor = float(limit_scale_factor)
    if max_data_points is not None:
        _PlotlyDefaults.max_data_points = int(max_data_points)
    if variables_per_row is not None:
        _PlotlyDefaults.variables_per_row = int(variables_per_row)
    if show_sliders is not None:
        _PlotlyDefaults.show_sliders = bool(show_sliders)
    if line_width_scale is not None:
        _PlotlyDefaults.line_width_scale = float(line_width_scale)
    if floor_line_width_scale is not None:
        _PlotlyDefaults.floor_line_width_scale = float(floor_line_width_scale)
    if marker_size_scale is not None:
        _PlotlyDefaults.marker_size_scale = float(marker_size_scale)

    return {
        key: value
        for key, value in vars(_PlotlyDefaults).items()
        if not key.startswith("_") and key not in {"get_size_for_class"}
    }


def _plot_curve_symbols(
    fig: Figure,
    symbol: PlotCurveSymbols,
    name: str,
    row: Optional[int] = None,
    col: Optional[int] = None,
    showlegend: bool = True,
) -> None:
    """Add curve symbols to a Plotly figure."""
    marker_symbol = _get_plotly_marker(symbol.marker)

    fig.add_trace(
        go.Scatter(
            x=symbol.xs,
            y=symbol.ys,
            mode="markers",
            marker=dict(
                symbol=marker_symbol,
                size=symbol.markersize * _PlotlyDefaults.marker_size_scale,
                color=plotly_color(symbol.color),
            ),
            name=name,
            showlegend=showlegend,
        ),
        row=row,
        col=col,
    )


def _plot_curve_line(
    fig: Figure,
    line: PlotCurveLine,
    name: Optional[str] = None,
    row: Optional[int] = None,
    col: Optional[int] = None,
    line_width_scale: float = 1.0,
    showlegend: bool = True,
) -> None:
    """Add curve line to a Plotly figure."""
    fig.add_trace(
        go.Scatter(
            x=line.xs,
            y=line.ys,
            mode="lines",
            line=dict(
                width=line.linewidth * line_width_scale,
                color=plotly_color(line.color),
            ),
            name=name,
            showlegend=showlegend,
        ),
        row=row,
        col=col,
    )


def _plot_curve(
    fig: Figure,
    curve: PlotCurve,
    row: Optional[int] = None,
    col: Optional[int] = None,
    line_width_scale: float = 1.0,
) -> None:
    """Plot a complete curve (line and/or symbols) on a Plotly figure."""
    name = pgplot.mathjax_string(curve.info["name"])

    if curve.line is not None:
        _plot_curve_line(
            fig=fig,
            line=curve.line,
            name=name,
            row=row,
            col=col,
            line_width_scale=line_width_scale,
            showlegend=True,
        )

    if curve.symbol is not None:
        showlegend = curve.line is None  # Only show legend if no line
        _plot_curve_symbols(
            fig=fig,
            symbol=curve.symbol,
            name=name,
            row=row,
            col=col,
            showlegend=showlegend,
        )


def _get_plotly_marker(marker: str) -> str:
    """Convert pgplot marker to Plotly marker symbol."""
    marker_map = {
        "dot": "circle",
        "circle": "circle",
        "square": "square",
        "diamond": "diamond",
        "triangle": "triangle-up",
        "cross": "cross",
        "plus": "cross",
        "star": "star",
        "x": "x",
    }
    return marker_map.get(marker, "circle")


def _add_patch_to_figure(
    fig: Figure,
    patch: PlotPatch,
    row: Optional[int] = None,
    col: Optional[int] = None,
    line_width: Optional[float] = None,
) -> None:
    """Add a patch to a Plotly figure."""
    if isinstance(patch, PlotPatchRectangle):
        _add_rectangle_patch(fig, patch, row, col, line_width)
    elif isinstance(patch, PlotPatchCircle):
        _add_circle_patch(fig, patch, row, col, line_width)
    elif isinstance(patch, PlotPatchEllipse):
        _add_ellipse_patch(fig, patch, row, col, line_width)
    elif isinstance(patch, PlotPatchPolygon):
        _add_polygon_patch(fig, patch, row, col, line_width)
    elif isinstance(patch, PlotPatchArc):
        _add_arc_patch(fig, patch, row, col, line_width)
    elif isinstance(patch, PlotPatchSbend):
        _add_sbend_patch(fig, patch, row, col, line_width)
    else:
        logger.warning(f"Patch type {type(patch).__name__} not implemented for Plotly")


def _add_rectangle_patch(
    fig: Figure,
    patch: PlotPatchRectangle,
    row: Optional[int],
    col: Optional[int],
    line_width: Optional[float],
) -> None:
    """Add a rectangle patch to Plotly figure."""
    x0, y0 = patch.xy
    width, height = patch.width, patch.height

    if patch.angle != 0:
        corners = _get_rotated_rectangle_corners(patch)
        x_coords = [p[0] for p in corners] + [corners[0][0]]
        y_coords = [p[1] for p in corners] + [corners[0][1]]

        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="lines",
                line=dict(
                    color=plotly_color(patch.color),
                    width=line_width or patch.linewidth,
                ),
                fill=None,
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    else:
        fig.add_shape(
            type="rect",
            x0=x0,
            y0=y0,
            x1=x0 + width,
            y1=y0 + height,
            line=dict(
                color=plotly_color(patch.color),
                width=line_width or patch.linewidth,
            ),
            fillcolor="rgba(0,0,0,0)",
            row=row,
            col=col,
        )


def _add_circle_patch(
    fig: Figure,
    patch: PlotPatchCircle,
    row: Optional[int],
    col: Optional[int],
    line_width: Optional[float],
) -> None:
    """Add a circle patch to Plotly figure."""
    x_center, y_center = patch.xy
    radius = patch.radius

    fig.add_shape(
        type="circle",
        x0=x_center - radius,
        y0=y_center - radius,
        x1=x_center + radius,
        y1=y_center + radius,
        line=dict(
            color=plotly_color(patch.color),
            width=line_width or patch.linewidth,
        ),
        fillcolor="rgba(0,0,0,0)" if not patch.fill else plotly_color(patch.color),
        row=row,
        col=col,
    )


def _add_ellipse_patch(
    fig: Figure,
    patch: PlotPatchEllipse,
    row: Optional[int],
    col: Optional[int],
    line_width: Optional[float],
) -> None:
    """Add an ellipse patch to Plotly figure."""
    # Plotly can't rotate built-in shapes, so we'll approximate with a polygon
    x_center, y_center = patch.xy
    a, b = patch.width / 2, patch.height / 2
    angle_rad = math.radians(patch.angle)

    # Generate ellipse points
    t = np.linspace(0, 2 * np.pi, 100)
    x_ellipse = a * np.cos(t)
    y_ellipse = b * np.sin(t)

    # Apply rotation
    cos_angle, sin_angle = np.cos(angle_rad), np.sin(angle_rad)
    x_rotated = x_ellipse * cos_angle - y_ellipse * sin_angle + x_center
    y_rotated = x_ellipse * sin_angle + y_ellipse * cos_angle + y_center

    fig.add_trace(
        go.Scatter(
            x=x_rotated,
            y=y_rotated,
            mode="lines",
            line=dict(
                color=plotly_color(patch.color),
                width=line_width or patch.linewidth,
            ),
            fill="toself" if patch.fill else None,
            fillcolor=plotly_color(patch.color) if patch.fill else None,
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def _add_polygon_patch(
    fig: Figure,
    patch: PlotPatchPolygon,
    row: Optional[int],
    col: Optional[int],
    line_width: Optional[float],
) -> None:
    """Add a polygon patch to Plotly figure."""
    vertices = patch.vertices + [patch.vertices[0]]  # Close the polygon
    x_coords = [v[0] for v in vertices]
    y_coords = [v[1] for v in vertices]

    fig.add_trace(
        go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="lines",
            line=dict(
                color=plotly_color(patch.color),
                width=line_width or patch.linewidth,
            ),
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def _add_arc_patch(
    fig: Figure,
    patch: PlotPatchArc,
    row: Optional[int],
    col: Optional[int],
    line_width: Optional[float],
) -> None:
    """Add an arc patch to Plotly figure."""
    x_center, y_center = patch.xy
    radius = patch.width / 2  # Assuming circular arc

    # Convert angles to radians
    theta1_rad = math.radians(patch.theta1)
    theta2_rad = math.radians(patch.theta2)

    # Generate arc points
    if theta2_rad < theta1_rad:
        theta2_rad += 2 * np.pi

    angles = np.linspace(theta1_rad, theta2_rad, 100)
    x_arc = x_center + radius * np.cos(angles)
    y_arc = y_center + radius * np.sin(angles)

    fig.add_trace(
        go.Scatter(
            x=x_arc,
            y=y_arc,
            mode="lines",
            line=dict(
                color=plotly_color(patch.color),
                width=line_width or patch.linewidth,
            ),
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def _add_sbend_patch(
    fig: Figure,
    patch: PlotPatchSbend,
    row: Optional[int],
    col: Optional[int],
    line_width: Optional[float],
) -> None:
    """Add an S-bend patch to Plotly figure."""
    # This is a complex shape - for now, we'll draw the splines as lines
    ((s1x0, s1y0), (s1cx0, s1cy0), (s1x1, s1y1)) = patch.spline1
    ((s2x0, s2y0), (s2cx0, s2cy0), (s2x1, s2y1)) = patch.spline2

    # Generate Bézier curve points for each spline
    t = np.linspace(0, 1, 50)

    # Spline 1
    x1 = (1 - t) ** 2 * s1x0 + 2 * (1 - t) * t * s1cx0 + t**2 * s1x1
    y1 = (1 - t) ** 2 * s1y0 + 2 * (1 - t) * t * s1cy0 + t**2 * s1y1

    # Spline 2
    x2 = (1 - t) ** 2 * s2x0 + 2 * (1 - t) * t * s2cx0 + t**2 * s2x1
    y2 = (1 - t) ** 2 * s2y0 + 2 * (1 - t) * t * s2cy0 + t**2 * s2y1

    # Connect the splines
    x_coords = np.concatenate([x1, [s2x0], x2, [s1x0]])
    y_coords = np.concatenate([y1, [s2y0], y2, [s1y0]])

    fig.add_trace(
        go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="lines",
            line=dict(
                color=plotly_color(patch.color),
                width=line_width or 1.0,
            ),
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def _get_rotated_rectangle_corners(patch: PlotPatchRectangle) -> List[Tuple[float, float]]:
    """Calculate corners of a rotated rectangle."""
    x0, y0 = patch.xy
    width, height = patch.width, patch.height
    angle_rad = math.radians(patch.angle)

    # Rectangle corners relative to center
    center_x = x0 + width / 2
    center_y = y0 + height / 2

    corners_relative = [
        (-width / 2, -height / 2),
        (width / 2, -height / 2),
        (width / 2, height / 2),
        (-width / 2, height / 2),
    ]

    # Rotate and translate corners
    cos_angle, sin_angle = math.cos(angle_rad), math.sin(angle_rad)
    corners = []
    for dx, dy in corners_relative:
        x_rot = dx * cos_angle - dy * sin_angle + center_x
        y_rot = dx * sin_angle + dy * cos_angle + center_y
        corners.append((x_rot, y_rot))

    return corners


TGraph = TypeVar("TGraph", bound=GraphBase)


class PlotlyGraphBase(ABC, Generic[TGraph]):
    """Base class for Plotly graph implementations."""

    manager: GraphManager
    graph: TGraph
    width: Optional[int]
    height: Optional[int]
    xlim: Tuple[float, float]
    ylim: Tuple[float, float]

    def __init__(
        self,
        manager: GraphManager,
        graph: TGraph,
        width: Optional[int] = None,
        height: Optional[int] = None,
        limit_scale_factor: Optional[float] = None,
    ) -> None:
        self.graph = graph
        self.manager = manager
        self.width = width
        self.height = height

        limit_scale_factor = limit_scale_factor or _PlotlyDefaults.limit_scale_factor
        self.xlim = util.apply_factor_to_limits(*graph.xlim, limit_scale_factor)
        self.ylim = util.apply_factor_to_limits(*graph.ylim, limit_scale_factor)

    @abstractmethod
    def create_figure(self) -> Figure:
        """Create the Plotly figure for this graph."""
        raise NotImplementedError()


class PlotlyBasicGraph(PlotlyGraphBase[BasicGraph]):
    """Plotly implementation of BasicGraph."""

    graph_type: ClassVar[str] = "basic"
    graph: BasicGraph
    num_points: int
    view_x_range: Tuple[float, float]

    def __init__(
        self,
        manager: GraphManager,
        graph: BasicGraph,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        super().__init__(
            manager=manager,
            graph=graph,
            width=width,
            height=height,
        )
        self.num_points = graph.get_num_points()
        self.view_x_range = graph.get_x_range()

    @property
    def tao(self) -> Tao:
        return self.manager.tao

    def update_plot(self, fig: Figure) -> None:
        """Update the plot with current data."""
        try:
            self.tao.cmd("set global lattice_calc_on = F")
            self.tao.cmd(f"set plot {self.graph.region_name} n_curve_pts = {self.num_points}")
            self.tao.cmd(
                f"x_scale {self.graph.region_name} {self.view_x_range[0]} {self.view_x_range[1]}"
            )
        except TaoCommandError as ex:
            logger.error(f"Failed to update plot extents: {ex.tao_output}")
        finally:
            self.tao.cmd("set global lattice_calc_on = T")

        try:
            updated = self.graph.update(self.manager)
            if updated is None:
                raise ValueError("update() returned None")
        except Exception:
            logger.exception("Failed to update graph")
            return

        if isinstance(updated, BasicGraph):
            fig.data = []
            for curve in updated.curves:
                _plot_curve(fig, curve, line_width_scale=_PlotlyDefaults.line_width_scale)

        # Update layout
        fig.update_layout(
            title=pgplot.mathjax_string(updated.title),
            xaxis_title=pgplot.mathjax_string(updated.xlabel),
            yaxis_title=pgplot.mathjax_string(updated.ylabel),
        )

    def create_figure(self) -> Figure:
        """Create a Plotly figure for this basic graph."""
        graph = self.graph

        fig = go.Figure()

        # Add curves
        for curve in graph.curves:
            _plot_curve(fig, curve, line_width_scale=_PlotlyDefaults.line_width_scale)

        # Configure layout
        fig.update_layout(
            title=pgplot.mathjax_string(graph.title),
            xaxis_title=pgplot.mathjax_string(graph.xlabel),
            yaxis_title=pgplot.mathjax_string(graph.ylabel),
            width=self.width or _PlotlyDefaults.width,
            height=self.height or _PlotlyDefaults.height,
            xaxis=dict(range=list(self.xlim)),
            yaxis=dict(range=list(self.ylim)),
            showlegend=True,
        )

        return fig


class PlotlyLatticeLayoutGraph(PlotlyGraphBase[LatticeLayoutGraph]):
    """Plotly implementation of LatticeLayoutGraph."""

    graph_type: ClassVar[str] = "lat_layout"
    graph: LatticeLayoutGraph

    def __init__(
        self,
        manager: GraphManager,
        graph: LatticeLayoutGraph,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        super().__init__(
            manager=manager,
            graph=graph,
            width=width,
            height=height,
        )

    def create_figure(self) -> Figure:
        """Create a Plotly figure for this lattice layout."""
        graph = self.graph

        fig = go.Figure()

        self._draw_layout_elements(fig)

        fig.update_layout(
            title=pgplot.mathjax_string(graph.title),
            xaxis_title=pgplot.mathjax_string(graph.xlabel),
            width=self.width or _PlotlyDefaults.width,
            height=self.height or _PlotlyDefaults.layout_height,
            xaxis=dict(
                range=list(self.xlim),
                tickmode="array",
                tickvals=[elem.info["ele_s_start"] for elem in graph.elements],
                ticktext=[elem.info["label_name"] for elem in graph.elements],
                tickangle=45,
            ),
            yaxis=dict(
                range=list(self.ylim),
                showticklabels=False,
                showgrid=False,
            ),
            showlegend=False,
        )

        # Add horizontal reference line
        fig.add_hline(y=0, line_width=1, line_color="black")

        return fig

    def _draw_layout_elements(self, fig: Figure) -> None:
        """Draw lattice layout elements on the figure."""
        for elem in self.graph.elements:
            color = plotly_color(elem.color)
            shape = elem.shape
            if not shape:
                continue

            lines = shape.to_lines()
            for line in lines:
                fig.add_trace(
                    go.Scatter(
                        x=line.xs,
                        y=line.ys,
                        mode="lines",
                        line=dict(
                            color=color,
                            width=shape.line_width * _PlotlyDefaults.line_width_scale,
                        ),
                        name=elem.name,
                        showlegend=False,
                        hovertemplate=f"<b>{elem.name}</b><br>"
                        + f"s_start: {elem.info['ele_s_start']:.3f} m<br>"
                        + f"s_end: {elem.info['ele_s_end']:.3f} m<extra></extra>",
                    )
                )

            if isinstance(shape, LayoutShape):
                for patch in shape.to_patches():
                    _add_patch_to_figure(
                        fig,
                        patch,
                        line_width=shape.line_width * _PlotlyDefaults.line_width_scale,
                    )

        self._draw_annotations(fig)

    def _draw_annotations(self, fig: Figure) -> None:
        """Draw annotations (text labels) on the figure."""
        annotations = []
        for elem in self.graph.elements:
            for annotation in elem.annotations:
                if annotation.text == elem.name:
                    # Skip element name labels as they're better as hover info
                    continue

                annotations.append(
                    dict(
                        x=annotation.x,
                        y=annotation.y,
                        text=pgplot.mathjax_string(annotation.text),
                        showarrow=False,
                        font=dict(
                            size=10,
                            color=plotly_color(annotation.color),
                        ),
                        textangle=annotation.rotation,
                        xanchor="center",
                        yanchor="middle",
                    )
                )

        fig.update_layout(annotations=annotations)


def _draw_floor_plan_shapes(fig, shapes):
    for shape in shapes:
        if isinstance(
            shape,
            (
                floor_plan_shapes.Box,
                floor_plan_shapes.XBox,
                floor_plan_shapes.BowTie,
                floor_plan_shapes.Diamond,
                floor_plan_shapes.Triangle,
            ),
        ):
            vx, vy = shape.vertices
            fig.add_trace(
                go.Scatter(
                    x=list(vx) + [vx[0]],  # Close the polygon
                    y=list(vy) + [vy[0]],
                    mode="lines",
                    line=dict(
                        color=plotly_color(shape.color),
                        width=shape.line_width * _PlotlyDefaults.floor_line_width_scale,
                    ),
                    name=shape.name,
                    showlegend=False,
                    hovertemplate=f"<b>{shape.name}</b><extra></extra>",
                )
            )
        else:
            for patch in shape.to_patches():
                _add_patch_to_figure(
                    fig,
                    patch,
                    line_width=shape.line_width * _PlotlyDefaults.floor_line_width_scale,
                )

            lines = shape.to_lines()
            for line in lines:
                fig.add_trace(
                    go.Scatter(
                        x=line.xs,
                        y=line.ys,
                        mode="lines",
                        line=dict(
                            color=plotly_color(line.color),
                            width=line.linewidth * _PlotlyDefaults.floor_line_width_scale,
                        ),
                        name=shape.name,
                        showlegend=False,
                        hovertemplate=f"<b>{shape.name}</b><extra></extra>",
                    )
                )


class PlotlyFloorPlanGraph(PlotlyGraphBase[FloorPlanGraph]):
    """Plotly implementation of FloorPlanGraph."""

    graph_type: ClassVar[str] = "floor_plan"
    graph: FloorPlanGraph

    def __init__(
        self,
        manager: GraphManager,
        graph: FloorPlanGraph,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        super().__init__(
            manager=manager,
            graph=graph,
            width=width,
            height=height,
        )

    @property
    def tao(self) -> Tao:
        return self.manager.tao

    def create_figure(self) -> Figure:
        """Create a Plotly figure for this floor plan."""
        graph = self.graph

        fig = go.Figure()

        # Draw floor plan elements
        self._draw_floor_plan_elements(fig)

        # Draw building walls
        self._draw_building_walls(fig)

        if graph.floor_orbits is not None:
            _plot_curve_symbols(
                fig,
                graph.floor_orbits.curve,
                name="floor_orbits",
                showlegend=True,
            )

        fig.update_layout(
            title=pgplot.mathjax_string(graph.title),
            xaxis_title=pgplot.mathjax_string(graph.xlabel),
            yaxis_title=pgplot.mathjax_string(graph.ylabel),
            width=self.width or _PlotlyDefaults.width,
            height=self.height or _PlotlyDefaults.height,
            xaxis=dict(
                scaleanchor="y",
                scaleratio=1,
                range=list(self.xlim),
            ),
            yaxis=dict(
                range=list(self.ylim),
            ),
            showlegend=True,
        )

        self._add_limit_border(fig)

        return fig

    def _draw_floor_plan_elements(self, fig: Figure) -> None:
        """Draw floor plan elements on the figure."""
        _draw_floor_plan_shapes(
            fig, [elem.shape for elem in self.graph.elements if elem.shape is not None]
        )

        self._draw_floor_plan_annotations(fig)

    def _draw_building_walls(self, fig: Figure) -> None:
        """Draw building walls on the figure."""
        building_walls = self.graph.building_walls

        for line in building_walls.lines:
            _plot_curve_line(fig, line, name="building_wall", showlegend=False)

        for patch in building_walls.patches:
            line_width = (
                patch.linewidth * _PlotlyDefaults.floor_line_width_scale
                if patch.linewidth is not None
                else 1.0
            )
            _add_patch_to_figure(fig, patch, line_width=line_width)

    def _draw_floor_plan_annotations(self, fig: Figure) -> None:
        """Draw floor plan annotations on the figure."""
        annotations = []
        for elem in self.graph.elements:
            for annotation in elem.annotations:
                annotations.append(
                    dict(
                        x=annotation.x,
                        y=annotation.y,
                        text=pgplot.mathjax_string(annotation.text),
                        showarrow=False,
                        font=dict(
                            size=8,
                            color=plotly_color(annotation.color),
                        ),
                        textangle=annotation.rotation,
                        xanchor="center",
                        yanchor="middle",
                    )
                )

        fig.update_layout(annotations=annotations)

    def _add_limit_border(self, fig: Figure) -> None:
        """Add a border showing the plot limits."""
        x0, x1 = self.xlim
        y0, y1 = self.ylim

        fig.add_shape(
            type="rect",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line=dict(color="gray", width=1, dash="dot"),
            fillcolor="rgba(128,128,128,0.1)",
        )


AnyPlotlyGraph = Union[PlotlyBasicGraph, PlotlyLatticeLayoutGraph, PlotlyFloorPlanGraph]


class PlotlyGraphManager(GraphManager):
    """Plotly backend graph manager."""

    _key_: ClassVar[str] = "plotly"

    @functools.wraps(set_plotly_defaults)
    def configure(self, **kwargs) -> Dict[str, Any]:
        return set_plotly_defaults(**kwargs)

    def to_plotly_graph(self, graph: AnyGraph) -> AnyPlotlyGraph:
        """
        Create a Plotly graph instance from the backend-agnostic AnyGraph version.

        Parameters
        ----------
        graph : AnyGraph

        Returns
        -------
        AnyPlotlyGraph
        """
        if isinstance(graph, BasicGraph):
            return PlotlyBasicGraph(self, graph)
        elif isinstance(graph, LatticeLayoutGraph):
            return PlotlyLatticeLayoutGraph(self, graph)
        elif isinstance(graph, FloorPlanGraph):
            return PlotlyFloorPlanGraph(self, graph)
        raise NotImplementedError(f"Graph type {type(graph).__name__} not supported")

    def plot_grid(
        self,
        templates: List[str],
        grid: Tuple[int, int],
        *,
        include_layout: bool = False,
        share_x: Optional[bool] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        figsize: Optional[Tuple[int, int]] = None,
        layout_height: Optional[int] = None,
        xlim: Union[OptionalLimit, Sequence[OptionalLimit]] = None,
        ylim: Union[OptionalLimit, Sequence[OptionalLimit]] = None,
        curves: Optional[List[CurveIndexToCurve]] = None,
        settings: Optional[List[TaoGraphSettings]] = None,
        save: Union[bool, str, pathlib.Path, None] = None,
    ) -> Tuple[List[AnyGraph], Figure]:
        """
        Plot graphs on a grid with Plotly.

        Parameters
        ----------
        templates : list of str
            Graph template names.
        grid : (nrows, ncols)
            Grid the provided graphs into this many rows and columns.
        include_layout : bool, default=False
            Include a layout plot at the bottom of each column.
        share_x : bool or None, default=None
            Share x-axes where sensible (`None`) or force sharing x-axes (True)
            for all plots.
        figsize : (int, int), optional
            Figure size. Alternative to specifying `width` and `height`.
        width : int, optional
            Width of the whole plot.
        height : int, optional
            Height of the whole plot.
        layout_height : int, optional
            Height of the layout plot.
        xlim : list of (float, float), optional
            X axis limits for each graph.
        ylim : list of (float, float), optional
            Y axis limits for each graph.
        curves : list of Dict[int, TaoCurveSettings], optional
            Curve settings per graph.
        settings : list of TaoGraphSettings, optional
            Graph customization settings, per graph.
        save : pathlib.Path or str, optional
            Save the plot to the given filename.

        Returns
        -------
        list of graphs, Figure
        """
        graphs = self.prepare_grid_by_names(
            template_names=templates,
            curves=curves,
            settings=settings,
            xlim=xlim,
            ylim=ylim,
        )

        if figsize is not None:
            width, height = figsize

        nrows, ncols = grid
        subplot_titles = [graph.title for graph in graphs]
        if include_layout:
            subplot_titles.extend([f"Layout {i + 1}" for i in range(ncols)])
            nrows += 1

        while ncols * nrows < len(graphs):
            nrows += 1
            # TODO
            # raise ValueError(
            #     f"Not enough rows x columns ({nrows}x{ncols}) to fit {len(graphs)} graphs"
            # )

        fig = plotly.subplots.make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=subplot_titles,
            shared_xaxes=share_x or False,
            vertical_spacing=_PlotlyDefaults.grid_spacing,
            horizontal_spacing=_PlotlyDefaults.grid_spacing,
        )

        for i, graph in enumerate(graphs):
            row = (i // ncols) + 1
            col = (i % ncols) + 1

            plotly_graph = self.to_plotly_graph(graph)
            single_fig = plotly_graph.create_figure()

            for trace in single_fig.data:
                fig.add_trace(trace, row=row, col=col)

        if include_layout:
            lattice_layout = self.to_plotly_graph(self.lattice_layout_graph)
            layout_fig = lattice_layout.create_figure()

            for col in range(1, ncols + 1):
                for trace in layout_fig.data:
                    fig.add_trace(trace, row=nrows, col=col)

        fig.update_layout(
            width=width or _PlotlyDefaults.width * ncols,
            height=height or _PlotlyDefaults.stacked_height * nrows,
            showlegend=False,
        )

        if save:
            if save is True:
                save = "plot_grid.html"
            if not pathlib.Path(save).suffix:
                save = f"{save}.html"
            fig.write_html(save)
            logger.info(f"Saving plot to {save!r}")

        return graphs, fig

    def plot(
        self,
        template: str,
        *,
        region_name: Optional[str] = None,
        include_layout: bool = True,
        figsize: Optional[Tuple[int, int]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        layout_height: Optional[int] = None,
        share_x: Optional[bool] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        save: Union[bool, str, pathlib.Path, None] = None,
        curves: Optional[Dict[int, TaoCurveSettings]] = None,
        settings: Optional[TaoGraphSettings] = None,
    ) -> Tuple[List[AnyGraph], Figure]:
        """
        Plot a graph with Plotly.

        Parameters
        ----------
        template : str
            Graph template name.
        region_name : str, optional
            Graph region name.
        include_layout : bool
            Include a layout plot at the bottom if appropriate.
        width : int, optional
            Width of each plot.
        height : int, optional
            Height of each plot.
        layout_height : int, optional
            Height of the layout plot.
        share_x : bool or None, default=None
            Share x-axes where sensible.
        xlim : (float, float), optional
            X axis limits.
        ylim : (float, float), optional
            Y axis limits.
        curves : Dict[int, TaoCurveSettings], optional
            Curve settings.
        settings : TaoGraphSettings, optional
            Graph customization settings.
        save : str or bool, optional
            Save the plot to a file.

        Returns
        -------
        list of graphs, Figure
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

        if figsize is not None:
            width, height = figsize

        if len(graphs) == 1 and not include_layout:
            plotly_graph = self.to_plotly_graph(graphs[0])
            fig = plotly_graph.create_figure()
        else:
            nrows = len(graphs)
            if include_layout and any(graph.is_s_plot for graph in graphs):
                nrows += 1

            subplot_titles = [graph.title for graph in graphs]
            if include_layout:
                subplot_titles.append("Lattice Layout")

            fig = plotly.subplots.make_subplots(
                rows=nrows,
                cols=1,
                subplot_titles=subplot_titles,
                shared_xaxes=share_x or (share_x is None),
                vertical_spacing=_PlotlyDefaults.grid_spacing,
            )

            for i, graph in enumerate(graphs):
                plotly_graph = self.to_plotly_graph(graph)
                single_fig = plotly_graph.create_figure()

                for trace in single_fig.data:
                    fig.add_trace(trace, row=i + 1, col=1)

            if include_layout and any(graph.is_s_plot for graph in graphs):
                lattice_layout = self.to_plotly_graph(self.lattice_layout_graph)
                layout_fig = lattice_layout.create_figure()

                for trace in layout_fig.data:
                    fig.add_trace(trace, row=nrows, col=1)

            fig.update_layout(
                width=width or _PlotlyDefaults.width,
                height=height or _PlotlyDefaults.stacked_height * nrows,
                showlegend=True,
            )

        if save:
            if save is True:
                save = f"{template}.html"
            if not pathlib.Path(save).suffix:
                save = f"{save}.html"
            fig.write_html(save)
            logger.info(f"Saving plot to {save!r}")

        return graphs, fig

    def plot_field(
        self,
        ele_id: str,
        *,
        colormap: Optional[str] = None,
        radius: float = 0.015,
        num_points: int = 100,
        x_scale: float = 1.0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        save: Union[bool, str, pathlib.Path, None] = None,
    ) -> Tuple[ElementField, Figure]:
        """
        Plot field information for a given element.

        Parameters
        ----------
        ele_id : str
            Element ID.
        colormap : str, optional
            Colormap for the plot.
        radius : float, default=0.015
            Radius.
        num_points : int, default=100
            Number of data points.
        x_scale : float, default=1.0
            X axis scaling factor.
        width : int, optional
        height : int, optional
        save : pathlib.Path or str, optional
            Save the plot to the given filename.

        Returns
        -------
        ElementField, Figure
        """
        field = ElementField.from_tao(self.tao, ele_id, num_points=num_points, radius=radius)

        fig = go.Figure()

        # Create heatmap
        by_data = np.asarray(field.by).T
        # s_min, s_max = np.min(field.s), np.max(field.s)

        fig.add_trace(
            go.Heatmap(
                z=by_data,
                x=np.asarray(field.s) * x_scale,
                y=np.linspace(-1, 1, by_data.shape[0]),  # Normalized y range
                colorscale=colormap or _PlotlyDefaults.colorscale,
                showscale=True,
                name="field",
            )
        )

        fig.update_layout(
            title=f"Field of {ele_id}",
            xaxis_title="s [m]",
            yaxis_title="Normalized Position",
            width=width or _PlotlyDefaults.width,
            height=height or _PlotlyDefaults.height,
        )

        if save:
            if save is True:
                save = f"{ele_id}_field.html"
            if not pathlib.Path(save).suffix:
                save = f"{save}.html"
            fig.write_html(save)
            logger.info(f"Saving plot to {save!r}")

        return field, fig


class PlotlyVariable:
    """Variable widget for Plotly plots in Jupyter notebooks."""

    def __init__(
        self,
        name: str,
        value: float,
        step: float,
        info: FloatVariableInfo,
        parameter: str = "model",
    ):
        self.name = name
        self.value = value
        self.step = step
        self.info = info
        self.parameter = parameter
        self._widgets = None
        self._update_callback = None

    def create_widgets(
        self,
        tao: Tao,
        update_callback: callable,
        show_sliders: bool = True,
    ) -> List[widgets.Widget]:
        """Create interactive widgets for this variable."""
        if not IPYWIDGETS_AVAILABLE:
            raise ImportError("ipywidgets is required for interactive variables")

        self._update_callback = update_callback

        # Create spinner (number input)
        spinner = widgets.FloatText(
            description=self.name,
            value=self.value,
            step=self.step,
            min=self.info["low_lim"],
            max=self.info["high_lim"],
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px"),
        )

        def on_spinner_change(change):
            if change["type"] == "change" and change["name"] == "value":
                self._handle_value_change(change["new"], tao)

        spinner.observe(on_spinner_change)

        if not show_sliders:
            self._widgets = [spinner]
            return self._widgets

        # Create slider
        slider = widgets.FloatSlider(
            description="",
            value=self.value,
            min=self.info["low_lim"],
            max=self.info["high_lim"],
            step=self.step,
            readout=False,
            layout=widgets.Layout(width="200px"),
        )

        def on_slider_change(change):
            if change["type"] == "change" and change["name"] == "value":
                spinner.value = change["new"]  # This will trigger spinner change

        slider.observe(on_slider_change)

        # Link widgets
        def on_spinner_change_linked(change):
            if change["type"] == "change" and change["name"] == "value":
                slider.value = change["new"]
                self._handle_value_change(change["new"], tao)

        spinner.observe(on_spinner_change_linked)

        self._widgets = [slider, spinner]
        return self._widgets

    def _handle_value_change(self, new_value: float, tao: Tao):
        """Handle value change from widgets."""
        try:
            self.set_value(tao, new_value)
            if self._update_callback:
                self._update_callback()
        except Exception as ex:
            print(f"Error updating {self.name}: {ex}")

    def set_value(self, tao: Tao, value: float):
        """Set the variable value in Tao."""
        self.value = value
        tao.cmd(f"set var {self.name}|{self.parameter} = {self.value}")

    @classmethod
    def from_tao(cls, tao: Tao, name: str, *, parameter: str = "model") -> PlotlyVariable:
        """Create a PlotlyVariable from Tao variable info."""
        from typing import cast

        info = cast(FloatVariableInfo, tao.var(name))
        return cls(
            name=name,
            info=info,
            step=info["key_delta"] or 0.01,
            value=info[f"{parameter}_value"],
            parameter=parameter,
        )

    @classmethod
    def from_tao_all(cls, tao: Tao, *, parameter: str = "model") -> List[PlotlyVariable]:
        """Create PlotlyVariables for all Tao variables."""
        return [
            cls.from_tao(
                tao=tao,
                name=f"{var_info['name']}[{idx}]",
                parameter=parameter,
            )
            for var_info in tao.var_general()
            for idx in range(var_info["lbound"], var_info["ubound"] + 1)
        ]


class PlotlyAppCreator:
    """
    A Plotly application creator for Jupyter notebooks with interactive widgets.
    """

    def __init__(
        self,
        manager: PlotlyGraphManager,
        graphs: List[AnyGraph],
        figure: Figure,
        variables: Optional[List[PlotlyVariable]] = None,
    ):
        self.manager = manager
        self.graphs = graphs
        self.figure = figure
        self.variables = variables or []
        self._current_figures = {}

    def create_variable_widgets(self) -> Optional[widgets.Widget]:
        """Create interactive widgets for variables."""
        if not self.variables or not IPYWIDGETS_AVAILABLE:
            return None

        def update_plots():
            """Update all plots when variables change."""
            try:
                # Re-create the plots with updated data
                updated_graphs = []
                for graph in self.graphs:
                    try:
                        updated = graph.update(self.manager)
                        if updated is not None:
                            updated_graphs.append(updated)
                    except Exception as ex:
                        print(f"Error updating graph {graph}: {ex}")
                        updated_graphs.append(graph)  # Keep original if update fails

                if updated_graphs:
                    # Create new figure with updated data
                    if len(updated_graphs) == 1:
                        plotly_graph = self.manager.to_plotly_graph(updated_graphs[0])
                        new_fig = plotly_graph.create_figure()
                    else:
                        # For multiple graphs, recreate the subplot
                        import plotly.subplots

                        new_fig = plotly.subplots.make_subplots(
                            rows=len(updated_graphs),
                            cols=1,
                            subplot_titles=[graph.title for graph in updated_graphs],
                        )

                        for i, graph in enumerate(updated_graphs):
                            plotly_graph = self.manager.to_plotly_graph(graph)
                            single_fig = plotly_graph.create_figure()

                            for trace in single_fig.data:
                                new_fig.add_trace(trace, row=i + 1, col=1)

                    # Update the display
                    with self.figure.batch_update():
                        self.figure.data = new_fig.data
                        self.figure.layout = new_fig.layout

            except Exception as ex:
                print(f"Error updating plots: {ex}")

        # Create widgets for each variable
        widget_rows = []
        widgets_per_row = _PlotlyDefaults.variables_per_row
        if _PlotlyDefaults.show_sliders:
            widgets_per_row *= 2

        all_widgets = []
        for var in self.variables:
            var_widgets = var.create_widgets(
                tao=self.manager.tao,
                update_callback=update_plots,
                show_sliders=_PlotlyDefaults.show_sliders,
            )
            all_widgets.extend(var_widgets)

        # Group widgets into rows
        for i in range(0, len(all_widgets), widgets_per_row):
            row_widgets = all_widgets[i : i + widgets_per_row]
            widget_rows.append(widgets.HBox(row_widgets))

        if widget_rows:
            return widgets.VBox(widget_rows)
        return None

    def show(self, notebook_handle: bool = True):
        """Display the plot in Jupyter notebook."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly is required for PlotlyNotebookGraphManager")

        # Create variable widgets if any
        variable_widget = self.create_variable_widgets()

        if variable_widget and IPYWIDGETS_AVAILABLE:
            # Display widgets above the plot
            display(variable_widget)

        # Display the plot
        if notebook_handle:
            pyo.iplot(self.figure)
        else:
            self.figure.show()

    def save(
        self,
        filename: AnyPath = "",
        *,
        title: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Optional[pathlib.Path]:
        """Save the plot to an HTML file."""
        if not filename:
            title = title or "plotly_plot"
            filename = f"{title}.html"
        if not pathlib.Path(filename).suffix:
            filename = f"{filename}.html"

        self.figure.write_html(filename)
        return pathlib.Path(filename)


class PlotlyNotebookGraphManager(PlotlyGraphManager):
    """Jupyter notebook Plotly backend graph manager."""

    _key_: ClassVar[str] = "plotly_notebook"

    def plot_grid(
        self,
        templates: List[str],
        grid: Tuple[int, int],
        *,
        curves: Optional[List[CurveIndexToCurve]] = None,
        settings: Optional[List[TaoGraphSettings]] = None,
        include_layout: bool = False,
        share_x: Optional[bool] = None,
        vars: bool = False,
        figsize: Optional[Tuple[int, int]] = None,
        layout_height: Optional[int] = None,
        xlim: Union[OptionalLimit, Sequence[OptionalLimit]] = None,
        ylim: Union[OptionalLimit, Sequence[OptionalLimit]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        save: Union[bool, str, pathlib.Path, None] = None,
        notebook_handle: bool = True,
    ) -> Tuple[List[AnyGraph], PlotlyAppCreator]:
        """
        Plot graphs on a grid with Plotly in Jupyter notebook.

        Parameters
        ----------
        templates : list of str
            Graph template names.
        grid : (nrows, ncols)
            Grid the provided graphs into this many rows and columns.
        include_layout : bool, default=False
            Include a layout plot at the bottom of each column.
        share_x : bool or None, default=None
            Share x-axes where sensible (`None`) or force sharing x-axes (True)
            for all plots.
        vars : bool, default=False
            Show Tao variables as adjustable widgets, like "single mode".
        figsize : (int, int), optional
            Figure size. Alternative to specifying `width` and `height`.
        width : int, optional
            Width of the whole plot.
        height : int, optional
            Height of the whole plot.
        layout_height : int, optional
            Height of the layout plot.
        xlim : list of (float, float), optional
            X axis limits for each graph.
        ylim : list of (float, float), optional
            Y axis limits for each graph.
        curves : list of Dict[int, TaoCurveSettings], optional
            Curve settings per graph.
        settings : list of TaoGraphSettings, optional
            Graph customization settings, per graph.
        save : pathlib.Path or str, optional
            Save the plot to the given filename.
        notebook_handle : bool, default=True
            Use notebook handle for interactive updates.

        Returns
        -------
        list of graphs, PlotlyAppCreator
        """
        graphs, figure = super().plot_grid(
            templates=templates,
            grid=grid,
            curves=curves,
            settings=settings,
            include_layout=include_layout,
            share_x=share_x,
            figsize=figsize,
            width=width,
            height=height,
            xlim=xlim,
            ylim=ylim,
            layout_height=layout_height,
            save=save,
        )

        variables = PlotlyVariable.from_tao_all(self.tao) if vars else []

        app = PlotlyAppCreator(
            manager=self,
            graphs=graphs,
            figure=figure,
            variables=variables,
        )

        app.show(notebook_handle=notebook_handle)
        return graphs, app

    def plot(
        self,
        template: str,
        *,
        region_name: Optional[str] = None,
        include_layout: bool = True,
        width: Optional[int] = None,
        height: Optional[int] = None,
        layout_height: Optional[int] = None,
        share_x: Optional[bool] = None,
        vars: bool = False,
        xlim: Optional[Limit] = None,
        ylim: Optional[Limit] = None,
        notebook_handle: bool = True,
        save: Union[bool, str, pathlib.Path, None] = None,
        curves: Optional[Dict[int, TaoCurveSettings]] = None,
        settings: Optional[TaoGraphSettings] = None,
    ) -> Tuple[List[AnyGraph], PlotlyAppCreator]:
        """
        Plot a graph with Plotly in Jupyter notebook.

        Parameters
        ----------
        template : str
            Graph template name.
        region_name : str, optional
            Graph region name.
        include_layout : bool
            Include a layout plot at the bottom if appropriate.
        width : int, optional
            Width of each plot.
        height : int, optional
            Height of each plot.
        layout_height : int, optional
            Height of the layout plot.
        share_x : bool or None, default=None
            Share x-axes where sensible.
        vars : bool, default=False
            Show Tao variables as adjustable widgets, like "single mode".
        xlim : (float, float), optional
            X axis limits.
        ylim : (float, float), optional
            Y axis limits.
        notebook_handle : bool, default=True
            Use notebook handle for interactive updates.
        curves : Dict[int, TaoCurveSettings], optional
            Curve settings.
        settings : TaoGraphSettings, optional
            Graph customization settings.
        save : str or bool, optional
            Save the plot to a file.

        Returns
        -------
        list of graphs, PlotlyAppCreator
        """
        graphs, figure = super().plot(
            template=template,
            region_name=region_name,
            include_layout=include_layout,
            width=width,
            height=height,
            layout_height=layout_height,
            xlim=xlim,
            ylim=ylim,
            curves=curves,
            settings=settings,
            share_x=share_x,
            save=save,
        )

        variables = PlotlyVariable.from_tao_all(self.tao) if vars else []

        app = PlotlyAppCreator(
            manager=self,
            graphs=graphs,
            figure=figure,
            variables=variables,
        )

        app.show(notebook_handle=notebook_handle)
        return graphs, app

    def plot_field(
        self,
        ele_id: str,
        *,
        colormap: Optional[str] = None,
        radius: float = 0.015,
        num_points: int = 100,
        width: Optional[int] = None,
        height: Optional[int] = None,
        x_scale: float = 1.0,
        save: Union[bool, str, pathlib.Path, None] = None,
        notebook_handle: bool = True,
    ) -> Tuple[ElementField, Figure]:
        """
        Plot field information for a given element in Jupyter notebook.

        Parameters
        ----------
        ele_id : str
            Element ID.
        colormap : str, optional
            Colormap for the plot.
        radius : float, default=0.015
            Radius.
        num_points : int, default=100
            Number of data points.
        width : int, optional
        height : int, optional
        x_scale : float, default=1.0
            X axis scaling factor.
        save : pathlib.Path or str, optional
            Save the plot to the given filename.
        notebook_handle : bool, default=True
            Use notebook handle for display.

        Returns
        -------
        ElementField, Figure
        """
        field, figure = super().plot_field(
            ele_id=ele_id,
            colormap=colormap,
            radius=radius,
            num_points=num_points,
            width=width,
            height=height,
            x_scale=x_scale,
            save=save,
        )

        if notebook_handle:
            pyo.iplot(figure)
        else:
            figure.show()

        return field, figure


@functools.cache
def select_graph_manager_class():
    """Select the appropriate Plotly graph manager class."""
    from . import util  # Assuming util has is_jupyter function

    if util.is_jupyter():
        return PlotlyNotebookGraphManager
    return PlotlyGraphManager
