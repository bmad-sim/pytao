from __future__ import annotations

import functools
import logging
import math
import os
import pathlib
import time
import typing
from abc import ABC, abstractmethod
from typing import (
    ClassVar,
    Dict,
    Generic,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import bokeh.colors.named
import bokeh.embed
import bokeh.events
import bokeh.io
import bokeh.layouts
import bokeh.models
import bokeh.plotting

# TODO remove mpl dep - only used in a single spot
import matplotlib
import matplotlib.patches
import numpy as np
from bokeh.core.enums import SizingModeType
from bokeh.document.callbacks import EventCallback
from bokeh.models.sources import ColumnDataSource
from bokeh.plotting import figure
from pydantic.dataclasses import dataclass
from typing_extensions import NotRequired, TypedDict

from ..interface_commands import AnyPath
from ..tao_ctypes.core import TaoCommandError
from . import floor_plan_shapes, pgplot, util
from .curves import CurveIndexToCurve, PlotCurveLine, PlotCurveSymbols, TaoCurveSettings
from .fields import ElementField
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
    AnyGraph,
    BasicGraph,
    FloorPlanElement,
    FloorPlanGraph,
    GraphBase,
    GraphManager,
    LatticeLayoutElement,
    LatticeLayoutGraph,
    PlotAnnotation,
    PlotCurve,
    UnsupportedGraphError,
)
from .settings import TaoGraphSettings
from .types import FloatVariableInfo
from .util import Limit, OptionalLimit, fix_grid_limits

if typing.TYPE_CHECKING:
    from .. import Tao


logger = logging.getLogger(__name__)


def bokeh_color(color):
    color = color.lower().replace("_", "")
    return getattr(bokeh.colors.named, color, "black")


class CurveData(TypedDict):
    line: NotRequired[ColumnDataSource]
    symbol: NotRequired[ColumnDataSource]


class _Defaults:
    """
    Defaults used for Bokeh plots internally.

    To change these values, use `set_defaults`.
    """

    width: int = 400
    height: int = 400
    stacked_height: int = 200
    layout_height: int = 100
    show_bokeh_logo: bool = False
    palette: str = "Magma256"
    tools: str = "pan,wheel_zoom,box_zoom,reset,hover,crosshair"
    grid_toolbar_location: str = "right"
    lattice_layout_tools: str = "pan,wheel_zoom,box_zoom,reset,hover,crosshair"
    floor_plan_tools: str = "pan,wheel_zoom,box_zoom,reset,hover,crosshair"
    floor_plan_annotate_elements: bool = True
    layout_font_size: str = "0.75em"
    floor_plan_font_size: str = "0.75em"
    limit_scale_factor: float = 1.01
    max_data_points: int = 10_000
    variables_per_row: int = 2
    show_sliders: bool = True

    @classmethod
    def get_size_for_class(
        cls,
        typ: Type[AnyBokehGraph],
        user_width: Optional[int] = None,
        user_height: Optional[int] = None,
    ) -> Tuple[int, int]:
        default = {
            BokehBasicGraph: (cls.width, cls.height),
            BokehLatticeLayoutGraph: (cls.width, cls.layout_height),
            BokehFloorPlanGraph: (cls.width, cls.height),
        }[typ]
        return (user_width or default[0], user_height or default[1])


def set_defaults(
    width: Optional[int] = None,
    height: Optional[int] = None,
    stacked_height: Optional[int] = None,
    layout_height: Optional[int] = None,
    palette: Optional[str] = None,
    show_bokeh_logo: Optional[bool] = None,
    tools: Optional[str] = None,
    grid_toolbar_location: Optional[str] = None,
    lattice_layout_tools: Optional[str] = None,
    floor_plan_tools: Optional[str] = None,
    floor_plan_annotate_elements: Optional[bool] = None,
    layout_font_size: Optional[str] = None,
    floor_plan_font_size: Optional[str] = None,
    limit_scale_factor: Optional[float] = None,
    max_data_points: Optional[int] = None,
    variables_per_row: Optional[int] = None,
    show_sliders: Optional[bool] = None,
):
    """
    Change defaults used for Bokeh plots.

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
    palette : str, optional
        Palette for `plot_field`.
    show_bokeh_logo : bool, optional
        Show Bokeh logo on each plot.
    tools : str, default="pan,wheel_zoom,box_zoom,reset,hover,crosshair"
        Bokeh tools to use.
    grid_toolbar_location : str, default="right"
        Toolbar location for gridded plots.
    lattice_layout_tools : str, optional
        Bokeh tools to use specifically for lattice layouts.
    floor_plan_tools : str, optional
        Bokeh tools to use specifically for floor plan layouts.
    layout_font_size : str, optional
        Font size to use in lattice layouts.
    floor_plan_font_size : str, optional
        Font size to use in floor plan layouts.
    limit_scale_factor : float, default=1.01
        View limits from Tao are scaled by this factor.  This can be used to
        ensure that all data is visible despite drawing method differences.
    max_data_points : int, optional
        Maximum number of data points to show in the slider.
    variables_per_row : int, default=2
        Variables to list per row when in single mode (i.e., `vars=True`).
    show_sliders : bool, default=True
        Show sliders alongside the spinners in single mode.
    """

    if width is not None:
        _Defaults.width = int(width)
    if height is not None:
        _Defaults.height = int(height)
    if stacked_height is not None:
        _Defaults.stacked_height = int(stacked_height)
    if layout_height is not None:
        _Defaults.layout_height = int(layout_height)
    if palette is not None:
        _Defaults.palette = palette
    if show_bokeh_logo is not None:
        _Defaults.show_bokeh_logo = bool(show_bokeh_logo)
    if tools is not None:
        _Defaults.tools = tools
    if grid_toolbar_location is not None:
        _Defaults.grid_toolbar_location = grid_toolbar_location
    if lattice_layout_tools is not None:
        _Defaults.lattice_layout_tools = lattice_layout_tools
    if floor_plan_tools is not None:
        _Defaults.floor_plan_tools = floor_plan_tools
    if floor_plan_annotate_elements is not None:
        _Defaults.floor_plan_annotate_elements = floor_plan_annotate_elements
    if layout_font_size is not None:
        _Defaults.layout_font_size = layout_font_size
    if floor_plan_font_size is not None:
        _Defaults.floor_plan_font_size = floor_plan_font_size
    if limit_scale_factor is not None:
        _Defaults.limit_scale_factor = float(limit_scale_factor)
    if max_data_points is not None:
        _Defaults.max_data_points = int(max_data_points)
    if variables_per_row is not None:
        _Defaults.variables_per_row = int(variables_per_row)
    if show_sliders is not None:
        _Defaults.show_sliders = bool(show_sliders)
    return {
        key: value
        for key, value in vars(_Defaults).items()
        if not key.startswith("_") and key not in {"get_size_for_class"}
    }


def _get_curve_data(curve: PlotCurve) -> CurveData:
    data: CurveData = {}
    if curve.line is not None:
        data["line"] = ColumnDataSource(
            data={
                "x": curve.line.xs,
                "y": curve.line.ys,
            }
        )
    if curve.symbol is not None:
        data["symbol"] = ColumnDataSource(
            data={
                "x": curve.symbol.xs,
                "y": curve.symbol.ys,
            }
        )
    return data


def _get_graph_data(graph) -> List[CurveData]:
    return [_get_curve_data(curve) for curve in graph.curves]


def share_x_axes(figs: List[figure]):
    if not figs:
        return
    fig0, *others = figs
    for other in others:
        other.x_range = fig0.x_range


class BGraphAndFigure(NamedTuple):
    bgraph: AnyBokehGraph
    fig: figure


T_Tool = TypeVar("T_Tool", bound=bokeh.models.Tool)


def get_tool_from_figure(fig: figure, tool_cls: Type[T_Tool]) -> Optional[T_Tool]:
    tools = [tool for tool in fig.tools if isinstance(tool, tool_cls)]
    return tools[0] if tools else None


def link_crosshairs(figs: List[figure]):
    first, *rest = figs
    crosshair = get_tool_from_figure(first, bokeh.models.CrosshairTool)
    if crosshair is None:
        return

    if crosshair.overlay == "auto":
        crosshair.overlay = (
            bokeh.models.Span(dimension="width", line_dash="dotted", line_width=1),
            bokeh.models.Span(dimension="height", line_dash="dotted", line_width=1),
        )

    for fig in rest:
        other_crosshair = get_tool_from_figure(fig, bokeh.models.CrosshairTool)
        if other_crosshair:
            other_crosshair.overlay = crosshair.overlay


def share_common_x_axes(
    pairs: List[BGraphAndFigure],
    crosshairs: bool = True,
) -> List[List[BGraphAndFigure]]:
    res: List[List[BGraphAndFigure]] = []

    s_plots = []
    for pair in pairs:
        if pair.bgraph.graph.is_s_plot:
            s_plots.append(pair)

    if s_plots:
        res.append(s_plots)

    by_xlabel: Dict[str, List[BGraphAndFigure]] = {}
    for pair in pairs:
        if pair in s_plots:
            continue
        by_xlabel.setdefault(pair.bgraph.graph.xlabel, []).append(pair)

    for sharing_set in by_xlabel.values():
        if len(sharing_set) > 1:
            res.append(sharing_set)

    for sharing_set in res:
        figs = [pair.fig for pair in sharing_set]
        share_x_axes(figs)
        if crosshairs:
            link_crosshairs(figs)

    return res


def _plot_curve_symbols(
    fig: figure,
    symbol: PlotCurveSymbols,
    name: str,
    source: Optional[ColumnDataSource] = None,
    legend_label: Optional[str] = None,
):
    marker = pgplot.bokeh_symbols.get(symbol.marker, "dot")
    if not marker:
        return

    if source is None:
        source = ColumnDataSource(data={})

    source.data.update(
        {
            "x": symbol.xs,
            "y": symbol.ys,
        }
    )

    if legend_label is not None:
        # Can't pass legend_label unless it's set to non-None
        kw = {"legend_label": legend_label}
    else:
        kw = {}
    return fig.scatter(
        "x",
        "y",
        source=source,
        fill_color=bokeh_color(symbol.color),
        marker=marker,
        size=symbol.markersize * 4 if marker == "dot" else symbol.markersize,
        name=name,
        **kw,
    )


def _plot_curve_line(
    fig: figure,
    line: PlotCurveLine,
    name: Optional[str] = None,
    source: Optional[ColumnDataSource] = None,
):
    if source is None:
        source = ColumnDataSource(data={})

    source.data.update({"x": line.xs, "y": line.ys})
    if name is not None:
        # Can't pass legend_label unless it's set to non-None
        kw = {"legend_label": name}
    else:
        kw = {}
    return fig.line(
        "x",
        "y",
        line_width=line.linewidth,
        source=source,
        color=bokeh_color(line.color),
        name=name,
        **kw,
    )


def _plot_curve(fig: figure, curve: PlotCurve, source: CurveData) -> None:
    name = pgplot.mathjax_string(curve.info["name"])
    if "line" in source and curve.line is not None:
        _plot_curve_line(fig=fig, line=curve.line, name=name, source=source["line"])

    if "symbol" in source and curve.symbol is not None:
        legend = None if "line" in source else name
        _plot_curve_symbols(
            fig=fig,
            symbol=curve.symbol,
            source=source["symbol"],
            name=name,
            legend_label=legend,
        )


def _plot_patch_arc(
    fig: figure,
    patch: PlotPatchArc,
    source: Optional[ColumnDataSource] = None,
    linewidth: Optional[float] = None,
):
    if source is None:
        source = ColumnDataSource(data={})
    if not np.isclose(patch.width, patch.height):
        logger.warning(
            "Arcs only support circular arcs for now (w=%f h=%f)",
            patch.width,
            patch.height,
        )

    source.data.update(
        {
            "x": [patch.xy[0]],
            "y": [patch.xy[1]],
            "radius": [patch.width / 2],
            "start_angle": [math.radians(patch.theta1)],
            "end_angle": [math.radians(patch.theta2)],
            # NOTE: debugging with aspect ratios...
            # "x": [0],
            # "y": [0],
            # "radius": [1],  # patch.width / 2],
            # "start_angle": [0],
            # "end_angle": [math.radians(345)],
        }
    )
    return fig.arc(
        x="x",
        y="y",
        radius="radius",
        start_angle="start_angle",
        end_angle="end_angle",
        line_width=linewidth if linewidth is not None else patch.linewidth,
        source=source,
    )


def _plot_sbend_patch(fig: figure, patch: PlotPatchSbend):
    ((s1x0, s1y0), (s1cx0, s1cy0), (s1x1, s1y1)) = patch.spline1
    ((s2x0, s2y0), (s2cx0, s2cy0), (s2x1, s2y1)) = patch.spline2
    fig.bezier(x0=s1x0, y0=s1y0, cx0=s1cx0, cy0=s1cy0, x1=s1x1, y1=s1y1)
    fig.line(x=[s1x1, s2x0], y=[s1y1, s2y0])

    fig.bezier(x0=s2x0, y0=s2y0, cx0=s2cx0, cy0=s2cy0, x1=s2x1, y1=s2y1)
    fig.line(x=[s2x1, s1x0], y=[s2y1, s1y0])


def _draw_layout_elems(
    fig: figure,
    elems: List[LatticeLayoutElement],
    skip_labels: bool = True,
):
    line_data = {
        "xs": [],
        "ys": [],
        "name": [],
        "s_start": [],
        "s_end": [],
        "line_width": [],
        "color": [],
    }
    rectangles: List[Tuple[LatticeLayoutElement, LayoutShape, PlotPatchRectangle]] = []

    _draw_annotations(
        fig,
        {elem.name: elem.annotations for elem in elems},
        font_size=_Defaults.layout_font_size,
        skip_labels=skip_labels,
    )

    for elem in elems:
        color = bokeh_color(elem.color)
        shape = elem.shape
        if not shape:
            continue

        lines = shape.to_lines()
        line_data["xs"].extend([line.xs for line in lines])
        line_data["ys"].extend([line.ys for line in lines])
        line_data["name"].extend([elem.name] * len(lines))
        line_data["s_start"].extend([elem.info["ele_s_start"]] * len(lines))
        line_data["s_end"].extend([elem.info["ele_s_end"]] * len(lines))
        line_data["line_width"].extend([shape.line_width] * len(lines))
        line_data["color"].extend([color] * len(lines))

        if isinstance(shape, LayoutShape):
            for patch in shape.to_patches():
                if isinstance(patch, PlotPatchRectangle):
                    rectangles.append((elem, shape, patch))
                else:
                    _plot_patch(fig, patch, line_width=shape.line_width)

    if rectangles:
        source = ColumnDataSource(
            data={
                "xs": [[[_patch_rect_to_points(patch)[0]]] for _, _, patch in rectangles],
                "ys": [[[_patch_rect_to_points(patch)[1]]] for _, _, patch in rectangles],
                "name": [shape.name for _, shape, _ in rectangles],
                "color": [bokeh_color(shape.color) for _, shape, _ in rectangles],
                "line_width": [shape.line_width for _, shape, _ in rectangles],
                "s_start": [elem.info["ele_s_start"] for elem, _, _ in rectangles],
                "s_end": [elem.info["ele_s_end"] for elem, _, _ in rectangles],
            }
        )
        fig.multi_polygons(
            xs="xs",
            ys="ys",
            color="color",
            line_width="line_width",
            source=source,
            fill_alpha=0.0,
        )

    if line_data:
        fig.multi_line(
            xs="xs",
            ys="ys",
            color="color",
            line_width="line_width",
            source=ColumnDataSource(data=line_data),
        )


def _draw_annotations(
    fig: figure,
    name_to_annotations: Dict[str, List[PlotAnnotation]],
    *,
    font_size: str,
    skip_labels: bool = False,
):
    data = {
        "x": [],
        "y": [],
        "name": [],
        "text": [],
        "color": [],
        "baseline": [],
        "align": [],
        "rotation": [],
        "font_size": [],
    }

    for name, annotations_ in name_to_annotations.items():
        for annotation in annotations_:
            if annotation.text == name and skip_labels:
                # We skip labels here as they work better as X tick labels
                continue

            baseline = annotation.verticalalignment
            if baseline == "center":
                baseline = "middle"
            data["x"].append(annotation.x)
            data["y"].append(annotation.y)
            data["text"].append(pgplot.mathjax_string(annotation.text))
            data["name"].append(name)
            data["rotation"].append(math.radians(annotation.rotation))
            data["align"].append(annotation.horizontalalignment)
            data["baseline"].append(baseline)
            data["color"].append(bokeh_color(annotation.color))
            data["font_size"].append(font_size)

    return fig.text(
        "x",
        "y",
        angle="rotation",
        text_align="align",
        text_baseline="baseline",
        text_font_size="font_size",
        color="color",
        source=ColumnDataSource(data=data),
    )


def _draw_floor_plan_shapes(
    fig: figure,
    elems: List[FloorPlanElement],
):
    polygon_data = {
        "xs": [],
        "ys": [],
        "name": [],
        "line_width": [],
        "color": [],
    }
    line_data = {
        "xs": [],
        "ys": [],
        "name": [],
        "line_width": [],
        "color": [],
    }
    for elem in elems:
        shape = elem.shape
        if not shape:
            continue

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
            polygon_data["xs"].append([[vx]])
            polygon_data["ys"].append([[vy]])
            polygon_data["name"].append(shape.name)
            polygon_data["line_width"].append(shape.line_width)
            polygon_data["color"].append(bokeh_color(shape.color))
        else:
            for patch in shape.to_patches():
                assert not isinstance(patch, (PlotPatchRectangle, PlotPatchPolygon))
                _plot_patch(fig, patch, line_width=shape.line_width)

            lines = shape.to_lines()
            if lines:
                line_data["xs"].extend([line.xs for line in lines])
                line_data["ys"].extend([line.ys for line in lines])
                line_data["name"].extend([shape.name] * len(lines))
                line_data["line_width"].extend([line.linewidth for line in lines])
                line_data["color"].extend([bokeh_color(line.color) for line in lines])

    if line_data["xs"]:
        fig.multi_line(
            xs="xs",
            ys="ys",
            color="color",
            line_width="line_width",
            source=ColumnDataSource(data=line_data),
        )

    if polygon_data["xs"]:
        fig.multi_polygons(
            xs="xs",
            ys="ys",
            color="color",
            line_width="line_width",
            source=ColumnDataSource(data=polygon_data),
            fill_alpha=0.0,
        )


def _patch_rect_to_points(patch: PlotPatchRectangle) -> Tuple[List[float], List[float]]:
    mpl_patch = matplotlib.patches.Rectangle(
        xy=patch.xy,
        width=patch.width,
        height=patch.height,
        angle=patch.angle,
        rotation_point=patch.rotation_point,
        **patch._patch_args,
    )

    points = mpl_patch.get_corners()
    return (
        points[:, 0].tolist() + [points[0, 0]],
        points[:, 1].tolist() + [points[0, 1]],
    )


def _draw_limit_border(
    fig: figure,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    alpha: float = 1.0,
):
    width = xlim[1] - xlim[0]
    height = ylim[1] - ylim[0]
    rect = PlotPatchRectangle(xy=(xlim[0], ylim[0]), width=width, height=height, alpha=alpha)
    px, py = _patch_rect_to_points(rect)

    return fig.line(px, py, alpha=alpha)


def _plot_patch(
    fig: figure,
    patch: PlotPatch,
    line_width: Optional[float] = None,
    source: Optional[ColumnDataSource] = None,
):
    if source is None:
        source = ColumnDataSource()

    line_width = line_width if line_width is not None else patch.linewidth
    if isinstance(patch, PlotPatchRectangle):
        cx, cy = patch.center
        source.data["x"] = [cx]
        source.data["y"] = [cy]
        source.data["width"] = [patch.width]
        source.data["height"] = [patch.height]
        return fig.rect(
            x="x",
            y="y",
            width="width",
            height="height",
            angle=math.radians(patch.angle),
            fill_alpha=0.0,
            line_alpha=1.0,
            line_color=bokeh_color(patch.color),
            line_width=line_width,
            source=source,
        )
    elif isinstance(patch, PlotPatchPolygon):
        source.data["xs"] = [p[0] for p in patch.vertices + patch.vertices[:1]]
        source.data["ys"] = [p[1] for p in patch.vertices + patch.vertices[:1]]
        return fig.line(
            x="xs",
            y="ys",
            line_width=line_width,
            color=bokeh_color(patch.color),
            source=source,
        )
    if isinstance(patch, PlotPatchArc):
        return _plot_patch_arc(fig, patch, source=source, linewidth=line_width)

    if isinstance(patch, PlotPatchCircle):
        source.data["x"], source.data["y"] = [patch.xy[0]], [patch.xy[1]]
        source.data["radii"] = [patch.radius]
        return fig.circle(
            x="x",
            y="y",
            radius="radii",
            line_width=line_width,
            fill_alpha=int(patch.fill),
            source=source,
        )
    if isinstance(patch, PlotPatchEllipse):
        source.data["x"], source.data["y"] = [patch.xy[0]], [patch.xy[1]]
        source.data["width"] = [patch.width]
        source.data["height"] = [patch.height]
        source.data["angle"] = [math.radians(patch.angle)]
        return fig.ellipse(
            x="x",
            y="y",
            width="width",
            height="height",
            angle="angle",
            line_width=line_width,
            fill_alpha=int(patch.fill),
            source=source,
        )
    if isinstance(patch, PlotPatchSbend):
        return _plot_sbend_patch(fig, patch)
    raise NotImplementedError(f"{type(patch).__name__}")


def _fields_to_data_source(fields: List[ElementField], x_scale: float = 1.0):
    return ColumnDataSource(
        data={
            "ele_id": [field.ele_id for field in fields],
            "by": [np.asarray(field.by).T for field in fields],
            "x": [np.min(field.s) for field in fields],
            "dw": [np.max(field.s) - np.min(field.s) for field in fields],
            "dh": [15.0 for _ in fields],
        }
    )


TGraph = TypeVar("TGraph", bound=GraphBase)


class BokehGraphBase(ABC, Generic[TGraph]):
    manager: GraphManager
    graph: TGraph
    sizing_mode: SizingModeType
    width: Optional[int]
    height: Optional[int]
    aspect_ratio: Optional[float]
    x_range: Optional[bokeh.models.Range]
    y_range: Optional[bokeh.models.Range]

    def __init__(
        self,
        manager: GraphManager,
        graph: TGraph,
        sizing_mode: SizingModeType,
        aspect_ratio: Optional[float] = None,  # w/h
        width: Optional[int] = None,
        height: Optional[int] = None,
        x_range: Optional[bokeh.models.Range] = None,
        y_range: Optional[bokeh.models.Range] = None,
        limit_scale_factor: Optional[float] = None,
    ) -> None:
        self.graph = graph
        self.manager = manager
        self.sizing_mode = sizing_mode
        self.width = width
        self.height = height
        self.aspect_ratio = aspect_ratio

        limit_scale_factor = limit_scale_factor or _Defaults.limit_scale_factor
        self.x_range = x_range or bokeh.models.Range1d(
            *util.apply_factor_to_limits(*graph.xlim, limit_scale_factor)
        )
        self.y_range = y_range or bokeh.models.Range1d(
            *util.apply_factor_to_limits(*graph.ylim, limit_scale_factor)
        )

    def create_widgets(self, fig: figure) -> List[bokeh.models.UIElement]:
        return []

    @abstractmethod
    def create_figure(
        self,
        *,
        tools: str = "pan,wheel_zoom,box_zoom,save,reset,crosshair",
        toolbar_location: str = "above",
    ) -> figure:
        raise NotImplementedError()


class BokehLatticeLayoutGraph(BokehGraphBase[LatticeLayoutGraph]):
    graph_type: ClassVar[str] = "lat_layout"
    graph: LatticeLayoutGraph

    def __init__(
        self,
        manager: GraphManager,
        graph: LatticeLayoutGraph,
        sizing_mode: SizingModeType = "inherit",
        width: Optional[int] = None,
        height: Optional[int] = None,
        aspect_ratio: Optional[float] = None,  # w/h
    ) -> None:
        super().__init__(
            manager=manager,
            graph=graph,
            sizing_mode=sizing_mode,
            aspect_ratio=aspect_ratio,
            width=width,
            height=height,
        )

    def update_plot(
        self,
        fig: figure,
        *,
        widgets: Optional[List[bokeh.models.Widget]] = None,
        tao: Optional[Tao] = None,
    ) -> None:
        if tao is None:
            return

    def create_figure(
        self,
        *,
        tools: Optional[str] = None,
        toolbar_location: str = "above",
    ) -> figure:
        if tools is None:
            tools = _Defaults.lattice_layout_tools

        add_named_hover_tool = isinstance(tools, str) and "hover" in tools.split(",")
        if add_named_hover_tool:
            tools = ",".join(tool for tool in tools.split(",") if tool != "hover")

        graph = self.graph
        fig = figure(
            title=pgplot.mathjax_string(graph.title),
            x_axis_label=pgplot.mathjax_string(graph.xlabel),
            # y_axis_label=pgplot.mathjax_string(graph.ylabel),
            toolbar_location=toolbar_location,
            tools=tools,
            aspect_ratio=self.aspect_ratio,
            sizing_mode=self.sizing_mode,
        )

        box_zoom = get_tool_from_figure(fig, bokeh.models.BoxZoomTool)
        if box_zoom is not None:
            box_zoom.match_aspect = False

        fig.xaxis.ticker = bokeh.models.FixedTicker(
            ticks=[elem.info["ele_s_start"] for elem in graph.elements],
            minor_ticks=[elem.info["ele_s_end"] for elem in graph.elements],
        )
        fig.xaxis.major_label_overrides = {
            elem.info["ele_s_start"]: elem.info["label_name"] for elem in graph.elements
        }
        fig.xaxis.major_label_orientation = math.pi / 4
        fig.yaxis.ticker = []
        fig.yaxis.visible = False

        _draw_layout_elems(fig, self.graph.elements, skip_labels=True)

        if add_named_hover_tool:
            hover = bokeh.models.HoverTool(
                renderers=get_hoverable_renderers(fig),
                tooltips=[
                    ("name", "@name"),
                    ("s start [m]", "@s_start"),
                    ("s end [m]", "@s_end"),
                ],
                mode="vline",
            )

            fig.add_tools(hover)

        fig.renderers.append(
            bokeh.models.Span(location=0, dimension="width", line_color="black", line_width=1)
        )

        if self.x_range is not None:
            fig.x_range = self.x_range
        if self.y_range is not None:
            fig.y_range = self.y_range
        return fig


class BokehBasicGraph(BokehGraphBase[BasicGraph]):
    graph_type: ClassVar[str] = "basic"
    graph: BasicGraph
    curve_data: List[CurveData]
    num_points: int
    view_x_range: Tuple[float, float]

    def __init__(
        self,
        manager: GraphManager,
        graph: BasicGraph,
        sizing_mode: SizingModeType = "inherit",
        width: Optional[int] = None,
        height: Optional[int] = None,
        aspect_ratio: Optional[float] = None,  # w/h
        variables: Optional[List[Variable]] = None,
    ) -> None:
        super().__init__(
            manager=manager,
            graph=graph,
            sizing_mode=sizing_mode,
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
        )
        self.curve_data = _get_graph_data(graph)
        self.num_points = graph.get_num_points()
        self.view_x_range = graph.get_x_range()
        self.variables = variables

    @property
    def tao(self) -> Tao:
        return self.manager.tao

    def _disable_widgets(self, widgets: List[bokeh.models.Widget]) -> None:
        for widget in widgets:
            if hasattr(widget, "disabled"):
                widget.disabled = True
            if hasattr(widget, "title"):
                widget.title = "(plot type changed; disabled)"

    def update_plot(
        self,
        fig: figure,
        *,
        widgets: Optional[List[bokeh.models.Widget]] = None,
    ) -> None:
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

        logger.debug(f"x={self.view_x_range} points={self.num_points}")

        try:
            updated = self.graph.update(self.manager)
            if updated is None:
                raise ValueError("update() returned None")
        except Exception:
            logger.exception("Failed to update graph")
            self._disable_widgets(widgets or [])
            return

        # In case the user mistakenly reuses the same plot region, ensure
        # that at least our axis labels are consistent.
        fig.title.text = pgplot.mathjax_string(updated.title)
        fig.xaxis.axis_label = pgplot.mathjax_string(updated.xlabel)
        fig.yaxis.axis_label = pgplot.mathjax_string(updated.ylabel)

        for orig_data, new_data in zip(self.curve_data, _get_graph_data(updated)):
            line = new_data.get("line", None)
            if line is not None:
                assert "line" in orig_data
                orig_data["line"].data = dict(line.data)

            symbol = new_data.get("symbol", None)
            if symbol is not None:
                assert "symbol" in orig_data
                orig_data["symbol"].data = dict(symbol.data)

    def create_figure(
        self,
        *,
        tools: Optional[str] = None,
        toolbar_location: str = "above",
        sizing_mode: SizingModeType = "inherit",
    ) -> figure:
        graph = self.graph

        if tools is None:
            tools = _Defaults.tools

        fig = figure(
            title=pgplot.mathjax_string(graph.title),
            x_axis_label=pgplot.mathjax_string(graph.xlabel),
            y_axis_label=pgplot.mathjax_string(graph.ylabel),
            toolbar_location=toolbar_location,
            tools=tools,
            sizing_mode=sizing_mode,
            width=self.width,
            height=self.height,
        )
        if self.x_range is not None:
            fig.x_range = self.x_range
        if self.y_range is not None:
            fig.y_range = self.y_range
        for curve, source in zip(graph.curves, self.curve_data):
            _plot_curve(fig, curve, source)
        return fig


def get_hoverable_renderers(fig: figure) -> List[bokeh.models.GlyphRenderer]:
    return [rend for rend in list(fig.renderers) if any(rend.data_source.data.get("name", []))]


class BokehFloorPlanGraph(BokehGraphBase[FloorPlanGraph]):
    graph_type: ClassVar[str] = "floor_plan"
    graph: FloorPlanGraph

    def __init__(
        self,
        manager: GraphManager,
        graph: FloorPlanGraph,
        sizing_mode: SizingModeType = "inherit",
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        super().__init__(
            manager=manager,
            graph=graph,
            sizing_mode=sizing_mode,
            width=width,
            height=height,
        )

    @property
    def tao(self) -> Tao:
        return self.manager.tao

    def create_figure(
        self,
        *,
        tools: Optional[str] = None,
        toolbar_location: str = "above",
        sizing_mode: SizingModeType = "inherit",
    ) -> figure:
        if tools is None:
            tools = _Defaults.floor_plan_tools

        add_named_hover_tool = isinstance(tools, str) and "hover" in tools.split(",")
        if add_named_hover_tool:
            tools = ",".join(tool for tool in tools.split(",") if tool != "hover")

        graph = self.graph
        fig = figure(
            title=pgplot.mathjax_string(graph.title),
            x_axis_label=pgplot.mathjax_string(graph.xlabel),
            y_axis_label=pgplot.mathjax_string(graph.ylabel),
            toolbar_location=toolbar_location,
            tools=tools,
            sizing_mode=sizing_mode,
            width=self.width,
            height=self.height,
            # This is vitally important for glyphs to render properly.
            # Compare how a circle centered at (0, 0) with a radius 1
            # looks with/without this setting
            match_aspect=True,
        )
        # TODO: specifying limits for floor plans can cause malformed glyphs.
        # Setting x_range/y_range apparently does away with `match_aspect`.
        # if self.x_range is not None:
        #     fig.x_range = self.x_range
        # if self.y_range is not None:
        #     fig.y_range = self.y_range

        box_zoom = get_tool_from_figure(fig, bokeh.models.BoxZoomTool)
        if box_zoom is not None:
            box_zoom.match_aspect = True

        _draw_floor_plan_shapes(fig, self.graph.elements)

        if add_named_hover_tool:
            hover = bokeh.models.HoverTool(
                renderers=get_hoverable_renderers(fig),
                tooltips=[
                    ("name", "@name"),
                    # ("Position [m]", "(@x, @y)"),
                ],
            )

            fig.add_tools(hover)

        for line in self.graph.building_walls.lines:
            _plot_curve_line(fig, line)
        for patch in self.graph.building_walls.patches:
            _plot_patch(fig, patch)
        orbits = self.graph.floor_orbits
        if orbits is not None:
            _plot_curve_symbols(fig, orbits.curve, name="floor_orbits")

        if _Defaults.floor_plan_annotate_elements:
            _draw_annotations(
                fig,
                {elem.name: elem.annotations for elem in self.graph.elements},
                font_size=_Defaults.floor_plan_font_size,
                skip_labels=False,
            )
        _draw_limit_border(fig, graph.xlim, graph.ylim, alpha=0.1)
        return fig

    def create_widgets(self, fig: figure) -> List[bokeh.models.UIElement]:
        controls = []
        try:
            (orbits,) = fig.select("floor_orbits")
        except ValueError:
            pass
        else:
            orbits.visible = False
            show_orbits_toggle = bokeh.models.Toggle(label="Show orbits", active=False)

            def orbits_toggled(_attr, _old, show):
                orbits.visible = show

            show_orbits_toggle.on_change("active", orbits_toggled)
            controls.append(show_orbits_toggle)

        if controls:
            return [bokeh.layouts.row(controls)]
        return []


AnyBokehGraph = Union[BokehBasicGraph, BokehLatticeLayoutGraph, BokehFloorPlanGraph]


UIGridLayoutList = List[Optional[bokeh.models.UIElement]]


class BokehAppState:
    pairs: List[BGraphAndFigure]
    layout_pairs: List[BGraphAndFigure]
    grid: List[UIGridLayoutList]

    def __init__(
        self,
        pairs: List[BGraphAndFigure],
        layout_pairs: List[BGraphAndFigure],
        grid: List[UIGridLayoutList],
    ) -> None:
        self.pairs = pairs
        self.layout_pairs = layout_pairs
        self.grid = grid

    def to_gridplot(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        **kwargs,
    ) -> bokeh.models.GridPlot:
        if not _Defaults.show_bokeh_logo:
            for pair in self.pairs + self.layout_pairs:
                pair.fig.toolbar.logo = None

        gridplot = bokeh.layouts.gridplot(
            children=self.grid,
            width=width,
            height=height,
            **kwargs,
        )
        gridplot.toolbar.tools.append(bokeh.models.SaveTool())
        return gridplot

    @property
    def figures(self) -> List[figure]:
        return [pair.fig for pair in [*self.pairs, *self.layout_pairs]]

    def to_html(
        self,
        title: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> str:
        layout = self.to_gridplot(width=width, height=height)
        return bokeh.embed.file_html(models=layout, title=title)

    def save(
        self,
        filename: AnyPath = "",
        *,
        title: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Optional[pathlib.Path]:
        title = title or self.pairs[0].bgraph.graph.title or f"plot-{time.time()}"
        if not filename:
            filename = f"{title}.html"
        if not pathlib.Path(filename).suffix:
            filename = f"{filename}.html"
        source = self.to_html(title=title, width=width, height=height)
        with open(filename, "wt") as fp:
            fp.write(source)
        return pathlib.Path(filename)


def _widgets_to_rows(widgets: Sequence[bokeh.models.UIElement], per_row: int):
    widgets = list(widgets)
    rows = []
    while widgets:
        rows.append(bokeh.layouts.row(widgets[:per_row]))
        widgets = widgets[per_row:]
    return rows


class BokehAppCreator:
    """
    A composite Bokeh application creator made up of 1 or more graphs.

    This can be used to:
    * Generate a static HTML page without Python widgets
    * Generate a Notebook (or standalone) application with Python widgets

    Interactive widgets will use the `Tao` object to adjust variables during
    callbacks resulting from user interaction.
    """

    manager: Union[BokehGraphManager, NotebookGraphManager]
    graphs: List[AnyGraph]
    bgraphs: List[AnyBokehGraph]
    share_x: Optional[bool]
    variables: List[Variable]
    grid: Tuple[int, int]
    width: Optional[int]
    height: Optional[int]
    include_layout: bool
    layout_height: Optional[int]
    xlim: List[OptionalLimit]
    ylim: List[OptionalLimit]
    figures: List[figure]
    graph_sizing_mode: Optional[SizingModeType]

    def __init__(
        self,
        manager: Union[BokehGraphManager, NotebookGraphManager],
        graphs: List[AnyGraph],
        share_x: Optional[bool] = None,
        include_variables: bool = False,
        grid: Optional[Tuple[int, int]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        include_layout: bool = False,
        graph_sizing_mode: Optional[SizingModeType] = None,
        layout_height: Optional[int] = None,
        xlim: Union[OptionalLimit, Sequence[OptionalLimit]] = None,
        ylim: Union[OptionalLimit, Sequence[OptionalLimit]] = None,
    ) -> None:
        if not len(graphs):
            raise ValueError("BokehAppCreator requires 1 or more graph")

        if any(isinstance(graph, LatticeLayoutGraph) for graph in graphs):
            include_layout = False
        elif not any(graph.is_s_plot for graph in graphs):
            include_layout = False

        if not grid:
            grid = (len(graphs), 1)

        if include_layout:
            grid = (grid[0] + 1, grid[1])

        if include_variables:
            variables = Variable.from_tao_all(manager.tao)
        else:
            variables = []

        self.manager = manager
        self.graphs = graphs
        self.share_x = share_x
        self.variables = variables
        self.grid = grid
        self.width = width
        self.height = height
        self.graph_sizing_mode = graph_sizing_mode
        self.include_layout = include_layout
        self.layout_height = layout_height
        self.xlim = fix_grid_limits(xlim, num_graphs=len(graphs))
        self.ylim = fix_grid_limits(ylim, num_graphs=len(graphs))

    def create_state(self) -> BokehAppState:
        """Create an independent application state based on the graph data."""
        pairs, layout_pairs = self._create_figures()
        grid = self._grid_figures(pairs, layout_pairs)
        return BokehAppState(
            pairs=pairs,
            layout_pairs=layout_pairs,
            grid=grid,
        )

    def save(
        self,
        filename: AnyPath = "",
        *,
        title: Optional[str] = None,
    ) -> Optional[pathlib.Path]:
        state = self.create_state()
        state.save(filename=filename, title=title, width=self.width, height=self.height)

    def _create_figures(self) -> Tuple[List[BGraphAndFigure], List[BGraphAndFigure]]:
        bgraphs = [self.manager.to_bokeh_graph(graph) for graph in self.graphs]
        figures = [
            bgraph.create_figure(
                tools=_Defaults.tools,
                toolbar_location=_Defaults.grid_toolbar_location,
            )
            for bgraph in bgraphs
        ]
        pairs = [
            BGraphAndFigure(bgraph=bgraph, fig=fig) for bgraph, fig in zip(bgraphs, figures)
        ]

        if not self.include_layout:
            layout_pairs = []
        else:
            lattice_layout = self.manager.to_bokeh_graph(self.manager.lattice_layout_graph)
            layout_pairs = [
                BGraphAndFigure(
                    fig=lattice_layout.create_figure(
                        tools=_Defaults.lattice_layout_tools,
                        toolbar_location=_Defaults.grid_toolbar_location,
                    ),
                    bgraph=lattice_layout,
                )
                for _ in range(self.ncols)
            ]

        if len(figures) > 1 or layout_pairs:
            if self.share_x is None:
                share_common_x_axes(pairs + layout_pairs)
            elif self.share_x:
                all_figs = figures + [pair.fig for pair in layout_pairs]
                share_x_axes(all_figs)
                link_crosshairs(all_figs)

        return pairs, layout_pairs

    def _grid_figures(
        self,
        pairs: List[BGraphAndFigure],
        layout_pairs: List[BGraphAndFigure],
    ) -> List[UIGridLayoutList]:
        nrows, ncols = self.grid
        rows = [[] for _ in range(nrows)]
        rows_cols = [(row, col) for row in range(nrows) for col in range(ncols)]

        for pair, xl, yl, (row, _col) in zip(pairs, self.xlim, self.ylim, rows_cols):
            fig = pair.fig

            if not isinstance(pair.bgraph, BokehFloorPlanGraph):
                if xl is not None:
                    fig.x_range = bokeh.models.Range1d(*xl)
                if yl is not None:
                    fig.y_range = bokeh.models.Range1d(*yl)

            if self.graph_sizing_mode is not None:
                fig.sizing_mode = self.graph_sizing_mode

            rows[row].append(fig)

        for pair in pairs + layout_pairs:
            is_layout = isinstance(pair.bgraph, BokehLatticeLayoutGraph)
            width, height = _Defaults.get_size_for_class(
                type(pair.bgraph),
                user_width=self.width,
                user_height=self.layout_height if is_layout else self.height,
            )

            pair.fig.frame_width = width
            pair.fig.frame_height = height
            logger.debug("fig %s width=%s height=%s", pair.fig, width, height)

        rows.append([pair.fig for pair in layout_pairs])

        for pair in layout_pairs:
            if pair.fig is not None:
                # pair.fig.min_border_bottom = 80
                pass

        for row in rows:
            for fig in row:
                # NOTE: this value is somewhat arbitrary; it helps align the X axes
                # between consecutive plots
                if fig is not None:
                    fig.min_border_left = 80

        return [row for row in rows if row]

    @property
    def nrows(self) -> int:
        return self.grid[0]

    @property
    def ncols(self) -> int:
        return self.grid[1]

    def _add_update_button(self, state: BokehAppState):
        update_button = bokeh.models.Button(label="Update")

        def update_plot():
            for pair in state.pairs:
                bgraph = pair.bgraph
                if not isinstance(bgraph, BokehBasicGraph):
                    continue

                try:
                    bgraph.update_plot(pair.fig, widgets=[update_button])
                except Exception:
                    logger.exception("Failed to update number of points")

        update_button.on_click(update_plot)
        return update_button

    def _add_num_points_slider(self, state: BokehAppState):
        num_points_slider = bokeh.models.Slider(
            title="Data Points",
            start=10,
            end=_Defaults.max_data_points,
            step=1_000,
            value=401,
        )

        def num_points_changed(_attr, _old, num_points: int):
            if num_points < 1:
                logger.error("Internal error: unexpected number of points")
                return

            for pair in state.pairs:
                bgraph = pair.bgraph
                if not isinstance(bgraph, BokehBasicGraph):
                    continue

                bgraph.num_points = num_points
                try:
                    bgraph.update_plot(pair.fig, widgets=[num_points_slider])
                except Exception:
                    logger.exception("Failed to update number of points")

        num_points_slider.on_change("value", num_points_changed)
        return num_points_slider

    def _monitor_range_updates(self, state: BokehAppState):
        def ranges_update(
            bgraph: BokehBasicGraph, fig: figure, event: bokeh.events.RangesUpdate
        ) -> None:
            x0, x1 = event.x0, event.x1
            if x0 is None or x1 is None or x1 < x0:
                logger.error(f"Internal error: unexpected range: {x0} {x1}")
                return

            new_xrange = bgraph.graph.clamp_x_range(x0, x1)
            if new_xrange != bgraph.view_x_range:
                bgraph.view_x_range = new_xrange

            try:
                bgraph.update_plot(fig, widgets=[])
            except Exception:
                logger.exception("Failed to update number ranges")

        callbacks = []
        for pair in state.pairs:
            if not isinstance(pair.bgraph, BokehBasicGraph):
                continue

            cb = cast(EventCallback, functools.partial(ranges_update, pair.bgraph, pair.fig))
            pair.fig.on_event(bokeh.events.RangesUpdate, cb)
            callbacks.append(cb)

        return callbacks

    def create_variable_widgets(self, state: BokehAppState):
        status_label = bokeh.models.PreText(height_policy="max", max_height=300)
        widgets = [
            var.create_widgets(
                tao=self.manager.tao,
                status_label=status_label,
                pairs=state.pairs,
                show_sliders=_Defaults.show_sliders,
            )
            for var in self.variables
        ]

        per_row = _Defaults.variables_per_row
        if _Defaults.show_sliders:
            per_row *= 2

        rows = _widgets_to_rows(sum(widgets, []), per_row=per_row)
        return bokeh.layouts.row(
            [
                bokeh.layouts.column(rows),
                bokeh.layouts.column([status_label]),
            ]
        )

    def create_app_ui(self):
        # Ensure we get a new set of data sources and figures for each app
        state = self.create_state()

        if not state.pairs:
            return

        widget_models: List[bokeh.layouts.UIElement] = []
        if self.variables:
            widget_models.append(self.create_variable_widgets(state))

        if any(isinstance(pair.bgraph, BokehBasicGraph) for pair in state.pairs):
            update_button = self._add_update_button(state)
            num_points_slider = self._add_num_points_slider(state)
            widget_models.insert(0, bokeh.layouts.row([update_button, num_points_slider]))

            self._monitor_range_updates(state)

        for pair in state.pairs:
            if isinstance(pair.bgraph, BokehFloorPlanGraph):
                widget_models.extend(pair.bgraph.create_widgets(pair.fig))
                break

        gridplot = state.to_gridplot(
            width=self.width,
            height=self.height,
            merge_tools=False,
            toolbar_options={} if _Defaults.show_bokeh_logo else {"logo": None},
        )

        all_elems: List[bokeh.models.UIElement] = [*widget_models, gridplot]
        return bokeh.layouts.column(all_elems)

    def create_full_app(self):
        if os.environ.get("PYTAO_BOKEH_NBCONVERT", "").lower() in {"1", "y"}:
            # Do not show full Bokeh server-backed applications when converting
            # Jupyter notebooks to HTML as they are not supported (and will
            # show up blank).  This is a way around it by only showing the
            # graphs without Python-backed widgets - similar to how static HTML
            # pages are saved.
            state = self.create_state()
            return state.to_gridplot()

        def bokeh_app(doc):
            doc.add_root(self.create_app_ui())

        return bokeh_app


@dataclass
class Variable:
    name: str
    value: float
    step: float
    info: FloatVariableInfo
    parameter: str = "model"

    def update_info(self, tao: Tao) -> FloatVariableInfo:
        self.info = cast(FloatVariableInfo, tao.var(self.name))
        return self.info

    def set_value(self, tao: Tao, value: float):
        self.value = value
        tao.cmd(f"set var {self.name}|{self.parameter} = {self.value}")

    def create_slider(self) -> bokeh.models.Slider:
        return bokeh.models.Slider(
            title=self.name,
            start=self.info["low_lim"],
            end=self.info["high_lim"],
            step=self.step,
            value=self.value,
            name="slider",
        )

    def create_widgets(
        self,
        tao: Tao,
        status_label: bokeh.models.PreText,
        pairs: List[BGraphAndFigure],
        show_sliders: bool,
    ) -> List[bokeh.models.UIElement]:
        spinner = self.create_spinner(tao, status_label, pairs)

        if not show_sliders:
            return [spinner]

        slider = self.create_slider()
        update_linked_value = bokeh.models.CustomJS(
            args=dict(slider=slider, spinner=spinner),
            code="""
            if (cb_obj.name == "slider") {
                spinner.value = slider.value
            } else {
                slider.value = spinner.value
            }
            """,
        )

        slider.js_on_change("value", update_linked_value)
        spinner.js_on_change("value", update_linked_value)
        return [slider, spinner]

    def create_spinner(
        self,
        tao: Tao,
        status_label: bokeh.models.PreText,
        pairs: List[BGraphAndFigure],
    ) -> bokeh.models.Spinner:
        spinner = bokeh.models.Spinner(
            title=self.name,
            value=self.value,
            step=self.step,
            low=self.info["low_lim"],
            high=self.info["high_lim"],
            name="spinner",
        )
        spinner.on_change(
            "value",
            functools.partial(self.ui_update, tao=tao, status_label=status_label, pairs=pairs),
        )
        return spinner

    @classmethod
    def from_tao(cls, tao: Tao, name: str, *, parameter: str = "model") -> Variable:
        info = cast(FloatVariableInfo, tao.var(name))
        return Variable(
            name=name,
            info=info,
            step=info["key_delta"] or 0.01,
            value=info[f"{parameter}_value"],
            parameter=parameter,
        )

    @classmethod
    def from_tao_all(cls, tao: Tao, *, parameter: str = "model") -> List[Variable]:
        return [
            cls.from_tao(
                tao=tao,
                name=f'{var_info["name"]}[{idx}]',
                parameter=parameter,
            )
            for var_info in tao.var_general()
            for idx in range(var_info["lbound"], var_info["ubound"] + 1)
        ]

    def ui_update(
        self,
        attr: str,
        old: float,
        new: float,
        *,
        tao: Tao,
        status_label: bokeh.models.PreText,
        pairs: List[BGraphAndFigure],
    ):
        status_label.text = ""

        def record_exception(ex: Exception) -> None:
            exc_text = _clean_tao_exception_for_user(
                getattr(ex, "tao_output", str(ex)),
                command="tao_set_invalid",
            )
            if status_label.text:
                status_label.text = "\n".join((status_label.text, exc_text))
            else:
                status_label.text = exc_text

        try:
            self.set_value(tao, new)
        except Exception as ex:
            record_exception(ex)

        for pair in pairs:
            if isinstance(pair.bgraph, (BokehBasicGraph, BokehLatticeLayoutGraph)):
                try:
                    pair.bgraph.update_plot(pair.fig)
                except Exception as ex:
                    record_exception(ex)


def _clean_tao_exception_for_user(text: str, command: str) -> str:
    def clean_line(line: str) -> str:
        # "[ERROR | 2024-JUL-22 09:20:20] tao_set_invalid:"
        if line.startswith("[") and line.endswith(f"{command}:"):
            return line.split(f"{command}:", 1)[1]
        return line

    text = text.replace("ERROR detected: ", "\n")
    lines = [clean_line(line.rstrip()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line.strip())


class BokehGraphManager(GraphManager):
    """Bokeh backend graph manager - for non-Jupyter contexts."""

    _key_: ClassVar[str] = "bokeh"

    @functools.wraps(set_defaults)
    def configure(self, **kwargs):
        return set_defaults(**kwargs)

    def to_bokeh_graph(self, graph: AnyGraph) -> AnyBokehGraph:
        """
        Create a Bokeh graph instance from the backend-agnostic AnyGraph version.

        For example, `BasicGraph` becomes `BokehBasicGraph`.

        Parameters
        ----------
        graph : AnyGraph

        Returns
        -------
        AnyBokehGraph
        """
        if isinstance(graph, BasicGraph):
            return BokehBasicGraph(self, graph)
        elif isinstance(graph, LatticeLayoutGraph):
            return BokehLatticeLayoutGraph(self, graph)
        elif isinstance(graph, FloorPlanGraph):
            return BokehFloorPlanGraph(self, graph)
        raise NotImplementedError(type(graph).__name__)

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
    ):
        """
        Plot graphs on a grid with Bokeh.

        Parameters
        ----------
        templates : list of str
            Graph template names.
        grid : (nrows, ncols), optional
            Grid the provided graphs into this many rows and columns.
        include_layout : bool, default=False
            Include a layout plot at the bottom of each column.
        share_x : bool or None, default=None
            Share x-axes where sensible (`None`) or force sharing x-axes (True)
            for all plots.
        figsize : (int, int), optional
            Figure size. Alternative to specifying `width` and `height`
            separately.  This takes precedence over `width` and `height`.
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

        BokehAppCreator
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

        app = BokehAppCreator(
            manager=self,
            graphs=graphs,
            share_x=share_x,
            include_variables=False,
            grid=grid,
            width=width or _Defaults.width,
            height=height or _Defaults.stacked_height,
            layout_height=layout_height or _Defaults.layout_height,
            include_layout=include_layout,
            xlim=xlim,
            ylim=ylim,
        )

        if save:
            if save is True:
                save = ""
            filename = app.save(save)
            logger.info(f"Saving plot to {filename!r}")
        return graphs, app

    def plot(
        self,
        template: str,
        *,
        region_name: Optional[str] = None,
        include_layout: bool = True,
        sizing_mode: Optional[SizingModeType] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        layout_height: Optional[int] = None,
        share_x: Optional[bool] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        save: Union[bool, str, pathlib.Path, None] = None,
        curves: Optional[Dict[int, TaoCurveSettings]] = None,
        settings: Optional[TaoGraphSettings] = None,
    ) -> Tuple[List[AnyGraph], BokehAppCreator]:
        """
        Plot a graph with Bokeh.

        Parameters
        ----------
        template : str
            Graph template name.
        region_name : str, optional
            Graph region name.
        include_layout : bool
            Include a layout plot at the bottom, if not already placed and if
            appropriate (i.e., another plot uses longitudinal coordinates on
            the x-axis).
        sizing_mode : Optional[SizingModeType]
            Set the sizing mode for all graphs.  Default is configured on a
            per-graph basis, typically "inherit".
        width : int, optional
            Width of each plot.
        height : int, optional
            Height of each plot.
        layout_height : int, optional
            Height of the layout plot.
        share_x : bool or None, default=None
            Share x-axes where sensible (`None`) or force sharing x-axes (True)
            for all plots.
        xlim : (float, float), optional
            X axis limits.
        ylim : (float, float), optional
            Y axis limits.
        curves : Dict[int, TaoCurveSettings], optional
            Dictionary of curve index to curve settings. These settings will be
            applied to the placed graph prior to plotting.
        settings : TaoGraphSettings, optional
            Graph customization settings.
        save : str or bool, optional
            Save the plot to a static HTML file with the given name.
            If `True`, saves to a filename based on the plot title.

        Returns
        -------
        list of graphs

        BokehAppCreator
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

        app = BokehAppCreator(
            manager=self,
            graphs=graphs,
            share_x=share_x,
            include_variables=False,
            grid=None,
            width=width or _Defaults.width,
            height=height or _Defaults.height,
            layout_height=layout_height or _Defaults.layout_height,
            include_layout=include_layout,
            graph_sizing_mode=sizing_mode,
            xlim=[xlim],
            ylim=[ylim],
        )

        if save:
            if save is True:
                save = ""
            filename = app.save(save)
            logger.info(f"Saving plot to {filename!r}")

        return graphs, app

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
        width : int, optional
        height : int, optional
        save : pathlib.Path or str, optional
            Save the plot to the given filename.

        Returns
        -------
        ElementField

        figure
        """
        field = ElementField.from_tao(self.tao, ele_id, num_points=num_points, radius=radius)
        fig = figure(title=f"Field of {ele_id}")

        palette = colormap or _Defaults.palette

        source = _fields_to_data_source([field], x_scale=x_scale)
        cmap = bokeh.models.LinearColorMapper(
            palette=palette or _Defaults.palette,
            low=np.min(source.data["by"]),
            high=np.max(source.data["by"]),
        )

        fig.image(
            image="by",
            x="x",
            y=-1,
            dw="dw",
            dh="dh",
            color_mapper=cmap,
            source=source,
            name="field_images",
        )
        color_bar = bokeh.models.ColorBar(color_mapper=cmap, location=(0, 0))
        fig.add_layout(color_bar, "right")

        fig.frame_width = width or _Defaults.width
        fig.frame_height = height or _Defaults.height

        if save:
            if save is True:
                save = f"{ele_id}_field.html"
            if not pathlib.Path(save).suffix:
                save = f"{save}.html"
            filename = bokeh.io.save(fig, filename=save)
            logger.info(f"Saving plot to {filename!r}")

        return field, fig


class NotebookGraphManager(BokehGraphManager):
    """Jupyter notebook Bokeh backend graph manager."""

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
    ):
        """
        Plot graphs on a grid with Bokeh.

        Parameters
        ----------
        templates : list of str
            Graph template names.
        grid : (nrows, ncols), optional
            Grid the provided graphs into this many rows and columns.
        include_layout : bool, default=False
            Include a layout plot at the bottom of each column.
        share_x : bool or None, default=None
            Share x-axes where sensible (`None`) or force sharing x-axes (True)
            for all plots.
        vars : bool, default=False
            Show Tao variables as adjustable widgets, like "single mode".
        figsize : (int, int), optional
            Figure size. Alternative to specifying `width` and `height`
            separately.  This takes precedence over `width` and `height`.
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

        BokehAppCreator
        """
        graphs, app = super().plot_grid(
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
        if vars:
            app.variables = Variable.from_tao_all(self.tao)
        bokeh.plotting.show(app.create_full_app())
        return graphs, app

    def plot(
        self,
        template: str,
        *,
        region_name: Optional[str] = None,
        include_layout: bool = True,
        sizing_mode: Optional[SizingModeType] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        layout_height: Optional[int] = None,
        share_x: Optional[bool] = None,
        vars: bool = False,
        xlim: Optional[Limit] = None,
        ylim: Optional[Limit] = None,
        notebook_handle: bool = False,
        save: Union[bool, str, pathlib.Path, None] = None,
        curves: Optional[Dict[int, TaoCurveSettings]] = None,
        settings: Optional[TaoGraphSettings] = None,
    ) -> Tuple[List[AnyGraph], BokehAppCreator]:
        """
        Plot a graph with Bokeh.

        Parameters
        ----------
        template : str
            Graph template name.
        region_name : str, optional
            Graph region name.
        include_layout : bool
            Include a layout plot at the bottom, if not already placed and if
            appropriate (i.e., another plot uses longitudinal coordinates on
            the x-axis).
        sizing_mode : Optional[SizingModeType]
            Set the sizing mode for all graphs.  Default is configured on a
            per-graph basis, typically "inherit".
        width : int, optional
            Width of each plot.
        height : int, optional
            Height of each plot.
        layout_height : int, optional
            Height of the layout plot.
        share_x : bool or None, default=None
            Share x-axes where sensible (`None`) or force sharing x-axes (True)
            for all plots.
        vars : bool, default=False
            Show Tao variables as adjustable widgets, like "single mode".
        xlim : (float, float), optional
            X axis limits.
        ylim : (float, float), optional
            Y axis limits.
        curves : Dict[int, TaoCurveSettings], optional
            Dictionary of curve index to curve settings. These settings will be
            applied to the placed graph prior to plotting.
        settings : TaoGraphSettings, optional
            Graph customization settings.
        save : str or bool, optional
            Save the plot to a static HTML file with the given name.
            If `True`, saves to a filename based on the plot title.

        Returns
        -------
        BokehAppCreator
        """
        graphs, app = super().plot(
            region_name=region_name,
            template=template,
            include_layout=include_layout,
            sizing_mode=sizing_mode,
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

        if vars:
            app.variables = Variable.from_tao_all(self.tao)

        bokeh.plotting.show(
            app.create_full_app(),
            notebook_handle=notebook_handle,
        )
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
        width : int, optional
        height : int, optional
        save : pathlib.Path or str, optional
            Save the plot to the given filename.
        """
        field, fig = super().plot_field(
            ele_id,
            colormap=colormap,
            radius=radius,
            num_points=num_points,
            width=width,
            height=height,
            save=save,
            x_scale=x_scale,
        )
        bokeh.plotting.show(fig, notebook_handle=True)

        return field, fig


@functools.cache
def select_graph_manager_class():
    if util.is_jupyter():
        initialize_jupyter()
        return NotebookGraphManager
    return BokehGraphManager


def initialize_jupyter():
    # Is this public bokeh API? An attempt at forward-compatibility
    try:
        from bokeh.io.state import curstate
    except ImportError:
        pass
    else:
        state = curstate()
        if getattr(state, "notebook", False):
            # Jupyter already initialized
            logger.debug("Bokeh output_notebook already called; not re-initializing")
            return

    from bokeh.plotting import output_notebook

    output_notebook()
