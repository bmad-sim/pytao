from __future__ import annotations

import functools
import logging
import math
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
import bokeh.resources
import numpy as np
from bokeh.core.enums import SizingModeType
from bokeh.document.callbacks import EventCallback
from bokeh.models.sources import ColumnDataSource
from bokeh.plotting import figure
from pydantic.dataclasses import dataclass
from typing_extensions import NotRequired, TypedDict

from ..interface_commands import AnyPath
from . import pgplot, util
from .curves import TaoCurveSettings
from .fields import ElementField
from .plot import (
    BasicGraph,
    FloorPlanGraph,
    GraphBase,
    GraphManager,
    LatticeLayoutElement,
    LatticeLayoutGraph,
    PlotAnnotation,
    PlotCurve,
    PlotCurveLine,
    PlotCurveSymbols,
    PlotPatch,
    PlotPatchArc,
    PlotPatchCircle,
    PlotPatchEllipse,
    PlotPatchPolygon,
    PlotPatchRectangle,
    PlotPatchSbend,
    make_graph,
)
from .types import FloatVariableInfo

if typing.TYPE_CHECKING:
    from .. import Tao


logger = logging.getLogger(__name__)


def bokeh_color(color):
    color = color.lower().replace("_", "")
    return getattr(bokeh.colors.named, color, "black")


class CurveData(TypedDict):
    line: NotRequired[ColumnDataSource]
    symbol: NotRequired[ColumnDataSource]


class Defaults:
    graph_size: Tuple[int, int] = (600, 400)
    lattice_layout_size: Tuple[int, int] = (600, 150)
    floor_plan_size: Tuple[int, int] = (600, 600)
    palette: str = "Magma256"

    @classmethod
    def get_size_for_class(
        cls,
        typ: Type[AnyBokehGraph],
        user_width: Optional[int] = None,
        user_height: Optional[int] = None,
    ) -> Tuple[int, int]:
        default = {
            BokehBasicGraph: cls.graph_size,
            BokehLatticeLayoutGraph: cls.lattice_layout_size,
            BokehFloorPlanGraph: cls.floor_plan_size,
        }[typ]
        return (user_width or default[0], user_height or default[1])


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


def move_layout_to_bottom(bgraphs: List[AnyBokehGraph]) -> List[AnyBokehGraph]:
    result = []
    layout = None
    for bgraph in bgraphs:
        if isinstance(bgraph, BokehLatticeLayoutGraph):
            layout = bgraph
        else:
            result.append(bgraph)
    if layout is not None:
        result.append(layout)
    return result


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

    source.data.update(
        {
            "x": line.xs,
            "y": line.ys,
        }
    )
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


def _patch_rect_to_points(patch: PlotPatchRectangle) -> Tuple[List[float], List[float]]:
    points = patch.to_mpl().get_corners()
    return (
        points[:, 0].tolist() + [points[0, 0]],
        points[:, 1].tolist() + [points[0, 1]],
    )


def _draw_annotation(
    fig: figure,
    annotation: PlotAnnotation,
    color: str = "black,",
    source: Optional[ColumnDataSource] = None,
):
    if source is None:
        source = ColumnDataSource()

    source.data.update(
        {
            "x": [annotation.x],
            "y": [annotation.y],
            "text": [pgplot.mathjax_string(annotation.text)],
        }
    )
    return fig.text(
        "x",
        "y",
        angle=math.radians(annotation.rotation),
        text_align="right",  # annotation.horizontalalignment,
        text_baseline=annotation.verticalalignment,
        color=color,
        source=source,
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
    if isinstance(patch, (PlotPatchRectangle, PlotPatchPolygon)):
        if isinstance(patch, PlotPatchRectangle):
            source.data["xs"], source.data["ys"] = _patch_rect_to_points(patch)
        else:
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
        return fig.ellipse(
            x="x",
            y="y",
            width=[patch.width],
            height=[patch.height],
            angle=[math.radians(patch.angle)],
            line_width=line_width,
            fill_alpha=int(patch.fill),
            source=source,
        )
    if isinstance(patch, PlotPatchSbend):
        return _plot_sbend_patch(fig, patch)
    raise NotImplementedError(f"{type(patch).__name__}")


def _draw_layout_element(
    fig: figure,
    elem: LatticeLayoutElement,
    skip_labels: bool = True,
):
    color = bokeh_color(elem.color)
    base_data = {
        "s_start": [elem.info["ele_s_start"]],
        "s_end": [elem.info["ele_s_end"]],
        "name": [elem.info["label_name"]],
        "color": [color],
    }
    all_lines: List[Tuple[List[float], List[float]]] = []
    for patch in elem.patches:
        source = ColumnDataSource(data=dict(base_data))
        if isinstance(patch, PlotPatchRectangle):
            all_lines.append(_patch_rect_to_points(patch))
        elif isinstance(patch, PlotPatchPolygon):
            all_lines.append(
                (
                    [p[0] for p in patch.vertices + patch.vertices[:1]],
                    [p[1] for p in patch.vertices + patch.vertices[:1]],
                )
            )
        else:
            _plot_patch(fig, patch, line_width=elem.width, source=source)

    if elem.lines:
        for line_points in elem.lines:
            all_lines.append(
                (
                    [pt[0] for pt in line_points],
                    [pt[1] for pt in line_points],
                )
            )

    if all_lines:
        source = ColumnDataSource(
            data={
                "xs": [line[0] for line in all_lines],
                "ys": [line[1] for line in all_lines],
                "s_start": base_data["s_start"] * len(all_lines),
                "s_end": base_data["s_end"] * len(all_lines),
                "name": base_data["name"] * len(all_lines),
                "color": base_data["color"] * len(all_lines),
            }
        )
        fig.multi_line(
            xs="xs",
            ys="ys",
            line_width=elem.width,
            color=color,
            source=source,
        )

    for annotation in elem.annotations:
        if annotation.text == elem.info["label_name"] and skip_labels:
            continue
        _draw_annotation(
            fig,
            annotation,
            color=color,
            source=ColumnDataSource(data=dict(base_data)),
        )


def _fields_to_data_source(
    fields: List[ElementField],
    x_scale: float = 1e3,
):
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
    width: int
    height: int
    aspect_ratio: float
    x_range: Optional[bokeh.models.Range]
    y_range: Optional[bokeh.models.Range]

    def __init__(
        self,
        manager: GraphManager,
        graph: TGraph,
        sizing_mode: SizingModeType,
        width: int,
        height: int,
        aspect_ratio: float,  # w/h
    ) -> None:
        self.graph = graph
        self.manager = manager
        self.sizing_mode = sizing_mode
        self.width = width
        self.height = height
        self.aspect_ratio = aspect_ratio
        self.x_range = None
        self.y_range = None

    def get_graph_info(self) -> TGraph:
        return self.graph

    def to_html(
        self,
        title: Optional[str] = None,
    ) -> str:
        return bokeh.embed.file_html(models=[self.create_figure()], title=title)

    def save(
        self,
        filename: AnyPath,
        title: Optional[str] = None,
    ):
        source = self.to_html(title=title)
        with open(filename, "wt") as fp:
            fp.write(source)

    def create_full_app(self):
        def bokeh_app(doc):
            primary_figure = self.create_figure()
            doc.add_root(primary_figure)
            for model in self.create_widgets(primary_figure):
                doc.add_root(model)

        return bokeh_app

    __call__ = create_full_app

    @abstractmethod
    def create_figure(
        self,
        *,
        tools: str = "pan,wheel_zoom,box_zoom,save,reset,help,crosshair",
        toolbar_location: str = "above",
    ) -> figure:
        raise NotImplementedError()

    @abstractmethod
    def create_widgets(self, fig: figure) -> List[bokeh.models.UIElement]:
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
        aspect_ratio: float = 4.0,  # w/h
    ) -> None:
        super().__init__(
            manager=manager,
            graph=graph,
            sizing_mode=sizing_mode,
            width=width or Defaults.lattice_layout_size[0],
            height=height or Defaults.lattice_layout_size[1],
            aspect_ratio=aspect_ratio,
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
        tools: str = "pan,wheel_zoom,box_zoom,save,reset,help,crosshair",
        toolbar_location: str = "above",
        add_named_hover_tool: bool = True,
    ) -> figure:
        graph = self.graph
        fig = figure(
            title=pgplot.mathjax_string(graph.title),
            x_axis_label=pgplot.mathjax_string(graph.xlabel),
            # y_axis_label=pgplot.mathjax_string(graph.ylabel),
            toolbar_location=toolbar_location,
            tools=tools,
            aspect_ratio=self.aspect_ratio,
            sizing_mode=self.sizing_mode,
            width=self.width,
            height=self.height,
        )
        if add_named_hover_tool:
            hover = bokeh.models.HoverTool(
                tooltips=[
                    ("name", "@name"),
                    ("s start [m]", "@s_start"),
                    ("s end [m]", "@s_end"),
                ]
            )

            fig.add_tools(hover)

        box_zoom = get_tool_from_figure(fig, bokeh.models.BoxZoomTool)
        if box_zoom is not None:
            box_zoom.match_aspect = True

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

        for elem in self.graph.elements:
            _draw_layout_element(fig, elem, skip_labels=True)

        if self.x_range is not None:
            fig.x_range = self.x_range
        if self.y_range is not None:
            fig.y_range = self.y_range
        return fig

    def create_widgets(self, fig: figure) -> List[bokeh.models.UIElement]:
        return []


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
        aspect_ratio: float = 1.5,  # w/h
        variables: Optional[List[Variable]] = None,
    ) -> None:
        super().__init__(
            manager=manager,
            graph=graph,
            sizing_mode=sizing_mode,
            width=width or Defaults.graph_size[0],
            height=height or Defaults.graph_size[1],
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
        tao: Optional[Tao] = None,
    ) -> None:
        try:
            self.tao.cmd("set global lattice_calc_on = F")
            self.tao.cmd(f"set plot {self.graph.region_name} n_curve_pts = {self.num_points}")
            self.tao.cmd(
                f"x_scale {self.graph.region_name} {self.view_x_range[0]} {self.view_x_range[1]}"
            )
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
        tools: str = "pan,wheel_zoom,box_zoom,save,reset,help,hover,crosshair",
        toolbar_location: str = "above",
        sizing_mode: SizingModeType = "inherit",
    ) -> figure:
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
        )
        if self.x_range is not None:
            fig.x_range = self.x_range
        if self.y_range is not None:
            fig.y_range = self.y_range
        for curve, source in zip(graph.curves, self.curve_data):
            _plot_curve(fig, curve, source)
        return fig

    def create_widgets(self, fig: figure) -> List[bokeh.models.UIElement]:
        update_button = bokeh.models.Button(label="Update")
        num_points_slider = bokeh.models.Slider(
            title="Data Points",
            start=10,
            end=10_000,
            step=1_000,
            value=401,
        )

        def update_plot():
            self.update_plot(fig, widgets=[update_button, num_points_slider])

        def ranges_update(event: bokeh.events.RangesUpdate) -> None:
            new_xrange = self.graph.clamp_x_range(event.x0, event.x1)
            if new_xrange != self.view_x_range:
                self.view_x_range = new_xrange

            try:
                update_plot()
            except Exception:
                logger.exception("Failed to update number ranges")

        def num_points_changed(_attr, _old, num_points: int):
            self.num_points = num_points
            try:
                update_plot()
            except Exception:
                logger.exception("Failed to update number of points")

        num_points_slider.on_change("value", num_points_changed)
        update_button.on_click(update_plot)
        fig.on_event(bokeh.events.RangesUpdate, cast(EventCallback, ranges_update))

        models: List[bokeh.models.UIElement] = [
            bokeh.layouts.column(bokeh.layouts.row(update_button, num_points_slider), fig)
        ]

        return models


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
        aspect_ratio: float = 1.0,  # w/h
    ) -> None:
        super().__init__(
            manager=manager,
            graph=graph,
            sizing_mode=sizing_mode,
            width=width or Defaults.floor_plan_size[0],
            height=height or Defaults.floor_plan_size[1],
            aspect_ratio=aspect_ratio,
        )

    @property
    def tao(self) -> Tao:
        return self.manager.tao

    def create_figure(
        self,
        *,
        tools: str = "pan,wheel_zoom,box_zoom,save,reset,help,hover,crosshair",
        toolbar_location: str = "above",
        sizing_mode: SizingModeType = "inherit",
    ) -> figure:
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
        if self.x_range is not None:
            fig.x_range = self.x_range
        if self.y_range is not None:
            fig.y_range = self.y_range

        box_zoom = get_tool_from_figure(fig, bokeh.models.BoxZoomTool)
        if box_zoom is not None:
            box_zoom.match_aspect = True

        for line in self.graph.building_walls.lines:
            _plot_curve_line(fig, line)
        for patch in self.graph.building_walls.patches:
            _plot_patch(fig, patch)
        orbits = self.graph.floor_orbits
        if orbits is not None:
            _plot_curve_symbols(fig, orbits.curve, name="floor_orbits")
        for elem in self.graph.elements:
            for line in elem.lines:
                _plot_curve_line(fig, line)
            for patch in elem.patches:
                _plot_patch(fig, patch, line_width=elem.info["line_width"])
            for annotation in elem.annotations:
                _draw_annotation(
                    fig,
                    annotation,
                    color=bokeh_color(elem.info["color"]),
                    source=ColumnDataSource(data={"name": [elem.info["label_name"]]}),
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

        controls_row = [bokeh.layouts.row(controls)] if controls else []
        return [bokeh.layouts.column([*controls_row, fig])]


AnyBokehGraph = Union[BokehBasicGraph, BokehLatticeLayoutGraph, BokehFloorPlanGraph]


OptionalLimit = Optional[Tuple[float, float]]
CurveIndexToCurve = Dict[int, TaoCurveSettings]
UIGridLayoutList = List[Optional[bokeh.models.UIElement]]


class BokehApp:
    """
    A composite Bokeh application made up of 1 or more graphs.

    This can be used to:
    * Generate a static HTML page without Python widgets
    * Generate a Notebook (or standalone) application with Python widgets

    Interactive widgets will use the `Tao` object to adjust variables during
    callbacks resulting from user interaction.
    """

    manager: Union[BokehGraphManager, NotebookGraphManager]
    bgraphs: List[AnyBokehGraph]
    share_x: Optional[bool]
    variables: List[Variable]
    grid: Tuple[int, int]
    graph_width: Optional[int]
    graph_height: Optional[int]
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
        bgraphs: List[AnyBokehGraph],
        share_x: Optional[bool] = None,
        include_variables: bool = False,
        grid: Optional[Tuple[int, int]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        include_layout: bool = False,
        graph_width: Optional[int] = None,
        graph_height: Optional[int] = None,
        graph_sizing_mode: Optional[SizingModeType] = None,
        layout_height: Optional[int] = None,
        xlim: Optional[List[OptionalLimit]] = None,
        ylim: Optional[List[OptionalLimit]] = None,
    ) -> None:
        assert len(bgraphs)

        xlim = list(xlim or [None])
        if len(xlim) < len(bgraphs):
            xlim.extend([xlim[-1]] * (len(bgraphs) - len(xlim)))

        ylim = list(ylim or [None])
        if len(ylim) < len(bgraphs):
            ylim.extend([ylim[-1]] * (len(bgraphs) - len(ylim)))

        if len(bgraphs) == 1 and isinstance(bgraphs[0], BokehLatticeLayoutGraph):
            include_layout = False

        if not grid:
            grid = (len(bgraphs), 1)

        if include_layout:
            grid = (grid[0] + 1, grid[1])

        if include_variables:
            variables = Variable.from_tao_all(manager.tao)
        else:
            variables = []

        self.manager = manager
        self.share_x = share_x
        self.variables = variables
        self.grid = grid
        self.width = width
        self.height = height
        self.graph_width = graph_width
        self.graph_height = graph_height
        self.graph_sizing_mode = graph_sizing_mode
        self.include_layout = include_layout
        self.layout_height = layout_height
        self.xlim = xlim
        self.ylim = ylim
        self.create_figures(bgraphs)

    def create_figures(self, bgraphs: List[AnyBokehGraph]) -> None:
        self.bgraphs = bgraphs
        self.figures = [bgraph.create_figure() for bgraph in bgraphs]
        self.pairs = [
            BGraphAndFigure(bgraph=bgraph, fig=fig)
            for bgraph, fig in zip(bgraphs, self.figures)
        ]
        if len(self.figures) <= 1:
            return

        if self.share_x is None:
            share_common_x_axes(self.pairs)
        elif self.share_x:
            share_x_axes(self.figures)

    def to_html(
        self,
        title: Optional[str] = None,
    ) -> str:
        layout = bokeh.layouts.layout(
            children=self._grid_figures(),
            width=self.width,
            height=self.height,
        )
        return bokeh.embed.file_html(models=layout, title=title)

    def save(
        self,
        filename: AnyPath = "",
        *,
        title: Optional[str] = None,
    ) -> Optional[pathlib.Path]:
        if not self.bgraphs:
            return

        title = self.bgraphs[0].graph.title or f"plot-{time.time()}"
        if not filename:
            filename = f"{title}.html"
        if not pathlib.Path(filename).suffix:
            filename = f"{filename}.html"
        source = self.to_html(title=title)
        with open(filename, "wt") as fp:
            fp.write(source)
        return pathlib.Path(filename)

    def _grid_figures(self) -> List[UIGridLayoutList]:
        nrows, ncols = self.grid
        rows = [[] for _ in range(nrows)]
        rows_cols = [(row, col) for row in range(nrows) for col in range(ncols)]

        for pair, xl, yl, (row, _col) in zip(self.pairs, self.xlim, self.ylim, rows_cols):
            fig = pair.fig
            bgraph = pair.bgraph
            if xl is not None:
                fig.x_range = bokeh.models.Range1d(*xl)
            if yl is not None:
                fig.y_range = bokeh.models.Range1d(*yl)

            is_layout = isinstance(bgraph, BokehLatticeLayoutGraph)

            if self.graph_sizing_mode is not None:
                fig.sizing_mode = self.graph_sizing_mode

            width, height = Defaults.get_size_for_class(
                type(bgraph),
                user_width=self.graph_width,
                user_height=self.layout_height if is_layout else self.graph_height,
            )

            fig.width = width
            fig.height = height

            rows[row].append(fig)
        return [row for row in rows if row]

    @property
    def nrows(self) -> int:
        return self.grid[0]

    @property
    def ncols(self) -> int:
        return self.grid[1]

    def _add_update_button(self):
        update_button = bokeh.models.Button(label="Update")

        def update_plot():
            for pair in self.pairs:
                bgraph = pair.bgraph
                if not isinstance(bgraph, BokehBasicGraph):
                    continue

                try:
                    bgraph.update_plot(pair.fig, widgets=[update_button])
                except Exception:
                    logger.exception("Failed to update number of points")

        update_button.on_click(update_plot)
        return update_button

    def _add_num_points_slider(self):
        num_points_slider = bokeh.models.Slider(
            title="Data Points",
            start=10,
            end=10_000,
            step=1_000,
            value=401,
        )

        def num_points_changed(_attr, _old, num_points: int):
            for pair in self.pairs:
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

    def _monitor_range_updates(self):
        def ranges_update(
            bgraph: BokehBasicGraph, fig: figure, event: bokeh.events.RangesUpdate
        ) -> None:
            new_xrange = bgraph.graph.clamp_x_range(event.x0, event.x1)
            if new_xrange != bgraph.view_x_range:
                bgraph.view_x_range = new_xrange

            try:
                bgraph.update_plot(fig, widgets=[])
            except Exception:
                logger.exception("Failed to update number ranges")

        for pair in self.pairs:
            if not isinstance(pair.bgraph, BokehBasicGraph):
                continue

            pair.fig.on_event(
                bokeh.events.RangesUpdate,
                cast(EventCallback, functools.partial(ranges_update, pair.bgraph, pair.fig)),
            )

    def create_ui(self):
        if not self.bgraphs:
            return

        widget_models: List[bokeh.layouts.UIElement] = []
        if self.variables:
            status_label = bokeh.models.PreText()
            spinners = _handle_variables(
                tao=self.manager.tao,
                variables=self.variables,
                status_label=status_label,
                pairs=self.pairs,
            )
            widget_models.insert(0, bokeh.layouts.row([status_label]))
            per_row = 6
            while spinners:
                row = bokeh.layouts.row(spinners[-per_row:])
                spinners = spinners[:-per_row]
                widget_models.insert(0, row)

        if any(isinstance(bgraph, BokehBasicGraph) for bgraph in self.bgraphs):
            update_button = self._add_update_button()
            num_points_slider = self._add_num_points_slider()
            widget_models.insert(0, bokeh.layouts.row([update_button, num_points_slider]))

            self._monitor_range_updates()

        rows = self._grid_figures()
        if self.include_layout:
            lattice_layout = self.manager.lattice_layout_graph
            lattice_layout.width, lattice_layout.height = Defaults.get_size_for_class(
                type(lattice_layout),
                user_width=self.graph_width,
                user_height=self.layout_height,
            )

            layout_figs: UIGridLayoutList = [
                lattice_layout.create_figure() for _ in range(self.ncols)
            ]
            rows.append(layout_figs)

            for fig in layout_figs:
                if fig is not None:
                    fig.min_border_bottom = 40

        for row in rows:
            for fig in row:
                # NOTE: this value is somewhat arbitrary; it helps align the X axes
                # between consecutive plots
                if fig is not None:
                    fig.min_border_left = 80

        all_elems: List[bokeh.models.UIElement] = [
            *widget_models,
            bokeh.layouts.gridplot(
                children=rows,
                width=self.width,
                height=self.height,
            ),
        ]
        return bokeh.layouts.column(all_elems)

    def create_full_app(self):
        def bokeh_app(doc):
            doc.add_root(self.create_ui())

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

    def create_spinner(self) -> bokeh.models.Spinner:
        return bokeh.models.Spinner(
            title=self.name,
            value=self.value,
            step=self.step,
            low=self.info["low_lim"],
            high=self.info["high_lim"],
        )

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


def _clean_tao_exception_for_user(text: str, command: str) -> str:
    def clean_line(line: str) -> str:
        # "[ERROR | 2024-JUL-22 09:20:20] tao_set_invalid:"
        if line.startswith("[") and line.endswith(f"{command}:"):
            return line.split(f"{command}:", 1)[1]
        return line

    text = text.replace("ERROR detected: ", "\n")
    lines = [clean_line(line.rstrip()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line.strip())


def _handle_variables(
    tao: Tao,
    variables: List[Variable],
    status_label: bokeh.models.PreText,
    pairs: List[BGraphAndFigure],
) -> List[bokeh.models.UIElement]:
    def variable_updated(attr: str, old: float, new: float, *, var: Variable):
        try:
            var.set_value(tao, new)
        except RuntimeError as ex:
            status_label.text = _clean_tao_exception_for_user(
                str(ex),
                command="tao_set_invalid",
            )
        else:
            status_label.text = ""

        for pair in pairs:
            if isinstance(pair.bgraph, (BokehBasicGraph, BokehLatticeLayoutGraph)):
                pair.bgraph.update_plot(pair.fig, tao=tao)

    spinners = []
    for var in variables:
        spinner = var.create_spinner()
        spinners.append(spinner)
        spinner.on_change("value", functools.partial(variable_updated, var=var))
    return spinners


class BokehGraphManager(
    GraphManager[AnyBokehGraph, BokehLatticeLayoutGraph, BokehFloorPlanGraph]
):
    """Bokeh backend graph manager - for non-Jupyter contexts."""

    _key_: ClassVar[str] = "bokeh"
    _lattice_layout_graph_type = BokehLatticeLayoutGraph
    _floor_plan_graph_type = BokehFloorPlanGraph

    def make_graph(self, region_name: str, graph_name: str) -> AnyBokehGraph:
        graph = make_graph(self.tao, region_name, graph_name)
        if isinstance(graph, BasicGraph):
            return BokehBasicGraph(self, graph)
        elif isinstance(graph, LatticeLayoutGraph):
            return BokehLatticeLayoutGraph(self, graph)
        elif isinstance(graph, FloorPlanGraph):
            return BokehFloorPlanGraph(self, graph)
        raise NotImplementedError(type(graph).__name__)

    def plot_grid(
        self,
        graph_names: List[str],
        grid: Tuple[int, int],
        *,
        include_layout: bool = False,
        share_x: Optional[bool] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        figsize: Optional[Tuple[int, int]] = None,
        layout_height: Optional[int] = None,
        xlim: Optional[List[OptionalLimit]] = None,
        ylim: Optional[List[OptionalLimit]] = None,
        reuse: bool = True,
        curves: Optional[List[Optional[CurveIndexToCurve]]] = None,
        save: Union[bool, str, pathlib.Path, None] = None,
    ) -> BokehApp:
        """
        Plot graphs on a grid with Bokeh.

        Parameters
        ----------
        graph_names : list of str
            Graph names.
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
        reuse : bool, default=True
            If an existing plot of the given template type exists, reuse the
            existing plot region rather than selecting a new empty region.
        curves : list of Dict[int, TaoCurveSettings], optional
            One dictionary per graph, with each dictionary mapping the curve
            index to curve settings. These settings will be applied to the
            placed graphs prior to plotting.
        save : pathlib.Path or str, optional
            Save the plot to the given filename.

        Returns
        -------
        BokehApp
        """
        if len(set(graph_names)) < len(graph_names):
            # Don't reuse existing regions if we place the same template more
            # than once
            reuse = False

        if not curves:
            curves = [None] * len(graph_names)
        elif len(curves) < len(graph_names):
            assert len(curves)
            curves = list(curves) + [None] * (len(graph_names) - len(curves))

        bgraphs = sum(
            (
                self.prepare_graphs_by_name(
                    graph_name=graph_name,
                    reuse=reuse,
                    curves=graph_curves,
                )
                for graph_name, graph_curves in zip(graph_names, curves or [])
            ),
            [],
        )

        if not bgraphs:
            raise ValueError(f"No supported plots from these templates: {graph_names}")

        if figsize is not None:
            width, height = figsize

        app = BokehApp(
            manager=self,
            bgraphs=bgraphs,
            share_x=share_x,
            include_variables=False,
            grid=grid,
            width=width,
            height=height,
            include_layout=include_layout,
            xlim=xlim,
            ylim=ylim,
            layout_height=layout_height,
        )

        if save:
            if save is True:
                save = ""
            filename = app.save(save)
            logger.info(f"Saving plot to {filename!r}")
        return app

    def plot(
        self,
        graph_name: str,
        *,
        region_name: Optional[str] = None,
        include_layout: bool = True,
        sizing_mode: Optional[SizingModeType] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        layout_height: Optional[int] = None,
        share_x: Optional[bool] = None,
        reuse: bool = True,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        save: Union[bool, str, pathlib.Path, None] = None,
        curves: Optional[Dict[int, TaoCurveSettings]] = None,
    ):
        """
        Plot a graph with Bokeh.

        Parameters
        ----------
        graph_name : str
            Graph template name.
        region_name : str, optional
            Graph region name.
        include_layout : bool
            Include a layout plot at the bottom, if not already placed and if
            appropriate (i.e., another plot uses longitudinal coordinates on
            the x-axis).
        update : bool, default=True
            Query Tao to update relevant graphs prior to plotting.
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
        reuse : bool, default=True
            If an existing plot of the given template type exists, reuse the
            existing plot region rather than selecting a new empty region.
        xlim : (float, float), optional
            X axis limits.
        ylim : (float, float), optional
            Y axis limits.
        curves : Dict[int, TaoCurveSettings], optional
            Dictionary of curve index to curve settings. These settings will be
            applied to the placed graph prior to plotting.
        save : str or bool, optional
            Save the plot to a static HTML file with the given name.
            If `True`, saves to a filename based on the plot title.

        Returns
        -------
        BokehApp
        """
        bgraphs = self.prepare_graphs_by_name(
            graph_name=graph_name,
            region_name=region_name,
            reuse=reuse,
            curves=curves,
        )
        if not bgraphs:
            return None

        app = BokehApp(
            manager=self,
            bgraphs=bgraphs,
            share_x=share_x,
            include_variables=False,
            grid=None,
            graph_width=width,
            graph_height=height,
            include_layout=include_layout,
            graph_sizing_mode=sizing_mode,
            layout_height=layout_height,
            xlim=[xlim],
            ylim=[ylim],
        )

        if save:
            if save is True:
                save = ""
            filename = app.save(save)
            logger.info(f"Saving plot to {filename!r}")

        return app

    def plot_field(
        self,
        ele_id: str,
        *,
        colormap: Optional[str] = None,
        radius: float = 0.015,
        num_points: int = 100,
        x_scale: float = 1e3,
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
        """
        field = ElementField.from_tao(self.tao, ele_id, num_points=num_points, radius=radius)
        fig = figure(title=f"Field of {ele_id}")

        palette = colormap or Defaults.palette

        source = _fields_to_data_source([field], x_scale=x_scale)
        cmap = bokeh.models.LinearColorMapper(
            palette=palette or Defaults.palette,
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

        if width is not None:
            fig.width = width
        if height is not None:
            fig.height = height

        if save:
            if save is True:
                save = f"{ele_id}_field.html"
            if not pathlib.Path(save).suffix:
                save = f"{save}.html"
            filename = bokeh.io.save(fig, filename=save)
            logger.info(f"Saving plot to {filename!r}")

        return fig


class NotebookGraphManager(BokehGraphManager):
    """Jupyter notebook Bokeh backend graph manager."""

    def plot_grid(
        self,
        graph_names: List[str],
        grid: Tuple[int, int],
        *,
        curves: Optional[List[Dict[int, TaoCurveSettings]]] = None,
        include_layout: bool = False,
        share_x: Optional[bool] = None,
        figsize: Optional[Tuple[int, int]] = None,
        layout_height: Optional[int] = None,
        xlim: Optional[List[Optional[Tuple[float, float]]]] = None,
        ylim: Optional[List[Optional[Tuple[float, float]]]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        reuse: bool = True,
        save: Union[bool, str, pathlib.Path, None] = None,
    ):
        """
        Plot graphs on a grid with Bokeh.

        Parameters
        ----------
        graph_names : list of str
            Graph names.
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
        reuse : bool, default=True
            If an existing plot of the given template type exists, reuse the
            existing plot region rather than selecting a new empty region.
        curves : list of Dict[int, TaoCurveSettings], optional
            One dictionary per graph, with each dictionary mapping the curve
            index to curve settings. These settings will be applied to the
            placed graphs prior to plotting.
        save : pathlib.Path or str, optional
            Save the plot to the given filename.

        Returns
        -------
        BokehApp
        """
        kwargs = {
            "graph_names": graph_names,
            "grid": grid,
            "curves": curves,
            "include_layout": include_layout,
            "share_x": share_x,
            "figsize": figsize,
            "width": width,
            "height": height,
            "reuse": reuse,
            "xlim": xlim,
            "ylim": ylim,
            "layout_height": layout_height,
        }
        if save:
            # NOTE/TODO: seems like we have to regenerate the bokeh glyphs
            # or they may be considered as owned by more than one document.
            # This is a workaround, but I assume there's a better way of
            # going about this...
            app = super().plot_grid(**kwargs, save=save)

        app = super().plot_grid(**kwargs, save=False)
        bokeh.plotting.show(app.create_full_app())
        return app

    def plot(
        self,
        graph_name: str,
        *,
        region_name: Optional[str] = None,
        include_layout: bool = True,
        sizing_mode: Optional[SizingModeType] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        layout_height: Optional[int] = None,
        share_x: Optional[bool] = None,
        vars: bool = False,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        notebook_handle: bool = False,
        reuse: bool = True,
        save: Union[bool, str, pathlib.Path, None] = None,
        curves: Optional[Dict[int, TaoCurveSettings]] = None,
    ):
        """
        Plot a graph with Bokeh.

        Parameters
        ----------
        graph_name : str
            Graph template name.
        region_name : str, optional
            Graph region name.
        include_layout : bool
            Include a layout plot at the bottom, if not already placed and if
            appropriate (i.e., another plot uses longitudinal coordinates on
            the x-axis).
        update : bool, default=True
            Query Tao to update relevant graphs prior to plotting.
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
        reuse : bool, default=True
            If an existing plot of the given template type exists, reuse the
            existing plot region rather than selecting a new empty region.
        xlim : (float, float), optional
            X axis limits.
        ylim : (float, float), optional
            Y axis limits.
        curves : Dict[int, TaoCurveSettings], optional
            Dictionary of curve index to curve settings. These settings will be
            applied to the placed graph prior to plotting.
        save : str or bool, optional
            Save the plot to a static HTML file with the given name.
            If `True`, saves to a filename based on the plot title.

        Returns
        -------
        BokehApp
        """
        kwargs = {
            "region_name": region_name,
            "graph_name": graph_name,
            "include_layout": include_layout,
            "sizing_mode": sizing_mode,
            "width": width,
            "height": height,
            "layout_height": layout_height,
            "reuse": reuse,
            "xlim": xlim,
            "ylim": ylim,
            "curves": curves,
            "share_x": share_x,
        }
        if save:
            # NOTE/TODO: seems like we have to regenerate the bokeh glyphs
            # or they may be considered as owned by more than one document.
            # This is a workaround, but I assume there's a better way of
            # going about this...
            app = super().plot(**kwargs, save=save)

        app = super().plot(**kwargs, save=False)
        if not app:
            return

        if vars:
            app.variables = Variable.from_tao_all(self.tao)

        bokeh.plotting.show(
            app.create_full_app(),
            notebook_handle=notebook_handle,
        )
        return app

    __call__ = plot

    def plot_field(
        self,
        ele_id: str,
        *,
        colormap: Optional[str] = None,
        radius: float = 0.015,
        num_points: int = 100,
        width: Optional[int] = None,
        height: Optional[int] = None,
        x_scale: float = 1e3,
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
        fig = super().plot_field(
            ele_id,
            colormap=colormap,
            radius=radius,
            num_points=num_points,
            width=width,
            height=height,
            save=save,
        )
        bokeh.plotting.show(fig, notebook_handle=True)

        return fig


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
