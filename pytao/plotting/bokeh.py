from __future__ import annotations

import functools
import logging
import math
import pathlib
import time
import typing
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
import numpy as np
from bokeh.core.enums import SizingModeType
from bokeh.document.callbacks import EventCallback
from bokeh.models.sources import ColumnDataSource
from bokeh.plotting import figure
from pydantic.dataclasses import dataclass
from typing_extensions import NotRequired, TypedDict, override

from ..interface_commands import AnyPath
from .fields import LatticeLayoutField
from .curves import TaoCurveSettings

from . import pgplot, util
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
    graphs: List[BGraphAndFigure],
    crosshairs: bool = True,
) -> List[List[BGraphAndFigure]]:
    res: List[List[BGraphAndFigure]] = []

    s_plots = []
    for item in graphs:
        if item.bgraph.graph.is_s_plot:
            s_plots.append(item)

    if s_plots:
        res.append(s_plots)

    by_xlabel: Dict[str, List[BGraphAndFigure]] = {}
    for item in graphs:
        if item in s_plots:
            continue
        by_xlabel.setdefault(item.bgraph.graph.xlabel, []).append(item)

    for sharing_set in by_xlabel.values():
        if len(sharing_set) > 1:
            res.append(sharing_set)

    for sharing_set in res:
        figs = [item.fig for item in sharing_set]
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
    fields: List[LatticeLayoutField],
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


def _draw_fields(
    fig: figure,
    fields: List[LatticeLayoutField],
    palette: str = "Magma256",
):
    source = _fields_to_data_source(fields)
    cmap = bokeh.models.LinearColorMapper(
        palette="Magma256",
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


TGraph = TypeVar("TGraph", bound=GraphBase)


class BokehGraphBase(Generic[TGraph]):
    manager: GraphManager
    graph: TGraph
    sizing_mode: SizingModeType
    width: int
    height: int
    aspect_ratio: float
    x_range: Optional[bokeh.models.Range]

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

    def create_figure(
        self,
        tools: str = "pan,wheel_zoom,box_zoom,save,reset,help,crosshair",
        toolbar_location: str = "above",
    ) -> figure:
        raise NotImplementedError()

    def create_app_figure(self) -> Tuple[figure, List[bokeh.models.UIElement]]:
        raise NotImplementedError()

    def create_app(self):
        def bokeh_app(doc):
            _primary_figure, models = self.create_app_figure()
            for model in models:
                doc.add_root(model)

        return bokeh_app

    __call__ = create_app


class BokehLatticeLayoutGraph(BokehGraphBase[LatticeLayoutGraph]):
    graph_type: ClassVar[str] = "lat_layout"
    graph: LatticeLayoutGraph
    show_fields: bool

    def __init__(
        self,
        manager: GraphManager,
        graph: LatticeLayoutGraph,
        sizing_mode: SizingModeType = "inherit",
        width: int = 900,
        height: int = 300,
        aspect_ratio: float = 3.0,  # w/h
        show_fields: bool = False,
    ) -> None:
        super().__init__(
            manager=manager,
            graph=graph,
            sizing_mode=sizing_mode,
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
        )
        self.show_fields = show_fields

    def update_plot(
        self,
        fig: figure,
        *,
        widgets: Optional[List[bokeh.models.Widget]] = None,
        tao: Optional[Tao] = None,
    ) -> None:
        if tao is None:
            return

        have_fields = len(self.graph.fields)
        # graph = make_graph(tao, self.graph.region_name, self.graph.graph_name)
        # assert isinstance(graph, LatticeLayoutGraph)
        # self.graph = graph

        if not have_fields:
            return

        self.graph.update_fields(tao)
        field_images_glyph = fig.select("field_images")
        field_images = field_images_glyph.data_source
        updated_source = _fields_to_data_source(self.graph.fields)
        field_images.data = dict(updated_source.data)

    @override
    def create_figure(
        self,
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

        if self.graph.show_fields:
            if not self.graph.fields:
                self.graph.update_fields(self.manager.tao)
            _draw_fields(fig, self.graph.fields)
        for elem in self.graph.elements:
            _draw_layout_element(fig, elem, skip_labels=True)

        if self.x_range is not None:
            fig.x_range = self.x_range
        return fig

    def create_app_figure(
        self,
        include_variables: bool = False,
    ) -> Tuple[figure, List[bokeh.models.UIElement]]:
        fig = self.create_figure()
        return fig, [fig]


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
        width: int = 900,
        height: int = 600,
        aspect_ratio: float = 1.5,  # w/h
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

    @override
    def create_figure(
        self,
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
        for curve, source in zip(graph.curves, self.curve_data):
            _plot_curve(fig, curve, source)
        return fig

    def create_app_figure(
        self,
        include_variables: bool = True,
    ) -> Tuple[figure, List[bokeh.models.UIElement]]:
        fig = self.create_figure()
        update_button = bokeh.models.Button(label="Update")  # , icon="reload")
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

        if include_variables and self.variables:
            status_label = bokeh.models.PreText()
            spinners = _handle_variables(
                self.tao,
                self.variables,
                status_label,
                [BGraphAndFigure(bgraph=self, fig=fig)],
            )
            models.insert(0, bokeh.layouts.row(spinners))
            models.insert(1, bokeh.layouts.row([status_label]))
        return fig, models


class BokehFloorPlanGraph(BokehGraphBase[FloorPlanGraph]):
    graph_type: ClassVar[str] = "floor_plan"
    graph: FloorPlanGraph

    def __init__(
        self,
        manager: GraphManager,
        graph: FloorPlanGraph,
        sizing_mode: SizingModeType = "inherit",
        width: int = 600,
        height: int = 600,
        aspect_ratio: float = 1.0,  # w/h
    ) -> None:
        super().__init__(
            manager=manager,
            graph=graph,
            sizing_mode=sizing_mode,
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
        )

    @property
    def tao(self) -> Tao:
        return self.manager.tao

    @override
    def create_figure(
        self,
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

    def create_app_figure(
        self,
        include_variables: bool = False,
    ) -> Tuple[figure, List[bokeh.models.UIElement]]:
        fig = self.create_figure()

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
        return fig, [bokeh.layouts.column([*controls_row, fig])]


AnyBokehGraph = Union[BokehBasicGraph, BokehLatticeLayoutGraph, BokehFloorPlanGraph]


class CompositeApp:
    tao: Tao
    bgraphs: List[AnyBokehGraph]
    share_x: Optional[bool]
    variables: List[Variable]

    def __init__(
        self,
        tao: Tao,
        bgraphs: List[AnyBokehGraph],
        share_x: Optional[bool] = None,
        variables: Optional[List[Variable]] = None,
    ) -> None:
        self.tao = tao
        self.bgraphs = bgraphs
        self.share_x = share_x
        self.variables = variables or []

    def to_html(
        self,
        title: Optional[str] = None,
    ) -> str:
        items = [
            BGraphAndFigure(fig=bgraph.create_figure(), bgraph=bgraph)
            for bgraph in self.bgraphs
        ]
        figures = [item.fig for item in items]
        share_common_x_axes(items)
        return bokeh.embed.file_html(models=figures, title=title)

    def save(
        self,
        filename: AnyPath,
        title: Optional[str] = None,
    ):
        source = self.to_html(title=title)
        with open(filename, "wt") as fp:
            fp.write(source)

    def create_ui(self):
        if not self.bgraphs:
            return

        items: List[BGraphAndFigure] = []
        models: List[bokeh.models.UIElement] = []
        for bgraph in self.bgraphs:
            primary_figure, fig_models = bgraph.create_app_figure(include_variables=False)
            items.append(BGraphAndFigure(bgraph, primary_figure))
            models.extend(fig_models)

        for item in items:
            # NOTE: this value is somewhat arbitrary; it helps align the X axes
            # between consecutive plots
            item.fig.min_border_left = 80
        # if isinstance(items[-1], BokehLatticeLayoutGraph):
        #     items[-1].fig.min_border_bottom = 80

        if self.share_x is None:
            share_common_x_axes(items)
        elif self.share_x:
            share_x_axes([item.fig for item in items])

        if self.variables:
            status_label = bokeh.models.PreText()
            spinners = _handle_variables(self.tao, self.variables, status_label, items)
            models.insert(0, bokeh.layouts.row(spinners))
            models.insert(1, bokeh.layouts.row([status_label]))

        return bokeh.layouts.column(models)

    def create_app(self):
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
    bgraphs: List[BGraphAndFigure],
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

        for item in bgraphs:
            if isinstance(item.bgraph, (BokehBasicGraph, BokehLatticeLayoutGraph)):
                item.bgraph.update_plot(item.fig, tao=tao)

    spinners = []
    for var in variables:
        spinner = var.create_spinner()
        spinners.append(spinner)
        spinner.on_change("value", functools.partial(variable_updated, var=var))
    return spinners


class BokehGraphManager(
    GraphManager[AnyBokehGraph, BokehLatticeLayoutGraph, BokehFloorPlanGraph]
):
    _key_: ClassVar[str] = "bokeh"
    _lattice_layout_graph_type = BokehLatticeLayoutGraph
    _floor_plan_graph_type = BokehFloorPlanGraph

    @override
    def make_graph(
        self,
        region_name: str,
        graph_name: str,
    ) -> AnyBokehGraph:
        graph = make_graph(self.tao, region_name, graph_name)
        if isinstance(graph, BasicGraph):
            return BokehBasicGraph(self, graph)
        elif isinstance(graph, LatticeLayoutGraph):
            return BokehLatticeLayoutGraph(self, graph)
        elif isinstance(graph, FloorPlanGraph):
            return BokehFloorPlanGraph(self, graph)
        raise NotImplementedError(type(graph).__name__)

    @override
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
        show_fields: bool = False,
        reuse: bool = True,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        save: Union[bool, str, pathlib.Path, None] = None,
        curves: Optional[Dict[int, TaoCurveSettings]] = None,
    ):
        """
        Plot a graph or all placed graphs with Bokeh.

        To plot a specific graph, specify `graph_name` (optionally `region_name`).
        The default is to plot all placed graphs.

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
        save : str or bool, optional
            Save the plot to a static HTML file with the given name.
            If `True`, saves to a filename based on the plot title.

        Returns
        -------
        list
        """
        bgraphs = self.prepare_graphs_by_name(
            graph_name=graph_name,
            region_name=region_name,
            reuse=reuse,
            curves=curves,
        )
        if not bgraphs:
            return []

        if (
            include_layout
            and any(bgraph.graph.is_s_plot for bgraph in bgraphs)
            and not any(isinstance(bgraph, BokehLatticeLayoutGraph) for bgraph in bgraphs)
        ):
            bgraphs.append(self.get_lattice_layout_graph())

        bgraphs = move_layout_to_bottom(bgraphs)

        for bgraph in bgraphs:
            is_layout = isinstance(bgraph, BokehLatticeLayoutGraph)
            if sizing_mode is not None:
                bgraph.sizing_mode = sizing_mode
            if width is not None:
                bgraph.width = width
            if is_layout:
                bgraph.show_fields = show_fields
                if show_fields and not bgraph.graph.fields:
                    bgraph.graph.update_fields(tao=self.tao)

                if layout_height is not None:
                    bgraph.height = layout_height
            else:
                if height is not None:
                    bgraph.height = height

        if save:
            title = bgraphs[0].graph.title or f"plot-{time.time()}"
            if save is True:
                save = f"{title}.html"
            if not pathlib.Path(save).suffix:
                save = f"{save}.html"

            logger.info(f"Saving plot to {save!r}")
            bokeh.io.save(
                bokeh.layouts.column([bgraph.create_figure() for bgraph in bgraphs]),
                filename=save,
                title=title,
            )

        return bgraphs


class NotebookGraphManager(BokehGraphManager):
    def plot_regions(
        self,
        regions: List[str],
        *,
        share_x: Optional[bool] = None,
        **kwargs,
    ):
        bgraphs = []
        for graph_name, graph_regions in list(self._graph_name_to_regions.items()):
            for region_name in graph_regions:
                if region_name in regions:
                    bgraphs.extend(
                        super().plot(graph_name=graph_name, region_name=region_name, **kwargs)
                    )

        bgraphs = move_layout_to_bottom(bgraphs)

        if not bgraphs:
            return None

        if len(bgraphs) == 1:
            (app,) = bgraphs
        else:
            app = CompositeApp(self.tao, bgraphs, share_x=share_x)

        return bokeh.plotting.show(app.create_app())

    @override
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
        show_fields: bool = False,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        notebook_handle: bool = False,
        reuse: bool = True,
        save: Union[bool, str, pathlib.Path, None] = None,
        curves: Optional[Dict[int, TaoCurveSettings]] = None,
    ):
        bgraphs = super().plot(
            region_name=region_name,
            graph_name=graph_name,
            include_layout=include_layout,
            sizing_mode=sizing_mode,
            width=width,
            height=height,
            layout_height=layout_height,
            show_fields=show_fields,
            reuse=reuse,
            xlim=xlim,
            ylim=ylim,
            curves=curves,
        )

        if not bgraphs:
            return None

        variables = Variable.from_tao_all(self.tao) if vars else None
        if len(bgraphs) == 1:
            (app,) = bgraphs
            if isinstance(app, BokehBasicGraph):
                app.variables = variables
        else:
            app = CompositeApp(self.tao, bgraphs, share_x=share_x, variables=variables)

        if save:
            title = bgraphs[0].graph.title or f"plot-{time.time()}"
            if save is True:
                save = f"{title}.html"
            if not pathlib.Path(save).suffix:
                save = f"{save}.html"

            logger.info(f"Saving plot to {save!r}")
            app.save(save, title=title)
            # NOTE/TODO: seems like we have to regenerate the bokeh glyphs
            # or they may be considered as owned by more than one document.
            # This is a workaround, but I assume there's a better way of
            # going about this...
            return self.plot(
                graph_name=graph_name,
                region_name=region_name,
                include_layout=include_layout,
                sizing_mode=sizing_mode,
                width=width,
                height=height,
                layout_height=layout_height,
                share_x=share_x,
                vars=vars,
                show_fields=show_fields,
                xlim=xlim,
                ylim=ylim,
                notebook_handle=notebook_handle,
                reuse=reuse,
                curves=curves,
                save=False,  # <-- important line
            )

        return bokeh.plotting.show(
            app.create_app(),
            notebook_handle=notebook_handle,
        )

    __call__ = plot


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
