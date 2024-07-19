from __future__ import annotations

import functools
import logging
import math
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
import bokeh.events
import bokeh.layouts
import bokeh.models
import bokeh.models.tools
import bokeh.plotting
import numpy as np
from bokeh.core.enums import SizingModeType
from bokeh.document.callbacks import EventCallback
from bokeh.models.sources import ColumnDataSource
from bokeh.plotting import figure
from typing_extensions import NotRequired, TypedDict, override

from . import pgplot, util
from .plot import (
    BasicGraph,
    FloorPlanGraph,
    GraphBase,
    GraphManager,
    LatticeLayoutElement,
    LatticeLayoutGraph,
    LayoutGraphNotFoundError,
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
)

if typing.TYPE_CHECKING:
    from .. import Tao


logger = logging.getLogger(__name__)


def bokeh_color(color):
    color = color.lower().replace("_", "")
    return getattr(bokeh.colors.named, color, "black")


class CurveData(TypedDict):
    line: NotRequired[ColumnDataSource]
    symbol: NotRequired[ColumnDataSource]


GraphData = List[CurveData]


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


T_Tool = TypeVar("T_Tool", bound=bokeh.models.tools.Tool)


def get_tool_from_figure(fig: figure, tool_cls: Type[T_Tool]) -> Optional[T_Tool]:
    tools = [tool for tool in fig.tools if isinstance(tool, tool_cls)]
    return tools[0] if tools else None


def link_crosshairs(figs: List[figure]):
    first, *rest = figs
    crosshair = get_tool_from_figure(first, bokeh.models.tools.CrosshairTool)
    if crosshair is None:
        return

    if crosshair.overlay == "auto":
        crosshair.overlay = (
            bokeh.models.tools.Span(dimension="width", line_dash="dotted", line_width=1),
            bokeh.models.tools.Span(dimension="height", line_dash="dotted", line_width=1),
        )

    for fig in rest:
        other_crosshair = get_tool_from_figure(fig, bokeh.models.tools.CrosshairTool)
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
    source = ColumnDataSource(
        data={
            "x": symbol.xs,
            "y": symbol.ys,
            **(source.data if source else {}),
        },
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
        name=name,
        **kw,
    )


def _plot_curve_line(
    fig: figure,
    line: PlotCurveLine,
    name: Optional[str] = None,
    source: Optional[ColumnDataSource] = None,
):
    source = ColumnDataSource(
        data={
            "x": line.xs,
            "y": line.ys,
            **(source.data if source else {}),
        },
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
    base_data: Optional[dict] = None,
    color: str = "black,",
):
    source = ColumnDataSource(
        data={
            "x": [annotation.x],
            "y": [annotation.y],
            "text": [pgplot.mathjax_string(annotation.text)],
            **(base_data or {}),
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
    xcenter = xlim[0] + width / 2.0
    ycenter = ylim[0] + height / 2.0
    rect = PlotPatchRectangle(xy=(xcenter, ycenter), width=width, height=height, alpha=alpha)
    px, py = _patch_rect_to_points(rect)

    return fig.line(px, py, alpha=alpha)


def _plot_patch(
    fig: figure,
    patch: PlotPatch,
    line_width: Optional[float] = None,
    source: Optional[ColumnDataSource] = None,
):
    source = ColumnDataSource(
        data={
            **(source.data if source else {}),
        },
    )
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
        _draw_annotation(fig, annotation, color=color, base_data=base_data)


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
    graph: LatticeLayoutGraph

    def __init__(
        self,
        manager: GraphManager,
        graph: LatticeLayoutGraph,
        sizing_mode: SizingModeType = "inherit",
        width: int = 900,
        height: int = 300,
        aspect_ratio: float = 3.0,  # w/h
    ) -> None:
        super().__init__(
            manager=manager,
            graph=graph,
            sizing_mode=sizing_mode,
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
        )

    def create_figure(
        self,
        tools: str = "pan,wheel_zoom,box_zoom,save,reset,help,crosshair",
        toolbar_location: str = "above",
        add_named_hover_tool: bool = True,
        set_xaxis_ticks: bool = True,
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
            hover = bokeh.models.tools.HoverTool(
                tooltips=[
                    ("name", "@name"),
                    ("s start [m]", "@s_start"),
                    ("s end [m]", "@s_end"),
                ]
            )

            fig.add_tools(hover)

        box_zoom = get_tool_from_figure(fig, bokeh.models.tools.BoxZoomTool)
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
        for elem in graph.elements:
            _draw_layout_element(fig, elem, skip_labels=set_xaxis_ticks)

        if self.x_range is not None:
            fig.x_range = self.x_range

        return fig

    def create_app_figure(self) -> Tuple[figure, List[bokeh.models.UIElement]]:
        fig = self.create_figure()
        return fig, [fig]


class BokehBasicGraph(BokehGraphBase[BasicGraph]):
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

    @property
    def tao(self) -> Tao:
        return self.manager.tao

    def update_plot(self) -> None:
        try:
            self.tao.cmd("set global lattice_calc_on = F")
            self.tao.cmd(f"set plot {self.graph.region_name} n_curve_pts = {self.num_points}")
            self.tao.cmd(
                f"x_scale {self.graph.region_name} {self.view_x_range[0]} {self.view_x_range[1]}"
            )
        finally:
            self.tao.cmd("set global lattice_calc_on = T")

        logger.debug(f"x={self.view_x_range} points={self.num_points}")

        updated = self.graph.update(self.manager)
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

    def create_app_figure(self) -> Tuple[figure, List[bokeh.models.UIElement]]:
        fig = self.create_figure()
        update_button = bokeh.models.Button(label="Update")  # , icon="reload")
        num_points = bokeh.models.Slider(
            title="Data Points",
            start=10,
            end=10_000,
            step=1_000,
            value=401,
        )

        def ranges_update(event: bokeh.events.RangesUpdate) -> None:
            new_xrange = self.graph.clamp_x_range(event.x0, event.x1)
            if new_xrange != self.view_x_range:
                self.view_x_range = new_xrange

            try:
                self.update_plot()
            except Exception:
                logger.exception("Failed to update number ranges")

        def num_points_changed(_attr, _old, num_points: int):
            self.num_points = num_points
            try:
                self.update_plot()
            except Exception:
                logger.exception("Failed to update number of points")

        num_points.on_change("value", num_points_changed)
        update_button.on_click(self.update_plot)
        fig.on_event(bokeh.events.RangesUpdate, cast(EventCallback, ranges_update))
        return fig, [bokeh.layouts.column(bokeh.layouts.row(update_button, num_points), fig)]


class BokehFloorPlanGraph(BokehGraphBase[FloorPlanGraph]):
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

        box_zoom = get_tool_from_figure(fig, bokeh.models.tools.BoxZoomTool)
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
            source = ColumnDataSource()
            for line in elem.lines:
                _plot_curve_line(fig, line, source=source)
            for patch in elem.patches:
                _plot_patch(fig, patch, line_width=elem.info["line_width"], source=source)
            for annotation in elem.annotations:
                _draw_annotation(
                    fig,
                    annotation,
                    color=bokeh_color(elem.info["color"]),
                    base_data={"name": [elem.info["label_name"]]},
                )

        _draw_limit_border(fig, graph.xlim, graph.ylim, alpha=0.1)

        return fig

    def create_app_figure(self) -> Tuple[figure, List[bokeh.models.UIElement]]:
        fig = self.create_figure()
        return fig, [fig]


AnyBokehGraph = Union[BokehBasicGraph, BokehLatticeLayoutGraph, BokehFloorPlanGraph]


class CompositeApp:
    bgraphs: List[AnyBokehGraph]
    share_x: Optional[bool]

    def __init__(self, bgraphs: List[AnyBokehGraph], share_x: Optional[bool] = None) -> None:
        self.bgraphs = bgraphs
        self.share_x = share_x

    def create_ui(self):
        items: List[BGraphAndFigure] = []
        models: List[bokeh.models.UIElement] = []
        for bgraph in self.bgraphs:
            primary_figure, fig_models = bgraph.create_app_figure()
            items.append(BGraphAndFigure(bgraph, primary_figure))
            models.extend(fig_models)

        for item in items:
            # NOTE: this value is somewhat arbitrary; it helps align the X axes
            # between consecutive plots
            item.fig.min_border_left = 80

        if self.share_x is None:
            share_common_x_axes(items)
        elif self.share_x:
            share_x_axes([item.fig for item in items])
        return bokeh.layouts.column(models)

    def create_app(self):
        def bokeh_app(doc):
            doc.add_root(self.create_ui())

        return bokeh_app


class BokehGraphManager(GraphManager):
    _key_: ClassVar[str] = "bokeh"

    @override
    def get_plot(
        self,
        region_name: str,
        graph_name: str,
        *,
        place: bool = True,
    ) -> AnyBokehGraph:
        if place:
            self.place()

        logger.debug(f"Plotting {region_name}.{graph_name}")
        graph = super().get_plot(region_name, graph_name, place=False)
        if isinstance(graph, BasicGraph):
            return BokehBasicGraph(self, graph)
        if isinstance(graph, LatticeLayoutGraph):
            return BokehLatticeLayoutGraph(self, graph)
        if isinstance(graph, FloorPlanGraph):
            return BokehFloorPlanGraph(self, graph)
        raise NotImplementedError(type(graph).__name__)

    @override
    def get_region(
        self,
        region_name: str,
        *,
        place: bool = True,
    ) -> Dict[str, AnyBokehGraph]:
        if place:
            self.place()

        res: Dict[str, AnyBokehGraph] = {}
        for graph_name in self.regions[region_name]:
            res[graph_name] = self.get_plot(
                region_name=region_name,
                graph_name=graph_name,
                place=False,
            )

        return res

    @override
    def get_all(
        self,
        *,
        place: bool = True,
    ) -> Dict[str, Dict[str, AnyBokehGraph]]:
        if place:
            self.place()

        res: Dict[str, Dict[str, AnyBokehGraph]] = {}
        for region_name in self.regions:
            res[region_name] = self.get_region(region_name, place=False)

        return res

    @override
    def plot(
        self,
        region_name: Optional[str] = None,
        graph_name: Optional[str] = None,
        *,
        include_layout: bool = True,
        place: bool = True,
        sizing_mode: Optional[SizingModeType] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        layout_height: Optional[int] = None,
        share_x: Optional[bool] = None,
    ):
        """
        Plot a graph, region, or all placed graphs.

        To plot a specific graph, specify `region_name` and `graph_name`.
        To plot a specific region, specify `region_name`.
        To plot all placed graphs, specify neither.

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
        sizing_mode : Optional[SizingModeType]
            Set the sizing mode for all graphs.  Default is configured on a
            per-graph basis, typically "inherit".
        width : int, optional
            Width of each plot.
        height : int, optional
            Height of each plot.
        layout_height : int, optional
            Height of the layout plot.

        Returns
        -------
        list
        """
        if graph_name and not region_name:
            raise ValueError("Must specify region_name if graph_name is specified")

        if place:
            self.place()

        if region_name and graph_name:
            bgraphs = [self.get_plot(region_name, graph_name, place=False)]
        elif region_name:
            region = self.get_region(region_name, place=False)
            bgraphs = list(region.values())
        else:
            by_region = self.get_all()
            bgraphs = [graph for region in by_region.values() for graph in region.values()]

        if not bgraphs:
            return []

        if (
            include_layout
            and not any(isinstance(bgraph, BokehLatticeLayoutGraph) for bgraph in bgraphs)
            and any(bgraph.graph.is_s_plot for bgraph in bgraphs)
        ):
            try:
                layout_graph = self.get_lattice_layout_graph(place=False)
            except LayoutGraphNotFoundError:
                logger.warning("Could not find lattice layout to include")
            else:
                bgraphs.append(
                    self.get_plot(
                        layout_graph.region_name,
                        layout_graph.graph_name,
                        place=False,
                    )
                )

        for bgraph in bgraphs:
            is_layout = isinstance(bgraph, BokehLatticeLayoutGraph)
            if sizing_mode is not None:
                bgraph.sizing_mode = sizing_mode
            if width is not None:
                bgraph.width = width
            if is_layout:
                if layout_height is not None:
                    bgraph.height = layout_height
            else:
                if height is not None:
                    bgraph.height = height
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
        for region in regions:
            bgraphs.extend(super().plot(region, **kwargs))

        if len(bgraphs) == 1:
            (app,) = bgraphs
        else:
            app = CompositeApp(bgraphs, share_x=share_x)

        return bokeh.plotting.show(app.create_app())

    @override
    def plot(
        self,
        region_name: Optional[str] = None,
        graph_name: Optional[str] = None,
        *,
        include_layout: bool = True,
        place: bool = True,
        sizing_mode: Optional[SizingModeType] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        layout_height: Optional[int] = None,
        share_x: Optional[bool] = None,
    ):
        bgraphs = super().plot(
            region_name=region_name,
            graph_name=graph_name,
            include_layout=include_layout,
            place=place,
            sizing_mode=sizing_mode,
            width=width,
            height=height,
            layout_height=layout_height,
        )

        if len(bgraphs) == 1:
            (app,) = bgraphs
        else:
            app = CompositeApp(bgraphs, share_x=share_x)

        return bokeh.plotting.show(app.create_app())

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
