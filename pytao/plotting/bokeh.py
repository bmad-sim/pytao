from __future__ import annotations

import logging
import math
import typing
from typing import (
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

from bokeh.core.enums import SizingModeType
import bokeh.events
import bokeh.models
import bokeh.models.tools
import bokeh.plotting

from bokeh.document.callbacks import EventCallback
from bokeh.layouts import column
from bokeh.models.sources import ColumnDataSource
from bokeh.plotting import figure
from typing_extensions import NotRequired, TypedDict

from . import pgplot
from .plot import (
    BasicGraph,
    GraphBase,
    GraphManager,
    LatticeLayoutElement,
    LatticeLayoutGraph,
    PlotCurve,
    PlotPatchRectangle,
    PlotPatchArc,
    PlotPatchCircle,
    PlotPatchEllipse,
    PlotPatchPolygon,
    PlotPatchCustom,
)

if typing.TYPE_CHECKING:
    from .. import Tao


logger = logging.getLogger(__name__)
# def bokeh_color(qp_color):
#     return getattr(qp_color.lower(), "black")


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


def _plot_curve(fig: figure, curve: PlotCurve, source: CurveData) -> None:
    name = pgplot.mathjax_string(curve.info["name"])
    if "line" in source and curve.line is not None:
        fig.line(
            "x",
            "y",
            line_width=curve.line.linewidth,
            source=source["line"],
            color=curve.line.color,
            name=name,
            legend_label=name,
        )

    if "symbol" in source and curve.symbol is not None:
        fig.scatter(
            "x",
            "y",
            source=source["symbol"],
            fill_color=curve.symbol.color,
            name=name,
            legend_label=None if "line" in source else name,
        )


def _plot_custom_patch(fig: figure, patch: PlotPatchCustom):
    raise NotImplementedError("plot custom patch")


def _draw_layout_element(
    fig: figure,
    elem: LatticeLayoutElement,
    skip_labels: bool = True,
):
    base_data = {
        "s_start": [elem.info["ele_s_start"]],
        "s_end": [elem.info["ele_s_end"]],
        "name": [elem.info["label_name"]],
        "color": [elem.color],
    }
    all_lines: List[Tuple[List[float], List[float]]] = []
    for patch in elem.patches:
        source = ColumnDataSource(data=dict(base_data))
        if isinstance(patch, PlotPatchRectangle):
            points = patch.to_mpl().get_corners()
            all_lines.append(
                (
                    points[:, 0].tolist() + [points[0, 0]],
                    points[:, 1].tolist() + [points[0, 1]],
                )
            )
        elif isinstance(patch, PlotPatchPolygon):
            all_lines.append(
                (
                    [p[0] for p in patch.vertices + patch.vertices[:1]],
                    [p[1] for p in patch.vertices + patch.vertices[:1]],
                )
            )
        elif isinstance(patch, PlotPatchArc):
            source.data["x"], source.data["y"] = [patch.xy[0]], [patch.xy[1]]
            fig.arc(
                x="x",
                y="y",
                start_angle=[patch.theta1],
                end_angle=[patch.theta2],
                line_width=elem.width,
                fill_alpha=int(patch.fill),
                source=source,
            )
        elif isinstance(patch, PlotPatchCircle):
            source.data["x"], source.data["y"] = [patch.xy[0]], [patch.xy[1]]
            fig.circle(
                x="x",
                y="y",
                radii=[patch.radius],
                line_width=elem.width,
                fill_alpha=int(patch.fill),
                source=source,
            )
        elif isinstance(patch, PlotPatchEllipse):
            source.data["x"], source.data["y"] = [patch.xy[0]], [patch.xy[1]]
            fig.ellipse(
                x="x",
                y="y",
                width=[patch.width],
                height=[patch.height],
                angle=[math.radians(patch.angle)],
                line_width=elem.width,
                fill_alpha=int(patch.fill),
                source=source,
            )
        elif isinstance(patch, PlotPatchCustom):
            _plot_custom_patch(fig, patch)
        else:
            raise NotImplementedError(f"{type(patch).__name__}")

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
            color=elem.color,
            source=source,
        )

    for annotation in elem.annotations:
        if annotation.text == elem.info["label_name"] and skip_labels:
            continue
        source = ColumnDataSource(
            data={
                "x": [annotation.x],
                "y": [annotation.y],
                "text": [pgplot.mathjax_string(annotation.text)],
                **base_data,
            }
        )
        fig.text(
            "x",
            "y",
            angle=math.radians(annotation.rotation),
            text_align="right",  # annotation.horizontalalignment,
            text_baseline=annotation.verticalalignment,
            color=elem.color,
            source=source,
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

    def create_app_figure(self) -> Tuple[figure, List[bokeh.models.UIElement]]:
        raise NotImplementedError()

    def create_app(self):
        def bokeh_app(doc):
            _primary_figure, models = self.create_app_figure()
            for model in models:
                doc.add_root(model)

        return bokeh_app

    __call__ = create_app


class BokehLatticeGraph(BokehGraphBase[LatticeLayoutGraph]):
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
        num_points = bokeh.models.Slider(
            title="Points",
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
        fig.on_event(bokeh.events.RangesUpdate, cast(EventCallback, ranges_update))
        return fig, [column(num_points, fig)]


AnyBokehGraph = Union[BokehBasicGraph, BokehLatticeGraph]


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
        return column(models)

    def create_app(self):
        def bokeh_app(doc):
            doc.add_root(self.create_ui())

        return bokeh_app


class BokehGraphManager(GraphManager):
    def plot(
        self,
        region_name: str,
        graph_name: str,
        *,
        place: bool = True,
        show: bool = False,
    ) -> AnyBokehGraph:
        if place:
            self.place_all_requested()

        logger.debug(f"Plotting {region_name}.{graph_name}")
        graph = self.regions[region_name][graph_name]
        if isinstance(graph, BasicGraph):
            return BokehBasicGraph(self, graph)
        if isinstance(graph, LatticeLayoutGraph):
            return BokehLatticeGraph(self, graph)
        raise NotImplementedError(type(graph).__name__)

    def plot_region(
        self,
        region_name: str,
        *,
        place: bool = True,
        show: bool = False,
        share_x: Optional[bool] = None,
    ) -> Dict[str, AnyBokehGraph]:
        if place:
            self.place_all_requested()

        res: Dict[str, AnyBokehGraph] = {}
        for graph_name in self.regions[region_name]:
            res[graph_name] = self.plot(
                region_name=region_name,
                graph_name=graph_name,
                place=False,
            )

        return res

    def plot_all(
        self,
        *,
        place: bool = True,
        show: bool = False,
        share_x: Optional[bool] = None,
    ) -> Dict[str, Dict[str, AnyBokehGraph]]:
        if place:
            self.place_all_requested()

        res: Dict[str, Dict[str, AnyBokehGraph]] = {}
        for region_name in self.regions:
            res[region_name] = self.plot_region(
                region_name,
                place=False,
                show=show,
            )

        return res


class NotebookGraphManager(BokehGraphManager):
    def show(
        self,
        region_name: Optional[str] = None,
        graph_name: Optional[str] = None,
        *,
        include_layout: bool = True,
        place: bool = True,
        sizing_mode: Optional[SizingModeType] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        share_x: Optional[bool] = None,
    ):
        if graph_name and not region_name:
            raise ValueError("Must specify region_name if graph_name is specified")

        if region_name and graph_name:
            bgraphs = [
                self.plot(
                    region_name,
                    graph_name,
                    place=place,
                )
            ]
        elif region_name:
            region = self.plot_region(
                region_name,
                place=place,
            )
            bgraphs = list(region.values())
        else:
            by_region = self.plot_all(
                place=place,
            )
            bgraphs = [graph for region in by_region.values() for graph in region.values()]

        if not bgraphs:
            return None

        if (
            include_layout
            and not any(isinstance(bgraph, BokehLatticeGraph) for bgraph in bgraphs)
            and any(bgraph.graph.is_s_plot for bgraph in bgraphs)
            and "layout" in self.regions
            and "g" in self.regions["layout"]
        ):
            bgraphs.append(self.plot("layout", "g"))

        for bgraph in bgraphs:
            if sizing_mode is not None:
                bgraph.sizing_mode = sizing_mode
            if width is not None:
                bgraph.width = width
            if height is not None:
                bgraph.height = height

        if len(bgraphs) == 1:
            (app,) = bgraphs
        else:
            app = CompositeApp(bgraphs, share_x=share_x)

        return bokeh.plotting.show(app.create_app())

    __call__ = show
