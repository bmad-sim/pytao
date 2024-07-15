from __future__ import annotations

import logging
import typing
from typing import List, Optional

# from bokeh.colors import named
import bokeh.events
import bokeh.plotting
from bokeh.layouts import column
from bokeh.models import Slider
from bokeh.models.sources import ColumnDataSource
from bokeh.plotting import figure
from typing_extensions import NotRequired, TypedDict

from .plot import BasicGraph, GraphManager, PlotCurve

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


def _plot_curve(fig: figure, curve: PlotCurve, source: CurveData) -> None:
    if "line" in source and curve.line is not None:
        fig.line(
            "x",
            "y",
            line_width=curve.line.linewidth,
            source=source["line"],
            color=curve.line.color,
        )

    if "symbol" in source and curve.symbol is not None:
        fig.scatter(
            "symbol_x",
            "symbol_y",
            source=source["symbol"],
            fill_color=curve.symbol.color,
        )


def _create_figure(graph: BasicGraph, curve_data: Optional[GraphData] = None) -> figure:
    if curve_data is None:
        curve_data = _get_graph_data(graph)

    fig = figure(
        title=graph.title,
        x_axis_label=graph.xlabel,
        y_axis_label=graph.ylabel,
    )
    for curve, source in zip(graph.curves, curve_data):
        _plot_curve(fig, curve, source)
    return fig


class BokehBasicGraph:
    def __init__(self, manager: GraphManager, graph: BasicGraph) -> None:
        self.graph = graph
        self.manager = manager
        self.curve_data = _get_graph_data(graph)
        self.num_points = graph.get_num_points()
        self.x_range = graph.get_x_range()

    @property
    def tao(self) -> Tao:
        return self.manager.tao

    def update_plot(self) -> None:
        try:
            self.tao.cmd("set global lattice_calc_on = F")
            self.tao.cmd(f"set plot {self.graph.region_name} n_curve_pts = {self.num_points}")
            self.tao.cmd(
                f"x_scale {self.graph.region_name} {self.x_range[0]} {self.x_range[1]}"
            )
        finally:
            self.tao.cmd("set global lattice_calc_on = T")

        logger.debug(f"x={self.x_range} points={self.num_points}")

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

    def create_app(self):
        def bokeh_app(doc):
            fig = _create_figure(self.graph, curve_data=self.curve_data)

            num_points = Slider(
                title="Points",
                start=10,
                end=10_000,
                step=1_000,
                value=401,
            )

            def ranges_update(event: bokeh.events.RangesUpdate) -> None:
                self.x_range = self.graph.clamp_x_range(event.x0, event.x1)
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
            fig.on_event(bokeh.events.RangesUpdate, ranges_update)
            doc.add_root(column(num_points, fig))
            return bokeh_app

        return bokeh_app

    __call__ = create_app


class BokehGraphManager(GraphManager):
    def plot(
        self,
        region_name: str,
        graph_name: str,
        *,
        include_layout: bool = True,
        show: bool = True,
    ):
        logger.debug(f"Plotting {region_name}.{graph_name}")
        graph = self.regions[region_name][graph_name]
        if isinstance(graph, BasicGraph):
            bgraph = BokehBasicGraph(self, graph).create_app()
            return bokeh.plotting.show(bgraph) if show else bgraph

    def plot_region(
        self,
        region_name: str,
        include_layout: bool = True,
    ):
        if region_name in self.to_place:
            self.place_all_requested()

        res = {}
        for graph_name in self.regions[region_name]:
            res[f"{region_name}.{graph_name}"] = self.plot(
                region_name=region_name,
                graph_name=graph_name,
                include_layout=include_layout,
            )
        return res

    def plot_all(
        self,
        include_layout: bool = True,
    ):
        self.place_all_requested()

        res = {}
        for region_name in self.regions:
            res.update(self.plot_region(region_name, include_layout=include_layout))
        return res
