from __future__ import annotations

import logging
import math
import typing
from typing import (
    ClassVar,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import matplotlib.axes
import matplotlib.collections
import matplotlib.patches
import matplotlib.path
import matplotlib.pyplot as plt
import matplotlib.text
import numpy as np
import pydantic.dataclasses as dataclasses
from matplotlib.ticker import AutoMinorLocator
from pydantic.dataclasses import Field
from typing_extensions import override


from . import pgplot, util
from .fields import LatticeLayoutField
from .types import (
    BuildingWallGraphInfo,
    BuildingWallInfo,
    FloorOrbitInfo,
    FloorPlanElementInfo,
    PlotCurveInfo,
    PlotGraphInfo,
    PlotHistogramInfo,
    PlotLatLayoutInfo,
    PlotPage,
    PlotRegionInfo,
    Point,
    WaveParams,
)

if typing.TYPE_CHECKING:
    from .. import Tao

logger = logging.getLogger(__name__)


class GraphInvalidError(Exception):
    pass


class NoLayoutError(Exception):
    pass


class NoCurveDataError(Exception):
    pass


class UnsupportedGraphError(NotImplementedError):
    pass


def _fix_limits(lim: Point, pad_factor: float = 0.0) -> Point:
    low, high = lim
    if np.isclose(low, 0.0) and np.isclose(high, 0.0):
        # TODO: matplotlib can sometimes get in a bad spot trying to plot empty data
        # with very small limits
        return (-0.001, 0.001)
    return (low - abs(low * pad_factor), high + abs(high * pad_factor))


def _should_use_symbol_color(symbol_type: str, fill_pattern: str) -> bool:
    if (
        symbol_type in ("dot", "1")
        or symbol_type.endswith("filled")
        or symbol_type.startswith("-")
    ):
        return True

    if pgplot.fills[fill_pattern] == "full":
        return True

    return False


@dataclasses.dataclass
class PlotAnnotation:
    x: float
    y: float
    text: str
    horizontalalignment: str = "left"
    verticalalignment: str = "baseline"
    clip_on: bool = False
    color: str = "black"
    rotation: float = 0
    rotation_mode: str = "default"

    def plot(self, ax: matplotlib.axes.Axes):
        return ax.annotate(
            xy=(self.x, self.y),
            text=pgplot.mpl_string(self.text),
            horizontalalignment=self.horizontalalignment,
            verticalalignment=self.verticalalignment,
            clip_on=self.clip_on,
            color=pgplot.mpl_color(self.color),
            rotation=self.rotation,
            rotation_mode=self.rotation_mode,
            fontsize=8,
        )


@dataclasses.dataclass
class PlotCurveLine:
    xs: List[float]
    ys: List[float]
    color: str = "black"
    linestyle: str = "solid"
    linewidth: float = 1.0

    def plot(self, ax: matplotlib.axes.Axes, label: Optional[str] = None):
        return ax.plot(
            self.xs,
            self.ys,
            color=pgplot.mpl_color(self.color),
            linestyle=self.linestyle,
            linewidth=self.linewidth,
            label=label,
        )


@dataclasses.dataclass
class PlotCurveSymbols:
    xs: List[float]
    ys: List[float]
    color: str
    markerfacecolor: str
    markersize: float
    marker: str
    markeredgewidth: float
    linewidth: float = 0

    def plot(self, ax: matplotlib.axes.Axes, label: Optional[str] = None):
        return ax.plot(
            self.xs,
            self.ys,
            color=pgplot.mpl_color(self.color),
            markerfacecolor=self.markerfacecolor,
            markersize=self.markersize,
            marker=pgplot.symbols.get(self.marker, "."),
            markeredgewidth=self.markeredgewidth,
            linewidth=self.linewidth,
            label=label,
        )


@dataclasses.dataclass
class PlotHistogram:
    xs: List[float]
    bins: Union[int, Sequence[float], str, None]
    weights: List[float]
    histtype: Literal["bar", "barstacked", "step", "stepfilled"]
    color: str

    def plot(self, ax: matplotlib.axes.Axes):
        return ax.hist(
            self.xs,
            bins=self.bins,
            weights=self.weights,
            histtype=self.histtype,
            color=pgplot.mpl_color(self.color),
        )


@dataclasses.dataclass
class PlotPatchBase:
    edgecolor: Optional[str] = None
    facecolor: Optional[str] = None
    color: Optional[str] = None
    linewidth: Optional[float] = None
    linestyle: Optional[str] = None
    antialiased: Optional[bool] = None
    hatch: Optional[str] = None
    fill: bool = True
    capstyle: Optional[str] = None
    joinstyle: Optional[str] = None
    alpha: float = 1.0

    @property
    def _patch_args(self):
        return {
            "edgecolor": self.edgecolor,
            "facecolor": self.facecolor,
            "color": pgplot.mpl_color(self.color or "black"),
            "linewidth": self.linewidth,
            "linestyle": self.linestyle,
            "antialiased": self.antialiased,
            "hatch": self.hatch,
            "fill": self.fill,
            "capstyle": self.capstyle,
            "joinstyle": self.joinstyle,
            "alpha": self.alpha,
        }

    def to_mpl(self):
        raise NotImplementedError(type(self))

    def plot(self, ax: matplotlib.axes.Axes):
        mpl = self.to_mpl()
        ax.add_patch(mpl)
        return mpl


_point_field = Field(default_factory=lambda: (0.0, 0.0))


@dataclasses.dataclass
class PlotPatchRectangle(PlotPatchBase):
    xy: Point = _point_field
    width: float = 0.0
    height: float = 0.0
    angle: float = 0.0
    rotation_point: Union[Literal["xy", "center"], Point] = "xy"

    def to_mpl(self) -> matplotlib.patches.Rectangle:
        return matplotlib.patches.Rectangle(
            xy=self.xy,
            width=self.width,
            height=self.height,
            angle=self.angle,
            rotation_point=self.rotation_point,
            **self._patch_args,
        )


@dataclasses.dataclass
class PlotPatchArc(PlotPatchBase):
    xy: Point = _point_field
    width: float = 0.0
    height: float = 0.0
    angle: float = 0.0
    theta1: float = 0.0
    theta2: float = 360.0
    fill: bool = False  # override

    def to_mpl(self) -> matplotlib.patches.Arc:
        return matplotlib.patches.Arc(
            xy=self.xy,
            width=self.width,
            height=self.height,
            angle=self.angle,
            theta1=self.theta1,
            theta2=self.theta2,
            **self._patch_args,
        )


@dataclasses.dataclass
class PlotPatchCircle(PlotPatchBase):
    xy: Point = _point_field
    radius: float = 0.0

    def to_mpl(self) -> matplotlib.patches.Ellipse:
        return matplotlib.patches.Circle(
            xy=self.xy,
            radius=self.radius,
            **self._patch_args,
        )


@dataclasses.dataclass
class PlotPatchPolygon(PlotPatchBase):
    vertices: List[Point] = Field(default_factory=list)

    def to_mpl(self) -> matplotlib.patches.Polygon:
        return matplotlib.patches.Polygon(
            xy=self.vertices,
            **self._patch_args,
        )


@dataclasses.dataclass
class PlotPatchEllipse(PlotPatchBase):
    xy: Point = _point_field
    width: float = 0.0
    height: float = 0.0
    angle: float = 0.0

    def to_mpl(self) -> matplotlib.patches.Ellipse:
        return matplotlib.patches.Ellipse(
            xy=self.xy,
            width=self.width,
            height=self.height,
            angle=self.angle,
            **self._patch_args,
        )


@dataclasses.dataclass
class PlotPatchSbend(PlotPatchBase):
    spline1: Tuple[Point, Point, Point] = Field(default_factory=tuple)
    spline2: Tuple[Point, Point, Point] = Field(default_factory=tuple)

    def to_mpl(self) -> matplotlib.patches.PathPatch:
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
            self.spline1[0],
            self.spline1[1],
            self.spline1[2],
            self.spline2[0],
            self.spline2[1],
            self.spline2[2],
            self.spline1[0],
        ]
        return matplotlib.patches.PathPatch(
            matplotlib.path.Path(vertices, codes),
            facecolor="green",
            alpha=0.5,
        )


PlotPatch = Union[
    PlotPatchRectangle,
    PlotPatchArc,
    PlotPatchCircle,
    PlotPatchEllipse,
    PlotPatchPolygon,
    PlotPatchSbend,
]


@dataclasses.dataclass
class GraphBase:
    info: PlotGraphInfo
    region_info: PlotRegionInfo
    region_name: str
    graph_name: str
    xlim: Point = _point_field
    ylim: Point = _point_field
    xlabel: str = ""
    ylabel: str = ""
    title: str = ""
    show_axes: bool = True
    draw_grid: bool = True
    draw_legend: bool = True

    def update(self, manager: GraphManager, *, error_on_new_type: bool = True):
        graphs = [
            graph.get_graph_info()
            for graph in manager.prepare_graphs_by_name(
                region_name=self.region_name,
                graph_name=self.graph_name,
                update=True,
            )
        ]
        # TODO
        for graph in graphs:
            if graph.graph_name == self.graph_name:
                if error_on_new_type and not isinstance(graph, type(self)):
                    raise ValueError(
                        f"Graph type changed from {type(self).__name__} to {type(graph).__name__}"
                    )
                return graph
        raise RuntimeError("Plot not found after update?")

    def get_graph_info(self):
        # For wrapping compatibility; messy cleanup reminder
        return self

    def _setup_axis(self, ax: matplotlib.axes.Axes, xticks: bool = True, yticks: bool = True):
        if not self.show_axes:
            ax.set_axis_off()

        ax.set_title(pgplot.mpl_string(self.title))
        ax.set_xlabel(pgplot.mpl_string(self.xlabel))
        ax.set_ylabel(pgplot.mpl_string(self.ylabel))
        ax.set_xlim(_fix_limits(self.xlim))
        ax.set_ylim(_fix_limits(self.ylim))
        ax.set_axisbelow(True)

        if self.draw_grid:
            ax.grid(self.draw_grid, which="major", axis="both")

        if xticks:
            if self.info["x_minor_div"] > 0:
                ax.xaxis.set_minor_locator(AutoMinorLocator(self.info["x_minor_div"]))
                ax.tick_params(axis="x", which="minor", length=4, color="black")

            if self.info["x_major_div_nominal"] > 2:
                ax.set_xticks(
                    np.linspace(
                        ax.get_xlim()[0],
                        ax.get_xlim()[1],
                        self.info["x_major_div_nominal"] - 1,
                    )
                )

        if yticks:
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(axis="y", which="minor", length=4, color="black")
            # ax.set_yticks(
            #     np.linspace(
            #         ax.get_ylim()[0],
            #         ax.get_ylim()[1],
            #         self.info["y_major_div_nominal"] - 1,
            #     )
            # )


@dataclasses.dataclass
class PlotCurve:
    info: PlotCurveInfo
    line: Optional[PlotCurveLine]
    symbol: Optional[PlotCurveSymbols]
    histogram: Optional[PlotHistogram] = None
    patches: Optional[List[PlotPatch]] = None

    def plot(self, ax: matplotlib.axes.Axes):
        res = []
        if self.line is not None:
            res.append(self.line.plot(ax, label=pgplot.mpl_string(self.legend_label)))
        if self.symbol is not None:
            res.append(
                self.symbol.plot(
                    ax,
                    label=pgplot.mpl_string(self.legend_label) if self.line is None else None,
                )
            )
        if self.histogram is not None:
            res.append(self.histogram.plot(ax))
        for patch in self.patches or []:
            res.append(patch.plot(ax))
        return res

    @property
    def legend_label(self) -> str:
        legend_text = self.info["legend_text"]
        if legend_text:
            return legend_text

        data_type = self.info["data_type"]
        return data_type if data_type == "physical_aperture" else ""

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        region_name: str,
        graph_name: str,
        curve_name: str,
        *,
        graph_type: Optional[str] = None,
    ) -> PlotCurve:
        full_name = f"{region_name}.{graph_name}.{curve_name}"
        curve_info = cast(PlotCurveInfo, tao.plot_curve(full_name))
        try:
            points = [
                (line["x"], line["y"])
                for line in tao.plot_line(region_name, graph_name, curve_name) or []
            ]
        except RuntimeError:
            points = []

        try:
            symbol_points = [
                (sym["x_symb"], sym["y_symb"])
                for sym in tao.plot_symbol(region_name, graph_name, curve_name, x_or_y="")
                or []
            ]
        except RuntimeError:
            symbol_points = []

        if graph_type is None:
            graph_info = get_plot_graph_info(tao, region_name, graph_name)
            graph_type = graph_info["graph^type"]

        if graph_type == "histogram":
            histogram_info = cast(PlotHistogramInfo, tao.plot_histogram(full_name))
        else:
            histogram_info = None

        wave_params = cast(WaveParams, tao.wave("params"))
        return cls.from_info(
            graph_type=graph_type,
            curve_info=curve_info,
            points=points,
            symbol_points=symbol_points,
            histogram_info=histogram_info,
            wave_params=wave_params,
        )

    @classmethod
    def from_info(
        cls,
        graph_type: str,
        curve_info: PlotCurveInfo,
        points: List[Point],
        symbol_points: List[Point],
        histogram_info: Optional[PlotHistogramInfo] = None,
        wave_params: Optional[WaveParams] = None,
    ) -> PlotCurve:
        line_color = pgplot.mpl_color(curve_info["line"]["color"])
        # TODO: line^pattern typo?
        line_style = pgplot.styles[curve_info["line"]["line^pattern"].lower()]
        if curve_info["draw_line"]:
            line_width = curve_info["line"]["width"]
        else:
            line_width = 0.0
        symbol_color = pgplot.mpl_color(curve_info["symbol"]["color"])

        # TODO: symbol^type typo?
        symbol_info = curve_info["symbol"]
        symbol_type = symbol_info["symbol^type"]
        if _should_use_symbol_color(
            symbol_type=symbol_type,
            fill_pattern=symbol_info["fill_pattern"],
        ):
            marker_color = symbol_info["color"]
        else:
            marker_color = "none"

        if curve_info["draw_symbols"] and pgplot.symbols[symbol_type]:
            marker_size = curve_info["symbol"]["height"]
        else:
            marker_size = 0

        symbol_line_width = curve_info["symbol"]["line_width"]

        xpoints = [p[0] for p in points]
        ypoints = [p[1] for p in points]
        symbol_xs = [p[0] for p in symbol_points]
        symbol_ys = [p[1] for p in symbol_points]
        if ypoints and symbol_ys:
            y_max = max(
                0.5 * max(max(ypoints), max(symbol_ys)),
                2 * max(max(ypoints), max(symbol_ys)),
            )
            y_min = min(
                0.5 * min(min(ypoints), min(symbol_ys)),
                2 * min(min(ypoints), min(symbol_ys)),
            )
        elif symbol_ys:
            y_max = max(symbol_ys)
            y_min = min(symbol_ys)
        elif ypoints:
            y_max = max(ypoints)
            y_min = min(ypoints)
        else:
            raise NoCurveDataError("No points found, make sure data is properly initialized")

        if xpoints:
            curve_line = PlotCurveLine(
                xs=xpoints,
                ys=ypoints,
                color=line_color,
                linestyle=line_style,
                linewidth=line_width / 2,
            )
        else:
            curve_line = None

        if symbol_xs:
            curve_symbols = PlotCurveSymbols(
                xs=symbol_xs,
                ys=symbol_ys,
                color=symbol_color,
                linewidth=0,
                markerfacecolor=marker_color,
                markersize=marker_size / 2,
                marker=symbol_type,
                markeredgewidth=symbol_line_width / 2,
            )
        else:
            curve_symbols = None

        if graph_type in {"data", "dynamic_aperture", "phase_space"}:
            return cls(
                info=curve_info,
                line=curve_line,
                symbol=curve_symbols,
            )

        if graph_type in {"wave.0", "wave.a", "wave.b"}:
            # Wave region boundaries
            # wave analysis rectangles
            if wave_params is None:
                raise ValueError(f"wave_params required for graph type: {graph_type}")
            if symbol_color in {"blue", "navy", "cyan", "green", "purple"}:
                wave_color = "orange"
            else:
                wave_color = "blue"

            patches = []
            if graph_type in {"wave.0", "wave.a"}:
                a1, a2 = wave_params["ix_a1"], wave_params["ix_a2"]
                patches.append(
                    PlotPatchRectangle(
                        xy=(a1, y_min),
                        width=a2 - a1,
                        height=y_max - y_min,
                        fill=False,
                        color=wave_color,
                    )
                )

            if graph_type in {"wave.0", "wave.b"}:
                b1, b2 = wave_params["ix_b1"], wave_params["ix_b2"]
                patches.append(
                    PlotPatchRectangle(
                        xy=(b1, y_min),
                        width=b2 - b1,
                        height=y_max - y_min,
                        fill=False,
                        color=wave_color,
                    )
                )

            return cls(
                info=curve_info,
                line=curve_line,
                symbol=curve_symbols,
                patches=patches,
            )

        if graph_type == "histogram":
            assert histogram_info is not None
            return cls(
                info=curve_info,
                line=None,
                symbol=None,
                histogram=PlotHistogram(
                    xs=xpoints,
                    bins=int(histogram_info["number"]),
                    weights=ypoints,
                    histtype="step",
                    color=symbol_color,
                ),
            )

        raise NotImplementedError(f"graph_type: {graph_type}")


@dataclasses.dataclass
class BasicGraph(GraphBase):
    curves: List[PlotCurve] = Field(default_factory=list)

    def clamp_x_range(self, x0: Optional[float], x1: Optional[float]) -> Tuple[float, float]:
        if x0 is None:
            x0 = self.get_x_range()[0]
        if x1 is None:
            x1 = self.get_x_range()[1]

        if self.is_s_plot:
            # Don't go to negative 's' values
            x0 = max((0.0, x0))
        return (x0, x1)

    @property
    def is_s_plot(self) -> bool:
        if self.region_info["x_axis_type"] == "s":
            return True
        return self.info["x_label"].lower().replace(" ", "") in {"s[m]", "s(m)"}

    def get_x_range(self) -> Tuple[float, float]:
        return (self.info["x_min"], self.info["x_max"])

    def get_num_points(self) -> int:
        for curve in self.curves:
            if curve.line is not None:
                return len(curve.line.xs)
        return 401  # per the docs, this is the default for n_curve_points

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        region_name: str,
        graph_name: str,
        *,
        info: Optional[PlotGraphInfo] = None,
    ) -> BasicGraph:
        if info is None:
            info = get_plot_graph_info(tao, region_name, graph_name)
        region_info = cast(PlotRegionInfo, tao.plot1(region_name))

        graph_type = info["graph^type"]
        if graph_type in {"lat_layout", "floor_plan", "key_table"}:
            raise ValueError(f"Incorrect graph type: {graph_type} for {cls.__name__}")

        if info["why_invalid"]:
            raise GraphInvalidError(f"Graph not valid: {info['why_invalid']}")

        all_curve_names = [info[f"curve[{idx}]"] for idx in range(1, info["num_curves"] + 1)]
        curves = []
        for curve_name in all_curve_names:
            try:
                curve = PlotCurve.from_tao(tao, region_name, graph_name, curve_name)
            except NoCurveDataError:
                logger.warning(f"No curve data for {region_name}.{graph_name}.{curve_name}")
            else:
                curves.append(curve)

        return cls(
            info=info,
            region_info=region_info,
            region_name=region_name,
            graph_name=graph_name,
            curves=curves,
            show_axes=info["draw_axes"],
            title="{title} {title_suffix}".format(**info),
            xlabel=info["x_label"],
            ylabel=info["y_label"],
            draw_grid=info["draw_grid"],
            xlim=(info["x_min"], info["x_max"]),
            ylim=(info["y_min"], info["y_max"]),
            draw_legend=info["draw_curve_legend"],
        )

    def plot(self, ax: Optional[matplotlib.axes.Axes] = None):
        if ax is None:
            _, ax = plt.subplots()
            assert ax is not None

        for curve in self.curves:
            assert not curve.info["use_y2"], "TODO: y2 support"
            curve.plot(ax)

        if self.draw_legend and any(curve.legend_label for curve in self.curves):
            ax.legend()

        self._setup_axis(ax)
        return ax


@dataclasses.dataclass
class LatticeLayoutElement:
    info: PlotLatLayoutInfo
    patches: List[PlotPatch]
    lines: List[List[Point]]
    annotations: List[PlotAnnotation]
    color: str
    width: float

    def plot(self, ax: matplotlib.axes.Axes):
        ax.add_collection(
            matplotlib.collections.LineCollection(
                self.lines,
                colors=pgplot.mpl_color(self.color),
                linewidths=self.width,
            )
        )
        for patch in self.patches:
            patch.plot(ax)
        for annotation in self.annotations:
            annotation.plot(ax)

    @classmethod
    def from_info(cls, graph_info: PlotGraphInfo, info: PlotLatLayoutInfo, y2_floor: float):
        s1 = info["ele_s_start"]
        s2 = info["ele_s_end"]
        y1 = info["y1"]
        y2 = -info["y2"]  # Note negative sign.
        width = info["line_width"]
        color = info["color"]
        shape = info["shape"]
        name = info["label_name"]

        patches = []
        lines = []
        annotations = []

        if ":" in shape:
            _shape_prefix, shape = shape.split(":", 1)
        else:
            _shape_prefix, shape = "", shape

        # Normal case where element is not wrapped around ends of lattice.
        if s2 - s1 > 0:
            box_patch = PlotPatchRectangle(
                xy=(s1, y1),
                width=s2 - s1,
                height=y2 - y1,
                linewidth=width,
                color=color,
                fill=False,
            )
            s_mid = (s1 + s2) / 2
            y_mid = (y1 + y2) / 2

            if shape == "box":
                patches.append(box_patch)
            elif shape == "xbox":
                patches.append(box_patch)
                lines.extend(
                    [
                        [(s1, y1), (s2, y2)],
                        [(s1, y2), (s2, y1)],
                    ]
                )
            elif shape == "x":
                lines.extend(
                    [
                        [(s1, y1), (s2, y2)],
                        [(s1, y2), (s2, y1)],
                    ]
                )
            elif shape == "bow_tie":
                lines.append([(s1, y1), (s2, y2), (s2, y1), (s1, y2), (s1, y1)])
            elif shape == "rbow_tie":
                lines.append([(s1, y1), (s2, y2), (s1, y2), (s2, y1), (s1, y1)])
            elif shape == "diamond":
                lines.append([(s1, 0), (s_mid, y1), (s2, 0), (s_mid, y2), (s1, 0)])
            elif shape == "circle":
                patches.append(
                    PlotPatchEllipse(
                        xy=(s_mid, 0),
                        width=y1 - y2,
                        height=y1 - y2,
                        linewidth=width,
                        color=color,
                        fill=False,
                    )
                )
            elif shape == "u_triangle":
                patches.append(
                    PlotPatchPolygon(
                        vertices=[
                            (s1, y2),
                            (s2, y2),
                            (s_mid, y1),
                        ],
                        linewidth=width,
                        color=color,
                        fill=False,
                    )
                )
            elif shape == "d_triangle":
                patches.append(
                    PlotPatchPolygon(
                        vertices=[
                            (s1, y1),
                            (s2, y1),
                            (s_mid, y2),
                        ],
                        linewidth=width,
                        color=color,
                        fill=False,
                    )
                )
            elif shape == "l_triangle":
                patches.append(
                    PlotPatchPolygon(
                        vertices=[
                            (s1, y_mid),
                            (s2, y2),
                            (s2, y1),
                        ],
                        linewidth=width,
                        color=color,
                        fill=False,
                    )
                )
            elif shape == "r_triangle":
                patches.append(
                    PlotPatchPolygon(
                        vertices=[
                            (s1, y1),
                            (s1, y2),
                            (s2, y_mid),
                        ],
                        linewidth=width,
                        color=color,
                        fill=False,
                    )
                )
            else:
                raise NotImplementedError(shape)

            if name:
                annotations.append(
                    PlotAnnotation(
                        x=(s1 + s2) / 2,
                        y=1.1 * y2_floor,
                        text=name,
                        horizontalalignment="center",
                        verticalalignment="top",
                        clip_on=False,
                        color=color,
                        rotation=90,
                    )
                )

        else:
            # Case where element is wrapped round the lattice ends.
            s_min = max((graph_info["x_min"], s1 + (s1 + s2) / 2.0))
            s_max = min((graph_info["x_max"], s1 - (s1 + s2) / 2.0))

            for xs, ys in _get_wrapped_shape_coords(
                shape=shape,
                s1=s1,
                s2=s2,
                y1=y1,
                y2=y2,
                s_min=s_min,
                s_max=s_max,
            ):
                lines.append(list(zip(xs, ys)))

            annotations.append(
                PlotAnnotation(
                    x=s_max,
                    y=1.1 * y2_floor,
                    text=name,
                    horizontalalignment="right",
                    verticalalignment="top",
                    clip_on=True,
                    color=color,
                )
            )
            annotations.append(
                PlotAnnotation(
                    x=s_min,
                    y=1.1 * y2_floor,
                    text=name,
                    horizontalalignment="left",
                    verticalalignment="top",
                    clip_on=True,
                    color=color,
                )
            )

        return cls(
            info=info,
            patches=patches,
            lines=lines,
            color=color,
            width=width,
            annotations=annotations,
        )


@dataclasses.dataclass
class LatticeLayoutGraph(GraphBase):
    elements: List[LatticeLayoutElement] = Field(default_factory=list)
    border_xlim: Point = _point_field
    universe: int = 0
    branch: int = 0
    y2_floor: float = 0
    fields: List[LatticeLayoutField] = Field(default_factory=list)
    show_fields: bool = False

    def update_fields(self, tao: Tao) -> List[LatticeLayoutField]:
        field_elems = [
            elem
            for elem in self.elements
            if elem.info["label_name"]
            and tao.ele_head(elem.info["label_name"])["key"] in {"Quadrupole"}
        ]
        fields = [
            LatticeLayoutField.from_tao(tao, ele_id=elem.info["label_name"])
            for elem in field_elems
        ]
        self.fields = fields
        return fields

    @property
    def is_s_plot(self) -> bool:
        return True

    def plot_fields(self, ax: Optional[matplotlib.axes.Axes] = None):
        if ax is None:
            _, ax = plt.subplots()
        assert ax is not None

        field_data = [field.by for field in self.fields]
        min_field = np.min(field_data or [0])
        max_field = np.max(field_data or [1])
        for field in self.fields:
            ax.pcolormesh(
                np.asarray(field.s),
                np.asarray(field.x) * 1e3,
                np.asarray(field.by),
                vmin=min_field,
                vmax=max_field,
                cmap="PRGn_r",
            )
        return ax

    def plot(self, ax: Optional[matplotlib.axes.Axes] = None):
        if ax is None:
            _, ax = plt.subplots()
        assert ax is not None

        ax.axhline(y=0, color="Black", linewidth=1)

        if self.show_fields:
            self.plot_fields(ax)
            # ax.colorbar(label="B1_GRADIENT (T/m)")
            # plt.xlabel(r"$s$ (m)")
            # plt.ylabel(r"$x$ (mm)")

        for elem in self.elements:
            elem.plot(ax)

        # Invisible line to give the lat layout enough vertical space.
        # Without this, the tops and bottoms of shapes could be cut off
        # ax.plot([0, 0], [-1.7 * self.y_max, 1.3 * self.y_max], alpha=0)

        ax.yaxis.set_visible(False)

        self._setup_axis(ax)
        # ax.set_xticks([elem.info["ele_s_start"] for elem in self.elements])
        # ax.set_xticklabels([elem.info["label_name"] for elem in self.elements], rotation=90)
        ax.grid(visible=False)
        return ax

    @property
    def y_max(self) -> float:
        ele_y1s = [elem.info["y1"] for elem in self.elements]
        ele_y2s = [elem.info["y2"] for elem in self.elements]
        return max(max(ele_y1s), max(ele_y2s))

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        region_name: str = "lat_layout",
        graph_name: str = "g",
        *,
        branch: Optional[int] = None,
        info: Optional[PlotGraphInfo] = None,
    ) -> LatticeLayoutGraph:
        if info is None:
            try:
                info = get_plot_graph_info(tao, region_name, graph_name)
            except RuntimeError:
                raise NoLayoutError(f"No layout named {region_name}.{graph_name}") from None
        region_info = cast(PlotRegionInfo, tao.plot1(region_name))

        graph_type = info["graph^type"]
        if graph_type != "lat_layout":
            raise ValueError(f"Incorrect graph type: {graph_type} for {cls.__name__}")

        universe = 1 if info["ix_universe"] == -1 else info["ix_universe"]
        branch = info["-1^ix_branch"]
        try:
            all_elem_info = tao.plot_lat_layout(ix_uni=universe, ix_branch=branch)
        except RuntimeError as ex:
            if branch != -1:
                raise

            logger.debug(
                f"Lat layout failed for universe={universe} branch={branch}; trying branch 0"
            )
            try:
                all_elem_info = tao.plot_lat_layout(ix_uni=universe, ix_branch=0)
            except RuntimeError:
                logger.error(f"Failed to plot layout: {ex}")
                raise

        all_elem_info = cast(List[PlotLatLayoutInfo], all_elem_info)

        ele_y2s = [elem["y2"] for elem in all_elem_info]
        y2_floor = -max(ele_y2s)  # Note negative sign

        elements = [
            LatticeLayoutElement.from_info(
                graph_info=info,
                info=elem,
                y2_floor=y2_floor,
            )
            for elem in all_elem_info
        ]

        return cls(
            info=info,
            region_info=region_info,
            region_name=region_name,
            graph_name=graph_name,
            xlim=(info["x_min"], info["x_max"]),
            ylim=(info["y_min"], info["y_max"]),
            border_xlim=(1.1 * info["x_min"], 1.1 * info["x_max"]),
            universe=universe,
            branch=branch,
            y2_floor=y2_floor,
            elements=elements,
        )


def _get_wrapped_shape_coords(
    shape: str,
    s1: float,
    s2: float,
    y1: float,
    y2: float,
    s_min: float,
    s_max: float,
):
    """Case where element is wrapped round the lattice ends."""
    if shape == "box":
        return [
            ([s1, s_max], [y1, y1]),
            ([s1, s_max], [y2, y2]),
            ([s_min, s2], [y1, y1]),
            ([s_min, s2], [y2, y2]),
            ([s1, s1], [y1, y2]),
            ([s2, s2], [y1, y2]),
        ]

    if shape == "xbox":
        return [
            ([s1, s_max], [y1, y1]),
            ([s1, s_max], [y2, y2]),
            ([s1, s_max], [y1, 0]),
            ([s1, s_max], [y2, 0]),
            ([s_min, s2], [y1, y1]),
            ([s_min, s2], [y2, y2]),
            ([s_min, s2], [0, y1]),
            ([s_min, s2], [0, y2]),
            ([s1, s1], [y1, y2]),
            ([s2, s2], [y1, y2]),
        ]

    if shape == "x":
        return [
            ([s1, s_max], [y1, 0]),
            ([s1, s_max], [y2, 0]),
            ([s_min, s2], [0, y1]),
            ([s_min, s2], [0, y2]),
        ]

    if shape == "bow_tie":
        return [
            ([s1, s_max], [y1, y1]),
            ([s1, s_max], [y2, y2]),
            ([s1, s_max], [y1, 0]),
            ([s1, s_max], [y2, 0]),
            ([s_min, s2], [y1, y1]),
            ([s_min, s2], [y2, y2]),
            ([s_min, s2], [0, y1]),
            ([s_min, s2], [0, y2]),
        ]

    if shape == "diamond":
        return [
            ([s1, s_max], [0, y1]),
            ([s1, s_max], [0, y2]),
            ([s_min, s2], [y1, 0]),
            ([s_min, s2], [y2, 0]),
        ]

    logger.warning("Unsupported shape: {shape}")
    return []


def _building_wall_to_arc(
    mx: float,
    my: float,
    kx: float,
    ky: float,
    k_radii: float,
    color: str,
    linewidth: float,
):
    (c0x, c0y), (c1x, c1y) = util.circle_intersection(mx, my, kx, ky, abs(k_radii))
    # radius and endpoints specify 2 possible circle centers for arcs
    mpx = (mx + kx) / 2
    mpy = (my + ky) / 2
    if (
        np.arctan2((my - mpy), (mx - mpx))
        < np.arctan2(c0y, c0x)
        < np.arctan2((my - mpy), (mx - mpx))
        and k_radii > 0
    ):
        center = (c1x, c1y)
    elif (
        np.arctan2((my - mpy), (mx - mpx))
        < np.arctan2(c0y, c0x)
        < np.arctan2((my - mpy), (mx - mpx))
        and k_radii < 0
    ):
        center = (c0x, c0y)
    elif k_radii > 0:
        center = (c0x, c0y)
    else:
        center = (c1x, c1y)

    m_angle = 360 + math.degrees(np.arctan2((my - center[1]), (mx - center[0])))
    k_angle = 360 + math.degrees(np.arctan2((ky - center[1]), (kx - center[0])))
    if k_angle > m_angle:
        t1 = m_angle
        t2 = k_angle
    else:
        t1 = k_angle
        t2 = m_angle

    if abs(k_angle - m_angle) > 180:
        t1, t2 = t2, t1

    return PlotPatchArc(
        xy=center,
        width=k_radii * 2,
        height=k_radii * 2,
        theta1=t1,
        theta2=t2,
        color=color,
        linewidth=linewidth,
    )


def _circle_to_patch(
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    off1: float,
    line_width: float,
    color: str,
):
    return PlotPatchCircle(
        xy=(x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2),
        radius=off1,
        linewidth=line_width,
        color=color,
        fill=False,
    )


def _box_to_patch(
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    off1: float,
    off2: float,
    line_width: float,
    color: str,
    angle_start: float,
):
    return PlotPatchRectangle(
        xy=(x1 + off2 * np.sin(angle_start), y1 - off2 * np.cos(angle_start)),
        width=np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2),
        height=off1 + off2,
        linewidth=line_width,
        color=color,
        fill=False,
        angle=math.degrees(angle_start),
    )


def _create_x_lines(
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    off1: float,
    off2: float,
    line_width: float,
    color: str,
    angle_start: float,
) -> List[PlotCurveLine]:
    return [
        PlotCurveLine(
            xs=[x1 + off2 * np.sin(angle_start), x2 - off1 * np.sin(angle_start)],
            ys=[y1 - off2 * np.cos(angle_start), y2 + off1 * np.cos(angle_start)],
            linewidth=line_width,
            color=color,
        ),
        PlotCurveLine(
            xs=[x1 - off1 * np.sin(angle_start), x2 + off2 * np.sin(angle_start)],
            ys=[y1 + off1 * np.cos(angle_start), y2 - off2 * np.cos(angle_start)],
            linewidth=line_width,
            color=color,
        ),
    ]


def _create_x_box(
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    off1: float,
    off2: float,
    line_width: float,
    color: str,
    angle_start: float,
) -> PlotCurveLine:
    px0 = x1 + off2 * np.sin(angle_start)
    py0 = y1 - off2 * np.cos(angle_start)

    px1 = x2 - off1 * np.sin(angle_start)
    py1 = y2 + off1 * np.cos(angle_start)

    px2 = x1 - off1 * np.sin(angle_start)
    py2 = y1 + off1 * np.cos(angle_start)

    px3 = x2 + off2 * np.sin(angle_start)
    py3 = y2 - off2 * np.cos(angle_start)
    return PlotCurveLine(
        xs=[px0, px1, px2, px3, px0, px2, px3, px1],
        ys=[py0, py1, py2, py3, py0, py2, py3, py1],
        linewidth=line_width,
        color=color,
    )


def _create_sbend_box(
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    off1: float,
    off2: float,
    line_width: float,
    color: str,
    angle_start: float,
    angle_end: float,
    rel_angle_start: float,
    rel_angle_end: float,
) -> List[PlotCurveLine]:
    return [
        PlotCurveLine(
            [
                x1 - off1 * np.sin(angle_start - rel_angle_start),
                x1 + off2 * np.sin(angle_start - rel_angle_start),
            ],
            [
                y1 + off1 * np.cos(angle_start - rel_angle_start),
                y1 - off2 * np.cos(angle_start - rel_angle_start),
            ],
            linewidth=line_width,
            color=color,
        ),
        PlotCurveLine(
            [
                x2 - off1 * np.sin(angle_end + rel_angle_end),
                x2 + off2 * np.sin(angle_end + rel_angle_end),
            ],
            [
                y2 + off1 * np.cos(angle_end + rel_angle_end),
                y2 - off2 * np.cos(angle_end + rel_angle_end),
            ],
            linewidth=line_width,
            color=color,
        ),
    ]


def _create_sbend(
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    off1: float,
    off2: float,
    line_width: float,
    color: str,
    angle_start: float,
    angle_end: float,
    rel_angle_start: float,
    rel_angle_end: float,
) -> Tuple[List[PlotCurveLine], List[PlotPatch]]:
    line1 = util.line(
        (x1 - off1 * np.sin(angle_start), y1 + off1 * np.cos(angle_start)),
        (x1 + off2 * np.sin(angle_start), y1 - off2 * np.cos(angle_start)),
    )
    line2 = util.line(
        (x2 - off1 * np.sin(angle_end), y2 + off1 * np.cos(angle_end)),
        (x2 + off2 * np.sin(angle_end), y2 - off2 * np.cos(angle_end + rel_angle_end)),
    )
    try:
        intersection = util.intersect(line1, line2)
    except util.NoIntersectionError:
        lines = [
            PlotCurveLine(
                [
                    x1 - off1 * np.sin(angle_start - rel_angle_start),
                    x2 - off1 * np.sin(angle_end + rel_angle_end),
                ],
                [
                    y1 + off1 * np.cos(angle_start - rel_angle_start),
                    y2 + off1 * np.cos(angle_end + rel_angle_end),
                ],
                linewidth=line_width,
                color=color,
            ),
            PlotCurveLine(
                [
                    x1 + off2 * np.sin(angle_start - rel_angle_start),
                    x2 + off2 * np.sin(angle_end + rel_angle_end),
                ],
                [
                    y1 - off2 * np.cos(angle_start - rel_angle_start),
                    y2 - off2 * np.cos(angle_end + rel_angle_end),
                ],
                linewidth=line_width,
                color=color,
            ),
        ]
        return lines, []

    # draw sbend edges if bend angle is 0
    angle1 = 360 + math.degrees(
        np.arctan2(
            y1 + off1 * np.cos(angle_start - rel_angle_start) - intersection[1],
            x1 - off1 * np.sin(angle_start - rel_angle_start) - intersection[0],
        )
    )
    angle2 = 360 + math.degrees(
        np.arctan2(
            y2 + off1 * np.cos(angle_end + rel_angle_end) - intersection[1],
            x2 - off1 * np.sin(angle_end + rel_angle_end) - intersection[0],
        )
    )
    # angles of further curve endpoints relative to center of circle
    angle3 = 360 + math.degrees(
        np.arctan2(
            y1 - off2 * np.cos(angle_start - rel_angle_start) - intersection[1],
            x1 + off2 * np.sin(angle_start - rel_angle_start) - intersection[0],
        )
    )
    angle4 = 360 + math.degrees(
        np.arctan2(
            y2 - off2 * np.cos(angle_end + rel_angle_end) - intersection[1],
            x2 + off2 * np.sin(angle_end + rel_angle_end) - intersection[0],
        )
    )
    # angles of closer curve endpoints relative to center of circle

    if abs(angle1 - angle2) < 180:
        a1 = min(angle1, angle2)
        a2 = max(angle1, angle2)
    else:
        a1 = max(angle1, angle2)
        a2 = min(angle1, angle2)

    if abs(angle3 - angle4) < 180:
        a3 = min(angle3, angle4)
        a4 = max(angle3, angle4)
    else:
        a3 = max(angle3, angle4)
        a4 = min(angle3, angle4)
    # determines correct start and end angles for arcs

    rel_sin = np.sin(angle_start - rel_angle_start)
    rel_cos = np.cos(angle_start - rel_angle_start)
    patches: List[PlotPatch] = [
        PlotPatchArc(
            xy=(intersection[0], intersection[1]),
            width=(
                2.0
                * np.sqrt(
                    (x1 - off1 * rel_sin - intersection[0]) ** 2
                    + (y1 + off1 * rel_cos - intersection[1]) ** 2
                )
            ),
            height=(
                2.0
                * np.sqrt(
                    (x1 - off1 * rel_sin - intersection[0]) ** 2
                    + (y1 + off1 * rel_cos - intersection[1]) ** 2
                )
            ),
            theta1=a1,
            theta2=a2,
            linewidth=line_width,
            color=color,
        ),
        PlotPatchArc(
            xy=(intersection[0], intersection[1]),
            width=(
                2.0
                * np.sqrt(
                    (x1 + off2 * rel_sin - intersection[0]) ** 2
                    + (y1 - off2 * rel_cos - intersection[1]) ** 2
                )
            ),
            height=(
                2.0
                * np.sqrt(
                    (x1 + off2 * rel_sin - intersection[0]) ** 2
                    + (y1 - off2 * rel_cos - intersection[1]) ** 2
                )
            ),
            theta1=a3,
            theta2=a4,
            linewidth=line_width,
            color=color,
        ),
    ]
    patch = _sbend_intersection_to_patch(
        intersection=intersection,
        x1=x1,
        x2=x2,
        y1=y1,
        y2=y2,
        off1=off1,
        off2=off2,
        angle_start=angle_start,
        angle_end=angle_end,
        rel_angle_start=rel_angle_start,
        rel_angle_end=rel_angle_end,
    )
    patches.append(patch)
    return [], patches


def _sbend_intersection_to_patch(
    intersection: util.Intersection,
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    off1: float,
    off2: float,
    angle_start: float,
    angle_end: float,
    rel_angle_start: float,
    rel_angle_end: float,
):
    sin_start = np.sin(angle_start - rel_angle_start)
    cos_start = np.cos(angle_start - rel_angle_start)
    sin_end = np.sin(angle_end + rel_angle_end)
    cos_end = np.cos(angle_end + rel_angle_end)

    # corners of sbend
    c1 = (x1 - off1 * sin_start, y1 + off1 * cos_start)
    c2 = (x2 - off1 * sin_end, y2 + off1 * cos_end)
    c3 = (x1 + off2 * sin_start, y1 - off2 * cos_start)
    c4 = (x2 + off2 * sin_end, y2 - off2 * cos_end)

    # radii of sbend arc edges
    outer_radius = np.sqrt(
        (x1 - off1 * sin_start - intersection[0]) ** 2
        + (y1 + off1 * cos_start - intersection[1]) ** 2
    )
    inner_radius = np.sqrt(
        (x1 + off2 * sin_start - intersection[0]) ** 2
        + (y1 - off2 * cos_start - intersection[1]) ** 2
    )
    if angle_start <= angle_end:
        outer_radius *= -1
        inner_radius *= -1

    # midpoints of top and bottom arcs in an sbend
    mid_angle = (angle_start + angle_end) / 2

    top = (
        intersection[0] - outer_radius * np.sin(mid_angle),
        intersection[1] + outer_radius * np.cos(mid_angle),
    )
    bottom = (
        intersection[0] - inner_radius * np.sin(mid_angle),
        intersection[1] + inner_radius * np.cos(mid_angle),
    )

    # corresponding control points for a quadratic Bezier curve that
    # passes through the corners and arc midpoint
    top_cp = (
        2 * (top[0]) - 0.5 * (c1[0]) - 0.5 * (c2[0]),
        2 * (top[1]) - 0.5 * (c1[1]) - 0.5 * (c2[1]),
    )
    bottom_cp = (
        2 * (bottom[0]) - 0.5 * (c3[0]) - 0.5 * (c4[0]),
        2 * (bottom[1]) - 0.5 * (c3[1]) - 0.5 * (c4[1]),
    )

    return PlotPatchSbend(
        spline1=(c1, top_cp, c2),
        spline2=(c4, bottom_cp, c3),
        facecolor="green",
        alpha=0.5,
    )


def _create_bow_tie(
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    off1: float,
    off2: float,
    line_width: float,
    color: str,
    angle_start: float,
) -> PlotCurveLine:
    l1x = [x1 + off2 * np.sin(angle_start), x2 - off1 * np.sin(angle_start)]
    l1y = [y1 - off2 * np.cos(angle_start), y2 + off1 * np.cos(angle_start)]
    l2x = [x1 - off1 * np.sin(angle_start), x2 + off2 * np.sin(angle_start)]
    l2y = [y1 + off1 * np.cos(angle_start), y2 - off2 * np.cos(angle_start)]
    return PlotCurveLine(
        [l1x[0], l1x[1], l2x[0], l2x[1], l1x[0]],
        [l1y[0], l1y[1], l2y[0], l2y[1], l1y[0]],
        linewidth=line_width,
        color=color,
    )


def _create_diamond(
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    off1: float,
    off2: float,
    line_width: float,
    color: str,
    angle_start: float,
):
    l1x1 = x1 + (x2 - x1) / 2 - off1 * np.sin(angle_start)
    l1y1 = y1 + (y2 - y1) / 2 + off1 * np.cos(angle_start)
    l2x1 = x1 + (x2 - x1) / 2 + off2 * np.sin(angle_start)
    l2y1 = y1 + (y2 - y1) / 2 - off2 * np.cos(angle_start)
    return PlotCurveLine(
        [x1, l1x1, x2, l2x1, x1],
        [y1, l1y1, y2, l2y1, y1],
        linewidth=line_width,
        color=color,
    )


@dataclasses.dataclass
class FloorPlanElement:
    branch_index: int
    index: int
    info: FloorPlanElementInfo
    lines: List[PlotCurveLine]
    patches: List[PlotPatch]
    annotations: List[PlotAnnotation]

    def plot(self, ax: matplotlib.axes.Axes):
        for line in self.lines:
            line.plot(ax)
        for patch in self.patches:
            patch.plot(ax)
        for annotation in self.annotations:
            annotation.plot(ax)

    @property
    def name(self) -> str:
        return self.info["label_name"]

    @classmethod
    def from_info(
        cls,
        info: FloorPlanElementInfo,
        plot_page: PlotPage,
    ):
        # Handle some renaming and reduce dictionary key usage
        return cls._from_info(
            info,
            branch_index=info["branch_index"],
            index=info["index"],
            ele_key=info["ele_key"],
            x1=info["end1_r1"],
            y1=info["end1_r2"],
            angle_start=info["end1_theta"],
            x2=info["end2_r1"],
            y2=info["end2_r2"],
            angle_end=info["end2_theta"],
            line_width=info["line_width"],
            shape=info["shape"],
            off1=info["y1"] * plot_page["floor_plan_shape_scale"],
            off2=info["y2"] * plot_page["floor_plan_shape_scale"],
            color=info["color"],
            label_name=info["label_name"],
            # ele_l=info["ele_l"],
            # ele_angle=info["ele_angle"],
            rel_angle_start=info.get("ele_e1", 0.0),
            rel_angle_end=info.get("ele_e", 0.0),
        )

    @classmethod
    def _from_info(
        cls,
        info: FloorPlanElementInfo,
        *,
        branch_index: int,
        index: int,
        ele_key: str,
        x1: float,
        y1: float,
        angle_start: float,
        x2: float,
        y2: float,
        angle_end: float,
        line_width: float,
        shape: str,
        off1: float,
        off2: float,
        color: str,
        label_name: str,
        # Only for sbend:
        rel_angle_start: float = 0.0,
        rel_angle_end: float = 0.0,
    ) -> FloorPlanElement:
        ele_key = ele_key.lower()

        lines: List[PlotCurveLine] = []
        patches: List[PlotPatch] = []
        annotations: List[PlotAnnotation] = []

        if ":" in shape:
            _shape_prefix, shape = shape.split(":", 1)
        else:
            _shape_prefix, shape = "", shape

        if ele_key == "drift" or ele_key == "kicker":
            lines.append(PlotCurveLine(xs=[x1, x2], ys=[y1, y2], color="black"))

        if off1 == 0 and off2 == 0 and ele_key != "sbend" and color:
            lines.append(
                PlotCurveLine(xs=[x1, x2], ys=[y1, y2], linewidth=line_width, color=color)
            )

        elif shape == "box" and ele_key != "sbend" and color:
            patches.append(
                _box_to_patch(
                    x1=x1,
                    x2=x2,
                    y1=y1,
                    y2=y2,
                    off1=off1,
                    off2=off2,
                    line_width=line_width,
                    color=color,
                    angle_start=angle_start,
                )
            )

        elif shape == "xbox" and ele_key != "sbend" and color:
            lines.append(
                _create_x_box(
                    x1=x1,
                    x2=x2,
                    y1=y1,
                    y2=y2,
                    off1=off1,
                    off2=off2,
                    line_width=line_width,
                    color=color,
                    angle_start=angle_start,
                )
            )

        elif shape == "x" and ele_key != "sbend" and color:
            lines.extend(
                _create_x_lines(
                    x1=x1,
                    x2=x2,
                    y1=y1,
                    y2=y2,
                    off1=off1,
                    off2=off2,
                    line_width=line_width,
                    color=color,
                    angle_start=angle_start,
                )
            )

        elif shape == "bow_tie" and ele_key != "sbend" and color:
            lines.append(
                _create_bow_tie(
                    x1=x1,
                    x2=x2,
                    y1=y1,
                    y2=y2,
                    off1=off1,
                    off2=off2,
                    line_width=line_width,
                    color=color,
                    angle_start=angle_start,
                )
            )
        elif shape == "diamond" and ele_key != "sbend" and color:
            lines.append(
                _create_diamond(
                    x1=x1,
                    x2=x2,
                    y1=y1,
                    y2=y2,
                    off1=off1,
                    off2=off2,
                    line_width=line_width,
                    color=color,
                    angle_start=angle_start,
                )
            )

        elif shape == "circle" and ele_key != "sbend" and color:
            patches.append(
                _circle_to_patch(
                    x1=x1,
                    x2=x2,
                    y1=y1,
                    y2=y2,
                    off1=off1,
                    line_width=line_width,
                    color=color,
                )
            )

        elif shape == "box" and ele_key == "sbend" and color:
            lines.extend(
                _create_sbend_box(
                    x1=x1,
                    x2=x2,
                    y1=y1,
                    y2=y2,
                    off1=off1,
                    off2=off2,
                    line_width=line_width,
                    color=color,
                    angle_start=angle_start,
                    angle_end=angle_end,
                    rel_angle_start=rel_angle_start,
                    rel_angle_end=rel_angle_end,
                )
            )
            sbend_lines, sbend_patches = _create_sbend(
                x1=x1,
                x2=x2,
                y1=y1,
                y2=y2,
                off1=off1,
                off2=off2,
                line_width=line_width,
                color=color,
                angle_start=angle_start,
                angle_end=angle_end,
                rel_angle_start=rel_angle_start,
                rel_angle_end=rel_angle_end,
            )
            lines.extend(sbend_lines or [])
            patches.extend(sbend_patches or [])
        elif shape:
            raise ValueError(f"unhandled shape: {shape}")

        if label_name and color:
            annotation_angle = math.degrees((angle_end + angle_start) / 2)
            if np.sin(((angle_end + angle_start) / 2)) > 0:
                annotation_angle += -90
                align = "right"
            else:
                annotation_angle += 90
                align = "left"

            annotations.append(
                PlotAnnotation(
                    x=x1 + (x2 - x1) / 2 - 1.3 * off1 * np.sin(angle_start),
                    y=y1 + (y2 - y1) / 2 + 1.3 * off1 * np.cos(angle_start),
                    text=label_name,
                    horizontalalignment=align,
                    verticalalignment="center",
                    color="black",
                    rotation=annotation_angle,
                    clip_on=True,
                    rotation_mode="anchor",
                )
            )

        return cls(
            branch_index=branch_index,
            index=index,
            info=info,
            lines=lines,
            patches=patches,
            annotations=annotations,
        )
        # path approximating sbend region for clickable region on graph using lines and quadratic Bezier curves

        # else:  # for non sbend click detection
        #     corner1[str(i)] = [
        #         x1 - off1 * np.sin(angle_start),
        #         y1 + off1 * np.cos(angle_start),
        #     ]
        #     corner2[str(i)] = [
        #         x2 - off1 * np.sin(angle_start),
        #         y2 + off1 * np.cos(angle_start),
        #     ]
        #     corner3[str(i)] = [
        #         x1 + off2 * np.sin(angle_start),
        #         y1 - off2 * np.cos(angle_start),
        #     ]
        #     corner4[str(i)] = [
        #         x2 + off2 * np.sin(angle_start),
        #         y2 - off2 * np.cos(angle_start),
        #     ]
        # coordinates of corners of a floor plan element for clickable region


def sort_building_wall_graph_info(
    info: List[BuildingWallGraphInfo],
) -> Dict[int, Dict[int, BuildingWallGraphInfo]]:
    res = {}
    for item in info:
        index = item["index"]
        point = item["point"]
        res.setdefault(index, {})[point] = item
    return res


@dataclasses.dataclass
class BuildingWalls:
    building_wall_graph: List[BuildingWallGraphInfo] = Field(default_factory=list)
    lines: List[PlotCurveLine] = Field(default_factory=list)
    patches: List[PlotPatch] = Field(default_factory=list)

    def plot(self, ax: matplotlib.axes.Axes):
        for line in self.lines:
            line.plot(ax)
        for patch in self.patches:
            patch.plot(ax)

    @classmethod
    def from_info(
        cls,
        building_wall_graph: List[BuildingWallGraphInfo],
        wall_list: List[BuildingWallInfo],
    ) -> BuildingWalls:
        walls = sort_building_wall_graph_info(building_wall_graph)
        wall_info_by_index = {info["index"]: info for info in wall_list}
        lines = []
        patches = []
        for wall_idx, wall in walls.items():
            wall_points = list(reversed(wall.values()))
            wall_info = wall_info_by_index[wall_idx]
            color = wall_info["color"]
            line_width = wall_info["line_width"]
            for point, next_point in zip(wall_points, wall_points[1:]):
                radius = point["radius"]
                p0x, p0y = point["offset_x"], point["offset_y"]
                p1x, p1y = next_point["offset_x"], next_point["offset_y"]
                if np.isclose(radius, 0.0):
                    lines.append(
                        PlotCurveLine(
                            xs=[p0x, p1x],
                            ys=[p0y, p1y],
                            color=color,
                            linewidth=line_width,
                        )
                    )

                else:
                    patches.append(
                        _building_wall_to_arc(
                            mx=p1x,
                            my=p1y,
                            kx=p0x,
                            ky=p0y,
                            k_radii=radius,
                            color=color,
                            linewidth=line_width,
                        )
                    )

        # plot floor plan building walls
        return cls(building_wall_graph=building_wall_graph, lines=lines, patches=patches)


@dataclasses.dataclass
class FloorOrbits:
    info: List[FloorOrbitInfo]
    curve: PlotCurveSymbols

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        region_name: str,
        graph_name: str,
        color: str,
    ) -> FloorOrbits:
        floor_orbit_info = cast(
            List[FloorOrbitInfo],
            tao.floor_orbit(f"{region_name}.{graph_name}"),
        )

        xs = []
        ys = []
        for info in floor_orbit_info:
            if info["ele_key"] == "x":
                xs.extend(info["orbits"])
            elif info["ele_key"] == "y":
                ys.extend(info["orbits"])

        return cls(
            info=floor_orbit_info,
            curve=PlotCurveSymbols(
                xs=xs,
                ys=ys,
                color=color,
                markerfacecolor=color,
                markersize=1,
                marker="circle_filled",
                markeredgewidth=1,
            ),
        )

    def plot(self, ax: matplotlib.axes.Axes):
        self.curve.plot(ax)


@dataclasses.dataclass
class FloorPlanGraph(GraphBase):
    building_walls: BuildingWalls = Field(default_factory=BuildingWalls)
    floor_orbits: Optional[FloorOrbits] = None
    elements: List[FloorPlanElement] = Field(default_factory=list)

    @property
    def is_s_plot(self) -> bool:
        return False

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        region_name: str,
        graph_name: str,
        *,
        info: Optional[PlotGraphInfo] = None,
        plot_page: Optional[PlotPage] = None,
    ) -> FloorPlanGraph:
        full_name = f"{region_name}.{graph_name}"
        if info is None:
            info = get_plot_graph_info(tao, region_name, graph_name)
        if plot_page is None:
            plot_page = cast(PlotPage, tao.plot_page())
        region_info = cast(PlotRegionInfo, tao.plot1(region_name))

        graph_type = info["graph^type"]
        if graph_type != "floor_plan":
            raise ValueError(f"Incorrect graph type: {graph_type} for {cls.__name__}")

        elem_infos = cast(
            List[FloorPlanElementInfo],
            tao.floor_plan(full_name),
        )
        elements = [FloorPlanElement.from_info(fpe_info, plot_page) for fpe_info in elem_infos]
        building_walls = BuildingWalls.from_info(
            building_wall_graph=cast(
                List[BuildingWallGraphInfo],
                tao.building_wall_graph(full_name),
            ),
            wall_list=cast(List[BuildingWallInfo], tao.building_wall_list()),
        )
        floor_orbits = None
        if float(info["floor_plan_orbit_scale"]) != 0:
            floor_orbits = FloorOrbits.from_tao(
                tao,
                region_name=region_name,
                graph_name=graph_name,
                color=info["floor_plan_orbit_color"].lower(),
            )

        return cls(
            info=info,
            region_info=region_info,
            region_name=region_name,
            graph_name=graph_name,
            elements=elements,
            building_walls=building_walls,
            floor_orbits=floor_orbits,
            title="{title} {title_suffix}".format(**info),
            xlabel=info["x_label"],
            ylabel=info["y_label"],
            draw_grid=info["draw_grid"],
            xlim=(info["x_min"], info["x_max"]),
            ylim=(info["y_min"], info["y_max"]),
            draw_legend=info["draw_curve_legend"],
        )

    def plot(self, ax: Optional[matplotlib.axes.Axes] = None):
        if ax is None:
            _, ax = plt.subplots()
            assert ax is not None

        for elem in self.elements:
            elem.plot(ax)

        self.building_walls.plot(ax)
        if self.floor_orbits is not None:
            self.floor_orbits.plot(ax)

        self._setup_axis(ax, xticks=False, yticks=False)
        return ax


def get_plots_in_region(tao: Tao, region_name: str):
    plot1_info = tao.plot1(region_name)

    if "num_graphs" not in plot1_info:
        raise RuntimeError("Plotting disabled?")

    return [plot1_info[f"graph[{idx}]"] for idx in range(1, plot1_info["num_graphs"] + 1)]


def make_graph(
    tao: Tao,
    region_name: str,
    graph_name: str,
) -> AnyGraph:
    graph_info = get_plot_graph_info(tao, region_name, graph_name)
    graph_type = graph_info["graph^type"]

    logger.debug(f"Creating graph {region_name}.{graph_name} ({graph_type})")

    if graph_type == "floor_plan":
        return FloorPlanGraph.from_tao(
            tao=tao,
            region_name=region_name,
            graph_name=graph_name,
            info=graph_info,
        )
    if graph_type == "lat_layout":
        return LatticeLayoutGraph.from_tao(
            tao,
            region_name=region_name,
            graph_name=graph_name,
            info=graph_info,
        )
    if graph_type == "key_table":
        raise UnsupportedGraphError(graph_type)

    return BasicGraph.from_tao(
        tao,
        region_name=region_name,
        graph_name=graph_name,
        info=graph_info,
    )


def get_plot_graph_info(tao: Tao, region_name: str, graph_name: str) -> PlotGraphInfo:
    return cast(PlotGraphInfo, tao.plot_graph(f"{region_name}.{graph_name}"))


def find_unused_plot_region(tao: Tao, skip: Set[str]) -> str:
    for info in tao.plot_list("r"):
        region_name = info["region"]
        if region_name not in skip and not info["plot_name"]:
            return region_name
    raise ValueError("No more available plot regions")


AnyGraph = Union[BasicGraph, LatticeLayoutGraph, FloorPlanGraph]
T_GraphType = TypeVar("T_GraphType")
T_LatticeLayoutGraph = TypeVar("T_LatticeLayoutGraph")
T_FloorPlanGraph = TypeVar("T_FloorPlanGraph")


class GraphManager(Generic[T_GraphType, T_LatticeLayoutGraph, T_FloorPlanGraph]):
    _key_: ClassVar[str] = "GraphManager"

    tao: Tao
    regions: Dict[str, List[T_GraphType]]
    _to_place: Dict[str, str]
    _graph_name_to_regions: Dict[str, Set[str]]
    _lattice_layout_graph_type: Type[T_LatticeLayoutGraph]
    _floor_plan_graph_type: Type[T_FloorPlanGraph]

    def __init__(self, tao: Tao) -> None:
        self.tao = tao
        self.regions = {}
        self._to_place = {}
        self._graph_name_to_regions = {}

    def _update_place_buffer(self):
        for item in self.tao.place_buffer():
            region = item["region"]
            graph = item["graph"]
            if graph == "none":
                if region == "*":
                    self._to_place.clear()
                else:
                    self._to_place.pop(region, None)
            else:
                self._to_place[region] = graph

    @property
    def to_place(self) -> Dict[str, str]:
        self._update_place_buffer()
        return self._to_place

    def get_lattice_layout_graph(self) -> T_LatticeLayoutGraph:
        for region in self.regions.values():
            for graph in region:
                if isinstance(graph, self._lattice_layout_graph_type):
                    return graph

        (graph,) = self.place("lat_layout")
        assert isinstance(graph, self._lattice_layout_graph_type)
        return graph

    def get_floor_plan_graph(self) -> T_FloorPlanGraph:
        for region in self.regions.values():
            for graph in region:
                if isinstance(graph, self._floor_plan_graph_type):
                    return graph
        (graph,) = self.place("floor_plan")
        assert isinstance(graph, self._floor_plan_graph_type)
        return graph

    @property
    def lattice_layout_graph(self) -> T_LatticeLayoutGraph:
        return self.get_lattice_layout_graph()

    @property
    def floor_plan_graph(self) -> T_FloorPlanGraph:
        return self.get_floor_plan_graph()

    def get_region_for_graph(self, graph_name: str) -> str:
        for region_name, to_place in self.to_place.items():
            if to_place == graph_name:
                logger.debug("Graph %s found in region %s", graph_name, region_name)
                return region_name

        if self._graph_name_to_regions.get(graph_name):
            logger.debug(
                "Graph %s reusing an existing region: %s",
                graph_name,
                self._graph_name_to_regions.get(graph_name),
            )
            return sorted(self._graph_name_to_regions[graph_name])[0]
        region_name = find_unused_plot_region(self.tao, set(self.to_place))
        logger.debug("New region for graph %s: %s", graph_name, region_name)
        return region_name

    def place_all(
        self,
        *,
        ignore_invalid: bool = True,
    ) -> Dict[str, List[T_GraphType]]:
        to_place = list(self.to_place.items())
        self.to_place.clear()

        logger.debug("Placing all plots: %s", to_place)
        result = {}
        for region_name, graph_name in to_place:
            result[region_name] = self.place(
                graph_name=graph_name,
                region_name=region_name,
                ignore_invalid=ignore_invalid,
            )

        return result

    def update_region(
        self,
        region_name: str,
        graph_name: str,
        ignore_invalid: bool = True,
    ) -> List[T_GraphType]:
        self._clear_region(region_name)

        result = []
        plot_names = get_plots_in_region(self.tao, region_name)
        for plot_name in plot_names:
            try:
                result.append(self.make_graph(region_name, plot_name))
            except GraphInvalidError as ex:
                if ignore_invalid:
                    logger.warning(f"Invalid graph in region {region_name}: {ex}")
                    continue
                raise

        self.regions[region_name] = result
        self._graph_name_to_regions.setdefault(graph_name, set()).add(region_name)
        logger.debug(
            "Updating region: %s graph: %s generated %d plots",
            region_name,
            graph_name,
            len(result),
        )
        return result

    def place(
        self,
        graph_name: str,
        *,
        region_name: Optional[str] = None,
        ignore_invalid: bool = True,
    ) -> List[T_GraphType]:
        if region_name is None:
            region_name = self.get_region_for_graph(graph_name)
            logger.debug(f"Picked {region_name} for {graph_name}")

        logger.debug(f"Placing {graph_name} in {region_name}")
        self.tao.cmd(f"place -no_buffer {region_name} {graph_name}")
        return self.update_region(
            region_name=region_name,
            graph_name=graph_name,
            ignore_invalid=ignore_invalid,
        )

    def _clear_region(self, region_name: str):
        if region_name == "*":
            self.regions.clear()
            logger.debug("Clearing all regions")
            self._graph_name_to_regions.clear()
            return

        logger.debug("Clearing region %s", region_name)
        if region_name in self.regions:
            self.regions[region_name].clear()

        for regions in self._graph_name_to_regions.values():
            if region_name in regions:
                regions.remove(region_name)

    def clear(self, region_name: str = "*"):
        try:
            self.tao.cmd(f"place -no_buffer {region_name} none")
        except RuntimeError as ex:
            logger.warning(f"Region clear failed: {ex}")

        self._clear_region(region_name)

    def plot_regions(
        self,
        regions: List[str],
        **kwargs,
    ):
        res = []
        for graph_name, graph_regions in list(self._graph_name_to_regions.items()):
            for region_name in graph_regions:
                if region_name in regions:
                    res.append(self.plot(graph_name, region_name=region_name, **kwargs))
        return res

    def prepare_graphs_by_name(
        self,
        graph_name: str,
        region_name: Optional[str] = None,
        update: bool = True,
        reuse: bool = True,
    ) -> List[T_GraphType]:
        if graph_name and not region_name:
            if reuse:
                region_name = self.get_region_for_graph(graph_name)
            else:
                region_name = find_unused_plot_region(self.tao, skip=set(self.to_place))

        if region_name in self.regions:
            if update:
                return self.update_region(region_name=region_name, graph_name=graph_name)
            return self.regions[region_name]

        return self.place(graph_name=graph_name, region_name=region_name)

    def make_graph(
        self,
        region_name: str,
        graph_name: str,
    ) -> T_GraphType:
        raise NotImplementedError()

    def plot(
        self,
        graph_name: str,
        *,
        region_name: Optional[str] = None,
        include_layout: bool = True,
        update: bool = True,
        reuse: bool = True,
        **kwargs,
    ):
        raise NotImplementedError()


class MatplotlibGraphManager(GraphManager[AnyGraph, LatticeLayoutGraph, FloorPlanGraph]):
    _key_: ClassVar[str] = "mpl"
    _lattice_layout_graph_type = LatticeLayoutGraph
    _floor_plan_graph_type = FloorPlanGraph

    def make_graph(
        self,
        region_name: str,
        graph_name: str,
    ) -> AnyGraph:
        # Matplotlib support is built-in here, so our graph classes (for better
        # or worse) are used directly with this backend
        return make_graph(self.tao, region_name, graph_name)

    @override
    def plot(
        self,
        graph_name: str,
        *,
        region_name: Optional[str] = None,
        show_fields: bool = False,
        include_layout: bool = True,
        width: int = 6,
        height: int = 6,
        layout_height: float = 0.5,
        figsize: Optional[Tuple[int, int]] = None,
        share_x: bool = True,
        update: bool = True,
        reuse: bool = True,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
    ):
        """
        Plot a graph, region, or all placed graphs with Matplotlib.

        To plot a specific graph, specify `graph_name` (optionally `region_name`).
        To plot a specific region, specify `region_name`.
        To plot all placed graphs, specify neither.

        For full details on available parameters, see the specific backend's
        graph manager. For example:

        In [1]: tao.bokeh.plot?
        In [2]: tao.matplotlib.plot?

        Parameters
        ----------
        graph_name : str, optional
            Graph name.
        region_name : str, optional
            Graph region name.
        include_layout : bool, optional
            Include a layout plot at the bottom, if not already placed and if
            appropriate (i.e., another plot uses longitudinal coordinates on
            the x-axis).
        place : bool, default=True
            Place all requested plots prior to continuing.
        update : bool, default=True
            Query Tao to update relevant graphs prior to plotting.
        width : int, optional
            Width of each plot.
        height : int, optional
            Height of each plot.
        layout_height : int, optional
            Height of the layout plot.
        share_x : bool or None, default=None
            Share x-axes where sensible (`None`) or force sharing x-axes (True)
            for all plots.
        backend : {"bokeh", "mpl"}, optional
            The backend to use.  Auto-detects Jupyter and availability of bokeh
            to select a backend.

        Returns
        -------
        matplotlib.Figure
            To gain access to the resulting plot objects, use the backend's
            `plot` method directly.
        array of matplotlib.Axes
        """
        graphs = self.prepare_graphs_by_name(
            graph_name=graph_name,
            region_name=region_name,
            update=update,
            reuse=reuse,
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
            layout_graph = self.get_lattice_layout_graph()
            graphs.append(layout_graph)

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

        if include_layout:
            layout_graph = self.get_lattice_layout_graph()
            layout_graph.show_fields = True
            if show_fields and not layout_graph.fields:
                layout_graph.update_fields(tao=self.tao)

        for ax, graph in zip(gs[:, 0], graphs):
            try:
                graph.plot(ax)
            except UnsupportedGraphError:
                continue

            if xlim is not None:
                ax.set_xlim(xlim)

            if ylim is not None:
                if not isinstance(graph, LatticeLayoutGraph) or len(graphs) == 1:
                    ax.set_ylim(ylim)

        return fig, gs

    __call__ = plot
