from __future__ import annotations

import logging
import math
import typing
from abc import ABC, abstractmethod
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)
from typing_extensions import Literal

import matplotlib.axes
import matplotlib.cm
import matplotlib.collections
import matplotlib.patches
import matplotlib.path
import matplotlib.pyplot as plt
import matplotlib.text
import numpy as np
import pydantic.dataclasses as dataclasses
from matplotlib.ticker import AutoMinorLocator
from pydantic import ConfigDict
from pydantic.fields import Field

from pytao.plotting.settings import TaoGraphSettings

from . import pgplot, layout_shapes, shapes
from .curves import (
    TaoCurveSettings,
    CurveIndexToCurve,
    PlotCurveLine,
    PlotCurveSymbols,
    PlotHistogram,
)
from .patches import (
    PlotPatch,
    PlotPatchArc,
    PlotPatchRectangle,
)
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


class AllPlotRegionsInUseError(Exception):
    pass


T = TypeVar("T")


def _clean_pytao_output(dct: dict, typ: Type[T]) -> T:
    return {key: dct.get(key, None) for key in typ.__required_keys__}


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


# We don't want a single new key from bmad commands to break our implementation,
# so in gneeral we should allow 'extra' keys:
_dcls_config = ConfigDict()

# However, for testing and development, this should be toggled on:
# _dcls_config = ConfigDict(extra="forbid")


@dataclasses.dataclass(config=_dcls_config)
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


_point_field = Field(default_factory=lambda: (0.0, 0.0))


@dataclasses.dataclass(config=_dcls_config)
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

    def update(
        self,
        manager: GraphManager,
        *,
        error_on_new_type: bool = True,
        raise_if_invalid: bool = False,
    ):
        try:
            graphs = [
                graph.get_graph_info()
                for graph in manager.prepare_graphs_by_name(
                    region_name=self.region_name,
                    graph_name=self.graph_name,
                    ignore_invalid=False,
                )
            ]
        except GraphInvalidError:
            if raise_if_invalid:
                raise
            return self

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


@dataclasses.dataclass(config=_dcls_config)
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


@dataclasses.dataclass(config=_dcls_config)
class BasicGraph(GraphBase):
    graph_type: ClassVar[str] = "basic"
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
        else:
            # We'll mutate it to remove curve names below to conform to PlotGraphInfo
            info = info.copy()

        region_info = _clean_pytao_output(tao.plot1(region_name), PlotRegionInfo)

        graph_type = info["graph^type"]
        if graph_type in {"lat_layout", "floor_plan", "key_table"}:
            raise ValueError(f"Incorrect graph type: {graph_type} for {cls.__name__}")

        if info["why_invalid"]:
            raise GraphInvalidError(f"Graph not valid: {info['why_invalid']}")

        curve_keys = [f"curve[{idx}]" for idx in range(1, info["num_curves"] + 1)]
        all_curve_names: List[str] = [info.pop(key) for key in curve_keys]

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


@dataclasses.dataclass(config=_dcls_config)
class LatticeLayoutElement:
    info: PlotLatLayoutInfo
    shape: Optional[layout_shapes.AnyLayoutShape]
    annotations: List[PlotAnnotation]
    color: str
    width: float

    def plot(self, ax: matplotlib.axes.Axes):
        if self.shape is not None:
            self.shape.plot(ax)
        # ax.add_collection(
        #     matplotlib.collections.LineCollection(
        #         self.lines,
        #         colors=pgplot.mpl_color(self.color),
        #         linewidths=self.width,
        #     )
        # )
        # for patch in self.patches:
        #     patch.plot(ax)
        for annotation in self.annotations:
            annotation.plot(ax)

    @classmethod
    def regular_shape(
        cls,
        s1: float,
        s2: float,
        y1: float,
        y2: float,
        width: float,
        color: str,
        shape: str,
        name: str,
        y2_floor: float,
    ) -> Tuple[Optional[layout_shapes.AnyLayoutShape], List[PlotAnnotation]]:
        assert s2 > s1
        if name:
            annotations = [
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
            ]
        else:
            annotations = []

        try:
            shape_cls = layout_shapes.shape_to_class[shape]
        except KeyError:
            logger.debug(f"Unsupported layout shape type: {shape}")
            shape_instance = None
        else:
            shape_instance = shape_cls(
                s1=s1,
                s2=s2,
                y1=y1,
                y2=y2,
                line_width=width,
                color=color,
                name=name,
                fill=False,
            )
        if isinstance(shape_instance, layout_shapes.LayoutTriangle):
            orientation = cast(Literal["u", "d", "l", "r"], shape[0])
            shape_instance.orientation = orientation

        return shape_instance, annotations

    @classmethod
    def wrapped_shape(
        cls,
        s1: float,
        s2: float,
        y1: float,
        y2: float,
        color: str,
        shape: str,
        name: str,
        x_min: float,
        x_max: float,
        y2_floor: float,
    ) -> Tuple[Optional[layout_shapes.AnyWrappedLayoutShape], List[PlotAnnotation]]:
        """
        Element is wrapped around the lattice ends, and s1 >= s2.

        `$ACC_ROOT_DIR/regression_tests/python_test/tao.init_wake` shows off
        this functionality.
        """
        assert s1 >= s2
        s_min = max((x_min, s1 + (s1 + s2) / 2.0))
        s_max = min((x_max, s1 - (s1 + s2) / 2.0))

        try:
            shape_cls = layout_shapes.wrapped_shape_to_class[shape]
        except KeyError:
            logger.debug(f"Unsupported wrappedlayout shape type: {shape}")
            shape_instance = None
        else:
            shape_instance = shape_cls(
                s1=s1,
                s2=s2,
                y1=y1,
                y2=y2,
                color=color,
                s_min=s_min,
                s_max=s_max,
                name=name,
                fill=False,
            )

        annotations = [
            PlotAnnotation(
                x=s_max,
                y=1.1 * y2_floor,
                text=name,
                horizontalalignment="right",
                verticalalignment="top",
                clip_on=True,
                color=color,
            ),
            PlotAnnotation(
                x=s_min,
                y=1.1 * y2_floor,
                text=name,
                horizontalalignment="left",
                verticalalignment="top",
                clip_on=True,
                color=color,
            ),
        ]

        return shape_instance, annotations

    @classmethod
    def from_info(
        cls,
        graph_info: PlotGraphInfo,
        info: PlotLatLayoutInfo,
        y2_floor: float,
        plot_page: PlotPage,
    ):
        s1 = info["ele_s_start"]
        s2 = info["ele_s_end"]
        y1 = info["y1"] * plot_page["lat_layout_shape_scale"]
        y2 = -info["y2"] * plot_page["lat_layout_shape_scale"]  # Note negative sign.
        width = info["line_width"]
        color = info["color"]
        shape = info["shape"]
        name = info["label_name"]

        if ":" in shape:
            _shape_prefix, shape = shape.split(":", 1)
        else:
            _shape_prefix, shape = "", shape

        # Normal case where element is not wrapped around ends of lattice.
        if s2 > s1:
            shape, annotations = cls.regular_shape(
                s1=s1,
                s2=s2,
                y1=y1,
                y2=y2,
                width=width,
                color=color,
                shape=shape,
                name=name,
                y2_floor=y2_floor,
            )

        else:
            shape, annotations = cls.wrapped_shape(
                s1=s1,
                s2=s2,
                y1=y1,
                y2=y2,
                color=color,
                shape=shape,
                name=name,
                x_min=graph_info["x_min"],
                x_max=graph_info["x_max"],
                y2_floor=y2_floor,
            )

        return cls(
            info=info,
            shape=shape,
            color=color,
            width=width,
            annotations=annotations,
        )


@dataclasses.dataclass(config=_dcls_config)
class LatticeLayoutGraph(GraphBase):
    graph_type: ClassVar[str] = "lat_layout"
    elements: List[LatticeLayoutElement] = Field(default_factory=list)
    border_xlim: Point = _point_field
    universe: int = 0
    branch: int = 0
    y2_floor: float = 0

    @property
    def is_s_plot(self) -> bool:
        return True

    def plot(self, ax: Optional[matplotlib.axes.Axes] = None):
        if ax is None:
            _, ax = plt.subplots()
        assert ax is not None

        ax.axhline(y=0, color="Black", linewidth=1)

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
        plot_page: Optional[PlotPage] = None,
    ) -> LatticeLayoutGraph:
        if info is None:
            try:
                info = get_plot_graph_info(tao, region_name, graph_name)
            except RuntimeError:
                raise NoLayoutError(f"No layout named {region_name}.{graph_name}") from None

        if plot_page is None:
            plot_page = cast(PlotPage, tao.plot_page())

        region_info = _clean_pytao_output(tao.plot1(region_name), PlotRegionInfo)

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
                plot_page=plot_page,
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


@dataclasses.dataclass(config=_dcls_config)
class FloorPlanElement:
    branch_index: int
    index: int
    info: FloorPlanElementInfo
    annotations: List[PlotAnnotation]
    shape: Optional[shapes.AnyShape]

    def plot(self, ax: matplotlib.axes.Axes):
        if self.shape is not None:
            self.shape.plot(ax)
        for annotation in self.annotations:
            annotation.plot(ax)

    @property
    def name(self) -> str:
        return self.info["label_name"]

    @classmethod
    def from_info(
        cls,
        info: FloorPlanElementInfo,
        graph_info: PlotGraphInfo,
        plot_page: PlotPage,
    ):
        is_absolute = graph_info["floor_plan_size_is_absolute"]
        floor_plan_shape_scale = plot_page["floor_plan_shape_scale"]
        if is_absolute or floor_plan_shape_scale == 1.0:
            # plot_page floor_plan_size_is_absolute=False means coordinates are
            # in units of points (72 DPI and as such 72 points per "inch" of
            # screen real estate).
            # This is not always set manually or correctly.
            # Let's assume (for now) that the default shape scale of 1.0 is
            # incorrect and that DPI units are to be assumed...
            scale = 1 / 72.0
        else:
            scale = 1.0

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
            off1=info["y1"] * scale,
            off2=info["y2"] * scale,
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

        annotations: List[PlotAnnotation] = []

        if ":" in shape:
            _shape_prefix, shape = shape.split(":", 1)
        else:
            _shape_prefix, shape = "", shape

        shape_cls = {
            "drift": shapes.DriftLine,
            "kicker": shapes.KickerLine,
            "box": shapes.Box,
            "xbox": shapes.XBox,
            "x": shapes.LetterX,
            "bow_tie": shapes.BowTie,
            "diamond": shapes.Diamond,
            "circle": shapes.Circle,
        }.get(shape, None)

        if ele_key == "sbend" and shape == "box":
            # An SBend box is a potentially curvy shape
            shape_cls = shapes.SBend
        elif off1 == 0 and off2 == 0:
            # Zero width/height -> just a line segment
            shape_cls = shapes.LineSegment

        if not color and shape_cls not in (shapes.DriftLine, shapes.KickerLine):
            # Don't draw colorless shapes, with a couple exceptions
            shape_cls = None

        if shape and not shape_cls:
            raise ValueError(f"Unhandled shape: {shape}")

        if shape_cls is None:
            shape_instance = None
        else:
            shape_instance = shape_cls(
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
                name=label_name,
            )

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
            annotations=annotations,
            shape=shape_instance,
        )


def sort_building_wall_graph_info(
    info: List[BuildingWallGraphInfo],
) -> Dict[int, Dict[int, BuildingWallGraphInfo]]:
    res = {}
    for item in info:
        index = item["index"]
        point = item["point"]
        res.setdefault(index, {})[point] = item
    return res


@dataclasses.dataclass(config=_dcls_config)
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
                        PlotPatchArc.from_building_wall(
                            mx=p1x,
                            my=p1y,
                            kx=p0x,
                            ky=p0y,
                            k_radii=radius,
                            color=color,
                            linewidth=line_width,
                        )
                    )

        return cls(building_wall_graph=building_wall_graph, lines=lines, patches=patches)


@dataclasses.dataclass(config=_dcls_config)
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


@dataclasses.dataclass(config=_dcls_config)
class FloorPlanGraph(GraphBase):
    graph_type: ClassVar[str] = "floor_plan"
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
        region_info = _clean_pytao_output(tao.plot1(region_name), PlotRegionInfo)

        graph_type = info["graph^type"]
        if graph_type != "floor_plan":
            raise ValueError(f"Incorrect graph type: {graph_type} for {cls.__name__}")

        elem_infos = cast(
            List[FloorPlanElementInfo],
            tao.floor_plan(full_name),
        )
        elements = [
            FloorPlanElement.from_info(
                info=fpe_info,
                graph_info=info,
                plot_page=plot_page,
            )
            for fpe_info in elem_infos
        ]
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

    raise AllPlotRegionsInUseError("No more available plot regions.")


AnyGraph = Union[BasicGraph, LatticeLayoutGraph, FloorPlanGraph]


class GraphManager(ABC):
    """
    Graph backend manager base class.
    """

    _key_: ClassVar[str] = "GraphManager"

    tao: Tao
    regions: Dict[str, List[AnyGraph]]
    _to_place: Dict[str, str]
    _graph_name_to_regions: Dict[str, Set[str]]

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
        """Graphs to place - region name to graph name."""
        self._update_place_buffer()
        return self._to_place

    @property
    def lattice_layout_graph(self) -> LatticeLayoutGraph:
        """The lattice layout graph.  Placed if not already available."""
        for region in self.regions.values():
            for graph in region:
                if isinstance(graph, LatticeLayoutGraph):
                    return graph

        (graph,) = self.place("lat_layout")
        assert isinstance(graph, LatticeLayoutGraph)
        return graph

    @property
    def floor_plan_graph(self) -> FloorPlanGraph:
        """The floor plan graph. Placed if not already available."""
        for region in self.regions.values():
            for graph in region:
                if isinstance(graph, FloorPlanGraph):
                    return graph
        (graph,) = self.place("floor_plan")
        assert isinstance(graph, FloorPlanGraph)
        return graph

    def get_region_for_graph(self, graph_name: str) -> str:
        """Get a region for placing the graph."""
        for region_name, to_place in self.to_place.items():
            if to_place == graph_name:
                logger.debug("Graph %s found in region %s", graph_name, region_name)
                return region_name

        try:
            region_name = find_unused_plot_region(self.tao, set(self.to_place))
        except AllPlotRegionsInUseError:
            region_name = list(self.regions)[0]
            plots_in_region = list(graph.graph_name for graph in self.regions[region_name])
            if plots_in_region:
                logger.warning(
                    f"All plot regions are in use; reusing plot region {region_name} which has graphs: {plots_in_region}"
                )
        else:
            logger.debug("New region for graph %s: %s", graph_name, region_name)
        return region_name

    def place_all(
        self,
        *,
        ignore_invalid: bool = True,
        ignore_unsupported: bool = True,
    ) -> Dict[str, List[AnyGraph]]:
        """
        Place all graphs in the place buffer.

        Side effect: clears `to_place`.

        Parameters
        ----------
        ignore_invalid : bool
            Ignore graphs marked as invalid by bmad.
        ignore_unsupported : bool
            Ignore unsupported graph types (e.g., key tables).

        Returns
        -------
        Dict[str, List[AnyGraph]]
            Region to list of graphs.
        """
        to_place = list(self.to_place.items())
        self.to_place.clear()

        logger.debug("Placing all plots: %s", to_place)
        result = {}
        for region_name, graph_name in to_place:
            try:
                result[region_name] = self.place(
                    graph_name=graph_name,
                    region_name=region_name,
                    ignore_invalid=ignore_invalid,
                )
            except UnsupportedGraphError:
                if not ignore_unsupported:
                    raise

        return result

    def update_region(
        self,
        region_name: str,
        graph_name: str,
        ignore_invalid: bool = True,
        ignore_unsupported: bool = True,
    ) -> List[AnyGraph]:
        """
        Update a specific region.

        Parameters
        ----------
        region_name : str, optional
            The region name to place it.  Determined automatically if unspecified.
        graph_name : str
            The graph template name.
        ignore_invalid : bool
            Ignore graphs marked as invalid by bmad.
        ignore_unsupported : bool
            Ignore unsupported graph types (e.g., key tables).

        Returns
        -------
        list of graphs
            The type of each graph is backend-dependent.
        """
        self._clear_region(region_name)

        result = []
        plot_names = get_plots_in_region(self.tao, region_name)
        for plot_name in plot_names:
            try:
                result.append(self.make_graph(region_name, plot_name))
            except UnsupportedGraphError as ex:
                if ignore_unsupported:
                    logger.debug(f"Unsupported graph in region {region_name}: {ex}")
                    continue
                raise
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

    def _place(
        self,
        graph_name: str,
        region_name: Optional[str] = None,
    ) -> str:
        if region_name is None:
            region_name = self.get_region_for_graph(graph_name)
            logger.debug(f"Picked {region_name} for {graph_name}")

        self.to_place.pop(region_name, None)

        logger.debug(f"Placing {graph_name} in {region_name}")
        self.tao.cmd(f"place -no_buffer {region_name} {graph_name}")
        return region_name

    def place(
        self,
        graph_name: str,
        *,
        region_name: Optional[str] = None,
        ignore_invalid: bool = True,
    ) -> List[AnyGraph]:
        """
        Place `graph_name` in `region_name`.

        Parameters
        ----------
        graph_name : str
            The graph template name.
        region_name : str, optional
            The region name to place it.  Determined automatically if unspecified.
        ignore_invalid : bool
            Ignore graphs marked as invalid by bmad.

        Returns
        -------
        list of graphs
            The type of each graph is backend-dependent.
        """
        region_name = self._place(graph_name, region_name)
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
        """
        Clear a single region or all regions.

        Parameters
        ----------
        region_name : str, optional
            Defaults to '*', which is all regions.
        """
        try:
            self.tao.cmd(f"place -no_buffer {region_name} none")
        except RuntimeError as ex:
            logger.warning(f"Region clear failed: {ex}")

        self._clear_region(region_name)

    def prepare_grid_by_names(
        self,
        graph_names: List[str],
        curves: Optional[List[CurveIndexToCurve]] = None,
        settings: Optional[List[TaoGraphSettings]] = None,
    ):
        """
        Prepare multiple graphs for a grid plot.

        Applies per-graph curve settings and also region/graph settings.

        Parameters
        ----------
        graph_names : list of str
            Graph names.
        curves : list of Dict[int, TaoCurveSettings], optional
            One dictionary per graph, with each dictionary mapping the curve
            index to curve settings. These settings will be applied to the
            placed graphs prior to plotting.
        settings : list of TaoGraphSettings, optional
            Graph customization settings.

        Returns
        -------
        list of graphs
        """
        if not curves:
            curves = [{}] * len(graph_names)
        elif len(curves) < len(graph_names):
            assert len(curves)
            curves = list(curves) + [{}] * (len(graph_names) - len(curves))

        if not settings:
            settings = [TaoGraphSettings()] * len(graph_names)
        elif len(settings) < len(graph_names):
            settings = list(settings) + [TaoGraphSettings()] * (
                len(graph_names) - len(settings)
            )

        graphs = sum(
            (
                self.prepare_graphs_by_name(
                    graph_name=graph_name,
                    curves=graph_curves,
                    settings=graph_settings,
                )
                for graph_name, graph_curves, graph_settings in zip(
                    graph_names,
                    curves,
                    settings,
                )
            ),
            [],
        )

        if not graphs:
            raise UnsupportedGraphError(
                f"No supported plots from these templates: {graph_names}"
            )
        return graphs

    def prepare_graphs_by_name(
        self,
        graph_name: str,
        region_name: Optional[str] = None,
        settings: Optional[TaoGraphSettings] = None,
        curves: Optional[Dict[int, TaoCurveSettings]] = None,
        ignore_unsupported: bool = True,
        ignore_invalid: bool = True,
    ) -> List[AnyGraph]:
        """
        Prepare a graph for plotting.

        Parameters
        ----------
        graph_name : str
            The graph template name.
        region_name : str, optional
            The region name to place it.  Determined automatically if unspecified.
        settings : TaoGraphSettings, optional
            Graph customization settings.
        curves : Dict[int, TaoCurveSettings], optional
            Curve settings, keyed by curve number.
        ignore_unsupported : bool
            Ignore unsupported graph types (e.g., key tables).
        ignore_invalid : bool
            Ignore graphs marked as invalid by bmad.

        Returns
        -------
        list of graphs
            The type of each graph is backend-dependent.
        """
        if not region_name:
            region_name = self.get_region_for_graph(graph_name)

        self._place(graph_name=graph_name, region_name=region_name)

        if settings:
            self.configure_graph(region_name, settings)

        if curves:
            self.configure_curves(region_name, curves)

        return self.update_region(
            region_name=region_name,
            graph_name=graph_name,
            ignore_unsupported=ignore_unsupported,
            ignore_invalid=ignore_invalid,
        )

    def configure_curves(
        self,
        region_name: str,
        settings: Dict[int, TaoCurveSettings],
        *,
        graph_name: Optional[str] = None,
    ):
        """
        Configure curves in a region.

        Parameters
        ----------
        region_name : str
            Already-placed region name.
        settings : Dict[int, TaoCurveSettings]
            Per-curve settings, keyed by integer curve index (starting at 1).
        graph_name : str, optional
            The graph name, if available.  If unspecified, settings will be
            applied to all plots in the region.
        """
        if not graph_name:
            for plot_name in get_plots_in_region(self.tao, region_name):
                self.configure_curves(region_name, settings=settings, graph_name=plot_name)
            return

        for curve_idx, curve in settings.items():
            for command in curve.get_commands(
                region_name,
                graph_name,
                curve_index=curve_idx,
            ):
                self.tao.cmd(command)

    def configure_graph(
        self,
        region_name: str,
        settings: TaoGraphSettings,
        *,
        graph_name: Optional[str] = None,
    ):
        """
        Configure graph settings for a region.

        Parameters
        ----------
        region_name : str
            Already-placed region name.
        settings : TaoGraphSettings
            Graph customization settings.
        graph_name : str, optional
            The graph name, if available.  If unspecified, settings will be
            applied to all plots in the region.
        """
        if not graph_name:
            for plot_name in get_plots_in_region(self.tao, region_name):
                self.configure_graph(region_name, settings=settings, graph_name=plot_name)
            return

        graph_info = get_plot_graph_info(self.tao, region_name, graph_name)
        graph_type = graph_info["graph^type"]
        for command in settings.get_commands(
            region_name,
            graph_name,
            graph_type=graph_type,
        ):
            self.tao.cmd(command)

    def plot_all(
        self, grid: Optional[Tuple[int, int]] = None, include_layout: bool = False, **kwargs
    ):
        """
        Plot all "placed" graphs.

        Parameters
        ----------
        grid : Tuple[int, int], optional
            Grid plots into this shape - (rows, cols).
        include_layout : bool, default=False
            Include a layout plot.
        **kwargs
            Keyword arguments are passed to `.plot_grid()`.
        """
        graph_names = list(self.to_place.values())
        if not grid:
            grid = (len(graph_names), 1)
        return self.plot_grid(
            graph_names,
            grid=grid,
            include_layout=include_layout,
            **kwargs,
        )

    def make_graph(self, region_name: str, graph_name: str) -> AnyGraph:
        """
        Create a graph instance from an already-placed graph.

        Parameters
        ----------
        region_name : str
            The region name of the graph.
        graph_name : str
            The graph name (not the template name).

        Returns
        -------
        AnyGraph
        """
        return make_graph(self.tao, region_name, graph_name)

    @abstractmethod
    def plot(
        self,
        graph_name: str,
        *,
        region_name: Optional[str] = None,
        include_layout: bool = True,
        settings: Optional[TaoGraphSettings] = None,
    ) -> Any:
        pass

    @abstractmethod
    def plot_grid(
        self,
        graph_names: List[str],
        grid: Tuple[int, int],
        *,
        include_layout: bool = False,
        curves: Optional[List[Dict[int, TaoCurveSettings]]] = None,
        settings: Optional[List[TaoGraphSettings]] = None,
    ) -> Any:
        pass

    @abstractmethod
    def plot_field(
        self,
        ele_id: str,
        *,
        colormap: Optional[str] = None,
        radius: float = 0.015,
        num_points: int = 100,
    ) -> Any:
        pass
