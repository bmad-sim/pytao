from __future__ import annotations
import logging
import math
from typing import List, Optional, Tuple, TypedDict, Union, cast

import matplotlib.axes
import matplotlib.collections
import matplotlib.patches
import matplotlib.path
import matplotlib.pyplot as plt
import numpy as np
from pytao import Tao

from . import pgplot, util

import pydantic.dataclasses as dataclasses
from pydantic.dataclasses import Field

logger = logging.getLogger(__name__)


class NoCurveDataError(Exception):
    pass


WaveParams = TypedDict(
    "WaveParams",
    {
        "ix_a1": float,
        "ix_a2": float,
        "ix_b1": float,
        "ix_b2": float,
    },
)

PlotCurveLineInfo = TypedDict(
    "PlotCurveLineInfo",
    {
        "width": int,
        "color": str,
        "line^pattern": str,
    },
)

PlotCurveSymbolInfo = TypedDict(
    "PlotCurveSymbolInfo",
    {
        "symbol^type": str,
        "color": str,
        "height": float,
        "fill_pattern": str,
        "line_width": int,
    },
)
PlotCurveInfo = TypedDict(
    "PlotCurveInfo",
    {
        "name": str,
        "data_source": str,
        "data_type_x": str,
        "data_type": str,
        "component": str,
        "ele_ref_name": str,
        "legend_text": str,
        "message_text": str,
        "why_invalid": str,
        "y_axis_scale_factor": float,
        "ix_universe": int,
        "symbol_every": int,
        "-1^ix_branch": int,
        "ix_ele_ref": int,
        "ix_ele_ref_track": int,
        "-1^ix_bunch": int,
        "use_y2": bool,
        "draw_line": bool,
        "draw_symbols": bool,
        "draw_symbol_index": bool,
        "draw_error_bars": bool,
        "smooth_line_calc": bool,
        "z_color_is_on": bool,
        "z_color_min": float,
        "z_color_max": float,
        "z_color_autoscale": bool,
        "z_color_data_type": str,
        "valid": bool,
        "line": PlotCurveLineInfo,
        "symbol": PlotCurveSymbolInfo,
        "symbol_line_width": int,
    },
)


PlotGraphInfo = TypedDict(
    "PlotGraphInfo",
    {
        # "curve[1..N]": str,
        "num_curves": int,
        "name": str,
        "graph^type": str,
        "title": str,
        "title_suffix": str,
        "why_invalid": str,
        "x_axis_scale_factor": float,
        "symbol_size_scale": float,
        "-1^ix_branch": int,
        "ix_universe": int,
        "clip": bool,
        "is_valid": bool,
        "y2_mirrors_y": bool,
        "limited": bool,
        "draw_axes": bool,
        "draw_curve_legend": bool,
        "draw_grid": bool,
        "draw_only_good_user_data_or_vars": bool,
        "floor_plan_view": str,
        "floor_plan_rotation": float,
        "floor_plan_flip_label_side": bool,
        "floor_plan_size_is_absolute": bool,
        "floor_plan_draw_building_wall": bool,
        "floor_plan_draw_only_first_pass": bool,
        "floor_plan_correct_distortion": bool,
        "floor_plan_orbit_scale": float,
        "floor_plan_orbit_color": str,
        "floor_plan_orbit_lattice": str,
        "floor_plan_orbit_width": int,
        "floor_plan_orbit_pattern": str,
        "x_label": str,
        "x_label_color": str,
        "x_label_offset": float,
        "x_max": float,
        "x_min": float,
        "x_axis^type": str,
        "x_bounds": str,
        "x_number_offset": float,
        "x_major_div_nominal": int,
        "x_minor_div": int,
        "x_minor_div_max": int,
        "x_draw_label": bool,
        "x_draw_numbers": bool,
        "x_tick_side": int,
        "x_number_side": int,
        "x_major_tick_len": float,
        "x_minor_tick_len": float,
        "y_label": str,
        "y_label_color": str,
        "y_label_offset": float,
        "y_max": float,
        "y_min": float,
        "y_axis^type": str,
        "y_bounds": str,
        "y_number_offset": float,
        "y_major_div_nominal": int,
        "y_minor_div": int,
        "y_minor_div_max": int,
        "y_draw_label": bool,
        "y_draw_numbers": bool,
        "y_tick_side": int,
        "y_number_side": int,
        "y_major_tick_len": float,
        "y_minor_tick_len": float,
        "y2_label": str,
        "y2_label_color": str,
        "y2_label_offset": float,
        "y2_max": float,
        "y2_min": float,
        "y2_axis^type": str,
        "y2_bounds": str,
        "y2_number_offset": float,
        "y2_major_div_nominal": int,
        "y2_minor_div": int,
        "y2_minor_div_max": int,
        "y2_draw_label": bool,
        "y2_draw_numbers": bool,
        "y2_tick_side": int,
        "y2_number_side": int,
        "y2_major_tick_len": float,
        "y2_minor_tick_len": float,
    },
)


PlotHistogramInfo = TypedDict(
    "PlotHistogramInfo",
    {
        "density_normalized": bool,
        "weight_by_charge": bool,
        "minimum": float,
        "maximum": float,
        "width": float,
        "center": float,
        "number": float,
    },
)


Point = Tuple[float, float]


def print_info(d):
    print({key: type(value).__name__ for key, value in d.items()})
    for key, value in d.items():
        if isinstance(value, dict):
            print(key, "->")
            print_info(value)


def _should_use_symbol_color(symbol_type: str, fill_pattern: str) -> bool:
    if (
        symbol_type in ("dot", "1")
        or symbol_type.endswith("filled")
        or symbol_type.startswith("-")
    ):
        return True

    if pgplot.fills[fill_pattern] == "solid":
        return True

    return False


@dataclasses.dataclass
class PlotCurveLine:
    xs: List[float]
    ys: List[float]
    color: str
    linestyle: str
    linewidth: float

    def plot(self, ax: matplotlib.axes.Axes):
        print("plot lines", self.xs)
        return ax.plot(
            self.xs,
            self.ys,
            color=self.color,
            linestyle=self.linestyle,
            linewidth=self.linewidth,
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

    def plot(self, ax: matplotlib.axes.Axes):
        return ax.plot(
            self.xs,
            self.ys,
            color=self.color,
            markerfacecolor=self.markerfacecolor,
            markersize=self.markersize,
            marker=self.marker,
            markeredgewidth=self.markeredgewidth,
            linewidth=self.linewidth,
        )


@dataclasses.dataclass
class PlotHistogram:
    xs: List[float]
    bins: float
    weights: List[float]
    histtype: str
    color: str

    def plot(self, ax: matplotlib.axes.Axes) -> None:
        return ax.hist(
            self.xs,
            bins=self.bins,
            weights=self.weights,
            histtype=self.histtype,
            color=self.color,
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

    @property
    def _patch_args(self):
        return {
            "edgecolor": self.edgecolor,
            "facecolor": self.facecolor,
            "color": self.color,
            "linewidth": self.linewidth,
            "linestyle": self.linestyle,
            "antialiased": self.antialiased,
            "hatch": self.hatch,
            "fill": self.fill,
            "capstyle": self.capstyle,
            "joinstyle": self.joinstyle,
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
    rotation_point: str = "xy"

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
    pass


@dataclasses.dataclass
class PlotPatchCircle(PlotPatchBase):
    pass


@dataclasses.dataclass
class PlotPatchEllipse(PlotPatchBase):
    pass


PlotPatch = Union[
    PlotPatchRectangle,
    PlotPatchArc,
    PlotPatchCircle,
    PlotPatchEllipse,
]


@dataclasses.dataclass
class PlotCurve:
    info: PlotCurveInfo
    line: Optional[PlotCurveLine]
    symbol: Optional[PlotCurveSymbols]
    histogram: Optional[PlotHistogram] = None
    patches: Optional[List[PlotPatch]] = None

    def plot(self, ax: matplotlib.axes.Axes) -> None:
        if self.line is not None:
            self.line.plot(ax)
        if self.symbol is not None:
            self.symbol.plot(ax)
        if self.histogram is not None:
            self.histogram.plot(ax)
        for patch in self.patches or []:
            patch.plot(ax)

    @property
    def legend_label(self) -> str:
        legend_text = pgplot.mpl_string(self.info["legend_text"])
        if legend_text:
            return legend_text

        data_type = pgplot.mpl_string(self.info["data_type"])
        return data_type if data_type == "physical_aperture" else ""

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        region_name: str,
        graph_name: str,
        curve_name: str,
        graph_type: Optional[str] = None,
    ) -> PlotCurve:
        full_name = f"{region_name}.{graph_name}.{curve_name}"
        curve_info = cast(PlotCurveInfo, tao.plot_curve(full_name))
        points = [
            (line["x"], line["y"])
            for line in tao.plot_line(region_name, graph_name, curve_name) or []
        ]
        try:
            symbol_points = [
                (sym["x_symb"], sym["y_symb"])
                for sym in tao.plot_symbol(region_name, graph_name, curve_name, x_or_y="")
                or []
            ]
        except RuntimeError:
            symbol_points = []

        if graph_type is None:
            graph_info = cast(PlotGraphInfo, tao.plot_graph(f"{region_name}.{graph_name}"))
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

        # marker
        marker = pgplot.symbols.get(symbol_type, ".")
        # symbol_line_width
        symbol_line_width = curve_info["symbol"]["line_width"]

        xpoints = [p[0] for p in points]
        ypoints = [p[1] for p in points]
        symbol_xs = [p[0] for p in symbol_points]
        symbol_ys = [p[1] for p in symbol_points]
        if symbol_ys:
            y_max = max(
                0.5 * max(max(ypoints), max(symbol_ys)),
                2 * max(max(ypoints), max(symbol_ys)),
            )
            y_min = min(
                0.5 * min(min(ypoints), min(symbol_ys)),
                2 * min(min(ypoints), min(symbol_ys)),
            )
        elif ypoints:
            y_max = max(ypoints)
            y_min = min(ypoints)
        else:
            raise NoCurveDataError("No points found, make sure data is properly initialized")
        # boundaries for wave analysis rectangles

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
                marker=marker,
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
                    bins=histogram_info["number"],
                    weights=ypoints,
                    histtype="step",
                    color=symbol_color,
                ),
            )

        raise NotImplementedError(f"graph_type: {graph_type}")


@dataclasses.dataclass
class PlotBasicGraph:
    info: PlotGraphInfo
    xlim: Point = _point_field
    ylim: Point = _point_field
    xlabel: str = ""
    ylabel: str = ""
    title: str = ""
    show_axes: bool = True
    draw_grid: bool = True
    draw_legend: bool = True
    curves: List[PlotCurve] = Field(default_factory=list)

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        region_name: str,
        graph_name: str,
        graph_info: Optional[PlotGraphInfo] = None,
    ) -> PlotBasicGraph:
        if graph_info is None:
            graph_info = tao.plot_graph(f"{region_name}.{graph_name}")
            assert graph_info is not None

        graph_type = graph_info["graph^type"]
        if graph_type == "key_table":
            raise NotImplementedError("Key table graphs")

        if graph_type == "lat_layout":
            raise ValueError()
        if graph_type == "floor_plan":
            raise ValueError()
        if graph_info["why_invalid"]:
            raise ValueError(f"Graph not valid: {graph_info['why_invalid']}")

        all_curve_names = [
            graph_info[f"curve[{i + 1}]"] for i in range(graph_info["num_curves"])
        ]
        curves = [
            PlotCurve.from_tao(tao, region_name, graph_name, curve_name)
            for curve_name in all_curve_names
        ]

        return cls(
            info=graph_info,
            curves=curves,
            show_axes=graph_info["draw_axes"],
            title=pgplot.mpl_string("{title} {title_suffix}".format(**graph_info)),
            xlabel=pgplot.mpl_string(graph_info["x_label"]),
            ylabel=pgplot.mpl_string(graph_info["y_label"]),
            draw_grid=graph_info["draw_grid"],
            xlim=(graph_info["x_min"], graph_info["x_max"]),
            ylim=(graph_info["y_min"], graph_info["y_max"]),
            draw_legend=graph_info["draw_curve_legend"],
        )

    def plot(self, ax: Optional[matplotlib.axes.Axes] = None):
        if ax is None:
            _, ax = plt.subplots()
            assert ax is not None

        for curve in self.curves:
            curve.plot(ax)

        # if self.draw_legend:
        #     ax.legend(legend_items, labels)

        if not self.show_axes:
            ax.set_axis_off()

        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.grid(self.draw_grid, which="major", axis="both")
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_axisbelow(True)
        return ax


def plot_normal_graph(
    tao: Tao,
    region_name: str,
    graph_name: str,
    graph_info: Optional[dict] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> PlotBasicGraph:
    result_graph = PlotBasicGraph.from_tao(
        tao,
        region_name=region_name,
        graph_name=graph_name,
        graph_info=graph_info,
    )

    result_graph.plot(ax)
    return result_graph


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
        yield [s1, s_max], [y1, y1]
        yield [s1, s_max], [y2, y2]
        yield [s_min, s2], [y1, y1]
        yield [s_min, s2], [y2, y2]
        yield [s1, s1], [y1, y2]
        yield [s2, s2], [y1, y2]

    elif shape == "xbox":
        yield [s1, s_max], [y1, y1]
        yield [s1, s_max], [y2, y2]
        yield [s1, s_max], [y1, 0]
        yield [s1, s_max], [y2, 0]
        yield [s_min, s2], [y1, y1]
        yield [s_min, s2], [y2, y2]
        yield [s_min, s2], [0, y1]
        yield [s_min, s2], [0, y2]
        yield [s1, s1], [y1, y2]
        yield [s2, s2], [y1, y2]

    elif shape == "x":
        yield [s1, s_max], [y1, 0]
        yield [s1, s_max], [y2, 0]
        yield [s_min, s2], [0, y1]
        yield [s_min, s2], [0, y2]

    elif shape == "bow_tie":
        yield [s1, s_max], [y1, y1]
        yield [s1, s_max], [y2, y2]
        yield [s1, s_max], [y1, 0]
        yield [s1, s_max], [y2, 0]
        yield [s_min, s2], [y1, y1]
        yield [s_min, s2], [y2, y2]
        yield [s_min, s2], [0, y1]
        yield [s_min, s2], [0, y2]

    elif shape == "diamond":
        yield [s1, s_max], [0, y1]
        yield [s1, s_max], [0, y2]
        yield [s_min, s2], [y1, 0]
        yield [s_min, s2], [y2, 0]


def plot_lat_layout(
    tao: Tao,
    region_name: str,
    graph_name: str,
    graph_info: Optional[dict] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
):
    if graph_info is None:
        graph_info = tao.plot_graph(f"{region_name}.{graph_name}")
        assert graph_info is not None

    if ax is None:
        _, ax = plt.subplots()
        assert ax is not None

    # List of parameter strings from tao command python plot_graph
    layout_info = tao.plot_graph("lat_layout.g")

    # Makes lat layout only have horizontal axis for panning and zooming
    twin_ax = ax.axes.twinx()
    plt.xlim(graph_info["x_min"], graph_info["x_max"])
    plt.ylim(
        layout_info["y_min"],
        layout_info["y_max"],
    )
    twin_ax.set_navigate(True)
    ax.axis("off")
    twin_ax.axis("off")

    # Sets axis limits and creates second axis to allow x panning and zooming
    ax.axes.set_navigate(False)
    ax.axhline(
        y=0,
        xmin=1.1 * layout_info["x_min"],
        xmax=1.1 * layout_info["x_max"],
        color="Black",
    )

    # Lat layout branch and universe information
    if layout_info["ix_universe"] != -1:
        universe = layout_info["ix_universe"]
    else:
        universe = 1
    branch = layout_info["-1^ix_branch"]

    # List of strings containing information about each element
    try:
        all_elem_info = tao.plot_lat_layout(ix_uni=universe, ix_branch=branch)
    except RuntimeError as ex:
        if branch != -1:
            raise

        logger.warning(
            f"Lat layout failed for universe={universe} branch={branch}; trying branch 0"
        )
        try:
            all_elem_info = tao.plot_lat_layout(ix_uni=universe, ix_branch=0)
        except RuntimeError:
            print(f"Failed to plot layout: {ex}")
            raise

    # Plotting line segments one-by-one can be slow if there are thousands of lattice elements.
    # So keep a list of line segments and plot all at once at the end.

    ele_y1s = [elem_info["y1"] for elem_info in all_elem_info]
    ele_y2s = [elem_info["y2"] for elem_info in all_elem_info]

    y_max = max(max(ele_y1s), max(ele_y2s))
    y2_floor = -max(ele_y2s)  # Note negative sign
    lines = []
    widths = []
    colors = []

    for elem_info in all_elem_info:
        s1 = elem_info["ele_s_start"]
        s2 = elem_info["ele_s"]
        y1 = elem_info["y1"]
        y2 = -elem_info["y2"]  # Note negative sign.
        wid = elem_info["line_width"]
        color = elem_info["color"]
        shape = elem_info["shape"]
        name = elem_info["label_name"]

        # Normal case where element is not wrapped around ends of lattice.
        if s2 - s1 > 0:
            # Draw box element
            if shape == "box":
                ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (s1, y1),
                        s2 - s1,
                        y2 - y1,
                        lw=wid,
                        color=color,
                        fill=False,
                    )
                )

            # Draw xbox element
            elif shape == "xbox":
                ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (s1, y1),
                        s2 - s1,
                        y2 - y1,
                        lw=wid,
                        color=color,
                        fill=False,
                    )
                )
                lines.extend([[(s1, y1), (s2, y2)], [(s1, y2), (s2, y1)]])
                colors.extend([color, color])
                widths.extend([wid, wid])

            # Draw x element
            elif shape == "x":
                lines.extend([[(s1, y1), (s2, y2)], [(s1, y2), (s2, y1)]])
                colors.extend([color, color])
                widths.extend([wid, wid])

            # Draw bow_tie element
            elif shape == "bow_tie":
                lines.extend(
                    [
                        [(s1, y1), (s2, y2)],
                        [(s1, y2), (s2, y1)],
                        [(s1, y1), (s1, y2)],
                        [(s2, y1), (s2, y2)],
                    ]
                )
                colors.extend([color, color, color, color])
                widths.extend([wid, wid, wid, wid])

            # Draw rbow_tie element
            elif shape == "rbow_tie":
                lines.extend(
                    [
                        [(s1, y1), (s2, y2)],
                        [(s1, y2), (s2, y1)],
                        [(s1, y1), (s2, y1)],
                        [(s1, y2), (s2, y2)],
                    ]
                )
                colors.extend([color, color, color, color])
                widths.extend([wid, wid, wid, wid])

            # Draw diamond element
            elif shape == "diamond":
                s_mid = (s1 + s2) / 2
                lines.extend(
                    [
                        [(s1, 0), (s_mid, y1)],
                        [(s1, 0), (s_mid, y2)],
                        [(s2, 0), (s_mid, y1)],
                        [(s2, 0), (s_mid, y2)],
                    ]
                )
                colors.extend([color, color, color, color])
                widths.extend([wid, wid, wid, wid])

            # Draw circle element
            elif shape == "circle":
                s_mid = (s1 + s2) / 2
                ax.add_patch(
                    matplotlib.patches.Ellipse(
                        (s_mid, 0),
                        y1 - y2,
                        y1 - y2,
                        lw=wid,
                        color=color,
                        fill=False,
                    )
                )

            # Draw element name
            ax.text(
                (s1 + s2) / 2,
                1.1 * y2_floor,
                name,
                ha="center",
                va="top",
                clip_on=True,
                color=color,
            )

        else:
            # Case where element is wrapped round the lattice ends.
            try:
                s_min = layout_info["x_min"]
                s_max = layout_info["x_max"]
            except KeyError:
                logger.exception("Missing xmin/xmax")
                continue

            # Draw wrapped box element
            for xs, ys in _get_wrapped_shape_coords(
                shape=shape,
                s1=s1,
                s2=s2,
                y1=y1,
                y2=y2,
                s_min=s_min,
                s_max=s_max,
            ):
                ax.plot(xs, ys, lw=wid, color=color)

            # Draw wrapped element name
            ax.text(
                s_max,
                1.1 * y2_floor,
                name,
                ha="right",
                va="top",
                clip_on=True,
                color=color,
            )
            ax.text(
                s_min,
                1.1 * y2_floor,
                name,
                ha="left",
                va="top",
                clip_on=True,
                color=color,
            )

    # Draw all line segments
    ax.add_collection(
        matplotlib.collections.LineCollection(lines, colors=colors, linewidths=widths)
    )

    # Invisible line to give the lat layout enough vertical space.
    # Without this, the tops and bottoms of shapes could be cut off
    ax.plot([0, 0], [-1.7 * y_max, 1.3 * y_max], lw=wid, color=color, alpha=0)


def _building_wall_to_arc(
    mx,
    my,
    kx,
    ky,
    k_radii,
    color: str,
):
    (c0x, c0y), (c1x, c1y) = util.circle_intersection(
        mx,
        my,
        kx,
        ky,
        abs(k_radii),
    )
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
    # find correct center

    m_angle = 360 + math.degrees(
        np.arctan2(
            (my - center[1]),
            (mx - center[0]),
        )
    )
    k_angle = 360 + math.degrees(
        np.arctan2(
            (ky - center[1]),
            (kx - center[0]),
        )
    )

    if abs(k_angle - m_angle) <= 180:
        if k_angle > m_angle:
            t1 = m_angle
            t2 = k_angle
        else:
            t1 = k_angle
            t2 = m_angle
    else:
        if k_angle > m_angle:
            t1 = k_angle
            t2 = m_angle
        else:
            t1 = m_angle
            t2 = k_angle
    # pick correct start and end angle for arc

    return matplotlib.patches.Arc(
        center,
        k_radii * 2,
        k_radii * 2,
        theta1=t1,
        theta2=t2,
        color=color,
    )
    # draw building wall arc


def plot_floor_plan(
    tao: Tao,
    region_name: str,
    graph_name: str,
    graph_info: Optional[dict] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
):
    if ax is None:
        _, ax = plt.subplots()
        assert ax is not None

    graph_full_name = f"{region_name}.{graph_name}"

    if graph_info is None:
        graph_info = tao.plot_graph(graph_full_name)
        assert graph_info is not None
    # list of plotting parameter strings from tao command python plot_graph

    # tao_parameter object names from python plot_graph for a floor plan
    # dictionary of tao_parameter name string keys to the corresponding tao_parameter object

    # if graph_info["ix_universe"] != -1:
    #     universe = graph_info["ix_universe"]
    #
    # else:
    #     universe = 1
    #
    floor_plan_elements = tao.floor_plan(graph_full_name)
    # list of plotting parameter strings from tao command python graph_info

    for info in floor_plan_elements:
        plot_floor_plan_element(ax=ax, **info)

    building_wall_graph = tao.building_wall_graph(graph_full_name)
    building_wall_curves = set(graph["index"] for graph in building_wall_graph)
    building_wall_types = {wall["index"]: wall["name"] for wall in tao.building_wall_list()}

    elem_to_color = {
        elem["ele_id"].split(":")[0]: pgplot.mpl_color(elem["color"])
        for elem in tao.shape_list("floor_plan")
    }

    for curve_name in sorted(building_wall_curves):
        points = []  # index of point in curve
        xs = []  # list of point x coordinates
        ys = []  # list of point y coordinates
        radii = []  # straight line if element has 0 or missing radius
        for bwg in building_wall_graph:
            if curve_name == bwg["index"]:
                points.append(bwg["point"])
                xs.append(bwg["offset_x"])
                ys.append(bwg["offset_y"])
                radii.append(bwg["radius"])

        k = max(points)  # max line index

        while k > 1:
            idx_k = points.index(k)
            idx_m = points.index(k - 1)  # adjacent point to connect to
            if building_wall_types[curve_name] not in elem_to_color:
                # TODO: This is a temporary fix to deal with building wall segments
                # that don't have an associated graph_info shape
                # Currently this will fail to match to wild cards
                # in the shape name (e.g. building_wall::* should match
                # to every building wall segment, but currently it
                # matches to none).  A more sophisticated way of getting the
                # graph_info shape settings for building walls will be required
                # in the future, either through a python command in tao or
                # with a method on the python to match wild cards to wall segment names
                logger.warning(
                    f"No graph_info shape defined for building_wall segment {building_wall_types[curve_name]}"
                )
                k -= 1
                continue

            color = elem_to_color[building_wall_types[curve_name]]
            if radii[idx_k] == 0:  # draw building wall line
                ax.plot(
                    [xs[idx_k], xs[idx_m]],
                    [ys[idx_k], ys[idx_m]],
                    color=color,
                )

            else:  # draw building wall arc
                ax.add_patch(
                    _building_wall_to_arc(
                        mx=xs[idx_m],
                        my=ys[idx_m],
                        kx=xs[idx_k],
                        ky=ys[idx_k],
                        k_radii=radii[idx_k],
                        color=color,
                    )
                )

            k = k - 1
    # plot floor plan building walls

    if float(graph_info["floor_plan_orbit_scale"]) != 0:
        floor_orbit_info = tao.floor_orbit(graph_full_name)

        floor_orbit_xs = []
        floor_orbit_ys = []
        for info in floor_orbit_info:
            if info["ele_key"] == "x":
                floor_orbit_xs.extend(info["orbits"])
            elif info["ele_key"] == "y":
                floor_orbit_ys.extend(info["orbits"])

        ax.plot(
            floor_orbit_xs,
            floor_orbit_ys,
            color=graph_info["floor_plan_orbit_color"].lower(),
        )
    # Lists of floor plan orbit point indices, x coordinates, and y coordinates
    # plot floor plan orbit
    ax.set_xlabel(pgplot.mpl_string(graph_info["x_label"]))
    ax.set_ylabel(pgplot.mpl_string(graph_info["y_label"]))
    # plot floor plan axis labels

    ax.grid(graph_info["draw_grid"], which="major", axis="both")
    ax.set_xlim(graph_info["x_min"], graph_info["x_max"])
    ax.set_ylim(graph_info["y_min"], graph_info["y_max"])
    ax.set_axisbelow(True)

    # plot floor plan grid


def _circle_to_patch(
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
    return matplotlib.patches.Circle(
        (x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2),
        off1,
        lw=line_width,
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
    return matplotlib.patches.Rectangle(
        (
            x1 + off2 * np.sin(angle_start),
            y1 - off2 * np.cos(angle_start),
        ),
        np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2),
        off1 + off2,
        lw=line_width,
        color=color,
        fill=False,
        angle=math.degrees(angle_start),
    )


def _draw_x_lines(
    ax: matplotlib.axes.Axes,
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
    ax.plot(
        [
            x1 + off2 * np.sin(angle_start),
            x2 - off1 * np.sin(angle_start),
        ],
        [
            y1 - off2 * np.cos(angle_start),
            y2 + off1 * np.cos(angle_start),
        ],
        lw=line_width,
        color=color,
    )
    ax.plot(
        [
            x1 - off1 * np.sin(angle_start),
            x2 + off2 * np.sin(angle_start),
        ],
        [
            y1 + off1 * np.cos(angle_start),
            y2 - off2 * np.cos(angle_start),
        ],
        lw=line_width,
        color=color,
    )


def _draw_sbend_box(
    ax: matplotlib.axes.Axes,
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
):
    ax.plot(
        [
            x1 - off1 * np.sin(angle_start - rel_angle_start),
            x1 + off2 * np.sin(angle_start - rel_angle_start),
        ],
        [
            y1 + off1 * np.cos(angle_start - rel_angle_start),
            y1 - off2 * np.cos(angle_start - rel_angle_start),
        ],
        lw=line_width,
        color=color,
    )
    ax.plot(
        [
            x2 - off1 * np.sin(angle_end + rel_angle_end),
            x2 + off2 * np.sin(angle_end + rel_angle_end),
        ],
        [
            y2 + off1 * np.cos(angle_end + rel_angle_end),
            y2 - off2 * np.cos(angle_end + rel_angle_end),
        ],
        lw=line_width,
        color=color,
    )


def _draw_sbend(
    ax: matplotlib.axes.Axes,
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
):
    _draw_sbend_box(
        ax=ax,
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

    try:
        intersection = util.intersect(
            util.line(
                (
                    x1 - off1 * np.sin(angle_start),
                    y1 + off1 * np.cos(angle_start),
                ),
                (
                    x1 + off2 * np.sin(angle_start),
                    y1 - off2 * np.cos(angle_start),
                ),
            ),
            util.line(
                (
                    x2 - off1 * np.sin(angle_end),
                    y2 + off1 * np.cos(angle_end),
                ),
                (
                    x2 + off2 * np.sin(angle_end),
                    y2 - off2 * np.cos(angle_end + rel_angle_end),
                ),
            ),
        )
        # center of circle used to draw arc edges of sbends
    except util.NoIntersectionError:
        intersection = None
        ax.plot(
            [
                x1 - off1 * np.sin(angle_start - rel_angle_start),
                x2 - off1 * np.sin(angle_end + rel_angle_end),
            ],
            [
                y1 + off1 * np.cos(angle_start - rel_angle_start),
                y2 + off1 * np.cos(angle_end + rel_angle_end),
            ],
            lw=line_width,
            color=color,
        )
        ax.plot(
            [
                x1 + off2 * np.sin(angle_start - rel_angle_start),
                x2 + off2 * np.sin(angle_end + rel_angle_end),
            ],
            [
                y1 - off2 * np.cos(angle_start - rel_angle_start),
                y2 - off2 * np.cos(angle_end + rel_angle_end),
            ],
            lw=line_width,
            color=color,
        )

    else:
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

        ax.add_patch(
            matplotlib.patches.Arc(
                (intersection[0], intersection[1]),
                np.sqrt(
                    (x1 - off1 * np.sin(angle_start - rel_angle_start) - intersection[0]) ** 2
                    + (y1 + off1 * np.cos(angle_start - rel_angle_start) - intersection[1])
                    ** 2
                )
                * 2,
                np.sqrt(
                    (x1 - off1 * np.sin(angle_start - rel_angle_start) - intersection[0]) ** 2
                    + (y1 + off1 * np.cos(angle_start - rel_angle_start) - intersection[1])
                    ** 2
                )
                * 2,
                theta1=a1,
                theta2=a2,
                lw=line_width,
                color=color,
            )
        )
        ax.add_patch(
            matplotlib.patches.Arc(
                (intersection[0], intersection[1]),
                np.sqrt(
                    (x1 + off2 * np.sin(angle_start - rel_angle_start) - intersection[0]) ** 2
                    + (y1 - off2 * np.cos(angle_start - rel_angle_start) - intersection[1])
                    ** 2
                )
                * 2,
                np.sqrt(
                    (x1 + off2 * np.sin(angle_start - rel_angle_start) - intersection[0]) ** 2
                    + (y1 - off2 * np.cos(angle_start - rel_angle_start) - intersection[1])
                    ** 2
                )
                * 2,
                theta1=a3,
                theta2=a4,
                lw=line_width,
                color=color,
            )
        )
        # draw sbend edges if bend angle is nonzero


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
    c1 = [
        x1 - off1 * np.sin(angle_start - rel_angle_start),
        y1 + off1 * np.cos(angle_start - rel_angle_start),
    ]
    c2 = [
        x2 - off1 * np.sin(angle_end + rel_angle_end),
        y2 + off1 * np.cos(angle_end + rel_angle_end),
    ]
    c3 = [
        x1 + off2 * np.sin(angle_start - rel_angle_start),
        y1 - off2 * np.cos(angle_start - rel_angle_start),
    ]
    c4 = [
        x2 + off2 * np.sin(angle_end + rel_angle_end),
        y2 - off2 * np.cos(angle_end + rel_angle_end),
    ]
    # corners of sbend

    if angle_start > angle_end:
        outer_radius = np.sqrt(
            (x1 - off1 * np.sin(angle_start - rel_angle_start) - intersection[0]) ** 2
            + (y1 + off1 * np.cos(angle_start - rel_angle_start) - intersection[1]) ** 2
        )
        inner_radius = np.sqrt(
            (x1 + off2 * np.sin(angle_start - rel_angle_start) - intersection[0]) ** 2
            + (y1 - off2 * np.cos(angle_start - rel_angle_start) - intersection[1]) ** 2
        )
    else:
        outer_radius = -np.sqrt(
            (x1 - off1 * np.sin(angle_start - rel_angle_start) - intersection[0]) ** 2
            + (y1 + off1 * np.cos(angle_start - rel_angle_start) - intersection[1]) ** 2
        )
        inner_radius = -np.sqrt(
            (x1 + off2 * np.sin(angle_start - rel_angle_start) - intersection[0]) ** 2
            + (y1 - off2 * np.cos(angle_start - rel_angle_start) - intersection[1]) ** 2
        )
    # radii of sbend arc edges

    mid_angle = (angle_start + angle_end) / 2

    top = [
        intersection[0] - outer_radius * np.sin(mid_angle),
        intersection[1] + outer_radius * np.cos(mid_angle),
    ]
    bottom = [
        intersection[0] - inner_radius * np.sin(mid_angle),
        intersection[1] + inner_radius * np.cos(mid_angle),
    ]
    # midpoints of top and bottom arcs in an sbend

    top_cp = [
        2 * (top[0]) - 0.5 * (c1[0]) - 0.5 * (c2[0]),
        2 * (top[1]) - 0.5 * (c1[1]) - 0.5 * (c2[1]),
    ]
    bottom_cp = [
        2 * (bottom[0]) - 0.5 * (c3[0]) - 0.5 * (c4[0]),
        2 * (bottom[1]) - 0.5 * (c3[1]) - 0.5 * (c4[1]),
    ]
    # corresponding control points for a quadratic Bezier curve that passes through the corners and arc midpoint

    verts = [c1, top_cp, c2, c4, bottom_cp, c3, c1]
    codes = [
        matplotlib.path.Path.MOVETO,
        matplotlib.path.Path.CURVE3,
        matplotlib.path.Path.CURVE3,
        matplotlib.path.Path.LINETO,
        matplotlib.path.Path.CURVE3,
        matplotlib.path.Path.CURVE3,
        matplotlib.path.Path.CLOSEPOLY,
    ]
    path = matplotlib.path.Path(verts, codes)
    return matplotlib.patches.PathPatch(path, facecolor="green", alpha=0.5)


def _draw_bow_tie(
    ax: matplotlib.axes.Axes,
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
    ax.plot(
        [
            x1 + off2 * np.sin(angle_start),
            x2 - off1 * np.sin(angle_start),
        ],
        [
            y1 - off2 * np.cos(angle_start),
            y2 + off1 * np.cos(angle_start),
        ],
        lw=line_width,
        color=color,
    )
    ax.plot(
        [
            x1 - off1 * np.sin(angle_start),
            x2 + off2 * np.sin(angle_start),
        ],
        [
            y1 + off1 * np.cos(angle_start),
            y2 - off2 * np.cos(angle_start),
        ],
        lw=line_width,
        color=color,
    )
    ax.plot(
        [
            x1 - off1 * np.sin(angle_start),
            x2 - off1 * np.sin(angle_start),
        ],
        [
            y1 + off1 * np.cos(angle_start),
            y2 + off1 * np.cos(angle_start),
        ],
        lw=line_width,
        color=color,
    )
    ax.plot(
        [
            x1 + off2 * np.sin(angle_start),
            x2 + off2 * np.sin(angle_start),
        ],
        [
            y1 - off2 * np.cos(angle_start),
            y2 - off2 * np.cos(angle_start),
        ],
        lw=line_width,
        color=color,
    )


def _draw_diamond(
    ax: matplotlib.axes.Axes,
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
    ax.plot(
        [x1, x1 + (x2 - x1) / 2 - off1 * np.sin(angle_start)],
        [y1, y1 + (y2 - y1) / 2 + off1 * np.cos(angle_start)],
        lw=line_width,
        color=color,
    )
    ax.plot(
        [x1 + (x2 - x1) / 2 - off1 * np.sin(angle_start), x2],
        [y1 + (y2 - y1) / 2 + off1 * np.cos(angle_start), y2],
        lw=line_width,
        color=color,
    )
    ax.plot(
        [x1, x1 + (x2 - x1) / 2 + off2 * np.sin(angle_start)],
        [y1, y1 + (y2 - y1) / 2 - off2 * np.cos(angle_start)],
        lw=line_width,
        color=color,
    )
    ax.plot(
        [x1 + (x2 - x1) / 2 + off2 * np.sin(angle_start), x2],
        [y1 + (y2 - y1) / 2 - off2 * np.cos(angle_start), y2],
        lw=line_width,
        color=color,
    )


def plot_floor_plan_element(
    ax: matplotlib.axes.Axes,
    *,
    branch_index: int,
    index: int,
    ele_key: str,
    end1_r1: float,
    end1_r2: float,
    end1_theta: float,
    end2_r1: float,
    end2_r2: float,
    end2_theta: float,
    line_width: float,
    shape: str,
    y1: float,
    y2: float,
    color: str,
    label_name: str,
    # Only for sbend:     #
    ele_l: float = 0.0,
    ele_angle: float = 0.0,
    ele_e1: float = 0.0,
    ele_e: float = 0.0,
) -> None:
    # A bit of renaming while porting this over...
    off1, off2 = y1, y2
    x1, y1, angle_start = end1_r1, end1_r2, end1_theta
    x2, y2, angle_end = end2_r1, end2_r2, end2_theta
    rel_angle_start, rel_angle_end = ele_e1, ele_e

    intersection = None

    if ele_key == "drift" or ele_key == "kicker":
        # draw drift element
        ax.plot([x1, x2], [y1, y2], color="black")

    if off1 == 0 and off2 == 0 and ele_key != "sbend" and color:
        # draw line element
        ax.plot([x1, x2], [y1, y2], lw=line_width, color=color)

    elif shape == "box" and ele_key != "sbend" and color:
        ax.add_patch(
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
        ax.add_patch(
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
        _draw_x_lines(
            ax,
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

    elif shape == "x" and ele_key != "sbend" and color:
        _draw_x_lines(
            ax,
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

    elif shape == "bow_tie" and ele_key != "sbend" and color:
        _draw_bow_tie(
            ax,
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

    elif shape == "diamond" and ele_key != "sbend" and color:
        _draw_diamond(
            ax,
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

    elif shape == "circle" and ele_key != "sbend" and color:
        ax.add_patch(
            _circle_to_patch(
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

    elif shape == "box" and ele_key == "sbend" and color:
        # draws straight sbend edges
        _draw_sbend(
            ax=ax,
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

    if label_name and color and np.sin(((angle_end + angle_start) / 2)) > 0:
        ax.text(
            x1 + (x2 - x1) / 2 - 1.3 * off1 * np.sin(angle_start),
            y1 + (y2 - y1) / 2 + 1.3 * off1 * np.cos(angle_start),
            label_name,
            ha="right",
            va="center",
            color="black",
            rotation=-90 + math.degrees((angle_end + angle_start) / 2),
            clip_on=True,
            rotation_mode="anchor",
        )

    elif label_name and color and np.sin(((angle_end + angle_start) / 2)) <= 0:
        ax.text(
            x1 + (x2 - x1) / 2 - 1.3 * off1 * np.sin(angle_start),
            y1 + (y2 - y1) / 2 + 1.3 * off1 * np.cos(angle_start),
            label_name,
            ha="left",
            va="center",
            color="black",
            rotation=90 + math.degrees((angle_end + angle_start) / 2),
            clip_on=True,
            rotation_mode="anchor",
        )
    # draw element name

    if ele_key == "sbend" and intersection:
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
        ax.add_patch(patch)

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


def get_graphs_in_region(tao: Tao, region_name: str):
    plot1_info = tao.plot1(region_name)

    if "num_graphs" not in plot1_info:
        raise RuntimeError("Plotting disabled?")

    return [plot1_info[f"graph[{i + 1}]"] for i in range(plot1_info["num_graphs"])]


def plot_graph(
    tao: Tao,
    region_name: str,
    graph_name: str,
    ax: Optional[matplotlib.axes.Axes] = None,
):
    graph_info = tao.plot_graph(f"{region_name}.{graph_name}")
    graph_type = graph_info["graph^type"]

    logger.debug(f"Plotting {region_name}.{graph_name} ({graph_type})")

    if ax is None:
        _, ax = plt.subplots()
        assert ax is not None

    if graph_type == "floor_plan":
        return plot_floor_plan(
            tao=tao,
            ax=ax,
            region_name=region_name,
            graph_name=graph_name,
            graph_info=graph_info,
        )
    if graph_type == "lat_layout":
        return plot_lat_layout(
            tao=tao,
            ax=ax,
            region_name=region_name,
            graph_name=graph_name,
            graph_info=graph_info,
        )
    return plot_normal_graph(
        tao,
        region_name=region_name,
        graph_name=graph_name,
        graph_info=graph_info,
        ax=ax,
    )


def plot_region(tao: Tao, region_name: str):
    fig = plt.figure()

    graph_names = get_graphs_in_region(tao, region_name=region_name)

    if not len(graph_names):
        return

    # gs = fig.add_gridspec(nrows=number_graphs, ncols=1, height_ratios=graph_heights)
    gs = fig.subplots(nrows=len(graph_names), ncols=1, sharex=True, squeeze=False)

    for ax, graph_name in zip(gs[:, 0], graph_names):
        plot_graph(
            tao=tao,
            region_name=region_name,
            graph_name=graph_name,
            ax=ax,
        )
