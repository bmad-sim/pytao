import math
import logging
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.collections
import matplotlib.patches
import matplotlib.path

from pytao import Tao

from . import pgplot, util


logger = logging.getLogger(__name__)


def plot_graph(
    tao: Tao,
    region_name: str,
    graph_name: str,
    ax: matplotlib.axes.Axes,
    graph_info: Optional[dict] = None,
):
    # Graph "region.graph" full name, EG: "r13.g" or "top.x"
    graph_full_name = f"{region_name}.{graph_name}"

    if graph_info is None:
        graph_info = tao.plot_graph(f"{region_name}.{graph_name}")
        assert graph_info is not None

    # List of curve names
    all_curve_names = [graph_info[f"curve[{i + 1}]"] for i in range(graph_info["num_curves"])]

    # List of curve parameters.
    curve_infos = [tao.plot_curve(graph_full_name + "." + i) for i in all_curve_names]

    histogram_infos = [tao.plot_histogram(f"{graph_full_name}.{i}") for i in all_curve_names]

    # Plot Data

    # List of data needed to plot line and symbol graphs
    # Includes points, and line and symbol information for each curve
    line_list = []
    for i, curve_info in enumerate(curve_infos):
        curve_name = curve_info["name"]
        points = [
            (line["x"], line["y"])
            for line in tao.plot_line(region_name, graph_name, curve_name)
        ]
        try:
            symbols = [
                (sym["x_symb"], sym["y_symb"])
                for sym in tao.plot_symbol(region_name, graph_name, curve_name, x_or_y="")
            ]
        except RuntimeError:
            symbols = []

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
        if (
            symbol_type in ("dot", "1")
            or symbol_type.endswith("filled")
            or symbol_type.startswith("-")
        ):  # determine if symbol should be filled
            marker_color = curve_info["symbol"]["color"]
        elif pgplot.fills[symbol_info["fill_pattern"]] == "solid":
            marker_color = curve_info["symbol"]["color"]
        else:
            marker_color = "none"

        # marker_size
        if curve_info["draw_symbols"] and pgplot.symbols[symbol_type]:
            # symbol size if drawn
            marker_size = curve_info["symbol"]["height"]
        else:
            marker_size = 0

        # marker
        marker = pgplot.symbols.get(symbol_type, ".")
        # symbol_line_width
        symbol_line_width = curve_info["symbol"]["line_width"]

        xpoints = [p[0] for p in points]
        ypoints = [p[1] for p in points]
        symbol_xs = [p[0] for p in symbols]
        symbol_ys = [p[1] for p in symbols]
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
            logger.error("No points found, make sure data is properly initialized")
            return
        # boundaries for wave analysis rectangles

        graph_type = graph_info["graph^type"]
        if graph_type in {"data", "wave.0", "wave.a", "wave.b"}:
            line_list.append(
                ax.plot(
                    xpoints,
                    ypoints,
                    color=line_color,
                    linestyle=line_style,
                    linewidth=line_width / 2,
                )
            )
            ax.plot(
                symbol_xs,
                symbol_ys,
                color=symbol_color,
                linewidth=0,
                markerfacecolor=marker_color,
                markersize=marker_size / 2,
                marker=marker,
                mew=symbol_line_width / 2,
            )

            # Wave region boundaries
            if graph_type != "data":  # wave analysis rectangles
                wave = tao.wave("params")
                a1, a2 = wave["ix_a1"], wave["ix_a2"]
                b1, b2 = wave["ix_b1"], wave["ix_b2"]

            if symbol_color in {"blue", "navy", "cyan", "green", "purple"}:
                wave_color = "orange"
            else:
                wave_color = "blue"
            # wave analysis rectangle color

            if graph_type in {"wave.0", "wave.a"}:
                ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (a1, y_min),
                        a2 - a1,
                        y_max - y_min,
                        fill=False,
                        color=wave_color,
                    )
                )
            elif graph_type in {"wave.0", "wave.b"}:
                ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (b1, y_min),
                        b2 - b1,
                        y_max - y_min,
                        fill=False,
                        color=wave_color,
                    )
                )
        # line and symbol graphs

        elif graph_type == "dynamic_aperture":
            line_list.append(
                ax.plot(
                    xpoints,
                    ypoints,
                    color=line_color,
                    linestyle=line_style,
                    linewidth=line_width / 2,
                )
            )
            ax.plot(
                symbol_xs,
                symbol_ys,
                color=symbol_color,
                linewidth=0,
                markerfacecolor=marker_color,
                markersize=marker_size / 2,
                marker=marker,
                mew=symbol_line_width / 2,
            )
        # dynamic aperture graphs

        elif graph_type == "phase_space":
            if xpoints:
                line_list.append(
                    ax.plot(
                        xpoints,
                        ypoints,
                        color=line_color,
                        linestyle=line_style,
                        linewidth=line_width / 2,
                    )
                )
                ax.plot(
                    symbol_xs,
                    symbol_ys,
                    color=symbol_color,
                    linewidth=0,
                    markerfacecolor=marker_color,
                    markersize=marker_size / 2,
                    marker=marker,
                    mew=symbol_line_width / 2,
                )
            else:
                line_list.append(
                    ax.plot(
                        symbol_xs,
                        symbol_ys,
                        color=symbol_color,
                        linewidth=0,
                        markerfacecolor=marker_color,
                        markersize=marker_size / 2,
                        marker=marker,
                        mew=symbol_line_width / 2,
                    )
                )
        # phase space graphs

        elif graph_type == "histogram":
            line_list.append(
                ax.hist(
                    xpoints,
                    bins=int(histogram_infos[i]["number"]),
                    weights=ypoints,
                    histtype="step",
                    color=symbol_color,
                )
            )
        # histogram

    graph_type = graph_info["graph^type"]
    if graph_type == "key_table":
        raise NotImplementedError("key table is not available in the GUI")

    if graph_type in {"lat_layout", "floor_plan"} or not graph_info["draw_axes"]:
        # hides axes if draw_axes is turned off
        plt.axis("off")

    if graph_info["why_invalid"]:
        raise ValueError(
            graph_info["why_invalid"] + ", make sure graph is properly initialized"
        )

    title, title_suffix = graph_info["title"], graph_info["title_suffix"]
    plt.title(pgplot.mpl_string(f"{title} {title_suffix}"))
    # plot title

    legend_items = []  # legends for each graph
    labels = []  # labels in each legend
    try:
        for idx, curve_info in enumerate(curve_infos):
            legend_items.append(line_list[idx][0])
            legend_text = pgplot.mpl_string(curve_info["legend_text"])
            data_type = pgplot.mpl_string(curve_info["data_type"])
            if legend_text:
                labels.append(legend_text)
            elif data_type == "physical_aperture":
                labels.append(pgplot.mpl_string(curve_info["data_type"]))
            else:
                labels.append("")
        # list of curves to be added to a legend and list of labels for each curve in the legend
    except IndexError:
        raise NotImplementedError("unknown graph type")

    if (
        (graph_info["draw_curve_legend"] and labels != [""])
        and graph_info["graph^type"] != "lat_layout"
        and graph_info["graph^type"] != "floor_plan"
    ):
        ax.legend(legend_items, labels)
    # plot legend

    plt.xlabel(pgplot.mpl_string(graph_info["x_label"]))
    plt.ylabel(pgplot.mpl_string(graph_info["y_label"]))
    # plot axis labels

    ax.grid(graph_info["draw_grid"], which="major", axis="both")
    # plot grid

    plt.xlim(graph_info["x_min"], graph_info["x_max"])
    plt.ylim(graph_info["y_min"], graph_info["y_max"])
    # set axis limits

    ax.set_axisbelow(True)
    # place graphs over grid lines


def plot_lat_layout(
    tao: Tao,
    ax: matplotlib.axes.Axes,
    region_name: str,
    graph_name: str,
    graph_info: Optional[dict] = None,
):
    if graph_info is None:
        graph_info = tao.plot_graph(f"{region_name}.{graph_name}")
        assert graph_info is not None

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

        # Case where element is wrapped round the lattice ends.
        else:
            try:
                s_min = layout_info["x_min"]
                s_max = layout_info["x_max"]
            except KeyError:
                logger.exception("Missing xmin/xmax")
                continue

            # Draw wrapped box element
            if shape == "box":
                ax.plot([s1, s_max], [y1, y1], lw=wid, color=color)
                ax.plot([s1, s_max], [y2, y2], lw=wid, color=color)
                ax.plot([s_min, s2], [y1, y1], lw=wid, color=color)
                ax.plot([s_min, s2], [y2, y2], lw=wid, color=color)
                ax.plot([s1, s1], [y1, y2], lw=wid, color=color)
                ax.plot([s2, s2], [y1, y2], lw=wid, color=color)

            # Draw wrapped xbox element
            elif shape == "xbox":
                ax.plot([s1, s_max], [y1, y1], lw=wid, color=color)
                ax.plot([s1, s_max], [y2, y2], lw=wid, color=color)
                ax.plot([s1, s_max], [y1, 0], lw=wid, color=color)
                ax.plot([s1, s_max], [y2, 0], lw=wid, color=color)
                ax.plot([s_min, s2], [y1, y1], lw=wid, color=color)
                ax.plot([s_min, s2], [y2, y2], lw=wid, color=color)
                ax.plot([s_min, s2], [0, y1], lw=wid, color=color)
                ax.plot([s_min, s2], [0, y2], lw=wid, color=color)
                ax.plot([s1, s1], [y1, y2], lw=wid, color=color)
                ax.plot([s2, s2], [y1, y2], lw=wid, color=color)

            # Draw wrapped x element
            elif shape == "x":
                ax.plot([s1, s_max], [y1, 0], lw=wid, color=color)
                ax.plot([s1, s_max], [y2, 0], lw=wid, color=color)
                ax.plot([s_min, s2], [0, y1], lw=wid, color=color)
                ax.plot([s_min, s2], [0, y2], lw=wid, color=color)

            # Draw wrapped bow tie element
            elif shape == "bow_tie":
                ax.plot([s1, s_max], [y1, y1], lw=wid, color=color)
                ax.plot([s1, s_max], [y2, y2], lw=wid, color=color)
                ax.plot([s1, s_max], [y1, 0], lw=wid, color=color)
                ax.plot([s1, s_max], [y2, 0], lw=wid, color=color)
                ax.plot([s_min, s2], [y1, y1], lw=wid, color=color)
                ax.plot([s_min, s2], [y2, y2], lw=wid, color=color)
                ax.plot([s_min, s2], [0, y1], lw=wid, color=color)
                ax.plot([s_min, s2], [0, y2], lw=wid, color=color)

            # Draw wrapped diamond element
            elif shape == "diamond":
                ax.plot([s1, s_max], [0, y1], lw=wid, color=color)
                ax.plot([s1, s_max], [0, y2], lw=wid, color=color)
                ax.plot([s_min, s2], [y1, 0], lw=wid, color=color)
                ax.plot([s_min, s2], [y2, 0], lw=wid, color=color)

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


def plot_floor_plan(
    tao: Tao,
    ax: matplotlib.axes.Axes,
    region_name: str,
    graph_name: str,
    graph_info: Optional[dict] = None,
):
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
    # list of plotting parameter strings from tao command python floor_building_wall

    building_wall_curves = set(graph["index"] for graph in building_wall_graph)

    building_wall_curves = list(set(building_wall_curves))  # list of unique curve indices

    building_wall_types = {wall["index"]: wall["name"] for wall in tao.building_wall_list()}
    # dictionary where keys are wall indices and values are the corresponding building wall types

    elem_to_color = {
        elem["ele_id"].split(":")[0]: pgplot.mpl_color(elem["color"])
        for elem in tao.shape_list("floor_plan")
    }

    for curve_name in sorted(building_wall_curves):
        fbwIndexList = []  # index of point in curve
        fbwXList = []  # list of point x coordinates
        fbwYList = []  # list of point y coordinates
        fbwRadiusList = []  # straight line if element has 0 or missing radius
        for bwg in building_wall_graph:
            if curve_name == bwg["index"]:
                fbwIndexList.append(bwg["point"])
                fbwXList.append(bwg["offset_x"])
                fbwYList.append(bwg["offset_y"])
                fbwRadiusList.append(bwg["radius"])

        k = max(fbwIndexList)  # max line index

        while k > 1:
            kIndex = fbwIndexList.index(k)
            mIndex = fbwIndexList.index(k - 1)  # adjacent point to connect to
            if building_wall_types[str(curve_name)] not in elem_to_color:
                # TODO: This is a temporary fix to deal with building wall segments
                # that don't have an associated graph_info shape
                # Currently this will fail to match to wild cards
                # in the shape name (e.g. building_wall::* should match
                # to every building wall segment, but currently it
                # matches to none).  A more sophisticated way of getting the
                # graph_info shape settings for building walls will be required
                # in the future, either through a python command in tao or
                # with a method on the python to match wild cards to wall segment names
                print(
                    "No graph_info shape defined for building_wall segment "
                    + building_wall_types[str(curve_name)]
                )
                k -= 1
                continue

            if fbwRadiusList[kIndex] == 0:  # draw building wall line
                ax.plot(
                    [fbwXList[kIndex], fbwXList[mIndex]],
                    [fbwYList[kIndex], fbwYList[mIndex]],
                    color=elem_to_color[building_wall_types[str(curve_name)]],
                )

            else:  # draw building wall arc
                centers = util.circle_intersection(
                    fbwXList[mIndex],
                    fbwYList[mIndex],
                    fbwXList[kIndex],
                    fbwYList[kIndex],
                    abs(fbwRadiusList[kIndex]),
                )
                # radius and endpoints specify 2 possible circle centers for arcs
                mpx = (fbwXList[mIndex] + fbwXList[kIndex]) / 2
                mpy = (fbwYList[mIndex] + fbwYList[kIndex]) / 2
                if (
                    np.arctan2((fbwYList[mIndex] - mpy), (fbwXList[mIndex] - mpx))
                    < np.arctan2(centers[0][1], centers[0][0])
                    < np.arctan2((fbwYList[mIndex] - mpy), (fbwXList[mIndex] - mpx))
                    and fbwRadiusList[kIndex] > 0
                ):
                    center = (centers[1][0], centers[1][1])
                elif (
                    np.arctan2((fbwYList[mIndex] - mpy), (fbwXList[mIndex] - mpx))
                    < np.arctan2(centers[0][1], centers[0][0])
                    < np.arctan2((fbwYList[mIndex] - mpy), (fbwXList[mIndex] - mpx))
                    and fbwRadiusList[kIndex] < 0
                ):
                    center = (centers[0][0], centers[0][1])
                elif fbwRadiusList[kIndex] > 0:
                    center = (centers[0][0], centers[0][1])
                else:
                    center = (centers[1][0], centers[1][1])
                # find correct center

                m_angle = 360 + math.degrees(
                    np.arctan2(
                        (fbwYList[mIndex] - center[1]),
                        (fbwXList[mIndex] - center[0]),
                    )
                )
                k_angle = 360 + math.degrees(
                    np.arctan2(
                        (fbwYList[kIndex] - center[1]),
                        (fbwXList[kIndex] - center[0]),
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

                ax.add_patch(
                    matplotlib.patches.Arc(
                        center,
                        fbwRadiusList[kIndex] * 2,
                        fbwRadiusList[kIndex] * 2,
                        theta1=t1,
                        theta2=t2,
                        color=elem_to_color[building_wall_types[str(curve_name)]],
                    )
                )
                # draw building wall arc

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


def plot_region(tao: Tao, region_name: str):
    # Creates plotting figure
    fig = plt.figure()

    # List of plotting parameter strings from tao command python plot1
    plot1_info = tao.plot1(region_name)

    # List of graph names and heights
    graph_names = [plot1_info[f"graph[{i + 1}]"] for i in range(plot1_info["num_graphs"])]

    # gs = fig.add_gridspec(nrows=number_graphs, ncols=1, height_ratios=graph_heights)
    gs = fig.subplots(nrows=len(graph_names), ncols=1, sharex=True, squeeze=False)

    if not len(graph_names):
        return

    graph_info = {}
    for ax, graph_name in zip(gs[:, 0], graph_names):
        # Create plots in figure, second line also makes x axes scale together

        graph_info = tao.plot_graph(f"{region_name}.{graph_name}")
        graph_type = graph_info["graph^type"]

        print(f"Plotting {region_name}.{graph_name} ({graph_type})")
        if graph_type == "floor_plan":
            plot_floor_plan(
                tao=tao,
                ax=ax,
                region_name=region_name,
                graph_name=graph_name,
                graph_info=graph_info,
            )
        elif graph_type == "lat_layout":
            plot_lat_layout(
                tao=tao,
                ax=ax,
                region_name=region_name,
                graph_name=graph_name,
                graph_info=graph_info,
            )
        else:
            plot_graph(
                tao,
                region_name=region_name,
                graph_name=graph_name,
                graph_info=graph_info,
                ax=ax,
            )


def main():
    import os

    if 1:
        init = os.path.expandvars(
            "-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_floor_orbit"
        )
        tao = Tao(init)
        plot_region(tao, "r33")
        plot_region(tao, "layout")
        # tao.cmd("place layout floor_plan")
        # plot_region(tao, "bottom")

    else:
        init = os.path.expandvars("-init $ACC_ROOT_DIR/bmad-doc/tao_examples/cesr/tao.init")
        tao = Tao(init)
        # init += " -noplot -external_plotting"
        plot_region(tao, "top")
        tao.cmd("place bottom floor_plan")
        plot_region(tao, "bottom")
        tao.cmd("place bottom lat_layout")
        plot_region(tao, "bottom")
    plt.show()


if __name__ == "__main__":
    main()
