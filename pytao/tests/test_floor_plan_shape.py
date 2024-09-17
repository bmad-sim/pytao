import logging
import math
import re
from typing import Union

import bokeh.io
import bokeh.layouts
import bokeh.plotting
import bokeh.resources
import matplotlib.axes
import matplotlib.pyplot as plt
import pytest
from bokeh.plotting import figure

from .. import SubprocessTao, Tao
from ..plotting.bokeh import _draw_floor_plan_shapes
from ..plotting.floor_plan_shapes import (
    AnyFloorPlanShape,
    BowTie,
    Box,
    Circle,
    Diamond,
    LetterX,
    SBend,
    Triangle,
    XBox,
)
from ..plotting.mpl import plot_floor_plan_shape as mpl_plot_floor_plan_shape
from ..plotting.plot import FloorPlanElement
from .conftest import test_artifacts

logger = logging.getLogger(__name__)


AnyTao = Union[Tao, SubprocessTao]


def draw_floor_plan_shape(
    fig: figure,
    shape: AnyFloorPlanShape,
):
    elem = FloorPlanElement(
        branch_index=0,
        index=0,
        info={
            "branch_index": 0,
            "color": "",
            "ele_key": "",
            "end1_r1": 0.0,
            "end1_r2": 0.0,
            "end1_theta": 0.0,
            "end2_r1": 0.0,
            "end2_r2": 0.0,
            "end2_theta": 0.0,
            "index": 0,
            "label_name": "",
            "line_width": 0.0,
            "shape": "",
            "y1": 0.0,
            "y2": 0.0,
        },
        annotations=[],
        shape=shape,
    )
    _draw_floor_plan_shapes(fig, elems=[elem])


@pytest.fixture(autouse=True, scope="function")
def _plot_show_to_savefig(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    # plot_backend: BackendName,
):
    index = 0

    def savefig():
        nonlocal index
        test_artifacts.mkdir(exist_ok=True)
        for fignum in plt.get_fignums():
            plt.figure(fignum)
            name = re.sub(r"[/\\]", "_", request.node.name)
            filename = test_artifacts / f"{name}_{index}.png"
            print(f"Saving figure (_plot_show_to_savefig fixture) to {filename}")
            plt.savefig(filename)
            index += 1
        plt.close("all")

    monkeypatch.setattr(plt, "show", savefig)
    yield
    plt.show()


def make_shapes(width: float, height: float, angle_low: int, angle_high: int):
    for angle in range(angle_low, angle_high, 5):
        x = angle
        for shape in [
            Box(
                x1=x,
                y1=0.0,
                x2=x + width,
                y2=0,
                off1=width,
                off2=height,
                angle_start=math.radians(angle),
            ),
            XBox(
                x1=x,
                y1=height * 3,
                x2=x + width,
                y2=height * 3,
                off1=width,
                off2=height,
                angle_start=math.radians(angle),
            ),
            LetterX(
                x1=x,
                y1=height * 6,
                x2=x + width,
                y2=height * 6,
                off1=width,
                off2=height,
                angle_start=math.radians(angle),
            ),
            Diamond(
                x1=x,
                y1=height * 9,
                x2=x + width,
                y2=height * 9,
                off1=width,
                off2=width,  # height,
                angle_start=math.radians(angle),
            ),
            SBend(
                x1=x,
                y1=height * 12,
                x2=x + width,
                y2=height * 12,
                off1=width,
                off2=height,  # height,
                angle_start=math.radians(angle % 90),
                angle_end=math.radians((angle + 1) % 90),
                rel_angle_start=0,
                rel_angle_end=0,
            ),
            Circle(
                x1=x,
                y1=height * 15,
                x2=x + width,
                y2=height * 15,
                off1=width,
                off2=height,
                angle_start=math.radians(angle),
            ),
            BowTie(
                x1=x,
                y1=height * 18,
                x2=x + width,
                y2=height * 18,
                off1=width,
                off2=height,
                angle_start=math.radians(angle),
            ),
            Triangle(
                orientation="u",
                x1=x,
                y1=height * 21,
                x2=x + width,
                y2=height * 21,
                off1=width,
                off2=height,
                angle_start=math.radians(angle),
            ),
            Triangle(
                orientation="d",
                x1=x,
                y1=height * 24,
                x2=x + width,
                y2=height * 24,
                off1=width,
                off2=height,
                angle_start=math.radians(angle),
            ),
            Triangle(
                orientation="l",
                x1=x,
                y1=height * 27,
                x2=x + width,
                y2=height * 27,
                off1=width,
                off2=height,
                angle_start=math.radians(angle),
            ),
            Triangle(
                orientation="r",
                x1=x,
                y1=height * 30,
                x2=x + width,
                y2=height * 30,
                off1=width,
                off2=height,
                angle_start=math.radians(angle),
            ),
        ]:
            yield shape


def test_floor_plan_shapes_mpl():
    fig = plt.figure(figsize=(12, 12))
    ax = fig.subplots()
    assert isinstance(ax, matplotlib.axes.Axes)
    for shape in make_shapes(width=1, height=2, angle_low=0, angle_high=90):
        mpl_plot_floor_plan_shape(shape, ax)

    plt.ylim(-5, 85)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.subplots()
    assert isinstance(ax, matplotlib.axes.Axes)
    for shape in make_shapes(width=1, height=2, angle_low=90, angle_high=180):
        mpl_plot_floor_plan_shape(shape, ax)

    plt.ylim(-5, 85)

    plt.show()


def test_floor_plan_shapes_bokeh(request: pytest.FixtureRequest):
    bokeh.io.output_file(test_artifacts / f"{request.node.name}.html")

    fig1 = bokeh.plotting.figure(match_aspect=True)
    for shape in make_shapes(width=1, height=2, angle_low=0, angle_high=90):
        draw_floor_plan_shape(fig1, shape)

    fig2 = bokeh.plotting.figure(match_aspect=True)
    for shape in make_shapes(width=1, height=2, angle_low=90, angle_high=180):
        draw_floor_plan_shape(fig2, shape)

    bokeh.io.save(bokeh.layouts.column([fig1, fig2]), resources=bokeh.resources.INLINE)
