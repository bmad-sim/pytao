import logging
import math
import re
from typing import Union

import bokeh.io
import bokeh.layouts
import bokeh.plotting

import matplotlib.axes
import matplotlib.pyplot as plt
import pytest

from ..plotting.floor_plan_shapes import BowTie, Box, Circle, Diamond, SBend, XBox, LetterX
from ..plotting.bokeh import _plot_floor_plan_shape as plot_floor_plan_shape


from .. import SubprocessTao, Tao
from .conftest import test_artifacts

logger = logging.getLogger(__name__)


AnyTao = Union[Tao, SubprocessTao]


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
        ]:
            yield shape


def test_plot_shapes_mpl():
    fig = plt.figure(figsize=(12, 12))
    ax = fig.subplots()
    assert isinstance(ax, matplotlib.axes.Axes)
    for shape in make_shapes(width=1, height=2, angle_low=0, angle_high=90):
        shape.plot(ax)

    plt.ylim(-5, 85)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.subplots()
    assert isinstance(ax, matplotlib.axes.Axes)
    for shape in make_shapes(width=1, height=2, angle_low=90, angle_high=180):
        shape.plot(ax)

    plt.ylim(-5, 85)

    plt.show()


def test_plot_shapes_bokeh():
    bokeh.io.output_file(test_artifacts / "test_plot_shapes_bokeh.html")

    fig1 = bokeh.plotting.figure(match_aspect=True)
    for shape in make_shapes(width=1, height=2, angle_low=0, angle_high=90):
        plot_floor_plan_shape(fig1, shape, line_width=1)

    fig2 = bokeh.plotting.figure(match_aspect=True)
    for shape in make_shapes(width=1, height=2, angle_low=90, angle_high=180):
        plot_floor_plan_shape(fig2, shape, line_width=1)

    bokeh.io.save(bokeh.layouts.column([fig1, fig2]))
