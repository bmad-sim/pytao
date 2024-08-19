import logging
import re
from typing import List, Type, Union

import bokeh.io
import bokeh.layouts
import bokeh.models
import bokeh.plotting
import matplotlib.axes
import matplotlib.pyplot as plt
import pytest

from .. import SubprocessTao, Tao
from ..plotting import layout_shapes
from ..plotting.bokeh import _draw_layout_elems as bokeh_draw_layout_elems
from ..plotting.bokeh import get_tool_from_figure
from ..plotting.mpl import plot_layout_shape as mpl_plot_layout_shape
from ..plotting.plot import LatticeLayoutElement
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


def make_shapes(
    width: float,
    height: float,
    kwarg_list: List[dict],
):
    separation = width * 2.5
    s = 0
    for kwargs in kwarg_list:
        for name, cls in layout_shapes.shape_to_class.items():
            s += separation
            yield cls(
                s1=s,
                s2=s + width,
                y1=0.0,
                y2=height,
                name=name,
                **kwargs,
            )
        s += separation


def test_plot_layout_shapes_mpl():
    fig = plt.figure(figsize=(12, 12))
    ax = fig.subplots()
    assert isinstance(ax, matplotlib.axes.Axes)
    shape = None
    for shape in make_shapes(
        width=1,
        height=2,
        kwarg_list=[
            {"color": "black"},
            {"color": "blue", "line_width": 1.0},
            {"color": "green", "line_width": 2.0},
        ],
    ):
        mpl_plot_layout_shape(shape, ax)

    assert shape is not None
    plt.xlim(-5, shape.s2 + 5)
    plt.ylim(-5, 5)

    plt.show()


def bokeh_draw_layout_shape(fig: bokeh.plotting.figure, shape: layout_shapes.AnyLayoutShape):
    bokeh_draw_layout_elems(
        fig=fig,
        skip_labels=False,
        elems=[
            LatticeLayoutElement(
                info={
                    "ix_branch": 0,
                    "ix_ele": 0,
                    "ele_s_start": 0.0,
                    "ele_s_end": 0.0,
                    "line_width": 0.0,
                    "shape": "",
                    "y1": 0.0,
                    "y2": 0.0,
                    "color": "",
                    "label_name": "",
                },
                shape=shape,
                annotations=[],
                color=shape.color,
                width=shape.line_width,
            ),
        ],
    )


def test_plot_layout_shapes_bokeh():
    bokeh.io.output_file(test_artifacts / "test_plot_layout_shapes_bokeh.html")

    fig1 = bokeh.plotting.figure(match_aspect=True)
    box_zoom = get_tool_from_figure(fig1, bokeh.models.BoxZoomTool)
    if box_zoom is not None:
        box_zoom.match_aspect = True
    for shape in make_shapes(
        width=1,
        height=2,
        kwarg_list=[
            {"color": "black"},
            {"color": "blue", "line_width": 1.0},
            {"color": "green", "line_width": 2.0},
        ],
    ):
        bokeh_draw_layout_shape(fig1, shape)

    bokeh.io.save(bokeh.layouts.column([fig1]))


shape_classes = pytest.mark.parametrize(
    ("shape_cls",),
    [
        pytest.param(cls, id=shape)
        for shape, cls in layout_shapes.wrapped_shape_to_class.items()
    ],
)


@shape_classes
def test_plot_layout_wrapped_shapes_mpl(shape_cls: Type[layout_shapes.AnyWrappedLayoutShape]):
    fig = plt.figure(figsize=(3, 3))
    ax = fig.subplots()
    assert isinstance(ax, matplotlib.axes.Axes)

    shape = shape_cls(
        s1=20,  # s1 > s2 is required
        s2=10,
        y1=0,
        y2=1,
        s_min=0,
        s_max=30,
    )
    mpl_plot_layout_shape(shape, ax)

    plt.xlim(-1, 31)
    plt.ylim(-5, 10)

    plt.show()


@shape_classes
def test_plot_layout_wrapped_shapes_bokeh(
    shape_cls: Type[layout_shapes.AnyWrappedLayoutShape],
):
    bokeh.io.output_file(test_artifacts / f"test_plot_shapes_bokeh_{shape_cls.__name__}.html")

    fig1 = bokeh.plotting.figure(match_aspect=True)
    shape = shape_cls(
        s1=20,  # s1 > s2 is required
        s2=10,
        y1=0,
        y2=1,
        s_min=0,
        s_max=30,
    )
    bokeh_draw_layout_shape(fig1, shape)

    bokeh.io.save(bokeh.layouts.column([fig1]))
