"""
Tests for per-element drawing shapes (`pipe ele:shape`) and offline plotting.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ..model import Element, ElementShapes, Lattice
from ..plotting.plot import (
    FloorPlanGraph,
    LatticeLayoutGraph,
    floor_plan_elements_from_elements,
    lat_layout_elements_from_elements,
)
from ..plotting.util import floor_to_screen
from .conftest import new_tao, test_artifacts

shape_inits = pytest.mark.parametrize(
    "init",
    [
        "-init $ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_shape",
        "-init $ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_floor_orbit",
        "-init $ACC_ROOT_DIR/regression_tests/pipe_test/cesr/tao.init",
    ],
)

floor_orbit_init = "-init $ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_floor_orbit"


def place_floor_plan(tao, region: str = "r12", *, view: str = "xz", rotation: float = 0.125):
    """Place a floor_plan graph with a non-default view and rotation; returns its graph info."""
    tao.cmd(f"place {region} floor_plan")
    tao.cmd(f"set graph {region} floor_plan%view = {view}")
    tao.cmd(f"set graph {region} floor_plan%rotation = {rotation}")
    graph_info = tao.plot_graph(f"{region}.g")
    assert graph_info["graph^type"] == "floor_plan"
    assert graph_info["floor_plan_view"] == view
    assert graph_info["floor_plan_rotation"] == pytest.approx(rotation)
    return graph_info


@shape_inits
def test_lat_layout_shapes_match_plot_lat_layout(tao_cls, init: str):
    with new_tao(tao_cls, init, external_plotting=False) as tao:
        scale = tao.plot_page()["lat_layout_shape_scale"]
        rows = {
            (row["ix_branch"], row["ix_ele"]): row
            for row in tao.plot_lat_layout(ix_uni=1, ix_branch=0)
        }
        eles = tao.eles("*", track_only=True, defaults=False, shapes=True)

    assert eles
    for ele in eles:
        assert ele.shapes is not None
        row = rows.get((ele.head.ix_branch, ele.head.ix_ele))
        drawn = [shape for shape in ele.shapes.lat_layout if shape.draw]
        if row is None:
            assert not drawn, ele.head.name
            continue
        (shape,) = drawn
        assert shape.shape == row["shape"]
        assert shape.color == row["color"]
        assert shape.label_name == row["label_name"]
        assert shape.line_width == row["line_width"]
        assert shape.y1 * scale == pytest.approx(row["y1"])
        assert shape.y2 * scale == pytest.approx(row["y2"])


def test_floor_plan_shapes_match_floor_plan(tao_cls):
    with new_tao(tao_cls, floor_orbit_init, external_plotting=False) as tao:
        graph_info = place_floor_plan(tao)
        scale = tao.plot_page()["floor_plan_shape_scale"]
        rows = tao.floor_plan("r12.g")
        eles = {
            (ele.head.ix_branch, ele.head.ix_ele): ele
            for ele in tao.eles("*", defaults=False, floor=True, shapes=True)
        }

    assert rows
    for row in rows:
        ele = eles[(row["branch_index"], row["index"])]
        assert ele.floor is not None and ele.shapes is not None
        begin, end = ele.floor.beginning.actual, ele.floor.end.actual
        assert begin is not None and end is not None

        for position, suffix in ((begin, "end1"), (end, "end2")):
            x, y, theta = floor_to_screen(
                position.x,
                position.y,
                position.z,
                position.theta,
                position.phi,
                view=graph_info["floor_plan_view"],
                rotation=graph_info["floor_plan_rotation"],
            )
            assert x == pytest.approx(row[f"{suffix}_r1"], abs=1e-6)
            assert y == pytest.approx(row[f"{suffix}_r2"], abs=1e-6)
            dtheta = (theta - row[f"{suffix}_theta"]) % (2 * math.pi)
            assert min(dtheta, 2 * math.pi - dtheta) == pytest.approx(0.0, abs=1e-6)

        drawn = [shape for shape in ele.shapes.floor_plan if shape.draw]
        if not drawn:
            assert row["shape"] == ""
            continue
        (shape,) = drawn
        assert shape.shape == row["shape"]
        assert shape.color == row["color"]
        assert shape.y1 * scale == pytest.approx(row["y1"])
        assert shape.y2 * scale == pytest.approx(row["y2"])


@pytest.mark.parametrize(
    ("view", "rotation", "expected"),
    [
        ("zx", 0.0, (3.0, 1.0)),
        ("xz", 0.0, (1.0, 3.0)),
        ("zy", 0.0, (3.0, 2.0)),
        ("zx", 0.25, (-1.0, 3.0)),
        ("zx", 0.5, (-3.0, -1.0)),
    ],
)
def test_floor_to_screen(view: str, rotation: float, expected: tuple[float, float]):
    x, y, theta = floor_to_screen(1.0, 2.0, 3.0, 0.0, 0.0, view=view, rotation=rotation)
    assert (x, y) == pytest.approx(expected)
    # theta=phi=0 points along +z; the screen angle is that direction plus the rotation.
    z_axis = {"zx": 0.0, "xz": math.pi / 2, "zy": 0.0}[view]
    assert theta == pytest.approx(z_axis + 2 * math.pi * rotation)


def test_floor_to_screen_bad_view():
    with pytest.raises(ValueError):
        floor_to_screen(0.0, 0.0, 0.0, 0.0, 0.0, view="q")


def test_shapes_default_on_and_roundtrip(tao_cls):
    with new_tao(tao_cls, floor_orbit_init, external_plotting=False) as tao:
        assert Element.from_tao(tao, "B1").shapes is not None
        assert Element.from_tao(tao, "B1", defaults=False).shapes is None
        ele = Element.from_tao(tao, "B1", defaults=False, shapes=True)
        shapes = ElementShapes.from_tao(tao, "B1")

    assert ele.shapes == shapes
    assert shapes.lat_layout and shapes.floor_plan
    assert shapes.lat_layout[0].base_shape == "box"
    assert shapes.lat_layout[0].prefix == ""

    restored = Element.model_validate(ele.model_dump(mode="json"))
    assert restored.shapes == shapes


def test_shape_prefix_split():
    from ..model.ele.sections import ElementShape

    shape = ElementShape(
        ix_shape=1,
        shape="asym_var:box",
        color="red",
        line_width=1,
        y1=1.0,
        y2=0.0,
        label_name="",
        draw=True,
        multi=False,
    )
    assert shape.prefix == "asym_var"
    assert shape.base_shape == "box"


def test_undrawn_shapes_reported(tao_cls):
    with new_tao(tao_cls, floor_orbit_init, external_plotting=False) as tao:
        (before,) = [shape for shape in tao.ele_shape("B1", who="lat_layout") if shape["draw"]]
        (config,) = [
            info
            for info in tao.shape_list("lat_layout")
            if info["shape_index"] == before["ix_shape"]
        ]
        assert config["shape_draw"]
        tao.shape_set(
            who="lat_layout",
            shape_index=config["shape_index"],
            ele_name=config["ele_name"],
            shape=config["shape"],
            color=config["color"],
            shape_size=config["shape_size"],
            type_label=config["type_label"],
            shape_draw="F",
            multi_shape="T" if config["multi_shape"] else "F",
            line_width=config["line_width"],
        )
        after = tao.ele_shape("B1", who="lat_layout")
        layout_rows = tao.plot_lat_layout(ix_uni=1, ix_branch=0)
        ele = Element.from_tao(tao, "B1", defaults=False, shapes=True)

    assert after[0]["ix_shape"] == before["ix_shape"]
    assert after[0]["draw"] is False
    assert after[0]["shape"] == before["shape"]
    drawn = [shape for shape in after if shape["draw"]]
    # Tao only lists B1 in the layout if some later shape still draws it.
    assert bool(drawn) == any(row["ix_ele"] == ele.head.ix_ele for row in layout_rows)
    assert ele.shapes is not None
    assert [shape.draw for shape in ele.shapes.lat_layout] == [
        shape["draw"] for shape in after
    ]
    offline = lat_layout_elements_from_elements([ele])
    assert len(offline) == len(drawn)


def test_offline_lat_layout_matches_tao(tao_cls):
    with new_tao(tao_cls, floor_orbit_init, external_plotting=False) as tao:
        ref = LatticeLayoutGraph.from_tao(tao, "layout", "g")
        eles = tao.eles("*", defaults=False, shapes=True)

    by_index = {(elem.info["ix_branch"], elem.info["ix_ele"]): elem for elem in ref.elements}
    # Tao's layout omits super slaves and multipass lords; mirror its selection.
    offline = lat_layout_elements_from_elements(
        [ele for ele in eles if (ele.head.ix_branch, ele.head.ix_ele) in by_index],
        x_min=ref.info["x_min"],
        x_max=ref.info["x_max"],
    )

    assert len(offline) == len(ref.elements) > 0
    for elem in offline:
        expected = by_index[(elem.info["ix_branch"], elem.info["ix_ele"])]
        assert elem.shape == expected.shape
        assert elem.color == expected.color
        assert elem.width == expected.width
        assert elem.annotations == expected.annotations


def _shapes_close(a, b) -> bool:
    if type(a) is not type(b):
        return False
    if a is None:
        return True
    da, db = a.__dict__, b.__dict__
    return all(
        np.allclose(da[key], db[key], atol=1e-6)
        if isinstance(da[key], (float, int))
        else da[key] == db[key]
        for key in da
    )


def test_offline_floor_plan_matches_tao(tao_cls):
    with new_tao(tao_cls, floor_orbit_init, external_plotting=False) as tao:
        graph_info = place_floor_plan(tao)
        ref = FloorPlanGraph.from_tao(tao, "r12", "g")
        eles = tao.eles("*", defaults=False, attrs=True, floor=True, shapes=True)

    by_index = {(elem.branch_index, elem.index): elem for elem in ref.elements}
    offline = floor_plan_elements_from_elements(
        [ele for ele in eles if (ele.head.ix_branch, ele.head.ix_ele) in by_index],
        view=graph_info["floor_plan_view"],
        rotation=graph_info["floor_plan_rotation"],
        size_is_absolute=graph_info["floor_plan_size_is_absolute"],
    )

    assert len(offline) == len(ref.elements) > 0
    for elem in offline:
        expected = by_index[(elem.branch_index, elem.index)]
        assert _shapes_close(elem.shape, expected.shape), (elem.shape, expected.shape)
        assert len(elem.annotations) == len(expected.annotations)


@pytest.fixture
def offline_lattice(tao_cls, tmp_path) -> Lattice:
    """A lattice archived with Tao and reloaded without it."""
    with new_tao(tao_cls, floor_orbit_init, external_plotting=False) as tao:
        Lattice.from_tao_tracking(tao).write(tmp_path / "lattice.msgpack")
    return Lattice.from_file(tmp_path / "lattice.msgpack")


def test_offline_mpl_plots(offline_lattice: Lattice, request: pytest.FixtureRequest):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..plotting import mpl

    layout = lat_layout_elements_from_elements(offline_lattice.elements)
    floor = floor_plan_elements_from_elements(offline_lattice.elements)
    assert layout and any(elem.shape is not None for elem in floor)

    fig, (ax_layout, ax_floor) = plt.subplots(2, 1)
    assert mpl.plot_lat_layout_elements(layout, ax_layout) is ax_layout
    assert mpl.plot_floor_plan_elements(floor, ax_floor) is ax_floor
    assert not ax_layout.yaxis.get_visible()
    assert ax_floor.get_aspect() == 1.0
    # Shapes landed as artists on the axes.
    assert ax_layout.collections or ax_layout.patches
    assert ax_floor.collections or ax_floor.patches

    test_artifacts.mkdir(exist_ok=True)
    fig.savefig(test_artifacts / f"{request.node.name}.png")
    plt.close(fig)

    # Creating the axes is optional.
    assert mpl.plot_lat_layout_elements(layout) is not None
    plt.close("all")


def test_offline_bokeh_figures(offline_lattice: Lattice):
    import bokeh.models

    from ..plotting import bokeh as pbokeh

    layout = lat_layout_elements_from_elements(offline_lattice.elements)
    floor = floor_plan_elements_from_elements(offline_lattice.elements)

    fig = pbokeh.lat_layout_figure(layout, title="layout", height=200, width=800)
    assert not fig.yaxis[0].visible
    assert isinstance(fig.xaxis[0].ticker, bokeh.models.FixedTicker)
    assert set(fig.xaxis[0].major_label_overrides.values()) == {e.name for e in layout}
    assert any(isinstance(tool, bokeh.models.HoverTool) for tool in fig.tools)
    assert fig.renderers

    fig = pbokeh.floor_plan_figure(floor, title="floor", height=400, width=800)
    assert fig.match_aspect
    assert fig.renderers
