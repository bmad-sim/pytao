from typing import List, Optional

import pytest
from pytest import FixtureRequest

from ..plotting import (
    TaoAxisSettings,
    TaoCurveSettings,
    TaoFloorPlanSettings,
    TaoGraphSettings,
)
from ..plotting.types import Limit
from .conftest import BackendName, get_example, test_artifacts


def test_curve_settings_empty():
    assert TaoCurveSettings().get_commands("a", "b", 0) == []


def test_graph_settings_empty():
    assert TaoGraphSettings().get_commands("a", "b", graph_type="lat_layout") == []


@pytest.mark.parametrize(
    ("settings", "expected_commands"),
    [
        pytest.param(
            TaoGraphSettings(text_legend={1: "test"}),
            ["set graph a text_legend(1) = test"],
        ),
        pytest.param(
            TaoGraphSettings(box={1: 2}),
            ["set graph a box(1) = 2"],
        ),
        pytest.param(
            TaoGraphSettings(component="abc"),
            ["set graph a.b component = abc"],
        ),
        pytest.param(
            TaoGraphSettings(curve_legend_origin=(1, 1, "abc")),
            [
                "set graph a.b curve_legend_origin%x = 1.0",
                "set graph a.b curve_legend_origin%y = 1.0",
                "set graph a.b curve_legend_origin%units = abc",
            ],
        ),
        pytest.param(
            TaoGraphSettings(margin=(1, 2, 3, 4, "abc")),
            [
                "set graph a.b margin%x1 = 1.0",
                "set graph a.b margin%x2 = 2.0",
                "set graph a.b margin%y1 = 3.0",
                "set graph a.b margin%y2 = 4.0",
                "set graph a.b margin%units = abc",
            ],
        ),
        pytest.param(
            TaoGraphSettings(x=TaoAxisSettings(bounds="zero_at_end", label="text")),
            [
                "set graph a x%bounds = zero_at_end",
                "set graph a x%label = text",
            ],
        ),
        pytest.param(
            TaoGraphSettings(floor_plan=TaoFloorPlanSettings(view="xz")),
            [
                "set graph a floor_plan%view = xz",
            ],
        ),
    ],
)
def test_graph_settings(settings: TaoGraphSettings, expected_commands: List[str]):
    assert settings.get_commands("a", "b", graph_type="lat_layout") == expected_commands


@pytest.mark.parametrize(
    ("xlim", "ylim", "expected_commands"),
    [
        pytest.param(None, None, [], id="no-lims"),
        pytest.param(
            (1.0, 2.0),
            None,
            ["x_scale a 1.0 2.0"],
            id="xlim",
        ),
        pytest.param(
            None,
            (1.0, 2.0),
            ["scale -y a 1.0 2.0"],
            id="ylim",
        ),
        pytest.param(
            (1.0, 2.0),
            (1.0, 2.0),
            ["x_scale a 1.0 2.0", "scale -y a 1.0 2.0"],
            id="both",
        ),
    ],
)
def test_graph_settings_xlim_ylim(
    xlim: Optional[Limit],
    ylim: Optional[Limit],
    expected_commands: List[str],
):
    settings = TaoGraphSettings()
    settings.xlim = xlim
    settings.ylim = ylim
    assert settings.get_commands("a", "b", graph_type="lat_layout") == expected_commands


def test_plot_settings_grid(plot_backend: BackendName, request: FixtureRequest):
    example = get_example("erl")
    example.plot = plot_backend
    with example.run_context(use_subprocess=True) as tao:
        manager = tao.plot_manager
        graphs, *_ = manager.plot_grid(
            templates=["zphase", "zphase"],
            grid=(3, 2),
            include_layout=True,
            curves=[
                {1: TaoCurveSettings(ele_ref_name=r"linac.beg\1")},
                {1: TaoCurveSettings(ele_ref_name=r"linac.end\1")},
            ],
            settings=[
                TaoGraphSettings(commands=["set graph {graph} title = Test Plot 1"]),
                TaoGraphSettings(title="Test Plot 2"),
            ],
            share_x=False,
            save=test_artifacts / request.node.name,
        )
        graph1, graph2, *_ = graphs
        assert graph1.title.startswith("Test Plot 1")
        assert graph2.title.startswith("Test Plot 2")


def test_plot_settings(plot_backend: BackendName, request: FixtureRequest):
    example = get_example("erl")
    example.plot = plot_backend
    with example.run_context(use_subprocess=True) as tao:
        manager = tao.plot_manager
        graphs, *_ = manager.plot(
            "zphase",
            include_layout=True,
            curves={1: TaoCurveSettings(ele_ref_name=r"linac.beg\1")},
            settings=TaoGraphSettings(
                title="Test Plot 1",
                y=TaoAxisSettings(
                    label="Y axis label",
                ),
            ),
            share_x=False,
            save=test_artifacts / request.node.name,
        )
        graph1, *_ = graphs
        assert graph1.title.startswith("Test Plot 1")
        assert graph1.ylabel == "Y axis label"
