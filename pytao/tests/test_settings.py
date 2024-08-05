import pytest

from typing import List

from ..plotting import (
    TaoCurveSettings,
    TaoGraphSettings,
    TaoAxisSettings,
    TaoFloorPlanSettings,
)


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
