from ..plotting import TaoCurveSettings, TaoGraphSettings


def test_curve_settings_empty():
    assert TaoCurveSettings().get_commands("a", "b", 0) == []


def test_graph_settings_empty():
    assert TaoGraphSettings().get_commands("a", "b") == []
