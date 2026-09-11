import pytest
from pytest import FixtureRequest

from .. import SubprocessTao, Tao
from ..plotting import (
    TaoAxisSettings,
    TaoCurveSettings,
    TaoFloorPlanSettings,
    TaoGraphSettings,
)
from ..plotting.plot import BasicGraph
from ..plotting.types import Limit
from .conftest import BackendName, get_example, test_artifacts

MULTI_UNI_INIT = (
    "$ACC_ROOT_DIR/bmad-doc/tutorial_bmad_tao/lattice_files/multiple_universes/tao.init"
)


@pytest.fixture(scope="function")
def multi_uni_tao():
    with SubprocessTao(init_file=MULTI_UNI_INIT, noplot=True) as tao:
        yield tao


@pytest.fixture(scope="function")
def multi_uni_mpl_tao():
    with SubprocessTao(init_file=MULTI_UNI_INIT, plot="mpl") as tao:
        yield tao


def test_curve_settings_empty():
    assert TaoCurveSettings().get_commands("a", "b", "c1") == []


@pytest.mark.parametrize(
    "curve_name",
    ["c1", "x", "y"],
    ids=["indexed", "named-x", "named-y"],
)
def test_curve_settings_uses_curve_name(curve_name: str):
    settings = TaoCurveSettings(ix_universe=2)
    assert settings.get_commands("r11", "g", curve_name) == [
        f"set curve r11.g.{curve_name} ix_universe = 2"
    ]


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
def test_graph_settings(settings: TaoGraphSettings, expected_commands: list[str]):
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
    xlim: Limit | None,
    ylim: Limit | None,
    expected_commands: list[str],
):
    settings = TaoGraphSettings()
    settings.xlim = xlim
    settings.ylim = ylim
    assert settings.get_commands("a", "b", graph_type="lat_layout") == expected_commands


def test_configure_curves_resolves_named_curves(multi_uni_tao: Tao):
    """
    Curve settings must target the actual Tao curve names.

    The ``orbit`` template names its curves ``x``/``y`` (not ``c1``/``c2``),
    so a positional ``c{index}`` reference raises ``CURVE NOT FOUND``.  This
    also exercises per-curve ``ix_universe`` selection for a second universe.
    """
    tao = multi_uni_tao
    manager = tao.plot_manager

    tao.cmd("place -no_buffer r11 orbit")

    manager.configure_curves(
        "r11",
        {
            1: TaoCurveSettings(ix_universe=2),
            2: TaoCurveSettings(ix_universe=2),
        },
    )

    assert tao.plot_curve("r11.g.x")["ix_universe"] == 2
    assert tao.plot_curve("r11.g.y")["ix_universe"] == 2


def _basic_graphs(graphs) -> list[BasicGraph]:
    return [graph for graph in graphs if isinstance(graph, BasicGraph)]


@pytest.mark.parametrize("ix_uni", [1, 2])
def test_prepare_graphs_ix_uni_sets_universe(multi_uni_mpl_tao: Tao, ix_uni: int):
    manager = multi_uni_mpl_tao.plot_manager
    graphs = _basic_graphs(manager.prepare_graphs_by_name("beta", ix_uni=ix_uni))
    assert graphs
    assert all(graph.info["ix_universe"] == ix_uni for graph in graphs)


def test_prepare_graphs_ix_uni_uses_universe_data(multi_uni_mpl_tao: Tao):
    """
    The example's universes start with beta_a of 10 m and 20 m, respectively.
    """
    manager = multi_uni_mpl_tao.plot_manager

    def get_first_beta(ix_uni: int) -> float:
        (graph,) = _basic_graphs(manager.prepare_graphs_by_name("beta", ix_uni=ix_uni))
        assert graph.curves[0].line is not None
        return graph.curves[0].line.ys[0]

    assert get_first_beta(1) == pytest.approx(10.0)
    assert get_first_beta(2) == pytest.approx(20.0)


def test_prepare_grid_ix_uni_sets_universe(multi_uni_mpl_tao: Tao):
    manager = multi_uni_mpl_tao.plot_manager
    graphs = _basic_graphs(manager.prepare_grid_by_names(["beta", "orbit"], ix_uni=2))
    assert len(graphs) == 2
    assert all(graph.info["ix_universe"] == 2 for graph in graphs)


def test_get_lattice_layout_graph_universe(multi_uni_mpl_tao: Tao):
    manager = multi_uni_mpl_tao.plot_manager
    assert manager.get_lattice_layout_graph(ix_uni=2).universe == 2
    # Unspecified universe should leave the layout untouched
    assert manager.get_lattice_layout_graph().universe == 2
    assert manager.get_lattice_layout_graph(ix_uni=1).universe == 1


def test_lattice_layout_resolves_default_universe(multi_uni_mpl_tao: Tao):
    tao = multi_uni_mpl_tao
    tao.cmd("set global default_universe = 2")
    assert tao.plot_manager.lattice_layout_graph.universe == 2


def test_include_layout_follows_ix_uni(multi_uni_mpl_tao: Tao):
    tao = multi_uni_mpl_tao
    tao.plot("beta", ix_uni=2, include_layout=True)
    layout = tao.plot_manager.lattice_layout_graph
    assert layout.universe == 2
    # The template's own region must not be left with a stale layout graph
    for region_name, graphs in tao.plot_manager.regions.items():
        for graph in graphs:
            assert graph.region_name == region_name


def test_bokeh_include_layout_follows_ix_uni():
    with SubprocessTao(init_file=MULTI_UNI_INIT, plot="bokeh") as tao:
        manager = tao.plot_manager
        _, app = manager.plot("beta", ix_uni=2, include_layout=True)
        app.create_state()
        assert manager.lattice_layout_graph.universe == 2


def test_floor_plan_all_universes(multi_uni_mpl_tao: Tao):
    manager = multi_uni_mpl_tao.plot_manager
    single = manager.get_floor_plan_graph(ix_uni=1)
    combined = manager.get_floor_plan_graph(ix_uni=-2)
    assert combined.info["ix_universe"] == -2
    assert len(combined.elements) == 2 * len(single.elements)


def test_curve_universe_legend_label(multi_uni_mpl_tao: Tao):
    """
    Curves explicitly pointed at a universe get a ``uni@`` legend prefix.
    """
    manager = multi_uni_mpl_tao.plot_manager
    graphs = manager.prepare_graphs_by_name(
        "orbit",
        curves={1: TaoCurveSettings(ix_universe=2)},
    )
    (graph,) = _basic_graphs(graphs)
    labels = {curve.info["name"]: curve.legend_label for curve in graph.curves}
    assert labels["x"] == "2@X"
    assert labels["y"] == "Y"


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
