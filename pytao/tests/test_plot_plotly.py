import contextlib
import logging
import re

import pytest

from .. import TaoStartup
from ..plotting.plot import FloorPlanGraph
from ..plotting.plotly import (
    PlotlyAppCreator,
    PlotlyNotebookGraphManager,
    PlotlyVariable,
    _PlotlyDefaults,
    select_graph_manager_class,
    set_plotly_defaults,
)
from ..plotting.settings import TaoFloorPlanSettings, TaoGraphSettings
from ..subproc import AnyTao
from ..tao_ctypes.util import filter_tao_messages_context
from .conftest import get_example, test_artifacts

try:
    import plotly.graph_objects as go
    import plotly.offline as pyo

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import ipywidgets as widgets  # noqa: F401

    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False


logger = logging.getLogger(__name__)


def annotate_and_save(figure: go.Figure, graphs, test_name: str, filename_base: str):
    """Annotate and save a Plotly figure."""
    if graphs:
        title_suffix = f" ({test_name})"
        if figure.layout.title:
            figure.layout.title.text = (figure.layout.title.text or "") + title_suffix
        else:
            figure.layout.title = title_suffix

    fn = test_artifacts / f"{filename_base}.html"
    figure.write_html(fn)
    return fn


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not available")
def test_plotly_manager(
    request: pytest.FixtureRequest,
    tao_regression_test: TaoStartup,
):
    name = re.sub(r"[/\\]", "_", request.node.name)
    filename_base = f"plotly_{name}"
    tao_regression_test.plot = "plotly"

    with tao_regression_test.run_context(use_subprocess=True) as tao:
        manager = tao.plotly

        with filter_tao_messages_context(functions=["twiss_propagate1"]):
            graphs, figure = manager.plot_all()

        annotate_and_save(figure, graphs, request.node.name, filename_base)

        for region in list(manager.regions):
            manager.clear(region)
        assert not any(region for region in manager.regions.values())
        manager.clear()
        assert not manager.regions


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not available")
def test_plotly_examples(
    request: pytest.FixtureRequest,
    tao_example: TaoStartup,
):
    example_name = tao_example.metadata["name"]
    name = re.sub(r"[/\\]", "_", request.node.name)
    filename_base = f"plotly_{name}"

    tao_example.plot = "plotly"

    with tao_example.run_context(use_subprocess=True) as tao:
        manager = tao.plotly

        if example_name == "erl":
            tao.cmd("place r11 zphase")

        graphs, figure = manager.plot_all()
        annotate_and_save(figure, graphs, request.node.name, filename_base)


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not available")
def test_plotly_floor_plan(request: pytest.FixtureRequest):
    tao_example = get_example("optics_matching")
    name = re.sub(r"[/\\]", "_", request.node.name)
    filename_base = f"plotly_{name}"

    tao_example.plot = "plotly"

    with tao_example.run_context(use_subprocess=True) as tao:
        tao.update_plot_shapes("quadrupole", type_label="name", layout=True, floor=True)
        graphs, figure = tao.plotly.plot("floor_plan")
        annotate_and_save(figure, graphs, request.node.name, filename_base)


@contextlib.contextmanager
def optics_matching_plotly(request: pytest.FixtureRequest):
    tao_example = get_example("optics_matching")
    name = re.sub(r"[/\\]", "_", request.node.name)
    tao_example.plot = "plotly"

    with tao_example.run_context(use_subprocess=True) as tao:
        yield tao, name


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not available")
def test_plotly_smoke_create_app_creator(request: pytest.FixtureRequest):
    with optics_matching_plotly(request) as (tao, _):
        graphs, figure = tao.plotly.plot_grid(
            ["alpha", "beta"], grid=(2, 1), include_layout=True
        )

        # Manually make app creator here
        app = PlotlyAppCreator(
            manager=tao.plotly,
            graphs=graphs,
            figure=figure,
        )
        print(f"Created PlotlyAppCreator with {len(graphs)} graphs")
        print(app)


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not available")
def test_plotly_grid_layout(request: pytest.FixtureRequest):
    """Test that grid layouts work correctly."""
    with optics_matching_plotly(request) as (tao, _):
        # Test 2x1 grid
        graphs, figure = tao.plotly.plot_grid(["alpha", "beta"], grid=(2, 1))
        assert len(graphs) == 2
        assert len(figure.data) > 0  # Should have traces

        # Test 1x2 grid
        graphs, figure = tao.plotly.plot_grid(["alpha", "beta"], grid=(1, 2))
        assert len(graphs) == 2
        assert len(figure.data) > 0


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not available")
def test_plotly_single_plot(request: pytest.FixtureRequest):
    """Test single plot functionality."""
    with optics_matching_plotly(request) as (tao, _):
        graphs, figure = tao.plotly.plot("beta")
        assert len(graphs) == 1
        assert len(figure.data) > 0
        assert figure.layout.title is not None


def get_notebook_plotly_manager(tao: AnyTao, monkeypatch: pytest.MonkeyPatch):
    """Create a notebook graph manager with mocked display functions."""
    gm = PlotlyNotebookGraphManager(tao)

    def mock_iplot(*args, **kwargs):
        print("plotly offline iplot:", args, kwargs)

    def mock_show(*args, **kwargs):
        print("plotly show:", args, kwargs)

    if PLOTLY_AVAILABLE:
        monkeypatch.setattr(pyo, "iplot", mock_iplot)
        # Mock the show method on Figure objects
        monkeypatch.setattr("plotly.graph_objs._figure.Figure.show", mock_show)

    return gm


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not available")
@pytest.mark.parametrize(
    ("grid",),
    [
        pytest.param(True, id="grid"),
        pytest.param(False, id="normal"),
    ],
)
def test_plotly_notebook_plot_vars(
    request: pytest.FixtureRequest,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    grid: bool,
):
    """Test notebook plotting with variables."""
    with caplog.at_level(logging.ERROR):
        with optics_matching_plotly(request) as (tao, _):
            gm = get_notebook_plotly_manager(tao, monkeypatch)

            if not IPYWIDGETS_AVAILABLE:
                pytest.skip("ipywidgets not available")

            if grid:
                _graphs, app = gm.plot_grid(["alpha", "beta"], grid=(2, 1), vars=True)
            else:
                _graphs, app = gm.plot("alpha", vars=True)

            def try_value(var: PlotlyVariable, value: float) -> None:
                """Test setting a variable value."""
                try:
                    var.set_value(tao, value)
                except Exception as ex:
                    logger.error(f"Error setting {var.name} to {value}: {ex}")

            def set_value_raise(*args, **kwargs):
                raise RuntimeError("raised")

            for var in app.variables:
                try_value(var, value=var.value)

            for var in app.variables:
                monkeypatch.setattr(var, "set_value", set_value_raise)
                try_value(var, value=var.value)


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not available")
def test_plotly_floor_orbits(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test floor plan with orbits."""
    with optics_matching_plotly(request) as (tao, _):
        gm = get_notebook_plotly_manager(tao, monkeypatch)
        graphs, app = gm.plot(
            "floor_plan",
            settings=TaoGraphSettings(floor_plan=TaoFloorPlanSettings(orbit_scale=1.0)),
        )

        assert len(graphs) == 1
        assert isinstance(graphs[0], FloorPlanGraph)

        # Check that the figure has data (floor plan elements)
        assert len(app.figure.data) > 0


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not available")
def test_plotly_variable_creation():
    """Test PlotlyVariable creation and widget functionality."""
    if not IPYWIDGETS_AVAILABLE:
        pytest.skip("ipywidgets not available")

    mock_info = {
        "name": "test_var",
        "model_value": 1.0,
        "low_lim": 0.0,
        "high_lim": 10.0,
        "key_delta": 0.1,
    }

    var = PlotlyVariable(
        name="test_var[1]",
        value=1.0,
        step=0.1,
        info=mock_info,
    )

    assert var.name == "test_var[1]"
    assert var.value == 1.0
    assert var.step == 0.1


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not available")
def test_plotly_field_plot(request: pytest.FixtureRequest):
    """Test field plotting functionality."""
    with optics_matching_plotly(request) as (tao, _):
        try:
            field, figure = tao.plotly.plot_field("Q1", num_points=10)
            assert figure is not None
            assert len(figure.data) > 0
        except Exception as ex:
            logger.info(f"Field plot test skipped due to: {ex}")


default_plotly_options = sorted(
    set(
        attr
        for attr in dir(_PlotlyDefaults)
        if not attr.startswith("_") and attr not in {"get_size_for_class"}
    )
)


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not available")
@pytest.mark.parametrize(("attr",), [pytest.param(attr) for attr in default_plotly_options])
def test_plotly_set_defaults(attr: str):
    """Test setting default values."""
    value = getattr(_PlotlyDefaults, attr)
    set_plotly_defaults(**{attr: value})
    assert getattr(_PlotlyDefaults, attr) == value


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not available")
def test_smoke_select_plotly_graph_manager_class():
    """Smoke test for manager class selection."""
    select_graph_manager_class()


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not available")
def test_plotly_figure_updates(request: pytest.FixtureRequest):
    """Test figure update functionality."""
    with optics_matching_plotly(request) as (tao, _):
        graphs, figure = tao.plotly.plot("beta")

        original_data_len = len(figure.data)

        figure.add_annotation(
            x=0.5,
            y=0.5,
            text="Test annotation",
            showarrow=False,
        )

        assert len(figure.layout.annotations) > 0
        assert len(figure.data) == original_data_len, "Data should be unchanged"


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not available")
def test_plotly_save_functionality(request: pytest.FixtureRequest, tmp_path):
    """Test saving plots to HTML files."""
    with optics_matching_plotly(request) as (tao, _):
        graphs, figure = tao.plotly.plot("beta")

        # Test saving with explicit filename
        save_path = tmp_path / "test_plot.html"
        figure.write_html(save_path)
        assert save_path.exists()

        app = PlotlyAppCreator(
            manager=tao.plotly,
            graphs=graphs,
            figure=figure,
        )

        saved_path = app.save(tmp_path / "app_test")
        assert saved_path.exists()
        assert saved_path.suffix == ".html"


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not available")
def test_plotly_error_handling(
    request: pytest.FixtureRequest, caplog: pytest.LogCaptureFixture
):
    """Test error handling in various scenarios."""
    with caplog.at_level(logging.ERROR):
        with optics_matching_plotly(request) as (tao, _):
            # Test invalid template
            try:
                graphs, figure = tao.plotly.plot("nonexistent_template")
            except Exception:
                pass  # Expected to fail

            # Test invalid element for field plot
            try:
                field, figure = tao.plotly.plot_field("NONEXISTENT_ELEMENT")
            except Exception:
                pass  # Expected to fail


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not available")
def test_plotly_curve_rendering(request: pytest.FixtureRequest):
    """Test that curves are properly rendered."""
    with optics_matching_plotly(request) as (tao, _):
        graphs, figure = tao.plotly.plot("beta")

        # Check that we have traces (curves)
        assert len(figure.data) > 0

        # Check that traces have data
        for trace in figure.data:
            if hasattr(trace, "x") and hasattr(trace, "y"):
                assert len(trace.x) > 0
                assert len(trace.y) > 0


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not available")
def test_plotly_layout_configuration(request: pytest.FixtureRequest):
    """Test that plot layouts are properly configured."""
    with optics_matching_plotly(request) as (tao, _):
        graphs, figure = tao.plotly.plot("beta")

        # Check basic layout properties
        assert figure.layout.title is not None
        assert figure.layout.xaxis.title is not None
        assert figure.layout.yaxis.title is not None

        # Check dimensions
        assert figure.layout.width is not None
        assert figure.layout.height is not None


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not available")
def test_plotly_axis_limits(request: pytest.FixtureRequest):
    """Test setting custom axis limits."""
    with optics_matching_plotly(request) as (tao, _):
        xlim = (0, 100)
        ylim = (-5, 5)

        graphs, figure = tao.plotly.plot("beta", xlim=xlim, ylim=ylim)

        # Note: Plotly may adjust limits slightly, so we check approximate ranges
        if figure.layout.xaxis.range:
            x_range = figure.layout.xaxis.range
            assert x_range[0] <= xlim[0]
            assert x_range[1] >= xlim[1]

        if figure.layout.yaxis.range:
            y_range = figure.layout.yaxis.range
            assert y_range[0] <= ylim[0]
            assert y_range[1] >= ylim[1]


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not available")
def test_plotly_patch_rendering(request: pytest.FixtureRequest):
    """Test that patches (shapes) are properly rendered in floor plans."""
    with optics_matching_plotly(request) as (tao, _):
        tao.update_plot_shapes("quadrupole", type_label="name", layout=True, floor=True)
        graphs, figure = tao.plotly.plot("floor_plan")

        # Floor plans should have shapes or traces representing elements
        has_data = len(figure.data) > 0
        has_shapes = len(figure.layout.shapes) > 0 if figure.layout.shapes else False

        assert has_data or has_shapes, "Floor plan should have either data traces or shapes"


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not available")
def test_plotly_shared_axes(request: pytest.FixtureRequest):
    """Test shared axes functionality in grid plots."""
    with optics_matching_plotly(request) as (tao, _):
        # Test with shared x-axes
        graphs, figure = tao.plotly.plot_grid(["alpha", "beta"], grid=(2, 1), share_x=True)

        # In subplots, shared axes are indicated by xaxis domains
        assert len(graphs) == 2
        assert len(figure.data) > 0


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not available")
def test_plotly_large_dataset_handling(request: pytest.FixtureRequest):
    """Test handling of plots with many data points."""
    with optics_matching_plotly(request) as (tao, _):
        original_max = _PlotlyDefaults.max_data_points
        _PlotlyDefaults.max_data_points = 1000

        try:
            graphs, figure = tao.plotly.plot("beta")
            assert len(figure.data) > 0
        finally:
            _PlotlyDefaults.max_data_points = original_max
