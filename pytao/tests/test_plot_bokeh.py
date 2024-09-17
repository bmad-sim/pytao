import contextlib
import logging
import re
import unittest.mock

import bokeh.events
import bokeh.models
import bokeh.plotting
import pytest
from bokeh.plotting import output_file

from .. import TaoStartup
from ..plotting.bokeh import (
    BokehAppCreator,
    BokehAppState,
    NotebookGraphManager,
    Variable,
    _Defaults,
    initialize_jupyter,
    select_graph_manager_class,
    set_defaults,
)
from ..plotting.plot import FloorPlanGraph
from ..plotting.settings import TaoFloorPlanSettings, TaoGraphSettings
from ..subproc import AnyTao
from .conftest import get_example, test_artifacts

logger = logging.getLogger(__name__)


def annotate_and_save(state: BokehAppState, test_name: str, filename_base: str):
    assert len(state.pairs)
    for pair in state.pairs:
        fig = pair.fig
        graph = pair.bgraph.graph
        fig.title.text = (
            f"{fig.title.text} ({graph.region_name}.{graph.graph_name} of {test_name})"
        )

    fn = test_artifacts / f"{filename_base}.html"
    state.save(fn)
    return fn


def test_bokeh_manager(
    request: pytest.FixtureRequest,
    tao_regression_test: TaoStartup,
):
    name = re.sub(r"[/\\]", "_", request.node.name)
    filename_base = f"bokeh_{name}"
    tao_regression_test.plot = "bokeh"
    with tao_regression_test.run_context(use_subprocess=True) as tao:
        manager = tao.bokeh

        output_file(test_artifacts / f"{filename_base}.html")

        _, app = manager.plot_all()

        annotate_and_save(app.create_state(), request.node.name, filename_base)

        for region in list(manager.regions):
            manager.clear(region)
        assert not any(region for region in manager.regions.values())
        manager.clear()
        assert not manager.regions


def test_bokeh_examples(
    request: pytest.FixtureRequest,
    tao_example: TaoStartup,
):
    example_name = tao_example.metadata["name"]
    name = re.sub(r"[/\\]", "_", request.node.name)
    filename_base = f"bokeh_{name}"

    tao_example.plot = "bokeh"

    with tao_example.run_context(use_subprocess=True) as tao:
        manager = tao.bokeh

        if example_name == "erl":
            tao.cmd("place r11 zphase")

        _, app = manager.plot_all()
        annotate_and_save(app.create_state(), request.node.name, filename_base)


def test_bokeh_floor_plan(request: pytest.FixtureRequest):
    tao_example = get_example("optics_matching")
    name = re.sub(r"[/\\]", "_", request.node.name)
    filename_base = f"bokeh_{name}"

    tao_example.plot = "bokeh"

    with tao_example.run_context(use_subprocess=True) as tao:
        tao.update_plot_shapes("quadrupole", type_label="name", layout=True, floor=True)
        _, app = tao.bokeh.plot("floor_plan")
        annotate_and_save(app.create_state(), request.node.name, filename_base)


@contextlib.contextmanager
def optics_matching(request: pytest.FixtureRequest):
    tao_example = get_example("optics_matching")
    name = re.sub(r"[/\\]", "_", request.node.name)
    tao_example.plot = "bokeh"

    with tao_example.run_context(use_subprocess=True) as tao:
        yield tao, name


def get_ui_from_app(app: BokehAppCreator):
    class Doc:
        def add_root(self, ui):
            self.ui = ui

    doc = Doc()
    app.create_full_app()(doc)
    return doc.ui


def test_bokeh_smoke_create_full_app(request: pytest.FixtureRequest):
    with optics_matching(request) as (tao, _):
        _, app = tao.bokeh.plot_grid(["alpha", "beta"], grid=(2, 1), include_layout=True)

        print(get_ui_from_app(app))


def test_bokeh_update_button(request: pytest.FixtureRequest, caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.ERROR):
        with optics_matching(request) as (tao, _):
            (_alpha, _beta), app = tao.bokeh.plot_grid(
                ["alpha", "beta"], grid=(2, 1), include_layout=True
            )

            state = app.create_state()
            button: bokeh.models.Button = app._add_update_button(state)

            caplog.clear()
            # NOTE: this is internal bokeh API and may break at some point
            # I think that's OK in the context of a test suite
            button._trigger_event(bokeh.events.ButtonClick(button))

    assert not caplog.messages


def test_bokeh_num_points(request: pytest.FixtureRequest, caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.ERROR):
        with optics_matching(request) as (tao, _):
            (_alpha, _beta), app = tao.bokeh.plot_grid(
                ["alpha", "beta"], grid=(2, 1), include_layout=True
            )

            state = app.create_state()
            slider: bokeh.models.Slider = app._add_num_points_slider(state)

            caplog.clear()
            slider.trigger("value", 0, 10)
            slider.trigger("value", 0, 100)
            assert not caplog.messages

            caplog.clear()
            slider.trigger("value", 0, -1)
            assert len(caplog.messages)


def test_bokeh_range_updates(request: pytest.FixtureRequest, caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.ERROR):
        with optics_matching(request) as (tao, _):
            (_alpha, _beta), app = tao.bokeh.plot_grid(
                ["alpha", "beta"], grid=(2, 1), include_layout=True
            )

            state = app.create_state()

            cbs = app._monitor_range_updates(state)

            caplog.clear()
            for cb in cbs:
                cb(bokeh.events.RangesUpdate(model=None, x0=0, x1=10))
            assert not caplog.messages

            caplog.clear()
            for cb in cbs:
                cb(bokeh.events.RangesUpdate(model=None, x0=10, x1=0))
            assert len(caplog.messages)

            caplog.clear()
            for cb in cbs:
                cb(bokeh.events.RangesUpdate(model=None, x0=10, x1=None))
            assert len(caplog.messages)


def get_notebook_graph_manager(tao: AnyTao, monkeypatch: pytest.MonkeyPatch):
    gm = NotebookGraphManager(tao)

    def show(*args, **kwargs):
        print("bokeh plotting show:", args, kwargs)

    monkeypatch.setattr(bokeh.plotting, "show", show)
    return gm


@pytest.mark.parametrize(
    ("grid",),
    [
        pytest.param(True, id="grid"),
        pytest.param(False, id="normal"),
    ],
)
def test_bokeh_notebook_plot_vars(
    request: pytest.FixtureRequest,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    grid: bool,
):
    with caplog.at_level(logging.ERROR):
        with optics_matching(request) as (tao, _):
            gm = get_notebook_graph_manager(tao, monkeypatch)
            if grid:
                _, app = gm.plot_grid(["alpha", "beta"], grid=(2, 1), vars=True)
            else:
                _, app = gm.plot("alpha", vars=True)

            state = app.create_state()

            status_label = bokeh.models.PreText()

            def try_value(var: Variable, value: float) -> None:
                var.ui_update(
                    "",
                    0.0,
                    value,
                    tao=tao,
                    status_label=status_label,
                    pairs=state.pairs,
                )

            def set_value_raise(*args, **kwargs):
                raise RuntimeError("raised")

            for var in app.variables:
                status_label.text = ""
                try_value(var, value=var.value)
                assert not str(status_label.text)

            for var in app.variables:
                status_label.text = ""
                monkeypatch.setattr(var, "set_value", set_value_raise)
                try_value(var, value=var.value)
                assert "raised" in str(status_label.text)


def test_bokeh_floor_orbits(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
):
    with optics_matching(request) as (tao, _):
        gm = get_notebook_graph_manager(tao, monkeypatch)
        (floor_plan,), app = gm.plot(
            "floor_plan",
            settings=TaoGraphSettings(floor_plan=TaoFloorPlanSettings(orbit_scale=1.0)),
        )

        assert isinstance(floor_plan, FloorPlanGraph)

        ui = get_ui_from_app(app)
        assert ui.children[0].children[0].label == "Show orbits"


default_options = sorted(
    set(
        attr
        for attr in dir(_Defaults)
        if not attr.startswith("_") and attr not in {"get_size_for_class"}
    )
)


@pytest.mark.parametrize(("attr",), [pytest.param(attr) for attr in default_options])
def test_bokeh_set_defaults(attr: str):
    value = getattr(_Defaults, attr)
    set_defaults(**{attr: value})
    assert getattr(_Defaults, attr) == value


def test_smoke_select_graph_manager_class():
    select_graph_manager_class()


def test_smoke_init_jupyter(monkeypatch: pytest.MonkeyPatch):
    output_notebook = unittest.mock.Mock()
    monkeypatch.setattr(bokeh.plotting, "output_notebook", output_notebook)
    initialize_jupyter()
    assert output_notebook.called
