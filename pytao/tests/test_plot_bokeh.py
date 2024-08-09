import logging
import re

import pytest
from bokeh.plotting import output_file

from pytao.plotting.bokeh import BokehAppState

from .. import TaoStartup
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
        _, app = tao.bokeh.plot("floor_plan")
        annotate_and_save(app.create_state(), request.node.name, filename_base)
