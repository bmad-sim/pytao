import logging
import pathlib
import re

import pytest
from bokeh.plotting import column, output_file, save

from .. import Tao
from ..plotting.bokeh import BGraphAndFigure, share_common_x_axes
from .conftest import new_tao, test_artifacts

logger = logging.getLogger(__name__)


def test_bokeh_manager(request: pytest.FixtureRequest, init_filename: pathlib.Path):
    name = re.sub(r"[/\\]", "_", request.node.name)
    filename_base = f"bokeh_{name}"

    # Floor orbit tries to set a plot setting on initialization which doesn't work
    # with external plotting.
    nostartup = init_filename.name == "tao.init_floor_orbit"
    with new_tao(Tao, f"-init {init_filename}", nostartup=nostartup) as tao:
        manager = tao.bokeh
        assert len(manager.place_all())

        output_file(test_artifacts / f"{filename_base}.html")

        bgraphs = sum(manager.plot_regions(list(manager.regions)), [])
        items = [
            BGraphAndFigure(bgraph=bgraph, fig=bgraph.create_figure()) for bgraph in bgraphs
        ]
        for item in items:
            fig = item.fig
            graph = item.bgraph.graph
            fig.title.text = f"{fig.title.text} ({graph.region_name}.{graph.graph_name} of {request.node.name})"

        share_common_x_axes(items)
        save(column([item.fig for item in items], sizing_mode="fixed"))

        for region in list(manager.regions):
            manager.clear(region)
        assert not any(region for region in manager.regions.values())
        manager.clear()
        assert not manager.regions
