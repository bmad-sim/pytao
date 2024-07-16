import logging
import pytest
import re
import pathlib

from ..plotting.bokeh import BokehGraphManager

from .conftest import new_tao, test_artifacts
from .. import Tao

logger = logging.getLogger(__name__)


def test_bokeh_manager(request: pytest.FixtureRequest, init_filename: pathlib.Path):
    from bokeh.plotting import output_file, save, column

    if init_filename.name in {"tao.init_wall", "tao.init_photon"}:
        pytest.skip(reason="bmad crash on 20240715.0 (TODO)")

    name = re.sub(r"[/\\]", "_", request.node.name)
    filename_base = f"bokeh_{name}"

    # Floor orbit tries to set a plot setting on initialization which doesn't work
    # with external plotting.  Enable plotting for it as a workaround.
    external_plotting = init_filename.name != "tao.init_floor_orbit"
    with new_tao(Tao, f"-init {init_filename}", external_plotting=external_plotting) as tao:
        manager = BokehGraphManager(tao)
        if external_plotting:
            assert len(manager.place_all_requested())

        output_file(test_artifacts / f"{filename_base}.html")
        figures = []
        for region in manager.regions:
            for name, graph in manager.plot_region(region).items():
                figures.append(graph.create_figure())
        save(column(figures, sizing_mode="scale_width"))

        for region in list(manager.regions):
            manager.clear(region)
        assert not manager.regions
        manager.clear_all()
        assert not manager.regions
