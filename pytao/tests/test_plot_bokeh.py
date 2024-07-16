import logging
import pathlib
import re
from typing import List

import pytest
from bokeh.plotting import column, output_file, save

from .. import Tao
from ..plotting.bokeh import (
    BGraphAndFigure,
    BokehGraphManager,
    share_common_x_axes,
)
from .conftest import new_tao, test_artifacts

logger = logging.getLogger(__name__)


def test_bokeh_manager(request: pytest.FixtureRequest, init_filename: pathlib.Path):
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
        items: List[BGraphAndFigure] = []

        for region_name, region in manager.plot_all().items():
            for name, bgraph in region.items():
                fig = bgraph.create_figure()
                fig.title.text = (
                    f"{fig.title.text} ({region_name}.{name} of {request.node.name})"
                )
                items.append(BGraphAndFigure(bgraph=bgraph, fig=fig))

        share_common_x_axes(items)
        save(column([item.fig for item in items], sizing_mode="fixed"))

        for region in list(manager.regions):
            manager.clear(region)
        assert not manager.regions
        manager.clear_all()
        assert not manager.regions
