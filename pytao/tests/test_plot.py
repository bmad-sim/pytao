import logging
import pytest
import re
import matplotlib.pyplot as plt
import pathlib

from .conftest import new_tao, test_artifacts
from .. import Tao
from ..plotting.plot import (
    MatplotlibGraphManager,
    make_graph,
    plot_all_requested,
    plot_all_visible,
    plot_graph,
    plot_region,
)

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True, scope="function")
def _plot_show_to_savefig(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = 0

    def savefig():
        nonlocal index
        test_artifacts.mkdir(exist_ok=True)
        for fignum in plt.get_fignums():
            plt.figure(fignum)
            name = re.sub(r"[/\\]", "_", request.node.name)
            filename = test_artifacts / f"{name}_{index}.png"
            print(f"Saving figure (_plot_show_to_savefig fixture) to {filename}")
            plt.savefig(filename)
            index += 1
        plt.close("all")

    monkeypatch.setattr(plt, "show", savefig)


def test_plot_floor_plan(tao_cls):
    with new_tao(
        tao_cls,
        "-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_wall",
    ) as tao:
        plot_all_requested(tao)
        # plot_region(tao, "floor_plan")
        # plt.show()


def test_plot_floor_layout(tao_cls):
    with new_tao(
        tao_cls,
        "-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_floor_orbit",
        # Floor orbit tries to set a plot setting on initialization which doesn't work
        # with external plotting.  Enable plotting for it as a workaround.
        external_plotting=False,
    ) as tao:
        # plot_all_requested(tao)
        plot_all_visible(tao)
        plot_region(tao, "r33")
        plt.show()

        plot_region(tao, "layout")
        plt.show()

        tao.cmd("place -no_buffer r12 floor_plan")
        graph = make_graph(tao, "r12", "g")
        plot_graph(tao, graph)
        plt.show()


def test_plot_data(tao_cls):
    with new_tao(
        tao_cls,
        "-init $ACC_ROOT_DIR/bmad-doc/tao_examples/cesr/tao.init",
    ) as tao:
        plot_all_requested(tao)
        plt.show()

        tao.cmd("place -no_buffer bottom floor_plan")
        # rich.print(plot_region(tao, "bottom"))
        plot_region(tao, "bottom")
        plt.show()

        tao.cmd("place -no_buffer bottom lat_layout")
        plot_region(tao, "bottom")
        plt.show()


def test_plot_all_requested(init_filename: pathlib.Path):
    # Floor orbit tries to set a plot setting on initialization which doesn't work
    # with external plotting.  Enable plotting for it as a workaround.
    external_plotting = init_filename.name != "tao.init_floor_orbit"
    with new_tao(Tao, f"-init {init_filename}", external_plotting=external_plotting) as tao:
        plot_all_requested(tao)
        plt.show()


def test_plot_manager(init_filename: pathlib.Path):
    # Floor orbit tries to set a plot setting on initialization which doesn't work
    # with external plotting.  Enable plotting for it as a workaround.
    external_plotting = init_filename.name != "tao.init_floor_orbit"
    with new_tao(Tao, f"-init {init_filename}", external_plotting=external_plotting) as tao:
        manager = MatplotlibGraphManager(tao)
        if external_plotting:
            assert len(manager.place())
        for region in manager.regions:
            manager.plot(region)
        plt.show()

        for region in list(manager.regions):
            manager.clear(region)
        assert not manager.regions
        manager.clear_all()
        assert not manager.regions
