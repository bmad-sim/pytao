import logging
import os
import pytest
import re
import matplotlib.pyplot as plt
import pathlib

import rich

from .conftest import new_tao, test_artifacts
from .. import Tao
from ..plotting.plot import GraphInvalidError, get_graphs_in_region, plot_graph, plot_region

logger = logging.getLogger(__name__)

init_files = list(
    pathlib.Path(os.path.expandvars("$ACC_ROOT_DIR/regression_tests/python_test/")).glob(
        "tao.init*"
    )
)


@pytest.fixture(params=init_files, ids=[fn.name for fn in init_files])
def init_filename(
    request: pytest.FixtureRequest,
) -> pathlib.Path:
    return request.param


@pytest.fixture(autouse=True, scope="function")
def _plot_show_to_savefig(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = 0

    def savefig():
        nonlocal index
        test_artifacts.mkdir(exist_ok=True)
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
        plotting=True,
    ) as tao:
        plot_region(tao, "floor_plan")
        plt.show()


def test_plot_floor_layout(tao_cls):
    with new_tao(
        tao_cls,
        "-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_floor_orbit",
        plotting=True,
    ) as tao:
        plot_region(tao, "r33")
        plt.show()

        plot_region(tao, "layout")
        plt.show()

        tao.cmd("place r12 floor_plan")
        _, graph = plot_graph(tao, "r12", "g")
        plt.show()
        rich.print(graph)


def test_plot_data(tao_cls):
    with new_tao(
        tao_cls,
        "-init $ACC_ROOT_DIR/bmad-doc/tao_examples/cesr/tao.init",
        plotting=True,
    ) as tao:
        # init += " -noplot -external_plotting"
        plot_region(tao, "top")
        plt.show()

        tao.cmd("place bottom floor_plan")

        # rich.print(plot_region(tao, "bottom"))
        plot_region(tao, "bottom")
        plt.show()

        tao.cmd("place bottom lat_layout")
        plot_region(tao, "bottom")
        plt.show()


def test_plot_all_visible(init_filename: pathlib.Path):
    with new_tao(Tao, f"-init {init_filename}", plotting=True) as tao:
        visible_plots = [plt for plt in tao.plot_list("r") if plt["visible"]]
        for plot in visible_plots:
            for graph_name in get_graphs_in_region(tao, plot["region"]):
                try:
                    _ax, _graph = plot_graph(tao, plot["region"], graph_name)
                except GraphInvalidError as ex:
                    # tao-reported; it's probably not our fault
                    logger.warning(f"Invalid graph error: {ex}")
                    continue

                plt.show()  # TODO uncomment
