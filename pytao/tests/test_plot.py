import logging
import re
from typing import Union

import matplotlib.pyplot as plt
import pytest
from typing_extensions import Literal

from .. import SubprocessTao, Tao, TaoStartup
from .conftest import test_artifacts

logger = logging.getLogger(__name__)


AnyTao = Union[Tao, SubprocessTao]
BackendName = Literal["mpl", "bokeh"]


@pytest.fixture(params=["bokeh", "mpl"])
def plot_backend(
    request: pytest.FixtureRequest,
) -> BackendName:
    return request.param


@pytest.fixture(params=[False, True], ids=["Tao", "SubprocessTao"])
def use_subprocess(
    request: pytest.FixtureRequest,
) -> bool:
    return request.param


@pytest.fixture(autouse=True, scope="function")
def _plot_show_to_savefig(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    plot_backend: BackendName,
):
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
    yield
    if plot_backend == "mpl":
        plt.show()


def test_plot_floor_plan(use_subprocess: bool, plot_backend: BackendName):
    with TaoStartup(
        init_file="$ACC_ROOT_DIR/regression_tests/python_test/tao.init_wall",
        plot=plot_backend,
    ).run_context(use_subprocess=use_subprocess) as tao:
        tao.plot("floor_plan")


def test_plot_floor_layout(use_subprocess: bool, plot_backend: BackendName):
    with TaoStartup(
        init_file="$ACC_ROOT_DIR/regression_tests/python_test/tao.init_floor_orbit",
        plot=plot_backend,
        nostartup=True,
    ).run_context(use_subprocess=use_subprocess) as tao:
        tao.plot("alpha")
        tao.plot("beta")
        tao.plot("lat_layout")

        tao.cmd("set plot_page%floor_plan_shape_scale = 0.01")
        tao.plot("floor_plan", region_name="r33")
        tao.cmd("set graph r33 floor_plan%orbit_scale = 1")
        tao.plot("floor_plan", region_name="r33", ylim=(-0.3, 0.1))


def test_plot_data(use_subprocess: bool, plot_backend: BackendName):
    with TaoStartup(
        init_file="$ACC_ROOT_DIR/bmad-doc/tao_examples/cesr/tao.init",
        plot=plot_backend,
    ).run_context(use_subprocess=use_subprocess) as tao:
        tao.plot_manager.place_all()
        tao.plot_manager.plot_regions(list(tao.plot_manager.regions))
        tao.plot("floor_plan")
        tao.plot("lat_layout")


def test_plot_all_requested(
    tao_regression_test: TaoStartup,
    plot_backend: BackendName,
    use_subprocess: bool,
):
    tao_regression_test.plot = plot_backend
    with tao_regression_test.run_context(use_subprocess=use_subprocess) as tao:
        tao.plot_manager.place_all()
        tao.plot_manager.plot_regions(list(tao.plot_manager.regions))


def test_plot_manager(
    tao_regression_test: TaoStartup,
    plot_backend: BackendName,
    use_subprocess: bool,
):
    tao_regression_test.plot = plot_backend
    with tao_regression_test.run_context(use_subprocess=use_subprocess) as tao:
        manager = tao.plot_manager
        assert len(manager.place_all())
        manager.plot_regions(list(manager.regions))

        for region in list(manager.regions):
            manager.clear(region)
        assert not any(region for region in manager.regions.values())
        manager.clear()
        assert not manager.regions
