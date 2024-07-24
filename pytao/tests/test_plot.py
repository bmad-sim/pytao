import contextlib
import logging
from typing_extensions import Literal
import pytest
import re
import matplotlib.pyplot as plt
import pathlib

from typing import Type, Union

from pytao.interface_commands import AnyPath
from pytao.subproc import SubprocessTao

from .conftest import test_artifacts
from .. import Tao

logger = logging.getLogger(__name__)


AnyTao = Union[Tao, SubprocessTao]
BackendName = Literal["mpl", "bokeh"]


@pytest.fixture(params=["bokeh", "mpl"])
def plot_backend(
    request: pytest.FixtureRequest,
) -> BackendName:
    return request.param


@contextlib.contextmanager
def new_tao(
    tao_cls: Type[AnyTao],
    init_file: AnyPath,
    plot: Literal["mpl", "bokeh"],
    **kwargs,
):
    tao = tao_cls(init_file=init_file, plot=plot, **kwargs)
    yield tao
    if isinstance(tao, SubprocessTao):
        print("Closing tao subprocess")
        tao.close_subprocess()

    if plot == "mpl":
        plt.show()


@pytest.fixture
def nostartup(init_filename: pathlib.Path) -> bool:
    return init_filename.name == "tao.init_floor_orbit"


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


def test_plot_floor_plan(tao_cls: Type[AnyTao], plot_backend: BackendName):
    with new_tao(
        tao_cls,
        "$ACC_ROOT_DIR/regression_tests/python_test/tao.init_wall",
        plot=plot_backend,
    ) as tao:
        tao.plot("floor_plan")


def test_plot_floor_layout(tao_cls: Type[AnyTao], plot_backend: BackendName):
    with new_tao(
        tao_cls,
        "$ACC_ROOT_DIR/regression_tests/python_test/tao.init_floor_orbit",
        plot=plot_backend,
        nostartup=True,
    ) as tao:
        tao.plot("alpha")
        tao.plot("beta")
        tao.plot("lat_layout")

        tao.cmd("set plot_page%floor_plan_shape_scale = 0.01")
        tao.plot("floor_plan", region_name="r33")
        tao.cmd("set graph r33 floor_plan%orbit_scale = 1")
        tao.plot("floor_plan", region_name="r33", ylim=(-0.3, 0.1))


def test_plot_data(tao_cls: Type[AnyTao], plot_backend: BackendName):
    with new_tao(
        tao_cls,
        "$ACC_ROOT_DIR/bmad-doc/tao_examples/cesr/tao.init",
        plot=plot_backend,
    ) as tao:
        tao.plot_manager.place_all()
        tao.plot_manager.plot_regions(list(tao.plot_manager.regions))
        tao.plot("floor_plan")
        tao.plot("lat_layout")


def test_plot_all_requested(
    init_filename: pathlib.Path,
    tao_cls: Type[AnyTao],
    plot_backend: BackendName,
    nostartup: bool,
):
    with new_tao(
        tao_cls,
        init_filename,
        plot=plot_backend,
        nostartup=nostartup,
    ) as tao:
        tao.plot_manager.place_all()
        tao.plot_manager.plot_regions(list(tao.plot_manager.regions))


def test_plot_manager(
    init_filename: pathlib.Path,
    nostartup: bool,
    plot_backend: BackendName,
):
    with new_tao(Tao, init_filename, plot=plot_backend, nostartup=nostartup) as tao:
        manager = tao.plot_manager
        assert len(manager.place_all())
        manager.plot_regions(list(manager.regions))

        for region in list(manager.regions):
            manager.clear(region)
        assert not any(region for region in manager.regions.values())
        manager.clear()
        assert not manager.regions
