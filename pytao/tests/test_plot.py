import logging
import re
from typing import Union

import matplotlib.pyplot as plt
import pytest

from .. import SubprocessTao, Tao, TaoStartup
from ..plotting import mpl
from ..plotting.curves import TaoCurveSettings
from .conftest import (
    BackendName,
    get_example,
    get_packaged_example,
    get_regression_test,
    test_artifacts,
)

logger = logging.getLogger(__name__)


AnyTao = Union[Tao, SubprocessTao]


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
    startup = get_regression_test("tao.init_wall")
    startup.plot = plot_backend
    with startup.run_context(use_subprocess=use_subprocess) as tao:
        tao.plot("floor_plan")


def test_plot_all_interface(plot_backend: BackendName):
    startup = get_regression_test("tao.init_floor_orbit")
    startup.plot = plot_backend
    with startup.run_context(use_subprocess=False) as tao:
        tao.plot()


@pytest.mark.parametrize(
    ("include_layout",),
    [
        pytest.param(True, id="include_layout"),
        pytest.param(False, id="no_layout"),
    ],
)
def test_plot_single_interface(plot_backend: BackendName, include_layout: bool):
    startup = get_regression_test("tao.init_floor_orbit")
    startup.plot = plot_backend
    with startup.run_context(use_subprocess=False) as tao:
        tao.plot("alpha", include_layout=include_layout)


def test_plot_grid_interface(plot_backend: BackendName):
    startup = get_regression_test("tao.init_floor_orbit")
    startup.plot = plot_backend
    with startup.run_context(use_subprocess=False) as tao:
        tao.plot(["alpha", "beta"])


def test_plot_floor_layout(use_subprocess: bool, plot_backend: BackendName):
    startup = get_regression_test("tao.init_floor_orbit")
    startup.plot = plot_backend
    startup.nostartup = True
    with startup.run_context(use_subprocess=use_subprocess) as tao:
        tao.plot("alpha")
        tao.plot("beta")
        tao.plot("lat_layout")

        tao.cmd("set plot_page%floor_plan_shape_scale = 0.01")
        tao.plot("floor_plan", region_name="r33")
        tao.cmd("set graph r33 floor_plan%orbit_scale = 1")
        tao.plot("floor_plan", region_name="r33", ylim=(-0.3, 0.1))


def test_plot_data(use_subprocess: bool, plot_backend: BackendName):
    startup = get_example("cesr")
    startup.plot = plot_backend
    with startup.run_context(use_subprocess=use_subprocess) as tao:
        tao.plot_manager.plot_all()
        tao.plot("floor_plan")
        tao.plot("lat_layout")


def test_plot_all_requested_regression_tests(
    tao_regression_test: TaoStartup,
    plot_backend: BackendName,
    use_subprocess: bool,
):
    tao_regression_test.plot = plot_backend
    with tao_regression_test.run_context(use_subprocess=use_subprocess) as tao:
        tao.plot_manager.plot_all()


def test_plot_all_requested_examples_mpl(tao_example: TaoStartup):
    tao_example.plot = "mpl"
    example_name = tao_example.metadata["name"]
    with tao_example.run_context(use_subprocess=True) as tao:
        if example_name == "erl":
            tao.cmd("place r11 zphase")
        tao.plot_manager.plot_all()


def test_plot_manager(
    tao_regression_test: TaoStartup,
    plot_backend: BackendName,
    use_subprocess: bool,
):
    tao_regression_test.plot = plot_backend
    with tao_regression_test.run_context(use_subprocess=use_subprocess) as tao:
        manager = tao.plot_manager
        manager.plot_all()

        for region in list(manager.regions):
            manager.clear(region)
        assert not any(region for region in manager.regions.values())
        manager.clear()
        assert not manager.regions


def test_plot_curve(plot_backend: BackendName):
    example = get_example("erl")
    example.plot = plot_backend
    with example.run_context(use_subprocess=True) as tao:
        manager = tao.plot_manager
        manager.plot(
            "zphase",
            curves={
                1: TaoCurveSettings(
                    ele_ref_name=r"linac.beg\1",
                    draw_line=True,
                    draw_symbols=True,
                    draw_symbol_index=True,
                ),
            },
            save=test_artifacts / f"test_plot_curve-{plot_backend}",
        )


def test_plot_grid(plot_backend: BackendName):
    example = get_example("erl")
    example.plot = plot_backend
    with example.run_context(use_subprocess=True) as tao:
        manager = tao.plot_manager
        manager.plot_grid(
            templates=["zphase", "zphase", "zphase", "zphase2"],
            grid=(2, 2),
            curves=[
                {1: TaoCurveSettings(ele_ref_name=r"linac.beg\1")},
                {1: TaoCurveSettings(ele_ref_name=r"linac.end\1")},
                {1: TaoCurveSettings(ele_ref_name=r"linac.beg\2")},
                {1: TaoCurveSettings(ele_ref_name=r"linac.end\2")},
            ],
            share_x=False,
            save=test_artifacts / f"test_plot_grid-{plot_backend}",
        )


def test_plot_grid_with_layout(plot_backend: BackendName):
    example = get_example("erl")
    example.plot = plot_backend
    with example.run_context(use_subprocess=True) as tao:
        manager = tao.plot_manager
        manager.plot_grid(
            templates=["zphase", "zphase", "zphase", "zphase2"],
            grid=(3, 2),
            include_layout=True,
            curves=[
                {1: TaoCurveSettings(ele_ref_name=r"linac.beg\1")},
                {1: TaoCurveSettings(ele_ref_name=r"linac.end\1")},
                {1: TaoCurveSettings(ele_ref_name=r"linac.beg\2")},
                {1: TaoCurveSettings(ele_ref_name=r"linac.end\2")},
            ],
            # figsize=(10, 10),
            share_x=False,
            save=test_artifacts / f"test_plot_grid_with_layout-{plot_backend}",
        )


def test_plot_update(plot_backend: BackendName):
    example = get_packaged_example("optics_matching_tweaked")
    example.plot = plot_backend
    with example.run_context(use_subprocess=True) as tao:
        manager = tao.plot_manager
        (graph,), *_ = manager.plot("alpha1", include_layout=False)
        updated = graph.update(manager)
        assert graph == updated


default_options = sorted(
    set(
        attr
        for attr in dir(mpl._Defaults)
        if not attr.startswith("_") and attr not in {"get_size_for_class"}
    )
)


@pytest.mark.parametrize(("attr",), [pytest.param(attr) for attr in default_options])
def test_mpl_set_defaults(attr: str):
    value = getattr(mpl._Defaults, attr)
    mpl.set_defaults(**{attr: value})
    assert getattr(mpl._Defaults, attr) == value
