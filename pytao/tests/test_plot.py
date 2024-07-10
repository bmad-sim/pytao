from .conftest import new_tao
from ..plotting.plot import plot_region


def test_plot_floor_plan(tao_cls):
    with new_tao(
        tao_cls,
        "-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_wall",
        plotting=True,
    ) as tao:
        # floor_plan.g
        plot_region(tao, "floor_plan")


def test_plot_floor_layout(tao_cls):
    with new_tao(
        tao_cls,
        "-init $ACC_ROOT_DIR/regression_tests/python_test/tao.init_floor_orbit",
        plotting=True,
    ) as tao:
        plot_region(tao, "r33")
        plot_region(tao, "layout")
        # tao.cmd("place layout floor_plan")
        # plot_region(tao, "bottom")


def test_plot_data(tao_cls):
    with new_tao(
        tao_cls,
        "-init $ACC_ROOT_DIR/bmad-doc/tao_examples/cesr/tao.init",
        plotting=True,
    ) as tao:
        # init += " -noplot -external_plotting"
        plot_region(tao, "top")
        tao.cmd("place bottom floor_plan")
        plot_region(tao, "bottom")
        tao.cmd("place bottom lat_layout")
        plot_region(tao, "bottom")
