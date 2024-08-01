import pytest

from .. import TaoStartup


def test_examples_can_init(tao_example: TaoStartup) -> None:
    assert tao_example.can_initialize


def test_regression_tests_can_init(tao_regression_test: TaoStartup) -> None:
    assert tao_regression_test.can_initialize


@pytest.mark.parametrize(
    ("startup"),
    [
        pytest.param(TaoStartup("-i foo")),
        pytest.param(TaoStartup("-init foo")),
        pytest.param(TaoStartup("-init_file foo")),
        pytest.param(TaoStartup("-la foo")),
        pytest.param(TaoStartup("-lat foo")),
        pytest.param(TaoStartup("-lattice_file foo")),
    ],
)
def test_can_init(startup: TaoStartup) -> None:
    assert startup.can_initialize


@pytest.mark.parametrize(
    ("startup"),
    [
        pytest.param(TaoStartup()),
        pytest.param(TaoStartup(external_plotting=True)),
        pytest.param(TaoStartup(var_file="no")),
    ],
)
def test_no_init(startup: TaoStartup) -> None:
    assert not startup.can_initialize


def test_plotting() -> None:
    assert (
        TaoStartup(plot="mpl", external_plotting=True, noplot=True).tao_init
        == "-external_plotting -noplot"
    )


def test_init_override() -> None:
    assert TaoStartup("-init_file foo", init_file="test").tao_init == "-init_file foo"
