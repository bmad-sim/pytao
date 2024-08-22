import contextlib
import logging
import os
import pathlib
from typing import Generator, Type, TypeVar

import matplotlib
import pytest
from typing_extensions import Literal

from .. import SubprocessTao, Tao, TaoStartup

matplotlib.use("Agg")

test_root = pathlib.Path(__file__).parent.resolve()
packaged_examples_root = test_root / "input_files"
test_artifacts = test_root / "artifacts"

regression_test_root = pathlib.Path("$ACC_ROOT_DIR/regression_tests/pipe_test/")
example_root = pathlib.Path("$ACC_ROOT_DIR/bmad-doc/tao_examples")
init_files = list(pathlib.Path(os.path.expandvars(regression_test_root)).glob("tao.init*"))
example_init_files = list(
    path for path in pathlib.Path(os.path.expandvars(example_root)).glob("*/tao.init")
)


@pytest.fixture
def rootdir():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def config_file(rootdir):
    return open(f"{rootdir}/test_files/iris_config.yml", "r")


@pytest.fixture(autouse=True)
def ensure_count():
    from ..util import parsers

    parsers.Settings.ensure_count = True


@contextlib.contextmanager
def ensure_successful_parsing(caplog):
    yield
    errors = [
        record for record in caplog.get_records("call") if record.levelno == logging.ERROR
    ]
    for error in errors:
        if "Failed to parse string data" in error.message:
            pytest.fail(error.message)


def get_packaged_example(name: str) -> TaoStartup:
    """PyTao packaged bmad input data."""
    init_file = packaged_examples_root / name / "tao.init"
    startup = TaoStartup(
        init_file=init_file,
        # nostartup=nostartup,
        metadata={"name": name},
    )
    print(f"Packaged example {name}: {startup.tao_init}")
    return startup


def get_example(name: str) -> TaoStartup:
    """Bmad-doc example startup data."""
    init_file = example_root / name / "tao.init"
    if name == "multi_turn_orbit":
        pytest.skip(
            "Multi-turn orbit example fails with: CANNOT SCALE GRAPH multi_turn.x SINCE NO DATA IS WITHIN THE GRAPH X-AXIS RANGE. "
        )
    if name == "custom_tao_with_measured_data":
        # Looks to require some additional compilation and such
        pytest.skip(
            "'custom tao with measured data' example fails with PARSER ERROR DETECTED FOR UNIVERSE: 1"
        )
    if name == "x_axis_param_plot":
        pytest.skip("'x_axis_param_plot' example fails saying no data is in range")

    nostartup = name in {
        # "multi_turn_orbit",
        "custom_tao_with_measured_data",
        "x_axis_param_plot",
    }
    startup = TaoStartup(
        init_file=init_file,
        nostartup=nostartup,
        metadata={"name": name},
    )
    print(f"Example {name}: {startup.tao_init}")
    return startup


def get_regression_test(name: str) -> TaoStartup:
    """Bmad-doc 'pipe' interface command regression test files."""
    init_file = regression_test_root / name
    nostartup = init_file.name == "tao.init_floor_orbit"
    return TaoStartup(
        init_file=init_file,
        nostartup=nostartup,
        metadata={"name": init_file.name},
    )


@pytest.fixture(params=init_files, ids=[f"regression_tests-{fn.name}" for fn in init_files])
def init_filename(
    request: pytest.FixtureRequest,
) -> pathlib.Path:
    return request.param


@pytest.fixture(params=init_files, ids=[f"regression_tests-{fn.name}" for fn in init_files])
def tao_regression_test(
    request: pytest.FixtureRequest,
) -> TaoStartup:
    return get_regression_test(request.param.name)


@pytest.fixture(params=example_init_files, ids=[fn.parts[-2] for fn in example_init_files])
def tao_example(
    request: pytest.FixtureRequest,
) -> TaoStartup:
    return get_example(request.param.parts[-2])


@pytest.fixture(
    params=[Tao, SubprocessTao],
    ids=["Tao", "SubprocessTao"],
)
def tao_cls(request: pytest.FixtureRequest):
    return request.param


T = TypeVar("T", bound=Tao)


@contextlib.contextmanager
def new_tao(
    tao_cls: Type[T],
    init: str = "",
    plot: bool = False,
    external_plotting: bool = True,
    **kwargs,
) -> Generator[T, None, None]:
    # init = os.path.expandvars(init)
    if external_plotting:
        init = " ".join((init, "-external_plotting"))
    if not plot:
        init = " ".join((init, "-noplot"))
    tao = tao_cls(init, **kwargs)
    yield tao
    if hasattr(tao, "close_subprocess"):
        print("Closing tao subprocess")
        tao.close_subprocess()


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
