import matplotlib
import contextlib
import logging
import os
import pathlib

import pytest

from .. import SubprocessTao, Tao

matplotlib.use("Agg")

test_root = pathlib.Path(__file__).parent.resolve()
test_artifacts = test_root / "artifacts"


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


@pytest.fixture(
    params=[Tao, SubprocessTao],
    ids=["Tao", "SubprocessTao"],
)
def tao_cls(request: pytest.FixtureRequest):
    return request.param


@contextlib.contextmanager
def new_tao(tao_cls, init, plot: bool = False, external_plotting: bool = True):
    init = os.path.expandvars(init)
    if external_plotting:
        init = " ".join((init, "-external_plotting"))
    if not plot:
        init = " ".join((init, "-noplot"))
    tao = tao_cls(init)
    yield tao
    if hasattr(tao, "close_subprocess"):
        print("Closing tao subprocess")
        tao.close_subprocess()
