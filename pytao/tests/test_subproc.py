from __future__ import annotations

import time
from typing import Any, Callable, Dict, Generator

import numpy as np
import pytest

from .. import SubprocessTao
from ..interface_commands import Tao
from ..subproc import SupportedKwarg, TaoDisconnectedError
from ..tao_ctypes.util import TaoCommandError


def test_crash_and_recovery() -> None:
    init = "-init $ACC_ROOT_DIR/regression_tests/pipe_test/csr_beam_tracking/tao.init -noplot"
    with SubprocessTao(init=init) as tao:
        # tao.init("-init regression_tests/pipe_test/tao.init_plot_line -external_plotting")
        bunch1 = tao.bunch1(ele_id="end", coordinate="x", which="model", ix_bunch="1")
        print("bunch1=", bunch1)

        assert tao._subproc_pipe_ is not None

        with pytest.raises(TaoDisconnectedError):
            # Close the pipe earlier than expected
            tao._subproc_pipe_.send_receive("quit", "", raises=True)
        time.sleep(0.5)

        print("Re-initializing:")
        tao.init(init)
        retry = tao.bunch1(ele_id="end", coordinate="x", which="model", ix_bunch="1")
        assert np.allclose(bunch1, retry)


def tao_custom_command(tao: Tao, value: Any):
    assert isinstance(tao, Tao)
    if isinstance(value, bool):
        return 23
    if isinstance(value, (int, np.ndarray)):
        return value + 1
    return value


@pytest.fixture(scope="module")
def subproc_tao() -> Generator[SubprocessTao, None, None]:
    with SubprocessTao(
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/csr_beam_tracking/tao.init",
        noplot=True,
    ) as tao:
        yield tao


@pytest.mark.parametrize(
    ("func", "kwargs", "expected_result"),
    [
        pytest.param(
            tao_custom_command,
            {"value": 3},
            4,
            id="int",
        ),
        pytest.param(
            tao_custom_command,
            {"value": np.zeros(3)},
            np.ones(3),
            id="ndarray",
        ),
        pytest.param(
            tao_custom_command,
            {"value": {"a": np.zeros(3), "b": {"c": np.ones(3)}}},
            {"a": np.zeros(3), "b": {"c": np.ones(3)}},
            id="nested-ndarray",
        ),
        pytest.param(
            tao_custom_command,
            {"value": "value"},
            "value",
            id="str",
        ),
        pytest.param(
            tao_custom_command,
            {"value": True},
            23,
            id="bool",
        ),
        pytest.param(
            tao_custom_command,
            {"value": 1.0},
            1.0,
            id="float",
        ),
        pytest.param(
            tao_custom_command,
            {"value": [1.0, {"a": np.zeros(3)}]},
            [1.0, {"a": np.zeros(3)}],
            id="list",
        ),
        pytest.param(
            tao_custom_command,
            {"value": {1.0, 2.0, 3.0}},
            {1.0, 2.0, 3.0},
            id="set",
        ),
        pytest.param(
            tao_custom_command,
            {"value": (1.0, {"a": np.zeros(3)})},
            (1.0, {"a": np.zeros(3)}),
            id="tuple",
        ),
    ],
)
def test_custom_command(
    subproc_tao: SubprocessTao,
    func: Callable,
    kwargs: Dict[str, SupportedKwarg],
    expected_result: Any,
) -> None:
    res = subproc_tao.subprocess_call(func, **kwargs)
    if isinstance(expected_result, np.ndarray):
        assert np.all(res == expected_result)
    elif "array(" in str(expected_result):
        # Lazily check the result's string equality
        assert str(res) == str(expected_result)
    else:
        assert res == expected_result


def failure_func(tao: Tao, **kwargs):
    raise ValueError(f"test got kwargs: {kwargs}")


def test_custom_command_exception(subproc_tao: SubprocessTao) -> None:
    with pytest.raises(TaoCommandError) as ex:
        subproc_tao.subprocess_call(failure_func, a=3)
    assert "ValueError: test got kwargs: {'a': 3}" in str(ex.value)
