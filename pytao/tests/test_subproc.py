from __future__ import annotations

import time
from collections.abc import Callable, Generator
from typing import Any

import numpy as np
import pytest

from .. import SubprocessTao
from ..errors import (
    TaoCommandError,
    capture_messages_from_functions,
    filter_tao_messages_context,
)
from ..subproc import SupportedKwarg, TaoDisconnectedError
from ..tao import Tao


def test_crash_and_recovery() -> None:
    init = "-init $ACC_ROOT_DIR/regression_tests/pipe_test/csr_beam_tracking/tao.init -noplot"
    with SubprocessTao(init=init) as tao:
        # tao.init("-init regression_tests/pipe_test/tao.init_plot_line -external_plotting")
        bunch1 = tao.bunch1(ele_id="end", coordinate="x", which="model", ix_bunch="1")

        assert tao._subproc_pipe_ is not None

        with pytest.raises(TaoDisconnectedError):
            # Close the pipe earlier than expected
            tao._subproc_pipe_.send_receive("quit", "", propagate_exceptions=True)
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
def subproc_tao() -> Generator[SubprocessTao]:
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
    kwargs: dict[str, SupportedKwarg],
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


def tao_custom_command_sleep(tao: Tao, delay: float):
    assert isinstance(tao, Tao)
    print(f"Sleeping for {delay}")
    time.sleep(delay)
    print("Done")
    return delay


def test_custom_command_timeout(subproc_tao: SubprocessTao) -> None:
    with SubprocessTao(
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/csr_beam_tracking/tao.init",
        noplot=True,
    ) as tao:
        with pytest.raises(TimeoutError):
            with tao.timeout(0.1):
                res = tao.subprocess_call(tao_custom_command_sleep, delay=10.0)
                print("subproc result was", res)


def test_custom_command_timeout_success(subproc_tao: SubprocessTao) -> None:
    with subproc_tao.timeout(10.0):
        res = subproc_tao.subprocess_call(tao_custom_command, value=1)
        assert res == 2


def _tao_cmd_raises(tao: Tao, cmd: str):
    """Run a command with raises=True inside the subprocess."""
    return tao.cmd(cmd, raises=True)


def test_error_filter_context_cmd(subproc_tao: SubprocessTao) -> None:
    bad_cmd = "sho ele 100000"

    with pytest.raises(TaoCommandError):
        subproc_tao.cmd(bad_cmd)

    res = subproc_tao.cmd(bad_cmd, raises=False)
    assert not res
    print(subproc_tao._last_output)
    func_names = [msg.function for msg in subproc_tao.last_messages]

    assert "lat_ele1_locator" in func_names

    with filter_tao_messages_context(functions=func_names):
        result = subproc_tao.cmd(bad_cmd, raises=True)
        assert not result
        # assert not subproc_tao.last_output


def test_error_filter_context_cmd_real(subproc_tao: SubprocessTao) -> None:
    arr = subproc_tao.lat_list("*", "ele.s")
    assert isinstance(arr, np.ndarray)
    assert len(arr) > 0

    with pytest.raises(TaoCommandError):
        subproc_tao.evaluate("1/0", raises=True)

    func_names = [msg.function for msg in subproc_tao.last_messages]

    str_list: list[str] = subproc_tao.lat_list("*", "ele.s", flags="-track_only")
    assert isinstance(str_list, list)

    arr = subproc_tao.lat_list("*", "ele.s")
    assert isinstance(arr, np.ndarray)

    assert len(str_list) == len(arr)

    with filter_tao_messages_context(functions=func_names):
        # no 'with raises' block here, as we're filtering it
        output = subproc_tao.evaluate("1/0", raises=True)
        # output still not really useful, but that's OK I suppose.
        assert isinstance(output, np.ndarray)
        assert output.shape == (0,)
        assert "Divide by zero" in subproc_tao.last_messages[0].message
        assert "Invalid expression" in subproc_tao.last_messages[1].message
