from __future__ import annotations

import dataclasses
import os
import pathlib
import pickle
import selectors
import signal
import stat
import sys
import time
from collections.abc import Generator
from typing import Any

import numpy as np
import pytest

from .. import SubprocessTao, Tao, TaoCommandError, TaoForkError
from ..fork import decode_fork_payloads, fork_map_payloads, resolve_max_workers
from ..subproc import _pickle_for_subprocess
from .conftest import new_tao

pytestmark = pytest.mark.filterwarnings(
    "ignore:This process.*multi-threaded:DeprecationWarning"
)

OPTICS_INIT = "-init $ACC_ROOT_DIR/bmad-doc/tao_examples/optics_matching/tao.init"
CSR_INIT = "-init $ACC_ROOT_DIR/regression_tests/pipe_test/csr_beam_tracking/tao.init"


@pytest.fixture
def tao() -> Generator[SubprocessTao]:
    with new_tao(SubprocessTao, OPTICS_INIT) as tao:
        yield tao


@pytest.fixture
def plain_tao() -> Generator[Tao]:
    """An in-process Tao for exercising the fork core directly."""
    with new_tao(Tao, OPTICS_INIT) as tao:
        yield tao


def fork_map_in_process(tao: Tao, func, items, **kwargs) -> list:
    """The fork core on an in-process Tao; `Tao` itself has no `fork_map`."""
    raises = kwargs.get("raises", True)
    return decode_fork_payloads(fork_map_payloads(tao, func, items, **kwargs), raises)


@pytest.fixture
def forked_pids(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the pids of every child forked by this process during the test."""
    pids: list[int] = []
    orig_fork = os.fork

    def recording_fork() -> int:
        pid = orig_fork()
        if pid:
            pids.append(pid)
        return pid

    monkeypatch.setattr(os, "fork", recording_fork)
    return pids


@pytest.fixture
def main_module_class(monkeypatch: pytest.MonkeyPatch) -> type:
    """A class that pickles by reference to `__main__`, which the subprocess lacks."""

    @dataclasses.dataclass
    class ForkItem:
        value: int

    ForkItem.__module__ = "__main__"
    ForkItem.__qualname__ = ForkItem.__name__ = "_PytaoForkItem"
    monkeypatch.setattr(sys.modules["__main__"], "_PytaoForkItem", ForkItem, raising=False)
    return ForkItem


def assert_all_reaped(pids: list[int]) -> None:
    for pid in pids:
        with pytest.raises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)


def set_k1_and_read(tao: Tao, k1: float) -> float:
    tao.cmd(f"set ele Q1 k1 = {k1}")
    return float(tao.ele_gen_attribs("Q1")["K1"])


def beta_for_k1(tao: Tao, k1: float) -> float:
    tao.cmd(f"set ele Q1 k1 = {k1}")
    return float(tao.ele_twiss("Q2")["beta_a"])


def sleep_then_return(tao: Tao, item: tuple[float, int]) -> int:
    delay, value = item
    time.sleep(delay)
    return value


def identity(tao: Tao, item: Any) -> Any:
    return item


def raise_on_two(tao: Tao, item: int) -> int:
    if item == 2:
        raise ValueError("boom")
    return item


def bad_tao_command(tao: Tao, item: Any) -> None:
    tao.cmd("set ele NOPE k1 = 1")


def exit_three(tao: Tao, item: Any) -> None:
    os._exit(3)


def kill_self(tao: Tao, item: Any) -> None:
    os.kill(os.getpid(), signal.SIGKILL)


def sleep_forever(tao: Tao, item: Any) -> None:
    time.sleep(30)


def open_pipe_fds(tao: Tao, item: Any) -> list[int]:
    fds = []
    for fd in range(3, 256):
        try:
            if stat.S_ISFIFO(os.fstat(fd).st_mode):
                fds.append(fd)
        except OSError:
            pass
    return fds


def sleep_then_return_pipe_fds(tao: Tao, item: int) -> list[int]:
    # Keep every child alive until all have been forked.
    time.sleep(0.5)
    return open_pipe_fds(tao, item)


def stdin_is_devnull(tao: Tao, item: Any) -> bool:
    return os.path.samestat(os.fstat(0), os.stat(os.devnull))


def write_pid_and_sleep(tao: Tao, pid_file: str) -> None:
    pathlib.Path(pid_file).write_text(str(os.getpid()))
    time.sleep(30)


def item_value(tao: Tao, item: Any) -> int:
    return item.value


def bunch_x(tao: Tao, item: Any) -> np.ndarray:
    return tao.bunch1(ele_id="end", coordinate="x", which="model", ix_bunch="1")


def test_independent_results(tao: SubprocessTao) -> None:
    k1_before = float(tao.ele_gen_attribs("Q1")["K1"])
    items = [1.0, 2.0, 3.0]

    assert tao.fork_map(set_k1_and_read, items) == items
    assert float(tao.ele_gen_attribs("Q1")["K1"]) == k1_before


def test_results_differ_per_item(tao: SubprocessTao) -> None:
    betas = tao.fork_map(beta_for_k1, [0.0, 0.5, 1.0, 1.5], max_workers=2)
    assert len(set(betas)) == 4


def test_order_preserved(tao: SubprocessTao) -> None:
    items = [(0.3, 0), (0.2, 1), (0.1, 2)]
    assert tao.fork_map(sleep_then_return, items, max_workers=3) == [0, 1, 2]


@pytest.mark.parametrize("max_workers", [1, 2, 4, None])
def test_max_workers(tao: SubprocessTao, max_workers: int | None) -> None:
    assert tao.fork_map(identity, range(6), max_workers=max_workers) == list(range(6))


@pytest.mark.parametrize(
    ("max_workers", "expected"),
    [(None, 8), (3, 3), (-1, 7), (-7, 1), (-8, 1), (-100, 1)],
)
def test_resolve_max_workers(
    monkeypatch: pytest.MonkeyPatch, max_workers: int | None, expected: int
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert resolve_max_workers(max_workers) == expected


def test_zero_max_workers(tao: SubprocessTao) -> None:
    with pytest.raises(ValueError, match="nonzero"):
        tao.fork_map(identity, [1], max_workers=0)


def test_negative_max_workers(tao: SubprocessTao) -> None:
    assert tao.fork_map(identity, range(4), max_workers=-1) == list(range(4))


def test_payloads_left_undecoded(plain_tao: Tao) -> None:
    payloads = fork_map_payloads(plain_tao, identity, [1, "a"])
    assert [type(payload) for payload, _ in payloads] == [bytes, bytes]
    assert [code for _, code in payloads] == [0, 0]
    assert decode_fork_payloads(payloads) == [1, "a"]


def test_empty_items(tao: SubprocessTao) -> None:
    assert tao.fork_map(identity, []) == []


def test_func_raises(tao: SubprocessTao) -> None:
    with pytest.raises(TaoForkError) as excinfo:
        tao.fork_map(raise_on_two, range(4))

    err = excinfo.value
    assert err.index == 2
    assert err.error_cls == "ValueError"
    assert "boom" in str(err)
    assert "ValueError: boom" in err.child_traceback


def test_func_raises_returned_in_place(tao: SubprocessTao) -> None:
    results = tao.fork_map(raise_on_two, range(4), raises=False)
    assert results[:2] == [0, 1]
    assert results[3] == 3
    assert isinstance(results[2], TaoForkError)
    assert results[2].index == 2


def test_tao_command_error_in_child(tao: SubprocessTao) -> None:
    with pytest.raises(TaoForkError) as excinfo:
        tao.fork_map(bad_tao_command, [0])

    assert excinfo.value.error_cls == "TaoCommandError"
    assert excinfo.value.tao_output


def test_child_exit_without_result(tao: SubprocessTao) -> None:
    with pytest.raises(TaoForkError, match="exited with code 3") as excinfo:
        tao.fork_map(exit_three, [0])
    assert excinfo.value.exit_code == 3


def test_child_killed_by_signal(tao: SubprocessTao) -> None:
    with pytest.raises(TaoForkError, match="SIGKILL") as excinfo:
        tao.fork_map(kill_self, [0])
    assert excinfo.value.exit_code == -signal.SIGKILL


def test_timeout(tao: SubprocessTao) -> None:
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        tao.fork_map(sleep_forever, range(3), timeout=0.5)
    assert time.monotonic() - t0 < 5.0

    assert tao.ele_twiss("Q2")["beta_a"] > 0


def test_timeout_reaps_children(plain_tao: Tao, forked_pids: list[int]) -> None:
    with pytest.raises(TimeoutError):
        fork_map_in_process(plain_tao, sleep_forever, range(3), timeout=0.5)

    assert len(forked_pids) == 3
    assert_all_reaped(forked_pids)


def test_failure_kills_remaining_children(plain_tao: Tao, forked_pids: list[int]) -> None:
    def fail_fast_or_sleep(tao: Tao, item: int) -> int:
        if item == 0:
            raise RuntimeError("first")
        time.sleep(30)
        return item

    t0 = time.monotonic()
    with pytest.raises(TaoForkError):
        fork_map_in_process(plain_tao, fail_fast_or_sleep, range(3), max_workers=3)
    assert time.monotonic() - t0 < 5.0
    assert_all_reaped(forked_pids)


def test_keyboard_interrupt_reaps_children(
    plain_tao: Tao,
    forked_pids: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupt(self, timeout=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(selectors.DefaultSelector, "select", interrupt)
    with pytest.raises(KeyboardInterrupt):
        fork_map_in_process(plain_tao, sleep_forever, range(2), max_workers=2)

    assert len(forked_pids) == 2
    assert_all_reaped(forked_pids)


def test_fork_failure_closes_pipes(plain_tao: Tao, monkeypatch: pytest.MonkeyPatch) -> None:
    orig_fork = os.fork
    forks = 0

    def fork_once() -> int:
        nonlocal forks
        forks += 1
        if forks > 1:
            raise BlockingIOError("fork limit")
        return orig_fork()

    before = open_pipe_fds(plain_tao, None)
    monkeypatch.setattr(os, "fork", fork_once)
    with pytest.raises(BlockingIOError):
        fork_map_in_process(plain_tao, sleep_forever, range(2), max_workers=2)
    assert open_pipe_fds(plain_tao, None) == before


def test_core_accepts_closure_and_lambda(plain_tao: Tao) -> None:
    offset = 10

    def add_offset(tao: Tao, item: int) -> int:
        return item + offset

    assert fork_map_in_process(plain_tao, add_offset, [1, 2]) == [11, 12]
    assert fork_map_in_process(plain_tao, lambda tao, item: item * 2, [1, 2]) == [2, 4]


def test_closure_and_lambda_in_subprocess() -> None:
    pytest.importorskip("cloudpickle")
    offset = 10

    def add_offset(tao: Tao, item: int) -> int:
        return item + offset

    with new_tao(SubprocessTao, OPTICS_INIT) as tao:
        assert tao.fork_map(add_offset, [1, 2]) == [11, 12]
        assert tao.fork_map(lambda tao, item: item * 2, [1, 2]) == [2, 4]


def test_subprocess_without_cloudpickle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pytao.subproc.cloudpickle", None)

    def local_func(tao: Tao, item: int) -> int:
        return item

    with new_tao(SubprocessTao, OPTICS_INIT) as tao:
        with pytest.raises(ValueError, match="cloudpickle"):
            tao.fork_map(local_func, [1])
        with pytest.raises(ValueError, match="cloudpickle"):
            tao.fork_map(lambda tao, item: item, [1])
        assert tao.fork_map(identity, [1, 2]) == [1, 2]


def test_numpy_result_round_trip() -> None:
    with new_tao(SubprocessTao, CSR_INIT) as tao:
        expected = bunch_x(tao, None)
        (result,) = tao.fork_map(bunch_x, [None])
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, expected)


def test_fork_error_pickle_round_trip() -> None:
    err = TaoForkError(
        "msg",
        tao_output="out",
        index=3,
        error_cls="ValueError",
        child_traceback="tb",
        exit_code=-9,
    )
    restored = pickle.loads(pickle.dumps(err))
    assert isinstance(restored, TaoForkError)
    assert str(restored) == "msg"
    assert restored.tao_output == "out"
    assert restored.index == 3
    assert restored.error_cls == "ValueError"
    assert restored.child_traceback == "tb"
    assert restored.exit_code == -9


def test_not_implemented_without_fork(plain_tao: Tao, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(os, "fork")
    with pytest.raises(NotImplementedError):
        fork_map_in_process(plain_tao, identity, [1])


def test_children_do_not_hold_sibling_pipes(tao: SubprocessTao) -> None:
    (baseline,) = tao.fork_map(open_pipe_fds, [None])
    results = tao.fork_map(sleep_then_return_pipe_fds, range(4), max_workers=4)
    assert [len(fds) for fds in results] == [len(baseline)] * 4


def test_subprocess_children_detach_from_pipes() -> None:
    with new_tao(SubprocessTao, OPTICS_INIT) as tao:
        (fds,) = tao.fork_map(open_pipe_fds, [None])
        assert len(fds) == 1, "only the child's own result pipe should be open"
        assert tao.fork_map(stdin_is_devnull, [None]) == [True]


def test_subprocess_timeout_kills_children(tmp_path: pathlib.Path) -> None:
    pid_files = [str(tmp_path / f"{i}.pid") for i in range(2)]
    with new_tao(SubprocessTao, OPTICS_INIT) as tao:
        t0 = time.monotonic()
        with pytest.raises(TimeoutError):
            with tao.timeout(1.0):
                tao.fork_map(write_pid_and_sleep, pid_files, max_workers=2)
        assert time.monotonic() - t0 < 10.0

    for pid_file in pid_files:
        with pytest.raises(ProcessLookupError):
            os.kill(int(pathlib.Path(pid_file).read_text()), 0)


def test_subprocess_items_from_main(main_module_class: type) -> None:
    pytest.importorskip("cloudpickle")
    with new_tao(SubprocessTao, OPTICS_INIT) as tao:
        assert tao.fork_map(item_value, [main_module_class(1), main_module_class(2)]) == [1, 2]


def test_subprocess_survives_unloadable_items(
    main_module_class: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("pytao.subproc.cloudpickle", None)
    with new_tao(SubprocessTao, OPTICS_INIT) as tao:
        with pytest.raises(TaoCommandError, match="_PytaoForkItem"):
            tao.fork_map(item_value, [main_module_class(1)])
        assert tao.subprocess_alive
        assert tao.fork_map(identity, [1, 2]) == [1, 2]


@pytest.mark.parametrize(
    "items",
    [
        pytest.param([1, "a", {"b": 2.0}], id="builtins"),
        pytest.param([np.arange(3)], id="ndarray"),
        pytest.param([TaoForkError("x")], id="importable-class"),
    ],
)
def test_items_use_plain_pickle_when_possible(items: list) -> None:
    assert _pickle_for_subprocess(items) == pickle.dumps(
        items, protocol=pickle.HIGHEST_PROTOCOL
    )


@pytest.mark.parametrize(
    "make_items",
    [
        pytest.param(lambda cls: [cls(1)], id="main-class"),
        pytest.param(lambda cls: [lambda: 1], id="lambda"),
    ],
)
def test_items_fall_back_to_cloudpickle(main_module_class: type, make_items) -> None:
    cloudpickle = pytest.importorskip("cloudpickle")
    items = make_items(main_module_class)
    assert _pickle_for_subprocess(items) == cloudpickle.dumps(
        items, protocol=pickle.HIGHEST_PROTOCOL
    )


def test_tao_has_no_fork_map() -> None:
    assert not hasattr(Tao, "fork_map")
    assert callable(SubprocessTao.fork_map)
