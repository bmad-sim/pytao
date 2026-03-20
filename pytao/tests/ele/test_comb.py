from __future__ import annotations

from collections.abc import Generator

import numpy as np
import pytest

from pytao import SubprocessTao
from pytao.model.ele import Comb


@pytest.fixture(scope="module")
def tao() -> Generator[SubprocessTao]:
    with SubprocessTao(
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall3d", noplot=True
    ) as tao:
        yield tao


@pytest.fixture(scope="function")
def comb() -> Comb:
    data = {
        key: np.arange(0, 1000) for key in Comb.model_fields if key not in {"command_args"}
    }
    return Comb(**data)


def test_comb_from_tao(tao: SubprocessTao):
    comb = Comb.from_tao(tao)
    print(repr(comb))
    assert "command_args" not in repr(comb)
    comb.query(tao)


def test_comb_slice(comb: Comb):
    sliced = comb.slice_by_s(0, 10, inclusive=True)
    expected = np.arange(0.0, 11.0).tolist()
    assert sliced.charge_live == expected
    assert sliced.s == expected

    expected = np.arange(1.0, 10.0).tolist()
    sliced = comb.slice_by_s(0, 10, inclusive=False)
    assert sliced.charge_live == expected
    assert sliced.s == expected
