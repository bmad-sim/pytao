from __future__ import annotations

import pathlib
from collections.abc import Generator
from typing import Literal

import numpy as np
import pytest

from pytao import SubprocessTao
from pytao.model.ele import Comb
from pytao.model.ele.comb import combine_combs


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
    assert np.array_equal(sliced.charge_live, expected)
    assert np.array_equal(sliced.s, expected)

    expected = np.arange(1.0, 10.0).tolist()
    sliced = comb.slice_by_s(0, 10, inclusive=False)
    assert np.array_equal(sliced.charge_live, expected)
    assert np.array_equal(sliced.s, expected)


def test_restore_backcompat():
    data = {"charge_live": [1, 2, 3]}
    assert Comb.model_validate(data) == Comb(charge_live=np.asarray([1, 2, 3]))


def test_restore_simple():
    data = {"charge_live": np.asarray([1, 2, 3])}
    assert Comb.model_validate(data) == Comb(charge_live=np.asarray([1, 2, 3]))


def test_combine():
    a = Comb(
        charge_live=np.asarray([4, 5, 6]),
        s=np.asarray([4, 5, 6]),
    )
    b = Comb(
        charge_live=np.asarray([1, 2, 3]),
        s=np.asarray([1, 2, 3]),
    )

    combined = combine_combs([a, b])
    expected = Comb(
        charge_live=np.asarray([1, 2, 3, 4, 5, 6]),
        s=np.asarray([1, 2, 3, 4, 5, 6]),
    )
    assert combined == expected


@pytest.mark.parametrize("format", ["msgpack", "json"])
@pytest.mark.parametrize("exclude_defaults", [False, True])
def test_save(
    tmp_path: pathlib.Path, format: Literal["msgpack", "json"], exclude_defaults: bool
):
    source = Comb(
        charge_live=np.asarray([4, 5, 6]),
        s=np.asarray([4, 5, 6]),
    )

    source.write(tmp_path, format=format, exclude_defaults=exclude_defaults)
    loaded = Comb.from_file(tmp_path, format=format)
    assert source == loaded
