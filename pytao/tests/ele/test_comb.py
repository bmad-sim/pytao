from __future__ import annotations

import pathlib
from collections.abc import Generator
from typing import Literal

import numpy as np
import pytest

from pytao import SubprocessTao
from pytao.model.base import _msgpack_default
from pytao.model import Comb, Element, ElementHead, Lattice
from pytao.model.ele.comb import (
    combine_combs,
    load_combs_from_lattice_data,
    load_combs_from_lattice_file,
)


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
    comb.query(tao)  # smoke


@pytest.mark.parametrize(
    "attr",
    (
        "max_delta",
        "min_delta",
        "px_max",
        "px_min",
        "py_max",
        "py_min",
        "x_max",
        "x_min",
        "y_max",
        "y_min",
        "z_max",
        "z_min",
    ),
)
def test_comb_smoke_properties(comb: Comb, attr: str) -> None:
    value = getattr(comb, attr)
    assert isinstance(value, np.ndarray)
    assert len(value) == len(comb.s)


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


def test_combine_empty():
    a = Comb()
    b = Comb()

    combined = combine_combs([a, b])
    expected = Comb()
    assert combined == expected


def test_load_from_lattice_data():
    d1 = {
        "elements": [
            {
                "comb": {"s": [4, 5, 6]},
            },
            {
                "comb": {"s": _msgpack_default(np.asarray([1, 2, 3]))},
            },
            {
                "comb": {"s": []},
            },
        ],
    }

    expected = Comb(s=np.asarray([4, 5, 6, 1, 2, 3]))
    combined = load_combs_from_lattice_data(d1)
    assert combined == expected

    expected = Comb(s=np.asarray([1, 2, 3, 4, 5, 6]))
    combined = load_combs_from_lattice_data(d1, sort=True)
    assert combined == expected


@pytest.mark.parametrize("format", ["msgpack", "json"])
@pytest.mark.parametrize("exclude_defaults", [False, True])
def test_load_from_lattice_file(
    tmp_path: pathlib.Path, format: Literal["msgpack", "json"], exclude_defaults: bool
):
    lat = Lattice(
        which="model",
        elements=(
            Element(
                ele_id="0",
                which="model",
                head=ElementHead(key="BEGINNING"),
                comb=Comb(s=np.asarray([4, 5, 6])),
            ),
            Element(
                ele_id="1",
                which="model",
                head=ElementHead(key="PIPE"),
                comb=Comb(s=np.asarray([1, 2, 3])),
            ),
        ),
    )

    lat.write(tmp_path, format=format, exclude_defaults=exclude_defaults)

    expected = Comb(s=np.asarray([4, 5, 6, 1, 2, 3]))
    combined = load_combs_from_lattice_file(tmp_path, format=format, sort=False)
    assert combined == expected

    expected = Comb(s=np.asarray([1, 2, 3, 4, 5, 6]))
    combined = load_combs_from_lattice_file(tmp_path, format=format, sort=True)
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
