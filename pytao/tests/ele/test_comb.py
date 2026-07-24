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

from ..conftest import set_gaussian


@pytest.fixture(scope="module")
def tao() -> Generator[SubprocessTao]:
    with SubprocessTao(
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall3d", noplot=True
    ) as tao:
        yield tao


@pytest.fixture(scope="function")
def comb() -> Comb:
    data = {
        key: np.arange(0, 1000)
        for key in Comb.model_fields
        if key not in {"command_args", "mc2"}
    }
    return Comb(mc2=5e6, **data)


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


@pytest.fixture(scope="function")
def raw_comb() -> Comb:
    """
    A Comb populated with distinct raw (Bmad-unit) arrays.

    Values are chosen so every derived property has an unambiguous expectation.
    """
    idx = np.arange(1, 5, dtype=float)
    return Comb(
        mc2=0.51099895e6,
        p0c=idx * 1e7,
        centroid_1=idx * 1.0,
        centroid_2=idx * 0.01,
        centroid_3=idx * 2.0,
        centroid_4=idx * 0.02,
        centroid_5=idx * 3.0,
        centroid_6=idx * 0.001,
        t=idx * 1e-9,
        sigma_11=idx * 4.0,
        sigma_22=idx * 5e-4,
        sigma_33=idx * 6.0,
        sigma_44=idx * 7e-4,
        sigma_55=idx * 8.0,
        sigma_66=idx * 9e-6,
        rel_min_1=-idx * 0.5,
        rel_min_2=-idx * 0.005,
        rel_min_3=-idx * 0.6,
        rel_min_4=-idx * 0.006,
        rel_min_5=-idx * 0.7,
        rel_min_6=-idx * 0.0007,
        rel_max_1=idx * 0.5,
        rel_max_2=idx * 0.005,
        rel_max_3=idx * 0.6,
        rel_max_4=idx * 0.006,
        rel_max_5=idx * 0.7,
        rel_max_6=idx * 0.0007,
    )


COMB_PROPERTIES = {
    "mean_x",
    "mean_px",
    "mean_y",
    "mean_py",
    "mean_z",
    "mean_delta",
    "mean_p",
    "mean_energy",
    "mean_t",
    "sigma_x",
    "sigma_px",
    "sigma_y",
    "sigma_py",
    "sigma_z",
    "sigma_p",
    "sigma_delta",
    "rel_min_x",
    "rel_min_px",
    "rel_min_y",
    "rel_min_py",
    "rel_min_z",
    "rel_min_p",
    "rel_min_delta",
    "rel_max_x",
    "rel_max_px",
    "rel_max_y",
    "rel_max_py",
    "rel_max_z",
    "rel_max_p",
    "rel_max_delta",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "z_min",
    "z_max",
    "px_min",
    "px_max",
    "py_min",
    "py_max",
    "min_delta",
    "max_delta",
}


@pytest.mark.parametrize("attr", list(COMB_PROPERTIES))
def test_comb_property_smoke(raw_comb: Comb, attr: str):
    getattr(raw_comb, attr)


def test_load_old_comb():
    fn = pathlib.Path(__file__).parent / "comb_old.json"
    comb = Comb.from_file(fn, format="json")

    assert isinstance(comb, Comb)
    assert len(comb.s) > 1
    assert comb.model_extra is not None
    for legacy_key in ("mean_x", "sigma_x", "cov_x__y", "norm_emit_x", "rel_min_delta"):
        assert legacy_key in comb.model_extra


@pytest.fixture(scope="module")
def tracked_comb() -> Comb:
    with SubprocessTao(
        init_file="$ACC_ROOT_DIR/bmad-doc/tao_examples/optics_matching/tao.init",
        noplot=True,
    ) as tao:
        set_gaussian(tao, n_particle=10)
        beam = tao.get_config().beam
        beam.comb_ds_save = 0.1
        beam.set(tao)
        tao.track_beam(use_progress_bar=False)
        return tao.get_comb()


def test_get_comb_tracked(tracked_comb: Comb):
    comb = tracked_comb
    assert len(comb.s) > 1
    assert comb.mc2 > 0
    assert np.all(np.diff(comb.s) >= 0)

    # Spot-check the unit conversions hold on real tracked data too.
    np.testing.assert_allclose(comb.mean_px, comb.centroid_2 * comb.p0c)
    np.testing.assert_allclose(comb.sigma_x, np.sqrt(comb.sigma_11))
    np.testing.assert_allclose(comb.sigma_delta, np.sqrt(comb.sigma_66))
    np.testing.assert_allclose(comb.mean_energy, np.hypot(comb.mean_p, comb.mc2))
    np.testing.assert_allclose(comb.x_min, comb.mean_x + comb.rel_min_x)

    assert not comb.model_extra
