import pathlib

import pytest

from pytao import SubprocessTao
from pytao.constraints.observables.datum import DatumLiteral, DatumObservable, DatumObservation
from pytao.constraints.observables.ele import (
    EleMaxObservable,
    EleMinObservable,
    EleObservable,
    EleLiteral,
    EleObservation,
)

DATA_DIR = pathlib.Path(__file__).parent / "data"
LAT_A = DATA_DIR / "lattices" / "lat_a.lat.bmad"


@pytest.fixture(scope="module")
def tao():
    with SubprocessTao(lattice_file=LAT_A, noplot=True) as t:
        yield t


@pytest.mark.parametrize(
    "obs, expected_type",
    [
        (DatumLiteral(model_value=1.0, design_value=0.0), DatumObservation),
        (EleLiteral(beta_a=5.0, alpha_a=0.5, beta_b=3.0, alpha_b=0.2), EleObservation),
    ],
)
def test_literal_observable(obs, expected_type):
    result = obs()
    assert isinstance(result, expected_type)
    assert result.elapsed_time >= 0.0
    if isinstance(result, DatumObservation):
        assert result.model_value == 1.0
        assert result.design_value == 0.0
    else:
        assert result.element is not None
        assert result.element.twiss.beta_a == 5.0


@pytest.mark.parametrize(
    "obs",
    [
        EleObservable(lattice_id="lat_a", ele_id="BEGINNING"),
        EleObservable(lattice_id="lat_a", ele_id="END"),
        EleMaxObservable(lattice_id="lat_a"),
        EleMinObservable(lattice_id="lat_a"),
    ],
)
def test_ele_lattice_observable(tao, obs):
    result = obs(tao)
    assert isinstance(result, EleObservation)
    assert result.elapsed_time >= 0.0
    assert result.element is not None


@pytest.mark.parametrize(
    "obs",
    [
        DatumObservable(lattice_id="lat_a", data_type="r56_compaction", ele_name="END"),
    ],
)
def test_datum_lattice_observable(tao, obs):
    result = obs(tao)
    assert isinstance(result, DatumObservation)
    assert result.elapsed_time >= 0.0
    assert isinstance(result.model_value, float)
    assert isinstance(result.design_value, float)
