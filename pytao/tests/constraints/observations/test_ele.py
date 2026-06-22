import pytest

from pytao.constraints.observables.ele import EleIsClose, EleLessThan, EleLiteral

from .utils import assert_result_fields

_ELE_IC = {
    "twiss_a": True,
    "twiss_b": True,
    "eta_x": True,
    "etap_x": True,
    "eta_y": True,
    "etap_y": True,
    "ref_energy": None,
    "p0c": True,
    "orbit": True,
    "floor_x": None,
    "floor_y": None,
    "floor_z": None,
}

_ELE_LT_NONE = {
    "beta_a": None,
    "alpha_a": None,
    "beta_b": None,
    "alpha_b": None,
    "eta_x": None,
    "etap_x": None,
    "eta_y": None,
    "etap_y": None,
    "ref_energy": None,
    "p0c": None,
    "floor_x": None,
    "floor_y": None,
    "floor_z": None,
}


@pytest.fixture
def ele_lit():
    return EleLiteral(beta_a=5.0, alpha_a=0.5, beta_b=3.0, alpha_b=0.3)


_ELE_IC_OP = EleIsClose(ref_energy=None)


@pytest.mark.parametrize(
    "overrides, outcome, fields",
    [
        ({"beta_a": 5.0}, True, _ELE_IC),
        ({"beta_a": 10.0}, False, {**_ELE_IC, "twiss_a": False}),
        ({"alpha_a": 0.5}, True, _ELE_IC),
        ({"alpha_a": -0.5}, False, {**_ELE_IC, "twiss_a": False}),
        ({"eta_x": 0.0}, True, _ELE_IC),
        ({"eta_x": 1.0}, False, {**_ELE_IC, "eta_x": False}),
        ({"p0c": 0.0}, True, _ELE_IC),
        ({"p0c": 1.0}, False, {**_ELE_IC, "p0c": False}),
    ],
)
def test_ele_is_satisfied(ele_lit, overrides, outcome, fields):
    obs_a = ele_lit.model_copy(update=overrides).observe()
    obs_b = ele_lit.observe()
    result = _ELE_IC_OP.compare(obs_a, obs_b)
    assert result.is_satisfied == outcome
    assert_result_fields(result, fields)


def test_ele_is_satisfied_missing_data_fails(ele_lit):
    obs_a = ele_lit.observe()
    obs_b = ele_lit.observe()
    result = EleIsClose().compare(obs_a, obs_b)
    assert not result.is_satisfied
    assert result.ref_energy is not None
    assert not result.ref_energy.passed
    assert result.ref_energy.detail


@pytest.mark.parametrize(
    "comparison, overrides, outcome, fields",
    [
        (EleLessThan(beta_a=True), {"beta_a": 4.0}, True, {**_ELE_LT_NONE, "beta_a": True}),
        (EleLessThan(beta_a=True), {"beta_a": 6.0}, False, {**_ELE_LT_NONE, "beta_a": False}),
        (EleLessThan(alpha_a=True), {"alpha_a": 0.3}, True, {**_ELE_LT_NONE, "alpha_a": True}),
        (
            EleLessThan(alpha_a=True),
            {"alpha_a": 0.5},
            False,
            {**_ELE_LT_NONE, "alpha_a": False},
        ),
        (EleLessThan(eta_x=True), {"eta_x": -0.1}, True, {**_ELE_LT_NONE, "eta_x": True}),
        (EleLessThan(eta_x=True), {"eta_x": 0.1}, False, {**_ELE_LT_NONE, "eta_x": False}),
        (EleLessThan(p0c=True), {"p0c": -1.0}, True, {**_ELE_LT_NONE, "p0c": True}),
        (EleLessThan(p0c=True), {"p0c": 1.0}, False, {**_ELE_LT_NONE, "p0c": False}),
    ],
)
def test_ele_less_than(ele_lit, comparison, overrides, outcome, fields):
    obs_a = ele_lit.model_copy(update=overrides).observe()
    obs_b = ele_lit.observe()
    result = comparison.compare(obs_a, obs_b)
    assert result.is_satisfied == outcome
    assert_result_fields(result, fields)


@pytest.mark.parametrize(
    "comparison, overrides, outcome, fields",
    [
        (
            EleIsClose(twiss_a=None, ref_energy=None),
            {"beta_a": 10.0},
            True,
            {**_ELE_IC, "twiss_a": None},
        ),
        (
            EleIsClose(twiss_b=None, ref_energy=None),
            {"beta_b": 10.0},
            True,
            {**_ELE_IC, "twiss_b": None},
        ),
        (
            EleIsClose(eta_x=None, ref_energy=None),
            {"eta_x": 1.0},
            True,
            {**_ELE_IC, "eta_x": None},
        ),
        (
            EleIsClose(etap_x=None, ref_energy=None),
            {"etap_x": 1.0},
            True,
            {**_ELE_IC, "etap_x": None},
        ),
        (
            EleIsClose(eta_y=None, ref_energy=None),
            {"eta_y": 1.0},
            True,
            {**_ELE_IC, "eta_y": None},
        ),
        (
            EleIsClose(p0c=None, ref_energy=None),
            {"p0c": 1.0},
            True,
            {**_ELE_IC, "p0c": None},
        ),
    ],
)
def test_ele_is_satisfied_turned_off(ele_lit, comparison, overrides, outcome, fields):
    obs_a = ele_lit.model_copy(update=overrides).observe()
    obs_b = ele_lit.observe()
    result = comparison.compare(obs_a, obs_b)
    assert result.is_satisfied == outcome
    assert_result_fields(result, fields)


@pytest.mark.parametrize(
    "comparison, overrides, outcome, fields",
    [
        (EleLessThan(beta_a=False), {"beta_a": 6.0}, True, _ELE_LT_NONE),
        (
            EleLessThan(beta_a=True, alpha_a=True),
            {"beta_a": 4.0, "alpha_a": 0.3},
            True,
            {**_ELE_LT_NONE, "beta_a": True, "alpha_a": True},
        ),
        (
            EleLessThan(beta_a=True, alpha_a=True),
            {"beta_a": 4.0, "alpha_a": 0.6},
            False,
            {**_ELE_LT_NONE, "beta_a": True, "alpha_a": False},
        ),
        (
            EleLessThan(beta_a=False, alpha_a=True),
            {"beta_a": 6.0, "alpha_a": 0.3},
            True,
            {**_ELE_LT_NONE, "alpha_a": True},
        ),
        (
            EleLessThan(eta_x=False, p0c=True),
            {"eta_x": 1.0, "p0c": -1.0},
            True,
            {**_ELE_LT_NONE, "p0c": True},
        ),
        (
            EleLessThan(eta_x=True, p0c=True),
            {"eta_x": 1.0, "p0c": -1.0},
            False,
            {**_ELE_LT_NONE, "eta_x": False, "p0c": True},
        ),
    ],
)
def test_ele_less_than_turned_off(ele_lit, comparison, overrides, outcome, fields):
    obs_a = ele_lit.model_copy(update=overrides).observe()
    obs_b = ele_lit.observe()
    result = comparison.compare(obs_a, obs_b)
    assert result.is_satisfied == outcome
    assert_result_fields(result, fields)
