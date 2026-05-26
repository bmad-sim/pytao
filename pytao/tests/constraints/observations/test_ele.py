import pytest

from pytao.constraints.observables.ele import EleIsClose, EleLessThan, EleLiteral


@pytest.fixture
def ele_lit():
    return EleLiteral(beta_a=5.0, alpha_a=0.5, beta_b=3.0, alpha_b=0.3)


@pytest.mark.parametrize(
    "overrides, outcome",
    [
        ({"beta_a": 5.0}, True),
        ({"beta_a": 10.0}, False),
        ({"alpha_a": 0.5}, True),
        ({"alpha_a": -0.5}, False),
        ({"eta_x": 0.0}, True),
        ({"eta_x": 1.0}, False),
        ({"p0c": 0.0}, True),
        ({"p0c": 1.0}, False),
    ],
)
def test_ele_is_close(ele_lit, overrides, outcome):
    obs_a = ele_lit.model_copy(update=overrides)()
    obs_b = ele_lit()
    result = EleIsClose()(obs_a, obs_b)
    assert result.is_close == outcome


@pytest.mark.parametrize(
    "comparison, overrides, outcome",
    [
        (EleLessThan(beta_a=True), {"beta_a": 4.0}, True),
        (EleLessThan(beta_a=True), {"beta_a": 6.0}, False),
        (EleLessThan(alpha_a=True), {"alpha_a": 0.3}, True),
        (EleLessThan(alpha_a=True), {"alpha_a": 0.5}, False),
        (EleLessThan(eta_x=True), {"eta_x": -0.1}, True),
        (EleLessThan(eta_x=True), {"eta_x": 0.1}, False),
        (EleLessThan(p0c=True), {"p0c": -1.0}, True),
        (EleLessThan(p0c=True), {"p0c": 1.0}, False),
    ],
)
def test_ele_less_than(ele_lit, comparison, overrides, outcome):
    obs_a = ele_lit.model_copy(update=overrides)()
    obs_b = ele_lit()
    result = comparison(obs_a, obs_b)
    assert result.is_close == outcome


@pytest.mark.parametrize(
    "comparison, overrides, outcome",
    [
        (EleIsClose(twiss_a_test=None), {"beta_a": 10.0}, True),
        (EleIsClose(twiss_b_test=None), {"beta_b": 10.0}, True),
        (EleIsClose(eta_x_test=None), {"eta_x": 1.0}, True),
        (EleIsClose(etap_x_test=None), {"etap_x": 1.0}, True),
        (EleIsClose(eta_y_test=None), {"eta_y": 1.0}, True),
        (EleIsClose(p0c_test=None), {"p0c": 1.0}, True),
    ],
)
def test_ele_is_close_turned_off(ele_lit, comparison, overrides, outcome):
    obs_a = ele_lit.model_copy(update=overrides)()
    obs_b = ele_lit()
    result = comparison(obs_a, obs_b)
    assert result.is_close == outcome


@pytest.mark.parametrize(
    "comparison, overrides, outcome",
    [
        (EleLessThan(beta_a=False), {"beta_a": 6.0}, True),
        (EleLessThan(beta_a=True, alpha_a=True), {"beta_a": 4.0, "alpha_a": 0.3}, True),
        (EleLessThan(beta_a=True, alpha_a=True), {"beta_a": 4.0, "alpha_a": 0.6}, False),
        (EleLessThan(beta_a=False, alpha_a=True), {"beta_a": 6.0, "alpha_a": 0.3}, True),
        (EleLessThan(eta_x=False, p0c=True), {"eta_x": 1.0, "p0c": -1.0}, True),
        (EleLessThan(eta_x=True, p0c=True), {"eta_x": 1.0, "p0c": -1.0}, False),
    ],
)
def test_ele_less_than_turned_off(ele_lit, comparison, overrides, outcome):
    obs_a = ele_lit.model_copy(update=overrides)()
    obs_b = ele_lit()
    result = comparison(obs_a, obs_b)
    assert result.is_close == outcome
