import pytest

from pytao.constraints.observables.datum import DatumIsClose, DatumLessThan, DatumLiteral
from pytao.constraints.observables.ele import TolComparison


@pytest.fixture
def datum_lit():
    return DatumLiteral(model_value=1.0, design_value=1.0)


@pytest.mark.parametrize(
    "overrides, outcome",
    [
        ({"model_value": 1.0}, True),
        ({"model_value": 1.0 + 1e-6}, True),
        ({"model_value": 2.0}, False),
        ({"design_value": 2.0}, True),
    ],
)
def test_datum_is_close(datum_lit, overrides, outcome):
    obs_a = datum_lit.model_copy(update=overrides)()
    obs_b = datum_lit()
    result = DatumIsClose()(obs_a, obs_b)
    assert result.is_close == outcome
    assert result.model_value is not None
    assert result.design_value is None


@pytest.mark.parametrize(
    "overrides, outcome",
    [
        ({"model_value": 0.5}, True),
        ({"model_value": 1.0}, False),
        ({"model_value": 2.0}, False),
    ],
)
def test_datum_less_than(datum_lit, overrides, outcome):
    obs_a = datum_lit.model_copy(update=overrides)()
    obs_b = datum_lit()
    result = DatumLessThan()(obs_a, obs_b)
    assert result.is_close == outcome
    assert result.model_value is not None
    assert result.design_value is None


@pytest.mark.parametrize(
    "comparison, overrides, outcome",
    [
        (DatumIsClose(model_value_test=None), {"model_value": 2.0}, True),
        (
            DatumIsClose(model_value_test=None, design_value_test=TolComparison()),
            {"design_value": 2.0},
            False,
        ),
        (
            DatumIsClose(model_value_test=None, design_value_test=TolComparison()),
            {"design_value": 1.0 + 1e-6},
            True,
        ),
        (
            DatumIsClose(model_value_test=None, design_value_test=None),
            {"model_value": 2.0},
            True,
        ),
        (DatumIsClose(design_value_test=TolComparison()), {"design_value": 2.0}, False),
        (DatumIsClose(model_value_test=TolComparison(atol=0.5)), {"model_value": 1.3}, True),
    ],
)
def test_datum_is_close_turned_off(datum_lit, comparison, overrides, outcome):
    obs_a = datum_lit.model_copy(update=overrides)()
    obs_b = datum_lit()
    result = comparison(obs_a, obs_b)
    assert result.is_close == outcome


@pytest.mark.parametrize(
    "comparison, overrides, outcome",
    [
        (DatumLessThan(model_value=False), {"model_value": 2.0}, True),
        (
            DatumLessThan(model_value=False, design_value=True),
            {"model_value": 2.0, "design_value": 0.5},
            True,
        ),
        (
            DatumLessThan(model_value=False, design_value=True),
            {"model_value": 2.0, "design_value": 2.0},
            False,
        ),
        (
            DatumLessThan(model_value=True, design_value=True),
            {"model_value": 0.5, "design_value": 0.5},
            True,
        ),
        (
            DatumLessThan(model_value=True, design_value=True),
            {"model_value": 0.5, "design_value": 2.0},
            False,
        ),
        (
            DatumLessThan(model_value=True, design_value=True),
            {"model_value": 2.0, "design_value": 0.5},
            False,
        ),
    ],
)
def test_datum_less_than_turned_off(datum_lit, comparison, overrides, outcome):
    obs_a = datum_lit.model_copy(update=overrides)()
    obs_b = datum_lit()
    result = comparison(obs_a, obs_b)
    assert result.is_close == outcome
