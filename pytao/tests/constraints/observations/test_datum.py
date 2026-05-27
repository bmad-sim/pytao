import pytest

from pytao.constraints.observables.datum import DatumIsClose, DatumLessThan, DatumLiteral
from pytao.constraints.observables.ele import TolComparison

from .utils import assert_result_fields

_IC = {"model_value": True, "design_value": None}
_LT = {"model_value": True, "design_value": None}


@pytest.fixture
def datum_lit():
    return DatumLiteral(model_value=1.0, design_value=1.0)


@pytest.mark.parametrize(
    "overrides, outcome, fields",
    [
        ({"model_value": 1.0}, True, _IC),
        ({"model_value": 1.0 + 1e-6}, True, _IC),
        ({"model_value": 2.0}, False, {**_IC, "model_value": False}),
        ({"design_value": 2.0}, True, _IC),
    ],
)
def test_datum_is_satisfied(datum_lit, overrides, outcome, fields):
    obs_a = datum_lit.model_copy(update=overrides)()
    obs_b = datum_lit()
    result = DatumIsClose()(obs_a, obs_b)
    assert result.is_satisfied == outcome
    assert_result_fields(result, fields)


@pytest.mark.parametrize(
    "overrides, outcome, fields",
    [
        ({"model_value": 0.5}, True, _LT),
        ({"model_value": 1.0}, False, {**_LT, "model_value": False}),
        ({"model_value": 2.0}, False, {**_LT, "model_value": False}),
    ],
)
def test_datum_less_than(datum_lit, overrides, outcome, fields):
    obs_a = datum_lit.model_copy(update=overrides)()
    obs_b = datum_lit()
    result = DatumLessThan()(obs_a, obs_b)
    assert result.is_satisfied == outcome
    assert_result_fields(result, fields)


@pytest.mark.parametrize(
    "comparison, overrides, outcome, fields",
    [
        (
            DatumIsClose(model_value_test=None),
            {"model_value": 2.0},
            True,
            {"model_value": None, "design_value": None},
        ),
        (
            DatumIsClose(model_value_test=None, design_value_test=TolComparison()),
            {"design_value": 2.0},
            False,
            {"model_value": None, "design_value": False},
        ),
        (
            DatumIsClose(model_value_test=None, design_value_test=TolComparison()),
            {"design_value": 1.0 + 1e-6},
            True,
            {"model_value": None, "design_value": True},
        ),
        (
            DatumIsClose(model_value_test=None, design_value_test=None),
            {"model_value": 2.0},
            True,
            {"model_value": None, "design_value": None},
        ),
        (
            DatumIsClose(design_value_test=TolComparison()),
            {"design_value": 2.0},
            False,
            {"model_value": True, "design_value": False},
        ),
        (
            DatumIsClose(model_value_test=TolComparison(atol=0.5)),
            {"model_value": 1.3},
            True,
            {"model_value": True, "design_value": None},
        ),
    ],
)
def test_datum_is_satisfied_turned_off(datum_lit, comparison, overrides, outcome, fields):
    obs_a = datum_lit.model_copy(update=overrides)()
    obs_b = datum_lit()
    result = comparison(obs_a, obs_b)
    assert result.is_satisfied == outcome
    assert_result_fields(result, fields)


@pytest.mark.parametrize(
    "comparison, overrides, outcome, fields",
    [
        (
            DatumLessThan(model_value=False),
            {"model_value": 2.0},
            True,
            {"model_value": None, "design_value": None},
        ),
        (
            DatumLessThan(model_value=False, design_value=True),
            {"model_value": 2.0, "design_value": 0.5},
            True,
            {"model_value": None, "design_value": True},
        ),
        (
            DatumLessThan(model_value=False, design_value=True),
            {"model_value": 2.0, "design_value": 2.0},
            False,
            {"model_value": None, "design_value": False},
        ),
        (
            DatumLessThan(model_value=True, design_value=True),
            {"model_value": 0.5, "design_value": 0.5},
            True,
            {"model_value": True, "design_value": True},
        ),
        (
            DatumLessThan(model_value=True, design_value=True),
            {"model_value": 0.5, "design_value": 2.0},
            False,
            {"model_value": True, "design_value": False},
        ),
        (
            DatumLessThan(model_value=True, design_value=True),
            {"model_value": 2.0, "design_value": 0.5},
            False,
            {"model_value": False, "design_value": True},
        ),
    ],
)
def test_datum_less_than_turned_off(datum_lit, comparison, overrides, outcome, fields):
    obs_a = datum_lit.model_copy(update=overrides)()
    obs_b = datum_lit()
    result = comparison(obs_a, obs_b)
    assert result.is_satisfied == outcome
    assert_result_fields(result, fields)
