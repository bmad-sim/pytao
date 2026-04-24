"""
Unit tests for :mod:`pytao.optimize.problem`.

These tests do not require a built Tao binary — they drive the code with a
FakeTao that returns scripted responses matching the structured output of
``var_general``, ``var_v_array``, ``var``, and the ``data_d*`` family.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from pytao.optimize import DatumInfo, TaoOptimizationProblem, VariableInfo
from pytao.optimize.problem import _finite_limit


# ---- FakeTao ------------------------------------------------------------


@dataclass
class FakeTao:
    """A stub that satisfies the introspection-subset of the ``_TaoLike`` protocol."""

    var_general_rows: list[dict[str, Any]] = field(default_factory=list)
    var_v_array_rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    var_detail: dict[str, dict[str, Any]] = field(default_factory=dict)
    d2_names: list[str] = field(default_factory=list)
    d2_names_by_universe: dict[int, list[str]] = field(default_factory=dict)
    d1_arrays: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    d_arrays: dict[tuple[str, str], list[dict[str, Any]]] = field(default_factory=dict)
    d_arrays_by_universe: dict[tuple[int, str, str], list[dict[str, Any]]] = field(
        default_factory=dict
    )
    default_universe: int = 1

    def var_general(self, *, raises: bool = True) -> list[dict[str, Any]]:  # noqa: ARG002
        return list(self.var_general_rows)

    def var_v_array(self, v1_var: str, *, raises: bool = True):  # noqa: ARG002
        return list(self.var_v_array_rows.get(v1_var, []))

    def var(self, var: str, *, raises: bool = True):  # noqa: ARG002
        return dict(self.var_detail[var])

    def data_d2_array(self, ix_uni: str = "", *, raises: bool = True):  # noqa: ARG002
        if ix_uni and int(ix_uni) in self.d2_names_by_universe:
            return list(self.d2_names_by_universe[int(ix_uni)])
        return list(self.d2_names)

    def data_d1_array(self, d2_datum: str, *, raises: bool = True):  # noqa: ARG002
        # d2_datum comes in as "{ix_uni}@{d2_name}"
        _, _, d2_name = d2_datum.partition("@")
        return list(self.d1_arrays.get(d2_name, []))

    def data_d_array(
        self,
        d2_name: str,
        d1_name: str,
        *,
        ix_uni: str = "",  # noqa: ARG002
        raises: bool = True,  # noqa: ARG002
    ):
        if ix_uni:
            key = (int(ix_uni), d2_name, d1_name)
            if key in self.d_arrays_by_universe:
                return list(self.d_arrays_by_universe[key])
        return list(self.d_arrays.get((d2_name, d1_name), []))

    def tao_global(self, *, raises: bool = True) -> dict[str, Any]:  # noqa: ARG002
        return {"default_universe": self.default_universe}


def _make_var_detail(
    model: float,
    low: float,
    high: float,
    step: float = 1e-4,
    weight: float = 0.0,
    merit_type: str = "target",
    ele_name: str = "Q1",
    attrib_name: str = "k1",
) -> dict[str, Any]:
    return {
        "model_value": model,
        "low_lim": low,
        "high_lim": high,
        "step": step,
        "weight": weight,
        "merit_type": merit_type,
        "ele_name": ele_name,
        "attrib_name": attrib_name,
    }


def _simple_problem_tao() -> FakeTao:
    """
    A realistic 3-variable, 2-datum FakeTao that several tests share.

    - v1 ``quad`` has 3 entries, 2 active (useit_opt True) and 1 inactive.
    - d2 ``twiss`` has a d1 ``end`` with 2 target datums, both active.
    """
    tao = FakeTao()
    tao.var_general_rows = [{"name": "quad", "line": "", "lbound": 1, "ubound": 3}]
    tao.var_v_array_rows = {
        "quad": [
            {
                "ix_v1": 1,
                "var_attrib_name": "k1",
                "meas_value": 0.0,
                "model_value": 0.5,
                "design_value": 0.5,
                "useit_opt": True,
                "good_user": True,
                "weight": 0.0,
            },
            {
                "ix_v1": 2,
                "var_attrib_name": "k1",
                "meas_value": 0.0,
                "model_value": -0.3,
                "design_value": -0.3,
                "useit_opt": True,
                "good_user": True,
                "weight": 0.0,
            },
            {
                "ix_v1": 3,
                "var_attrib_name": "k1",
                "meas_value": 0.0,
                "model_value": 0.1,
                "design_value": 0.1,
                "useit_opt": False,
                "good_user": False,
                "weight": 0.0,
            },
        ]
    }
    tao.var_detail = {
        "quad[1]": _make_var_detail(0.5, -5.0, 5.0, ele_name="Q1"),
        "quad[2]": _make_var_detail(-0.3, -1e30, 1e30, ele_name="Q2"),
        "quad[3]": _make_var_detail(0.1, -5.0, 5.0, ele_name="Q3"),
    }
    tao.d2_names = ["twiss"]
    tao.d1_arrays = {"twiss": [{"name": "end"}]}
    tao.d_arrays = {
        ("twiss", "end"): [
            {
                "ix_d1": 1,
                "data_type": "beta.a",
                "merit_type": "target",
                "ele_ref_name": "",
                "ele_start_name": "",
                "ele_name": "END",
                "meas_value": 12.5,
                "model_value": 10.0,
                "design_value": 10.0,
                "useit_opt": True,
                "useit_plot": True,
                "good_user": True,
                "weight": 10.0,
                "exists": True,
            },
            {
                "ix_d1": 2,
                "data_type": "alpha.a",
                "merit_type": "target",
                "ele_ref_name": "",
                "ele_start_name": "",
                "ele_name": "END",
                "meas_value": -1.0,
                "model_value": 0.0,
                "design_value": 0.0,
                "useit_opt": True,
                "useit_plot": True,
                "good_user": True,
                "weight": 100.0,
                "exists": True,
            },
        ]
    }
    return tao


# ---- extraction --------------------------------------------------------


def test_problem_extracts_only_active_variables():
    tao = _simple_problem_tao()
    p = TaoOptimizationProblem.from_tao(tao)
    assert p.n_var == 2
    assert [v.name for v in p.variables] == ["quad[1]", "quad[2]"]
    assert p.variables[0].ele_name == "Q1"
    assert p.variables[1].ele_name == "Q2"


def test_problem_extracts_only_active_datums():
    tao = _simple_problem_tao()
    tao.d_arrays[("twiss", "end")][1]["useit_opt"] = False
    p = TaoOptimizationProblem.from_tao(tao)
    assert p.n_data == 1
    assert p.datums[0].name == "twiss.end[1]"


def test_variable_preserves_merit_and_attribute_metadata():
    tao = _simple_problem_tao()
    tao.var_detail["quad[1]"]["merit_type"] = "limit"
    p = TaoOptimizationProblem.from_tao(tao)
    assert p.variables[0].merit_type == "limit"
    assert p.variables[0].attrib_name == "k1"
    assert p.variables[0].step == pytest.approx(1e-4)


def test_datum_preserves_merit_type_and_weight():
    tao = _simple_problem_tao()
    tao.d_arrays[("twiss", "end")][0]["merit_type"] = "max"
    p = TaoOptimizationProblem.from_tao(tao)
    assert p.datums[0].merit_type == "max"
    assert p.datums[0].weight == 10.0


def test_unbounded_sentinels_become_inf():
    tao = _simple_problem_tao()
    p = TaoOptimizationProblem.from_tao(tao)
    assert math.isinf(p.variables[1].low_lim) and p.variables[1].low_lim < 0
    assert math.isinf(p.variables[1].high_lim) and p.variables[1].high_lim > 0


def test_finite_limits_preserved():
    tao = _simple_problem_tao()
    p = TaoOptimizationProblem.from_tao(tao)
    assert p.variables[0].low_lim == -5.0
    assert p.variables[0].high_lim == 5.0


def test_x0_matches_model_values():
    tao = _simple_problem_tao()
    p = TaoOptimizationProblem.from_tao(tao)
    np.testing.assert_array_equal(p.x0, np.array([0.5, -0.3]))


def test_x0_returns_copy_not_reference():
    """Mutating the returned x0 must not affect the problem's internal state."""
    tao = _simple_problem_tao()
    p = TaoOptimizationProblem.from_tao(tao)
    x = p.x0
    x[0] = 999.0
    np.testing.assert_array_equal(p.x0, np.array([0.5, -0.3]))


def test_bounds_and_bounds_array_shapes():
    tao = _simple_problem_tao()
    p = TaoOptimizationProblem.from_tao(tao)
    assert p.bounds == [(-5.0, 5.0), (-math.inf, math.inf)]
    lb, ub = p.bounds_array
    assert lb.shape == (2,) and ub.shape == (2,)
    np.testing.assert_array_equal(lb, [-5.0, -math.inf])
    np.testing.assert_array_equal(ub, [5.0, math.inf])


def test_weights_vector():
    tao = _simple_problem_tao()
    p = TaoOptimizationProblem.from_tao(tao)
    np.testing.assert_array_equal(p.weights, [10.0, 100.0])


def test_variable_names_order_stable():
    tao = _simple_problem_tao()
    p = TaoOptimizationProblem.from_tao(tao)
    assert p.variable_names == ["quad[1]", "quad[2]"]


def test_multiple_v1_groups_are_all_enumerated():
    tao = _simple_problem_tao()
    tao.var_general_rows.append({"name": "bend", "line": "", "lbound": 1, "ubound": 1})
    tao.var_v_array_rows["bend"] = [
        {
            "ix_v1": 1,
            "var_attrib_name": "angle",
            "meas_value": 0.0,
            "model_value": 0.05,
            "design_value": 0.05,
            "useit_opt": True,
            "good_user": True,
            "weight": 0.0,
        }
    ]
    tao.var_detail["bend[1]"] = _make_var_detail(
        0.05, -math.inf, math.inf, ele_name="B1", attrib_name="angle"
    )
    p = TaoOptimizationProblem.from_tao(tao)
    assert [v.v1_name for v in p.variables] == ["quad", "quad", "bend"]
    assert p.n_var == 3


def test_multiple_d1_and_d2_groups_enumerated():
    tao = _simple_problem_tao()
    tao.d1_arrays["twiss"].append({"name": "max"})
    tao.d_arrays[("twiss", "max")] = [
        {
            "ix_d1": 1,
            "data_type": "beta.a",
            "merit_type": "max",
            "ele_ref_name": "",
            "ele_start_name": "Q1",
            "ele_name": "END",
            "meas_value": 100.0,
            "model_value": 20.0,
            "design_value": 20.0,
            "useit_opt": True,
            "useit_plot": True,
            "good_user": True,
            "weight": 5.0,
            "exists": True,
        }
    ]
    p = TaoOptimizationProblem.from_tao(tao)
    assert p.n_data == 3
    assert {d.d1_name for d in p.datums} == {"end", "max"}


# ---- warnings ----------------------------------------------------------


def test_warns_when_no_active_variables(caplog):
    tao = _simple_problem_tao()
    for row in tao.var_v_array_rows["quad"]:
        row["useit_opt"] = False
    with caplog.at_level("WARNING", logger="pytao.optimize.problem"):
        TaoOptimizationProblem.from_tao(tao)
    assert any("no active variables" in rec.message for rec in caplog.records)


def test_warns_when_no_active_datums(caplog):
    tao = _simple_problem_tao()
    for row in tao.d_arrays[("twiss", "end")]:
        row["useit_opt"] = False
    with caplog.at_level("WARNING", logger="pytao.optimize.problem"):
        TaoOptimizationProblem.from_tao(tao)
    assert any("no active datums" in rec.message for rec in caplog.records)


# ---- weight validation -------------------------------------------------


def test_negative_datum_weight_is_rejected():
    tao = _simple_problem_tao()
    tao.d_arrays[("twiss", "end")][0]["weight"] = -1.0
    with pytest.raises(ValueError, match="negative weight"):
        TaoOptimizationProblem.from_tao(tao)


def test_negative_variable_weight_is_rejected():
    tao = _simple_problem_tao()
    tao.var_detail["quad[1]"]["weight"] = -2.0
    with pytest.raises(ValueError, match="negative weight"):
        TaoOptimizationProblem.from_tao(tao)


def test_zero_weight_is_accepted():
    tao = _simple_problem_tao()
    # Default weights are 0.0 on variables; make sure that's fine.
    p = TaoOptimizationProblem.from_tao(tao)
    assert p.n_var == 2


# ---- _finite_limit low-level sanity -----------------------------------


def test_finite_limit_preserves_within_threshold():
    assert _finite_limit(5.0, -1) == 5.0
    assert _finite_limit(-5.0, -1) == -5.0
    assert _finite_limit(1e19, 1) == pytest.approx(1e19)


def test_finite_limit_translates_sentinels_to_inf():
    assert math.isinf(_finite_limit(-1e30, -1)) and _finite_limit(-1e30, -1) < 0
    assert math.isinf(_finite_limit(1e30, 1)) and _finite_limit(1e30, 1) > 0


# ---- dataclass immutability -------------------------------------------


def test_variable_info_is_frozen():
    v = VariableInfo(
        name="q[1]",
        v1_name="q",
        index=1,
        initial_value=0.0,
        low_lim=-1.0,
        high_lim=1.0,
        step=1e-4,
        weight=0.0,
        merit_type="target",
        ele_name="Q1",
        attrib_name="k1",
    )
    with pytest.raises(Exception):  # FrozenInstanceError
        v.initial_value = 1.0  # type: ignore[misc]


def test_datum_info_is_frozen():
    d = DatumInfo(
        name="d.end[1]",
        d2_name="d",
        d1_name="end",
        ix_d1=1,
        data_type="x",
        merit_type="target",
        meas_value=0.0,
        model_value=0.0,
        design_value=0.0,
        weight=1.0,
    )
    with pytest.raises(Exception):
        d.weight = 2.0  # type: ignore[misc]


# ---- read-only snapshot contract ---------------------------------------


def test_variables_is_immutable_tuple():
    tao = _simple_problem_tao()
    p = TaoOptimizationProblem.from_tao(tao)
    assert isinstance(p.variables, tuple)
    with pytest.raises(AttributeError):
        p.variables.append(p.variables[0])  # type: ignore[attr-defined]


def test_datums_is_immutable_tuple():
    tao = _simple_problem_tao()
    p = TaoOptimizationProblem.from_tao(tao)
    assert isinstance(p.datums, tuple)
    with pytest.raises(AttributeError):
        p.datums.append(p.datums[0])  # type: ignore[attr-defined]


# ---- universe selection -----------------------------------------------


def test_universe_defaults_to_live_tao_global():
    """When universe=None, snapshot pins to Tao's live default_universe."""
    tao = _simple_problem_tao()
    tao.default_universe = 3
    # Put the only datum under universe 3 specifically; universe 1 has none.
    tao.d2_names_by_universe = {3: ["twiss"]}
    tao.d_arrays_by_universe = {(3, "twiss", "end"): tao.d_arrays[("twiss", "end")]}
    tao.d_arrays = {}  # universe=1 sees nothing
    p = TaoOptimizationProblem.from_tao(tao)
    assert p.universe == 3
    assert p.n_data == 2


def test_explicit_universe_overrides_default():
    tao = _simple_problem_tao()
    tao.default_universe = 1
    # universe=2 sees a different set of datums.
    tao.d2_names_by_universe = {2: ["twiss"]}
    tao.d_arrays_by_universe = {
        (
            2,
            "twiss",
            "end",
        ): [
            {
                "ix_d1": 1,
                "data_type": "beta.a",
                "merit_type": "target",
                "ele_ref_name": "",
                "ele_start_name": "",
                "ele_name": "END",
                "meas_value": 5.0,
                "model_value": 4.0,
                "design_value": 4.0,
                "useit_opt": True,
                "useit_plot": True,
                "good_user": True,
                "weight": 1.0,
                "exists": True,
            }
        ]
    }
    p = TaoOptimizationProblem.from_tao(tao, universe=2)
    assert p.universe == 2
    assert p.n_data == 1
    assert p.datums[0].meas_value == 5.0


def test_universe_falls_back_when_tao_global_missing(caplog):
    """A Tao-like object without tao_global() falls back to universe=1."""

    class MinimalFakeTao(FakeTao):
        tao_global = None  # type: ignore[assignment]

    tao = MinimalFakeTao()
    simple = _simple_problem_tao()
    tao.var_general_rows = simple.var_general_rows
    tao.var_v_array_rows = simple.var_v_array_rows
    tao.var_detail = simple.var_detail
    tao.d2_names = simple.d2_names
    tao.d1_arrays = simple.d1_arrays
    tao.d_arrays = simple.d_arrays
    with caplog.at_level("WARNING", logger="pytao.optimize.problem"):
        p = TaoOptimizationProblem.from_tao(tao)
    assert p.universe == 1
    assert any("does not expose tao_global" in rec.message for rec in caplog.records)
