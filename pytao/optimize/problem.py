"""
Read-only introspection of a Tao optimization setup.

:class:`TaoOptimizationProblem` reads a live :class:`~pytao.Tao` instance and
reports the set of active optimization variables and datums, along with their
bounds, weights, and merit types. This is the foundation for Python-driven
optimization, but the current module does not mutate Tao state or evaluate the
merit function — those capabilities are the subject of follow-up work.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)


# Finite sentinel used when Tao reports an unbounded side. Tao stores
# ``low_lim`` / ``high_lim`` as plain Fortran reals and uses very large values
# (±1e30) to mean "no limit". scipy's bounded solvers accept ±inf, so we
# translate anything past this threshold to ±inf.
_UNBOUNDED_THRESHOLD = 1e20


class _TaoLike(Protocol):
    """Structural type covering the bits of the Tao API we use for introspection."""

    def var_general(self, *, raises: bool = True) -> list[dict[str, Any]]: ...
    def var_v_array(self, v1_var: str, *, raises: bool = True) -> list[dict[str, Any]]: ...
    def var(self, var: str, *, raises: bool = True) -> dict[str, Any]: ...
    def data_d2_array(self, ix_uni: str = "", *, raises: bool = True) -> list[str]: ...
    def data_d1_array(self, d2_datum: str, *, raises: bool = True) -> list[dict[str, Any]]: ...
    def data_d_array(
        self, d2_name: str, d1_name: str, *, ix_uni: str = "", raises: bool = True
    ) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class VariableInfo:
    """
    Description of a single Tao optimization variable.

    Attributes
    ----------
    name : str
        Fully-qualified variable name in the form ``"v1_name[ix]"``, suitable
        for use with Tao's ``set var`` command syntax.
    v1_name : str
        Name of the v1 variable group (e.g. ``"quad"``).
    index : int
        Index within ``v1_name`` (matches Tao's ``ix_v1``).
    initial_value : float
        ``model_value`` at the time the problem was constructed.
    low_lim : float
        Lower bound; ``-inf`` if Tao reports no limit.
    high_lim : float
        Upper bound; ``+inf`` if Tao reports no limit.
    step : float
        Step size suggested by Tao (useful for finite-difference fallbacks
        in downstream optimizers).
    weight : float
        Merit weight applied to limit-type variable violations.
    merit_type : str
        ``"target"`` (default), ``"limit"``, etc. — mirrors
        ``tao_var_struct%merit_type``.
    ele_name : str
        Name of the lattice element the variable operates on, if any.
    attrib_name : str
        Attribute being varied (e.g. ``"k1"``).
    """

    name: str
    v1_name: str
    index: int
    initial_value: float
    low_lim: float
    high_lim: float
    step: float
    weight: float
    merit_type: str
    ele_name: str
    attrib_name: str


@dataclass(frozen=True)
class DatumInfo:
    """
    Description of a single Tao datum that contributes to the merit.

    Attributes
    ----------
    name : str
        Fully-qualified datum name: ``"d2_name.d1_name[ix_d1]"``.
    d2_name, d1_name : str
        Datum group names.
    ix_d1 : int
        Index within the d1 array.
    data_type : str
        Underlying observable (e.g. ``"beta.a"``).
    merit_type : str
        Determines how Tao computes the datum's merit contribution. One of
        ``"target"``, ``"min"``, ``"max"``, ``"abs_min"``, ``"abs_max"``,
        ``"average"``, ``"rms"``, ``"integral"``, ``"max-min"``.
    meas_value, model_value, design_value : float
        Target value, current value, and the design reference.
    weight : float
        Merit weight (applied as ``weight * delta**2``).
    """

    name: str
    d2_name: str
    d1_name: str
    ix_d1: int
    data_type: str
    merit_type: str
    meas_value: float
    model_value: float
    design_value: float
    weight: float


@dataclass
class TaoOptimizationProblem:
    """
    A snapshot of a Tao optimization problem, exposed for Python inspection.

    Build one of these from a running :class:`~pytao.Tao` instance. The
    constructor enumerates the active variables and datums (those with
    ``useit_opt == True``), translates Tao's ``±1e30`` "no limit" sentinels to
    ``±inf``, and validates that weights are non-negative.

    Parameters
    ----------
    tao : Tao
        A live Tao instance. Any object satisfying the :class:`_TaoLike`
        protocol also works, which is how the unit tests exercise the logic
        without a real Tao binary.
    universe : int, default=1
        Universe index to query for datums. Tao allows multi-universe
        optimization but most setups use a single universe.

    Notes
    -----
    Tao's merit function is

    .. math::

        M = \\sum_i w_i \\, \\Delta_i^2 + \\sum_j w_j \\, \\Delta_j^2

    where the first sum is over active datums and the second over active
    variables (variables only contribute when ``merit_type == 'limit'``).
    Constraints declared as ``merit_type = 'max'`` / ``'min'`` on datums are
    therefore *soft* — they're folded into the merit via their weight. This
    first release only reports the configuration; mutation and merit
    evaluation will follow in subsequent slices.
    """

    tao: _TaoLike
    universe: int = 1
    variables: list[VariableInfo] = field(init=False)
    datums: list[DatumInfo] = field(init=False)

    def __post_init__(self) -> None:
        self.variables = _collect_active_variables(self.tao)
        self.datums = _collect_active_datums(self.tao, self.universe)
        self._x0 = np.array([v.initial_value for v in self.variables], dtype=float)
        _validate_weights(self.variables, self.datums)
        if not self.variables:
            logger.warning(
                "TaoOptimizationProblem built with no active variables "
                "(check good_user / good_opt flags)."
            )
        if not self.datums:
            logger.warning(
                "TaoOptimizationProblem built with no active datums "
                "(check good_user / good_opt flags)."
            )

    # ---- static views ----------------------------------------------------

    @property
    def n_var(self) -> int:
        """Number of active optimization variables."""
        return len(self.variables)

    @property
    def n_data(self) -> int:
        """Number of active datums contributing to the merit."""
        return len(self.datums)

    @property
    def x0(self) -> np.ndarray:
        """Initial variable vector captured at construction."""
        return self._x0.copy()

    @property
    def variable_names(self) -> list[str]:
        """Fully-qualified names of the active variables, in vector order."""
        return [v.name for v in self.variables]

    @property
    def bounds(self) -> list[tuple[float, float]]:
        """
        Per-variable ``(low, high)`` tuples, using ``±inf`` for unbounded.

        Suitable for :func:`scipy.optimize.minimize` bounded methods once a
        downstream adapter starts consuming the problem.
        """
        return [(v.low_lim, v.high_lim) for v in self.variables]

    @property
    def bounds_array(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Bounds reshaped for :func:`scipy.optimize.least_squares`.

        Returns
        -------
        (lb, ub) : tuple of ndarray
            Each of shape ``(n_var,)``. Unbounded sides are ``±inf``.
        """
        lb = np.array([v.low_lim for v in self.variables], dtype=float)
        ub = np.array([v.high_lim for v in self.variables], dtype=float)
        return lb, ub

    @property
    def weights(self) -> np.ndarray:
        """Per-datum merit weights, in datum vector order."""
        return np.array([d.weight for d in self.datums], dtype=float)


# ---- internal helpers ----------------------------------------------------


def _finite_limit(value: float, sign: int) -> float:
    """Translate Tao's ±1e30 "no limit" sentinels to ±inf."""
    if abs(value) > _UNBOUNDED_THRESHOLD:
        return math.copysign(math.inf, sign)
    return float(value)


def _validate_weights(variables: list[VariableInfo], datums: list[DatumInfo]) -> None:
    """
    Reject negative weights at the boundary.

    Tao's merit is :math:`\\sum w \\Delta^2` with :math:`w \\geq 0` by design.
    Downstream consumers will take ``sqrt(weight)``; catching negatives here
    produces a clear error instead of silently returning NaNs later.
    """
    for d in datums:
        if d.weight < 0:
            raise ValueError(
                f"Datum {d.name!r} has negative weight {d.weight!r}; "
                "Tao's merit requires non-negative weights."
            )
    for v in variables:
        if v.weight < 0:
            raise ValueError(
                f"Variable {v.name!r} has negative weight {v.weight!r}; "
                "Tao's merit requires non-negative weights."
            )


def _collect_active_variables(tao: _TaoLike) -> list[VariableInfo]:
    """Enumerate every ``useit_opt == True`` variable from Tao."""
    results: list[VariableInfo] = []
    for v1 in tao.var_general():
        v1_name = v1["name"]
        rows = tao.var_v_array(v1_name)
        rows_sorted = sorted(rows, key=lambda r: int(r["ix_v1"]))
        for row in rows_sorted:
            if not row.get("useit_opt", False):
                continue
            ix_v1 = int(row["ix_v1"])
            full_name = f"{v1_name}[{ix_v1}]"
            detail = tao.var(full_name)
            low = _finite_limit(float(detail.get("low_lim", -math.inf)), -1)
            high = _finite_limit(float(detail.get("high_lim", math.inf)), 1)
            results.append(
                VariableInfo(
                    name=full_name,
                    v1_name=v1_name,
                    index=ix_v1,
                    initial_value=float(detail.get("model_value", row["model_value"])),
                    low_lim=low,
                    high_lim=high,
                    step=float(detail.get("step", 0.0)),
                    weight=float(detail.get("weight", row.get("weight", 0.0))),
                    merit_type=str(detail.get("merit_type", "target")),
                    ele_name=str(detail.get("ele_name", "")),
                    attrib_name=str(detail.get("attrib_name", row.get("var_attrib_name", ""))),
                )
            )
    return results


def _collect_active_datums(tao: _TaoLike, universe: int) -> list[DatumInfo]:
    """Enumerate every ``useit_opt == True`` datum for ``universe``."""
    results: list[DatumInfo] = []
    for d2 in tao.data_d2_array(str(universe)):
        d2_name = d2 if isinstance(d2, str) else d2.get("name", "")
        if not d2_name:
            continue
        d2_ref = f"{universe}@{d2_name}"
        for d1 in tao.data_d1_array(d2_ref):
            d1_name = d1["name"] if isinstance(d1, dict) else d1
            rows = tao.data_d_array(d2_name, d1_name, ix_uni=str(universe))
            rows_sorted = sorted(rows, key=lambda r: int(r["ix_d1"]))
            for row in rows_sorted:
                if not row.get("useit_opt", False):
                    continue
                ix_d1 = int(row["ix_d1"])
                results.append(
                    DatumInfo(
                        name=f"{d2_name}.{d1_name}[{ix_d1}]",
                        d2_name=d2_name,
                        d1_name=d1_name,
                        ix_d1=ix_d1,
                        data_type=str(row.get("data_type", "")),
                        merit_type=str(row.get("merit_type", "target")),
                        meas_value=float(row["meas_value"]),
                        model_value=float(row["model_value"]),
                        design_value=float(row["design_value"]),
                        weight=float(row.get("weight", 0.0)),
                    )
                )
    return results
