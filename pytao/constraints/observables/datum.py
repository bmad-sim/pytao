from __future__ import annotations

from enum import Enum
from typing import Literal


from pydantic import computed_field

from pytao import Tao
from pytao.errors import TaoCommandError
from pytao.constraints.observables.base import (
    CheckResult,
    IsClose,
    IsCloseResult,
    IsLess,
    IsLessResult,
    LatticeObservable,
    LiteralObservable,
    Observation,
)
from pytao.constraints.observables.ele import TolComparison


class DataSource(str, Enum):
    """Data source for a Tao datum evaluation."""

    lat = "lat"
    data = "data"
    var = "var"
    beam = "beam"


class EvalPoint(str, Enum):
    """Element evaluation point for a Tao datum."""

    beginning = "beginning"
    center = "center"
    end = "end"


_D2_NAME = "_pytao_tmp"
_D1_NAME = "v"
_DATUM_NAME = f"{_D2_NAME}.{_D1_NAME}[1]"


class DatumObservation(Observation):
    """Observation containing the result of a Tao datum evaluation.

    Attributes
    ----------
    obs_type : str
        Discriminator literal. Always ``"datum"``.
    model_value : float
        Model value of the datum.
    design_value : float
        Design value of the datum.
    """

    obs_type: Literal["datum"] = "datum"
    model_value: float
    design_value: float


class DatumIsCloseResult(IsCloseResult):
    """Result of a DatumIsClose comparison with per-field check results.

    Each field is ``None`` if the corresponding comparison was not run.

    Attributes
    ----------
    result_type : str
        Discriminator literal. Always ``"datum_is_close"``.
    model_value : CheckResult or None
        Model value comparison result.
    design_value : CheckResult or None
        Design value comparison result.
    """

    result_type: Literal["datum_is_close"] = "datum_is_close"
    model_value: CheckResult | None = None
    design_value: CheckResult | None = None

    @computed_field
    @property
    def is_satisfied(self) -> bool:
        if not super().is_satisfied:
            return False
        ran = [r for r in [self.model_value, self.design_value] if r is not None]
        return all(ran) if ran else True


class DatumIsClose(IsClose[DatumObservation]):
    """IsClose operator comparing two DatumObservation instances.

    Set a field to ``None`` to skip that comparison.

    Attributes
    ----------
    model_value_test : TolComparison or None
        Comparison for the model value.
    design_value_test : TolComparison or None
        Comparison for the design value.
    """

    model_value_test: TolComparison | None = TolComparison()
    design_value_test: TolComparison | None = None

    def compare(self, obja: DatumObservation, objb: DatumObservation) -> DatumIsCloseResult:
        model_value = None
        design_value = None

        if self.model_value_test is not None:
            model_value = self.model_value_test(obja.model_value, objb.model_value)
        if self.design_value_test is not None:
            design_value = self.design_value_test(obja.design_value, objb.design_value)

        return DatumIsCloseResult(model_value=model_value, design_value=design_value)


class DatumLessThanResult(IsLessResult):
    """Result of a DatumLessThan comparison with per-field less-than check results.

    Each field is ``None`` if the corresponding component was not checked.

    Attributes
    ----------
    result_type : str
        Discriminator literal. Always ``"datum_is_less"``.
    model_value : CheckResult or None
        Model value comparison result.
    design_value : CheckResult or None
        Design value comparison result.
    """

    result_type: Literal["datum_is_less"] = "datum_is_less"
    model_value: CheckResult | None = None
    design_value: CheckResult | None = None

    @computed_field
    @property
    def is_satisfied(self) -> bool:
        if not super().is_satisfied:
            return False
        ran = [r for r in [self.model_value, self.design_value] if r is not None]
        return all(ran) if ran else True


class DatumLessThan(IsLess[DatumObservation]):
    """Component-wise less-than comparison between two DatumObservations.

    Set a field to ``True`` to enable the less-than check for that component.

    Attributes
    ----------
    model_value : bool
        Check model value.
    design_value : bool
        Check design value.
    """

    model_value: bool = True
    design_value: bool = False

    def _check(self, va: float, vb: float) -> CheckResult:
        passed = va < vb
        return CheckResult(
            passed=passed, detail="" if passed else f"a={va:.6g} not < b={vb:.6g}"
        )

    def compare(self, obja: DatumObservation, objb: DatumObservation) -> DatumLessThanResult:
        model_value = (
            self._check(obja.model_value, objb.model_value) if self.model_value else None
        )
        design_value = (
            self._check(obja.design_value, objb.design_value) if self.design_value else None
        )
        return DatumLessThanResult(model_value=model_value, design_value=design_value)


class DatumLiteral(LiteralObservable[DatumObservation]):
    """Literal datum observable with user-specified model and design values.

    Attributes
    ----------
    obs_type : str
        Discriminator literal. Always ``"datum_literal"``.
    model_value : float
        Model value for the produced observation.
    design_value : float
        Design value for the produced observation.
    """

    obs_type: Literal["datum_literal"] = "datum_literal"
    model_value: float
    design_value: float

    @property
    def label(self) -> str:
        return "literal"

    def _make_observation(self) -> DatumObservation:
        return DatumObservation(model_value=self.model_value, design_value=self.design_value)


class DatumObservable(LatticeObservable[DatumObservation]):
    """Observable that creates a temporary Tao datum and evaluates it.

    Attributes
    ----------
    obs_type : str
        Discriminator literal. Always ``"datum"``.
    data_type : str
        Tao datum data type (e.g. ``"orbit.x"``).
    ele_name : str
        Name of the element at which the datum is evaluated.
    ele_start_name : str
        Start element name for range datums.
    ele_ref_name : str
        Reference element name.
    eval_point : EvalPoint
        Where along the element to evaluate (beginning, center, or end).
    data_source : DataSource
        Source of the data (lat, data, var, or beam).
    """

    obs_type: Literal["datum"] = "datum"
    data_type: str
    ele_name: str
    ele_start_name: str = ""
    ele_ref_name: str = ""
    eval_point: EvalPoint = EvalPoint.end
    data_source: DataSource = DataSource.lat

    @property
    def label(self) -> str:
        return f"{self.lattice_id}[{self.data_type}@{self.ele_name}]"

    def _make_observation(self, tao: Tao) -> DatumObservation:
        tao.data_d2_create(_D2_NAME, "1", f"{_D1_NAME}^^1^^1")
        tao.datum_create(
            _DATUM_NAME,
            self.data_type,
            ele_name=self.ele_name,
            ele_start_name=self.ele_start_name,
            ele_ref_name=self.ele_ref_name,
            eval_point=self.eval_point.value,
            data_source=self.data_source.value,
        )
        tao.data_set_design_value()
        try:
            result = tao.data_d_array(_D2_NAME, _D1_NAME)[0]

            if not result["exists"]:
                raise TaoCommandError(
                    f"DatumObservable Failed: Could not create datum. {self!r}"
                )

            return DatumObservation(
                model_value=result["model_value"],
                design_value=result["design_value"],
            )
        finally:
            tao.data_d2_destroy(_D2_NAME)
