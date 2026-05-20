from typing import Annotated, Union

from pydantic import Field

from pytao.constraints.observables.base import CheckResult, Comparison, ConstraintResult, IsClose, IsCloseResult, IsLess, IsLessResult, Observable, Observation
from pytao.constraints.observables.datum import (
    DatumIsClose,
    DatumIsCloseResult,
    DatumLessThan,
    DatumLessThanResult,
    DatumObservable,
    DatumObservation,
)
from pytao.constraints.observables.ele import (
    EleIsClose,
    EleIsCloseResult,
    EleLessThan,
    EleLessThanResult,
    EleObservable,
    EleObservation,
    TolComparison,
)
from pytao.constraints.observables.twiss import (
    BmagTwissComparison,
    DummyTwissComparison,
    TwissComparisonMethod,
    twiss_comparison_types,
)

observable_types = Annotated[Union[EleObservable, DatumObservable], Field(discriminator="obs_type")]
observation_types = Annotated[Union[EleObservation, DatumObservation], Field(discriminator="obs_type")]
constraint_result_types = Annotated[Union[EleIsCloseResult, DatumIsCloseResult, EleLessThanResult, DatumLessThanResult, IsCloseResult, IsLessResult], Field(discriminator="result_type")]

__all__ = [
    "BmagTwissComparison",
    "CheckResult",
    "Comparison",
    "ConstraintResult",
    "DatumIsClose",
    "DatumIsCloseResult",
    "DatumLessThan",
    "DatumLessThanResult",
    "DatumObservable",
    "DatumObservation",
    "DummyTwissComparison",
    "EleIsClose",
    "EleIsCloseResult",
    "EleLessThan",
    "EleLessThanResult",
    "EleObservable",
    "EleObservation",
    "IsClose",
    "IsCloseResult",
    "IsLess",
    "IsLessResult",
    "Observable",
    "Observation",
    "TolComparison",
    "TwissComparisonMethod",
    "constraint_result_types",
    "observable_types",
    "observation_types",
    "twiss_comparison_types",
]
