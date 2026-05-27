from typing import Annotated, Union

from pydantic import Field

from pytao.constraints.observables.base import (
    CheckResult,
    Comparison,
    ComparisonResult,
    IsClose,
    IsCloseResult,
    IsLess,
    IsLessResult,
    LatticeObservable,
    LiteralObservable,
    Observable,
    Observation,
    ResultT,
)
from pytao.constraints.observables.datum import (
    DataSource,
    DatumIsClose,
    DatumIsCloseResult,
    DatumLessThan,
    DatumLessThanResult,
    DatumLiteral,
    DatumObservable,
    DatumObservation,
    EvalPoint,
)
from pytao.constraints.observables.ele import (
    EleIsClose,
    EleIsCloseResult,
    EleLessThan,
    EleLessThanResult,
    EleLiteral,
    EleMaxObservable,
    EleMinObservable,
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

observable_types = Annotated[
    Union[
        EleObservable,
        EleMaxObservable,
        EleMinObservable,
        DatumObservable,
        EleLiteral,
        DatumLiteral,
    ],
    Field(discriminator="obs_type"),
]
observation_types = Annotated[
    Union[EleObservation, DatumObservation], Field(discriminator="obs_type")
]
constraint_result_types = Annotated[
    Union[
        EleIsCloseResult,
        DatumIsCloseResult,
        EleLessThanResult,
        DatumLessThanResult,
        IsCloseResult,
        IsLessResult,
    ],
    Field(discriminator="result_type"),
]

__all__ = [
    "BmagTwissComparison",
    "CheckResult",
    "Comparison",
    "ComparisonResult",
    "DataSource",
    "DatumIsClose",
    "DatumIsCloseResult",
    "DatumLessThan",
    "DatumLessThanResult",
    "DatumLiteral",
    "DatumObservable",
    "DatumObservation",
    "DummyTwissComparison",
    "EleIsClose",
    "EleIsCloseResult",
    "EleLessThan",
    "EleLessThanResult",
    "EleLiteral",
    "EleMaxObservable",
    "EleMinObservable",
    "EleObservable",
    "EleObservation",
    "EvalPoint",
    "IsClose",
    "IsCloseResult",
    "IsLess",
    "IsLessResult",
    "LatticeObservable",
    "LiteralObservable",
    "Observable",
    "Observation",
    "ResultT",
    "TolComparison",
    "TwissComparisonMethod",
    "constraint_result_types",
    "observable_types",
    "observation_types",
    "twiss_comparison_types",
]
