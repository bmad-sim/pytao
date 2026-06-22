from typing import Annotated, Union

from pydantic import Field

from .base import (
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
from .datum import (
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
from .ele import (
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
from .twiss import (
    AnyTwissComparison,
    BmagTwissComparison,
    TwissComparisonMethod,
)

AnyObservable = Annotated[
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
AnyObservation = Annotated[
    Union[EleObservation, DatumObservation], Field(discriminator="obs_type")
]
AnyComparisonResult = Annotated[
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
    "AnyComparisonResult",
    "AnyObservable",
    "AnyObservation",
    "AnyTwissComparison",
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
]
