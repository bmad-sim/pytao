from typing import Annotated, Union

from pydantic import Field

from .base import (
    CheckResult,
    Comparison,
    ComparisonResult,
    IsClose,
    IsLess,
    LatticeObservable,
    LiteralObservable,
    Observable,
    Observation,
)
from .datum import (
    DataSource,
    DatumIsClose,
    DatumLessThan,
    DatumLiteral,
    DatumObservable,
    DatumObservation,
    EvalPoint,
)
from .ele import (
    EleIsClose,
    EleLessThan,
    EleLiteral,
    EleReduceObservable,
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
        EleReduceObservable,
        DatumObservable,
        EleLiteral,
        DatumLiteral,
    ],
    Field(discriminator="type"),
]
AnyObservation = Annotated[
    Union[EleObservation, DatumObservation], Field(discriminator="type")
]
AnyComparison = Annotated[
    Union[
        EleIsClose,
        DatumIsClose,
        EleLessThan,
        DatumLessThan,
    ],
    Field(discriminator="type"),
]
__all__ = [
    "AnyObservable",
    "AnyObservation",
    "AnyTwissComparison",
    "BmagTwissComparison",
    "CheckResult",
    "Comparison",
    "ComparisonResult",
    "DataSource",
    "DatumIsClose",
    "DatumLessThan",
    "DatumLiteral",
    "DatumObservable",
    "DatumObservation",
    "EleIsClose",
    "EleLessThan",
    "EleLiteral",
    "EleReduceObservable",
    "EleObservable",
    "EleObservation",
    "EvalPoint",
    "IsClose",
    "IsLess",
    "LatticeObservable",
    "LiteralObservable",
    "Observable",
    "Observation",
    "TolComparison",
    "TwissComparisonMethod",
]
