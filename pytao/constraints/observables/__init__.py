from typing import Annotated

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
    EleObservable,
    EleObservation,
    EleReduceObservable,
    TolComparison,
)
from .twiss import (
    AnyTwissComparison,
    BmagTwissComparison,
    TwissComparisonMethod,
)

AnyObservable = Annotated[
    EleObservable | EleReduceObservable | DatumObservable | EleLiteral | DatumLiteral,
    Field(discriminator="type"),
]
AnyObservation = Annotated[EleObservation | DatumObservation, Field(discriminator="type")]
AnyComparison = Annotated[
    EleIsClose | DatumIsClose | EleLessThan | DatumLessThan,
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
    "EleObservable",
    "EleObservation",
    "EleReduceObservable",
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
