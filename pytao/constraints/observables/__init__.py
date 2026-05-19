from typing import Annotated, Union

from pydantic import Field

from pytao.constraints.observables.base import CheckResult, IsClose, IsCloseResult, Observable, Observation
from pytao.constraints.observables.ele import (
    EleIsClose,
    EleIsCloseResult,
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

observable_types = Annotated[Union[EleObservable], Field(discriminator="obs_type")]
observation_types = Annotated[Union[EleObservation], Field(discriminator="obs_type")]

__all__ = [
    "BmagTwissComparison",
    "CheckResult",
    "DummyTwissComparison",
    "EleIsClose",
    "EleIsCloseResult",
    "EleObservable",
    "EleObservation",
    "IsClose",
    "IsCloseResult",
    "Observable",
    "Observation",
    "TolComparison",
    "TwissComparisonMethod",
    "observable_types",
    "observation_types",
    "twiss_comparison_types",
]
