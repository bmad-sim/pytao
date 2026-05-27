from abc import abstractmethod
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from pytao.check.observables import (
    ComparisonResult,
    DatumIsClose,
    DatumIsCloseResult,
    DatumLessThan,
    DatumLessThanResult,
    DatumLiteral,
    DatumObservable,
    EleIsClose,
    EleIsCloseResult,
    EleLessThan,
    EleLessThanResult,
    EleLiteral,
    EleMaxObservable,
    EleMinObservable,
    EleObservable,
    IsClose,
    IsLess,
    Observable,
    Observation,
)
from pytao.startup import TaoStartup


EleObservables = Annotated[
    Union[EleObservable, EleMaxObservable, EleMinObservable, EleLiteral],
    Field(discriminator="obs_type"),
]

DatumObservables = Annotated[
    Union[DatumObservable, DatumLiteral],
    Field(discriminator="obs_type"),
]


class Constraint(BaseModel):
    """Abstract base for all constraint types."""

    description: str = Field(default="", description="Short one-line name used on labels")
    comment: str = Field(
        default="", description="Detailed description or notes about the constraint"
    )

    @property
    @abstractmethod
    def required_observables(self) -> frozenset[Observable]: ...

    @abstractmethod
    def is_satisfied(
        self, observations: dict[Observable, Observation]
    ) -> ComparisonResult: ...

    @abstractmethod
    def error_result(self, error: str) -> ComparisonResult: ...


class EqualityConstraint(Constraint):
    """Base for constraints that use an IsClose comparison operator."""

    comparison: IsClose


class IsLessConstraint(Constraint):
    """Base for constraints that use an IsLess comparison operator."""

    comparison: IsLess


class EleIsCloseConstraint(EqualityConstraint):
    constraint_type: Literal["ele_eq"] = "ele_eq"
    obs_a: EleObservables
    obs_b: EleObservables
    comparison: EleIsClose = EleIsClose()

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset((self.obs_a, self.obs_b))

    def is_satisfied(self, observations: dict[Observable, Observation]) -> EleIsCloseResult:
        return self.comparison(observations[self.obs_a], observations[self.obs_b])

    def error_result(self, error: str) -> EleIsCloseResult:
        return EleIsCloseResult(error=error)


class EleLessThanConstraint(IsLessConstraint):
    constraint_type: Literal["ele_lt"] = "ele_lt"
    obs_a: EleObservables
    obs_b: EleObservables
    comparison: EleLessThan = EleLessThan()

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset((self.obs_a, self.obs_b))

    def is_satisfied(self, observations: dict[Observable, Observation]) -> EleLessThanResult:
        return self.comparison(observations[self.obs_a], observations[self.obs_b])

    def error_result(self, error: str) -> EleLessThanResult:
        return EleLessThanResult(error=error)


class DatumIsCloseConstraint(EqualityConstraint):
    constraint_type: Literal["datum_eq"] = "datum_eq"
    obs_a: DatumObservables
    obs_b: DatumObservables
    comparison: DatumIsClose = DatumIsClose()

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset((self.obs_a, self.obs_b))

    def is_satisfied(self, observations: dict[Observable, Observation]) -> DatumIsCloseResult:
        return self.comparison(observations[self.obs_a], observations[self.obs_b])

    def error_result(self, error: str) -> DatumIsCloseResult:
        return DatumIsCloseResult(error=error)


class DatumLessThanConstraint(IsLessConstraint):
    constraint_type: Literal["datum_lt"] = "datum_lt"
    obs_a: DatumObservables
    obs_b: DatumObservables
    comparison: DatumLessThan = DatumLessThan()

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset((self.obs_a, self.obs_b))

    def is_satisfied(self, observations: dict[Observable, Observation]) -> DatumLessThanResult:
        return self.comparison(observations[self.obs_a], observations[self.obs_b])

    def error_result(self, error: str) -> DatumLessThanResult:
        return DatumLessThanResult(error=error)


AnyConstraint = Annotated[
    Union[
        EleIsCloseConstraint,
        EleLessThanConstraint,
        DatumIsCloseConstraint,
        DatumLessThanConstraint,
    ],
    Field(discriminator="constraint_type"),
]


class ConstraintsConfig(BaseModel):
    lattices: dict[str, TaoStartup] = Field(
        default_factory=dict,
        description="Mapping from unique lattice identifier to lattice loading information",
    )
    constraints: list[AnyConstraint] = Field(
        default_factory=list, description="Constraints to check across lattices"
    )
