from abc import abstractmethod
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from pytao.constraints.observables import (
    ConstraintResult,
    DatumIsClose,
    DatumIsCloseResult,
    DatumLessThan,
    DatumLessThanResult,
    DatumLiteral,
    DatumObservable,
    DatumObservation,
    EleIsClose,
    EleIsCloseResult,
    EleLessThan,
    EleLessThanResult,
    EleLiteral,
    EleMaxObservable,
    EleMinObservable,
    EleObservable,
    EleObservation,
    IsClose,
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
    ) -> ConstraintResult: ...

    @abstractmethod
    def error_result(self, error: str) -> ConstraintResult: ...


class EqualityConstraint(Constraint):
    """Base for constraints that use an IsClose comparison operator."""

    comparison: IsClose


class EleIsCloseConstraint(EqualityConstraint):
    constraint_type: Literal["ele_eq"] = "ele_eq"
    obs_a: EleObservables
    obs_b: EleObservables
    comparison: EleIsClose = Field(default_factory=EleObservation.is_close_cls)

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset((self.obs_a, self.obs_b))

    def is_satisfied(self, observations: dict[Observable, Observation]) -> EleIsCloseResult:
        return self.comparison(observations[self.obs_a], observations[self.obs_b])

    def error_result(self, error: str) -> EleIsCloseResult:
        return EleIsCloseResult(is_close=False, error=error)


class EleLessThanConstraint(Constraint):
    constraint_type: Literal["ele_lt"] = "ele_lt"
    obs_a: EleObservables
    obs_b: EleObservables
    comparison: EleLessThan = Field(default_factory=EleObservation.is_less_cls)

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset((self.obs_a, self.obs_b))

    def is_satisfied(self, observations: dict[Observable, Observation]) -> EleLessThanResult:
        return self.comparison(observations[self.obs_a], observations[self.obs_b])

    def error_result(self, error: str) -> EleLessThanResult:
        return EleLessThanResult(is_less=False, error=error)


class DatumIsCloseConstraint(EqualityConstraint):
    constraint_type: Literal["datum_eq"] = "datum_eq"
    obs_a: DatumObservables
    obs_b: DatumObservables
    comparison: DatumIsClose = Field(default_factory=DatumObservation.is_close_cls)

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset((self.obs_a, self.obs_b))

    def is_satisfied(self, observations: dict[Observable, Observation]) -> DatumIsCloseResult:
        return self.comparison(observations[self.obs_a], observations[self.obs_b])

    def error_result(self, error: str) -> DatumIsCloseResult:
        return DatumIsCloseResult(is_close=False, error=error)


class DatumLessThanConstraint(Constraint):
    constraint_type: Literal["datum_lt"] = "datum_lt"
    obs_a: DatumObservables
    obs_b: DatumObservables
    comparison: DatumLessThan = Field(default_factory=DatumObservation.is_less_cls)

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset((self.obs_a, self.obs_b))

    def is_satisfied(self, observations: dict[Observable, Observation]) -> DatumLessThanResult:
        return self.comparison(observations[self.obs_a], observations[self.obs_b])

    def error_result(self, error: str) -> DatumLessThanResult:
        return DatumLessThanResult(is_less=False, error=error)


constraint_types = Annotated[
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
    constraints: list[constraint_types] = Field(
        default_factory=list, description="Constraints to check across lattices"
    )
