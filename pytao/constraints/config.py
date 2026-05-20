from abc import abstractmethod
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Discriminator, Field, Tag

from pytao.constraints.observables import ConstraintResult, IsClose, Observable, Observation
from pytao.constraints.observables.datum import DatumIsClose, DatumIsCloseResult, DatumLessThan, DatumLessThanResult, DatumLiteral, DatumObservable, DatumObservation
from pytao.constraints.observables.ele import EleIsClose, EleIsCloseResult, EleLessThan, EleLessThanResult, EleLiteral, EleObservable, EleObservation
from pytao.startup import TaoStartup


def _has_lattice_id(v) -> bool:
    if isinstance(v, dict):
        return "lattice_id" in v
    return isinstance(v, Observable)


def _ele_discriminator(v) -> str:
    return "observable" if _has_lattice_id(v) else "literal"


def _datum_discriminator(v) -> str:
    return "observable" if _has_lattice_id(v) else "literal"


EleObservableOrLiteral = Annotated[
    Union[Annotated[EleObservable, Tag("observable")], Annotated[EleLiteral, Tag("literal")]],
    Discriminator(_ele_discriminator),
]

DatumObservableOrLiteral = Annotated[
    Union[Annotated[DatumObservable, Tag("observable")], Annotated[DatumLiteral, Tag("literal")]],
    Discriminator(_datum_discriminator),
]


class Constraint(BaseModel):
    """Abstract base for all constraint types."""

    comment: str = ""

    @property
    @abstractmethod
    def required_observables(self) -> frozenset[Observable]: ...

    @abstractmethod
    def is_satisfied(self, observations: dict[Observable, Observation]) -> ConstraintResult: ...


class EqualityConstraint(Constraint):
    """Base for constraints that use an IsClose comparison operator."""

    comparison: IsClose


class ElementPairEquality(EqualityConstraint):
    constraint_type: Literal["ele_eq"] = "ele_eq"
    ele_a: EleObservableOrLiteral
    ele_b: EleObservableOrLiteral
    comparison: EleIsClose = Field(default_factory=EleIsClose)

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset(x for x in (self.ele_a, self.ele_b) if isinstance(x, EleObservable))

    def is_satisfied(self, observations: dict[Observable, Observation]) -> EleIsCloseResult:
        obs_a = observations[self.ele_a] if isinstance(self.ele_a, EleObservable) else self.ele_a.to_observation()
        obs_b = observations[self.ele_b] if isinstance(self.ele_b, EleObservable) else self.ele_b.to_observation()
        if not isinstance(obs_a, EleObservation):
            raise TypeError(f"expected EleObservation for obs_a, got {type(obs_a)}")
        if not isinstance(obs_b, EleObservation):
            raise TypeError(f"expected EleObservation for obs_b, got {type(obs_b)}")
        return self.comparison(obs_a, obs_b)


class DatumPairEquality(EqualityConstraint):
    constraint_type: Literal["datum_eq"] = "datum_eq"
    datum_a: DatumObservableOrLiteral
    datum_b: DatumObservableOrLiteral
    comparison: DatumIsClose = Field(default_factory=DatumIsClose)

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset(x for x in (self.datum_a, self.datum_b) if isinstance(x, DatumObservable))

    def is_satisfied(self, observations: dict[Observable, Observation]) -> DatumIsCloseResult:
        obs_a = observations[self.datum_a] if isinstance(self.datum_a, DatumObservable) else self.datum_a.to_observation()
        obs_b = observations[self.datum_b] if isinstance(self.datum_b, DatumObservable) else self.datum_b.to_observation()
        if not isinstance(obs_a, DatumObservation):
            raise TypeError(f"expected DatumObservation for obs_a, got {type(obs_a)}")
        if not isinstance(obs_b, DatumObservation):
            raise TypeError(f"expected DatumObservation for obs_b, got {type(obs_b)}")
        return self.comparison(obs_a, obs_b)


class ElementPairLessThan(Constraint):
    constraint_type: Literal["ele_lt"] = "ele_lt"
    ele_a: EleObservableOrLiteral
    ele_b: EleObservableOrLiteral
    comparison: EleLessThan = Field(default_factory=EleLessThan)

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset(x for x in (self.ele_a, self.ele_b) if isinstance(x, EleObservable))

    def is_satisfied(self, observations: dict[Observable, Observation]) -> EleLessThanResult:
        obs_a = observations[self.ele_a] if isinstance(self.ele_a, EleObservable) else self.ele_a.to_observation()
        obs_b = observations[self.ele_b] if isinstance(self.ele_b, EleObservable) else self.ele_b.to_observation()
        if not isinstance(obs_a, EleObservation):
            raise TypeError(f"expected EleObservation for obs_a, got {type(obs_a)}")
        if not isinstance(obs_b, EleObservation):
            raise TypeError(f"expected EleObservation for obs_b, got {type(obs_b)}")
        return self.comparison(obs_a, obs_b)


class DatumPairLessThan(Constraint):
    constraint_type: Literal["datum_lt"] = "datum_lt"
    datum_a: DatumObservableOrLiteral
    datum_b: DatumObservableOrLiteral
    comparison: DatumLessThan = Field(default_factory=DatumLessThan)

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset(x for x in (self.datum_a, self.datum_b) if isinstance(x, DatumObservable))

    def is_satisfied(self, observations: dict[Observable, Observation]) -> DatumLessThanResult:
        obs_b = observations[self.datum_b] if isinstance(self.datum_b, DatumObservable) else self.datum_b.to_observation()
        obs_a = observations[self.datum_a] if isinstance(self.datum_a, DatumObservable) else self.datum_a.to_observation()
        if not isinstance(obs_a, DatumObservation):
            raise TypeError(f"expected DatumObservation for obs_a, got {type(obs_a)}")
        if not isinstance(obs_b, DatumObservation):
            raise TypeError(f"expected DatumObservation for obs_b, got {type(obs_b)}")
        return self.comparison(obs_a, obs_b)


constraint_types = Annotated[
    Union[ElementPairEquality, DatumPairEquality, ElementPairLessThan, DatumPairLessThan],
    Field(discriminator="constraint_type"),
]


class ConstraintsConfig(BaseModel):
    lattices: dict[str, TaoStartup] = Field(default_factory=dict, description="Mapping from unique lattice identifier to lattice loading information")
    constraints: list[constraint_types] = Field(default_factory=list, description="Constraints to check across lattices")
