from abc import abstractmethod
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Discriminator, Field, Tag

from pytao.constraints.observables import ConstraintResult, IsClose, LatticeObservable, Observable, Observation
from pytao.constraints.observables.datum import DatumIsClose, DatumIsCloseResult, DatumLessThan, DatumLessThanResult, DatumLiteral, DatumObservable
from pytao.constraints.observables.ele import EleIsClose, EleIsCloseResult, EleLessThan, EleLessThanResult, EleLiteral, EleMaxObservable, EleMinObservable, EleObservable
from pytao.startup import TaoStartup


def _has_lattice_id(v) -> bool:
    if isinstance(v, dict):
        return "lattice_id" in v
    return isinstance(v, LatticeObservable)


def _ele_discriminator(v) -> str:
    return "observable" if _has_lattice_id(v) else "literal"


def _datum_discriminator(v) -> str:
    return "observable" if _has_lattice_id(v) else "literal"


EleObservableOrLiteral = Annotated[
    Union[Annotated[EleObservable, Tag("observable")], Annotated[EleLiteral, Tag("literal")]],
    Discriminator(_ele_discriminator),
]

EleMaxObservableOrLiteral = Annotated[
    Union[Annotated[EleMaxObservable, Tag("observable")], Annotated[EleLiteral, Tag("literal")]],
    Discriminator(_ele_discriminator),
]

EleMinObservableOrLiteral = Annotated[
    Union[Annotated[EleMinObservable, Tag("observable")], Annotated[EleLiteral, Tag("literal")]],
    Discriminator(_ele_discriminator),
]

DatumObservableOrLiteral = Annotated[
    Union[Annotated[DatumObservable, Tag("observable")], Annotated[DatumLiteral, Tag("literal")]],
    Discriminator(_datum_discriminator),
]


class Constraint(BaseModel):
    """Abstract base for all constraint types."""

    description: str = Field(default="", description="Short one-line name used on labels")
    comment: str = Field(default="", description="Detailed description or notes about the constraint")

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
        return frozenset((self.ele_a, self.ele_b))

    def is_satisfied(self, observations: dict[Observable, Observation]) -> EleIsCloseResult:
        return self.comparison(observations[self.ele_a], observations[self.ele_b])


class DatumPairEquality(EqualityConstraint):
    constraint_type: Literal["datum_eq"] = "datum_eq"
    datum_a: DatumObservableOrLiteral
    datum_b: DatumObservableOrLiteral
    comparison: DatumIsClose = Field(default_factory=DatumIsClose)

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset((self.datum_a, self.datum_b))

    def is_satisfied(self, observations: dict[Observable, Observation]) -> DatumIsCloseResult:
        return self.comparison(observations[self.datum_a], observations[self.datum_b])


class ElementPairLessThan(Constraint):
    constraint_type: Literal["ele_lt"] = "ele_lt"
    ele_a: EleObservableOrLiteral
    ele_b: EleObservableOrLiteral
    comparison: EleLessThan = Field(default_factory=EleLessThan)

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset((self.ele_a, self.ele_b))

    def is_satisfied(self, observations: dict[Observable, Observation]) -> EleLessThanResult:
        return self.comparison(observations[self.ele_a], observations[self.ele_b])


class DatumPairLessThan(Constraint):
    constraint_type: Literal["datum_lt"] = "datum_lt"
    datum_a: DatumObservableOrLiteral
    datum_b: DatumObservableOrLiteral
    comparison: DatumLessThan = Field(default_factory=DatumLessThan)

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset((self.datum_a, self.datum_b))

    def is_satisfied(self, observations: dict[Observable, Observation]) -> DatumLessThanResult:
        return self.comparison(observations[self.datum_a], observations[self.datum_b])


class ElementMaxEquality(EqualityConstraint):
    constraint_type: Literal["ele_max_eq"] = "ele_max_eq"
    ele_a: EleMaxObservableOrLiteral
    ele_b: EleMaxObservableOrLiteral
    comparison: EleIsClose = Field(default_factory=EleIsClose)

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset((self.ele_a, self.ele_b))

    def is_satisfied(self, observations: dict[Observable, Observation]) -> EleIsCloseResult:
        return self.comparison(observations[self.ele_a], observations[self.ele_b])


class ElementMinEquality(EqualityConstraint):
    constraint_type: Literal["ele_min_eq"] = "ele_min_eq"
    ele_a: EleMinObservableOrLiteral
    ele_b: EleMinObservableOrLiteral
    comparison: EleIsClose = Field(default_factory=EleIsClose)

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset((self.ele_a, self.ele_b))

    def is_satisfied(self, observations: dict[Observable, Observation]) -> EleIsCloseResult:
        return self.comparison(observations[self.ele_a], observations[self.ele_b])


class ElementMaxLessThan(Constraint):
    constraint_type: Literal["ele_max_lt"] = "ele_max_lt"
    ele_a: EleMaxObservableOrLiteral
    ele_b: EleMaxObservableOrLiteral
    comparison: EleLessThan = Field(default_factory=EleLessThan)

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset((self.ele_a, self.ele_b))

    def is_satisfied(self, observations: dict[Observable, Observation]) -> EleLessThanResult:
        return self.comparison(observations[self.ele_a], observations[self.ele_b])


class ElementMinLessThan(Constraint):
    constraint_type: Literal["ele_min_lt"] = "ele_min_lt"
    ele_a: EleMinObservableOrLiteral
    ele_b: EleMinObservableOrLiteral
    comparison: EleLessThan = Field(default_factory=EleLessThan)

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset((self.ele_a, self.ele_b))

    def is_satisfied(self, observations: dict[Observable, Observation]) -> EleLessThanResult:
        return self.comparison(observations[self.ele_a], observations[self.ele_b])


constraint_types = Annotated[
    Union[
        ElementPairEquality, DatumPairEquality,
        ElementPairLessThan, DatumPairLessThan,
        ElementMaxEquality, ElementMinEquality,
        ElementMaxLessThan, ElementMinLessThan,
    ],
    Field(discriminator="constraint_type"),
]


class ConstraintsConfig(BaseModel):
    lattices: dict[str, TaoStartup] = Field(default_factory=dict, description="Mapping from unique lattice identifier to lattice loading information")
    constraints: list[constraint_types] = Field(default_factory=list, description="Constraints to check across lattices")
