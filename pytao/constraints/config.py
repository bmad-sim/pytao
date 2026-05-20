from abc import abstractmethod
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Discriminator, Field, Tag

from pytao.constraints.observables import IsCloseResult, Observable, Observation
from pytao.constraints.observables.datum import DatumIsClose, DatumIsCloseResult, DatumLiteral, DatumObservable, DatumObservation
from pytao.constraints.observables.ele import EleIsClose, EleIsCloseResult, EleLiteral, EleObservable, EleObservation


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


class EqualityConstraint(BaseModel):
    """Abstract base for equality constraints between two observables."""

    @property
    @abstractmethod
    def required_observables(self) -> frozenset[Observable]: ...

    @abstractmethod
    def compare(self, observations: dict[Observable, Observation]) -> IsCloseResult: ...


class ElementPairEquality(EqualityConstraint):
    constraint_type: Literal["ele"] = "ele"
    ele_a: EleObservableOrLiteral
    ele_b: EleObservableOrLiteral
    comparison: EleIsClose = Field(default_factory=EleIsClose)

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset(x for x in (self.ele_a, self.ele_b) if isinstance(x, EleObservable))

    def compare(self, observations: dict[Observable, Observation]) -> EleIsCloseResult:
        obs_a = observations[self.ele_a] if isinstance(self.ele_a, EleObservable) else self.ele_a.to_observation()
        obs_b = observations[self.ele_b] if isinstance(self.ele_b, EleObservable) else self.ele_b.to_observation()
        if not isinstance(obs_a, EleObservation):
            raise TypeError(f"expected EleObservation for obs_a, got {type(obs_a)}")
        if not isinstance(obs_b, EleObservation):
            raise TypeError(f"expected EleObservation for obs_b, got {type(obs_b)}")
        return self.comparison(obs_a, obs_b)


class DatumPairEquality(EqualityConstraint):
    constraint_type: Literal["datum"] = "datum"
    datum_a: DatumObservableOrLiteral
    datum_b: DatumObservableOrLiteral
    comparison: DatumIsClose = Field(default_factory=DatumIsClose)

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset(x for x in (self.datum_a, self.datum_b) if isinstance(x, DatumObservable))

    def compare(self, observations: dict[Observable, Observation]) -> DatumIsCloseResult:
        obs_a = observations[self.datum_a] if isinstance(self.datum_a, DatumObservable) else self.datum_a.to_observation()
        obs_b = observations[self.datum_b] if isinstance(self.datum_b, DatumObservable) else self.datum_b.to_observation()
        if not isinstance(obs_a, DatumObservation):
            raise TypeError(f"expected DatumObservation for obs_a, got {type(obs_a)}")
        if not isinstance(obs_b, DatumObservation):
            raise TypeError(f"expected DatumObservation for obs_b, got {type(obs_b)}")
        return self.comparison(obs_a, obs_b)


equality_constraint_types = Annotated[
    Union[ElementPairEquality, DatumPairEquality],
    Field(discriminator="constraint_type"),
]


class LatticeConfig(BaseModel):
    lattice_file: str | None = None
    init_file: str | None = None


class ConstraintsConfig(BaseModel):
    lattices: dict[str, LatticeConfig] = Field(default_factory=dict, description="Mapping from unique lattice identifier to lattice loading information")
    equality_constraints: list[equality_constraint_types] = Field(default_factory=list, description="Equality constraints to check across lattices")
