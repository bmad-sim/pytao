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
    def obs_a(self) -> Observable | None: ...

    @property
    @abstractmethod
    def obs_b(self) -> Observable | None: ...

    @abstractmethod
    def compare(self, obs_a: Observation | None, obs_b: Observation | None) -> IsCloseResult: ...


class ElementPairEquality(EqualityConstraint):
    constraint_type: Literal["ele"] = "ele"
    ele_a: EleObservableOrLiteral
    ele_b: EleObservableOrLiteral
    comparison: EleIsClose = Field(default_factory=EleIsClose)

    @property
    def obs_a(self) -> EleObservable | None:
        return self.ele_a if isinstance(self.ele_a, EleObservable) else None

    @property
    def obs_b(self) -> EleObservable | None:
        return self.ele_b if isinstance(self.ele_b, EleObservable) else None

    def compare(self, obs_a: Observation | None, obs_b: Observation | None) -> EleIsCloseResult:
        if isinstance(self.ele_a, EleLiteral):
            obs_a = self.ele_a.to_observation()
        if isinstance(self.ele_b, EleLiteral):
            obs_b = self.ele_b.to_observation()
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
    def obs_a(self) -> DatumObservable | None:
        return self.datum_a if isinstance(self.datum_a, DatumObservable) else None

    @property
    def obs_b(self) -> DatumObservable | None:
        return self.datum_b if isinstance(self.datum_b, DatumObservable) else None

    def compare(self, obs_a: Observation | None, obs_b: Observation | None) -> DatumIsCloseResult:
        if isinstance(self.datum_a, DatumLiteral):
            if not isinstance(obs_b, DatumObservation):
                raise TypeError(f"expected DatumObservation for obs_b, got {type(obs_b)}")
            obs_a = self.datum_a.to_observation(obs_b)
        if isinstance(self.datum_b, DatumLiteral):
            if not isinstance(obs_a, DatumObservation):
                raise TypeError(f"expected DatumObservation for obs_a, got {type(obs_a)}")
            obs_b = self.datum_b.to_observation(obs_a)
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
