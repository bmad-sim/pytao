from abc import abstractmethod
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from pytao.constraints.observables import IsCloseResult, Observable, Observation
from pytao.constraints.observables.datum import DatumIsClose, DatumIsCloseResult, DatumObservable, DatumObservation, DatumLiteral
from pytao.constraints.observables.ele import EleIsClose, EleIsCloseResult, EleObservable, EleObservation, EleLiteral


class EqualityConstraint(BaseModel):
    """Abstract base for equality constraints between two observables."""

    @property
    @abstractmethod
    def obs_a(self) -> Observable: ...

    @property
    @abstractmethod
    def obs_b(self) -> Observable: ...

    @abstractmethod
    def compare(self, obs_a: Observation, obs_b: Observation) -> IsCloseResult: ...


class ElementPairEquality(EqualityConstraint):
    constraint_type: Literal["ele"] = "ele"
    ele_a: EleObservable
    ele_b: EleObservable
    comparison: EleIsClose = Field(default_factory=EleIsClose)

    @property
    def obs_a(self) -> EleObservable:
        return self.ele_a

    @property
    def obs_b(self) -> EleObservable:
        return self.ele_b

    def compare(self, obs_a: Observation, obs_b: Observation) -> EleIsCloseResult:
        return self.comparison(obs_a, obs_b)


class DatumPairEquality(EqualityConstraint):
    constraint_type: Literal["datum"] = "datum"
    datum_a: DatumObservable
    datum_b: DatumObservable
    comparison: DatumIsClose = Field(default_factory=DatumIsClose)

    @property
    def obs_a(self) -> DatumObservable:
        return self.datum_a

    @property
    def obs_b(self) -> DatumObservable:
        return self.datum_b

    def compare(self, obs_a: Observation, obs_b: Observation) -> DatumIsCloseResult:
        return self.comparison(obs_a, obs_b)


class EleLiteralEquality(EqualityConstraint):
    constraint_type: Literal["ele_literal"] = "ele_literal"
    ele: EleObservable
    expected: EleLiteral
    comparison: EleIsClose = Field(default_factory=EleIsClose)

    @property
    def obs_a(self) -> EleObservable:
        return self.ele

    @property
    def obs_b(self) -> EleObservable:
        return self.ele

    def compare(self, obs_a: Observation, obs_b: Observation) -> EleIsCloseResult:
        if not isinstance(obs_a, EleObservation):
            raise TypeError(f"expected EleObservation, got {type(obs_a)}")
        return self.comparison(obs_a, self.expected.to_observation())


class DatumLiteralEquality(EqualityConstraint):
    constraint_type: Literal["datum_literal"] = "datum_literal"
    datum: DatumObservable
    expected: DatumLiteral
    comparison: DatumIsClose = Field(default_factory=DatumIsClose)

    @property
    def obs_a(self) -> DatumObservable:
        return self.datum

    @property
    def obs_b(self) -> DatumObservable:
        return self.datum

    def compare(self, obs_a: Observation, obs_b: Observation) -> DatumIsCloseResult:
        if not isinstance(obs_a, DatumObservation):
            raise TypeError(f"expected DatumObservation, got {type(obs_a)}")
        literal = self.expected.to_observation(obs_a)
        return self.comparison(obs_a, literal)


equality_constraint_types = Annotated[Union[ElementPairEquality, EleLiteralEquality, DatumPairEquality, DatumLiteralEquality], Field(discriminator="constraint_type")]


class LatticeConfig(BaseModel):
    lattice_file: str | None = None
    init_file: str | None = None


class ConstraintsConfig(BaseModel):
    lattices: dict[str, LatticeConfig] = Field(default_factory=dict, description="Mapping from unique lattice identifier to lattice loading information")
    equality_constraints: list[equality_constraint_types] = Field(default_factory=list, description="Equality constraints to check across lattices")
