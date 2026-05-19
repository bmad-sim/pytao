from abc import abstractmethod
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from pytao.constraints.observables import IsCloseResult, Observable, Observation
from pytao.constraints.observables.ele import EleIsClose, EleIsCloseResult, EleObservable


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


class ElementPair(EqualityConstraint):
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


equality_constraint_types = Annotated[Union[ElementPair], Field(discriminator="constraint_type")]


class LatticeConfig(BaseModel):
    lattice_file: str | None = None
    init_file: str | None = None


class UnittestConfig(BaseModel):
    lattices: dict[str, LatticeConfig] = Field(default_factory=dict, description="Mapping from unique lattice identifier to lattice loading information")
    equality_constraints: list[equality_constraint_types] = Field(default_factory=list, description="Equality constraints to check across lattices")
