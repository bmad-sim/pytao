from pydantic import BaseModel, Field

from pytao.constraints.observables.ele import EleIsClose, EleObservable


class LatticeConfig(BaseModel):
    lattice_file: str | None = None
    init_file: str | None = None


class ElementPair(BaseModel):
    ele_a: EleObservable
    ele_b: EleObservable
    comparison: EleIsClose = Field(default_factory=EleIsClose)


class UnittestConfig(BaseModel):
    lattices: dict[str, LatticeConfig] = Field(default_factory=dict, description="Mapping from unique lattice identifier to lattice loading information")
    ele_equality: list[ElementPair] = Field(default_factory=list, description="Element pairs to compare across lattices")
