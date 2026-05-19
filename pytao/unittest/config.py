from pydantic import BaseModel, Field

from pytao.unittest.observables.ele import EleIsClose


class LatticeConfig(BaseModel):
    lattice_file: str | None = None
    init_file: str | None = None


class ElementPair(BaseModel):
    lattice_a_id: str
    element_a: str
    lattice_b_id: str
    element_b: str
    comparison: EleIsClose = Field(default_factory=EleIsClose)


class UnittestConfig(BaseModel):
    lattices: dict[str, LatticeConfig] = Field(default_factory=dict, description="Mapping from unique lattice identifier to lattice loading information")
    pair_equality: list[ElementPair] = Field(default_factory=list, description="Element pairs to compare across lattices")
