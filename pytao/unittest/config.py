from pydantic import BaseModel, Field


class LatticeConfig(BaseModel):
    lattice_file: str | None = None
    init_file: str | None = None


class UnittestConfig(BaseModel):
    lattices: list[LatticeConfig] = Field(default_factory=list)
