from pydantic import BaseModel, Field


class LatticeConfig(BaseModel):
    lattice_file: str | None = None
    init_file: str | None = None


class UnittestConfig(BaseModel):
    lattices: dict[str, LatticeConfig] = Field(default_factory=list, description="Mapping from unique lattice identifier to lattice loading information")
