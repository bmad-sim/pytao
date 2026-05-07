from pydantic import BaseModel, Field

from pytao.unittest.tests import unit_test_types


class LatticeConfig(BaseModel):
    lattice_file: str | None = None
    init_file: str | None = None


class UnittestConfig(BaseModel):
    lattices: dict[str, LatticeConfig] = Field(default_factory=dict, description="Mapping from unique lattice identifier to lattice loading information")
    tests: list[unit_test_types] = Field(default_factory=list, description="Unit tests to run against the lattices")
