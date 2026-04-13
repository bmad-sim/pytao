from pydantic import BaseModel


class LatticeResult(BaseModel):
    lattice_file: str | None
    init_file: str | None
    loaded: bool
    error: str | None = None
    load_time: float


class UnittestResults(BaseModel):
    lattices: list[LatticeResult]
