from pydantic import BaseModel


class LatticeResult(BaseModel):
    lattice_file: str | None
    init_file: str | None
    loaded: bool
    error: str | None = None
    load_time: float


class TestResult(BaseModel):
    test_type: str
    passed: bool
    error: str | None = None


class UnittestResults(BaseModel):
    lattices: dict[str, LatticeResult]
    tests: list[TestResult]
