from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class TestResult(BaseModel):
    result_type: Literal["TestResult"] = "TestResult"
    test_type: str
    description: str
    passed: bool
    error: str | None = None


class PairMatchResult(TestResult):
    result_type: Literal["PairMatchResult"] = "PairMatchResult"
    twiss_a: bool | None = None
    twiss_b: bool | None = None
    eta_x: bool | None = None
    etap_x: bool | None = None
    eta_y: bool | None = None
    etap_y: bool | None = None
    ref_energy: bool | None = None
    p0c: bool | None = None
    orbit: bool | None = None
    floor_x: bool | None = None
    floor_y: bool | None = None
    floor_z: bool | None = None


test_result_types = Annotated[Union[PairMatchResult, TestResult], Field(discriminator="result_type")]


class LatticeResult(BaseModel):
    lattice_file: str | None
    init_file: str | None
    loaded: bool
    error: str | None = None
    load_time: float


class UnittestResults(BaseModel):
    lattices: dict[str, LatticeResult]
    tests: list[test_result_types]
