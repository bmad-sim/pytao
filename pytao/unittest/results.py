from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class TestResult(BaseModel):
    result_type: Literal["TestResult"] = "TestResult"
    test_type: str
    description: str
    passed: bool
    error: str | None = None

    def print_failure_detail(self) -> None:
        if self.error:
            for line in self.error.splitlines():
                print(f"    {line}")


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

    def print_failure_detail(self) -> None:
        checks = {
            "twiss_a": self.twiss_a,
            "twiss_b": self.twiss_b,
            "eta_x": self.eta_x,
            "etap_x": self.etap_x,
            "eta_y": self.eta_y,
            "etap_y": self.etap_y,
            "ref_energy": self.ref_energy,
            "p0c": self.p0c,
            "orbit": self.orbit,
            "floor_x": self.floor_x,
            "floor_y": self.floor_y,
            "floor_z": self.floor_z,
        }
        ran = {name: result for name, result in checks.items() if result is not None}
        if ran:
            width = max(len(name) for name in ran)
            for name, result in ran.items():
                status = "PASS" if result else "FAIL"
                print(f"    {name:<{width}}  {status}")
        if self.error:
            for line in self.error.splitlines():
                print(f"    {line}")


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
