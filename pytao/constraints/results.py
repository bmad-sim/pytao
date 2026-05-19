from typing import Literal

from pydantic import BaseModel

from pytao.constraints.observables.base import CheckResult
from pytao.constraints.observables.ele import EleIsCloseResult


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
    twiss_a: CheckResult | None = None
    twiss_b: CheckResult | None = None
    eta_x: CheckResult | None = None
    etap_x: CheckResult | None = None
    eta_y: CheckResult | None = None
    etap_y: CheckResult | None = None
    ref_energy: CheckResult | None = None
    p0c: CheckResult | None = None
    orbit: CheckResult | None = None
    floor_x: CheckResult | None = None
    floor_y: CheckResult | None = None
    floor_z: CheckResult | None = None

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
                status = "PASS" if result.passed else "FAIL"
                detail = f"  {result.detail}" if not result.passed and result.detail else ""
                print(f"    {name:<{width}}  {status}{detail}")
        if self.error:
            for line in self.error.splitlines():
                print(f"    {line}")


class PairEqualityResult(BaseModel):
    lattice_a_id: str
    element_a: str
    lattice_b_id: str
    element_b: str
    result: EleIsCloseResult


class LatticeResult(BaseModel):
    lattice_file: str | None
    init_file: str | None
    loaded: bool
    error: str | None = None
    load_time: float


class UnittestResults(BaseModel):
    lattices: dict[str, LatticeResult]
    ele_equality: list[PairEqualityResult]
