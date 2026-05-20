from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from pytao.constraints.observables import (
    constraint_result_types,
    observable_types,
    observation_types,
)


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


class ConstraintResult(BaseModel):
    observables: list[observable_types]
    comment: str = ""
    result: constraint_result_types


class RegressionResult(BaseModel):
    observable: observable_types
    result: constraint_result_types


class SavedEntry(BaseModel):
    observable: observable_types
    observation: observation_types


class SavedObservations(BaseModel):
    entries: list[SavedEntry]


class LatticeResult(BaseModel):
    lattice_file: str | None
    init_file: str | None
    loaded: bool
    error: str | None = None
    load_time: float


class ConstraintResults(BaseModel):
    started_at: datetime
    finished_at: datetime
    lattices: dict[str, LatticeResult]
    constraints: list[ConstraintResult]
    regression: list[RegressionResult] = Field(default_factory=list)
