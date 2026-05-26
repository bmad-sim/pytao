from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from pytao.constraints.observables import (
    LatticeObservable,
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
    description: str = ""
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

    @classmethod
    def from_obs_map(
        cls, obs_map: dict[observable_types, observation_types]
    ) -> "SavedObservations":
        return cls(
            entries=[
                SavedEntry(observable=obs, observation=obs_val)
                for obs, obs_val in obs_map.items()
                if isinstance(obs, LatticeObservable)
            ]
        )

    @property
    def obs_map(self) -> dict[observable_types, observation_types]:
        return {e.observable: e.observation for e in self.entries}


class LatticeResult(BaseModel):
    lattice_file: str | None
    init_file: str | None
    loaded: bool
    error: str | None = None
    load_time: float
    obs_time: float = 0.0


class ConstraintResults(BaseModel):
    started_at: datetime
    finished_at: datetime
    lattices: dict[str, LatticeResult]
    constraints: list[ConstraintResult]
    regression: list[RegressionResult] = Field(default_factory=list)
