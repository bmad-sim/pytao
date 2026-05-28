from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from pytao.constraints.observables import (
    AnyComparisonResult,
    AnyObservable,
    AnyObservation,
    LatticeObservable,
)
from pytao.startup import TaoStartup


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
    group: str | None = None
    observables: list[AnyObservable]
    description: str = ""
    comment: str = ""
    result: AnyComparisonResult


class RegressionResult(BaseModel):
    observable: AnyObservable
    result: AnyComparisonResult


class SavedEntry(BaseModel):
    observable: AnyObservable
    observation: AnyObservation


class SavedObservations(BaseModel):
    entries: list[SavedEntry]

    @classmethod
    def from_obs_map(cls, obs_map: dict[AnyObservable, AnyObservation]) -> SavedObservations:
        return cls(
            entries=[
                SavedEntry(observable=obs, observation=obs_val)
                for obs, obs_val in obs_map.items()
                if isinstance(obs, LatticeObservable)
            ]
        )

    @property
    def obs_map(self) -> dict[AnyObservable, AnyObservation]:
        return {e.observable: e.observation for e in self.entries}


class LatticeResult(BaseModel):
    lattice_file: str | None
    init_file: str | None
    loaded: bool
    error: str | None = None
    load_time: float
    obs_time: float = 0.0

    @classmethod
    def from_startup(
        cls,
        lat_startup: TaoStartup,
        *,
        loaded: bool,
        error: str | None,
        load_time: float,
        obs_time: float,
    ) -> LatticeResult:
        return cls(
            lattice_file=str(lat_startup.lattice_file) if lat_startup.lattice_file else None,
            init_file=str(lat_startup.init_file) if lat_startup.init_file else None,
            loaded=loaded,
            error=error,
            load_time=load_time,
            obs_time=obs_time,
        )


class ConstraintResults(BaseModel):
    started_at: datetime
    finished_at: datetime
    lattices: dict[str, LatticeResult]
    constraints: list[ConstraintResult]
    regression: list[RegressionResult] = []

    @property
    def constraints_by_group(self) -> dict[str | None, list[ConstraintResult]]:
        groups: dict[str | None, list[ConstraintResult]] = {}
        for cr in self.constraints:
            groups.setdefault(cr.group, []).append(cr)
        return groups
