from __future__ import annotations

from datetime import datetime

from pytao.constraints.pydantic import ConstraintsBase

from pytao.constraints.observables import (
    AnyComparisonResult,
    AnyObservable,
    AnyObservation,
    LatticeObservable,
)
from pytao.startup import TaoStartup


class ConstraintResult(ConstraintsBase):
    group: str | None = None
    label: str = ""
    observables: list[AnyObservable]
    description: str = ""
    comment: str = ""
    result: AnyComparisonResult


class RegressionResult(ConstraintsBase):
    group: str | None = None
    label: str = ""
    description: str = ""
    comment: str = ""
    observable: AnyObservable
    result: AnyComparisonResult


class SavedEntry(ConstraintsBase):
    observable: AnyObservable
    observation: AnyObservation


class SavedObservations(ConstraintsBase):
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

    def __len__(self):
        return len(self.entries)


class LatticeResult(ConstraintsBase):
    tao_startup: TaoStartup
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
            tao_startup=lat_startup,
            loaded=loaded,
            error=error,
            load_time=load_time,
            obs_time=obs_time,
        )


class ConstraintResults(ConstraintsBase):
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
