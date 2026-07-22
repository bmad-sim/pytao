from __future__ import annotations

from collections.abc import Iterator
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
    particle_survived: bool | None = None
    error: str = ""
    load_time: float = 0.0
    obs_time: float = 0.0

    @property
    def failed(self) -> bool:
        return not self.loaded or self.particle_survived is False


class ConstraintResultsGroup(ConstraintsBase):
    started_at: datetime
    finished_at: datetime
    lattices: dict[str, LatticeResult]
    constraints: dict[str | None, list[ConstraintResult]]
    regression: dict[str | None, list[RegressionResult]] = {}

    def iter_constraints(self) -> Iterator[tuple[str | None, ConstraintResult]]:
        for group, crs in self.constraints.items():
            for cr in crs:
                yield group, cr

    def iter_regression(self) -> Iterator[tuple[str | None, RegressionResult]]:
        for group, rrs in self.regression.items():
            for rr in rrs:
                yield group, rr
