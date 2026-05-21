import time
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from pytao import Tao
from typing import Literal


class CheckResult(BaseModel):
    passed: bool
    detail: str = ""

    def __bool__(self) -> bool:
        return self.passed


class Observation(BaseModel):
    """Concrete output from a lattice observation."""
    elapsed_time: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Observable(BaseModel, frozen=True):
    """Abstract base for all observables."""

    @property
    def label(self) -> str:
        return ""


class LatticeObservable(Observable):
    """Observable that fetches data from a lattice via Tao."""
    lattice_id: str

    @property
    def label(self) -> str:
        return self.lattice_id

    def _make_observation(self, tao: Tao) -> Observation:
        ...

    def __call__(self, tao: Tao) -> Observation:
        created_at = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        result = self._make_observation(tao)
        result.elapsed_time = time.perf_counter() - t0
        result.created_at = created_at
        return result


class LiteralObservable(Observable):
    """Observable whose observation is a constant value."""

    def _make_observation(self) -> Observation:
        ...

    def __call__(self) -> Observation:
        created_at = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        result = self._make_observation()
        result.elapsed_time = time.perf_counter() - t0
        result.created_at = created_at
        return result


class Comparison(BaseModel):
    """Abstract base for comparison operators between two observations."""


class ConstraintResult(BaseModel):
    """Base for all constraint check results."""
    error: str | None = None


class IsCloseResult(ConstraintResult):
    result_type: Literal["IsCloseResult"] = "IsCloseResult"
    is_close: bool


class IsClose(Comparison):
    """Approximate equality operator between two observations."""
    def __call__(self, obja: Observation, objb: Observation) -> IsCloseResult:
        ...


class IsLessResult(ConstraintResult):
    result_type: Literal["IsLessResult"] = "IsLessResult"
    is_less: bool

    @property
    def is_close(self) -> bool:
        return self.is_less


class IsLess(Comparison):
    """Component-wise less-than operator between two observations."""
    def __call__(self, obja: Observation, objb: Observation) -> IsLessResult:
        ...
