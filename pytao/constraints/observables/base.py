import time
from datetime import datetime, timezone
from pydantic import BaseModel, ConfigDict, Field
from pytao import Tao
from typing import Generic, Literal, TypeVar


class CheckResult(BaseModel):
    passed: bool
    detail: str = ""

    def __bool__(self) -> bool:
        return self.passed


class Observation(BaseModel):
    """Concrete output from a lattice observation."""

    elapsed_time: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


ObsT = TypeVar("ObsT", bound=Observation)


class Observable(BaseModel, Generic[ObsT]):
    """Abstract base for all observables."""

    model_config = ConfigDict(frozen=True)

    @property
    def label(self) -> str:
        return ""


class LatticeObservable(Observable[ObsT]):
    """Observable that fetches data from a lattice via Tao."""

    lattice_id: str

    @property
    def label(self) -> str:
        return self.lattice_id

    def _make_observation(self, tao: Tao) -> ObsT: ...

    def __call__(self, tao: Tao) -> ObsT:
        created_at = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        result = self._make_observation(tao)
        result.elapsed_time = time.perf_counter() - t0
        result.created_at = created_at
        return result


class LiteralObservable(Observable[ObsT]):
    """Observable whose observation is a constant value."""

    def _make_observation(self) -> ObsT: ...

    def __call__(self) -> ObsT:
        created_at = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        result = self._make_observation()
        result.elapsed_time = time.perf_counter() - t0
        result.created_at = created_at
        return result


class Comparison(BaseModel):
    """Abstract base for comparison operators between two observations."""


class ComparisonResult(BaseModel):
    """Base for all constraint check results."""

    error: str | None = None


class IsCloseResult(ComparisonResult):
    result_type: Literal["IsCloseResult"] = "IsCloseResult"
    is_close: bool


class IsClose(Comparison, Generic[ObsT]):
    """Approximate equality operator between two observations."""

    def __call__(self, obja: ObsT, objb: ObsT) -> IsCloseResult: ...


class IsLessResult(ComparisonResult):
    result_type: Literal["IsLessResult"] = "IsLessResult"
    is_less: bool

    @property
    def is_close(self) -> bool:
        return self.is_less


class IsLess(Comparison, Generic[ObsT]):
    """Component-wise less-than operator between two observations."""

    def __call__(self, obja: ObsT, objb: ObsT) -> IsLessResult: ...
