import time
from datetime import datetime, timezone
from pydantic import ConfigDict, Field, model_validator

from pytao.constraints.pydantic import ConstraintsBase
from pytao import Tao
from typing import Generic, Literal, TypeVar


class CheckResult(ConstraintsBase):
    """Result of a single scalar or array comparison check.

    Attributes
    ----------
    passed : bool
        Whether the check passed.
    detail : str
        Human-readable detail shown on failure.
    """

    passed: bool
    detail: str = ""

    def __bool__(self) -> bool:
        return self.passed

    def format_detail(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        if not self.passed and self.detail:
            return f"{status}  {self.detail}"
        return status


class Observation(ConstraintsBase):
    """Base class for all observation outputs.

    Attributes
    ----------
    elapsed_time : float
        Wall-clock time taken to produce the observation, in seconds.
    created_at : datetime
        UTC timestamp at which the observation was created.
    """

    elapsed_time: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


ObsT = TypeVar("ObsT", bound=Observation)


class Observable(ConstraintsBase, Generic[ObsT]):
    """Abstract base for all observables.

    Generic over ``ObsT``, the ``Observation`` subclass this observable produces.
    All observable instances are frozen (immutable) Pydantic models.
    """

    model_config = ConfigDict(frozen=True)

    @property
    def label(self) -> str:
        return ""


class LatticeObservable(Observable[ObsT]):
    """Observable that fetches data from a lattice via Tao.

    Subclasses implement ``_make_observation`` to retrieve and package data.

    Attributes
    ----------
    lattice_id : str
        Identifier for the lattice this observable is associated with.
    """

    lattice_id: str

    @property
    def label(self) -> str:
        return self.lattice_id

    def _make_observation(self, tao: Tao) -> ObsT: ...

    def observe(self, tao: Tao) -> ObsT:
        created_at = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        result = self._make_observation(tao)
        result.elapsed_time = time.perf_counter() - t0
        result.created_at = created_at
        return result


class LiteralObservable(Observable[ObsT]):
    """Observable whose observation is a constant value independent of the lattice.

    Subclasses implement ``_make_observation`` to build the fixed observation.
    """

    def _make_observation(self) -> ObsT: ...

    def observe(self) -> ObsT:
        created_at = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        result = self._make_observation()
        result.elapsed_time = time.perf_counter() - t0
        result.created_at = created_at
        return result


class ComparisonResult(ConstraintsBase):
    """Base class for all constraint check results.

    Attributes
    ----------
    error : str or None
        Set to a non-empty string when evaluation failed (e.g. a Tao error).
        When set, ``is_satisfied`` returns ``False`` regardless of per-field results.
    """

    error: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _strip_computed(cls, data: object) -> object:
        if isinstance(data, dict):
            data.pop("is_satisfied", None)
        return data

    @property
    def is_satisfied(self) -> bool:
        return not bool(self.error)

    def check_results(self) -> dict[str, CheckResult]:
        return {
            name: getattr(self, name)
            for name in type(self).model_fields
            if isinstance(getattr(self, name), CheckResult)
        }


ResultT = TypeVar("ResultT", bound=ComparisonResult)


class Comparison(ConstraintsBase, Generic[ResultT]):
    """Abstract base for comparison operators between two observations."""


class IsCloseResult(ComparisonResult):
    """Base result type for approximate-equality comparisons.

    Attributes
    ----------
    result_type : str
        Discriminator literal. Always ``"is_close"``.
    """

    result_type: Literal["is_close"] = "is_close"


class IsClose(Comparison[IsCloseResult], Generic[ObsT]):
    """Approximate equality operator between two observations."""

    def compare(self, obja: ObsT, objb: ObsT) -> IsCloseResult: ...


class IsLessResult(ComparisonResult):
    """Base result type for less-than comparisons.

    Attributes
    ----------
    result_type : str
        Discriminator literal. Always ``"is_less"``.
    """

    result_type: Literal["is_less"] = "is_less"


class IsLess(Comparison[IsLessResult], Generic[ObsT]):
    """Component-wise less-than operator between two observations."""

    def compare(self, obja: ObsT, objb: ObsT) -> IsLessResult: ...
