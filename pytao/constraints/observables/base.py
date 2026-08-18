import time
from datetime import datetime, timezone
from pydantic import ConfigDict, Field, model_validator

from pytao.constraints.pydantic import ConstraintsBase
from pytao import Tao
from typing import Generic, TypeVar


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


ObservationT = TypeVar("ObservationT", bound=Observation)


class Observable(ConstraintsBase, Generic[ObservationT]):
    """Abstract base for all observables.

    Generic over ``ObservationT``, the ``Observation`` subclass this observable produces.
    All observable instances are frozen (immutable) Pydantic models.
    """

    model_config = ConfigDict(frozen=True)

    @property
    def label(self) -> str: ...

    def observe(self, *args, **kwargs) -> ObservationT: ...


ObservableT = TypeVar("ObservableT", bound=Observable)


class LatticeObservable(Observable[ObservationT]):
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

    def _make_observation(self, tao: Tao) -> ObservationT: ...

    def observe(self, tao: Tao) -> ObservationT:
        created_at = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        result = self._make_observation(tao)
        result.elapsed_time = time.perf_counter() - t0
        result.created_at = created_at
        return result


class LiteralObservable(Observable[ObservationT]):
    """Observable whose observation is a constant value independent of the lattice.

    Subclasses implement ``_make_observation`` to build the fixed observation.
    """

    def _make_observation(self) -> ObservationT: ...

    def observe(self) -> ObservationT:
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


class Comparison(ConstraintsBase, Generic[ObservationT, ResultT]):
    """Abstract base for comparison operators between two observations."""

    def compare(self, obja: ObservationT, objb: ObservationT) -> ResultT: ...


class IsClose(Comparison[ObservationT, ResultT]):
    """
    Approximate equality operator between two observations.

    This class retained to restrict RegressionConstraints to only IsClose operations
    """

    def compare(self, obja: ObservationT, objb: ObservationT) -> ResultT: ...


class IsLess(Comparison[ObservationT, ResultT]):
    """Component-wise less-than operator between two observations."""

    def compare(self, obja: ObservationT, objb: ObservationT) -> ResultT: ...
