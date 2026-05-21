from pydantic import BaseModel
from pytao import Tao
from typing import Literal


class CheckResult(BaseModel):
    passed: bool
    detail: str = ""

    def __bool__(self) -> bool:
        return self.passed


class Observation(BaseModel):
    """Concrete output from a lattice observation."""
    ...


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

    def __call__(self, tao: Tao) -> Observation:
        ...


class LiteralObservable(Observable):
    """Observable whose observation is a constant value."""

    def get_observation(self) -> Observation:
        ...


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
