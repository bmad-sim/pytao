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
    """Configuration and action to make an observation from a lattice."""
    lattice_id: str

    @property
    def label(self) -> str:
        return self.lattice_id

    def __call__(self, tao: Tao) -> Observation:
        ...


class IsCloseResult(BaseModel):
    result_type: Literal["IsCloseResult"] = "IsCloseResult"
    is_close: bool
    error: str | None = None


class IsClose(BaseModel):
    """Approximate equality operator between two observations."""
    def __call__(self, obja: Observation, objb: Observation) -> IsCloseResult:
        ...


class IsLessResult(BaseModel):
    result_type: Literal["IsLessResult"] = "IsLessResult"
    is_less: bool
    error: str | None = None

    @property
    def is_close(self) -> bool:
        return self.is_less


class IsLess(BaseModel):
    """Component-wise less-than operator between two observations."""
    def __call__(self, obja: Observation, objb: Observation) -> IsLessResult:
        ...
