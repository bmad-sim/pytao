from pydantic import BaseModel
from pytao import Tao
from typing import Literal, TypeVar


class Observation(BaseModel):
    """Concrete output from a lattice observation."""
    ...


class Observable(BaseModel, frozen=True):
    """Configuration and action to make an observation from a lattice."""
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
