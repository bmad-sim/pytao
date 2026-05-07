from pydantic import BaseModel
from pytao import Tao
from pytao.model import Element


class Observation(BaseModel):
    """Concrete output from a lattice observation."""
    ...


class Observable(BaseModel):
    """Configuration and action to make an observation from a lattice."""
    def __call__(self, tao: Tao) -> Observation:
        ...


class EleObservation(Observation):
    element: Element

    @classmethod
    def from_ele(cls, ele: Element) -> "EleObservation":
        return cls(element=ele)


class EleObservable(Observable):
    """Observable that gets information from the lattice at one element."""
    ele: str | int

    def __call__(self, tao: Tao) -> EleObservation:
        return EleObservation.from_ele(tao.ele(self.ele))
