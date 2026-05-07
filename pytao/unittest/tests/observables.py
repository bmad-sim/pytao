from pydantic import BaseModel
from pytao import Tao

#####################################################################################
# Base classes 
#####################################################################################

class Observation(BaseModel):
    """
    Represents the concrete output from a lattice observation.
    """
    ...


class Observable(BaseModel):
    """
    Represents the configurations required to make an observation from a lattice and the action to do it.
    """
    def __call__(self, tao: Tao) -> Observation:
        """Perform the observation returning an observed quantity."""
        ...


#####################################################################################
# Concrete
#####################################################################################

class EleObservation(Observation):
    @classmethod
    def from_ele(cls, ele):
        pass
    

class EleObservable(Observable):
    """Observable that gets information from the lattice at one element."""
    ele: str | int
    
    def __call__(self, tao: Tao) -> Observation:
        """Perform the observation returning an observed quantity."""
        return EleObservation.from_ele(tao.ele(self.ele))
