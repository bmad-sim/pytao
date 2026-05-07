from pydantic import BaseModel
from pytao import Tao
from abc import ABC, abstractmethod

from pytao.unittest.tests.observables import Observable

class UnitTest(BaseModel, ABC):
    @abstractmethod
    @property
    def associated_lattices(self) -> list[str] | None:
        """
        On which lattices are we required to make observations for this test. None means all lattices.

        Returns
        -------
        list[str] | None
            A list of lattice IDs for which lattices we need, or None meaning all.
        """
        ...
        
    @abstractmethod
    @property
    def observable(self) -> Observable:
        """
        The observable that will be run on each associated lattice.
        """
        ...
    
    