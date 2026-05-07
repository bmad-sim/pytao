from abc import ABC, abstractmethod
from pydantic import BaseModel

from pytao.unittest.observables import Observable, Observation


class UnitTest(BaseModel, ABC):
    @property
    @abstractmethod
    def observables(self) -> dict[str, Observable]:
        """
        Mapping of lattice_id to the Observable to run on that lattice.

        Returns
        -------
        dict[str, Observable]
        """
        ...

    @abstractmethod
    def run(self, observations: dict[str, Observation]) -> bool:
        """
        Run the test given observations keyed by lattice_id.

        Parameters
        ----------
        observations : dict[str, Observation]

        Returns
        -------
        bool
        """
        ...
