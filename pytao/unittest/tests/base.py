from abc import ABC, abstractmethod
from pydantic import BaseModel

from pytao.unittest.observables import Observable, Observation
from pytao.unittest.results import TestResult


class UnitTest(BaseModel, ABC):
    description: str = ""
    comment: str = ""

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
    def run(self, observations: dict[str, Observation]) -> TestResult:
        """
        Run the test given observations keyed by lattice_id.

        Parameters
        ----------
        observations : dict[str, Observation]

        Returns
        -------
        TestResult
        """
        ...
