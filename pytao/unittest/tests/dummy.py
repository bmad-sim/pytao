from typing import Literal

from pytao.unittest.tests.base import UnitTest
from pytao.unittest.observables import Observable, Observation


class DummyUnitTest(UnitTest):
    type: Literal["DummyUnitTest"] = "DummyUnitTest"

    @property
    def observables(self) -> dict[str, Observable]:
        return {}

    def run(self, observations: dict[str, Observation]) -> bool:
        return True
