from typing import Literal

from pytao.unittest.observables import Observable, Observation
from pytao.unittest.results import TestResult
from pytao.unittest.tests.base import UnitTest


class DummyUnitTest(UnitTest):
    type: Literal["DummyUnitTest"] = "DummyUnitTest"

    @property
    def observables(self) -> dict[str, Observable]:
        return {}

    def run(self, observations: dict[str, Observation]) -> TestResult:
        return TestResult(
            test_type=type(self).__name__,
            description=self.description,
            passed=True,
        )
