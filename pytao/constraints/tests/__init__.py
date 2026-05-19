from typing import Annotated, Union

from pydantic import Field

from pytao.constraints.tests.base import UnitTest
from pytao.constraints.tests.dummy import DummyUnitTest
from pytao.constraints.tests.pair_match import PairMatch, TolComparison
from pytao.constraints.observables.twiss import (
    BmagTwissComparison,
    DummyTwissComparison,
    TwissComparisonMethod,
    twiss_comparison_types,
)

unit_test_types = Annotated[Union[DummyUnitTest, PairMatch], Field(discriminator="type")]

__all__ = [
    "BmagTwissComparison",
    "DummyUnitTest",
    "DummyTwissComparison",
    "PairMatch",
    "TolComparison",
    "TwissComparisonMethod",
    "UnitTest",
    "twiss_comparison_types",
    "unit_test_types",
]
