from typing import Annotated, Union

from pydantic import Field

from pytao.unittest.tests.base import UnitTest
from pytao.unittest.tests.dummy import DummyUnitTest
from pytao.unittest.tests.pair_match import PairMatch, TolComparison
from pytao.unittest.tests.twiss import (
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
