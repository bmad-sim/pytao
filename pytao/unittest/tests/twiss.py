from typing import Literal, Annotated, Union
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

from pytao.unittest.results import CheckResult


class TwissComparisonMethod(BaseModel, ABC):
    @abstractmethod
    def __call__(self, beta0, alpha0, beta1, alpha1) -> CheckResult:
        ...


class DummyTwissComparison(TwissComparisonMethod):
    type: Literal["dummy"] = "dummy"

    def __call__(self, _beta0, _alpha0, _beta1, _alpha1) -> CheckResult:
        return CheckResult(passed=True)


class BmagTwissComparison(TwissComparisonMethod):
    type: Literal["bmag"] = "bmag"
    max_bmag: float = 1.01
    min_bmag: float = 0.99

    def __call__(self, beta0, alpha0, beta1, alpha1) -> CheckResult:
        bmag = 0.5 * ((beta0 / beta1 + beta1 / beta0) + beta1 * beta0 * (alpha1 / beta1 - alpha0 / beta0) ** 2)
        passed = self.min_bmag <= bmag <= self.max_bmag
        detail = (
            ""
            if passed
            else (
                f"bmag={bmag:.4f} outside [{self.min_bmag}, {self.max_bmag}]"
                f"  (beta0={beta0:.4g}, alpha0={alpha0:.4g})"
                f"  (beta1={beta1:.4g}, alpha1={alpha1:.4g})"
            )
        )
        return CheckResult(passed=passed, detail=detail)


twiss_comparison_types = Annotated[Union[DummyTwissComparison, BmagTwissComparison], Field(discriminator="type")]
