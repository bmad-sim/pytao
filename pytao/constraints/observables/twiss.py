from abc import ABC, abstractmethod
from typing import Annotated, Literal, Union

from pydantic import Field

from pytao.constraints.pydantic import ConstraintsBase

from pytao.constraints.observables.base import CheckResult


class TwissComparisonMethod(ConstraintsBase, ABC):
    """Abstract base for Twiss parameter comparison methods.

    Subclasses implement ``__call__`` to compare a (beta, alpha) pair
    from two elements and return a ``CheckResult``.
    """

    @abstractmethod
    def __call__(self, beta0, alpha0, beta1, alpha1) -> CheckResult: ...


class BmagTwissComparison(TwissComparisonMethod):
    """Twiss comparison using the Bmag beam transfer matrix invariant.

    Attributes
    ----------
    max_bmag : float
        Upper bound for an acceptable Bmag value.
    min_bmag : float
        Lower bound for an acceptable Bmag value.
    """

    type: Literal["bmag"] = "bmag"
    max_bmag: float = 1.01
    min_bmag: float = 0.99

    def __call__(self, beta0, alpha0, beta1, alpha1) -> CheckResult:
        bmag = 0.5 * (
            (beta0 / beta1 + beta1 / beta0)
            + beta1 * beta0 * (alpha1 / beta1 - alpha0 / beta0) ** 2
        )
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


AnyTwissComparison = Annotated[Union[BmagTwissComparison], Field(discriminator="type")]
