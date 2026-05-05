from pydantic import BaseModel, Field
from typing import Literal, Annotated, Union
from abc import ABC, abstractmethod
import numpy as np


class TwissComparisonMethod(BaseModel, ABC):
    @abstractmethod
    def __call__(self, beta0, alpha0, beta1, alpha1):
        ...


class DummyTwissComparison(TwissComparisonMethod):
    type: Literal["dummy"] = "dummy"
    
    def __call__(self, beta0, alpha0, beta1, alpha1):
        return True


class BmagTwissComparison(TwissComparisonMethod):
    type: Literal["bmag"] = "bmag"
    max_bmag: float = 1.1
    min_bmag: float = 0.9
    
    def __call__(self, beta0, alpha0, beta1, alpha1):
        bmag = 0.5 * ( (beta0/beta1 + beta1/beta0) + beta1*beta0*(alpha1/beta1 - alpha0/beta0)**2 )
        return self.min_bmag <= bmag <= self.max_bmag

twiss_comparison_types = Annotated[Union[DummyTwissComparison, BmagTwissComparison], Field(discriminator="type")]
