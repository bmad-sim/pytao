from pydantic import BaseModel, Field
from typing import Literal, Annotated, Union
from abc import ABC, abstractmethod
import numpy as np

from pytao.unittest.tests.twiss import twiss_comparison_types, BmagTwissComparison, DispersionComparison


class TolComparison(BaseModel):
    atol: float = 1e-8
    rtol: float = 1e-5
    
    def __call__(self, x0, x1):
        return np.allclose(x0, x1, rtol=self.rtol, atol=self.atol)


class PairMatch(BaseModel):
    type: Literal["PairMatch"] = "PairMatch"
    
    # Define the element and lattice which we are comparing
    lattice_a_id: str
    element_a: str
    lattice_b_id: str
    element_b: str
    
    # Twiss tests
    twiss_a_test: twiss_comparison_types | None = Field(default_factory=BmagTwissComparison)
    twiss_b_test: twiss_comparison_types | None = Field(default_factory=BmagTwissComparison)
    
    # Dispersion tests
    eta_x_test: TolComparison | None = Field(default_factory=TolComparison)
    etap_x_test: TolComparison | None = Field(default_factory=TolComparison)
    eta_y_test: TolComparison | None = Field(default_factory=TolComparison)
    etap_y_test: TolComparison | None = Field(default_factory=TolComparison)

    # Other tests
    p0c_test: TolComparison | None = Field(default_factory=TolComparison)
    orbit_test: TolComparison | None = Field(default_factory=TolComparison)
    floor_x_test: TolComparison | None = Field(default_factory=TolComparison)
    floor_y_test: TolComparison | None = Field(default_factory=TolComparison)
    floor_z_test: TolComparison | None = Field(default_factory=TolComparison)
    