from typing import Literal, cast
import numpy as np
from pydantic import BaseModel, Field

from pytao.unittest.observables import EleObservable, EleObservation, Observable, Observation
from pytao.unittest.tests.twiss import twiss_comparison_types, BmagTwissComparison
from pytao.unittest.tests.base import UnitTest


class TolComparison(BaseModel):
    atol: float = 1e-8
    rtol: float = 1e-5

    def __call__(self, x0, x1) -> bool:
        return bool(np.allclose(x0, x1, rtol=self.rtol, atol=self.atol))


class PairMatch(UnitTest):
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
    ref_energy_test: TolComparison | None = Field(default_factory=TolComparison)
    p0c_test: TolComparison | None = Field(default_factory=TolComparison)
    orbit_test: TolComparison | None = Field(default_factory=TolComparison)
    floor_x_test: TolComparison | None = None
    floor_y_test: TolComparison | None = None
    floor_z_test: TolComparison | None = None

    @property
    def observables(self) -> dict[str, Observable]:
        return {
            self.lattice_a_id: EleObservable(ele=self.element_a),
            self.lattice_b_id: EleObservable(ele=self.element_b),
        }

    def run(self, observations: dict[str, Observation]) -> bool:
        obs_a = cast(EleObservation, observations[self.lattice_a_id])
        obs_b = cast(EleObservation, observations[self.lattice_b_id])
        ea, eb = obs_a.element, obs_b.element

        checks = []

        if ea.twiss is not None and eb.twiss is not None:
            ta, tb = ea.twiss, eb.twiss
            if self.twiss_a_test is not None:
                checks.append(self.twiss_a_test(ta.beta_a, ta.alpha_a, tb.beta_a, tb.alpha_a))
            if self.twiss_b_test is not None:
                checks.append(self.twiss_b_test(ta.beta_b, ta.alpha_b, tb.beta_b, tb.alpha_b))
            if self.eta_x_test is not None:
                checks.append(self.eta_x_test(ta.eta_x, tb.eta_x))
            if self.etap_x_test is not None:
                checks.append(self.etap_x_test(ta.etap_x, tb.etap_x))
            if self.eta_y_test is not None:
                checks.append(self.eta_y_test(ta.eta_y, tb.eta_y))
            if self.etap_y_test is not None:
                checks.append(self.etap_y_test(ta.etap_y, tb.etap_y))

        if ea.orbit is not None and eb.orbit is not None:
            oa, ob = ea.orbit, eb.orbit
            if self.p0c_test is not None:
                checks.append(self.p0c_test(oa.p0c, ob.p0c))
            if self.orbit_test is not None:
                vec_a = np.array([oa.x, oa.px, oa.y, oa.py, oa.z, oa.pz])
                vec_b = np.array([ob.x, ob.px, ob.y, ob.py, ob.z, ob.pz])
                checks.append(self.orbit_test(vec_a, vec_b))

        if ea.attrs is not None and eb.attrs is not None and self.ref_energy_test is not None:
            try:
                e_tot_a = float(ea.attrs["e_tot"].data)
                e_tot_b = float(eb.attrs["e_tot"].data)
                checks.append(self.ref_energy_test(e_tot_a, e_tot_b))
            except (KeyError, TypeError, ValueError):
                pass

        if ea.floor is not None and eb.floor is not None:
            fa = ea.floor.end.actual
            fb = eb.floor.end.actual
            if fa is not None and fb is not None:
                if self.floor_x_test is not None:
                    checks.append(self.floor_x_test(fa.x, fb.x))
                if self.floor_y_test is not None:
                    checks.append(self.floor_y_test(fa.y, fb.y))
                if self.floor_z_test is not None:
                    checks.append(self.floor_z_test(fa.z, fb.z))

        return all(checks)
