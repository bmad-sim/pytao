from typing import Literal, cast
import numpy as np
from pydantic import BaseModel, Field

from pytao.unittest.observables import EleObservable, EleObservation, Observable, Observation
from pytao.unittest.observables.base import CheckResult
from pytao.unittest.results import PairMatchResult
from pytao.unittest.observables.twiss import BmagTwissComparison, twiss_comparison_types
from pytao.unittest.tests.base import UnitTest


class TolComparison(BaseModel):
    atol: float = 1e-8
    rtol: float = 1e-5

    def __call__(self, x0, x1) -> CheckResult:
        passed = bool(np.allclose(x0, x1, rtol=self.rtol, atol=self.atol))
        if passed:
            return CheckResult(passed=True)
        x0a, x1a = np.asarray(x0), np.asarray(x1)
        if x0a.ndim == 0:
            detail = f"a={float(x0a):.6g}, b={float(x1a):.6g}, diff={abs(float(x0a) - float(x1a)):.3e}"
        else:
            detail = f"max|diff|={np.max(np.abs(x0a - x1a)):.3e}"
        return CheckResult(passed=False, detail=detail)


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

    def run(self, observations: dict[str, Observation]) -> PairMatchResult:
        obs_a = cast(EleObservation, observations[self.lattice_a_id])
        obs_b = cast(EleObservation, observations[self.lattice_b_id])
        ea, eb = obs_a.element, obs_b.element

        twiss_a: CheckResult | None = None
        twiss_b: CheckResult | None = None
        eta_x: CheckResult | None = None
        etap_x: CheckResult | None = None
        eta_y: CheckResult | None = None
        etap_y: CheckResult | None = None
        ref_energy: CheckResult | None = None
        p0c: CheckResult | None = None
        orbit: CheckResult | None = None
        floor_x: CheckResult | None = None
        floor_y: CheckResult | None = None
        floor_z: CheckResult | None = None

        if ea.twiss is not None and eb.twiss is not None:
            ta, tb = ea.twiss, eb.twiss
            if self.twiss_a_test is not None:
                twiss_a = self.twiss_a_test(ta.beta_a, ta.alpha_a, tb.beta_a, tb.alpha_a)
            if self.twiss_b_test is not None:
                twiss_b = self.twiss_b_test(ta.beta_b, ta.alpha_b, tb.beta_b, tb.alpha_b)
            if self.eta_x_test is not None:
                eta_x = self.eta_x_test(ta.eta_x, tb.eta_x)
            if self.etap_x_test is not None:
                etap_x = self.etap_x_test(ta.etap_x, tb.etap_x)
            if self.eta_y_test is not None:
                eta_y = self.eta_y_test(ta.eta_y, tb.eta_y)
            if self.etap_y_test is not None:
                etap_y = self.etap_y_test(ta.etap_y, tb.etap_y)

        if ea.orbit is not None and eb.orbit is not None:
            oa, ob = ea.orbit, eb.orbit
            if self.p0c_test is not None:
                p0c = self.p0c_test(oa.p0c, ob.p0c)
            if self.orbit_test is not None:
                vec_a = np.array([oa.x, oa.px, oa.y, oa.py, oa.z, oa.pz])
                vec_b = np.array([ob.x, ob.px, ob.y, ob.py, ob.z, ob.pz])
                orbit = self.orbit_test(vec_a, vec_b)

        if ea.attrs is not None and eb.attrs is not None and self.ref_energy_test is not None:
            try:
                e_tot_a = float(ea.attrs["e_tot"].data)
                e_tot_b = float(eb.attrs["e_tot"].data)
            except (KeyError, TypeError, ValueError):
                pass
            else:
                ref_energy = self.ref_energy_test(e_tot_a, e_tot_b)

        if ea.floor is not None and eb.floor is not None:
            fa = ea.floor.end.actual
            fb = eb.floor.end.actual
            if fa is not None and fb is not None:
                if self.floor_x_test is not None:
                    floor_x = self.floor_x_test(fa.x, fb.x)
                if self.floor_y_test is not None:
                    floor_y = self.floor_y_test(fa.y, fb.y)
                if self.floor_z_test is not None:
                    floor_z = self.floor_z_test(fa.z, fb.z)

        ran = [r for r in [twiss_a, twiss_b, eta_x, etap_x, eta_y, etap_y,
                            ref_energy, p0c, orbit, floor_x, floor_y, floor_z]
               if r is not None]

        return PairMatchResult(
            test_type=type(self).__name__,
            description=self.description,
            passed=all(ran) if ran else True,
            twiss_a=twiss_a,
            twiss_b=twiss_b,
            eta_x=eta_x,
            etap_x=etap_x,
            eta_y=eta_y,
            etap_y=etap_y,
            ref_energy=ref_energy,
            p0c=p0c,
            orbit=orbit,
            floor_x=floor_x,
            floor_y=floor_y,
            floor_z=floor_z,
        )
