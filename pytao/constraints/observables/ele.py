from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field, computed_field

from pytao.constraints.pydantic import ConstraintsBase

from pytao import Tao
from pytao.constraints.observables.base import (
    CheckResult,
    IsClose,
    IsCloseResult,
    IsLess,
    IsLessResult,
    LatticeObservable,
    LiteralObservable,
    Observation,
)
from pytao.constraints.observables.twiss import AnyTwissComparison, BmagTwissComparison
from pytao.model import (
    ElementFloor,
    ElementFloorAll,
    ElementFloorPosition,
    ElementOrbit,
    ElementTwiss,
)
from pytao.model import _generated as tao_classes
from pytao.model.ele.ele import Element


class EleObservation(Observation):
    """Observation with all of the data available in a Bmad element. Ie, all of the information you get from a pytao
    ``tao.ele(...)`` call.

    Attributes
    ----------
    obs_type : str
        Discriminator literal. Always ``"ele"``.
    element : Element
        Element data including Twiss parameters, orbit, floor position, and attributes.
    """

    obs_type: Literal["ele"] = "ele"
    element: Element


class TolComparison(ConstraintsBase):
    """Scalar or array approximate-equality check using ``np.allclose``.

    Attributes
    ----------
    atol : float
        Absolute tolerance.
    rtol : float
        Relative tolerance.
    """

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


class EleIsCloseResult(IsCloseResult):
    """Result of an EleIsClose comparison with per-field check results.

    Each field is ``None`` if the corresponding comparison was not run.

    Attributes
    ----------
    result_type : str
        Discriminator literal. Always ``"ele_is_close"``.
    twiss_a : CheckResult or None
        Mode A Twiss comparison (beta_a, alpha_a).
    twiss_b : CheckResult or None
        Mode B Twiss comparison (beta_b, alpha_b).
    eta_x : CheckResult or None
        Horizontal dispersion.
    etap_x : CheckResult or None
        Horizontal dispersion slope.
    eta_y : CheckResult or None
        Vertical dispersion.
    etap_y : CheckResult or None
        Vertical dispersion slope.
    ref_energy : CheckResult or None
        Total reference energy (e_tot).
    p0c : CheckResult or None
        Reference momentum.
    orbit : CheckResult or None
        Orbit, as 6D vector.
    floor_x : CheckResult or None
        Global floor x coordinate.
    floor_y : CheckResult or None
        Global floor y coordinate.
    floor_z : CheckResult or None
        Global floor z coordinate.
    """

    result_type: Literal["ele_is_close"] = "ele_is_close"
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

    @computed_field
    @property
    def is_satisfied(self) -> bool:
        if self.error:
            return False
        ran = [
            r
            for r in [
                self.twiss_a,
                self.twiss_b,
                self.eta_x,
                self.etap_x,
                self.eta_y,
                self.etap_y,
                self.ref_energy,
                self.p0c,
                self.orbit,
                self.floor_x,
                self.floor_y,
                self.floor_z,
            ]
            if r is not None
        ]
        return all(ran) if ran else True


class EleIsClose(IsClose[EleObservation]):
    """IsClose operator comparing two EleObservation instances across all available data.

    Set a field to ``None`` to skip that comparison.

    Attributes
    ----------
    twiss_a : AnyTwissComparison or None
        Comparison method for mode A Twiss parameters.
    twiss_b : AnyTwissComparison or None
        Comparison method for mode B Twiss parameters.
    eta_x : TolComparison or None
        Comparison for horizontal dispersion.
    etap_x : TolComparison or None
        Comparison for horizontal dispersion slope.
    eta_y : TolComparison or None
        Comparison for vertical dispersion.
    etap_y : TolComparison or None
        Comparison for vertical dispersion slope.
    ref_energy : TolComparison or None
        Comparison for total reference energy.
    p0c : TolComparison or None
        Comparison for reference momentum.
    orbit : TolComparison or None
        Comparison for orbit, as 6D vector.
    floor_x : TolComparison or None
        Comparison for global floor x coordinate.
    floor_y : TolComparison or None
        Comparison for global floor y coordinate.
    floor_z : TolComparison or None
        Comparison for global floor z coordinate.
    """

    twiss_a: AnyTwissComparison | None = BmagTwissComparison()
    twiss_b: AnyTwissComparison | None = BmagTwissComparison()

    eta_x: TolComparison | None = TolComparison()
    etap_x: TolComparison | None = TolComparison()
    eta_y: TolComparison | None = TolComparison()
    etap_y: TolComparison | None = TolComparison()

    ref_energy: TolComparison | None = TolComparison()
    p0c: TolComparison | None = TolComparison()
    orbit: TolComparison | None = TolComparison()
    floor_x: TolComparison | None = None
    floor_y: TolComparison | None = None
    floor_z: TolComparison | None = None

    def __call__(self, obja: EleObservation, objb: EleObservation) -> EleIsCloseResult:
        ea, eb = obja.element, objb.element

        twiss_a = twiss_b = eta_x = etap_x = eta_y = etap_y = None
        ref_energy = p0c = orbit = floor_x = floor_y = floor_z = None

        ta, tb = ea.twiss, eb.twiss
        twiss_ok = ta is not None and tb is not None
        no_twiss = CheckResult(passed=False, detail="twiss data not available")
        if self.twiss_a is not None:
            twiss_a = (
                self.twiss_a(ta.beta_a, ta.alpha_a, tb.beta_a, tb.alpha_a)
                if twiss_ok
                else no_twiss
            )
        if self.twiss_b is not None:
            twiss_b = (
                self.twiss_b(ta.beta_b, ta.alpha_b, tb.beta_b, tb.alpha_b)
                if twiss_ok
                else no_twiss
            )
        if self.eta_x is not None:
            eta_x = self.eta_x(ta.eta_x, tb.eta_x) if twiss_ok else no_twiss
        if self.etap_x is not None:
            etap_x = self.etap_x(ta.etap_x, tb.etap_x) if twiss_ok else no_twiss
        if self.eta_y is not None:
            eta_y = self.eta_y(ta.eta_y, tb.eta_y) if twiss_ok else no_twiss
        if self.etap_y is not None:
            etap_y = self.etap_y(ta.etap_y, tb.etap_y) if twiss_ok else no_twiss

        oa, ob = ea.orbit, eb.orbit
        orbit_ok = oa is not None and ob is not None
        no_orbit = CheckResult(passed=False, detail="orbit data not available")
        if self.p0c is not None:
            p0c = self.p0c(oa.p0c, ob.p0c) if orbit_ok else no_orbit
        if self.orbit is not None:
            if orbit_ok:
                vec_a = np.array([oa.x, oa.px, oa.y, oa.py, oa.z, oa.pz])
                vec_b = np.array([ob.x, ob.px, ob.y, ob.py, ob.z, ob.pz])
                orbit = self.orbit(vec_a, vec_b)
            else:
                orbit = no_orbit

        if self.ref_energy is not None:
            ref_energy_ok = False
            e_tot_a = e_tot_b = 0.0
            if ea.attrs is not None and eb.attrs is not None:
                try:
                    e_tot_a = float(ea.attrs["e_tot"].data)
                    e_tot_b = float(eb.attrs["e_tot"].data)
                    ref_energy_ok = True
                except (KeyError, TypeError, ValueError):
                    pass
            ref_energy = (
                self.ref_energy(e_tot_a, e_tot_b)
                if ref_energy_ok
                else CheckResult(passed=False, detail="ref_energy data not available")
            )

        fa = ea.floor.end.actual if ea.floor is not None else None
        fb = eb.floor.end.actual if eb.floor is not None else None
        floor_ok = fa is not None and fb is not None
        no_floor = CheckResult(passed=False, detail="floor data not available")
        if self.floor_x is not None:
            floor_x = self.floor_x(fa.x, fb.x) if floor_ok else no_floor
        if self.floor_y is not None:
            floor_y = self.floor_y(fa.y, fb.y) if floor_ok else no_floor
        if self.floor_z is not None:
            floor_z = self.floor_z(fa.z, fb.z) if floor_ok else no_floor

        return EleIsCloseResult(
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


class EleLessThanResult(IsLessResult):
    """Result of an EleLessThan comparison with per-field less-than check results.

    Each field is ``None`` if the corresponding component was not checked.

    Attributes
    ----------
    result_type : str
        Discriminator literal. Always ``"ele_is_less"``.
    beta_a : CheckResult or None
        Mode A beta function.
    alpha_a : CheckResult or None
        Mode A alpha function.
    beta_b : CheckResult or None
        Mode B beta function.
    alpha_b : CheckResult or None
        Mode B alpha function.
    eta_x : CheckResult or None
        Horizontal dispersion.
    etap_x : CheckResult or None
        Horizontal dispersion slope.
    eta_y : CheckResult or None
        Vertical dispersion.
    etap_y : CheckResult or None
        Vertical dispersion slope.
    ref_energy : CheckResult or None
        Total reference energy.
    p0c : CheckResult or None
        Reference momentum.
    floor_x : CheckResult or None
        Global floor x coordinate.
    floor_y : CheckResult or None
        Global floor y coordinate.
    floor_z : CheckResult or None
        Global floor z coordinate.
    """

    result_type: Literal["ele_is_less"] = "ele_is_less"
    beta_a: CheckResult | None = None
    alpha_a: CheckResult | None = None
    beta_b: CheckResult | None = None
    alpha_b: CheckResult | None = None
    eta_x: CheckResult | None = None
    etap_x: CheckResult | None = None
    eta_y: CheckResult | None = None
    etap_y: CheckResult | None = None
    ref_energy: CheckResult | None = None
    p0c: CheckResult | None = None
    floor_x: CheckResult | None = None
    floor_y: CheckResult | None = None
    floor_z: CheckResult | None = None

    @computed_field
    @property
    def is_satisfied(self) -> bool:
        if self.error:
            return False
        ran = [
            r
            for r in [
                self.beta_a,
                self.alpha_a,
                self.beta_b,
                self.alpha_b,
                self.eta_x,
                self.etap_x,
                self.eta_y,
                self.etap_y,
                self.ref_energy,
                self.p0c,
                self.floor_x,
                self.floor_y,
                self.floor_z,
            ]
            if r is not None
        ]
        return all(ran) if ran else True


class EleLessThan(IsLess[EleObservation]):
    """Component-wise less-than comparison between two EleObservations.

    Set a field to ``True`` to enable the less-than check for that component.

    Attributes
    ----------
    beta_a : bool
        Check mode A beta function.
    alpha_a : bool
        Check mode A alpha function.
    beta_b : bool
        Check mode B beta function.
    alpha_b : bool
        Check mode B alpha function.
    eta_x : bool
        Check horizontal dispersion.
    etap_x : bool
        Check horizontal dispersion slope.
    eta_y : bool
        Check vertical dispersion.
    etap_y : bool
        Check vertical dispersion slope.
    ref_energy : bool
        Check total reference energy.
    p0c : bool
        Check reference momentum.
    floor_x : bool
        Check global floor x coordinate.
    floor_y : bool
        Check global floor y coordinate.
    floor_z : bool
        Check global floor z coordinate.
    """

    beta_a: bool = False
    alpha_a: bool = False
    beta_b: bool = False
    alpha_b: bool = False
    eta_x: bool = False
    etap_x: bool = False
    eta_y: bool = False
    etap_y: bool = False
    ref_energy: bool = False
    p0c: bool = False
    floor_x: bool = False
    floor_y: bool = False
    floor_z: bool = False

    def _check(self, va: float, vb: float) -> CheckResult:
        passed = va < vb
        return CheckResult(
            passed=passed, detail="" if passed else f"a={va:.6g} not < b={vb:.6g}"
        )

    def __call__(self, obja: EleObservation, objb: EleObservation) -> EleLessThanResult:
        ea, eb = obja.element, objb.element
        beta_a = alpha_a = beta_b = alpha_b = None
        eta_x = etap_x = eta_y = etap_y = None
        ref_energy = p0c = floor_x = floor_y = floor_z = None

        ta, tb = ea.twiss, eb.twiss
        twiss_ok = ta is not None and tb is not None
        no_twiss = CheckResult(passed=False, detail="twiss data not available")
        if self.beta_a:
            beta_a = self._check(ta.beta_a, tb.beta_a) if twiss_ok else no_twiss
        if self.alpha_a:
            alpha_a = self._check(ta.alpha_a, tb.alpha_a) if twiss_ok else no_twiss
        if self.beta_b:
            beta_b = self._check(ta.beta_b, tb.beta_b) if twiss_ok else no_twiss
        if self.alpha_b:
            alpha_b = self._check(ta.alpha_b, tb.alpha_b) if twiss_ok else no_twiss
        if self.eta_x:
            eta_x = self._check(ta.eta_x, tb.eta_x) if twiss_ok else no_twiss
        if self.etap_x:
            etap_x = self._check(ta.etap_x, tb.etap_x) if twiss_ok else no_twiss
        if self.eta_y:
            eta_y = self._check(ta.eta_y, tb.eta_y) if twiss_ok else no_twiss
        if self.etap_y:
            etap_y = self._check(ta.etap_y, tb.etap_y) if twiss_ok else no_twiss

        oa, ob = ea.orbit, eb.orbit
        orbit_ok = oa is not None and ob is not None
        if self.p0c:
            p0c = (
                self._check(oa.p0c, ob.p0c)
                if orbit_ok
                else CheckResult(passed=False, detail="orbit data not available")
            )

        if self.ref_energy:
            ref_energy_ok = False
            e_tot_a = e_tot_b = 0.0
            if ea.attrs is not None and eb.attrs is not None:
                try:
                    e_tot_a = float(ea.attrs["e_tot"].data)
                    e_tot_b = float(eb.attrs["e_tot"].data)
                    ref_energy_ok = True
                except (KeyError, TypeError, ValueError):
                    pass
            ref_energy = (
                self._check(e_tot_a, e_tot_b)
                if ref_energy_ok
                else CheckResult(passed=False, detail="ref_energy data not available")
            )

        fa = ea.floor.end.actual if ea.floor is not None else None
        fb = eb.floor.end.actual if eb.floor is not None else None
        floor_ok = fa is not None and fb is not None
        no_floor = CheckResult(passed=False, detail="floor data not available")
        if self.floor_x:
            floor_x = self._check(fa.x, fb.x) if floor_ok else no_floor
        if self.floor_y:
            floor_y = self._check(fa.y, fb.y) if floor_ok else no_floor
        if self.floor_z:
            floor_z = self._check(fa.z, fb.z) if floor_ok else no_floor

        return EleLessThanResult(
            beta_a=beta_a,
            alpha_a=alpha_a,
            beta_b=beta_b,
            alpha_b=alpha_b,
            eta_x=eta_x,
            etap_x=etap_x,
            eta_y=eta_y,
            etap_y=etap_y,
            ref_energy=ref_energy,
            p0c=p0c,
            floor_x=floor_x,
            floor_y=floor_y,
            floor_z=floor_z,
        )


def _build_ele_observation(
    *,
    beta_a: float | None,
    alpha_a: float | None,
    beta_b: float | None,
    alpha_b: float | None,
    eta_x: float | None,
    etap_x: float | None,
    eta_y: float | None,
    etap_y: float | None,
    p0c: float | None,
    floor_x: float | None,
    floor_y: float | None,
    floor_z: float | None,
) -> EleObservation:
    new_twiss = ElementTwiss(
        **{
            k: v
            for k, v in [
                ("beta_a", beta_a),
                ("alpha_a", alpha_a),
                ("beta_b", beta_b),
                ("alpha_b", alpha_b),
                ("eta_x", eta_x),
                ("etap_x", etap_x),
                ("eta_y", eta_y),
                ("etap_y", etap_y),
            ]
            if v is not None
        }
    )
    new_orbit = ElementOrbit(**{k: v for k, v in [("p0c", p0c)] if v is not None})
    dummy_floor = ElementFloor(
        which="model", where="end", actual=None, reference=None, slaves={}
    )
    new_floor = ElementFloorAll(
        which="model",
        beginning=dummy_floor,
        center=dummy_floor,
        end=ElementFloor(
            which="model",
            where="end",
            actual=ElementFloorPosition.model_construct(
                wmat=np.identity(3),
                **{
                    k: v
                    for k, v in [("x", floor_x), ("y", floor_y), ("z", floor_z)]
                    if v is not None
                },
            ),
            reference=None,
            slaves={},
        ),
    )
    element = Element.model_construct(
        ele_id="_literal",
        which="model",
        head=tao_classes.ElementHead(name="_literal", key="literal"),
        twiss=new_twiss,
        orbit=new_orbit,
        floor=new_floor,
    )
    return EleObservation(element=element)


class EleLiteral(LiteralObservable[EleObservation]):
    """Literal element observable with user-specified field values.

    Only non-``None`` fields are included in the produced observation.

    Attributes
    ----------
    obs_type : str
        Discriminator literal. Always ``"ele_literal"``.
    beta_a : float or None
        Mode A beta function.
    alpha_a : float or None
        Mode A alpha function.
    beta_b : float or None
        Mode B beta function.
    alpha_b : float or None
        Mode B alpha function.
    eta_x : float or None
        Horizontal dispersion.
    etap_x : float or None
        Horizontal dispersion slope.
    eta_y : float or None
        Vertical dispersion.
    etap_y : float or None
        Vertical dispersion slope.
    p0c : float or None
        Reference momentum.
    floor_x : float or None
        Global floor x coordinate.
    floor_y : float or None
        Global floor y coordinate.
    floor_z : float or None
        Global floor z coordinate.
    """

    obs_type: Literal["ele_literal"] = "ele_literal"
    beta_a: float | None = None
    alpha_a: float | None = None
    beta_b: float | None = None
    alpha_b: float | None = None
    eta_x: float | None = None
    etap_x: float | None = None
    eta_y: float | None = None
    etap_y: float | None = None
    p0c: float | None = None
    floor_x: float | None = None
    floor_y: float | None = None
    floor_z: float | None = None

    @property
    def label(self) -> str:
        return "literal"

    def _make_observation(self) -> EleObservation:
        return _build_ele_observation(**self.model_dump(exclude={"obs_type"}))


def _ele_reduce(
    tao: Tao, reduce_fn, ix_uni: str = "1", ix_branch: str = "0"
) -> EleObservation:
    beta_a = alpha_a = beta_b = alpha_b = None
    eta_x = etap_x = eta_y = etap_y = None
    p0c = None
    floor_x = floor_y = floor_z = None

    for ix_ele in tao.lat_list("*", "ele.ix_ele", ix_uni=ix_uni, ix_branch=ix_branch):
        ele = tao.ele(ix_ele, ix_uni=ix_uni, ix_branch=ix_branch)
        if ele.twiss is not None:
            t = ele.twiss
            beta_a = reduce_fn(beta_a, t.beta_a) if beta_a is not None else t.beta_a
            alpha_a = reduce_fn(alpha_a, t.alpha_a) if alpha_a is not None else t.alpha_a
            beta_b = reduce_fn(beta_b, t.beta_b) if beta_b is not None else t.beta_b
            alpha_b = reduce_fn(alpha_b, t.alpha_b) if alpha_b is not None else t.alpha_b
            eta_x = reduce_fn(eta_x, t.eta_x) if eta_x is not None else t.eta_x
            etap_x = reduce_fn(etap_x, t.etap_x) if etap_x is not None else t.etap_x
            eta_y = reduce_fn(eta_y, t.eta_y) if eta_y is not None else t.eta_y
            etap_y = reduce_fn(etap_y, t.etap_y) if etap_y is not None else t.etap_y
        if ele.orbit is not None:
            p0c = reduce_fn(p0c, ele.orbit.p0c) if p0c is not None else ele.orbit.p0c
        if ele.floor is not None and ele.floor.end.actual is not None:
            fa = ele.floor.end.actual
            floor_x = reduce_fn(floor_x, fa.x) if floor_x is not None else fa.x
            floor_y = reduce_fn(floor_y, fa.y) if floor_y is not None else fa.y
            floor_z = reduce_fn(floor_z, fa.z) if floor_z is not None else fa.z

    return _build_ele_observation(
        beta_a=beta_a,
        alpha_a=alpha_a,
        beta_b=beta_b,
        alpha_b=alpha_b,
        eta_x=eta_x,
        etap_x=etap_x,
        eta_y=eta_y,
        etap_y=etap_y,
        p0c=p0c,
        floor_x=floor_x,
        floor_y=floor_y,
        floor_z=floor_z,
    )


class EleObservable(LatticeObservable[EleObservation]):
    """Observable that fetches element data from the lattice.

    Attributes
    ----------
    obs_type : str
        Discriminator literal. Always ``"ele"``.
    ele_id : str or int
        Element index or name passed to ``tao.ele()``.
    ix_uni : int
        Universe index.
    ix_branch : int
        Branch index.
    """

    obs_type: Literal["ele"] = "ele"
    ele_id: str | int
    ix_uni: int = Field(default=1, ge=0)
    ix_branch: int = Field(default=0, ge=0)

    @property
    def label(self) -> str:
        suffix = (
            f"@{self.ix_uni}:{self.ix_branch}"
            if self.ix_uni != 1 or self.ix_branch != 0
            else ""
        )
        return f"{self.lattice_id}[{self.ele_id}{suffix}]"

    def _make_observation(self, tao: Tao) -> EleObservation:
        return EleObservation(
            element=tao.ele(
                self.ele_id, ix_uni=str(self.ix_uni), ix_branch=str(self.ix_branch)
            )
        )


class EleMaxObservable(LatticeObservable[EleObservation]):
    """Observable yielding the per-field maximum across all tracking elements.

    Attributes
    ----------
    obs_type : str
        Discriminator literal. Always ``"ele_max"``.
    ix_uni : int
        Universe index.
    ix_branch : int
        Branch index.
    """

    obs_type: Literal["ele_max"] = "ele_max"
    ix_uni: int = Field(default=1, ge=0)
    ix_branch: int = Field(default=0, ge=0)

    @property
    def label(self) -> str:
        suffix = (
            f"@{self.ix_uni}:{self.ix_branch}"
            if self.ix_uni != 1 or self.ix_branch != 0
            else ""
        )
        return f"{self.lattice_id}[max{suffix}]"

    def _make_observation(self, tao: Tao) -> EleObservation:
        return _ele_reduce(tao, max, ix_uni=str(self.ix_uni), ix_branch=str(self.ix_branch))


class EleMinObservable(LatticeObservable[EleObservation]):
    """Observable yielding the per-field minimum across all tracking elements.

    Attributes
    ----------
    obs_type : str
        Discriminator literal. Always ``"ele_min"``.
    ix_uni : int
        Universe index.
    ix_branch : int
        Branch index.
    """

    obs_type: Literal["ele_min"] = "ele_min"
    ix_uni: int = Field(default=1, ge=0)
    ix_branch: int = Field(default=0, ge=0)

    @property
    def label(self) -> str:
        suffix = (
            f"@{self.ix_uni}:{self.ix_branch}"
            if self.ix_uni != 1 or self.ix_branch != 0
            else ""
        )
        return f"{self.lattice_id}[min{suffix}]"

    def _make_observation(self, tao: Tao) -> EleObservation:
        return _ele_reduce(tao, min, ix_uni=str(self.ix_uni), ix_branch=str(self.ix_branch))
