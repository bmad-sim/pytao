from abc import abstractmethod
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from pytao.constraints.observables import IsCloseResult, Observable, Observation
from pytao.constraints.observables.datum import DatumIsClose, DatumIsCloseResult, DatumObservable, DatumObservation
from pytao.constraints.observables.ele import EleIsClose, EleIsCloseResult, EleObservable, EleObservation, TolComparison


class EqualityConstraint(BaseModel):
    """Abstract base for equality constraints between two observables."""

    @property
    @abstractmethod
    def obs_a(self) -> Observable: ...

    @property
    @abstractmethod
    def obs_b(self) -> Observable: ...

    @abstractmethod
    def compare(self, obs_a: Observation, obs_b: Observation) -> IsCloseResult: ...


class ElementPair(EqualityConstraint):
    constraint_type: Literal["ele"] = "ele"
    ele_a: EleObservable
    ele_b: EleObservable
    comparison: EleIsClose = Field(default_factory=EleIsClose)

    @property
    def obs_a(self) -> EleObservable:
        return self.ele_a

    @property
    def obs_b(self) -> EleObservable:
        return self.ele_b

    def compare(self, obs_a: Observation, obs_b: Observation) -> EleIsCloseResult:
        return self.comparison(obs_a, obs_b)


class DatumPair(EqualityConstraint):
    constraint_type: Literal["datum"] = "datum"
    datum_a: DatumObservable
    datum_b: DatumObservable
    comparison: DatumIsClose = Field(default_factory=DatumIsClose)

    @property
    def obs_a(self) -> DatumObservable:
        return self.datum_a

    @property
    def obs_b(self) -> DatumObservable:
        return self.datum_b

    def compare(self, obs_a: Observation, obs_b: Observation) -> DatumIsCloseResult:
        return self.comparison(obs_a, obs_b)


class EleLiteral(EqualityConstraint):
    constraint_type: Literal["ele_literal"] = "ele_literal"
    ele: EleObservable
    beta_a: float | None = None
    alpha_a: float | None = None
    beta_b: float | None = None
    alpha_b: float | None = None
    eta_x: float | None = None
    etap_x: float | None = None
    eta_y: float | None = None
    etap_y: float | None = None
    ref_energy: float | None = None
    p0c: float | None = None
    floor_x: float | None = None
    floor_y: float | None = None
    floor_z: float | None = None
    comparison: TolComparison = Field(default_factory=TolComparison)

    @property
    def obs_a(self) -> EleObservable:
        return self.ele

    @property
    def obs_b(self) -> EleObservable:
        return self.ele

    def compare(self, obs_a: Observation, obs_b: Observation) -> EleIsCloseResult:
        assert isinstance(obs_a, EleObservation)
        ea = obs_a.element

        twiss_a = None
        twiss_b = None
        eta_x = None
        etap_x = None
        eta_y = None
        etap_y = None
        ref_energy = None
        p0c = None
        floor_x = None
        floor_y = None
        floor_z = None

        if ea.twiss is not None:
            ta = ea.twiss
            twiss_a_checks = [
                self.comparison(ta.beta_a, self.beta_a) if self.beta_a is not None else None,
                self.comparison(ta.alpha_a, self.alpha_a) if self.alpha_a is not None else None,
            ]
            twiss_a_checks = [c for c in twiss_a_checks if c is not None]
            twiss_a = next((c for c in twiss_a_checks if not c.passed), twiss_a_checks[0] if twiss_a_checks else None)
            twiss_b_checks = [
                self.comparison(ta.beta_b, self.beta_b) if self.beta_b is not None else None,
                self.comparison(ta.alpha_b, self.alpha_b) if self.alpha_b is not None else None,
            ]
            twiss_b_checks = [c for c in twiss_b_checks if c is not None]
            twiss_b = next((c for c in twiss_b_checks if not c.passed), twiss_b_checks[0] if twiss_b_checks else None)
            if self.eta_x is not None:
                eta_x = self.comparison(ta.eta_x, self.eta_x)
            if self.etap_x is not None:
                etap_x = self.comparison(ta.etap_x, self.etap_x)
            if self.eta_y is not None:
                eta_y = self.comparison(ta.eta_y, self.eta_y)
            if self.etap_y is not None:
                etap_y = self.comparison(ta.etap_y, self.etap_y)

        if ea.orbit is not None:
            oa = ea.orbit
            if self.p0c is not None:
                p0c = self.comparison(oa.p0c, self.p0c)

        if ea.attrs is not None and self.ref_energy is not None:
            try:
                e_tot = float(ea.attrs["e_tot"].data)
            except (KeyError, TypeError, ValueError):
                pass
            else:
                ref_energy = self.comparison(e_tot, self.ref_energy)

        if ea.floor is not None:
            fa = ea.floor.end.actual
            if fa is not None:
                if self.floor_x is not None:
                    floor_x = self.comparison(fa.x, self.floor_x)
                if self.floor_y is not None:
                    floor_y = self.comparison(fa.y, self.floor_y)
                if self.floor_z is not None:
                    floor_z = self.comparison(fa.z, self.floor_z)

        ran = [r for r in [twiss_a, twiss_b, eta_x, etap_x, eta_y, etap_y,
                            ref_energy, p0c, floor_x, floor_y, floor_z]
               if r is not None]

        return EleIsCloseResult(
            is_close=all(ran) if ran else True,
            twiss_a=twiss_a,
            twiss_b=twiss_b,
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


class DatumLiteral(EqualityConstraint):
    constraint_type: Literal["datum_literal"] = "datum_literal"
    datum: DatumObservable
    expected_model_value: float | None = None
    expected_design_value: float | None = None
    comparison: DatumIsClose = Field(default_factory=DatumIsClose)

    @property
    def obs_a(self) -> DatumObservable:
        return self.datum

    @property
    def obs_b(self) -> DatumObservable:
        return self.datum

    def compare(self, obs_a: Observation, obs_b: Observation) -> DatumIsCloseResult:
        literal = DatumObservation(
            model_value=self.expected_model_value if self.expected_model_value is not None else obs_a.model_value,
            design_value=self.expected_design_value if self.expected_design_value is not None else obs_a.design_value,
        )
        return self.comparison(obs_a, literal)


equality_constraint_types = Annotated[Union[ElementPair, EleLiteral, DatumPair, DatumLiteral], Field(discriminator="constraint_type")]


class LatticeConfig(BaseModel):
    lattice_file: str | None = None
    init_file: str | None = None


class ConstraintsConfig(BaseModel):
    lattices: dict[str, LatticeConfig] = Field(default_factory=dict, description="Mapping from unique lattice identifier to lattice loading information")
    equality_constraints: list[equality_constraint_types] = Field(default_factory=list, description="Equality constraints to check across lattices")
