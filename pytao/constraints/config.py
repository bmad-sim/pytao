from abc import abstractmethod
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from pytao.constraints.observables import IsCloseResult, Observable, Observation
from pytao.constraints.observables.datum import DatumIsClose, DatumIsCloseResult, DatumObservable, DatumObservation
from pytao.model import ElementFloor, ElementFloorAll, ElementFloorPosition, ElementOrbit, ElementTwiss

from pytao.model.ele.ele import Element

from pytao.constraints.observables.ele import EleIsClose, EleIsCloseResult, EleObservable, EleObservation


class EleLiteralValues(BaseModel):
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

    def to_observation(self, base: Element) -> Element:
        new_twiss = ElementTwiss(
            **{k: v for k, v in [
                ("beta_a", self.beta_a), ("alpha_a", self.alpha_a),
                ("beta_b", self.beta_b), ("alpha_b", self.alpha_b),
                ("eta_x", self.eta_x), ("etap_x", self.etap_x),
                ("eta_y", self.eta_y), ("etap_y", self.etap_y),
            ] if v is not None}
        )
        new_orbit = ElementOrbit(
            **{k: v for k, v in [("p0c", self.p0c)] if v is not None}
        )
        dummy_floor = ElementFloor(which="model", where="end", actual=None, reference=None, slaves={})
        new_floor = ElementFloorAll(
            which="model",
            beginning=dummy_floor,
            center=dummy_floor,
            end=ElementFloor(
                which="model",
                where="end",
                actual=ElementFloorPosition(
                    **{k: v for k, v in [
                        ("x", self.floor_x), ("y", self.floor_y), ("z", self.floor_z)
                    ] if v is not None}
                ),
                reference=None,
                slaves={},
            ),
        )
        element = base.model_copy(update={"twiss": new_twiss, "orbit": new_orbit, "floor": new_floor})
        return EleObservation(element=element)


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
    expected: EleLiteralValues
    comparison: EleIsClose = Field(default_factory=EleIsClose)

    @property
    def obs_a(self) -> EleObservable:
        return self.ele

    @property
    def obs_b(self) -> EleObservable:
        return self.ele

    def compare(self, obs_a: Observation, obs_b: Observation) -> EleIsCloseResult:
        if not isinstance(obs_a, EleObservation):
            raise TypeError(f"expected EleObservation, got {type(obs_a)}")
        return self.comparison(obs_a, lement=self.expected.to_observation(obs_a.element))


class DatumLiteralValues(BaseModel):
    model_value: float | None = None
    design_value: float | None = None

    def to_observation(self, base: DatumObservation) -> DatumObservation:
        return DatumObservation(
            model_value=self.model_value if self.model_value is not None else base.model_value,
            design_value=self.design_value if self.design_value is not None else base.design_value,
        )


class DatumLiteral(EqualityConstraint):
    constraint_type: Literal["datum_literal"] = "datum_literal"
    datum: DatumObservable
    expected: DatumLiteralValues
    comparison: DatumIsClose = Field(default_factory=DatumIsClose)

    @property
    def obs_a(self) -> DatumObservable:
        return self.datum

    @property
    def obs_b(self) -> DatumObservable:
        return self.datum

    def compare(self, obs_a: Observation, obs_b: Observation) -> DatumIsCloseResult:
        if not isinstance(obs_a, DatumObservation):
            raise TypeError(f"expected DatumObservation, got {type(obs_a)}")
        literal = self.expected.to_observation(obs_a)
        return self.comparison(obs_a, literal)


equality_constraint_types = Annotated[Union[ElementPair, EleLiteral, DatumPair, DatumLiteral], Field(discriminator="constraint_type")]


class LatticeConfig(BaseModel):
    lattice_file: str | None = None
    init_file: str | None = None


class ConstraintsConfig(BaseModel):
    lattices: dict[str, LatticeConfig] = Field(default_factory=dict, description="Mapping from unique lattice identifier to lattice loading information")
    equality_constraints: list[equality_constraint_types] = Field(default_factory=list, description="Equality constraints to check across lattices")
