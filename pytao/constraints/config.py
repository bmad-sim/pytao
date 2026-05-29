from abc import abstractmethod
from typing import Annotated, Literal, Union

from pydantic import Field

from pytao.constraints.pydantic import ConstraintsBase

from pytao.constraints.observables import (
    ComparisonResult,
    DatumIsClose,
    DatumIsCloseResult,
    DatumLessThan,
    DatumLessThanResult,
    DatumLiteral,
    DatumObservable,
    DatumObservation,
    EleIsClose,
    EleIsCloseResult,
    EleLessThan,
    EleLessThanResult,
    EleLiteral,
    EleMaxObservable,
    EleMinObservable,
    EleObservable,
    EleObservation,
    IsClose,
    IsLess,
    LatticeObservable,
    LiteralObservable,
    Observable,
    Observation,
)
from pytao.startup import TaoStartup


EleObservables = Annotated[
    Union[EleObservable, EleMaxObservable, EleMinObservable, EleLiteral],
    Field(discriminator="obs_type"),
]

DatumObservables = Annotated[
    Union[DatumObservable, DatumLiteral],
    Field(discriminator="obs_type"),
]


class Constraint(ConstraintsBase):
    """Abstract base for all constraint types."""

    description: str = Field(default="", description="Short one-line name used on labels")
    comment: str = Field(
        default="", description="Detailed description or notes about the constraint"
    )

    @property
    @abstractmethod
    def label(self) -> str: ...

    @property
    @abstractmethod
    def required_observables(self) -> frozenset[Observable]: ...

    @abstractmethod
    def error_result(self, error: str) -> ComparisonResult: ...


class ComparisonConstraint(Constraint):
    """Base for constraints that compare two observations against each other."""

    @abstractmethod
    def is_satisfied(
        self, observations: dict[Observable, Observation]
    ) -> ComparisonResult: ...


class IsCloseConstraint(ComparisonConstraint):
    """Base for constraints that use an IsClose comparison operator."""

    comparison: IsClose


class IsLessConstraint(ComparisonConstraint):
    """Base for constraints that use an IsLess comparison operator."""

    comparison: IsLess


class RegressionConstraint(Constraint):
    """Base for constraints that compare current observations against a saved reference."""

    comparison: IsClose

    @abstractmethod
    def evaluate(self, current: Observation, reference: Observation) -> ComparisonResult: ...


class EleIsCloseConstraint(IsCloseConstraint):
    """Constraint checking that two element observables are approximately equal.

    Attributes
    ----------
    obs_a : EleObservables
        First element observable.
    obs_b : EleObservables
        Second element observable.
    comparison : EleIsClose
        Comparison operator applied to the two observations.
    """

    constraint_type: Literal["ele_eq"] = "ele_eq"
    obs_a: EleObservables
    obs_b: EleObservables
    comparison: EleIsClose = EleIsClose()

    @property
    def label(self) -> str:
        if self.obs_a == self.obs_b:
            return self.obs_a.label
        return f"{self.obs_a.label} == {self.obs_b.label}"

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset((self.obs_a, self.obs_b))

    def is_satisfied(self, observations: dict[Observable, Observation]) -> EleIsCloseResult:
        return self.comparison(observations[self.obs_a], observations[self.obs_b])

    def error_result(self, error: str) -> EleIsCloseResult:
        return EleIsCloseResult(error=error)


class EleLessThanConstraint(IsLessConstraint):
    """Constraint checking that ``obs_a`` is component-wise less than ``obs_b``.

    Attributes
    ----------
    obs_a : EleObservables
        Left-hand side observable.
    obs_b : EleObservables
        Right-hand side observable.
    comparison : EleLessThan
        Component selector and less-than operator.
    """

    constraint_type: Literal["ele_lt"] = "ele_lt"
    obs_a: EleObservables
    obs_b: EleObservables
    comparison: EleLessThan = EleLessThan()

    @property
    def label(self) -> str:
        return f"{self.obs_a.label} < {self.obs_b.label}"

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset((self.obs_a, self.obs_b))

    def is_satisfied(self, observations: dict[Observable, Observation]) -> EleLessThanResult:
        return self.comparison(observations[self.obs_a], observations[self.obs_b])

    def error_result(self, error: str) -> EleLessThanResult:
        return EleLessThanResult(error=error)


class DatumIsCloseConstraint(IsCloseConstraint):
    constraint_type: Literal["datum_eq"] = "datum_eq"
    obs_a: DatumObservables
    obs_b: DatumObservables
    comparison: DatumIsClose = DatumIsClose()

    @property
    def label(self) -> str:
        if self.obs_a == self.obs_b:
            return self.obs_a.label
        return f"{self.obs_a.label} == {self.obs_b.label}"

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset((self.obs_a, self.obs_b))

    def is_satisfied(self, observations: dict[Observable, Observation]) -> DatumIsCloseResult:
        return self.comparison(observations[self.obs_a], observations[self.obs_b])

    def error_result(self, error: str) -> DatumIsCloseResult:
        return DatumIsCloseResult(error=error)


class DatumLessThanConstraint(IsLessConstraint):
    constraint_type: Literal["datum_lt"] = "datum_lt"
    obs_a: DatumObservables
    obs_b: DatumObservables
    comparison: DatumLessThan = DatumLessThan()

    @property
    def label(self) -> str:
        return f"{self.obs_a.label} < {self.obs_b.label}"

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset((self.obs_a, self.obs_b))

    def is_satisfied(self, observations: dict[Observable, Observation]) -> DatumLessThanResult:
        return self.comparison(observations[self.obs_a], observations[self.obs_b])

    def error_result(self, error: str) -> DatumLessThanResult:
        return DatumLessThanResult(error=error)


class EleRegressionConstraint(RegressionConstraint):
    """Constraint comparing current element observations against a saved reference.

    Attributes
    ----------
    obs : EleObservables
        Element observable to evaluate and compare.
    comparison : EleIsClose
        Comparison operator used to check current against reference.
    """

    constraint_type: Literal["ele_reg"] = "ele_reg"
    obs: EleObservables
    comparison: EleIsClose = EleIsClose()

    @property
    def label(self) -> str:
        return self.obs.label

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset({self.obs})

    def evaluate(self, current: EleObservation, reference: EleObservation) -> EleIsCloseResult:
        return self.comparison(current, reference)

    def error_result(self, error: str) -> EleIsCloseResult:
        return EleIsCloseResult(error=error)


class DatumRegressionConstraint(RegressionConstraint):
    constraint_type: Literal["datum_reg"] = "datum_reg"
    obs: DatumObservables
    comparison: DatumIsClose = DatumIsClose()

    @property
    def label(self) -> str:
        return self.obs.label

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset({self.obs})

    def evaluate(
        self, current: DatumObservation, reference: DatumObservation
    ) -> DatumIsCloseResult:
        return self.comparison(current, reference)

    def error_result(self, error: str) -> DatumIsCloseResult:
        return DatumIsCloseResult(error=error)


AnyConstraint = Annotated[
    Union[
        EleIsCloseConstraint,
        EleLessThanConstraint,
        DatumIsCloseConstraint,
        DatumLessThanConstraint,
        EleRegressionConstraint,
        DatumRegressionConstraint,
    ],
    Field(discriminator="constraint_type"),
]


class ConstraintsConfig(ConstraintsBase):
    lattices: dict[str, TaoStartup] = Field(
        default_factory=dict,
        description="Mapping from unique lattice identifier to lattice loading information",
    )
    constraints: list[AnyConstraint] | dict[str, list[AnyConstraint]] = Field(
        default_factory=list,
        description="Flat list (ungrouped) or mapping of group name to list of constraints",
    )

    @property
    def constraints_by_group(self) -> dict[str | None, list[AnyConstraint]]:
        if isinstance(self.constraints, list):
            return {None: self.constraints}
        return dict(self.constraints)

    @property
    def all_constraints(self) -> list[AnyConstraint]:
        if isinstance(self.constraints, list):
            return self.constraints
        return [c for cs in self.constraints.values() for c in cs]

    @property
    def required_lattice_observables(self) -> dict[str, set[LatticeObservable]]:
        needed: dict[str, set[LatticeObservable]] = {lat_id: set() for lat_id in self.lattices}
        for constraint in self.all_constraints:
            for obs in constraint.required_observables:
                if isinstance(obs, LatticeObservable):
                    needed[obs.lattice_id].add(obs)
        return needed

    @property
    def required_literal_observables(self) -> set[LiteralObservable]:
        return {
            obs
            for constraint in self.all_constraints
            for obs in constraint.required_observables
            if isinstance(obs, LiteralObservable)
        }
