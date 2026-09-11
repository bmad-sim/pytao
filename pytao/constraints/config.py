from abc import abstractmethod
from typing import Annotated, Any, Generic, Literal, TypeVar, Union, cast

from pydantic import Field, model_validator

from pytao.constraints.pydantic import ConstraintsBase

from pytao.constraints.observables import (
    AnyComparison,
    Comparison,
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
from pytao.constraints.results import ConstraintResult, RegressionResult
from pytao.startup import TaoStartup

CompT = TypeVar("CompT", bound=Comparison[Any])


EleObservables = Annotated[
    Union[EleObservable, EleMaxObservable, EleMinObservable, EleLiteral],
    Field(discriminator="obs_type"),
]

DatumObservables = Annotated[
    Union[DatumObservable, DatumLiteral],
    Field(discriminator="obs_type"),
]


class Constraint(ConstraintsBase):
    """Abstract base for all constraint types.

    Attributes
    ----------
    description : str
        Short one-line name used on labels.
    comment : str
        Detailed notes about the constraint.
    """

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

    @abstractmethod
    def run(
        self,
        obs_map: dict[Observable, Observation],
        expected_obs_map: dict[Observable, Observation] | None,
        common_comparison_map: dict[str, AnyComparison],
        group: str | None,
    ) -> tuple[list[ConstraintResult], list[RegressionResult]]: ...


class ComparisonConstraint(Constraint, Generic[CompT]):
    """Base for constraints that compare two observations against each other."""

    comparison: CompT | str

    # comparison object to be filled by `.run()`
    _comparison_obj: CompT | None = None

    def is_satisfied(self, observations: dict[Observable, Observation]) -> ComparisonResult:
        if self._comparison_obj is None:
            raise RuntimeError(
                "Comparison has not been run, cannot return a meaningful result"
            )

    def run(
        self,
        obs_map: dict[Observable, Observation],
        expected_obs_map: dict[Observable, Observation] | None,
        common_comparisons_map: dict[str, AnyComparison],
        group: str | None,
    ) -> tuple[list[ConstraintResult], list[RegressionResult]]:
        # replace string comparison reference with real comparison
        if isinstance(self.comparison, str):
            if self.comparison not in common_comparisons_map:
                raise ValueError(f"Referenced comparison ({self.comparison}) not defined")
            self._comparison_obj = cast(CompT, common_comparisons_map[self.comparison])
        else:
            self._comparison_obj = self.comparison

        missing = [obs for obs in self.required_observables if obs not in obs_map]
        if missing:
            missing_labels = ", ".join(obs.label for obs in missing)
            result = self.error_result(f"Missing observations: {missing_labels}")
        else:
            result = self.is_satisfied(
                {obs: obs_map[obs] for obs in self.required_observables}
            )
        cr = ConstraintResult(
            label=self.label,
            observables=list(self.required_observables),
            description=self.description,
            comment=self.comment,
            result=result,
        )
        return [cr], []


class IsCloseConstraint(ComparisonConstraint[CompT]):
    """Base for constraints that use an IsClose comparison operator.

    When ``regression_check`` is ``True`` and a comparison baseline is available,
    each required observable is also compared against its saved value using the
    same ``comparison`` operator.

    Attributes
    ----------
    comparison : IsClose
        Operator used to evaluate approximate equality between two observations.
    regression_check : bool
        Whether to implicitly define regression checks on the observations from this constraint.
    """

    comparison: CompT | str
    regression_check: bool = True

    def run(
        self,
        obs_map: dict[Observable, Observation],
        expected_obs_map: dict[Observable, Observation] | None,
        common_comparison_map: dict[str, AnyComparison],
        group: str | None,
    ) -> tuple[list[ConstraintResult], list[RegressionResult]]:
        """
        Run the constraint, determining if the observables are close

        Parameters
        ----------
        obs_map : dict[Observable, Observation]
            mapping specified observable -> complete observation
        expected_obs_map : dict[Observable, Observation] | None
            mapping observable -> complete observation from e.g. lattice
        common_comparison_map : dict[str, AnyComparison]
            mapping string -> shared comparisons
        group : str | None
            Name of group this constraint belongs to, Optional

        Returns
        -------
        tuple[list[ConstraintResult], list[RegressionResult]]

        Raises
        ------
        ValueError
            if comparison is a reference (string) that is not defined
        TypeError
            if comparison references a non-IsClose comparison
        """
        crs, _ = super().run(obs_map, expected_obs_map, common_comparison_map, group)
        reg: list[RegressionResult] = []
        if not isinstance(self._comparison_obj, IsClose):
            raise TypeError(
                f"Referenced comparison ({self._comparison_obj}) is of "
                f"incorrect type: {type(self._comparison_obj)}"
            )

        if self.regression_check and expected_obs_map is not None:
            for obs in self.required_observables:
                if obs not in obs_map or obs not in expected_obs_map:
                    reg_result = self.error_result("Missing observation")
                else:
                    reg_result = self._comparison_obj.compare(
                        obs_map[obs], expected_obs_map[obs]
                    )
                reg.append(
                    RegressionResult(
                        group=group,
                        label=self.label,
                        description=self.description,
                        comment=self.comment,
                        observable=obs,
                        result=reg_result,
                    )
                )
        return crs, reg


class IsLessConstraint(ComparisonConstraint[CompT]):
    """Base for constraints that use an IsLess comparison operator.

    Attributes
    ----------
    comparison : IsLess
        Operator used to evaluate component-wise less-than between two observations.
    """

    comparison: CompT | str

    def run(
        self,
        obs_map: dict[Observable, Observation],
        expected_obs_map: dict[Observable, Observation] | None,
        common_comparison_map: dict[str, AnyComparison],
        group: str | None,
    ) -> tuple[list[ConstraintResult], list[RegressionResult]]:
        crs, reg = super().run(obs_map, expected_obs_map, common_comparison_map, group)
        if not isinstance(self._comparison_obj, IsLess):
            raise TypeError(
                f"Referenced comparison ({self._comparison_obj}) is of "
                f"incorrect type: {type(self._comparison_obj)}"
            )

        return crs, reg


class RegressionConstraint(Constraint, Generic[CompT]):
    """Base for constraints that compare current observations against a saved reference.

    Attributes
    ----------
    comparison : IsClose
        Operator used to compare the current observation against the reference.
    """

    comparison: CompT | str

    # comparison object to be filled by `.run()`
    _comparison_obj: CompT | None = None

    @abstractmethod
    def evaluate(self, current: Observation, reference: Observation) -> ComparisonResult: ...

    def run(
        self,
        obs_map: dict[Observable, Observation],
        expected_obs_map: dict[Observable, Observation] | None,
        common_comparison_map: dict[str, AnyComparison],
        group: str | None,
    ) -> tuple[list[ConstraintResult], list[RegressionResult]]:
        if expected_obs_map is None:
            return [], []
        obs = next(iter(self.required_observables))
        if isinstance(self.comparison, str):
            if self.comparison not in common_comparison_map:
                raise ValueError(f"Referenced comparison ({self.comparison}) not defined")
            ref_comp = common_comparison_map[self.comparison]
            if not isinstance(ref_comp, IsClose):
                raise TypeError(
                    f"Referenced comparison ({self.comparison}) is of "
                    f"incorrect type: {type(ref_comp)}"
                )
            self._comparison_obj = cast(CompT, ref_comp)
        else:
            self._comparison_obj = self.comparison

        if obs not in obs_map or obs not in expected_obs_map:
            result = self.error_result("Missing observation")
        else:
            result = self.evaluate(obs_map[obs], expected_obs_map[obs])
        return [], [
            RegressionResult(
                group=group,
                label=self.label,
                description=self.description,
                comment=self.comment,
                observable=obs,
                result=result,
            )
        ]


class EleIsCloseConstraint(IsCloseConstraint[EleIsClose]):
    """Constraint checking that two element observables are approximately equal.

    Attributes
    ----------
    constraint_type : str
        Discriminator literal. Always ``"ele_eq"``.
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
    comparison: EleIsClose | str = EleIsClose()

    @property
    def label(self) -> str:
        if self.obs_a == self.obs_b:
            return self.obs_a.label
        return f"{self.obs_a.label} == {self.obs_b.label}"

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset((self.obs_a, self.obs_b))

    def is_satisfied(self, observations: dict[Observable, Observation]) -> EleIsCloseResult:
        super().is_satisfied(observations=observations)
        return self._comparison_obj.compare(observations[self.obs_a], observations[self.obs_b])

    def error_result(self, error: str) -> EleIsCloseResult:
        return EleIsCloseResult(error=error)


class EleLessThanConstraint(IsLessConstraint[EleLessThan]):
    """Constraint checking that ``obs_a`` is component-wise less than ``obs_b``.

    Attributes
    ----------
    constraint_type : str
        Discriminator literal. Always ``"ele_lt"``.
    obs_a : EleObservables
        Left-hand side observable.
    obs_b : EleObservables
        Right-hand side observable.
    comparison : EleLessThan
        Less-than operator configuration.
    """

    constraint_type: Literal["ele_lt"] = "ele_lt"
    obs_a: EleObservables
    obs_b: EleObservables
    comparison: EleLessThan | str = EleLessThan()

    @property
    def label(self) -> str:
        return f"{self.obs_a.label} < {self.obs_b.label}"

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset((self.obs_a, self.obs_b))

    def is_satisfied(self, observations: dict[Observable, Observation]) -> EleLessThanResult:
        super().is_satisfied(observations=observations)
        return self._comparison_obj.compare(observations[self.obs_a], observations[self.obs_b])

    def error_result(self, error: str) -> EleLessThanResult:
        return EleLessThanResult(error=error)


class DatumIsCloseConstraint(IsCloseConstraint[DatumIsClose]):
    """Constraint checking that two datum observables are approximately equal.

    Attributes
    ----------
    constraint_type : str
        Discriminator literal. Always ``"datum_eq"``.
    obs_a : DatumObservables
        First datum observable.
    obs_b : DatumObservables
        Second datum observable.
    comparison : DatumIsClose
        Comparison operator applied to the two observations.
    """

    constraint_type: Literal["datum_eq"] = "datum_eq"
    obs_a: DatumObservables
    obs_b: DatumObservables
    comparison: DatumIsClose | str = DatumIsClose()

    @property
    def label(self) -> str:
        if self.obs_a == self.obs_b:
            return self.obs_a.label
        return f"{self.obs_a.label} == {self.obs_b.label}"

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset((self.obs_a, self.obs_b))

    def is_satisfied(self, observations: dict[Observable, Observation]) -> DatumIsCloseResult:
        super().is_satisfied(observations=observations)
        return self._comparison_obj.compare(observations[self.obs_a], observations[self.obs_b])

    def error_result(self, error: str) -> DatumIsCloseResult:
        return DatumIsCloseResult(error=error)


class DatumLessThanConstraint(IsLessConstraint[DatumLessThan]):
    """Constraint checking that ``obs_a`` is component-wise less than ``obs_b``.

    Attributes
    ----------
    constraint_type : str
        Discriminator literal. Always ``"datum_lt"``.
    obs_a : DatumObservables
        Left-hand side observable.
    obs_b : DatumObservables
        Right-hand side observable.
    comparison : DatumLessThan
        Less-than operator configuration.
    """

    constraint_type: Literal["datum_lt"] = "datum_lt"
    obs_a: DatumObservables
    obs_b: DatumObservables
    comparison: DatumLessThan | str = DatumLessThan()

    @property
    def label(self) -> str:
        return f"{self.obs_a.label} < {self.obs_b.label}"

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset((self.obs_a, self.obs_b))

    def is_satisfied(self, observations: dict[Observable, Observation]) -> DatumLessThanResult:
        super().is_satisfied(observations=observations)
        return self._comparison_obj.compare(observations[self.obs_a], observations[self.obs_b])

    def error_result(self, error: str) -> DatumLessThanResult:
        return DatumLessThanResult(error=error)


class EleRegressionConstraint(RegressionConstraint[EleIsClose]):
    """Constraint comparing current element observations against a saved reference.

    Attributes
    ----------
    constraint_type : str
        Discriminator literal. Always ``"ele_reg"``.
    obs : EleObservables
        Element observable to evaluate and compare.
    comparison : EleIsClose | str
        Comparison operator used to check current against reference.
    """

    constraint_type: Literal["ele_reg"] = "ele_reg"
    obs: EleObservables
    comparison: EleIsClose | str = EleIsClose()

    @property
    def label(self) -> str:
        return self.obs.label

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset({self.obs})

    def evaluate(self, current: EleObservation, reference: EleObservation) -> EleIsCloseResult:
        return self._comparison_obj.compare(current, reference)

    def error_result(self, error: str) -> EleIsCloseResult:
        return EleIsCloseResult(error=error)


class DatumRegressionConstraint(RegressionConstraint[DatumIsClose]):
    """Constraint comparing current datum observations against a saved reference.

    Attributes
    ----------
    constraint_type : str
        Discriminator literal. Always ``"datum_reg"``.
    obs : DatumObservables
        Datum observable to evaluate and compare.
    comparison : DatumIsClose
        Comparison operator used to check current against reference.
    """

    constraint_type: Literal["datum_reg"] = "datum_reg"
    obs: DatumObservables
    comparison: DatumIsClose | str = DatumIsClose()

    @property
    def label(self) -> str:
        return self.obs.label

    @property
    def required_observables(self) -> frozenset[Observable]:
        return frozenset({self.obs})

    def evaluate(
        self, current: DatumObservation, reference: DatumObservation
    ) -> DatumIsCloseResult:
        return self._comparison_obj.compare(current, reference)

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
    comparisons: dict[str, AnyComparison] = Field(
        default_factory=dict,
        description="Mapping from unique comparison identifier to reusable comparison settings",
    )

    @model_validator(mode="before")
    @classmethod
    def _default_lattice_startup(cls, data: Any) -> Any:
        if isinstance(data, dict):
            lattices = data.get("lattices")
            if isinstance(lattices, dict):
                for startup in lattices.values():
                    if isinstance(startup, dict):
                        startup.setdefault("noinit", True)
                        startup.setdefault("noplot", True)
        return data

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
