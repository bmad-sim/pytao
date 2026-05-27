import argparse
import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml

from pytao import SubprocessTao

from .config import ConstraintsConfig, EqualityConstraint
from .observables import (
    ComparisonResult,
    IsCloseResult,
    LatticeObservable,
    LiteralObservable,
    Observable,
    Observation,
)
from .results import (
    ConstraintResult,
    ConstraintResults,
    LatticeResult,
    RegressionResult,
    SavedObservations,
)

logger = logging.getLogger(__name__)


def run(
    config: ConstraintsConfig,
    config_dir: Path,
    compare: SavedObservations | None = None,
) -> tuple[SavedObservations, ConstraintResults]:
    """
    Run all constraints in the given config and return observations and results.

    Parameters
    ----------
    config : ConstraintsConfig
        Parsed constraints configuration.
    config_dir : Path
        Directory used to resolve relative paths in the config.
    compare : SavedObservations, optional
        Previously saved observations for regression comparison.

    Returns
    -------
    tuple[SavedObservations, ConstraintResults]
        Saved lattice observations and the full constraint results.
    """
    started_at = datetime.now(timezone.utc)

    # Build lattice_id -> set of lattice observables, and collect literal observables separately
    needed: dict[str, set[LatticeObservable]] = {lat_id: set() for lat_id in config.lattices}
    literal_obs: set[LiteralObservable] = set()
    for constraint in config.constraints:
        for obs in constraint.required_observables:
            if isinstance(obs, LatticeObservable):
                needed[obs.lattice_id].add(obs)
            elif isinstance(obs, LiteralObservable):
                literal_obs.add(obs)
            else:
                raise ValueError(f"Unrecognized observable type: {type(obs)}")

    # Run observables: observable -> observation
    obs_map: dict[Observable, Observation] = {}
    lattice_results: dict[str, LatticeResult] = {}

    for lat_id, lat_startup in config.lattices.items():
        params = dict(lat_startup.with_path_prefix(config_dir).tao_class_params)
        params["noplot"] = True

        loaded = False
        error: str | None = None
        load_time = 0.0
        obs_time = 0.0
        t0 = time.perf_counter()

        try:
            with SubprocessTao(**params) as tao:
                load_time = time.perf_counter() - t0
                loaded = True
                for obs in needed[lat_id]:
                    try:
                        obs_map[obs] = obs(tao)
                    except Exception:
                        logger.debug(
                            "Observable %r failed for lattice %r:\n%s",
                            obs,
                            lat_id,
                            traceback.format_exc().strip(),
                        )
                obs_time = sum(
                    obs_map[obs].elapsed_time for obs in needed[lat_id] if obs in obs_map
                )
        except Exception:
            if not load_time:
                load_time = time.perf_counter() - t0
            error = traceback.format_exc().strip()

        lattice_results[lat_id] = LatticeResult.from_startup(
            lat_startup,
            loaded=loaded,
            error=error,
            load_time=load_time,
            obs_time=obs_time,
        )

    for obs in literal_obs:
        obs_map[obs] = obs()

    saved = SavedObservations.from_obs_map(obs_map)

    # Run each constraint comparison
    constraint_results: list[ConstraintResult] = []
    for constraint in config.constraints:
        missing = [obs for obs in constraint.required_observables if obs not in obs_map]
        if missing:
            missing_labels = ", ".join(obs.label for obs in missing)
            result = constraint.error_result(f"Missing observations: {missing_labels}")
        else:
            result = constraint.is_satisfied(
                {obs: obs_map[obs] for obs in constraint.required_observables}
            )
        constraint_results.append(
            ConstraintResult(
                observables=list(constraint.required_observables),
                description=constraint.description,
                comment=constraint.comment,
                result=result,
            )
        )

    # Regression comparisons against saved observations
    regression_results: list[RegressionResult] = []
    if compare is not None:
        compare_map = compare.obs_map
        for constraint in config.constraints:
            if not isinstance(constraint, EqualityConstraint):
                continue
            for obs in constraint.required_observables:
                if obs not in obs_map or obs not in compare_map:
                    continue
                try:
                    result = constraint.comparison(obs_map[obs], compare_map[obs])
                except Exception:
                    result = IsCloseResult(
                        is_close=False,
                        error=traceback.format_exc().strip(),
                    )
                regression_results.append(RegressionResult(observable=obs, result=result))

    return saved, ConstraintResults(
        started_at=started_at,
        finished_at=datetime.now(timezone.utc),
        lattices=lattice_results,
        constraints=constraint_results,
        regression=regression_results,
    )


def _print_check_detail(res: ComparisonResult) -> None:
    checks = res.check_results()
    if checks:
        width = max(len(name) for name in checks)
        for name, check in checks.items():
            print(f"    {name:<{width}}  {check.format_detail()}")
    if res.error:
        for line in res.error.splitlines():
            print(f"    {line}")


def _print_results(results: ConstraintResults) -> None:
    print("Lattices:")
    for lat_id, lat in results.lattices.items():
        status = "OK  " if lat.loaded else "FAIL"
        print(
            f"  [{status}] {lat_id}  loaded in {lat.load_time:.2f}s, observables in {lat.obs_time:.2f}s"
        )
        if lat.error:
            for line in lat.error.splitlines():
                print(f"         {line}")

    print()
    print("Constraints:")
    for cr in results.constraints:
        status = "PASS" if cr.result.is_close else "FAIL"
        label = " == ".join(obs.label for obs in cr.observables)
        suffix = f"  {cr.description}" if cr.description else ""
        print(f"  [{status}] {label}{suffix}")

    if results.regression:
        print()
        print("Regression:")
        for rr in results.regression:
            status = "PASS" if rr.result.is_close else "FAIL"
            print(f"  [{status}] {rr.observable.label}")

    failures_eq = [cr for cr in results.constraints if not cr.result.is_close]
    failures_reg = [rr for rr in results.regression if not rr.result.is_close]

    if failures_eq or failures_reg:
        print()
        print("=" * 60)
        print("FAILURES")
        print("=" * 60)
        for cr in failures_eq:
            label = " == ".join(obs.label for obs in cr.observables)
            header = f"{label}  {cr.description}" if cr.description else label
            print(f"\n  {header}")
            if cr.comment:
                print(f"  {cr.comment}")
            print("  " + "-" * 56)
            _print_check_detail(cr.result)
        for rr in failures_reg:
            print(f"\n  regression: {rr.observable.label}")
            print("  " + "-" * 56)
            _print_check_detail(rr.result)

    n_passed = sum(1 for cr in results.constraints if cr.result.is_close)
    n_total = len(results.constraints)
    print()
    print(f"{n_passed}/{n_total} constraints passed")

    if results.regression:
        n_reg_passed = sum(1 for rr in results.regression if rr.result.is_close)
        n_reg_total = len(results.regression)
        print(f"{n_reg_passed}/{n_reg_total} regression checks passed")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pytao-constraints",
        description="Run pytao constraints checks against Bmad lattice files.",
    )
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument(
        "--save-observations",
        metavar="FILE",
        help="Path to write a JSON snapshot of current observations",
    )
    parser.add_argument(
        "--save-results",
        metavar="FILE",
        help="Path to write a JSON snapshot of the results",
    )
    parser.add_argument(
        "--compare-path",
        metavar="FILE",
        help="Path to a previously saved observations JSON for regression comparison",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config_path = Path(args.config).resolve()
    with config_path.open() as fh:
        raw = yaml.safe_load(fh)

    config = ConstraintsConfig.model_validate(raw)

    compare: SavedObservations | None = None
    if args.compare_path:
        compare = SavedObservations.model_validate_json(Path(args.compare_path).read_text())

    save_obs_path = Path(args.save_observations) if args.save_observations else None

    saved, results = run(config, config_dir=config_path.parent, compare=compare)

    _print_results(results)

    if save_obs_path is not None:
        save_obs_path.write_text(saved.model_dump_json(indent=2))
        print(f"\nObservations saved to {save_obs_path}")

    if args.save_results:
        results_path = Path(args.save_results)
        results_path.write_text(results.model_dump_json(indent=2))
        print(f"\nResults saved to {results_path}")
