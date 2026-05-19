import argparse
import time
import traceback
from pathlib import Path

import yaml

from pytao import SubprocessTao

from .config import UnittestConfig
from .observables import EleIsCloseResult, IsCloseResult, Observable, Observation
from .results import (
    EqualityConstraintResult,
    LatticeResult,
    RegressionResult,
    SavedEntry,
    SavedObservations,
    UnittestResults,
)


def run(
    config: UnittestConfig,
    config_dir: Path,
    save_path: Path | None = None,
    compare: SavedObservations | None = None,
) -> UnittestResults:
    # Build lattice_id -> set of observables needed for that lattice
    needed: dict[str, set[Observable]] = {lat_id: set() for lat_id in config.lattices}
    for constraint in config.equality_constraints:
        for obs in [constraint.obs_a, constraint.obs_b]:
            if obs.lattice_id in needed:
                needed[obs.lattice_id].add(obs)

    # Run observables: observable -> observation
    obs_map: dict[Observable, Observation] = {}
    lattice_results: dict[str, LatticeResult] = {}

    for lat_id, lat_config in config.lattices.items():
        kwargs: dict = {"noplot": True}
        if lat_config.init_file:
            kwargs["init_file"] = str(config_dir / lat_config.init_file)
        if lat_config.lattice_file:
            kwargs["lattice_file"] = str(config_dir / lat_config.lattice_file)

        loaded = False
        error: str | None = None
        t0 = time.monotonic()

        try:
            with SubprocessTao(**kwargs) as tao:
                loaded = True
                for obs in needed[lat_id]:
                    obs_map[obs] = obs(tao)
        except Exception:
            error = traceback.format_exc().strip()
        finally:
            load_time = time.monotonic() - t0

        lattice_results[lat_id] = LatticeResult(
            lattice_file=lat_config.lattice_file,
            init_file=lat_config.init_file,
            loaded=loaded,
            error=error,
            load_time=load_time,
        )

    if save_path is not None:
        saved = SavedObservations(entries=[
            SavedEntry(observable=obs, observation=obs_val)
            for obs, obs_val in obs_map.items()
        ])
        save_path.write_text(saved.model_dump_json(indent=2))

    # Run each equality constraint comparison
    constraint_results: list[EqualityConstraintResult] = []
    for constraint in config.equality_constraints:
        try:
            result = constraint.compare(obs_map[constraint.obs_a], obs_map[constraint.obs_b])
        except Exception:
            result = IsCloseResult(
                is_close=False,
                error=traceback.format_exc().strip(),
            )
        constraint_results.append(EqualityConstraintResult(
            obs_a=constraint.obs_a,
            obs_b=constraint.obs_b,
            result=result,
        ))

    # Regression comparisons against saved observations
    regression_results: list[RegressionResult] = []
    if compare is not None:
        compare_map = {e.observable: e.observation for e in compare.entries}
        for constraint in config.equality_constraints:
            for obs in [constraint.obs_a, constraint.obs_b]:
                try:
                    result = constraint.compare(obs_map[obs], compare_map[obs])
                except Exception:
                    result = IsCloseResult(
                        is_close=False,
                        error=traceback.format_exc().strip(),
                    )
                regression_results.append(RegressionResult(observable=obs, result=result))

    return UnittestResults(
        lattices=lattice_results,
        equality_constraints=constraint_results,
        regression=regression_results,
    )


def _print_check_detail(res: IsCloseResult) -> None:
    if isinstance(res, EleIsCloseResult):
        checks = {
            "twiss_a": res.twiss_a,
            "twiss_b": res.twiss_b,
            "eta_x": res.eta_x,
            "etap_x": res.etap_x,
            "eta_y": res.eta_y,
            "etap_y": res.etap_y,
            "ref_energy": res.ref_energy,
            "p0c": res.p0c,
            "orbit": res.orbit,
            "floor_x": res.floor_x,
            "floor_y": res.floor_y,
            "floor_z": res.floor_z,
        }
        ran = {name: check for name, check in checks.items() if check is not None}
        if ran:
            width = max(len(name) for name in ran)
            for name, check in ran.items():
                check_status = "PASS" if check.passed else "FAIL"
                detail = f"  {check.detail}" if not check.passed and check.detail else ""
                print(f"    {name:<{width}}  {check_status}{detail}")
    if res.error:
        for line in res.error.splitlines():
            print(f"    {line}")


def _print_results(results: UnittestResults) -> None:
    print("Lattices:")
    for lat_id, lat in results.lattices.items():
        status = "OK  " if lat.loaded else "FAIL"
        print(f"  [{status}] {lat_id}  ({lat.load_time:.2f}s)")
        if lat.error:
            for line in lat.error.splitlines():
                print(f"         {line}")

    print()
    print("Equality constraints:")
    for cr in results.equality_constraints:
        status = "PASS" if cr.result.is_close else "FAIL"
        label = f"{cr.obs_a.label} == {cr.obs_b.label}"
        print(f"  [{status}] {label}")

    if results.regression:
        print()
        print("Regression:")
        for rr in results.regression:
            status = "PASS" if rr.result.is_close else "FAIL"
            print(f"  [{status}] {rr.observable.label}")

    failures_eq = [cr for cr in results.equality_constraints if not cr.result.is_close]
    failures_reg = [rr for rr in results.regression if not rr.result.is_close]

    if failures_eq or failures_reg:
        print()
        print("=" * 60)
        print("FAILURES")
        print("=" * 60)
        for cr in failures_eq:
            label = f"{cr.obs_a.label} == {cr.obs_b.label}"
            print(f"\n  {label}")
            print("  " + "-" * 56)
            _print_check_detail(cr.result)
        for rr in failures_reg:
            print(f"\n  regression: {rr.observable.label}")
            print("  " + "-" * 56)
            _print_check_detail(rr.result)

    n_passed = sum(1 for cr in results.equality_constraints if cr.result.is_close)
    n_total = len(results.equality_constraints)
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
        "-o",
        "--output",
        metavar="OUTPUT",
        help="Path to write JSON results file (optional)",
    )
    parser.add_argument(
        "--save",
        metavar="OBSERVATIONS",
        help="Path to write a JSON snapshot of current observations",
    )
    parser.add_argument(
        "--compare-path",
        metavar="OBSERVATIONS",
        help="Path to a previously saved observations JSON for regression comparison",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with config_path.open() as fh:
        raw = yaml.safe_load(fh)

    config = UnittestConfig.model_validate(raw)

    compare: SavedObservations | None = None
    if args.compare_path:
        compare = SavedObservations.model_validate_json(Path(args.compare_path).read_text())

    save_path = Path(args.save) if args.save else None

    results = run(config, config_dir=config_path.parent, save_path=save_path, compare=compare)

    _print_results(results)

    if save_path is not None:
        print(f"\nObservations saved to {save_path}")

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(results.model_dump_json(indent=2))
        print(f"\nResults written to {output_path}")
