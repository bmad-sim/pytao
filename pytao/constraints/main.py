import argparse
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml

from pytao import SubprocessTao

from .config import (
    ComparisonConstraint,
    ConstraintsConfig,
    IsCloseConstraint,
    RegressionConstraint,
)
from .observables import (
    ComparisonResult,
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

_MD_ESCAPE = str.maketrans({c: f"\\{c}" for c in r"\[]*_`|"})


def _escape_md(text: str) -> str:
    return text.translate(_MD_ESCAPE)


def _md_status(passed: bool) -> str:
    return ":white_check_mark:" if passed else ":x:"


def run(
    config: ConstraintsConfig,
    config_dir: Path,
    compare: SavedObservations | None = None,
    verbose: bool = False,
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

    needed = config.required_lattice_observables
    literal_obs = config.required_literal_observables

    if verbose:
        n_lat = len(config.lattices)
        n_obs = sum(len(v) for v in needed.values())
        n_constraints = len(config.all_constraints)
        print(
            f"Beginning constraints check with {n_lat} lattice(s), {n_constraints} constraint(s), "
            f"and {n_obs} observable(s)"
        )
        print("Loading Lattices:")

    # Run observables: observable -> observation
    obs_map: dict[Observable, Observation] = {}
    lattice_results: dict[str, LatticeResult] = {}

    for lat_id, lat_startup in config.lattices.items():
        params = dict(lat_startup.with_path_prefix(config_dir).tao_class_params)
        params["noplot"] = True

        loaded = False
        particle_survived: bool | None = None
        error: str | None = None
        load_time = 0.0
        obs_time = 0.0
        t0 = time.perf_counter()

        try:
            with SubprocessTao(**params) as tao:
                load_time = time.perf_counter() - t0
                loaded = True
                try:
                    states = tao.lat_list("end", "orbit.state", flags="-array_out")
                    particle_survived = bool(states[0] == 1)
                except Exception:
                    logger.debug(
                        "Particle survival check failed for lattice %r:\n%s",
                        lat_id,
                        traceback.format_exc().strip(),
                    )
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

        if verbose:
            if not loaded:
                first_line = error.splitlines()[-1] if error else "unknown error"
                print(f"  [FAIL] {lat_id}  {first_line}")
            else:
                n_obs = len([obs for obs in needed[lat_id] if obs in obs_map])
                tag = "[LOST]" if particle_survived is False else "[OK  ]"
                print(
                    f"  {tag} {lat_id}  loaded in {load_time:.2f}s, {n_obs} observables in {obs_time:.2f}s"
                )

        lattice_results[lat_id] = LatticeResult(
            tao_startup=lat_startup,
            loaded=loaded,
            particle_survived=particle_survived,
            error=error,
            load_time=load_time,
            obs_time=obs_time,
        )

    for obs in literal_obs:
        obs_map[obs] = obs()

    saved = SavedObservations.from_obs_map(obs_map)

    constraint_results: list[ConstraintResult] = []
    regression_results: list[RegressionResult] = []
    compare_map = compare.obs_map if compare is not None else None

    for group, constraints in config.constraints_by_group.items():
        for constraint in constraints:
            if isinstance(constraint, ComparisonConstraint):
                missing = [
                    obs for obs in constraint.required_observables if obs not in obs_map
                ]
                if missing:
                    missing_labels = ", ".join(obs.label for obs in missing)
                    result = constraint.error_result(f"Missing observations: {missing_labels}")
                else:
                    result = constraint.is_satisfied(
                        {obs: obs_map[obs] for obs in constraint.required_observables}
                    )
                constraint_results.append(
                    ConstraintResult(
                        group=group,
                        label=constraint.label,
                        observables=list(constraint.required_observables),
                        description=constraint.description,
                        comment=constraint.comment,
                        result=result,
                    )
                )
                if (
                    isinstance(constraint, IsCloseConstraint)
                    and constraint.regression_check
                    and compare_map is not None
                ):
                    for obs in constraint.required_observables:
                        if obs not in obs_map or obs not in compare_map:
                            reg_result = constraint.error_result("Missing observation")
                        else:
                            reg_result = constraint.comparison(obs_map[obs], compare_map[obs])
                        regression_results.append(
                            RegressionResult(
                                group=group,
                                label=constraint.label,
                                description=constraint.description,
                                comment=constraint.comment,
                                observable=obs,
                                result=reg_result,
                            )
                        )
            elif isinstance(constraint, RegressionConstraint):
                if compare_map is None:
                    continue
                obs = next(iter(constraint.required_observables))
                if obs not in obs_map or obs not in compare_map:
                    result = constraint.error_result("Missing observation")
                else:
                    result = constraint.evaluate(obs_map[obs], compare_map[obs])
                regression_results.append(
                    RegressionResult(
                        group=group,
                        label=constraint.label,
                        description=constraint.description,
                        comment=constraint.comment,
                        observable=obs,
                        result=result,
                    )
                )
            else:
                raise ValueError(f"Unrecognized constraint type: {type(constraint)}")

    return saved, ConstraintResults(
        started_at=started_at,
        finished_at=datetime.now(timezone.utc),
        lattices=lattice_results,
        constraints=constraint_results,
        regression=regression_results,
    )


def _md_check_detail_rows(res: ComparisonResult) -> str:
    lines = []
    checks = res.check_results()
    if checks:
        lines.append("| Check | Result |")
        lines.append("|-------|--------|")
        for name, check in checks.items():
            status = _md_status(check.passed)
            detail = _escape_md(check.detail) if check.detail else ""
            result_cell = f"{status} {detail}".strip()
            lines.append(f"| {_escape_md(name)} | {result_cell} |")
    if res.error:
        lines.append("")
        lines.append("```")
        lines.append(res.error)
        lines.append("```")
    return "\n".join(lines)


def _print_results_markdown(results: ConstraintResults) -> None:
    grouped = any(cr.group is not None for cr in results.constraints)

    print("## Lattices")
    print()
    print("| Lattice | Status | Load | Obs |")
    print("|---------|--------|------|-----|")
    for lat_id, lat in results.lattices.items():
        if not lat.loaded:
            status = f"{_md_status(False)} failed"
        elif lat.particle_survived is False:
            status = f"{_md_status(False)} particle lost"
        else:
            status = f"{_md_status(True)} loaded"
        print(
            f"| {_escape_md(lat_id)} | {status} | {lat.load_time:.2f}s | {lat.obs_time:.2f}s |"
        )

    lat_errors = [(lat_id, lat) for lat_id, lat in results.lattices.items() if lat.error]
    if lat_errors:
        print()
        print("<details><summary>Lattice errors</summary>")
        print()
        for lat_id, lat in lat_errors:
            print(f"**{_escape_md(lat_id)}**")
            print()
            print("```")
            print(lat.error)
            print("```")
            print()
        print("</details>")

    print()
    print("## Constraints")
    for group, crs in results.constraints_by_group.items():
        if grouped:
            print()
            print(f"### {group}")
        print()
        print("| Status | Constraint | Description |")
        print("|--------|------------|-------------|")
        for cr in crs:
            status = _md_status(cr.result.is_satisfied)
            label = _escape_md(cr.label)
            desc = _escape_md(cr.description)
            print(f"| {status} | {label} | {desc} |")

    if results.regression:
        print()
        print("## Regression")
        print()
        print("| Status | Observable | Description |")
        print("|--------|------------|-------------|")
        for rr in results.regression:
            status = _md_status(rr.result.is_satisfied)
            desc = _escape_md(rr.description)
            print(f"| {status} | {_escape_md(rr.label)} | {desc} |")

    lat_failures = [
        (lat_id, lat)
        for lat_id, lat in results.lattices.items()
        if not lat.loaded or lat.particle_survived is False
    ]
    failures_eq = [cr for cr in results.constraints if not cr.result.is_satisfied]
    failures_reg = [rr for rr in results.regression if not rr.result.is_satisfied]

    if lat_failures or failures_eq or failures_reg:
        print()
        print("## Failures")
        print()
        for lat_id, lat in lat_failures:
            if not lat.loaded:
                summary = f"{_md_status(False)} lattice {_escape_md(lat_id)}: failed to load"
                print("<details>")
                print(f"<summary>{summary}</summary>")
                print()
                if lat.error:
                    print("```")
                    print(lat.error)
                    print("```")
                    print()
                print("</details>")
                print()
            else:
                print(
                    f"{_md_status(False)} lattice {_escape_md(lat_id)}: particle lost before end"
                )
                print()
        for cr in failures_eq:
            # Raw label in <summary>: content is HTML, not markdown, so _escape_md
            # would produce literal backslashes instead of consumed escape sequences.
            label = cr.label
            prefix = f"[{cr.group}] " if grouped and cr.group else ""
            summary = f"{_md_status(False)} {prefix}{label}"
            if cr.description:
                summary += f"  {cr.description}"
            print("<details>")
            print(f"<summary>{summary}</summary>")
            print()
            if cr.comment:
                print(_escape_md(cr.comment))
                print()
            print(_md_check_detail_rows(cr.result))
            print()
            print("</details>")
            print()
        for rr in failures_reg:
            summary = f"{_md_status(False)} regression: {rr.label}"
            if rr.description:
                summary += f"  {rr.description}"
            print("<details>")
            print(f"<summary>{summary}</summary>")
            print()
            if rr.comment:
                print(_escape_md(rr.comment))
                print()
            print(_md_check_detail_rows(rr.result))
            print()
            print("</details>")
            print()

    n_passed = sum(1 for cr in results.constraints if cr.result.is_satisfied)
    n_total = len(results.constraints)
    print()
    print(f"**{n_passed}/{n_total} constraints passed**")

    if results.regression:
        n_reg_passed = sum(1 for rr in results.regression if rr.result.is_satisfied)
        n_reg_total = len(results.regression)
        print(f"**{n_reg_passed}/{n_reg_total} regression checks passed**")


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
    grouped = any(cr.group is not None for cr in results.constraints)

    print("Constraints:")
    for group, crs in results.constraints_by_group.items():
        if grouped:
            print(f"  [{group}]")
        indent = "    " if grouped else "  "
        for cr in crs:
            status = "PASS" if cr.result.is_satisfied else "FAIL"
            label = cr.label
            suffix = f"  {cr.description}" if cr.description else ""
            print(f"{indent}[{status}] {label}{suffix}")

    if results.regression:
        print()
        print("Regression:")
        for rr in results.regression:
            status = "PASS" if rr.result.is_satisfied else "FAIL"
            suffix = f"  {rr.description}" if rr.description else ""
            print(f"  [{status}] {rr.label}{suffix}")

    lat_failures = [
        (lat_id, lat)
        for lat_id, lat in results.lattices.items()
        if not lat.loaded or lat.particle_survived is False
    ]
    failures_eq = [cr for cr in results.constraints if not cr.result.is_satisfied]
    failures_reg = [rr for rr in results.regression if not rr.result.is_satisfied]

    if lat_failures or failures_eq or failures_reg:
        print()
        print("=" * 60)
        print("FAILURES")
        print("=" * 60)
        for lat_id, lat in lat_failures:
            if not lat.loaded:
                reason = lat.error.splitlines()[-1] if lat.error else "unknown error"
                print(f"\n  lattice {lat_id}: failed to load")
                print(f"    {reason}")
            else:
                print(f"\n  lattice {lat_id}: particle lost before end")
        for cr in failures_eq:
            label = cr.label
            prefix = f"[{cr.group}] " if grouped and cr.group else ""
            header = (
                f"{prefix}{label}  {cr.description}" if cr.description else f"{prefix}{label}"
            )
            print(f"\n  {header}")
            if cr.comment:
                print(f"  {cr.comment}")
            print("  " + "-" * 56)
            _print_check_detail(cr.result)
        for rr in failures_reg:
            header = f"regression: {rr.label}"
            if rr.description:
                header += f"  {rr.description}"
            print(f"\n  {header}")
            if rr.comment:
                print(f"  {rr.comment}")
            print("  " + "-" * 56)
            _print_check_detail(rr.result)

    n_passed = sum(1 for cr in results.constraints if cr.result.is_satisfied)
    n_total = len(results.constraints)
    print()
    print(f"{n_passed}/{n_total} constraints passed")

    if results.regression:
        n_reg_passed = sum(1 for rr in results.regression if rr.result.is_satisfied)
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
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Emit GitHub-flavored markdown suitable for GITHUB_STEP_SUMMARY",
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

    saved, results = run(config, config_dir=config_path.parent, compare=compare, verbose=True)

    if args.markdown:
        _print_results_markdown(results)
    else:
        _print_results(results)

    if save_obs_path is not None:
        save_obs_path.write_text(saved.model_dump_json(indent=2))
        print(f"\n{len(saved)} observations saved to {save_obs_path}")

    if args.save_results:
        results_path = Path(args.save_results)
        results_path.write_text(results.model_dump_json(indent=2))
        print(f"\nResults saved to {results_path}")

    failed = (
        any(not lat.loaded for lat in results.lattices.values())
        or any(lat.particle_survived is False for lat in results.lattices.values())
        or any(not cr.result.is_satisfied for cr in results.constraints)
    )
    if failed:
        sys.exit(1)
