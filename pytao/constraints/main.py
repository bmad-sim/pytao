import argparse
import logging
import sys
import traceback
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import yaml

from pytao.startup import TaoStartup
from pytao.subproc import (
    MAX_AUTO_JOBS,
    TaoInitResult,
    parallel_subprocess_taos,
    resolve_job_count,
)

from .config import ConstraintsConfig
from .observables import (
    ComparisonResult,
    LatticeObservable,
    Observable,
    Observation,
)
from .results import (
    ConstraintResult,
    ConstraintResultsGroup,
    LatticeResult,
    SavedObservations,
)

logger = logging.getLogger(__name__)

_MD_ESCAPE = str.maketrans({c: f"\\{c}" for c in r"\[]*_`|"})


def _escape_md(text: str) -> str:
    return text.translate(_MD_ESCAPE)


def _md_status(passed: bool) -> str:
    return ":white_check_mark:" if passed else ":x:"


def _observe_lattice(
    lat_id: str,
    lat_startup: TaoStartup,
    init_result: TaoInitResult,
    observables: Iterable[LatticeObservable],
) -> tuple[LatticeResult, dict[Observable, Observation], str]:
    """
    Evaluate observables against an already-initialized lattice.

    This never raises: initialization and observation failures are captured in
    the returned `LatticeResult`.

    Parameters
    ----------
    lat_id : str
        Identifier of the lattice in the configuration.
    lat_startup : TaoStartup
        Startup settings as configured, before path prefixing. Recorded on the
        result so that saved output stays relative to the config file.
    init_result : TaoInitResult
        Outcome of initializing Tao for this lattice.
    observables : iterable of LatticeObservable
        Observables to evaluate against this lattice.

    Returns
    -------
    tuple[LatticeResult, dict[Observable, Observation], str]
        The lattice result, its observations, and a one-line status message.
    """
    obs_map: dict[Observable, Observation] = {}
    particle_survived: bool | None = None
    error = ""
    load_time = init_result.elapsed_time
    obs_time = 0.0

    tao = init_result.tao
    if tao is None:
        error = "".join(traceback.format_exception(init_result.error)).strip()
        first_line = error.splitlines()[-1] if error else "unknown error"
        status_line = f"[FAIL] {lat_id}  {first_line}"
    else:
        try:
            states = tao.lat_list("end", "orbit.state", flags="-array_out")
            particle_survived = bool(states[0] == 1)
        except Exception:
            logger.debug(
                "Particle survival check failed for lattice %r:\n%s",
                lat_id,
                traceback.format_exc().strip(),
            )
        for obs in observables:
            try:
                obs_map[obs] = obs.observe(tao)
            except Exception:
                logger.debug(
                    "Observable %r failed for lattice %r:\n%s",
                    obs,
                    lat_id,
                    traceback.format_exc().strip(),
                )
        obs_time = sum(observation.elapsed_time for observation in obs_map.values())
        tag = "[LOST]" if particle_survived is False else "[OK  ]"
        status_line = (
            f"{tag} {lat_id}  loaded in {load_time:.2f}s, "
            f"{len(obs_map)} observables in {obs_time:.2f}s"
        )

    result = LatticeResult(
        tao_startup=lat_startup,
        loaded=tao is not None,
        particle_survived=particle_survived,
        error=error,
        load_time=load_time,
        obs_time=obs_time,
    )
    return result, obs_map, status_line


def run(
    config: ConstraintsConfig,
    config_dir: Path,
    compare: SavedObservations | None = None,
    verbose: bool = False,
    jobs: int | None = None,
) -> tuple[SavedObservations, ConstraintResultsGroup]:
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
    verbose : bool, default=False
        Print lattice loading progress to stdout.
    jobs : int, optional
        Number of lattices to load in parallel. Each lattice runs in its own
        Tao subprocess, so this scales with available CPUs and memory. Defaults
        to an automatic count capped at `MAX_AUTO_JOBS`; use ``1`` to load
        lattices sequentially.

    Returns
    -------
    tuple[SavedObservations, ConstraintResultsGroup]
        Saved lattice observations and the full constraint results.
    """
    started_at = datetime.now(timezone.utc)

    needed = config.required_lattice_observables
    literal_obs = config.required_literal_observables

    n_lat = len(config.lattices)
    n_obs_total = sum(len(v) for v in needed.values())
    n_constraints = len(config.all_constraints)
    n_jobs = resolve_job_count(jobs, n_lat)
    summary = (
        f"Beginning constraints check with {n_lat} lattice(s), {n_constraints} constraint(s), "
        f"and {n_obs_total} observable(s)"
    )
    loading_header = (
        "Loading Lattices:" if n_jobs == 1 else f"Loading Lattices: ({n_jobs} in parallel)"
    )
    logger.info(summary)
    logger.info(loading_header)
    if verbose:
        print(summary)
        print(loading_header)

    # Run observables: observable -> observation
    obs_map: dict[Observable, Observation] = {}
    lattice_results: dict[str, LatticeResult] = {}

    lat_ids = list(config.lattices)
    startups = [config.lattices[lat_id].with_path_prefix(config_dir) for lat_id in lat_ids]

    with parallel_subprocess_taos(startups, jobs=n_jobs) as init_results:
        for lat_id, init_result in zip(lat_ids, init_results):
            result, lat_obs_map, status_line = _observe_lattice(
                lat_id, config.lattices[lat_id], init_result, needed[lat_id]
            )
            logger.info(status_line)
            if verbose:
                print(f"  {status_line}")
            obs_map.update(lat_obs_map)
            lattice_results[lat_id] = result

    for obs in literal_obs:
        obs_map[obs] = obs.observe()

    saved = SavedObservations.from_obs_map(obs_map)

    constraint_results: dict[str | None, list[ConstraintResult]] = {}
    regression_results: dict[str | None, list] = {}
    compare_map = compare.obs_map if compare is not None else None

    for group, constraints in config.constraints_by_group.items():
        constraint_results[group] = []
        regression_results[group] = []
        for constraint in constraints:
            crs, reg = constraint.run(obs_map, compare_map, group)
            constraint_results[group].extend(crs)
            regression_results[group].extend(reg)

    return saved, ConstraintResultsGroup(
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


def _print_results_markdown(results: ConstraintResultsGroup) -> None:
    grouped = any(group is not None for group in results.constraints)

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
    for group, crs in results.constraints.items():
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
        for _, rr in results.iter_regression():
            status = _md_status(rr.result.is_satisfied)
            desc = _escape_md(rr.description)
            print(f"| {status} | {_escape_md(rr.label)} | {desc} |")

    lat_failures = [(lat_id, lat) for lat_id, lat in results.lattices.items() if lat.failed]
    failures_eq = [
        (group, cr) for group, cr in results.iter_constraints() if not cr.result.is_satisfied
    ]
    failures_reg = [rr for _, rr in results.iter_regression() if not rr.result.is_satisfied]

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
        for group, cr in failures_eq:
            # Raw label in <summary>: content is HTML, not markdown, so _escape_md
            # would produce literal backslashes instead of consumed escape sequences.
            label = cr.label
            prefix = f"[{group}] " if grouped and group else ""
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

    n_passed = sum(1 for _, cr in results.iter_constraints() if cr.result.is_satisfied)
    n_total = sum(len(v) for v in results.constraints.values())
    print()
    print(f"**{n_passed}/{n_total} constraints passed**")

    if results.regression:
        n_reg_passed = sum(1 for _, rr in results.iter_regression() if rr.result.is_satisfied)
        n_reg_total = sum(len(v) for v in results.regression.values())
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


def _print_results(results: ConstraintResultsGroup) -> None:
    grouped = any(group is not None for group in results.constraints)

    print("Constraints:")
    for group, crs in results.constraints.items():
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
        for _, rr in results.iter_regression():
            status = "PASS" if rr.result.is_satisfied else "FAIL"
            suffix = f"  {rr.description}" if rr.description else ""
            print(f"  [{status}] {rr.label}{suffix}")

    lat_failures = [(lat_id, lat) for lat_id, lat in results.lattices.items() if lat.failed]
    failures_eq = [
        (group, cr) for group, cr in results.iter_constraints() if not cr.result.is_satisfied
    ]
    failures_reg = [rr for _, rr in results.iter_regression() if not rr.result.is_satisfied]

    if lat_failures or failures_eq or failures_reg:
        print()
        print("=" * 60)
        print("FAILURES")
        print("=" * 60)
        for lat_id, lat in lat_failures:
            if not lat.loaded:
                error_lines = lat.error.splitlines()[-10:] if lat.error else ["unknown error"]
                print(f"\n  lattice {lat_id}: failed to load")
                for line in error_lines:
                    print(f"    {line}")
            else:
                print(f"\n  lattice {lat_id}: particle lost before end")
        for group, cr in failures_eq:
            label = cr.label
            prefix = f"[{group}] " if grouped and group else ""
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

    n_passed = sum(1 for _, cr in results.iter_constraints() if cr.result.is_satisfied)
    n_total = sum(len(v) for v in results.constraints.values())
    print()
    print(f"{n_passed}/{n_total} constraints passed")

    if results.regression:
        n_reg_passed = sum(1 for _, rr in results.iter_regression() if rr.result.is_satisfied)
        n_reg_total = sum(len(v) for v in results.regression.values())
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
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of lattices to load in parallel (default: automatic, "
            f"up to {MAX_AUTO_JOBS}). Use 1 to load lattices sequentially."
        ),
    )
    parser.add_argument(
        "--log-file",
        metavar="FILE",
        help="Write pytao/Tao log output to FILE",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level for --log-file (default: INFO)",
    )
    args = parser.parse_args()

    if args.log_file:
        logging.basicConfig(
            filename=args.log_file,
            level=getattr(logging, args.log_level),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    config_path = Path(args.config).resolve()
    with config_path.open() as fh:
        raw = yaml.safe_load(fh)

    config = ConstraintsConfig.model_validate(raw)

    compare: SavedObservations | None = None
    if args.compare_path:
        compare = SavedObservations.model_validate_json(Path(args.compare_path).read_text())

    save_obs_path = Path(args.save_observations) if args.save_observations else None

    saved, results = run(
        config,
        config_dir=config_path.parent,
        compare=compare,
        verbose=not args.markdown,
        jobs=args.jobs,
    )

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

    failed = any(lat.failed for lat in results.lattices.values()) or any(
        not cr.result.is_satisfied for _, cr in results.iter_constraints()
    )
    if failed:
        sys.exit(1)
