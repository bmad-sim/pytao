import argparse
import time
import traceback
from pathlib import Path

import yaml

from pytao import SubprocessTao

from .config import UnittestConfig
from .observables import Observable, Observation
from .results import LatticeResult, TestResult, UnittestResults


def run(config: UnittestConfig, config_dir: Path) -> UnittestResults:
    # Build lattice_id -> set of unique observables needed
    needed: dict[str, set[Observable]] = {lat_id: set() for lat_id in config.lattices}
    for test in config.tests:
        for lat_id, obs in test.observables.items():
            if lat_id in needed:
                needed[lat_id].add(obs)

    # Run observables: lattice_id -> {observable -> observation}
    obs_map: dict[str, dict[Observable, Observation]] = {}
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
                obs_map[lat_id] = {obs: obs(tao) for obs in needed[lat_id]}
        except Exception:
            error = traceback.format_exc().strip()
            obs_map[lat_id] = {}
        finally:
            load_time = time.monotonic() - t0

        lattice_results[lat_id] = LatticeResult(
            lattice_file=lat_config.lattice_file,
            init_file=lat_config.init_file,
            loaded=loaded,
            error=error,
            load_time=load_time,
        )

    # Run each test, resolving its observations from the shared map
    test_results: list[test_result_types] = []
    for test in config.tests:
        try:
            observations: dict[str, Observation] = {
                lat_id: obs_map[lat_id][obs]
                for lat_id, obs in test.observables.items()
            }
            result = test.run(observations)
        except Exception:
            result = TestResult(
                test_type=type(test).__name__,
                description=test.description,
                passed=False,
                error=traceback.format_exc().strip(),
            )
        test_results.append(result)

    return UnittestResults(lattices=lattice_results, tests=test_results)


def _print_results(results: UnittestResults) -> None:
    print("Lattices:")
    for lat_id, lat in results.lattices.items():
        status = "OK  " if lat.loaded else "FAIL"
        print(f"  [{status}] {lat_id}  ({lat.load_time:.2f}s)")
        if lat.error:
            for line in lat.error.splitlines():
                print(f"         {line}")

    print()
    print("Tests:")
    for result in results.tests:
        status = "PASS" if result.passed else "FAIL"
        suffix = f"  {result.description}" if result.description else ""
        print(f"  [{status}] {result.test_type}{suffix}")

    failures = [r for r in results.tests if not r.passed]
    if failures:
        print()
        print("=" * 60)
        print("FAILURES")
        print("=" * 60)
        for result in failures:
            desc = f"  {result.description}" if result.description else ""
            print(f"\n  {result.test_type}{desc}")
            print("  " + "-" * 56)
            result.print_failure_detail()

    n_passed = sum(1 for t in results.tests if t.passed)
    n_total = len(results.tests)
    print()
    print(f"{n_passed}/{n_total} tests passed")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pytao-unittest",
        description="Run pytao unit tests against Bmad lattice files.",
    )
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument(
        "-o",
        "--output",
        metavar="OUTPUT",
        help="Path to write JSON results file (optional)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with config_path.open() as fh:
        raw = yaml.safe_load(fh)

    config = UnittestConfig.model_validate(raw)
    results = run(config, config_dir=config_path.parent)

    _print_results(results)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(results.model_dump_json(indent=2))
        print(f"\nResults written to {output_path}")
