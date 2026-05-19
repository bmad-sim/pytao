import argparse
import time
import traceback
from pathlib import Path

import yaml

from pytao import SubprocessTao

from .config import UnittestConfig
from .observables.ele import EleIsCloseResult, EleObservable, EleObservation
from .results import LatticeResult, PairEqualityResult, UnittestResults


def run(config: UnittestConfig, config_dir: Path) -> UnittestResults:
    # Build lattice_id -> set of unique observables needed
    needed: dict[str, set[EleObservable]] = {lat_id: set() for lat_id in config.lattices}
    for pair in config.pair_equality:
        if pair.lattice_a_id in needed:
            needed[pair.lattice_a_id].add(EleObservable(ele=pair.element_a))
        if pair.lattice_b_id in needed:
            needed[pair.lattice_b_id].add(EleObservable(ele=pair.element_b))

    # Run observables: lattice_id -> {observable -> observation}
    obs_map: dict[str, dict[EleObservable, EleObservation]] = {}
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

    # Run each pair comparison
    pair_results: list[PairEqualityResult] = []
    for pair in config.pair_equality:
        obs_a_key = EleObservable(ele=pair.element_a)
        obs_b_key = EleObservable(ele=pair.element_b)
        try:
            obs_a = obs_map[pair.lattice_a_id][obs_a_key]
            obs_b = obs_map[pair.lattice_b_id][obs_b_key]
            result = pair.comparison(obs_a, obs_b)
        except Exception:
            result = EleIsCloseResult(
                is_close=False,
                error=traceback.format_exc().strip(),
            )
        pair_results.append(PairEqualityResult(
            lattice_a_id=pair.lattice_a_id,
            element_a=pair.element_a,
            lattice_b_id=pair.lattice_b_id,
            element_b=pair.element_b,
            result=result,
        ))

    return UnittestResults(lattices=lattice_results, pair_equality=pair_results)


def _print_results(results: UnittestResults) -> None:
    print("Lattices:")
    for lat_id, lat in results.lattices.items():
        status = "OK  " if lat.loaded else "FAIL"
        print(f"  [{status}] {lat_id}  ({lat.load_time:.2f}s)")
        if lat.error:
            for line in lat.error.splitlines():
                print(f"         {line}")

    print()
    print("Pair equality:")
    for pr in results.pair_equality:
        status = "PASS" if pr.result.is_close else "FAIL"
        label = f"{pr.lattice_a_id}[{pr.element_a}] == {pr.lattice_b_id}[{pr.element_b}]"
        print(f"  [{status}] {label}")

    failures = [pr for pr in results.pair_equality if not pr.result.is_close]
    if failures:
        print()
        print("=" * 60)
        print("FAILURES")
        print("=" * 60)
        for pr in failures:
            label = f"{pr.lattice_a_id}[{pr.element_a}] == {pr.lattice_b_id}[{pr.element_b}]"
            print(f"\n  {label}")
            print("  " + "-" * 56)
            res = pr.result
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

    n_passed = sum(1 for pr in results.pair_equality if pr.result.is_close)
    n_total = len(results.pair_equality)
    print()
    print(f"{n_passed}/{n_total} pairs passed")


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
