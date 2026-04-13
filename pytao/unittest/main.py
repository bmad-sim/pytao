import argparse
import time
import traceback
from pathlib import Path

from pytao import SubprocessTao

from .config import LatticeConfig, UnittestConfig
from .results import LatticeResult, UnittestResults


def _run_lattice(config: LatticeConfig) -> LatticeResult:
    kwargs: dict = {"noplot": True}
    if config.init_file:
        kwargs["init_file"] = config.init_file
    if config.lattice_file:
        kwargs["lattice_file"] = config.lattice_file

    loaded = False
    error: str | None = None
    t0 = time.monotonic()
    try:
        with SubprocessTao(**kwargs):
            loaded = True
    except Exception:
        error = traceback.format_exc().strip()
    finally:
        load_time = time.monotonic() - t0

    return LatticeResult(
        lattice_file=config.lattice_file,
        init_file=config.init_file,
        loaded=loaded,
        error=error,
        load_time=load_time,
    )


def run(config: UnittestConfig) -> UnittestResults:
    return UnittestResults(lattices=[_run_lattice(lat) for lat in config.lattices])


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pytao-unittest",
        description="Test Bmad lattice files by loading them with Tao.",
    )
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="OUTPUT",
        help="Path to write JSON results file",
    )
    args = parser.parse_args()

    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required: pip install pyyaml") from exc

    config_path = Path(args.config)
    with config_path.open() as fh:
        raw = yaml.safe_load(fh)

    config = UnittestConfig.model_validate(raw)
    results = run(config)

    output_path = Path(args.output)
    output_path.write_text(results.model_dump_json(indent=2))
    print(f"Results written to {output_path}")
