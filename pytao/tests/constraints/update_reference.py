import json
import pathlib
import tempfile
from unittest.mock import patch

from pytao.constraints.main import main
from pytao.constraints.results import ConstraintResultsGroup


DATA_DIR = pathlib.Path(__file__).parent / "data"
OUT_PATH = DATA_DIR / "reference_results.json"


def _normalize_results(data: dict) -> dict:
    data.pop("started_at", None)
    data.pop("finished_at", None)
    for lat in data.get("lattices", {}).values():
        lat.pop("load_time", None)
        lat.pop("obs_time", None)
    for group in data.get("constraints", {}).values():
        for constraint in group:
            constraint["observables"].sort(key=lambda o: json.dumps(o, sort_keys=True))
    return data


def run() -> None:
    with tempfile.TemporaryDirectory() as dir:
        save_path = pathlib.Path(dir) / "out.json"
        with patch(
            "sys.argv",
            [
                "pytao-constraints",
                str(DATA_DIR / "constraints.yaml"),
                "--save-results",
                str(save_path),
            ],
        ):
            try:
                main()
            except SystemExit:
                pass
        loaded = ConstraintResultsGroup.model_validate_json(
            pathlib.Path(save_path).read_text()
        )
        data = _normalize_results(loaded.model_dump(mode="json"))
        OUT_PATH.write_text(json.dumps(data, indent=2))
        print(f"Written: {OUT_PATH}")


if __name__ == "__main__":
    run()
