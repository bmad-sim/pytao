import json
import pathlib
import tempfile
from unittest.mock import patch
from pytao.constraints.main import main
from pytao.constraints.results import SavedObservations


DATA_DIR = pathlib.Path(__file__).parent / "data"
OUT_PATH = DATA_DIR / "reference_observations.json"


def _normalize_observations(data: dict) -> dict:
    entries = data.get("entries", [])
    for entry in entries:
        obs = entry.get("observation", {})
        obs.pop("elapsed_time", None)
        obs.pop("created_at", None)
    entries.sort(key=lambda e: json.dumps(e.get("observable", {}), sort_keys=True))
    return data


def run() -> None:
    with tempfile.TemporaryDirectory() as dir:
        save_path = pathlib.Path(dir) / "out.json"
        with patch(
            "sys.argv",
            [
                "pytao-constraints",
                str(DATA_DIR / "constraints.yaml"),
                "--save-observations",
                str(save_path),
            ],
        ):
            main()
        loaded = SavedObservations.model_validate_json(pathlib.Path(save_path).read_text())
        data = _normalize_observations(loaded.model_dump(mode="json"))
        OUT_PATH.write_text(json.dumps(data, indent=2))
        print(f"Written: {OUT_PATH}")


if __name__ == "__main__":
    run()
