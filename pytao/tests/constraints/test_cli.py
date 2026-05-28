import pathlib
from unittest.mock import patch

import pytest

from pytao.constraints.main import main
from pytao.constraints.results import SavedObservations

DATA_DIR = pathlib.Path(__file__).parent / "data"
CONFIGS = ["constraints.yaml", "constraints_grouped.yaml"]


def _run_cli(*args: str, capsys) -> tuple[str, int]:
    with patch("sys.argv", ["pytao-constraints", *args]):
        with pytest.raises(SystemExit) as exc_info:
            main()
    return capsys.readouterr().out, exc_info.value.code


@pytest.mark.parametrize("config_file", CONFIGS)
@pytest.mark.parametrize("markdown", [False, True], ids=["plain", "markdown"])
def test_cli_output_format(config_file, markdown, capsys):
    args = [str(DATA_DIR / config_file)]
    if markdown:
        args.append("--markdown")
    out, code = _run_cli(*args, capsys=capsys)
    assert code == 1
    if markdown:
        assert "## Lattices" in out
        assert "## Constraints" in out
    else:
        assert "Lattices:" in out
        assert "Constraints:" in out
    if "grouped" in config_file:
        assert "Lattice-Consistency" in out


@pytest.mark.parametrize("config_file", CONFIGS)
def test_cli_save_observations(config_file, capsys, tmp_path):
    save_path = tmp_path / "obs.json"
    _, code = _run_cli(
        str(DATA_DIR / config_file),
        "--save-observations",
        str(save_path),
        capsys=capsys,
    )
    assert code == 1
    assert save_path.exists()
    loaded = SavedObservations.model_validate_json(save_path.read_text())
    assert len(loaded.entries) > 0


@pytest.mark.parametrize("config_file", CONFIGS)
@pytest.mark.parametrize("markdown", [False, True], ids=["plain", "markdown"])
def test_cli_compare(config_file, markdown, capsys, tmp_path):
    config_path = str(DATA_DIR / config_file)
    save_path = tmp_path / "obs.json"

    _run_cli(config_path, "--save-observations", str(save_path), capsys=capsys)

    args = [config_path, "--compare-path", str(save_path)]
    if markdown:
        args.append("--markdown")
    out, code = _run_cli(*args, capsys=capsys)
    assert code == 1

    if "grouped" in config_file:
        if markdown:
            assert "## Regression" in out
        else:
            assert "Regression:" in out
