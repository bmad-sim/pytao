from __future__ import annotations

import operator
import pathlib
from typing import Generator

import numpy as np
import pytest
from beamphysics import single_particle
from beamphysics.units import pmd_unit
from pydantic import TypeAdapter

from pytao import SubprocessTao, TaoCommandError
from pytao.model import Beam, BeamInit, SpaceChargeCom, TaoConfig, TaoGlobal
from pytao.model.types import NDArray, PydanticParticleGroup, PydanticPmdUnit


@pytest.fixture(scope="module")
def tao() -> Generator[SubprocessTao, None, None]:
    with SubprocessTao(
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall3d", noplot=True
    ) as tao:
        yield tao


def test_tao_config(tao: SubprocessTao) -> None:
    config = TaoConfig.from_tao(tao)
    print(repr(config))

    config.beam_init.a_emit = 1.0
    assert "set beam_init a_emit = 1.0" in config.set_commands
    assert f"set beam_init center(1) = {config.beam_init.center[0]}" in config.set_commands

    assert config.set(tao, allow_errors=False)
    assert TaoConfig.from_tao(tao).beam_init.a_emit == 1.0


def test_tao_config_write_read(tao: SubprocessTao, tmp_path: pathlib.Path) -> None:
    config = TaoConfig.from_tao(tao)

    dest = tmp_path / "info.json"
    config.write(dest)
    config.write(dest)  # write twice to check the backup mechanism
    assert len(list(tmp_path.glob("*.json"))) == 2

    restored = config.from_file(dest)
    assert config == restored


@pytest.mark.parametrize(
    ("with_tao",),
    [
        pytest.param(True, id="with-tao"),
        pytest.param(False, id="without-tao"),
    ],
)
def test_tao_config_shell_script(
    tao: SubprocessTao, tmp_path: pathlib.Path, with_tao: bool
) -> None:
    config = TaoConfig.from_tao(tao)

    config.write_bash_loader_script(
        tmp_path,
        prefix="foo",
        tao=tao if with_tao else None,
    )
    assert (tmp_path / "foo.sh").exists()
    assert (tmp_path / "foo.cmd").exists()

    if with_tao:
        assert (tmp_path / "foo.lat.bmad").exists()
    else:
        assert not (tmp_path / "foo.lat.bmad").exists()


def test_beam_init(tao: SubprocessTao) -> None:
    beam_init = BeamInit.from_tao(tao)
    print(repr(beam_init))

    assert beam_init == beam_init

    beam_init.a_emit = 1.0
    assert "set beam_init a_emit = 1.0" in beam_init.set_commands


def test_beam(tao: SubprocessTao, monkeypatch: pytest.MonkeyPatch) -> None:
    beam = Beam.from_tao(tao)
    print(repr(beam))

    assert beam == beam
    beam.always_reinit = True
    assert "set beam always_reinit = True" in beam.set_commands

    orig_cmd = tao.cmd

    def should_pass(cmd: str) -> bool:
        return cmd == "pipe global" or "global lattice_calc" in cmd or "global plot_on" in cmd

    def mock_command_raise(cmd: str, raises: bool = True):
        if should_pass(cmd):
            return orig_cmd(cmd, raises=raises)

        raise TaoCommandError("No")

    monkeypatch.setattr(tao, "cmd", mock_command_raise)

    assert (
        beam.set(tao, allow_errors=True, suppress_lattice_calc=False, suppress_plotting=False)
        is False
    )

    def mock_command_ok(cmd: str, raises: bool = True):
        if should_pass(cmd):
            return orig_cmd(cmd, raises=raises)

        return []

    monkeypatch.setattr(tao, "cmd", mock_command_ok)
    with beam.set_context(tao):
        pass


def test_space_charge_com(tao: SubprocessTao) -> None:
    space_charge = SpaceChargeCom.from_tao(tao)
    print(repr(space_charge))

    assert space_charge == space_charge  # checking the equality helper
    space_charge.abs_tol_tracking = 1.0
    assert "set space_charge_com abs_tol_tracking = 1.0" in space_charge.set_commands


def test_global(tao: SubprocessTao) -> None:
    glob = TaoGlobal.from_tao(tao)
    print(repr(glob))

    assert glob == glob  # checking the equality helper
    glob.debug_on = True
    assert "set global debug_on = True" in glob.set_commands
    glob.set(tao, allow_errors=False)
    assert TaoGlobal.from_tao(tao) == glob


def test_global_nthreads(tao: SubprocessTao) -> None:
    glob = TaoGlobal.from_tao(tao)
    print(repr(glob))

    orig_nthreads = glob.n_threads
    if orig_nthreads > 1:
        glob.n_threads = 1
    else:
        glob.n_threads = 2

    assert f"set global n_threads = {glob.n_threads}" in glob.set_commands
    try:
        glob.set(tao, allow_errors=False)
    except TaoCommandError as ex:
        if "Multithreading support with OpenMP is not available" in str(ex):
            pytest.skip("Multithreading support unavailable to test")
        raise

    assert TaoGlobal.from_tao(tao) == glob


@pytest.mark.parametrize(
    "type_annotation, obj, comparator",
    [
        pytest.param(
            PydanticPmdUnit,
            pmd_unit.from_symbol("m"),
            operator.eq,
            id="pmd_unit",
        ),
        pytest.param(
            PydanticParticleGroup,
            single_particle(),
            operator.eq,
            id="particlegroup",
        ),
        pytest.param(
            NDArray,
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array_equal,
            id="ndarray",
        ),
    ],
)
def test_json_serialization(type_annotation, obj, comparator):
    adapter = TypeAdapter(type_annotation)
    assert comparator(adapter.validate_python(adapter.dump_python(obj)), obj)
    assert comparator(adapter.validate_json(adapter.dump_json(obj)), obj)
