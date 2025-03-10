from typing import Type

import numpy as np

from .. import AnyTao
from .test_interface_commands import new_tao


def test_get_active_beam_track_element(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall"
    ) as tao:
        assert tao.get_active_beam_track_element() == -1


def set_gaussian(
    tao: AnyTao,
    n_particle: int,
    a_norm_emit: float = 1.0e-6,
    b_norm_emit: float = 1.0e-6,
    bunch_charge: float = 1e-9,
    sig_pz0: float = 2e-6,
    sig_z: float = 200e-6,
    center_pz: float = 0.0,
    chirp: float = 0.0,  # 1/m
):
    sig_pz = np.hypot(sig_pz0, chirp * sig_z)

    cmds = [
        f"set beam_init n_particle = {n_particle}",
        "set beam_init random_engine = quasi",
        "set beam_init saved_at = MARKER::*, BEGINNING, END",
        f"set beam_init a_norm_emit = {a_norm_emit}",
        f"set beam_init b_norm_emit = {b_norm_emit}",
        f"set beam_init bunch_charge = {bunch_charge}",
        f"set beam_init sig_pz = {sig_pz}",
        f"set beam_init sig_z = {sig_z}",
        f"set beam_init dpz_dz = {chirp}",
        f"set beam_init center(6) = {center_pz}",
    ]
    tao.cmds(cmds)
    tao.cmd("set global lattice_calc_on = T")


def test_cli_progress_bar(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall"
    ) as tao:
        set_gaussian(tao, n_particle=1)
        tao.track_beam(use_progress_bar=False)
        tao.track_beam(use_progress_bar=True)


def test_cli_progress_bar_track_start(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall"
    ) as tao:
        set_gaussian(tao, n_particle=1)
        tao.cmd("set global lattice_calc_on = F")
        tao.track_beam("BEGinning", "eND", use_progress_bar=True)
        assert not tao.tao_global()["lattice_calc_on"]
        assert tao.beam(ix_branch=0)["track_start"] == "BEGinning"
        assert tao.beam(ix_branch=0)["track_end"] == "eND"

        tao.cmd("set global lattice_calc_on = T")
        assert tao.tao_global()["lattice_calc_on"]
        tao.track_beam(track_end="END", use_progress_bar=True)
        assert tao.tao_global()["lattice_calc_on"]

        assert tao.beam(ix_branch=0)["track_start"] == "BEGinning"
        assert tao.beam(ix_branch=0)["track_end"] == "END"
