import numpy as np

import threading
import time

from .. import AnyTao, SubprocessTao
from .test_interface_commands import new_tao


def test_get_active_beam_track_element(tao_cls: type[AnyTao]):
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


def test_cli_progress_bar(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall"
    ) as tao:
        set_gaussian(tao, n_particle=1)
        tao.track_beam(use_progress_bar=False)
        tao.track_beam(use_progress_bar=True)


def _sleep_cmd(tao: SubprocessTao):
    time.sleep(1)


def test_shm_read_during_tracking():
    with new_tao(
        SubprocessTao,
        init_file="$ACC_ROOT_DIR/bmad-doc/tao_examples/optics_matching/tao.init",
    ) as tao:
        observed = []
        start = threading.Event()
        stop = threading.Event()

        def poll_active_element():
            start.wait()
            while not stop.is_set():
                observed.append(tao.get_active_beam_track_element())
                time.sleep(0.1)

        poller = threading.Thread(target=poll_active_element, daemon=True)
        poller.start()
        try:
            start.set()
            # tao.track_beam(use_progress_bar=False)
            tao.subprocess_call(_sleep_cmd)
        finally:
            stop.set()
            poller.join(timeout=2)

        assert len(observed) > 5, "Should observe at least 5 element calls during the sleep"


def test_cli_progress_bar_track_start(tao_cls: type[AnyTao]):
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
