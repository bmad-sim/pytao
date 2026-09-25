import threading
import time

from .. import AnyTao, SubprocessTao
from .conftest import set_gaussian
from .test_interface_commands import new_tao


def test_get_active_beam_track_element(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall"
    ) as tao:
        assert tao.get_active_beam_track_element() == -1


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
