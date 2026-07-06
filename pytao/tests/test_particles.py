import pytest

from pytao import SubprocessTao, Tao


@pytest.fixture(scope="function")
def tao():
    with SubprocessTao(
        init_file="$ACC_ROOT_DIR/bmad-doc/tutorial_bmad_tao/lattice_files/multiple_universes/tao.init",
        noplot=True,
    ) as tao:
        yield tao


def test_initial_particles_empty(tao: Tao):
    for ix_uni in tao.inum("ix_universe"):
        assert tao.get_initial_particles(ix_uni=ix_uni) is None


def test_initial_particles_set(tao: Tao):
    # for ix_uni in tao.inum("ix_universe"):
    ix_uni = 1

    assert tao.get_initial_particles(ix_uni=ix_uni) is None

    beam_init = tao.get_config(ix_uni=ix_uni).beam_init
    beam_init.a_norm_emit = 1e-6
    beam_init.b_norm_emit = 1e-7
    beam_init.set(tao)

    bi = tao.get_config(ix_uni=ix_uni).beam_init

    assert bi.a_norm_emit == 1e-6
    assert bi.b_norm_emit == 1e-7

    charge = 1e-10
    tao.init_particles(10, charge, ix_uni=1)
    # tao.init_particles(100, 1e-15, ix_uni=2)

    P1 = tao.get_initial_particles(ix_uni=ix_uni)
    assert P1 is not None
    assert len(P1) == 10
    assert P1.charge == pytest.approx(charge)
