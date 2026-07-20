import pathlib

import pytest

from beamphysics import ParticleGroup
from pytao import SubprocessTao, Tao


@pytest.fixture(scope="function")
def tao():
    with SubprocessTao(
        init_file="$ACC_ROOT_DIR/bmad-doc/tutorial_bmad_tao/lattice_files/multiple_universes/tao.init",
        noplot=True,
    ) as tao:
        yield tao


@pytest.fixture(scope="function")
def initial_particles():
    with SubprocessTao(
        init_file="$ACC_ROOT_DIR/bmad-doc/tutorial_bmad_tao/lattice_files/multiple_universes/tao.init",
        noplot=True,
    ) as tao:
        beam_init = tao.get_config().beam_init
        beam_init.a_norm_emit = 1e-6
        beam_init.b_norm_emit = 1e-6
        beam_init.n_particle = 10
        beam_init.bunch_charge = 1e-10
        beam_init.set(tao, only_changed=True)

        tao.track_beam(use_progress_bar=False)
        return tao.particles("BEGINNING")


def test_initial_particles_empty(tao: Tao):
    for ix_uni in tao.inum("ix_universe"):
        assert tao.get_initial_particles(ix_uni=ix_uni) is None


def test_initial_particles_set(tao: Tao, initial_particles: ParticleGroup):
    unis = tao.inum("ix_universe")
    particles_by_universe: dict[int, ParticleGroup] = {
        ix_uni: initial_particles.split(n_chunks=ix_uni)[0] for ix_uni in unis
    }

    # Set all of the particles first
    for ix_uni, P0 in particles_by_universe.items():
        assert tao.get_initial_particles(ix_uni=ix_uni) is None
        tao.set_initial_particles(P0, ix_uni=ix_uni)

    # Then make sure they are all readable
    for ix_uni, P0 in particles_by_universe.items():
        P1 = tao.get_initial_particles(ix_uni=ix_uni)
        assert P1 is not None
        assert len(P1) == len(P0)
        assert P1.charge == pytest.approx(P0.charge)

        assert P0 == P1

    tao.set_initial_particles(initial_particles, n_particle=5)
    P1 = tao.get_initial_particles()
    assert P1 is not None
    assert len(P1) == 5


def test_initial_particles_set_by_filename(
    tao: Tao, initial_particles: ParticleGroup, tmp_path: pathlib.Path
):
    unis = tao.inum("ix_universe")
    particles_by_universe: dict[int, ParticleGroup] = {
        ix_uni: initial_particles.split(n_chunks=ix_uni)[0] for ix_uni in unis
    }

    # Set all of the particles first
    for ix_uni, P0 in particles_by_universe.items():
        assert tao.get_initial_particles(ix_uni=ix_uni) is None

        fn = tmp_path / f"{ix_uni}.h5"
        P0.write(fn)
        tao.set_initial_particles(fn, ix_uni=ix_uni)

    # Then make sure they are all readable
    for ix_uni, P0 in particles_by_universe.items():
        fn = tmp_path / f"{ix_uni}.h5"

        P1 = tao.get_initial_particles(ix_uni=ix_uni)
        assert P1 is not None
        assert len(P1) == len(P0)
        assert P1.charge == pytest.approx(P0.charge)

        assert P0 == P1

    tao.set_initial_particles(initial_particles, n_particle=5)
    P1 = tao.get_initial_particles()
    assert P1 is not None
    assert len(P1) == 5
