import pathlib

import numpy as np
import pytest
from beamphysics import ParticleGroup

from pytao import SubprocessTao, Tao
from pytao.tao import PARTICLE_EXTRAS
from pytao.model import Lattice


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

        tao.cmd(f"set beam {ix_uni}@track_end = b")
        tao.set_initial_particles(fn, ix_uni=ix_uni)

    # Then make sure they are all readable
    for ix_uni, P0 in particles_by_universe.items():
        fn = tmp_path / f"{ix_uni}.h5"

        P1 = tao.get_initial_particles(ix_uni=ix_uni)
        assert P1 is not None
        assert len(P1) == len(P0)
        assert P1.charge == pytest.approx(P0.charge)

        assert P0 == P1

        # Ensure track_end was reset from 'beginning' to 'b'
        assert tao.get_config(ix_uni=ix_uni).beam.track_end == "b"

    tao.set_initial_particles(initial_particles, n_particle=5)
    P1 = tao.get_initial_particles()
    assert P1 is not None
    assert len(P1) == 5


@pytest.fixture(scope="module")
def csr_tao():
    with SubprocessTao(
        init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/csr_beam_tracking/tao.init",
        noplot=True,
    ) as tao:
        yield tao


@pytest.mark.parametrize(
    "key", ["s", "ix_branch", "ix_ele", "location", "p0c", "spin_x", "spin_y", "spin_z"]
)
def test_particles_extras_match_write_beam(csr_tao: Tao, tmp_path: pathlib.Path, key: str):
    fn = tmp_path / "beam.h5"
    csr_tao.cmd(f"write beam -at end {fn}")
    expected = ParticleGroup(str(fn))
    P = csr_tao.particles("end", extras=[key])
    assert getattr(P, key) == pytest.approx(getattr(expected, key))


def test_bunch_data_extras_all(csr_tao: Tao):
    data = csr_tao.bunch_data("end", extras=True)
    assert "id" not in data  # ix_user is unset
    n = len(data["x"])
    for key in ["s", "ix_branch", "ix_ele", "location", "p0c", "spin_x", "charge_state"]:
        assert len(data[key]) == n
    assert set(data["charge_state"]) == {-1}


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        pytest.param({}, {"p0c"}, id="default"),
        pytest.param({"extras": False}, set(), id="false"),
        pytest.param({"extras": ["s"]}, {"s"}, id="explicit"),
    ],
)
def test_bunch_data_extras_selection(csr_tao: Tao, kwargs: dict, expected: set[str]):
    data = csr_tao.bunch_data("end", **kwargs)
    assert set(data) & set(PARTICLE_EXTRAS) == expected


def test_bunch_data_extras_unknown(csr_tao: Tao):
    with pytest.raises(ValueError, match="Unsupported extras"):
        csr_tao.bunch_data("end", extras=["bogus"])


floor_extras = ["ix_ele", "ix_branch", "location", "s"]


def _positions(P: ParticleGroup) -> np.ndarray:
    return np.stack([P.x, P.y, P.z], axis=-1)


def test_element_particles_to_floor(csr_tao: Tao):
    P = csr_tao.particles("m1", extras=floor_extras)
    ele = csr_tao.ele("m1")
    Pf = ele.particles_to_floor(P)

    reference = ele.floor.end.reference
    # m1 sits after part of a bend, so the conversion includes a rotation.
    assert not np.allclose(reference.wmat[:, 2], [0, 0, 1])
    expected = _positions(P) @ reference.wmat.T + [reference.x, reference.y, reference.z]
    np.testing.assert_allclose(_positions(Pf), expected, atol=1e-14)
    np.testing.assert_allclose(Pf.p, P.p)
    np.testing.assert_array_equal(Pf.t, P.t)

    table = csr_tao.floor_table()
    np.testing.assert_allclose(
        _positions(table.particles_to_floor(P)), _positions(Pf), atol=1e-14
    )


def test_element_particles_to_floor_checks_index(csr_tao: Tao):
    P = csr_tao.particles("m1", extras=floor_extras)
    with pytest.raises(ValueError, match="ix_ele"):
        csr_tao.ele("Mar.CSR").particles_to_floor(P)


def test_lattice_floor_table(csr_tao: Tao, tmp_path: pathlib.Path):
    P1 = csr_tao.particles("Mar.CSR", extras=floor_extras)
    P2 = csr_tao.particles("m1", extras=floor_extras)
    P = P1 + P2

    lat = Lattice.from_tao_eles(
        csr_tao, ["Mar.CSR", "m1"], defaults=False, floor=True, attrs=True
    )
    Pf = lat.floor_table().particles_to_floor(P)
    expected = np.concatenate(
        [
            _positions(csr_tao.ele("Mar.CSR").particles_to_floor(P1)),
            _positions(csr_tao.ele("m1").particles_to_floor(P2)),
        ]
    )
    np.testing.assert_allclose(_positions(Pf), expected, atol=1e-14)

    # Saved and reloaded, the lattice converts without Tao.
    lat.write(tmp_path / "lat.json")
    offline = Lattice.from_file(tmp_path / "lat.json")
    np.testing.assert_array_equal(
        _positions(offline.floor_table().particles_to_floor(P)), _positions(Pf)
    )


def test_particles_to_floor_needs_filled_elements(csr_tao: Tao):
    P = csr_tao.particles("m1", extras=floor_extras)
    lat = Lattice.from_tao_eles(csr_tao, ["m1"], defaults=False)
    with pytest.raises(ValueError, match="Fill it first"):
        lat.floor_table()
    with pytest.raises(ValueError, match="Fill it first"):
        lat.elements[0].particles_to_floor(P)


def test_floor_table_unknown_element(csr_tao: Tao):
    P = csr_tao.particles("m1", extras=floor_extras)
    lat = Lattice.from_tao_eles(csr_tao, ["Mar.CSR"], defaults=False, floor=True)
    with pytest.raises(ValueError, match=r"\(0, 4\)"):
        lat.floor_table().particles_to_floor(P)


def test_tao_particles_never_carry_slice_index(csr_tao: Tao):
    """Particles tracked through Bmad's temporary element slices keep the parent's index."""
    for ele_id in ["Mar.CSR", "m1", "Mar.END", "end"]:
        P = csr_tao.particles(ele_id, extras=["ix_ele"])
        assert np.all(P.ix_ele >= 0), ele_id
