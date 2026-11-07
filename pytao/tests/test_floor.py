import pathlib

import numpy as np
import pytest
from beamphysics import ParticleGroup

from pytao import SubprocessTao
from pytao.model import FloorTable, w_matrix_from_angles

TILTED_LATTICE = """
parameter[geometry] = open
parameter[particle] = electron
parameter[e_tot] = 1e9
beginning[beta_a] = 10
beginning[beta_b] = 10
d1: drift, L = 0.5
b1: sbend, L = 1.0, angle = 0.3, ref_tilt = 0.7
d2: drift, L = 0.5
b2: sbend, L = 0.8, angle = -0.25, ref_tilt = -0.4
q1: quadrupole, L = 0.3, k1 = 0.1
b3: sbend, L = 0.6, angle = 0.2
d3: drift, L = 0.3
m1: marker
ln: line = (d1, b1, d2, b2, q1, b3, d3, m1)
use, ln
"""


def make_table(n: int = 1, **columns) -> FloorTable:
    """`n` straight 1 m elements from the origin along +Z, with columns overridden."""
    fields = {
        "ix_branch": [0] * n,
        "ix_ele": [1] * n,
        "name": ["E"] * n,
        "key": ["Drift"] * n,
        "s": [1.0] * n,
        "g": [0.0] * n,
        "ref_tilt": [0.0] * n,
        "beginning": [[0.0] * 6] * n,
        "end": [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]] * n,
    }
    return FloorTable(**{**fields, **columns})


@pytest.fixture
def quarter_bend() -> FloorTable:
    """
    A 90 degree bend with g = 1 entering at the origin along +Z.

    Positive g bends toward -x, so the arc is centered at (-1, 0, 0) and the
    exit is at (-1, 0, 1) heading along -X.
    """
    return make_table(
        key=["SBend"],
        s=[np.pi / 2],
        g=[1.0],
        end=[[-1.0, 0.0, 1.0, -np.pi / 2, 0.0, 0.0]],
    )


def make_particles(n: int = 4, **records) -> ParticleGroup:
    data = {
        "x": np.linspace(-0.01, 0.02, n),
        "px": 0.0,
        "y": np.linspace(0.005, -0.015, n),
        "py": 0.0,
        "z": 0.0,
        "pz": 1e6,
        "t": np.linspace(0, 1e-12, n),
        "weight": 1e-12,
        "status": 1,
        "species": "electron",
        "ix_ele": 1,
        **records,
    }
    return ParticleGroup(data=data)


def positions(P: ParticleGroup) -> np.ndarray:
    return np.stack([P.x, P.y, P.z], axis=-1)


def momenta(P: ParticleGroup) -> np.ndarray:
    return np.stack([P.px, P.py, P.pz], axis=-1)


@pytest.mark.parametrize(
    ("angles", "expected"),
    [
        pytest.param((np.pi / 2, 0, 0), [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], id="yaw"),
        pytest.param((0, np.pi / 2, 0), [[1, 0, 0], [0, 0, 1], [0, -1, 0]], id="pitch"),
        pytest.param((0, 0, np.pi / 2), [[0, -1, 0], [1, 0, 0], [0, 0, 1]], id="roll"),
    ],
)
def test_w_matrix_from_angles(angles, expected):
    np.testing.assert_allclose(w_matrix_from_angles(*angles), expected, atol=1e-15)


def test_w_matrix_vectorised_and_orthonormal():
    rng = np.random.default_rng(0)
    theta, phi, psi = rng.uniform(-np.pi, np.pi, (3, 5))
    w = w_matrix_from_angles(theta, phi, psi)
    assert w.shape == (5, 3, 3)
    np.testing.assert_allclose(
        w @ np.swapaxes(w, -1, -2), np.broadcast_to(np.eye(3), w.shape), atol=1e-14
    )
    np.testing.assert_allclose(np.linalg.det(w), 1.0)
    np.testing.assert_allclose(w[2], w_matrix_from_angles(theta[2], phi[2], psi[2]))


def test_quarter_bend_interior(quarter_bend: FloorTable):
    angle = np.array([0.0, 0.3, 0.9, np.pi / 2])
    P = make_particles(location=0, s=angle)
    Pf = quarter_bend.particles_to_floor(P)

    x = P.x
    expected = np.stack([-1 + (1 + x) * np.cos(angle), P.y, (1 + x) * np.sin(angle)], axis=-1)
    np.testing.assert_allclose(positions(Pf), expected, atol=1e-14)
    tangent = np.stack([-np.sin(angle), np.zeros(4), np.cos(angle)], axis=-1)
    np.testing.assert_allclose(momenta(Pf), 1e6 * tangent, atol=1e-8)


@pytest.mark.parametrize(("location", "angle"), [(-1, 0.0), (1, np.pi / 2)])
def test_quarter_bend_ends(quarter_bend: FloorTable, location: int, angle: float):
    P = make_particles(location=location)
    Pf = quarter_bend.particles_to_floor(P)
    expected = np.stack(
        [-1 + (1 + P.x) * np.cos(angle), P.y, (1 + P.x) * np.sin(angle)], axis=-1
    )
    np.testing.assert_allclose(positions(Pf), expected, atol=1e-14)


def test_default_location_is_downstream_end(quarter_bend: FloorTable):
    P = make_particles()
    Pf = quarter_bend.particles_to_floor(P)
    np.testing.assert_allclose(
        positions(Pf), positions(quarter_bend.particles_to_floor(make_particles(location=1)))
    )


def test_straight_interior_and_local_z():
    table = make_table(s=[3.0], end=[[1.0, 0.0, 3.0, 0.0, 0.0, 0.0]])
    P = make_particles(location=0, s=2.5, z=0.1)
    Pf = table.particles_to_floor(P)
    np.testing.assert_allclose(Pf.x, 1.0 + P.x)
    np.testing.assert_allclose(Pf.y, P.y)
    np.testing.assert_allclose(Pf.z, 2.6)


def test_keeps_other_records(quarter_bend: FloorTable):
    P = make_particles(location=1, s=np.pi / 2, ix_branch=0)
    P.assign_id()
    Pf = quarter_bend.particles_to_floor(P)
    np.testing.assert_allclose(Pf.p, P.p)
    np.testing.assert_array_equal(Pf.t, P.t)
    np.testing.assert_array_equal(Pf.weight, P.weight)
    np.testing.assert_array_equal(Pf.status, P.status)
    for key in ("id", "s", "ix_ele", "ix_branch", "location"):
        np.testing.assert_array_equal(Pf[key], P[key])
    assert Pf.species == P.species


def test_patch_interior_not_supported():
    table = make_table(key=["Patch"])
    with pytest.raises(NotImplementedError, match="Patch"):
        table.particles_to_floor(make_particles(location=0, s=0.5))
    # Its ends use the stored entrance and exit poses.
    Pf = table.particles_to_floor(make_particles(location=-1))
    np.testing.assert_allclose(Pf.z, 0.0)


@pytest.mark.parametrize(
    ("P", "match"),
    [
        pytest.param(make_particles(location=0), "need an `s` record", id="inside-without-s"),
        pytest.param(make_particles(location=3), "Invalid location", id="bad-location"),
        pytest.param(make_particles(ix_ele=7), r"\(0, 7\)", id="unknown-element"),
    ],
)
def test_conversion_errors(quarter_bend: FloorTable, P: ParticleGroup, match: str):
    with pytest.raises(ValueError, match=match):
        quarter_bend.particles_to_floor(P)


def test_ix_ele_optional_for_single_element(quarter_bend: FloorTable):
    data = make_particles().data
    del data["ix_ele"]
    P = ParticleGroup(data=data)
    np.testing.assert_array_equal(
        positions(quarter_bend.particles_to_floor(P)),
        positions(quarter_bend.particles_to_floor(make_particles())),
    )
    with pytest.raises(ValueError, match="ix_ele"):
        make_table(n=2, ix_ele=[1, 2]).particles_to_floor(P)


def test_ambiguous_branch():
    table = make_table(n=2, ix_branch=[0, 1])
    with pytest.raises(ValueError, match="no `ix_branch`"):
        table.particles_to_floor(make_particles())
    Pf = table.particles_to_floor(make_particles(ix_branch=1))
    assert len(Pf) == 4


def test_column_lengths_checked():
    table = make_table(n=2, ix_ele=[1, 2])
    with pytest.raises(ValueError, match="FloorTable columns must all have the same length"):
        FloorTable.model_validate({**table.model_dump(), "s": [1.0]})


@pytest.mark.parametrize(
    "filename", ["floor.json", "floor.json.gz", "floor.msgpack", "floor.yaml"]
)
def test_archive_round_trip(quarter_bend: FloorTable, tmp_path: pathlib.Path, filename: str):
    quarter_bend.write(tmp_path / filename)
    assert FloorTable.from_file(tmp_path / filename) == quarter_bend


@pytest.fixture(scope="module")
def tilted_tao(tmp_path_factory: pytest.TempPathFactory):
    path = tmp_path_factory.mktemp("floor") / "tilted.bmad"
    path.write_text(TILTED_LATTICE)
    with SubprocessTao(lattice_file=str(path), noplot=True) as tao:
        yield tao


@pytest.fixture(scope="module")
def tilted_table(tilted_tao: SubprocessTao) -> FloorTable:
    return tilted_tao.floor_table()


def test_tilted_table_from_tao(tilted_table: FloorTable):
    assert list(tilted_table.ix_ele) == list(range(len(tilted_table)))
    is_bend = tilted_table.g != 0
    assert [name for name, bend in zip(tilted_table.name, is_bend) if bend] == [
        "B1",
        "B2",
        "B3",
    ]
    np.testing.assert_allclose(tilted_table.g[is_bend], [0.3, -0.3125, 1 / 3])
    np.testing.assert_allclose(tilted_table.ref_tilt[is_bend], [0.7, -0.4, 0.0])
    # Each entrance pose is the previous element's exit pose.
    np.testing.assert_array_equal(tilted_table.beginning[1:], tilted_table.end[:-1])


def test_from_tao_branches(tmp_path: pathlib.Path):
    """All branches are fetched by default, each matching Tao's element floor positions."""
    forked = TILTED_LATTICE.replace(
        "ln: line = (d1, b1, d2,",
        """f1: fork, to_line = side
d4: drift, L = 0.4
b4: sbend, L = 0.5, angle = 0.1, ref_tilt = 0.2
side: line = (d4, b4)
ln: line = (d1, b1, f1, d2,""",
    )
    assert forked != TILTED_LATTICE
    (tmp_path / "forked.bmad").write_text(forked)
    with SubprocessTao(lattice_file=str(tmp_path / "forked.bmad"), noplot=True) as tao:
        table = tao.floor_table()
        assert sorted(set(table.ix_branch.tolist())) == [0, 1]
        assert tao.floor_table(ix_branch=[1, 0, 1]) == table
        assert set(tao.floor_table(ix_branch=1).ix_branch.tolist()) == {1}
        for ix_branch, ix_ele, end in zip(table.ix_branch, table.ix_ele, table.end):
            floor = tao.ele_floor(f"{ix_branch}>>{ix_ele}", where="end")
            np.testing.assert_allclose(end, floor["Reference"], atol=1e-14)


@pytest.mark.parametrize("where", ["beginning", "center", "end"])
def test_matches_tao_floor(tilted_tao: SubprocessTao, tilted_table: FloorTable, where: str):
    """Points on and off the reference orbit match Tao's own element floor positions."""
    offset = np.array([0.01, -0.02, 0.0])
    for row in range(len(tilted_table)):
        ix, s_end = int(tilted_table.ix_ele[row]), tilted_table.s[row]
        L = s_end - tilted_table.s[max(row - 1, 0)]
        s = {"beginning": s_end - L, "center": s_end - L / 2, "end": s_end}[where]
        floor = tilted_tao.ele_floor(f"0>>{ix}", where=where)
        R = floor["Reference"][:3]
        W = floor["Reference-W"].reshape(3, 3).T

        P = make_particles(
            n=2, x=[0.0, offset[0]], y=[0.0, offset[1]], ix_ele=ix, location=0, s=s
        )
        Pf = tilted_table.particles_to_floor(P)
        name = tilted_table.name[row]
        np.testing.assert_allclose(positions(Pf)[0], R, atol=1e-12, err_msg=name)
        np.testing.assert_allclose(positions(Pf)[1], R + W @ offset, atol=1e-12, err_msg=name)
        np.testing.assert_allclose(momenta(Pf)[0] / 1e6, W[:, 2], atol=1e-12, err_msg=name)


def test_offline_from_saved_table(tilted_table: FloorTable, tmp_path: pathlib.Path):
    tilted_table.write(tmp_path / "floor.json")
    offline = FloorTable.from_file(tmp_path / "floor.json")
    P = make_particles(ix_ele=2, location=0, s=1.0)
    np.testing.assert_array_equal(
        positions(offline.particles_to_floor(P)), positions(tilted_table.particles_to_floor(P))
    )
