"""
Compare the floor-coordinate port in `pytao.model.floor` against Bmad, via pybmad.

Each tracking element of several lattices is converted with `FloorTable` and
with Bmad's own `coords_local_curvilinear_to_floor`, at both ends and at points
inside the element. Bends are also checked against `ele_geometry`, which gives
the reference orbit's floor pose part way through an element.

The table is built from pybmad's parsed lattice, so no Tao is involved.
"""

import os
import pathlib
import typing
from dataclasses import dataclass, field

import numpy as np
import pytest
from beamphysics import ParticleGroup

from pytao.model import FloorTable, w_matrix_from_angles
from pytao.model.floor import (
    BEND_KEYS,
    DOWNSTREAM_END,
    INSIDE,
    NO_INTERIOR_KEYS,
    UPSTREAM_END,
)

if typing.TYPE_CHECKING:
    import pybmad as pb

pb = pytest.importorskip("pybmad")

# Bmad key index to name, for the keys the conversion treats specially. All
# other keys are straight elements.
KEY_NAMES = {pb.key_name_to_key_index(name): name for name in BEND_KEYS | NO_INTERIOR_KEYS}

# Local (x, y) offsets (m) of the points checked in each element.
OFFSETS = [(0.0, 0.0), (0.01, -0.02)]

# Fractions of each element's length at which interior points are checked.
FRACTIONS = [0.0, 0.25, 0.5, 1.0]

POSITION_ATOL = 1e-9  # m; ring lattices reach hundreds of meters.
W_ATOL = 1e-12

GEOMETRY_LATTICE = """
no_digested
parameter[geometry] = open
parameter[particle] = electron
parameter[e_tot] = 1e9
beginning[beta_a] = 10
beginning[beta_b] = 10
beginning[x_position] = 0.3
beginning[y_position] = -0.2
beginning[z_position] = 1.5
beginning[theta_position] = 0.4
beginning[phi_position] = 0.1
beginning[psi_position] = -0.05
d1: drift, L = 0.5
b1: sbend, L = 1.0, angle = 0.3, ref_tilt = 0.7
vb: sbend, L = 0.6, angle = -0.2, ref_tilt = pi/2
q1: quadrupole, L = 0.3, k1 = 0.1
pt: patch, x_offset = 0.05, y_offset = -0.02, z_offset = 0.3, x_pitch = 0.02, y_pitch = -0.03, tilt = 0.1
fs: floor_shift, x_offset = 0.1, z_offset = 0.2, y_pitch = 0.05, tilt = -0.1
b2: sbend, L = 0.8, angle = -0.25, ref_tilt = -0.4
m1: marker
ln: line = (d1, b1, vb, q1, pt, fs, b2, m1)
use, ln
"""

EXAMPLE_LATTICES = {
    "csr": "$ACC_ROOT_DIR/bmad-doc/tao_examples/csr_beam_tracking/lat.bmad",
    "chicane": "$ACC_ROOT_DIR/regression_tests/pipe_test/floor_orbit.bmad",
    "cesr": "$ACC_ROOT_DIR/bmad-doc/tao_examples/cesr/bmad_L9A18A000-_MOVEREC.lat",
}


@pytest.fixture(scope="module", params=["geometry", *EXAMPLE_LATTICES])
def branch(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory):
    """Branch 0 of a lattice parsed by pybmad."""
    if request.param == "geometry":
        path = tmp_path_factory.mktemp("pybmad") / "geometry.bmad"
        path.write_text(GEOMETRY_LATTICE)
    else:
        path = pathlib.Path(os.path.expandvars(EXAMPLE_LATTICES[request.param]))
        if not path.exists():
            pytest.skip(f"Lattice not found: {path}")
    return pb.bmad_parser(str(path)).lat.branch[0]


def floor_pose(floor) -> list[float]:
    return [*np.asarray(floor.r), floor.theta, floor.phi, floor.psi]


def floor_position_struct(r) -> pb.FloorPositionStruct:
    position = pb.FloorPositionStruct()
    position.r = [float(v) for v in r]
    return position


def table_from_branch(branch) -> FloorTable:
    """A `FloorTable` built from pybmad's elements, as `FloorTable.from_tao` builds one from Tao."""
    eles = [branch.ele[ix] for ix in range(branch.n_ele_track + 1)]
    keys = [KEY_NAMES.get(ele.key, "straight") for ele in eles]
    end = [floor_pose(ele.floor) for ele in eles]
    return FloorTable(
        ix_branch=[branch.ix_branch] * len(eles),
        ix_ele=list(range(len(eles))),
        name=[ele.name for ele in eles],
        key=keys,
        s=[ele.s for ele in eles],
        g=[ele.value[pb.G] if key in BEND_KEYS else 0.0 for ele, key in zip(eles, keys)],
        ref_tilt=[
            ele.value[pb.REF_TILT_TOT] if key in BEND_KEYS else 0.0
            for ele, key in zip(eles, keys)
        ],
        orientation=[ele.orientation for ele in eles],
        beginning=[end[0], *end[:-1]],
        end=end,
    )


def bmad_to_floor(ele, r, end_origin: int | None = None):
    """Floor position and W of a local point, from Bmad's `coords_local_curvilinear_to_floor`."""
    out = pb.coords_local_curvilinear_to_floor(
        floor_position_struct(r), ele, end_origin=end_origin
    )
    return np.asarray(out.global_position.r), np.asarray(out.w_mat)


@dataclass
class Checks:
    """Local points to convert, each with Bmad's floor position and W for it."""

    labels: list[str] = field(default_factory=list)
    points: list[tuple[int, float, float, int, float]] = field(default_factory=list)
    bmad: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)

    def add(self, label: str, ix_ele: int, x: float, y: float, location: int, s: float, bmad):
        self.labels.append(label)
        self.points.append((ix_ele, x, y, location, s))
        self.bmad.append(bmad)

    def assert_match(self, table: FloorTable) -> None:
        """Convert every point in one call and report all that differ from Bmad."""
        assert self.labels
        # Three particles per point, with unit momenta along the local axes:
        # their floor momenta are the columns of W.
        ix_ele, x, y, location, s = np.repeat(np.array(self.points), 3, axis=0).T
        px, py, pz = np.tile(np.eye(3), (len(self.labels), 1)).T
        P = ParticleGroup(
            data={
                "x": x,
                "y": y,
                "z": np.zeros_like(x),
                "px": px,
                "py": py,
                "pz": pz,
                "t": np.zeros_like(x),
                "weight": np.ones_like(x),
                "status": np.ones(len(x), dtype=int),
                "species": "electron",
                "ix_ele": ix_ele.astype(int),
                "location": location.astype(int),
                "s": s,
            }
        )
        Pf = table.particles_to_floor(P)
        position = np.stack([Pf.x, Pf.y, Pf.z], axis=-1)[::3]
        w = np.stack([Pf.px, Pf.py, Pf.pz], axis=-1).reshape(-1, 3, 3).transpose(0, 2, 1)

        messages = []
        for label, r, w_mat, (bmad_r, bmad_w) in zip(self.labels, position, w, self.bmad):
            dr = np.abs(r - bmad_r).max()
            dw = np.abs(w_mat - bmad_w).max()
            if dr > POSITION_ATOL or dw > W_ATOL:
                messages.append(f"{label}: position error {dr:.3g} m, W error {dw:.3g}")
        assert not messages, (
            f"{len(messages)} of {len(self.labels)} checks differ from Bmad:\n"
            + "\n".join(messages[:20])
        )


@pytest.mark.parametrize("seed", range(5))
def test_w_matrix_matches_bmad(seed: int):
    rng = np.random.default_rng(seed)
    for theta, phi, psi in rng.uniform(-np.pi, np.pi, (20, 3)):
        expected = np.asarray(pb.floor_angles_to_w_mat(theta, phi, psi).w_mat)
        np.testing.assert_allclose(w_matrix_from_angles(theta, phi, psi), expected, atol=1e-14)


def test_element_ends_match_bmad(branch):
    """Particles at each element's upstream and downstream ends."""
    table = table_from_branch(branch)
    checks = Checks()
    for ix, (name, key) in enumerate(zip(table.name, table.key)):
        ele = branch.ele[ix]
        length = ele.value[pb.L]
        for x, y in OFFSETS:
            label = f"{name} ({ix}, {key}) at ({x}, {y})"

            # A floor shift or fiducial changes frame with no length, and Bmad's
            # routine only special-cases patches, so the upstream end of those
            # is checked as the downstream end of the element before.
            if key in NO_INTERIOR_KEYS and key != "patch" and ix > 0:
                previous = branch.ele[ix - 1]
                upstream = bmad_to_floor(
                    previous,
                    [x, y, previous.value[pb.L]],
                    end_origin=pb.DOWNSTREAM_END,
                )
            else:
                upstream = bmad_to_floor(ele, [x, y, 0.0], end_origin=pb.UPSTREAM_END)
            downstream = bmad_to_floor(ele, [x, y, length], end_origin=pb.DOWNSTREAM_END)

            checks.add(f"{label} upstream", ix, x, y, UPSTREAM_END, ele.s, upstream)
            checks.add(f"{label} downstream", ix, x, y, DOWNSTREAM_END, ele.s, downstream)
    checks.assert_match(table)


def test_element_interiors_match_bmad(branch):
    """Particles inside each element, at several points along it."""
    table = table_from_branch(branch)
    checks = Checks()
    for ix, (name, key) in enumerate(zip(table.name, table.key)):
        if key in NO_INTERIOR_KEYS:
            continue
        ele = branch.ele[ix]
        length = ele.value[pb.L]
        for fraction in FRACTIONS:
            z_rel = fraction * length
            for x, y in OFFSETS:
                checks.add(
                    f"{name} ({ix}, {key}) at ({x}, {y}, {z_rel:.4g})",
                    ix,
                    x,
                    y,
                    INSIDE,
                    ele.s - length + z_rel,
                    bmad_to_floor(ele, [x, y, z_rel]),
                )
    checks.assert_match(table)


def test_bend_interiors_match_ele_geometry(branch):
    """The reference orbit inside bends follows Bmad's `ele_geometry`."""
    table = table_from_branch(branch)
    checks = Checks()
    for ix, (name, key) in enumerate(zip(table.name, table.key)):
        ele = branch.ele[ix]
        length = ele.value[pb.L]
        if key not in BEND_KEYS or length == 0:
            continue
        for fraction in FRACTIONS:
            floor = pb.ele_geometry(branch.ele[ix - 1].floor, ele, len_scale=fraction)
            checks.add(
                f"{name} ({ix}) at fraction {fraction}",
                ix,
                0.0,
                0.0,
                INSIDE,
                ele.s - (1 - fraction) * length,
                (np.asarray(floor.r), np.asarray(floor.w)),
            )
    if not checks.labels:
        pytest.skip("No bends in this lattice")
    checks.assert_match(table)
