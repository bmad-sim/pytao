"""
Floor (global) coordinates of particles from a compact per-element table.

Bmad expresses particle coordinates in the curvilinear frame of the element a
particle is in. `FloorTable` holds, for each tracking element, just enough
geometry to map those coordinates to the global "floor" frame without a running
Tao: the entrance and exit floor poses, and for bends the curvature and
reference tilt. The geometry follows Bmad's
`coords_local_curvilinear_to_floor` and `bend_shift`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pydantic

from .base import TaoBaseModel
from .ele.sections import ElementFloorPosition, Which, w_matrix_from_angles
from .types import NDArray

if TYPE_CHECKING:
    from beamphysics import ParticleGroup
    from typing_extensions import Self

    from ..tao import Tao
    from .ele import Element


# Element keys with curved (arc) geometry.
BEND_KEYS = frozenset({"sbend", "rf_bend"})

# Element keys whose interior has no simple curvilinear geometry. Particles at
# either end are supported through the stored entrance and exit poses.
NO_INTERIOR_KEYS = frozenset(
    {"patch", "floor_shift", "fiducial", "mirror", "crystal", "multilayer_mirror"}
)

# openPMD `locationInElement` values, as carried by `ParticleGroup.location`.
UPSTREAM_END = -1
INSIDE = 0
DOWNSTREAM_END = 1

_POSE_FIELDS = ("x", "y", "z", "theta", "phi", "psi")


def _sinc(x: np.ndarray) -> np.ndarray:
    return np.sinc(x / np.pi)


def _pose(position: ElementFloorPosition) -> list[float]:
    return [getattr(position, name) for name in _POSE_FIELDS]


def _local_to_floor(
    pose: np.ndarray,
    g: np.ndarray,
    ref_tilt: np.ndarray,
    ds: np.ndarray,
    position: np.ndarray,
    momentum: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert local positions and momenta to the floor frame.

    Each point is first moved along its element to the frame of `pose`, a
    distance `ds` downstream, as in Bmad's `bend_shift`. The shift is written
    with sinc so that `g = 0` gives a straight element.

    Parameters
    ----------
    pose : numpy.ndarray
        Shape `(n, 6)` floor poses `(x, y, z, theta, phi, psi)`.
    g : numpy.ndarray
        Shape `(n,)` curvature (1/m); 0 for a straight element.
    ref_tilt : numpy.ndarray
        Shape `(n,)` reference tilt (rad) of the bend plane.
    ds : numpy.ndarray
        Shape `(n,)` arc length (m) from each point to the frame of `pose`.
    position : numpy.ndarray
        Shape `(n, 3)` local `(x, y, z)` positions (m).
    momentum : numpy.ndarray
        Shape `(n, 3)` local momenta; any units.

    Returns
    -------
    position, momentum : numpy.ndarray
        Shape `(n, 3)` floor positions (m) and momenta (input units).
    """
    angle = g * ds
    tilt = w_matrix_from_angles(0, 0, ref_tilt)
    # Rotation from the local frame at the point to the frame of `pose`.
    S = tilt @ w_matrix_from_angles(angle, 0, 0) @ tilt.transpose(0, 2, 1)
    arc = -ds[:, np.newaxis] * np.stack(
        [np.sin(angle / 2) * _sinc(angle / 2), np.zeros_like(angle), _sinc(angle)],
        axis=-1,
    )
    shifted = np.matvec(S, position) + np.matvec(tilt, arc)

    W = w_matrix_from_angles(pose[:, 3], pose[:, 4], pose[:, 5])
    return pose[:, :3] + np.matvec(W, shifted), np.matvec(W @ S, momentum)


class FloorTable(TaoBaseModel):
    """
    Per-element floor geometry for converting particles to floor coordinates.

    One entry per tracking element: the floor pose at the element's entrance
    (`beginning`) and exit (`end`), and for bends the curvature and reference
    tilt. That is enough to place a particle anywhere in a straight element or
    bend, and at either end of any element, without Tao.

    | Field                 | Unit     | Description                                   |
    | --------------------- | -------- | --------------------------------------------- |
    | `ix_branch`, `ix_ele` |          | Branch and element index                      |
    | `name`, `key`         |          | Element name and key, e.g. `SBend`            |
    | `s`                   | m        | Exit s-position                               |
    | `g`, `ref_tilt`       | 1/m, rad | Bend curvature and tilt; 0 for non-bends      |
    | `beginning`, `end`    | m, rad   | Entrance and exit floor poses, shape `(n, 6)` |

    Each pose is `(x, y, z, theta, phi, psi)`, as in `ElementFloorPosition`.
    `ref_tilt` is Bmad's `ref_tilt_tot`, which includes any offset girder's tilt.

    Build one with `FloorTable.from_tao` (or `Tao.floor_table`) or
    `FloorTable.from_elements` (or `Lattice.floor_table`), save it with `write`
    (json, yaml, msgpack), and load it later with `from_file`.

    Reversed elements (orientation -1) are not supported.
    """

    which: Which = "model"
    ix_branch: NDArray
    ix_ele: NDArray
    name: tuple[str, ...]
    key: tuple[str, ...]
    s: NDArray
    g: NDArray
    ref_tilt: NDArray
    beginning: NDArray
    end: NDArray

    @pydantic.model_validator(mode="after")
    def _check_lengths(self) -> Self:
        lengths = {
            name: len(getattr(self, name))
            for name in type(self).model_fields
            if name != "which"
        }
        if len(set(lengths.values())) > 1:
            raise ValueError(
                f"FloorTable columns must all have the same length; got {lengths}"
            )
        return self

    def __len__(self) -> int:
        return len(self.ix_ele)

    def __repr__(self) -> str:
        branches = np.unique(self.ix_branch).tolist()
        where = f"branch {branches[0]}" if len(branches) == 1 else f"branches {branches}"
        return (
            f"<FloorTable which={self.which} with {len(self)} elements "
            f"({np.count_nonzero(self.g)} bends) in {where}>"
        )

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        *,
        ix_branch: int | Sequence[int] | None = None,
        ix_uni: str | int = "",
        which: Which = "model",
    ) -> Self:
        """
        Fetch the floor table of one or more lattice branches from Tao.

        Uses a fixed number of `lat_list` calls per branch. Each element's
        entrance pose is the exit pose of the element before it.

        Parameters
        ----------
        tao : Tao
        ix_branch : int or sequence of int, optional
            Branch or branches to include. Default is every branch of the universe.
        ix_uni : str or int, optional
            Universe index. Default is the current universe.
        which : "base", "model", or "design", optional
            Default is "model".

        Returns
        -------
        FloorTable
        """
        if ix_branch is None:
            branches = [branch["index"] for branch in tao.lat_branch_list(ix_uni=str(ix_uni))]
        elif isinstance(ix_branch, int):
            branches = [ix_branch]
        else:
            branches = sorted(set(ix_branch))
        columns = [_branch_columns(tao, branch, ix_uni, which) for branch in branches]
        return cls(
            which=which,
            **{name: [value for col in columns for value in col[name]] for name in columns[0]},
        )

    @classmethod
    def from_elements(cls, elements: Iterable[Element], *, which: Which = "model") -> Self:
        """
        Build a table from pytao `Element`s with `floor` filled.

        Bends also need `attrs` filled for their curvature and tilt. Lords with
        slaves, which have no floor pose of their own, are skipped since
        particles are never in them.

        Raises
        ------
        ValueError
            If an element's `floor`, or a bend's `attrs`, is not filled.
        """
        rows: dict[tuple[int, int], dict[str, Any]] = {}
        for ele in elements:
            head = ele.head
            if ele.floor is None:
                raise ValueError(
                    f"`floor` is unavailable for {head.name}. Fill it first using the "
                    f"Tao object (see `Element.fill`)"
                )
            beginning = ele.floor.beginning.reference
            end = ele.floor.end.reference
            if beginning is None or end is None:
                continue

            g = ref_tilt = 0.0
            if head.key.lower() in BEND_KEYS:
                g = float(ele.attribs.get("g") or 0.0)
                ref_tilt = float(ele.attribs.get("ref_tilt_tot") or 0.0)

            rows[head.ix_branch, head.ix_ele] = {
                "ix_branch": head.ix_branch,
                "ix_ele": head.ix_ele,
                "name": head.name,
                "key": head.key,
                "s": head.s,
                "g": g,
                "ref_tilt": ref_tilt,
                "beginning": _pose(beginning),
                "end": _pose(end),
            }
        ordered = [rows[pair] for pair in sorted(rows)]
        names = [name for name in cls.model_fields if name != "which"]
        return cls(which=which, **{name: [row[name] for row in ordered] for name in names})

    def particles_to_floor(
        self, P: ParticleGroup, *, magnification: float = 1.0
    ) -> ParticleGroup:
        """
        Convert particles from element-local to floor coordinates.

        Each particle is placed using its element (`ix_branch`, `ix_ele`) and
        its `location` in that element: -1 upstream end, 1 downstream end, or 0
        inside at longitudinal position `s`. Without a `location` record,
        particles are taken to be at the downstream end, where Bmad saves bunches.
        `P.z` is a straight offset along the local z axis; it is 0 for Bmad bunch
        data.

        Parameters
        ----------
        P : ParticleGroup
            Particles with an `ix_ele` record, which may be omitted when the
            table holds a single element. `ix_branch` may be omitted when the
            table holds a single branch. Particles inside an element need `s`.
        magnification : float, optional
            Scale the local `x` and `y` offsets from the reference orbit by this
            factor before converting, to make a small beam visible when drawn
            at the scale of the machine. Default is 1.

        Returns
        -------
        ParticleGroup
            A new group with `x`, `y`, `z` the floor position (m) and `px`, `py`,
            `pz` the momentum (eV/c) in the floor frame. Time, weight, status,
            species and every optional record are unchanged.

        Raises
        ------
        ValueError
            If `ix_ele` is missing, `ix_branch` is ambiguous, an element is not in
            the table, or a particle inside an element has no `s`.
        NotImplementedError
            For a particle inside an element with no simple interior geometry,
            such as a patch.
        """
        from beamphysics import ParticleGroup

        row = self._particle_rows(P)
        n = len(row)
        location = (
            np.asarray(P.location, dtype=int)
            if "location" in P
            else np.full(n, DOWNSTREAM_END)
        )
        bad = set(np.unique(location).tolist()) - {UPSTREAM_END, INSIDE, DOWNSTREAM_END}
        if bad:
            raise ValueError(f"Invalid location values {sorted(bad)}; expected -1, 0 or 1")

        # Arc length from each particle to the exit frame of its element.
        ds = np.zeros(n)
        inside = location == INSIDE
        if inside.any():
            self._check_interior(row[inside])
            if "s" not in P:
                raise ValueError(
                    "Particles inside an element (location 0) need an `s` record to "
                    "place them along the element."
                )
            ds[inside] = self.s[row[inside]] - np.asarray(P.s, dtype=float)[inside]

        upstream = (location == UPSTREAM_END)[:, np.newaxis]
        position, momentum = _local_to_floor(
            pose=np.where(upstream, self.beginning[row], self.end[row]),
            g=self.g[row],
            ref_tilt=self.ref_tilt[row],
            ds=ds,
            position=np.stack([magnification * P.x, magnification * P.y, P.z], axis=-1),
            momentum=np.stack([P.px, P.py, P.pz], axis=-1),
        )
        return ParticleGroup(
            data={
                **P.data,
                "x": position[:, 0],
                "y": position[:, 1],
                "z": position[:, 2],
                "px": momentum[:, 0],
                "py": momentum[:, 1],
                "pz": momentum[:, 2],
            }
        )

    def points_to_floor(
        self,
        s: np.ndarray,
        x: np.ndarray | float = 0.0,
        y: np.ndarray | float = 0.0,
        *,
        ix_branch: int | None = None,
        magnification: float = 1.0,
    ) -> np.ndarray:
        """
        Convert points along a branch from local to floor coordinates.

        Each point is placed inside the element that contains its `s` position.
        Use this for anything known as a function of `s`, such as an orbit or
        a beam envelope.

        Parameters
        ----------
        s : numpy.ndarray
            Shape `(n,)` longitudinal positions (m).
        x, y : numpy.ndarray or float, optional
            Local offsets (m) from the reference orbit at each `s`. Default is 0,
            the reference orbit itself.
        ix_branch : int, optional
            Branch the points are in. May be omitted when the table holds a
            single branch.
        magnification : float, optional
            Scale `x` and `y` by this factor before converting. Default is 1.

        Returns
        -------
        numpy.ndarray
            Shape `(n, 3)` floor `(x, y, z)` positions (m).

        Raises
        ------
        ValueError
            If `ix_branch` is ambiguous or not in the table, or `s` is past the
            end of the branch.
        NotImplementedError
            For a point inside an element with no simple interior geometry,
            such as a patch.
        """
        branches = np.unique(self.ix_branch).tolist()
        if ix_branch is None:
            if len(branches) != 1:
                raise ValueError(
                    f"`ix_branch` is required; the table has branches {branches}."
                )
            ix_branch = branches[0]
        rows = np.flatnonzero(self.ix_branch == ix_branch)
        if len(rows) == 0:
            raise ValueError(f"Branch {ix_branch} is not in the table; it has {branches}.")
        rows = rows[np.argsort(self.s[rows], kind="stable")]

        s, x, y = np.broadcast_arrays(np.atleast_1d(np.asarray(s, dtype=float)), x, y)
        # The table holds exit positions only, so the start of the first
        # element is unknown and there is no lower bound to check.
        s_max = self.s[rows[-1]]
        if np.any(s > s_max + 1e-9 * max(1.0, abs(s_max))):
            raise ValueError(
                f"`s` must not be past the end of branch {ix_branch} at {s_max} m; "
                f"got up to {s.max()} m."
            )
        s = np.minimum(s, s_max)

        # The first element whose exit is at or past each point contains it.
        row = rows[np.searchsorted(self.s[rows], s)]
        ds = self.s[row] - s
        self._check_interior(row[ds > 0])
        position, _ = _local_to_floor(
            pose=self.end[row],
            g=self.g[row],
            ref_tilt=self.ref_tilt[row],
            ds=ds,
            position=np.stack(
                [magnification * x, magnification * y, np.zeros_like(s)], axis=-1
            ),
            momentum=np.zeros((len(s), 3)),
        )
        return position

    def _check_interior(self, rows: np.ndarray) -> None:
        unsupported = sorted(
            f"{self.name[i]} ({self.key[i]})"
            for i in np.unique(rows)
            if self.key[i].lower() in NO_INTERIOR_KEYS
        )
        if unsupported:
            raise NotImplementedError(
                f"Points inside {', '.join(unsupported)} cannot be converted; "
                f"only the upstream and downstream ends are supported."
            )

    def _particle_rows(self, P: ParticleGroup) -> np.ndarray:
        if "ix_ele" not in P:
            if len(self) != 1:
                raise ValueError(
                    "Particles need an `ix_ele` record to find their element; for "
                    "particles all in one element, use `Element.particles_to_floor`."
                )
            return np.zeros(len(P), dtype=int)

        ix_ele = np.asarray(P.ix_ele, dtype=int)
        if "ix_branch" in P:
            ix_branch = np.asarray(P.ix_branch, dtype=int)
        else:
            branches = np.unique(self.ix_branch)
            if len(branches) != 1:
                raise ValueError(
                    f"Particles have no `ix_branch` record and the table has branches "
                    f"{branches.tolist()}."
                )
            ix_branch = np.full(len(ix_ele), branches[0])

        lookup = {
            pair: row
            for row, pair in enumerate(zip(self.ix_branch.tolist(), self.ix_ele.tolist()))
        }
        pairs = list(zip(ix_branch.tolist(), ix_ele.tolist()))
        missing = sorted(set(pairs) - set(lookup))
        if missing:
            raise ValueError(
                f"Particles are in elements (ix_branch, ix_ele) not in the floor table: "
                f"{missing}. ix_ele -2 is Bmad's marker for a temporary element slice, "
                f"which is not a lattice element."
            )
        return np.array([lookup[pair] for pair in pairs], dtype=int)


def _branch_columns(
    tao: Tao, ix_branch: int, ix_uni: str | int, which: Which
) -> dict[str, Any]:
    def lat_list(elements: str, who: str, flags: str = "-array_out -track_only -index_order"):
        return tao.lat_list(
            elements,
            who,
            ix_uni=str(ix_uni),
            ix_branch=str(ix_branch),
            which=which,
            flags=flags,
        )

    ix_ele = np.asarray(lat_list("*", "ele.ix_ele"), dtype=int)
    g = np.zeros(len(ix_ele))
    ref_tilt = np.zeros(len(ix_ele))
    for bend_key in BEND_KEYS:
        bend_ix_ele = lat_list(f"{bend_key}::*", "ele.ix_ele")
        if len(bend_ix_ele) == 0:
            continue
        # -index_order sorts ix_ele, so this finds each bend's row.
        bend_rows = np.searchsorted(ix_ele, bend_ix_ele)
        g[bend_rows] = lat_list(f"{bend_key}::*", "ele.g")
        ref_tilt[bend_rows] = lat_list(f"{bend_key}::*", "ele.ref_tilt_tot")

    end = np.stack([lat_list("*", f"ele.{name}_position") for name in _POSE_FIELDS], axis=-1)
    return {
        "ix_branch": np.full(len(ix_ele), ix_branch),
        "ix_ele": ix_ele,
        "name": lat_list("*", "ele.name", flags="-track_only -index_order"),
        "key": lat_list("*", "ele.key", flags="-track_only -index_order"),
        "s": lat_list("*", "ele.s"),
        "g": g,
        "ref_tilt": ref_tilt,
        "beginning": np.concatenate([end[:1], end[:-1]]),
        "end": end,
    }
