from __future__ import annotations

import pathlib
import typing

import numpy as np
from beamphysics.species import mass_of
from typing_extensions import Self

from ..base import ArchiveFormat, TaoModel, load_model_data
from ..types import FloatSequence

if typing.TYPE_CHECKING:
    from pytao import Tao

comb_phase_space_int = {
    "x": "1",
    "px": "2",
    "y": "3",
    "py": "4",
    "z": "5",
    "pz": "6",
}
comb_phase_space_label = {
    1: "x",
    2: "px",
    3: "y",
    4: "py",
    5: "z",
    6: "p",
}


def comb_covariance_data(tao: Tao):
    """
    Returns all covariance (sigma matrix) data from the bunch comb.

    Returns
    -------
    dict with keys:
        'cov_{a}__{b}'
            where a and b are one of:
                'x':
                    x position in meters
                'px':
                    x momentum in eV/c
                'y':
                    y position in meters
                'py':
                    y momentum in eV/c
                'z':
                    z position in meters
                    Note this is really Bmad's z = -beta*c*(t-t_ref)
                'p':
                    total momentum in eV/c
        'sigma_{a}'
            sqrt of the diagonals above
        'p0c':
            reference momentum in eV
        'sigma_delta':
            relative energy spread for convenience
    """

    # remove Bmad units
    p0c = tao.bunch_comb("p0c")

    data = {}
    data["p0c"] = p0c

    for i in comb_phase_space_label:
        for j in comb_phase_space_label:
            if i <= j:
                k1 = comb_phase_space_label[i]
                k2 = comb_phase_space_label[j]

                key = f"cov_{k1}__{k2}"
                value = tao.bunch_comb(f"sigma.{i}{j}")

                # Convert from Bmad units px/p0, py/p0, p/p0 - 1
                if i % 2 == 0:
                    value = value * p0c
                if j % 2 == 0:
                    value = value * p0c

                data[key] = value

    # Add diagonals for convenience
    for _, x in comb_phase_space_label.items():
        data[f"sigma_{x}"] = np.sqrt(data[f"cov_{x}__{x}"])

    # Add back delta for convenience
    data["sigma_delta"] = data["sigma_p"] / p0c

    return data


def comb_misc_data(tao: Tao):
    """
    Returns a dict of miscellaneous comb data:
        'charge_live'
        'n_particle_live'
        'ix_ele'
        'n_particle_lost_in_ele'

    """
    keys = "charge_live", "n_particle_live", "ix_ele", "n_particle_lost_in_ele"
    return {key: tao.bunch_comb(key) for key in keys}


def comb_emittance_data(tao: Tao):
    """
    Returns a dict of comb arrays with keys:
        'norm_emit_x'
        'norm_emit_y'
    """
    data = {}
    for plane in ("x", "y"):
        data[f"norm_emit_{plane}"] = tao.bunch_comb(f"{plane}.norm_emit")
    return data


def comb_energy_data(tao: Tao, ix_branch: int = 0):
    """
    Returns a dict of bunch comb energy info, with keys
        'mean_p'
        'mean_energy'
    """
    data = {}
    p0c = tao.bunch_comb("p0c")
    species = tao.beam_init(ix_branch=ix_branch)["species"]
    if species == "":
        species = "positron"

    mc2 = mass_of(species)
    mean_p = (1 + tao.bunch_comb("pz")) * p0c
    data["mean_p"] = mean_p
    data["mean_energy"] = np.hypot(mean_p, mc2)

    return data


def comb_twiss_data(
    tao, planes=("x", "y", "a", "b"), properties=("beta", "alpha", "phi", "eta")
):
    """
    Returns a dict all combinations of twiss data for various planes, in the form:
        'twiss_{plane}_{property}'
        e.g.: 'twiss_beta_y'
    """
    data = {}
    for plane in planes:
        for prop in properties:
            data[f"twiss_{prop}_{plane}"] = tao.bunch_comb(f"{plane}.{prop}")
    return data


def comb_data_from_tao(tao: Tao, ix_branch: int = 0):
    """
    Extract all comb data from tao.

    This converts tao.bunch_comb data from Bmad units into standard
    openPMD-beamphysics units.
    """
    data = {
        **comb_covariance_data(tao),
        **comb_twiss_data(tao),
        **comb_emittance_data(tao),
        **comb_misc_data(tao),
    }

    # for scaling
    p0c = data["p0c"]

    # rel_min, rel_max, mean
    for i in comb_phase_space_label:
        label = comb_phase_space_label[i]

        if i in (2, 4):
            scale = p0c
        elif i == 6:
            label = "delta"
            scale = 1
        else:
            scale = 1
        data[f"rel_min_{label}"] = (tao.bunch_comb(f"rel_min.{i}")) * scale
        data[f"rel_max_{label}"] = (tao.bunch_comb(f"rel_max.{i}")) * scale

        # Means
        if label == "delta":
            bkey = "pz"
        else:
            bkey = label
        data[f"mean_{label}"] = tao.bunch_comb(f"{bkey}") * scale

    # s, time
    data["mean_t"] = tao.bunch_comb("t")
    data["s"] = tao.bunch_comb("s")

    # Also form mean_p, mean_energy for convenience
    species = tao.beam_init(ix_branch=ix_branch)["species"]
    if species == "":
        species = "positron"
    mc2 = mass_of(species)
    mean_p = (data["mean_delta"] + 1) * p0c
    data["mean_p"] = mean_p
    data["mean_energy"] = np.hypot(mean_p, mc2)
    return data


class Comb(TaoModel, extra="allow"):
    charge_live: FloatSequence = []
    cov_p__p: FloatSequence = []
    cov_px__p: FloatSequence = []
    cov_px__px: FloatSequence = []
    cov_px__py: FloatSequence = []
    cov_px__y: FloatSequence = []
    cov_px__z: FloatSequence = []
    cov_py__p: FloatSequence = []
    cov_py__py: FloatSequence = []
    cov_py__z: FloatSequence = []
    cov_x__p: FloatSequence = []
    cov_x__px: FloatSequence = []
    cov_x__py: FloatSequence = []
    cov_x__x: FloatSequence = []
    cov_x__y: FloatSequence = []
    cov_x__z: FloatSequence = []
    cov_y__p: FloatSequence = []
    cov_y__py: FloatSequence = []
    cov_y__y: FloatSequence = []
    cov_y__z: FloatSequence = []
    cov_z__p: FloatSequence = []
    cov_z__z: FloatSequence = []
    ix_ele: FloatSequence = []
    mean_delta: FloatSequence = []
    mean_energy: FloatSequence = []
    mean_p: FloatSequence = []
    mean_px: FloatSequence = []
    mean_py: FloatSequence = []
    mean_t: FloatSequence = []
    mean_x: FloatSequence = []
    mean_y: FloatSequence = []
    mean_z: FloatSequence = []
    n_particle_live: FloatSequence = []
    n_particle_lost_in_ele: FloatSequence = []
    norm_emit_x: FloatSequence = []
    norm_emit_y: FloatSequence = []
    p0c: FloatSequence = []
    rel_max_delta: FloatSequence = []
    rel_max_px: FloatSequence = []
    rel_max_py: FloatSequence = []
    rel_max_x: FloatSequence = []
    rel_max_y: FloatSequence = []
    rel_max_z: FloatSequence = []
    rel_min_delta: FloatSequence = []
    rel_min_px: FloatSequence = []
    rel_min_py: FloatSequence = []
    rel_min_x: FloatSequence = []
    rel_min_y: FloatSequence = []
    rel_min_z: FloatSequence = []
    s: FloatSequence = []
    sigma_delta: FloatSequence = []
    sigma_p: FloatSequence = []
    sigma_px: FloatSequence = []
    sigma_py: FloatSequence = []
    sigma_x: FloatSequence = []
    sigma_y: FloatSequence = []
    sigma_z: FloatSequence = []
    twiss_alpha_a: FloatSequence = []
    twiss_alpha_b: FloatSequence = []
    twiss_alpha_x: FloatSequence = []
    twiss_alpha_y: FloatSequence = []
    twiss_beta_a: FloatSequence = []
    twiss_beta_b: FloatSequence = []
    twiss_beta_x: FloatSequence = []
    twiss_beta_y: FloatSequence = []
    twiss_eta_a: FloatSequence = []
    twiss_eta_b: FloatSequence = []
    twiss_eta_x: FloatSequence = []
    twiss_eta_y: FloatSequence = []
    twiss_phi_a: FloatSequence = []
    twiss_phi_b: FloatSequence = []
    twiss_phi_x: FloatSequence = []
    twiss_phi_y: FloatSequence = []

    def query(self, tao: Tao) -> Self:
        return self.from_tao(tao)

    @classmethod
    def from_tao(cls: type[Self], tao: Tao, **kwargs) -> Self:
        """
        Create a Comb instance from Tao.

        Parameters
        ----------
        tao : Tao
        **kwargs : dict
            Additional keyword arguments to pass to `comb_data_from_tao`.

        Returns
        -------
        Comb
        """
        return cls(**comb_data_from_tao(tao, **kwargs))

    def slice_by_s(self, s_start: float, s_end: float, *, inclusive: bool = True) -> Comb:
        """
        Slice the Comb data by 's' position between specified start and end values.

        Parameters
        ----------
        s_start : float
            The starting s value of the slice.
        s_end : float
            The ending s value of the slice.
        inclusive : bool, default=True
            If True, the slice includes `s_start` and `s_end`.
            Otherwise, it excludes these boundaries.

        Returns
        -------
        Comb
            A new instance of the Comb class with the sliced data.
        """
        s = np.asarray(self.s)
        if inclusive:
            (indices,) = np.where((s <= s_end) & (s >= s_start))
        else:
            (indices,) = np.where((s < s_end) & (s > s_start))

        def fix_value(value):
            if isinstance(value, (list, np.ndarray)):
                return np.asarray(value)[indices]
            return value

        data = {key: fix_value(value) for key, value in self.model_dump().items()}
        return type(self)(**data)


def combine_combs(combs: typing.Sequence[Comb]) -> Comb:
    """Combine the given combs into a single one."""
    attrs = set(Comb.model_fields) - {"command_args"}
    res = Comb()

    for comb in combs:
        for attr in attrs:
            getattr(res, attr).extend(getattr(comb, attr))

    order = np.argsort(res.s)
    for attr in attrs:
        value = getattr(res, attr)

        setattr(res, attr, np.asarray(value)[order])
    return res


def load_combs_from_lattice_data(lat_data) -> Comb:
    comb_data = {}
    for ele in lat_data["elements"]:
        if "comb" in ele:
            for key, value in ele["comb"].items():
                if isinstance(value, list):
                    comb_data.setdefault(key, [])
                    comb_data[key].extend(value)
    return Comb(**comb_data)


def load_combs_from_lattice_file(fn: pathlib.Path, format: ArchiveFormat | None = None):
    data = load_model_data(fn, format=format)

    from .. import Lattice

    if isinstance(data, Lattice):
        return combine_combs([ele.comb for ele in data.elements if ele.comb is not None])
    # Otherwise, just raw lattice data
    return load_combs_from_lattice_data(data)
