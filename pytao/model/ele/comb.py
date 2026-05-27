from __future__ import annotations

import pathlib
import typing

import numpy as np
from typing_extensions import Self

from ..base import ArchiveFormat, TaoModel, load_model_data
from ..types import NDArray, deserialize_ndarray, empty_ndarray

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

    from beamphysics.species import mass_of

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
    from beamphysics.species import mass_of

    mc2 = mass_of(species)
    mean_p = (data["mean_delta"] + 1) * p0c
    data["mean_p"] = mean_p
    data["mean_energy"] = np.hypot(mean_p, mc2)
    return data


class Comb(TaoModel, extra="allow"):
    """
    Beam comb (saved bunch statistics) along the lattice.

    Note
    ----
    Arrays are indexed by element. Phase-space momenta follow Bmad's
    normalized convention and are dimensionless: ``px`` = p_x / p0,
    ``py`` = p_y / p0, and the longitudinal ``p`` = ``delta`` = (p - p0) / p0.
    This differs from openPMD-beamphysics, where px/py/p carry units of eV/c.

    Attributes
    ----------
    s : np.ndarray
        Longitudinal position (m).
    p0c : np.ndarray
        Reference momentum energy equivalent, p0*c (eV).
    charge_live : np.ndarray
        Total live charge (C).
    ix_ele : np.ndarray
        Element index (dimensionless).
    n_particle_live : np.ndarray
        Number of live particles (dimensionless).
    n_particle_lost_in_ele : np.ndarray
        Number of particles lost in this element (dimensionless).
    mean_x : np.ndarray
        Mean horizontal position (m).
    mean_y : np.ndarray
        Mean vertical position (m).
    mean_z : np.ndarray
        Mean longitudinal position, z = -beta*c*(t - t_ref) (m).
    mean_px : np.ndarray
        Mean normalized horizontal momentum p_x/p0 (dimensionless).
    mean_py : np.ndarray
        Mean normalized vertical momentum p_y/p0 (dimensionless).
    mean_p : np.ndarray
        Mean total momentum, (delta + 1) * p0c (eV/c).
    mean_delta : np.ndarray
        Mean relative momentum deviation (p - p0) / p0 (dimensionless).
    mean_energy : np.ndarray
        Mean total relativistic energy (eV).
    mean_t : np.ndarray
        Mean time coordinate (s).
    sigma_x : np.ndarray
        RMS horizontal beam size (m).
    sigma_y : np.ndarray
        RMS vertical beam size (m).
    sigma_z : np.ndarray
        RMS longitudinal beam size (m).
    sigma_px : np.ndarray
        RMS normalized horizontal momentum spread (dimensionless).
    sigma_py : np.ndarray
        RMS normalized vertical momentum spread (dimensionless).
    sigma_p : np.ndarray
        RMS relative momentum spread (dimensionless).
    sigma_delta : np.ndarray
        RMS relative momentum spread (dimensionless).
    norm_emit_x : np.ndarray
        Normalized RMS horizontal emittance (m).
    norm_emit_y : np.ndarray
        Normalized RMS vertical emittance (m).
    rel_min_x : np.ndarray
        Minimum horizontal position relative to mean (m).
    rel_max_x : np.ndarray
        Maximum horizontal position relative to mean (m).
    rel_min_y : np.ndarray
        Minimum vertical position relative to mean (m).
    rel_max_y : np.ndarray
        Maximum vertical position relative to mean (m).
    rel_min_z : np.ndarray
        Minimum longitudinal position relative to mean (m).
    rel_max_z : np.ndarray
        Maximum longitudinal position relative to mean (m).
    rel_min_px : np.ndarray
        Minimum normalized horizontal momentum relative to mean (dimensionless).
    rel_max_px : np.ndarray
        Maximum normalized horizontal momentum relative to mean (dimensionless).
    rel_min_py : np.ndarray
        Minimum normalized vertical momentum relative to mean (dimensionless).
    rel_max_py : np.ndarray
        Maximum normalized vertical momentum relative to mean (dimensionless).
    rel_min_delta : np.ndarray
        Minimum relative momentum deviation relative to mean (dimensionless).
    rel_max_delta : np.ndarray
        Maximum relative momentum deviation relative to mean (dimensionless).
    cov_x__x : np.ndarray
        Covariance <x*x> - <x><x> (m^2).
    cov_x__px : np.ndarray
        Covariance <x*px> - <x><px> (m).
    cov_x__y : np.ndarray
        Covariance <x*y> - <x><y> (m^2).
    cov_x__py : np.ndarray
        Covariance <x*py> - <x><py> (m).
    cov_x__z : np.ndarray
        Covariance <x*z> - <x><z> (m^2).
    cov_x__p : np.ndarray
        Covariance <x*delta> - <x><delta> (m).
    cov_px__px : np.ndarray
        Covariance <px*px> - <px><px> (dimensionless).
    cov_px__y : np.ndarray
        Covariance <px*y> - <px><y> (m).
    cov_px__py : np.ndarray
        Covariance <px*py> - <px><py> (dimensionless).
    cov_px__z : np.ndarray
        Covariance <px*z> - <px><z> (m).
    cov_px__p : np.ndarray
        Covariance <px*delta> - <px><delta> (dimensionless).
    cov_y__y : np.ndarray
        Covariance <y*y> - <y><y> (m^2).
    cov_y__py : np.ndarray
        Covariance <y*py> - <y><py> (m).
    cov_y__z : np.ndarray
        Covariance <y*z> - <y><z> (m^2).
    cov_y__p : np.ndarray
        Covariance <y*delta> - <y><delta> (m).
    cov_py__py : np.ndarray
        Covariance <py*py> - <py><py> (dimensionless).
    cov_py__z : np.ndarray
        Covariance <py*z> - <py><z> (m).
    cov_py__p : np.ndarray
        Covariance <py*delta> - <py><delta> (dimensionless).
    cov_z__z : np.ndarray
        Covariance <z*z> - <z><z> (m^2).
    cov_z__p : np.ndarray
        Covariance <z*delta> - <z><delta> (m).
    cov_p__p : np.ndarray
        Covariance <delta*delta> - <delta><delta> (dimensionless).
    twiss_beta_x : np.ndarray
        Horizontal beta function (m).
    twiss_beta_y : np.ndarray
        Vertical beta function (m).
    twiss_beta_a : np.ndarray
        Mode-a beta function (m).
    twiss_beta_b : np.ndarray
        Mode-b beta function (m).
    twiss_alpha_x : np.ndarray
        Horizontal alpha function (dimensionless).
    twiss_alpha_y : np.ndarray
        Vertical alpha function (dimensionless).
    twiss_alpha_a : np.ndarray
        Mode-a alpha function (dimensionless).
    twiss_alpha_b : np.ndarray
        Mode-b alpha function (dimensionless).
    twiss_phi_x : np.ndarray
        Horizontal phase advance (rad).
    twiss_phi_y : np.ndarray
        Vertical phase advance (rad).
    twiss_phi_a : np.ndarray
        Mode-a phase advance (rad).
    twiss_phi_b : np.ndarray
        Mode-b phase advance (rad).
    twiss_eta_x : np.ndarray
        Horizontal dispersion function (m).
    twiss_eta_y : np.ndarray
        Vertical dispersion function (m).
    twiss_eta_a : np.ndarray
        Mode-a dispersion function (m).
    twiss_eta_b : np.ndarray
        Mode-b dispersion function (m).
    """

    charge_live: NDArray = empty_ndarray()
    cov_p__p: NDArray = empty_ndarray()
    cov_px__p: NDArray = empty_ndarray()
    cov_px__px: NDArray = empty_ndarray()
    cov_px__py: NDArray = empty_ndarray()
    cov_px__y: NDArray = empty_ndarray()
    cov_px__z: NDArray = empty_ndarray()
    cov_py__p: NDArray = empty_ndarray()
    cov_py__py: NDArray = empty_ndarray()
    cov_py__z: NDArray = empty_ndarray()
    cov_x__p: NDArray = empty_ndarray()
    cov_x__px: NDArray = empty_ndarray()
    cov_x__py: NDArray = empty_ndarray()
    cov_x__x: NDArray = empty_ndarray()
    cov_x__y: NDArray = empty_ndarray()
    cov_x__z: NDArray = empty_ndarray()
    cov_y__p: NDArray = empty_ndarray()
    cov_y__py: NDArray = empty_ndarray()
    cov_y__y: NDArray = empty_ndarray()
    cov_y__z: NDArray = empty_ndarray()
    cov_z__p: NDArray = empty_ndarray()
    cov_z__z: NDArray = empty_ndarray()
    ix_ele: NDArray = empty_ndarray()
    mean_delta: NDArray = empty_ndarray()
    mean_energy: NDArray = empty_ndarray()
    mean_p: NDArray = empty_ndarray()
    mean_px: NDArray = empty_ndarray()
    mean_py: NDArray = empty_ndarray()
    mean_t: NDArray = empty_ndarray()
    mean_x: NDArray = empty_ndarray()
    mean_y: NDArray = empty_ndarray()
    mean_z: NDArray = empty_ndarray()
    n_particle_live: NDArray = empty_ndarray()
    n_particle_lost_in_ele: NDArray = empty_ndarray()
    norm_emit_x: NDArray = empty_ndarray()
    norm_emit_y: NDArray = empty_ndarray()
    p0c: NDArray = empty_ndarray()
    rel_max_delta: NDArray = empty_ndarray()
    rel_max_px: NDArray = empty_ndarray()
    rel_max_py: NDArray = empty_ndarray()
    rel_max_x: NDArray = empty_ndarray()
    rel_max_y: NDArray = empty_ndarray()
    rel_max_z: NDArray = empty_ndarray()
    rel_min_delta: NDArray = empty_ndarray()
    rel_min_px: NDArray = empty_ndarray()
    rel_min_py: NDArray = empty_ndarray()
    rel_min_x: NDArray = empty_ndarray()
    rel_min_y: NDArray = empty_ndarray()
    rel_min_z: NDArray = empty_ndarray()
    s: NDArray = empty_ndarray()
    sigma_delta: NDArray = empty_ndarray()
    sigma_p: NDArray = empty_ndarray()
    sigma_px: NDArray = empty_ndarray()
    sigma_py: NDArray = empty_ndarray()
    sigma_x: NDArray = empty_ndarray()
    sigma_y: NDArray = empty_ndarray()
    sigma_z: NDArray = empty_ndarray()
    twiss_alpha_a: NDArray = empty_ndarray()
    twiss_alpha_b: NDArray = empty_ndarray()
    twiss_alpha_x: NDArray = empty_ndarray()
    twiss_alpha_y: NDArray = empty_ndarray()
    twiss_beta_a: NDArray = empty_ndarray()
    twiss_beta_b: NDArray = empty_ndarray()
    twiss_beta_x: NDArray = empty_ndarray()
    twiss_beta_y: NDArray = empty_ndarray()
    twiss_eta_a: NDArray = empty_ndarray()
    twiss_eta_b: NDArray = empty_ndarray()
    twiss_eta_x: NDArray = empty_ndarray()
    twiss_eta_y: NDArray = empty_ndarray()
    twiss_phi_a: NDArray = empty_ndarray()
    twiss_phi_b: NDArray = empty_ndarray()
    twiss_phi_x: NDArray = empty_ndarray()
    twiss_phi_y: NDArray = empty_ndarray()

    @property
    def x_min(self) -> np.ndarray:
        """Minimum horizontal position, mean_x + rel_min_x (m)."""
        return self.mean_x + self.rel_min_x

    @property
    def y_min(self) -> np.ndarray:
        """Minimum vertical position, mean_y + rel_min_y (m)."""
        return self.mean_y + self.rel_min_y

    @property
    def z_min(self) -> np.ndarray:
        """Minimum longitudinal position, mean_z + rel_min_z (m)."""
        return self.mean_z + self.rel_min_z

    @property
    def x_max(self) -> np.ndarray:
        """Maximum horizontal position, mean_x + rel_max_x (m)."""
        return self.mean_x + self.rel_max_x

    @property
    def y_max(self) -> np.ndarray:
        """Maximum vertical position, mean_y + rel_max_y (m)."""
        return self.mean_y + self.rel_max_y

    @property
    def z_max(self) -> np.ndarray:
        """Maximum longitudinal position, mean_z + rel_max_z (m)."""
        return self.mean_z + self.rel_max_z

    @property
    def px_min(self) -> np.ndarray:
        """Minimum normalized horizontal momentum, mean_px + rel_min_px (dimensionless)."""
        return self.mean_px + self.rel_min_px

    @property
    def py_min(self) -> np.ndarray:
        """Minimum normalized vertical momentum, mean_py + rel_min_py (dimensionless)."""
        return self.mean_py + self.rel_min_py

    @property
    def px_max(self) -> np.ndarray:
        """Maximum normalized horizontal momentum, mean_px + rel_max_px (dimensionless)."""
        return self.mean_px + self.rel_max_px

    @property
    def py_max(self) -> np.ndarray:
        """Maximum normalized vertical momentum, mean_py + rel_max_py (dimensionless)."""
        return self.mean_py + self.rel_max_py

    @property
    def min_delta(self) -> np.ndarray:
        """Minimum relative momentum deviation, mean_delta + rel_min_delta (dimensionless)."""
        return self.mean_delta + self.rel_min_delta

    @property
    def max_delta(self) -> np.ndarray:
        """Maximum relative momentum deviation, mean_delta + rel_max_delta (dimensionless)."""
        return self.mean_delta + self.rel_max_delta

    def query(self, tao: Tao) -> Self:
        """Re-query Tao for updated Comb data."""
        return self.from_tao(tao)

    def sort_by_s(self) -> Comb:
        """Sort array data by `s` position."""
        res = Comb()
        order = np.argsort(self.s)
        for attr in _comb_array_attrs:
            value = getattr(self, attr)

            if value.size:
                setattr(res, attr, np.asarray(value)[order])
        return res

    @classmethod
    def from_tao(
        cls: type[Self], tao: Tao, *, check_ds_save: bool = True, ix_branch: int = 0, **kwargs
    ) -> Self:
        """
        Create a Comb instance from Tao.

        Parameters
        ----------
        tao : Tao

        ix_branch : int, optional

        **kwargs : dict
            Additional keyword arguments to pass to `comb_data_from_tao`.

        Returns
        -------
        Comb
        """
        if check_ds_save:
            if tao.beam(ix_branch)["ds_save"] <= 0:
                return cls()

        return cls(**comb_data_from_tao(tao, ix_branch=ix_branch))

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


_comb_array_attrs = set(Comb.model_fields) - {"command_args"}


def combine_combs(combs: typing.Sequence[Comb], sort: bool = True) -> Comb:
    """Combine the given combs into a single one."""
    res = Comb()

    for attr in _comb_array_attrs:
        parts = [getattr(comb, attr) for comb in combs]
        if parts:
            setattr(res, attr, np.concat(parts))

    return res.sort_by_s() if sort else res


def load_combs_from_lattice_data(lat_data, sort: bool = False) -> Comb:
    """
    Load comb data from raw lattice data.

    This can be used to speed up loading only comb data from an archive.

    Parameters
    ----------
    lat_data : dict
        Raw Lattice model data.
    sort : bool, optional
        Sort comb data by s position. Defaults to True.

    Returns
    -------
    Comb

    Example
    -------
    >>> lattice_data = load_model_data(lattice_dump_fn, raw=True)
    >>> comb = load_combs_from_lattice_data(lattice_data)
    """
    comb_data = {}
    for ele in lat_data["elements"]:
        if "comb" in ele:
            for key, value in ele["comb"].items():
                if key in _comb_array_attrs:
                    if isinstance(value, np.ndarray):
                        arr = value
                    else:
                        arr = deserialize_ndarray(value)
                        ele["comb"][key] = arr

                    comb_data.setdefault(key, [])
                    comb_data[key].extend(arr.tolist())
    comb = Comb(**comb_data)
    return comb.sort_by_s() if sort else comb


def load_combs_from_lattice_file(
    fn: pathlib.Path,
    format: ArchiveFormat | None = None,
    sort: bool = True,
) -> Comb:
    """
    Load only comb data from a lattice file.

    Parameters
    ----------
    fn : pathlib.Path
        The Lattice filename dump to load from.
    format : ArchiveFormat or None, optional
        The format of the archive.
    sort : bool, optional
        Sort comb data by s position. Defaults to True.

    Returns
    -------
    Comb
    """
    data = load_model_data(fn, format=format)

    from .. import Lattice

    if isinstance(data, Lattice):
        return combine_combs(
            [ele.comb for ele in data.elements if ele.comb is not None],
            sort=sort,
        )
    # Otherwise, just raw lattice data
    return load_combs_from_lattice_data(data, sort=sort)
