from __future__ import annotations

import pathlib
import typing

import numpy as np
from typing_extensions import Self

from ...errors import TaoCommandError
from ..base import ArchiveFormat, TaoModel, load_model_data
from ..types import NDArray, deserialize_ndarray, empty_ndarray

if typing.TYPE_CHECKING:
    from pytao import Tao


def _get_branch_species(tao: Tao, ix_uni: int = 1, ix_branch: int = 0) -> str:
    species = tao.beam_init(ix_uni=str(ix_uni), ix_branch=str(ix_branch))["species"]
    if species not in {"H-", "H2+"}:
        # NOTE/TODO: Bmad has many more species types, but beamphysics only
        # supports 7 as of the time of writing.
        # Tao accepts arbitrarily-cased named subatomic particles, so we
        # normalize here ahead of time.
        species = species.lower()

    return species or "positron"


def _get_branch_mc2(tao: Tao, ix_uni: int = 1, ix_branch: int = 0) -> float:
    from beamphysics.species import mass_of

    return mass_of(_get_branch_species(tao, ix_uni=ix_uni, ix_branch=ix_branch))


def comb_data_from_tao(
    tao: Tao, ix_uni: int = 1, ix_branch: int | str = 0, ix_bunch: int = 1
) -> dict[str, float | np.ndarray]:
    """
    Extract all comb data from tao.
    """

    def get(who: str) -> np.ndarray:
        return typing.cast(
            np.ndarray,
            tao.bunch_comb(
                who,
                ix_uni=str(ix_uni),
                ix_branch=str(ix_branch),
                ix_bunch=str(ix_bunch),
                flags="-array_out",
            ),
        )

    data: dict[str, float | np.ndarray] = {
        "mc2": _get_branch_mc2(tao, ix_uni=ix_uni, ix_branch=ix_branch),
        "p0c": get("p0c"),
        # Centroid
        "centroid_1": get("x"),
        "centroid_2": get("px"),
        "centroid_3": get("y"),
        "centroid_4": get("py"),
        "centroid_5": get("z"),
        "centroid_6": get("pz"),
        "t": get("t"),
        "s": get("s"),
        "centroid_spin_x": get("spin.x"),
        "centroid_spin_y": get("spin.y"),
        "centroid_spin_z": get("spin.z"),
        "centroid_beta": get("beta"),
        # Single sigma (more below)
        # Verbatim
        "charge_live": get("charge_live"),
        "n_particle_live": get("n_particle_live"),
        "n_particle_lost_in_ele": get("n_particle_lost_in_ele"),
        "ix_ele": get("ix_ele"),
    }

    for plane in "xyzabc":
        for prop in (
            "beta",
            "alpha",
            "gamma",
            "phi",
            "eta",
            "etap",
            "deta_ds",
            "dbeta_dpz",
            "dalpha_dpz",
            "deta_dpz",
            "detap_dpz",
            "sigma",
            "sigma_p",
            "emit",
            "norm_emit",
        ):
            data[f"twiss_{prop}_{plane}"] = get(f"{plane}.{prop}")

    try:
        data["sigma_t"] = get("t.sigma")
    except TaoCommandError:
        # bug in Bmad <=20260713-0
        data["sigma_t"] = np.zeros_like(data["s"])

    for i in range(1, 7):
        data[f"rel_min_{i}"] = get(f"rel_min.{i}")
        data[f"rel_max_{i}"] = get(f"rel_max.{i}")
        for j in range(1, 7):
            data[f"sigma_{i}{j}"] = get(f"sigma.{i}{j}")

    return data


class Comb(TaoModel, extra="allow"):
    """
    Beam comb (saved bunch statistics) along the lattice.

    Note
    ----
    Arrays are indexed by element.

    Phase-space momenta follow Bmad's normalized convention and are
    dimensionless:

    * ``px`` = p_x / p0
    * ``py`` = p_y / p0
    * ``pz`` = p/p0 - 1 (also ``vec(6)`` or ``delta`` in Bmad)

    Where p is the total momentum and p0 is the reference momentum.

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
        Mean horizontal momentum, (p_x/p0) * p0c (eV/c).
    mean_py : np.ndarray
        Mean vertical momentum, (p_y/p0) * p0c (eV/c).
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
        RMS horizontal momentum spread, sqrt(sigma_22) * p0c (eV/c).
    sigma_py : np.ndarray
        RMS vertical momentum spread, sqrt(sigma_44) * p0c (eV/c).
    sigma_p : np.ndarray
        RMS momentum spread, sqrt(sigma_66) * p0c (eV/c).
    sigma_delta : np.ndarray
        RMS fractional momentum spread, sqrt(sigma_66) (dimensionless).
    twiss_norm_emit_x : np.ndarray
        Normalized RMS horizontal emittance (m·rad).
    twiss_norm_emit_y : np.ndarray
        Normalized RMS vertical emittance (m·rad).
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
        Minimum horizontal momentum relative to mean, rel_min(2) * p0c (eV/c).
    rel_max_px : np.ndarray
        Maximum horizontal momentum relative to mean, rel_max(2) * p0c (eV/c).
    rel_min_py : np.ndarray
        Minimum vertical momentum relative to mean, rel_min(4) * p0c (eV/c).
    rel_max_py : np.ndarray
        Maximum vertical momentum relative to mean, rel_max(4) * p0c (eV/c).
    rel_min_delta : np.ndarray
        Minimum relative momentum deviation relative to mean (dimensionless).
    rel_max_delta : np.ndarray
        Maximum relative momentum deviation relative to mean (dimensionless).
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

    mc2: float = 0.0

    centroid_1: NDArray = empty_ndarray()
    centroid_2: NDArray = empty_ndarray()
    centroid_3: NDArray = empty_ndarray()
    centroid_4: NDArray = empty_ndarray()
    centroid_5: NDArray = empty_ndarray()
    centroid_6: NDArray = empty_ndarray()
    centroid_beta: NDArray = empty_ndarray()
    centroid_spin_x: NDArray = empty_ndarray()
    centroid_spin_y: NDArray = empty_ndarray()
    centroid_spin_z: NDArray = empty_ndarray()
    charge_live: NDArray = empty_ndarray()
    ix_ele: NDArray = empty_ndarray()
    n_particle_live: NDArray = empty_ndarray()
    n_particle_lost_in_ele: NDArray = empty_ndarray()
    p0c: NDArray = empty_ndarray()
    rel_max_1: NDArray = empty_ndarray()
    rel_max_2: NDArray = empty_ndarray()
    rel_max_3: NDArray = empty_ndarray()
    rel_max_4: NDArray = empty_ndarray()
    rel_max_5: NDArray = empty_ndarray()
    rel_max_6: NDArray = empty_ndarray()
    rel_min_1: NDArray = empty_ndarray()
    rel_min_2: NDArray = empty_ndarray()
    rel_min_3: NDArray = empty_ndarray()
    rel_min_4: NDArray = empty_ndarray()
    rel_min_5: NDArray = empty_ndarray()
    rel_min_6: NDArray = empty_ndarray()
    s: NDArray = empty_ndarray()
    sigma_11: NDArray = empty_ndarray()
    sigma_12: NDArray = empty_ndarray()
    sigma_13: NDArray = empty_ndarray()
    sigma_14: NDArray = empty_ndarray()
    sigma_15: NDArray = empty_ndarray()
    sigma_16: NDArray = empty_ndarray()
    sigma_21: NDArray = empty_ndarray()
    sigma_22: NDArray = empty_ndarray()
    sigma_23: NDArray = empty_ndarray()
    sigma_24: NDArray = empty_ndarray()
    sigma_25: NDArray = empty_ndarray()
    sigma_26: NDArray = empty_ndarray()
    sigma_31: NDArray = empty_ndarray()
    sigma_32: NDArray = empty_ndarray()
    sigma_33: NDArray = empty_ndarray()
    sigma_34: NDArray = empty_ndarray()
    sigma_35: NDArray = empty_ndarray()
    sigma_36: NDArray = empty_ndarray()
    sigma_41: NDArray = empty_ndarray()
    sigma_42: NDArray = empty_ndarray()
    sigma_43: NDArray = empty_ndarray()
    sigma_44: NDArray = empty_ndarray()
    sigma_45: NDArray = empty_ndarray()
    sigma_46: NDArray = empty_ndarray()
    sigma_51: NDArray = empty_ndarray()
    sigma_52: NDArray = empty_ndarray()
    sigma_53: NDArray = empty_ndarray()
    sigma_54: NDArray = empty_ndarray()
    sigma_55: NDArray = empty_ndarray()
    sigma_56: NDArray = empty_ndarray()
    sigma_61: NDArray = empty_ndarray()
    sigma_62: NDArray = empty_ndarray()
    sigma_63: NDArray = empty_ndarray()
    sigma_64: NDArray = empty_ndarray()
    sigma_65: NDArray = empty_ndarray()
    sigma_66: NDArray = empty_ndarray()
    sigma_t: NDArray = empty_ndarray()
    t: NDArray = empty_ndarray()
    twiss_alpha_a: NDArray = empty_ndarray()
    twiss_alpha_b: NDArray = empty_ndarray()
    twiss_alpha_c: NDArray = empty_ndarray()
    twiss_alpha_x: NDArray = empty_ndarray()
    twiss_alpha_y: NDArray = empty_ndarray()
    twiss_alpha_z: NDArray = empty_ndarray()
    twiss_beta_a: NDArray = empty_ndarray()
    twiss_beta_b: NDArray = empty_ndarray()
    twiss_beta_c: NDArray = empty_ndarray()
    twiss_beta_x: NDArray = empty_ndarray()
    twiss_beta_y: NDArray = empty_ndarray()
    twiss_beta_z: NDArray = empty_ndarray()
    twiss_dalpha_dpz_a: NDArray = empty_ndarray()
    twiss_dalpha_dpz_b: NDArray = empty_ndarray()
    twiss_dalpha_dpz_c: NDArray = empty_ndarray()
    twiss_dalpha_dpz_x: NDArray = empty_ndarray()
    twiss_dalpha_dpz_y: NDArray = empty_ndarray()
    twiss_dalpha_dpz_z: NDArray = empty_ndarray()
    twiss_dbeta_dpz_a: NDArray = empty_ndarray()
    twiss_dbeta_dpz_b: NDArray = empty_ndarray()
    twiss_dbeta_dpz_c: NDArray = empty_ndarray()
    twiss_dbeta_dpz_x: NDArray = empty_ndarray()
    twiss_dbeta_dpz_y: NDArray = empty_ndarray()
    twiss_dbeta_dpz_z: NDArray = empty_ndarray()
    twiss_deta_dpz_a: NDArray = empty_ndarray()
    twiss_deta_dpz_b: NDArray = empty_ndarray()
    twiss_deta_dpz_c: NDArray = empty_ndarray()
    twiss_deta_dpz_x: NDArray = empty_ndarray()
    twiss_deta_dpz_y: NDArray = empty_ndarray()
    twiss_deta_dpz_z: NDArray = empty_ndarray()
    twiss_deta_ds_a: NDArray = empty_ndarray()
    twiss_deta_ds_b: NDArray = empty_ndarray()
    twiss_deta_ds_c: NDArray = empty_ndarray()
    twiss_deta_ds_x: NDArray = empty_ndarray()
    twiss_deta_ds_y: NDArray = empty_ndarray()
    twiss_deta_ds_z: NDArray = empty_ndarray()
    twiss_detap_dpz_a: NDArray = empty_ndarray()
    twiss_detap_dpz_b: NDArray = empty_ndarray()
    twiss_detap_dpz_c: NDArray = empty_ndarray()
    twiss_detap_dpz_x: NDArray = empty_ndarray()
    twiss_detap_dpz_y: NDArray = empty_ndarray()
    twiss_detap_dpz_z: NDArray = empty_ndarray()
    twiss_emit_a: NDArray = empty_ndarray()
    twiss_emit_b: NDArray = empty_ndarray()
    twiss_emit_c: NDArray = empty_ndarray()
    twiss_emit_x: NDArray = empty_ndarray()
    twiss_emit_y: NDArray = empty_ndarray()
    twiss_emit_z: NDArray = empty_ndarray()
    twiss_eta_a: NDArray = empty_ndarray()
    twiss_eta_b: NDArray = empty_ndarray()
    twiss_eta_c: NDArray = empty_ndarray()
    twiss_eta_x: NDArray = empty_ndarray()
    twiss_eta_y: NDArray = empty_ndarray()
    twiss_eta_z: NDArray = empty_ndarray()
    twiss_etap_a: NDArray = empty_ndarray()
    twiss_etap_b: NDArray = empty_ndarray()
    twiss_etap_c: NDArray = empty_ndarray()
    twiss_etap_x: NDArray = empty_ndarray()
    twiss_etap_y: NDArray = empty_ndarray()
    twiss_etap_z: NDArray = empty_ndarray()
    twiss_gamma_a: NDArray = empty_ndarray()
    twiss_gamma_b: NDArray = empty_ndarray()
    twiss_gamma_c: NDArray = empty_ndarray()
    twiss_gamma_x: NDArray = empty_ndarray()
    twiss_gamma_y: NDArray = empty_ndarray()
    twiss_gamma_z: NDArray = empty_ndarray()
    twiss_norm_emit_a: NDArray = empty_ndarray()
    twiss_norm_emit_b: NDArray = empty_ndarray()
    twiss_norm_emit_c: NDArray = empty_ndarray()
    twiss_norm_emit_x: NDArray = empty_ndarray()
    twiss_norm_emit_y: NDArray = empty_ndarray()
    twiss_norm_emit_z: NDArray = empty_ndarray()
    twiss_phi_a: NDArray = empty_ndarray()
    twiss_phi_b: NDArray = empty_ndarray()
    twiss_phi_c: NDArray = empty_ndarray()
    twiss_phi_x: NDArray = empty_ndarray()
    twiss_phi_y: NDArray = empty_ndarray()
    twiss_phi_z: NDArray = empty_ndarray()
    twiss_sigma_a: NDArray = empty_ndarray()
    twiss_sigma_b: NDArray = empty_ndarray()
    twiss_sigma_c: NDArray = empty_ndarray()
    twiss_sigma_p_a: NDArray = empty_ndarray()
    twiss_sigma_p_b: NDArray = empty_ndarray()
    twiss_sigma_p_c: NDArray = empty_ndarray()
    twiss_sigma_p_x: NDArray = empty_ndarray()
    twiss_sigma_p_y: NDArray = empty_ndarray()
    twiss_sigma_p_z: NDArray = empty_ndarray()
    twiss_sigma_x: NDArray = empty_ndarray()
    twiss_sigma_y: NDArray = empty_ndarray()
    twiss_sigma_z: NDArray = empty_ndarray()

    def __repr__(self):
        range_info = ""
        if len(self.s):
            min_s, max_s = np.min(self.s), np.max(self.s)
            range_info = f" from s={min_s:.4g} to s={max_s:.4g}"

            if len(self.s) > 1:
                step = self.s[1] - self.s[0]
                range_info = f"{range_info} with step {step:.4g}"

        return f"<{type(self).__name__} with {len(self.s)} data points{range_info}>"

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
        cls: type[Self],
        tao: Tao,
        *,
        check_ds_save: bool = True,
        ix_uni: int = 1,
        ix_branch: int | str = 0,
        ix_bunch: int | str = 1,
    ) -> Self:
        """
        Get Comb data, if available.

        Parameters
        ----------
        ix_uni : int, optional
            Defaults to the primary universe, universe 1.
        ix_branch : str or int, optional
            Defaults to the primary branch, branch 0.
        ix_bunch : int, optional
            The bunch index. Defaults to 1.
        check_ds_save : bool, optional
            Check if Comb data should be saved first (`beam
            ds_save` is set). Defaults to True.
            When True, returns an empty `Comb` instance if comb data is unavailable.
            When False, raises `TaoCommandError` if no comb data available.

        Raises
        ------
        TaoCommandError
            If `check_ds_save` is False and comb data is unavailable.

        Returns
        -------
        Comb
            Comb data.
        """
        if check_ds_save:
            if tao.beam(ix_branch)["ds_save"] <= 0:
                return cls()

        args = {"ix_uni": ix_uni, "ix_branch": ix_branch, "ix_bunch": ix_bunch}
        data = comb_data_from_tao(tao, **args)
        return cls(**data, command_args=args)

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
            if not len(indices):
                return value

            if isinstance(value, (list, np.ndarray)):
                return np.asarray(value)[indices]
            return value

        data = {key: fix_value(value) for key, value in self.model_dump().items()}
        return type(self)(**data)

    @property
    def mean_x(self) -> np.ndarray:
        """Centroid mean x (m)."""
        return self.centroid_1  # m

    @property
    def mean_px(self) -> np.ndarray:
        """Centroid mean px (eV/c)."""
        return self.centroid_2 * self.p0c  # eV/c

    @property
    def mean_y(self) -> np.ndarray:
        """Centroid mean y (m)."""
        return self.centroid_3  # m

    @property
    def mean_py(self) -> np.ndarray:
        """Centroid mean py (eV/c)."""
        return self.centroid_4 * self.p0c  # eV/c

    @property
    def mean_z(self) -> np.ndarray:
        """Centroid mean z (m)."""
        return self.centroid_5

    @property
    def mean_p(self) -> np.ndarray:
        """Centroid mean total momentum, (1 + pz) * p0c (eV/c)."""
        return (1 + self.centroid_6) * self.p0c  # eV/c

    @property
    def mean_delta(self) -> np.ndarray:
        """Centroid mean fractional momentum deviation, pz = (p - p0) / p0 (dimensionless)."""
        return self.centroid_6

    @property
    def mean_energy(self) -> np.ndarray:
        """Mean energy (eV)."""
        return np.hypot(self.mean_p, self.mc2)

    @property
    def mean_t(self) -> np.ndarray:
        """Centroid mean time coordinate (s)."""
        return self.t

    @property
    def sigma_x(self) -> np.ndarray:
        """Sigma x (m)."""
        return np.sqrt(self.sigma_11)  # m

    @property
    def sigma_px(self) -> np.ndarray:
        """Sigma px (eV/c)."""
        return np.sqrt(self.sigma_22) * self.p0c  # eV/c

    @property
    def sigma_y(self) -> np.ndarray:
        """Sigma y (m)."""
        return np.sqrt(self.sigma_33)

    @property
    def sigma_py(self) -> np.ndarray:
        """Sigma py (eV/c)."""
        return np.sqrt(self.sigma_44) * self.p0c  # eV/c

    @property
    def sigma_z(self) -> np.ndarray:
        """Sigma z (m)."""
        return np.sqrt(self.sigma_55)  # m

    @property
    def sigma_p(self) -> np.ndarray:
        """RMS momentum spread, sqrt(sigma_66) * p0c (eV/c)."""
        return np.sqrt(self.sigma_66) * self.p0c  # eV/c

    @property
    def sigma_delta(self) -> np.ndarray:
        """RMS fractional momentum spread, sqrt(sigma_66) (dimensionless)."""
        return np.sqrt(self.sigma_66)

    @property
    def rel_min_x(self) -> np.ndarray:
        """Relative minimum x (m)."""
        return self.rel_min_1

    @property
    def rel_min_px(self) -> np.ndarray:
        """Relative minimum px (eV/c)."""
        return self.rel_min_2 * self.p0c

    @property
    def rel_min_y(self) -> np.ndarray:
        """Relative minimum y (m)."""
        return self.rel_min_3

    @property
    def rel_min_py(self) -> np.ndarray:
        """Relative minimum py (eV/c)."""
        return self.rel_min_4 * self.p0c

    @property
    def rel_min_z(self) -> np.ndarray:
        """Relative minimum z (m)."""
        return self.rel_min_5

    @property
    def rel_min_p(self) -> np.ndarray:
        """Minimum momentum, (1 + rel_min(6)) * p0c (eV/c)."""
        return (1 + self.rel_min_6) * self.p0c

    @property
    def rel_max_x(self) -> np.ndarray:
        """Relative maximum x (m)."""
        return self.rel_max_1

    @property
    def rel_max_px(self) -> np.ndarray:
        """Relative maximum px (eV/c)."""
        return self.rel_max_2 * self.p0c

    @property
    def rel_max_y(self) -> np.ndarray:
        """Relative maximum y (m)."""
        return self.rel_max_3

    @property
    def rel_max_py(self) -> np.ndarray:
        """Relative maximum py (eV/c)."""
        return self.rel_max_4 * self.p0c

    @property
    def rel_max_z(self) -> np.ndarray:
        """Relative maximum z (m)."""
        return self.rel_max_5

    @property
    def rel_max_p(self) -> np.ndarray:
        """Maximum momentum, (1 + rel_max(6)) * p0c (eV/c)."""
        return (1 + self.rel_max_6) * self.p0c

    @property
    def rel_min_delta(self) -> np.ndarray:
        """Minimum fractional momentum deviation relative to mean (dimensionless)."""
        return self.rel_min_6

    @property
    def rel_max_delta(self) -> np.ndarray:
        """Maximum fractional momentum deviation relative to mean (dimensionless)."""
        return self.rel_max_6

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
        """Minimum horizontal momentum, mean_px + rel_min_px (eV/c)."""
        return self.mean_px + self.rel_min_px

    @property
    def py_min(self) -> np.ndarray:
        """Minimum vertical momentum, mean_py + rel_min_py (eV/c)."""
        return self.mean_py + self.rel_min_py

    @property
    def px_max(self) -> np.ndarray:
        """Maximum horizontal momentum, mean_px + rel_max_px (eV/c)."""
        return self.mean_px + self.rel_max_px

    @property
    def py_max(self) -> np.ndarray:
        """Maximum vertical momentum, mean_py + rel_max_py (eV/c)."""
        return self.mean_py + self.rel_max_py

    @property
    def min_delta(self) -> np.ndarray:
        """Minimum fractional momentum deviation, mean_delta + rel_min_delta (dimensionless)."""
        return self.mean_delta + self.rel_min_delta

    @property
    def max_delta(self) -> np.ndarray:
        """Maximum fractional momentum deviation, mean_delta + rel_max_delta (dimensionless)."""
        return self.mean_delta + self.rel_max_delta


_comb_array_attrs = set(Comb.model_fields) - {"command_args", "mc2"}


def combine_combs(combs: typing.Sequence[Comb], sort: bool = True) -> Comb:
    """Combine the given combs into a single one."""
    res = Comb()

    for attr in _comb_array_attrs:
        parts = [getattr(comb, attr) for comb in combs]
        if parts:
            # NOTE: this may fail if you add a scalar attribute to Comb and
            # _comb_array_attrs doesn't get updated!
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
